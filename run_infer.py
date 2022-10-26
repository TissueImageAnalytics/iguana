"""run_infer.py

Process slides with IGUANA.

Usage:
  run_infer.py [--gpu=<id>] [--mode=<mode>] [--model=<path>] [--batch_size=<n>] [--num_workers=<n>] [--input_dir=<path>] 
  run_infer.py (-h | --help)
  run_infer.py --version
  
Options:
  -h --help            Show this string.
  --version            Show version.
  --gpu=<id>           GPU list. [default: 0]
  --model=<path>       Path to saved checkpoint.
  --batch_size=<n>     Batch size. [default: 1]
  --num_workers=<n>    Number of workers. [default: 8]
"""

import os
from docopt import docopt
import tqdm
import numpy as np
import pandas as pd
from importlib import import_module
import glob
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import torch
from torch_geometric.data import DataLoader

from dataloader.graph_loader import FileLoader
from metrics.stats_utils import get_auc_pr_sen_spec_metrics_abnormal

import warnings
warnings.filterwarnings('ignore')


def get_labels_scores(wsi_names, scores, gt, binarize=True):
    """Align the scores and labels."""
    gt = pd.read_csv(gt)
    labels_output = []
    scores_output = []
    for idx, wsi_name in enumerate(wsi_names):
        score = scores[idx]
        gt_subset = gt[gt["wsi_id"] == wsi_name]
        lab = list(gt_subset["label_id"])
        if len(lab) > 0:
            lab = int(lab[0])
            if binarize:
                if lab > 0:
                    lab = 1
            labels_output.append(lab)
            scores_output.append(score)
    return labels_output, scores_output

class InferBase(object):
    def __init__(self, **kwargs):
        self.run_step = None
        for variable, value in kwargs.items():
            self.__setattr__(variable, value)
        self.__load_model()
        return

    def __load_model(self):
        """Create the model, load the checkpoint and define
        associated run steps to process each data batch

        """
        model_desc = import_module('models.net_desc')
        model_creator = getattr(model_desc, 'create_model')

        # TODO: deal with parsing multi level model desc
        net = model_creator(model_name=self.model_name, nr_features=25).to('cuda')
        saved_state_dict = torch.load(self.model_path)
        net.load_state_dict(saved_state_dict['desc'], strict=True)

        run_desc = import_module('models.run_desc')
        self.run_step = lambda input_batch: getattr(
            run_desc, 'infer_step')(input_batch, net)
        return


####
class Infer(InferBase):

    def __run_model(self, file_list):

        print('Loading feature statistics...')
        mean = np.load(f"{self.stats_path}/mean.npy")
        median = np.load(f"{self.stats_path}/median.npy")
        std = np.load(f"{self.stats_path}/std.npy")
        perc_25 = np.load(f"{self.stats_path}/perc_25.npy")
        perc_75 = np.load(f"{self.stats_path}/perc_75.npy")

        feat_stats = [mean, median, std, perc_25, perc_75]

        input_dataset = FileLoader(file_list, feat_stats=feat_stats, norm="standard", data_clean="std")
        
        dataloader = DataLoader(input_dataset,
                                num_workers=self.nr_procs,
                                batch_size=self.batch_size,
                                shuffle=False,
                                drop_last=False
                                )

        pbar = tqdm.tqdm(desc='Processsing', leave=True,
                         total=int(len(dataloader)),
                         ncols=80, ascii=True, position=0)

        pred_all = []
        prob_all = []
        true_all = []
        wsi_name_all = []
        for _, batch_data in enumerate(dataloader):
            
            batch_output = self.run_step(batch_data)
            pred_list = []
            prob_list = []
            true_list = []
            wsi_name_list = []

            prob = batch_output['prob']
            true = batch_output['true']
            wsi_info = batch_output['wsi_info']
            num_examples = len(batch_output['true'])
            for idx in range(num_examples):
                pred_tmp = torch.argmax(prob[idx])
                prob_tmp = prob[idx][1]
                true_tmp = true[idx]
                pred_list.append(pred_tmp.cpu())
                prob_list.append(prob_tmp.cpu())
                true_list.append(true_tmp.cpu())
                wsi_name_list.append(wsi_info[0][idx][0])

            pred_all.extend(pred_list)
            prob_all.extend(prob_list)
            true_all.extend(true_list)
            wsi_name_all.extend(wsi_name_list)

            pbar.update()
        pbar.close()
        return np.array(pred_all), np.array(prob_all), np.array(true_all), np.array(wsi_name_all)
    

    def __get_stats(self, prob, true):
        # AUC-ROC
        auc_roc = roc_auc_score(true, prob)
        # AUC-PR
        pr, re, _ = precision_recall_curve(true, prob)
        auc_pr = auc(re, pr)
        
        # specificity @ given sensitivity
        _, _, spec_95, spec_97, spec_98, spec_99, spec_100 = get_auc_pr_sen_spec_metrics_abnormal(true, prob)
        
        print('='*50)
        print("AUC-ROC:", auc_roc)
        print("AUC-PR:", auc_pr)
        print("Specifity_at_97_Sensitivity:", spec_97)
        print("Specifity_at_98_Sensitivity:", spec_98)
        print("Specifity_at_99_Sensitivity:", spec_99)

    def process_files(self):
        
        file_list = []
        for dir_path in self.data_path:
            file_list.extend(glob.glob('%s/*.dat' % dir_path))
        file_list.sort()  # to always ensure same input ordering
        
        print('Number of WSI graphs:', len(file_list))
        print('-'*50)

        pred, prob, true, wsi_names = self.__run_model(file_list)
        
        # save results to a single csv file
        df = pd.DataFrame(data = {'wsi_name': wsi_names, 'score': prob, "pred": pred, 'label': true})
        df.to_csv(f"{self.output_path}/results.csv")
        
        # get stats
        true, prob  = get_labels_scores(wsi_names, prob, self.gt_path)
        self.__get_stats(prob, true)


#-------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    args = docopt(__doc__, version='IGUANA Inference v1.0')

    os.environ['CUDA_VISIBLE_DEVICES'] = args['--gpu']
    
    fold_nr = 1
    model_name = "pna" # keep as 'pna' for iguana
    dataset_name = "imp"
    
    ckpt_root = f"/root/lsf_workspace/iguana_data/weights/"
    data_dir = [f"/root/lsf_workspace/iguana_data/graph_data/{dataset_name}/"]
    gt_root = "/root/lsf_workspace/iguana_data/ground_truth/"
    stats_path = "/root/lsf_workspace/iguana_data/stats/"
    
    output_path = "output_test/"
        
    args = {
        "model_name": model_name,
        "model_path": f"{ckpt_root}/iguana_fold{fold_nr}.tar",
        "stats_path": stats_path,
        "data_path": data_dir,
        "gt_path": f"{gt_root}/{dataset_name}_gt.csv",
        "batch_size": int(args["--batch_size"]), #! do not change
        "nr_procs": int(args["--num_workers"]),
        "output_path": output_path,
    }
 
    infer = Infer(**args)
    infer.process_files()

