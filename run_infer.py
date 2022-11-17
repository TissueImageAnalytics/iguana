"""run_infer.py

Process slides with IGUANA.

Usage:
  run_infer.py [--gpu=<id>] [--model_path=<path>] [--model_name=<str>] [--data_dir=<path>] \
    [--data_info=<path>] [--stats_dir=<path>] [--output_dir=<path>] [--batch_size=<n>] \
    [--fold_nr=<n>] [--split_nr=<n>] [--num_workers=<n>]
  run_infer.py (-h | --help)
  run_infer.py --version
  
Options:
  -h --help              Show this string.
  --version              Show version.
  --gpu=<id>             GPU list. [default: 0]
  --model_path=<path>    Path to saved checkpoint.
  --model_name=<str>     Type of graph convolution used. [default: pna]
  --data_dir=<path>      Path to where graph data is stored.
  --data_info=<path>     Path to where data information csv file is stored
  --stats_dir=<path>     Location of feaure stats directory for input standardisation.
  --output_dir=<path>    Path where output will be saved. [default: output/]
  --batch_size=<n>       Batch size. [default: 1]
  --fold_nr=<n>          Fold number considered during cross validation. Don't change if considering independent test set. [default: 1]
  --split_nr=<n>         Only consider slides in the data info csv according to this selected number. [default: 3]
  --num_workers=<n>      Number of workers. [default: 8]

"""

import os
import yaml
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
from metrics.stats_utils import get_sens_spec_metrics
from misc.utils import rm_n_mkdir

import warnings
warnings.filterwarnings('ignore')


def get_labels_scores(wsi_names, scores, gt, binarize=True):
    """Align the scores and labels."""
    labels_output = []
    scores_output = []
    for idx, wsi_name in enumerate(wsi_names):
        score = scores[idx]
        gt_subset = gt[gt["wsi_name"] == wsi_name]
        lab = list(gt_subset["label"])
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
        net = model_creator(
            model_name=self.model_name,
            nr_features=len(self.feat_names),
            node_degree=self.node_degree).to('cuda')
        saved_state_dict = torch.load(self.model_path)
        net.load_state_dict(saved_state_dict['desc'], strict=True)

        run_desc = import_module('models.run_desc')
        self.run_step = lambda input_batch: getattr(
            run_desc, 'infer_step')(input_batch, net)
        return


class Infer(InferBase):
    def __run_model(self, file_list):

        print('Loading feature statistics...')
        with open(f"{self.stats_path}/stats_dict.yml") as fptr:
            stats_dict = yaml.full_load(fptr)

        input_dataset = FileLoader(
            file_list, self.feat_names, feat_stats=stats_dict, norm="standard", data_clean="std"
        )
        
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
            wsi_name = batch_output['wsi_name'][0]
            num_examples = len(batch_output['true'])
            
            for idx in range(num_examples):
                pred_tmp = torch.argmax(prob[idx])
                prob_tmp = prob[idx][1]
                true_tmp = true[idx]
                pred_list.append(pred_tmp.cpu())
                prob_list.append(prob_tmp.cpu())
                true_list.append(true_tmp.cpu())
                wsi_name_list.append(wsi_name)

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
        spec_95, spec_97, spec_98, spec_99, spec_100 = get_sens_spec_metrics(true, prob)
        
        print('='*50)
        print("AUC-ROC:", auc_roc)
        print("AUC-PR:", auc_pr)
        print("Specifity_at_97_Sensitivity:", spec_97)
        print("Specifity_at_98_Sensitivity:", spec_98)
        print("Specifity_at_99_Sensitivity:", spec_99)

    def process_files(self):
        
        # select the slides according to selected fold_nr and split_nr
        # independent test set should have split_nr all equal to 3
        data_info = pd.read_csv(self.data_info)
        file_list = []
        for row in data_info.iterrows():
            wsi_name = row[1].iloc[0]
            if row[1].iloc[self.fold_nr] == self.split_nr:
                file_list.append(f"{self.data_path}/{wsi_name}.dat")
        file_list.sort()  # to always ensure same input ordering
        
        print('Number of WSI graphs:', len(file_list))
        print('-'*50)

        pred, prob, true, wsi_names = self.__run_model(file_list)
        
        # save results to a single csv file
        df = pd.DataFrame(data = {'wsi_name': wsi_names, 'score': prob, "pred": pred, 'label': true})
        df.to_csv(f"{self.output_path}/results.csv")
        
        # get stats
        true, prob  = get_labels_scores(wsi_names, prob, data_info)
        self.__get_stats(prob, true)


#-------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    args = docopt(__doc__, version='IGUANA Inference v1.0')
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args['--gpu']
    
    # get the subset of features to be input to the GNN
    with open("features.yml") as fptr:
        feat_names = list(yaml.full_load(fptr).values())[0]
    
    # load node degree
    stats_path = args["--stats_dir"]
    if args["--model_name"] == "pna":
        node_degree = np.load(f"{stats_path}/node_deg.npy")
    else:
        node_degree = None

    if not os.path.exists(args["--output_dir"]):
        rm_n_mkdir(args["--output_dir"])
    
    #TODO Batch size must be set at 1 at the moment - fix this!
    args = {
        "model_name": args["--model_name"],
        "model_path": args["--model_path"],
        "stats_path": stats_path,
        "node_degree": node_degree,
        "data_path": args["--data_dir"],
        "data_info": args["--data_info"],
        "feat_names": feat_names,
        "batch_size": int(args["--batch_size"]), 
        "nr_procs": int(args["--num_workers"]),
        "output_path": args["--output_dir"],
        "fold_nr": int(args["--fold_nr"]),
        "split_nr": int(args["--split_nr"]),
    }
 
    infer = Infer(**args)
    infer.process_files()

