import glob
import os
import shutil
import joblib
import numpy as np
import cv2
import torch

import sys 
sys.path.append("../")


def normalize(mask, dtype=np.uint8):
    return (255 * mask / np.amax(mask)).astype(dtype)


def get_bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def cropping_center(x, crop_shape, batch=False):
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]
    return x


def rm_n_mkdir(dir_path):
    if (os.path.isdir(dir_path)):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def get_files(data_dir_list, data_ext):
    """Given a list of directories containing data with extention 'data_ext',
    generate a list of paths for all files within these directories

    """
    data_files = []
    for sub_dir in data_dir_list:
        files_list = glob.glob(sub_dir + '/*' + data_ext)
        files_list.sort()  # ensure same order
        data_files.extend(files_list)

    return data_files


def remap_label(pred, by_size=False, ds_factor=None):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)

    """ 
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1   

    return new_pred


def get_inst_centroid(inst_map):
    inst_centroid_list = []
    inst_id_list = list(np.unique(inst_map))
    for inst_id in inst_id_list[1:]:  # avoid 0 i.e background
        mask = np.array(inst_map == inst_id, np.uint8)
        inst_moment = cv2.moments(mask)
        inst_centroid = [(inst_moment["m10"] / inst_moment["m00"]),
                         (inst_moment["m01"] / inst_moment["m00"])]
        inst_centroid_list.append(inst_centroid)
    return np.array(inst_centroid_list)


def get_local_feat_stats(file_list):
    """Calculate mean and standard deviation from features- used for normalisation 
    of input features before input to GCN.
    
    Args:
        file_list: list of .dat files containing features
    
    """
    # read the first file to get the number of features!
    feats_tmp = joblib.load(file_list[0])
    nr_feats = feats_tmp["local_feats"].shape[-1]  # must be same number of features in each file
    del feats_tmp

    print("Getting local feature statistics...")
    mean = []
    median = []
    std = []
    perc_25 = []
    perc_75 = []
    for idx in range(nr_feats):
        # first 2 indexes are not actually features (wsi name and object id)
        if idx == 0 or idx == 1:
            mean.append(0)
            median.append(0)
            std.append(0)
            perc_25.append(0)
            perc_75.append(0)
        else:
            accumulated_feats_tmp = []
            for filepath in file_list:
                feats = joblib.load(filepath)["local_feats"]  # hard assumption on .dat file
                feats = feats[:, idx].tolist()
                accumulated_feats_tmp.extend(np.float32(feats))
            mean.append(np.nanmean(np.array(accumulated_feats_tmp)))
            median.append(np.nanmedian(np.array(accumulated_feats_tmp)))
            std.append(np.nanstd(np.array(accumulated_feats_tmp)))
            perc_25.append(np.nanpercentile(np.array(accumulated_feats_tmp), q=25))
            perc_75.append(np.nanpercentile(np.array(accumulated_feats_tmp), q=75))

    del accumulated_feats_tmp

    # convert to numpy array
    local_mean = np.array(mean)
    local_median = np.array(median)
    local_std = np.array(std)
    local_perc_25 = np.array(perc_25)
    local_perc_75 = np.array(perc_75)

    print('Done!')

    return (
        local_mean, 
        local_median, 
        local_std, 
        local_perc_25, 
        local_perc_75
    )


def get_global_feat_stats(file_list):
    """Calculate mean and standard deviation from features- used for normalisation 
    of input features before input to GCN.
    
    Args:
        file_list: list of .dat files containing features
    
    """
    feats_tmp = joblib.load(file_list[0])
    nr_feats = feats_tmp["global_feats"].shape[-1]  # must be same number of features in each file
    del feats_tmp

    print("Getting global feature statistics...")    
    step = 200 # number of features to consider at a time (prevent blow up in memory)
    mean = []
    median = []
    std = []
    perc_25 = []
    perc_75 = []
    for idx in range(0, nr_feats+1, step):
        accumulated_feats = []
        for filepath in file_list:
            feats = joblib.load(filepath)["global_feats"]
            feats = np.array(feats)
            feats_sub = feats[idx:idx+step] # subset of feats
            accumulated_feats.append(np.float32(feats_sub))
        accumulated_feats = np.array(accumulated_feats)

        accumulated_feats[accumulated_feats == np.inf] = 0 # remove inf values (replace with 0)
        mean_tmp = np.nanmean(accumulated_feats, axis=0).tolist()
        median_tmp = np.nanmedian(accumulated_feats, axis=0).tolist()
        std_tmp = np.nanstd(accumulated_feats, axis=0).tolist()
        perc_25_tmp = np.nanpercentile(accumulated_feats, q=25, axis=0).tolist()
        perc_75_tmp = np.nanpercentile(accumulated_feats, q=75, axis=0).tolist()
        mean.extend(mean_tmp)
        median.extend(median_tmp)
        std.extend(std_tmp)
        perc_25.extend(perc_25_tmp)
        perc_75.extend(perc_75_tmp)

    # convert to numpy array
    global_mean = np.array(mean)
    global_median = np.array(median)
    global_std = np.array(std)
    global_perc_25 = np.array(perc_25)
    global_perc_75 = np.array(perc_75)

    print('Done!')

    return (
        global_mean, 
        global_median, 
        global_std, 
        global_perc_25, 
        global_perc_75
    )


def get_pna_deg(data_list, file_ext, save_path):
    """Compute the maximum in-degree in the training data. Only needed for PNA Conv."""

    from dataloader.graph_loader import FileLoader
    from torch_geometric.utils import degree

    print("Computing maximum node degree for PNA conv...")

    file_list = []
    for dir_name in data_list:
        file_list.extend(glob.glob("%s/*%s" % (dir_name, file_ext)))

    input_dataset = FileLoader(file_list, feat_stats=None, norm=None, data_clean=None)
    
    max_degree = -1
    for data in input_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in input_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    np.save(f"{save_path}/deg.npy", deg)

def ranking_loss(pred, true):
    """Ranking loss.
    
    Args:
        pred: prediction array
        true: ground truth array
    
    """
    loss = 0
    c = 0 
    z = torch.Tensor([0]).to("cuda").type(torch.float32)
    for i in range(len(true)-1):
        for j in range(i+1, len(true)):
            if true[i] != true[j]:
                c += 1
                dz = pred[i,-1]-pred[j,-1]
                dy = true[i]-true[j]                   
                loss += torch.max(z, 1.0-dy*dz)
    return loss/c

def refine_files(file_list, wsi_info):
    """Remove unwanted categories."""

    wsi_info["Diagnostic Category"].replace(
        {
            "Normal ": "Normal",
            "Abnormal: Non-Neoplastic ": "Abnormal: Non-Neoplastic",
            "Abnormal: Neoplastic ": "Abnormal: Neoplastic",
        },
        inplace=True,
    )

    refined_list = []
    for filename in file_list:
        wsiname = os.path.basename(filename)
        wsiname = wsiname[:-4]

        subset = wsi_info.loc[wsi_info["WSI no"] == wsiname]
        diagnosis = np.array(subset["Specific Diagnosis"])[0]
        if not isinstance(diagnosis, float):
            diagnosis = diagnosis.lower()

            category = np.array(subset["Diagnostic Category"])[0]
            if category == "Normal":
                category = 0
            if category == "Abnormal: Non-Neoplastic":
                category = 1
            if category == "Abnormal: Neoplastic":
                category = 2

            # clean up
            diagnosis = diagnosis.replace(",", " ")
            diagnosis = diagnosis.replace(":", " ")
            diagnosis = diagnosis.replace(".", " ")
            diagnosis = diagnosis.replace("?", " ")
            diagnosis = diagnosis.replace("-", " ")
            diagnosis_split = diagnosis.split(" ")

            if "spirochetosis" not in diagnosis_split and "melanosis" not in diagnosis_split and "malanosis" not in diagnosis_split:
                refined_list.append(filename)
    
    return refined_list


def get_feat_stats(file_list):
    """Calculate mean and standard deviation from features- used for normalisation
    of input features before input to GCN.

    Args:
        file_list: list of .mat files containing features

    """
    # read the first file to get the number of features!
    feats_tmp = sio.loadmat(file_list[0])["feats"]  # hard assumption on .mat file
    nr_feats = feats_tmp.shape[-1]  # must be same number of features in each file

    print("Getting feature statistics...")
    mean = []
    median = []
    std = []
    perc_25 = []
    perc_75 = []
    for idx in range(nr_feats):
        accumulated_feats_tmp = []
        for filepath in file_list:
            feats = sio.loadmat(filepath)["feats"]  # hard assumption on .mat file
            feats = feats[:, idx].tolist()
            accumulated_feats_tmp.extend(feats)
        mean.append(np.mean(np.array(accumulated_feats_tmp)))
        median.append(np.median(np.array(accumulated_feats_tmp)))
        std.append(np.std(np.array(accumulated_feats_tmp)))
        perc_25.append(np.percentile(np.array(accumulated_feats_tmp), q=25))
        perc_75.append(np.percentile(np.array(accumulated_feats_tmp), q=75))

    # convert to numpy array
    mean = np.array(mean)
    median = np.array(median)
    std = np.array(std)
    perc_25 = np.array(perc_25)
    perc_75 = np.array(perc_75)

    return mean, median, std, perc_25, perc_75