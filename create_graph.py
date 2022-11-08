"""create_graph.py

Generates a graph along with node-level features from a WSI and it's instance-level mask

Usage:
  create_graph.py [--dist_thresh=<n>] [--k=<n>] [--save_graph]
  create_graph.py (-h | --help)
  create_graph.py --version

Options:
  -h --help           Show this string.
  --version           Show version.
  --dist_thresh=<n>   Only connect nodes if the distance between them is below a threshold [default: 500]
  --save_graph        Whether to save the connected graph each time
  
"""

import glob
import os
import math
import joblib
from threading import local
import pandas as pd
import numpy as np
from docopt import docopt



def preprocess_local_feats(feats, fill_mode="mean"):
    """Preprocess local feats. Fill in nan values."""

    all_feats = []
    id_list = []
    for obj_id, feats_local_dict in feats.items():
        feats_list = []
        id_list.append(obj_id)
        for feat_name, feats_local in feats_local_dict.items():
            if math.isnan(feats_local):
                feats_local = np.nan
            else:
                feats_local = float(feats_local)
            feats_list.append(feats_local)
        all_feats.append(feats_list)
    all_feats = np.array(all_feats)

    nr_feats = all_feats.shape[1]
    output_list = []
    for feat_idx in range(nr_feats):
        feats_tmp = all_feats[:, feat_idx]
        if fill_mode == "mean":
            fill = np.nanmean(feats_tmp)
        elif fill_mode == "median":
            fill = np.nanmedian(feats_tmp)
        elif fill_mode == "ones":
            fill = 0
        
        if math.isnan(fill):
            fill = 0

        # fill nans
        feats_tmp[np.isnan(feats_tmp)] = fill 
        output_list.append(feats_tmp)
    
    processed_output = np.squeeze(np.dstack(output_list))
    id_list = np.array(id_list)

    return id_list, processed_output


def get_edge_index(dst_matrix, tissue_idx, dist_thresh):
    """Get the edge matrix from adjacency matrix.
    
    Returns:
        edge_matrix: 2xN array of object indicies showing the presence of edges
    
    """
    
    nr_objs = dst_matrix.shape[0]
    origin = []
    dest = []

    for i in range(nr_objs):
        for j in range(nr_objs):
            # check to make sure distance is under a threshold
            if dst_matrix[i, j] < dist_thresh:
                # check to make sure two nodes are within the same tissue region
                if tissue_idx[i] == tissue_idx[j]:
                    origin.append(i)
                    dest.append(j)

    origin = np.array(origin)
    origin = np.expand_dims(origin, 0)
    dest = np.array(dest)
    dest = np.expand_dims(dest, 0)

    edge_index = np.concatenate((origin, dest), 0)
    edge_index = edge_index.astype("int32")

    return edge_index


def construct_graphs(files, dist_thresh, output_path):
    """Construct graph."""
    
    for fold_nr, files_info_all in files.items():
        print(f'Creating graph data from fold {fold_nr}')
        for files_info in files_info_all: 
            local_feats_file = files_info[0]
            # check file exists - folder may have been created without features file!
            if os.path.isfile(local_feats_file):
                dst_matrix_file = files_info[2]
                label = files_info[3]
                wsi_name = files_info[4]
                tissue_idx_file = files_info[5]

                if not os.path.isfile(f"{output_path}{fold_nr}/{wsi_name}.dat"):

                    local_feats = joblib.load(local_feats_file)
                    id_list, local_feats_proc = preprocess_local_feats(local_feats, fill_mode="mean")

                    # merge wsi_name and id_list - easier to work with during GCN
                    wsi_name_list = [wsi_name] * id_list.shape[0]
                    wsi_name_array = np.expand_dims(np.array(wsi_name_list), -1)
                    id_list = np.expand_dims(id_list, -1)

                    merged_feats = np.concatenate((wsi_name_array, id_list, local_feats_proc), axis=-1)
                    
                    dst_matrix = np.load(dst_matrix_file)
                    tissue_idx = np.load(tissue_idx_file)
                    edge_index = get_edge_index(dst_matrix, tissue_idx, dist_thresh)

                    data = {"local_feats": merged_feats, "edge_index": edge_index, "label": label}
                    joblib.dump(data, f"{output_path}{fold_nr}/{wsi_name}.dat")


def get_files(list_files, dev_info, test=False):
    """Get files."""
    
    out_dict = {}
    fold1_list = []
    fold2_list = []
    fold3_list = []
    test_list = []
    list_names = list(dev_info["wsi_id"])
    for filename in list_files:
        local_feats = f"{filename}/local_feats.dat"
        dst_matrix = f"{filename}/dst_matrix.npy"
        tissue_idx = f"{filename}/tissue_idx.npy"
        if os.path.isfile(local_feats) and os.path.isfile(dst_matrix) and os.path.isfile(tissue_idx):
            basename = os.path.basename(filename)
            if basename in list_names:
                label = int(dev_info[dev_info["wsi_id"] == basename].label_id)
                if not test:
                    fold1_id = int(dev_info[dev_info["wsi_id"] == basename].fold1_id)
                    fold2_id = int(dev_info[dev_info["wsi_id"] == basename].fold2_id)
                    fold3_id = int(dev_info[dev_info["wsi_id"] == basename].fold3_id)
                    if fold1_id == 1:
                        fold1_list.append([local_feats, dst_matrix, label, basename, tissue_idx])
                    elif fold2_id == 1:
                        fold2_list.append([local_feats, dst_matrix, label, basename, tissue_idx])
                    elif fold3_id == 1:
                        fold3_list.append([local_feats, dst_matrix, label, basename, tissue_idx])
                else:
                    test_list.append([local_feats, dst_matrix, label, basename, tissue_idx])
    if not test:
        out_dict['split1'] = fold1_list
        out_dict['split2'] = fold2_list
        out_dict['split3'] = fold3_list
    else:
        out_dict['test'] = test_list
    
    return out_dict


#---------------------------------------------------------------------------------------

if __name__ == '__main__':
    args = docopt(__doc__)

    dist_thresh = int(args['--dist_thresh'])
    save_graph = args['--save_graph']

    input_files = "/root/lsf_workspace/proc_slides/imp/feats/"
    dev_info = pd.read_csv("/root/lsf_workspace/graph_data/imp/test_info_imp.csv")

    output_path = "/root/lsf_workspace/graph_data/imp/graph_data/"

    list_files = glob.glob(input_files + "*")

    files_per_fold = get_files(list_files, dev_info, test=True)

    construct_graphs(files_per_fold, dist_thresh, output_path)


    