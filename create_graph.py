"""create_graph.py

Generates a graph along with node-level features from a WSI and it's instance-level mask.

Usage:
  create_graph.py [--input_dir=<path>] [--output_dir=<path>] [--dist_thresh=<n>] [--k=<n>] [--data_info==<path>]
  create_graph.py (-h | --help)
  create_graph.py --version

Options:
  -h --help             Show this string.
  --version             Show version.
  --input_dir=<path>    Path to where input features are stored.
  --output_dir=<path>   Path where graph data will be saved.
  --dist_thresh=<n>     Only connect nodes if the distance between them is below a threshold [default: 500]
  --data_info=<path>    Data csv file containing fold information and labels.
  
"""

import glob
import os
import joblib
import pandas as pd
import numpy as np
from docopt import docopt

from misc.utils import rm_n_mkdir


def preprocess_local_feats(feats, fill_mode="mean"):
    """Preprocess local feats. Fill in nan values."""

    output_feats = {} # output deals with missing values (denoted as nan)
    for feat_name, feat_vals in feats.items():
        feat_vals = np.array(feat_vals)
        if feat_name != "obj_id":
            if fill_mode == "mean":
                fill = np.nanmean(feat_vals)
            elif fill_mode == "median":
                fill = np.nanmedian(feat_vals)
            elif fill_mode == "zeros":
                fill = 0
            
            # fill nans
            feat_vals[np.isnan(feat_vals)] = fill 
        output_feats[feat_name] = feat_vals

    return output_feats


def get_edge_index(dst_matrix, tissue_idx, dist_thresh):
    """Get the edge matrix from adjacency matrix. Connects nodes if they are within a certain distance.
    
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


def construct_graphs(list_files, data_info, dist_thresh, output_path):
    """Construct graph."""
    
    list_names = list(data_info["wsi_name"])
    for filename in list_files:
        local_feats = f"{filename}/local_feats.dat"
        dst_matrix = f"{filename}/dst_matrix.npy"
        tissue_idx = f"{filename}/tissue_idx.npy"
        if os.path.isfile(local_feats) and os.path.isfile(dst_matrix) and os.path.isfile(tissue_idx):
            wsi_name = os.path.basename(filename)
            if wsi_name in list_names:
                label = int(data_info[data_info["wsi_name"] == wsi_name].label)
                # check if file exists - folder may have been created without generating features!
                if os.path.isfile(local_feats):
                    # make sure graph has not already been constructed
                    if not os.path.isfile(f"{output_path}/{wsi_name}.dat"):
                        # load local features and perform preprocessing
                        local_feats = joblib.load(local_feats)
                        local_feats_proc = preprocess_local_feats(local_feats, fill_mode="mean")
                        
                        # load other graph-related data
                        dst_matrix = np.load(dst_matrix)
                        tissue_idx = np.load(tissue_idx)
                        
                        # compute edge index
                        edge_index = get_edge_index(dst_matrix, tissue_idx, dist_thresh)

                        data = {"local_feats": local_feats_proc, "edge_index": edge_index, "label": label, "wsi_name": wsi_name}
                        joblib.dump(data, f"{output_path}/{wsi_name}.dat")


#---------------------------------------------------------------------------------------

if __name__ == '__main__':
    args = docopt(__doc__)

    dist_thresh = int(args['--dist_thresh'])
    input_files = args["--input_dir"]
    output_path = args["--output_dir"]
    data_info = pd.read_csv(args["--data_info"])

    list_files = glob.glob(input_files + "*")
    
    # create output directory
    if not os.path.exists(output_path):
        rm_n_mkdir(output_path)

    construct_graphs(list_files, data_info, dist_thresh, output_path)


    