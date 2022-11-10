"""Dataset info. Modify this with your own data directories for training IGUANA."""

import glob
import pandas as pd
import numpy as np


class __CoBi(object):
    """Defines the Colon Biopsy (CoBi) graph dataset"""

    def __init__(self, fold_nr):

        file_ext = ".dat"
        root_dir = 'test_data/'
        
        self.stats_path = f"{root_dir}/stats"
        
        self.all_data = glob.glob(f"{root_dir}/data/*{file_ext}")
        
        # csv file - 1st column indicates the WSI name and each subsequent column gives the fold info
        # eg column 2 gives the info for fold1, column 3, gives the info for fold2, etc 
        # for fold info: 1 denotes training, 2 denotes validation and 3 denotes testing
        # if the dataset is an independent test set, use 1 fold info column, with all cells set to 3
        fold_info = pd.read_csv(f"{root_dir}/data_info.csv")

        wsi_names = np.array(fold_info.iloc[:, 0])
        fold_info = np.array(fold_info.iloc[:, fold_nr])

        wsi_train = wsi_names[fold_info==1]
        wsi_valid = wsi_names[fold_info==2]
        
        self.train_list = []
        for wsi_name in wsi_train:
            self.train_list.append(f"{root_dir}/data/{wsi_name}{file_ext}")
        
        self.valid_list = []
        for wsi_name in wsi_valid:
            self.valid_list.append(f"{root_dir}/data/{wsi_name}{file_ext}")


def get_dataset(name):
    """Return a pre-defined dataset object associated with `name`"""
    if name.lower() == "cobi_fold1":
        return __CoBi(fold_nr=1)
    elif name.lower() == "cobi_fold2":
        return __CoBi(fold_nr=2)
    elif name.lower() == "cobi_fold3":
        return __CoBi(fold_nr=3)
    else:
        assert False, "Unknown dataset `%s`" % name
