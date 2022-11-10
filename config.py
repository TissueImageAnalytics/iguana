import importlib
import yaml

from dataset import get_dataset


class Config(object):
    """Configuration file."""

    def __init__(self):

        self.seed = 30 # random seed
        self.logging = True

        # iguana used `pna` - but we include options to use other graph convolution layers
        self.model_name = "pna"  # choose from `gin`, `graphsage`, `pna`, `edge`, `gat`

        fold_nr = 1
        # determines which dataset to be used during training / inference. The appropriate class
        # is initialised in dataset.py. Refer to dataset.py for info regarding data paths.
        # choose from `cobi_fold1`, `cobi_fold2`, `cobi_fold3`
        #! alternatively for a custom dataset, make your own class in dataset.py
        self.dataset_name = f"cobi_fold{fold_nr}"

        # log directory where checkpoints are saved
        exp_nr = "v1.0"
        log_root = "logs"
        self.log_dir = f"{log_root}/{self.model_name}/fold{fold_nr}/{exp_nr}/" 
        
        # get the subset of feature names that will be considered by the GNN
        with open("features.yml") as fptr:
            self.feat_names = list(yaml.full_load(fptr).values())[0]

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)
        self.model_config_file = importlib.import_module("models.opt")
