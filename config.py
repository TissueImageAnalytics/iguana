import importlib

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
        log_root = "/root/lsf_workspace/output/gland_graphs/logs"
        self.log_dir = f"{log_root}/{self.model_name}/fold{fold_nr}/{exp_nr}/" 

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)
        self.model_config_file = importlib.import_module("models.opt")
