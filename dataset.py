"""Dataset info. Modify this with your own data directories for training IGUANA."""


class __CoBi(object):
    """Defines the Colon Biopsy (CoBi) graph dataset"""

    def __init__(self, fold_nr):

        self.file_ext = ".dat"
        root_dir = '/mnt/gpfs01/lsf-workspace/tialab-simon/graph_data/cobi/graph_data_refined'
        self.all_dir_list = [f'{root_dir}/split1/', f'{root_dir}/split2/', f'{root_dir}/split3/']

        if fold_nr == 1:
            self.train_dir_list = [f'{root_dir}/split2/', f'{root_dir}/split3/']
            self.valid_dir_list = [f'{root_dir}/split1/']

        elif fold_nr == 2:
            self.train_dir_list = [f"{root_dir}/split1/", f"{root_dir}/split3/"]
            self.valid_dir_list = [f"{root_dir}/split2/"]

        elif fold_nr == 3:
            self.train_dir_list = [f"{root_dir}/split1/", f"{root_dir}/split2/"]
            self.valid_dir_list = [f"{root_dir}/split3/"]


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
