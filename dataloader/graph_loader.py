import numpy as np
import joblib
import torch

from torch_geometric.data import Data


class FileLoader(torch.utils.data.Dataset):
    """Data loader for graph data. The loader will use features defined in features.yml."""

    def __init__(self, file_list, feat_names, feat_stats, norm, data_clean):
        self.file_list = file_list
        self.feat_stats = feat_stats
        self.feat_names = feat_names
        
        if feat_stats is not None:
            self.mean_stats = np.array([feat_stats["mean"][k] for k in self.feat_names])
            self.median_stats = np.array([feat_stats["median"][k] for k in self.feat_names])
            self.std_stats = np.array([feat_stats["std"][k] for k in self.feat_names])
            self.perc_25_stats= np.array([feat_stats["perc_25"][k] for k in self.feat_names])
            self.perc_75_stats= np.array([feat_stats["perc_75"][k] for k in self.feat_names])
        
        # get lower and upper bounds for clipping data (outlier removal)
        if data_clean == 'std':
            self.local_lower_bounds = self.mean_stats - 3 * self.std_stats
            self.local_upper_bounds = self.mean_stats + 3 * self.std_stats
        elif data_clean == 'iqr':
            iqr = self.perc_75_stats - self.perc_25_stats
            self.local_lower_bounds = self.perc_25_stats - 2 * iqr
            self.local_upper_bounds = self.perc_75_stats + 2 * iqr

        self.norm = norm

        assert (
            self.norm == "standard" or self.norm == "robust" or self.norm == None
        ), "`norm` must be `standard` or `robust`."
        self.data_clean = data_clean
        assert (
            self.data_clean == "std" or self.data_clean == "iqr" or self.data_clean == None
        ), "`data_clean` must be `std` or `iqr`."
        return

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        data = joblib.load(path)

        edge_index = np.array(data["edge_index"])
        feats = data["local_feats"]
        
        nr_local_feats = len(self.feat_names)

        feats_sub = dict((k, feats[k]) for k in self.feat_names) # get subset of features
        feats_sub = np.array(list(feats_sub.values())).astype("float32") # convert to array
        feats_sub = np.transpose(feats_sub) # ensure NxF
        wsi_name = data["wsi_name"]
        obj_id = feats["obj_id"]
    
        # clean up data - deal with outliers!
        if self.data_clean is not None:
            # local feats
            clipped_feats = []
            for idx in range(nr_local_feats):
                feat_single = feats_sub[:, idx]
                feat_single[feat_single > self.local_upper_bounds[idx]] = self.local_upper_bounds[idx]
                feat_single[feat_single < self.local_lower_bounds[idx]] = self.local_lower_bounds[idx]
                clipped_feats.append(feat_single)
            feats_sub = np.squeeze(np.dstack(clipped_feats), axis=0)

        # normalise the feature subset
        if self.norm == "standard":
            feats_sub = ((feats_sub - self.mean_stats) + 1e-8) / (self.std_stats + 1e-8)
        elif self.norm == "robust":
            feats_sub = ((feats_sub - self.median_stats) + 1e-8) / ((self.perc_75_stats - self.perc_25_stats) + 1e-8)

        label = np.array([data["label"]])
    
        x = torch.Tensor(feats_sub).type(torch.float)
        edge_index = torch.Tensor(edge_index).type(torch.long)
        label = torch.Tensor(label).type(torch.float)

        return Data(x=x, edge_index=edge_index, y=label, obj_id=obj_id, wsi_name=wsi_name)
