import numpy as np
import joblib
import torch
import yaml

from torch_geometric.data import Data


class FileLoader(torch.utils.data.Dataset):
    """Data loader for graph data. The loader will use features defined in features.yml."""

    def __init__(self, file_list, feat_names, feat_stats, norm, data_clean):
        self.file_list = file_list
        self.feat_stats = feat_stats
        
        # get lower and upper bounds for clipping data (outlier removal)
        if data_clean == 'std':
            self.local_lower_bounds = feat_stats[0][self.local_feats_idx] - 3 * feat_stats[2][self.local_feats_idx]
            self.local_upper_bounds = feat_stats[0][self.local_feats_idx] + 3 * feat_stats[2][self.local_feats_idx]
        elif data_clean == 'iqr':
            iqr = feat_stats[4][self.local_feats_idx] - feat_stats[3][self.local_feats_idx]
            self.local_lower_bounds = feat_stats[3][self.local_feats_idx] - 2 * iqr
            self.local_upper_bounds = feat_stats[4][self.local_feats_idx] + 2 * iqr

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
        feats = np.array(data["local_feats"])
        
        nr_local_feats = len(self.feat_names)
        
        feats_sub = feats[:, self.local_feats_idx].astype('float32') #! this isn't the right way of subsetting a dict
        wsi_name = feats["wsi_name"]
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
            # local feats
            local_mean = self.feat_stats[0][self.local_feats_idx]
            local_std = self.feat_stats[2][self.local_feats_idx]
            feats_sub = ((feats_sub - local_mean) + 1e-8) / local_std + 1e-8
        elif self.norm == "robust":
            # local feats
            local_median = self.feat_stats[1][self.local_feats_idx]
            local_perc_25 = self.feat_stats[3][self.local_feats_idx]
            local_perc_75 = self.feat_stats[4][self.local_feats_idx]
            feats_sub = ((feats_sub - local_median) + 1e-8) / ((local_perc_75 - local_perc_25) + 1e-8)

        label = np.array([data["label"]])

        # data is 3-class -> convert to 2 class (normal vs abnormal)
        # label[label > 1] = 1

        x = torch.Tensor(feats_sub).type(torch.float)
        edge_index = torch.Tensor(edge_index).type(torch.long)
        label = torch.Tensor(label).type(torch.float)

        return Data(x=x, edge_index=edge_index, y=label, obj_id=obj_id, wsi_name=wsi_name)
