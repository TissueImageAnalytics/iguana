import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d

from torch_geometric.nn import (
    GINConv, 
    SAGEConv, 
    PNAConv, 
    EdgeConv, 
    GATConv,
    GlobalAttention,
    global_mean_pool, 
)
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import reset
from torch_scatter import scatter_add


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        

class GlobalAtt(torch.nn.Module):
    """GlobalAttenion but returning the attention scores."""
    def __init__(self, gate_nn, nn=None):
        super(GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)
        
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None,
        size: Optional[int] = None):
        r"""
        Args:
            x (Tensor): The input node features.
            batch (LongTensor, optional): A vector that maps each node to its
                respective graph identifier. (default: :obj:`None`)
            size (int, optional): The number of graphs in the batch.
                (default: :obj:`None`)
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.int64)

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = int(batch.max()) + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out, gate

class GraphSageLayer(nn.Module):
    """GraphSage layer.

    Args:
        in_ch: number of input channels.
        out_ch: number of output channels.

    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.gconv = SAGEConv(in_ch, out_ch)

    def forward(self, x, edge_index, freeze=False):
        if self.training:
            with torch.set_grad_enabled(not freeze):
                new_feat = self.gconv(x, edge_index)
        else:
            new_feat = self.gconv(x, edge_index)

        return new_feat

class GATLayer(nn.Module):
    """GAT layer.

    Args:
        in_ch: number of input channels.
        out_ch: number of output channels.

    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.gconv = GATConv(in_ch, out_ch)

    def forward(self, x, edge_index, freeze=False):
        if self.training:
            with torch.set_grad_enabled(not freeze):
                new_feat = self.gconv(x, edge_index)
        else:
            new_feat = self.gconv(x, edge_index)

        return new_feat


class GINLayer(nn.Module):
    """Graph Isomorphic Network layer.

    Args:
        in_ch: number of input channels.
        out_ch: number of output channels.

    """

    def __init__(self, in_ch, out_ch, act=False):
        super().__init__()
        act_dict = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "identity": nn.Sequential()}
        self.gconv = GINConv(
            nn.Sequential(
                nn.Linear(in_ch, out_ch),
                BatchNorm1d(out_ch),
                act_dict[act],
                nn.Linear(out_ch, out_ch),
                act_dict[act],
            )
        )

    def forward(self, x, edge_index, freeze=False):
        if self.training:
            with torch.set_grad_enabled(not freeze):
                new_feat = self.gconv(x, edge_index)
        else:
            new_feat = self.gconv(x, edge_index)

        return new_feat


class EdgeConvLayer(nn.Module):
    """Edge Conv layer.

    Args:
        in_ch: number of input channels.
        out_ch: number of output channels.

    """

    def __init__(self, in_ch, out_ch, act=False):
        super().__init__()
        act_dict = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "identity": nn.Sequential()}
        self.gconv = EdgeConv(
            nn.Sequential(
                nn.Linear(in_ch*2, out_ch),
                BatchNorm1d(out_ch),
                act_dict[act],
                nn.Linear(out_ch, out_ch),
                act_dict[act],
            ),
            aggr = 'mean'
        )

    def forward(self, x, edge_index, freeze=False):
        if self.training:
            with torch.set_grad_enabled(not freeze):
                new_feat = self.gconv(x, edge_index)
        else:
            new_feat = self.gconv(x, edge_index)

        return new_feat


class PNALayer(nn.Module):
    """PNA layer.

    Args:
        in_ch: number of input channels.
        out_ch: number of output channels.

    """

    def __init__(
        self,
        in_ch, 
        out_ch, 
        aggregators=['mean', 'min', 'max', 'std'], 
        scalers = ['identity', 'amplification', 'attenuation'],
        deg=None
        ):
        super().__init__()
        
        self.gconv = PNAConv(in_ch, out_ch, aggregators=aggregators, scalers=scalers, deg=deg)

    def forward(self, x, edge_index, freeze=False):
        if self.training:
            with torch.set_grad_enabled(not freeze):
                new_feat = self.gconv(x, edge_index)
        else:
            new_feat = self.gconv(x, edge_index)

        return new_feat


class NetDesc(nn.Module):
    """Network description."""

    def __init__(
        self,
        model_name,
        nr_features,
        nhid=[12,12,12],
        grph_dim=10,
        dropout_rate=0.03,
        use_edges=0,
        label_dim=2,
        agg="attention",
        return_prob=False,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.use_edges = use_edges
        self.agg = agg
        self.return_prob = return_prob

        if model_name == "graphsage":
            self.gconv1 = GraphSageLayer(nhid[0], nhid[1])
            self.gconv2 = GraphSageLayer(nhid[1], nhid[2])
        elif model_name == "gat":
            self.gconv1 = GATLayer(nhid[0], nhid[1])
            self.gconv2 = GATLayer(nhid[1], nhid[2])
        elif model_name == "gin":
            self.gconv1 = GINLayer(nhid[0], nhid[1], act="relu")
            self.gconv2 = GINLayer(nhid[1], nhid[2], act="relu")
        elif model_name == "pna":
            #! hard-coded at the moment!
            #TODO refactor
            deg = np.load("/root/lsf_workspace/graph_data/cobi/graph_data_refined/deg.npy")
            deg = torch.Tensor(deg)
            ######
            self.gconv1 = PNALayer(nhid[0], nhid[1], deg=deg)
            self.gconv2 = PNALayer(nhid[1], nhid[2], deg=deg)
        elif model_name == "edge":
            self.gconv1 = EdgeConvLayer(nhid[0], nhid[1], act="relu")
            self.gconv2 = EdgeConvLayer(nhid[1], nhid[2], act="relu")
        elif model_name == "linear":
            self.gconv1 = nn.Linear(nhid[0], nhid[1])
            self.gconv2 = nn.Linear(nhid[1], nhid[2])

        ## local 
        self.lin0 = nn.Linear(nr_features, nhid[0])
        self.lin_emb0 = nn.Linear(nhid[0], grph_dim)
        self.lin_emb1 = nn.Linear(nhid[1], grph_dim) 
        self.lin_emb2 = nn.Linear(nhid[2], grph_dim)

        self.lin_merge = nn.Linear(grph_dim*3, grph_dim)

        gate_nn = nn.Sequential(nn.Linear(grph_dim, 1))
        v_nn = nn.Sequential(nn.Linear(grph_dim, grph_dim))
        self.gpool = GlobalAtt(gate_nn, v_nn)
        # self.gpool = GlobalAttention(gate_nn, v_nn)
        
        self.lin_out = nn.Linear(grph_dim, label_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch):
        edge_attr = None
        x0 = self.lin0(x)
        x1 = self.gconv1(x0, edge_index)
        x2 = self.gconv2(x1, edge_index)
        ###
        x0 = self.dropout(F.relu(self.lin_emb0(x0)))
        x1 = self.dropout(F.relu(self.lin_emb1(x1)))
        x2 = self.dropout(F.relu(self.lin_emb2(x2)))
        ###
        x_combined = F.relu(self.lin_merge(torch.cat((x0, x1, x2), dim=1)))

        # pool over node-level features
        if self.agg == 'attention':
            att_pool = self.gpool(x_combined, batch)
            logits = att_pool[0]
            scores = att_pool[1]
            # logits = self.gpool(x_combined, batch)
            scores = None
        elif self.agg == 'mean':
            scores = self.lin_scores(x_combined)
            logits = global_mean_pool(scores, batch)

        output1 = F.softmax(self.lin_out(logits), dim=-1)
        output2 = self.lin_out(logits)
        
        if self.return_prob:
            output = output1
        else:
            output = {"output": output1, "output_log": output2, "node_scores": scores}

        return output

####
def create_model(**kwargs):
    return NetDesc(**kwargs)
