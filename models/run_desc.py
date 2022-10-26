import numpy as np
import torch
import torch.nn.functional as F
from itertools import chain
from sklearn.metrics import confusion_matrix, roc_auc_score

import sys
sys.path.append("..")
from metrics.stats_utils import get_auc_pr_sen_spec_metrics_abnormal
from misc.utils import ranking_loss


def train_step(batch_data, run_info, device="cuda"):
    run_info, state_info = run_info

    weight = torch.tensor([1, 1]).to(device).type(torch.float32)

    loss_func_dict = {
        "ce": torch.nn.CrossEntropyLoss(weight),
        "ranking": ranking_loss
    }
    # use 'ema' to add for EMA calculation, must be scalar!
    result_dict = {"EMA": {}}
    track_value = lambda name, value: result_dict["EMA"].update({name: value})

    ####
    batch = batch_data.batch
    edge_index = batch_data.edge_index
    feats = batch_data.x
    label = batch_data.y

    #! the first 2 columns are the wsi name and object ids, respectively. 
    #! Need to be dropped before passing to GCN!
    # feats = feats[: 2:]

    # data is 3-class -> convert to 2 class (normal vs abnormal)
    label_orig = label.clone()  # make copy of original 3 class label for evaluation
    label[label > 1] = 1

    batch = batch.to(device).type(torch.int64)
    edge_index = edge_index.to(device).type(torch.long)
    feats = feats.to(device).type(torch.float32)
    label = torch.squeeze(label.to(device).type(torch.int64))

    ####
    model = run_info["net"]["desc"]
    optimizer = run_info["net"]["optimizer"]
    ####
    model.train()
    model.zero_grad()  # not rnn so not accumulate
    out_dict = model(feats, edge_index, batch)
    out = out_dict["output_log"]
    prob = out

    ####
    loss = 0
    loss_opts = run_info["net"]["extra_info"]["loss"]

    for loss_name, loss_weight in loss_opts.items():
        loss_func = loss_func_dict[loss_name]
        loss_args = [out, label]
        term_loss = loss_func(*loss_args)
        track_value("loss_%s" % loss_name, term_loss.cpu().item())
        loss += loss_weight * term_loss
    loss = torch.unsqueeze(loss, 0)

    track_value("overall_loss", loss.cpu().item())
    # * gradient update

    # torch.set_printoptions(precision=10)
    loss.backward()
    optimizer.step()
    ####

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict["raw"] = {"true": label, "true_orig": label_orig, "prob": prob}

    return result_dict


def valid_step(batch_data, run_info, device="cuda"):
    run_info, state_info = run_info
    ####
    batch = batch_data.batch
    edge_index = batch_data.edge_index
    feats = batch_data.x
    label = batch_data.y

    # data is 3-class -> convert to 2 class (normal vs abnormal)
    label_orig = label.clone()  # make copy of original 3 class label for evaluation
    label[label > 1] = 1

    batch = batch.to(device).type(torch.int64)
    edge_index = edge_index.to(device).type(torch.long)
    feats = feats.to(device).type(torch.float32)
    label = label.to(device).type(torch.int64)

    ####
    model = run_info["net"]["desc"]
    model.eval()  # infer mode

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        out_dict = model(feats, edge_index, batch)
        prob = out_dict["output"]

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict = {"raw": {"true": label, "true_orig": label_orig, "prob": prob}}

    return result_dict


def infer_step(batch_data, model, device="cuda"):

    ####
    batch = batch_data.batch
    edge_index = batch_data.edge_index
    feats = batch_data.x
    label = batch_data.y
    wsi_info = batch_data.wsi_info

    # data is 3-class -> convert to 2 class (normal vs abnormal)
    label_orig = label.clone()  # make copy of original 3 class label for evaluation
    label[label > 1] = 1

    batch = batch.to(device).type(torch.int64)
    edge_index = edge_index.to(device).type(torch.long)
    feats = feats.to(device).type(torch.float32)
    label = label.to(device).type(torch.int64)

    ####
    model.eval()  # infer mode

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        out_dict = model(feats, edge_index, batch)
        prob = out_dict["output"]
        node_scores = out_dict["node_scores"]

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict = {"true": label, "true_orig": label_orig, "prob": prob, "node_scores": node_scores, "wsi_info": wsi_info}

    return result_dict


def proc_valid_step_output(raw_data):
    # TODO: add auto populate from main state track list
    track_dict = {"scalar": {}, "image": {}}

    def track_value(name, value, vtype):
        return track_dict[vtype].update({name: value})

    def longlist2array(longlist):
        tmp = list(chain.from_iterable(longlist))
        return np.array(tmp).reshape((len(longlist),) + longlist[0].shape)

    prob = raw_data["prob"]
    true = raw_data["true"]
    true_orig = np.array(raw_data["true_orig"])
    num_examples = len(raw_data["true"])

    prob_list = []
    true_list = []
    for idx in range(num_examples):
        graph_prob = prob[idx][1]
        graph_true = true[idx]
        prob_list.append(graph_prob.cpu())
        true_list.append(graph_true.cpu())

    prob = np.array(prob_list)
    pred = prob.copy()
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    true = np.array(true_list)
    
    auc = roc_auc_score(true, prob)
    conf = confusion_matrix(true, pred)

    _, _, spec_95, spec_97, spec_98, spec_99, spec_100 = get_auc_pr_sen_spec_metrics_abnormal(true, prob)

    track_value("AUC-ROC", auc, "scalar")
    track_value("Specifity_at_95_Sensitivity", spec_95, "scalar")
    track_value("Specifity_at_97_Sensitivity", spec_97, "scalar")
    track_value("Specifity_at_98_Sensitivity", spec_98, "scalar")
    track_value("Specifity_at_99_Sensitivity", spec_99, "scalar")

    return track_dict

