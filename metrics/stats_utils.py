import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score


def get_auc_pr_sen_spec_metrics_abnormal(targets, probs):
    """Fill in stats."""
    num_classes = 2
    binary_labels = np.array(targets.copy())
    # binary_labels = label_binarize(np.array(targets), classes=[i for i in range(num_classes)])
    temp_auc = []
    temp_ap = []
    specificity_at_95 = []
    specificity_at_97 = []
    specificity_at_98 = []
    specificity_at_99 = []
    specificity_at_100 = []

    fpr, tpr, thresholds_auc = roc_curve(binary_labels, probs)
    temp_auc.append(auc(fpr, tpr))
    precision, recall, thresholds_pr = precision_recall_curve(binary_labels, probs)
    #temp_ap.append(auc(recall, precision))
    temp_ap.append(average_precision_score(binary_labels, probs))
    tnr = 1 - fpr
    specificity_at_95 = tnr[tpr >= 0.95][0]
    specificity_at_97 = tnr[tpr >= 0.97][0]
    specificity_at_98 = tnr[tpr >= 0.98][0]
    specificity_at_99 = tnr[tpr >= 0.99][0]
    specificity_at_100 = tnr[tpr >= 1.0][0]

    return temp_auc, temp_ap, specificity_at_95, specificity_at_97, specificity_at_98, specificity_at_99, specificity_at_100