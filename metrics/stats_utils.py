import numpy as np
from sklearn.metrics import roc_curve


def get_sens_spec_metrics(targets, probs):
    """Get the specificity at different sensitivity cut-offs.
    
    Args:
        targets (list): list of true labels
        probs (list): list of predicted scores
    
    Returns:

        """
    binary_labels = np.array(targets.copy())
    specificity_at_95 = []
    specificity_at_97 = []
    specificity_at_98 = []
    specificity_at_99 = []
    specificity_at_100 = []

    fpr, tpr, _ = roc_curve(binary_labels, probs)
    tnr = 1 - fpr
    specificity_at_95 = tnr[tpr >= 0.95][0]
    specificity_at_97 = tnr[tpr >= 0.97][0]
    specificity_at_98 = tnr[tpr >= 0.98][0]
    specificity_at_99 = tnr[tpr >= 0.99][0]
    specificity_at_100 = tnr[tpr >= 1.0][0]

    return specificity_at_95, specificity_at_97, specificity_at_98, specificity_at_99, specificity_at_100