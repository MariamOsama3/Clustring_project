from sklearn.metrics import silhouette_score
from scipy.stats import mode
import numpy as np

def calculate_silhouette_score(data, labels):
    """Calculate silhouette score"""
    return silhouette_score(data, labels)

def purity_score(y_true, y_pred):
    """Calculate clustering purity score"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    clusters = np.unique(y_pred)
    correct_preds = 0

    for cluster in clusters:
        indices = np.where(y_pred == cluster)[0]
        true_labels = y_true[indices]

        if len(true_labels) == 0:
            continue

        majority_label = mode(true_labels, keepdims=True).mode[0]
        correct_preds += np.sum(true_labels == majority_label)

    return correct_preds / len(y_true)