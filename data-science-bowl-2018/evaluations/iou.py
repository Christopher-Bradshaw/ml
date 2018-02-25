"""
Code taken originally from https://www.kaggle.com/wcukierski/example-metric-implementation

I don't fully understand this but have tested that it works...
"""
import numpy as np
from scipy.ndimage import label

import plot_helpers.plot as plot

def iou_loss(pred, true):
    iou = _compute_iou(pred, true)
    if iou is None:
        return 0
    perc = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = _precision_at(t, iou)
        p = tp / (tp + fp + fn)
        perc.append(p)
    return np.mean(perc)

def log_iou(pred, true):
    iou = _compute_iou(pred, true)
    perc = []
    print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = _precision_at(t, iou)
        assert tp + fp + fn != 0
        p = tp / (tp + fp + fn)
        print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        perc.append(p)
    print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(perc)))

def _compute_iou(pred, true):
    assert pred.shape == true.shape
    if np.max(pred) == 0 or np.max(true) == 0:
        return None

    pred, num_pred = label(pred)
    true, num_true = label(true)
    num_pred += 1
    num_true += 1 # To account for the empty 0s
    # print(num_pred, num_true)
    # plot.imshow(pred)

    intersection = np.histogram2d(true.flatten(), pred.flatten(), bins=(num_pred, num_true))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(true, bins = num_true)[0]
    area_true = np.expand_dims(area_true, 0)

    area_pred = np.histogram(pred, bins = num_pred)[0]
    area_pred = np.expand_dims(area_pred, -1)

    union = area_true + area_pred - intersection
    union = union[1:,1:]
    intersection = intersection[1:,1:]


    iou = intersection / union
    return iou

# Precision helper function
def _precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn
