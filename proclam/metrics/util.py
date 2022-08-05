"""
Utility functions for PLAsTiCC metrics
"""

from __future__ import absolute_import, division
__all__ = ['sanitize_predictions',
           'weight_sum', 'check_weights', 'averager',
           'cm_to_rate', 'precision',
           'auc', 'check_auc_grid', 'prep_curve',
           'det_to_prob', 'prob_to_det',
           'det_to_cm']

import collections
import numpy as np
# import pycm
import sys
from scipy.integrate import trapz

RateMatrix = collections.namedtuple('rates', 'TPR FPR FNR TNR TP FP FN TN')


def sanitize_predictions(predictions, epsilon=1.e-8):
    """
    Replaces 0 and 1 with 0+epsilon, 1-epsilon

    Parameters
    ----------
    predictions: numpy.ndarray, float
        N*M matrix of probabilities per object, may have 0 or 1 values
    epsilon: float
        small placeholder number, defaults to floating point precision

    Returns
    -------
    predictions: numpy.ndarray, float
        N*M matrix of probabilities per object, no 0 or 1 values
    """
    assert epsilon > 0. and epsilon < 0.0005
    mask1 = (predictions < epsilon)
    mask2 = (predictions > 1.0 - epsilon)

    predictions[mask1] = epsilon
    predictions[mask2] = 1.0 - epsilon
    predictions = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
    return predictions


def weight_sum(per_class_metrics, weight_vector):
    """
    Calculates the weighted metric

    Parameters
    ----------
    per_class_metrics: numpy.ndarray, float
        vector of per-class scores
    weight_vector: numpy.ndarray, float
        vector of per-class weights

    Returns
    -------
    weight_sum: np.float
        The weighted metric
    """
    weight_sum = np.dot(weight_vector, per_class_metrics)
    return weight_sum


def check_weights(avg_info, M, chosen=None, truth=None):
    """
    Converts standard weighting schemes to weight vectors for weight_sum

    Parameters
    ----------
    avg_info: str or numpy.ndarray, float
        keyword about how to calculate weighted average metric
    M: int
        number of classes
    chosen: int, optional
        which class is to be singled out for down/up-weighting
    truth: numpy.ndarray, int, optional
        true class assignments

    Returns
    -------
    weights: numpy.ndarray, float
        relative weights per class

    Notes
    -----
    Assumes a random class
    """
    if type(avg_info) != str:
        avg_info = np.asarray(avg_info)
        weights = avg_info / np.sum(avg_info)
        assert(np.isclose(sum(weights), 1.))
    elif avg_info == 'per_class':
        weights = np.ones(M) / float(M)
    elif avg_info == 'per_item':
        classes, counts = np.unique(truth, return_counts=True)
        weights = np.zeros(M)
        weights[classes] = counts / float(len(truth))
        assert len(weights) == M
    elif avg_info == 'flat':
        weights = np.ones(M)
    elif avg_info == 'up' or avg_info == 'down':
        if chosen is None:
            chosen = np.random.randint(M)
        if avg_info == 'up':
            weights = np.ones(M) / np.float(M)
            weights[chosen] = 1.
        elif avg_info == 'down':
            weights = np.ones(M)
            weights[chosen] = 1./np.float(M)
    else:
        print('something has gone wrong with avg_info '+str(avg_info))
        weights = None
    return weights


def averager(per_object_metrics, truth, M, vb=False):
    """
    Creates a list with the metrics per object, separated by class

    Notes
    -----
    There is currently a kludge for when there are no true class members, causing an improvement when that class is upweighted due to increasing the weight of 0.
    """
    group_metric = per_object_metrics
    class_metric = np.empty(M)
    for m in range(M):
        true_indices = np.where(truth == m)[0]
        how_many_in_class = len(true_indices)
        try:
            assert(how_many_in_class > 0)
            per_class_metric = group_metric[true_indices]
            # assert(~np.all(np.isnan(per_class_metric)))
            class_metric[m] = np.average(per_class_metric)
        except AssertionError:
            class_metric[m] = 0.
        if vb:
            print('by request '+str((m, how_many_in_class, class_metric[m])))
    return class_metric


def cm_to_rate(cm, vb=False):
    """
    Turns a confusion matrix into true/false positive/negative rates

    Parameters
    ----------
    cm: numpy.ndarray, int or float
        confusion matrix, first axis is predictions, second axis is truth
    vb: boolean, optional
        print progress to stdout?

    Returns
    -------
    rates: named tuple, float
        RateMatrix named tuple

    Notes
    -----
    This can be done with a mask to weight the classes differently here.
    """
    cm = cm.astype(float)
    # if vb: print('by request cm '+str(cm))
    tot = np.sum(cm)
    # mask = range(len(cm))
    # if vb: print('by request sum '+str(tot))

    T = np.sum(cm, axis=1)
    F = tot[np.newaxis] - T
    P = np.sum(cm, axis=0)
    N = tot[np.newaxis] - P
    # if vb: print('by request T, F, P, N'+str((T, F, P, N)))

    TP = np.diag(cm)
    FN = P - TP
    TN = F - FN
    FP = T - TP
    # if vb: print('by request TP, FP, FN, TN'+str((TP, FP, FN, TN)))

    TPR = TP / P
    FPR = FP / N
    FNR = FN / P
    TNR = TN / N
    # if vb: print('by request TPR, FPR, FNR, TNR'+str((TPR, FPR, FNR, TNR)))

    rates = RateMatrix(TPR=TPR, FPR=FPR, FNR=FNR, TNR=TNR, TP=TP, FN=FN, TN=TN, FP=FP)
    # if vb: print('by request TPR, FPR, FNR, TNR '+str(rates))

    return rates


def prep_curve(x, y):
    """
    Makes a curve for AUC

    Parameters
    ----------
    x: numpy.ndarray, float
        x-axis
    y: numpy.ndarray, float
        y-axis

    Returns
    -------
    x: numpy.ndarray, float
        x-axis
    y: numpy.ndarray, float
        y-axis
    """
    x = np.concatenate(([0.], x, [1.]),)
    y = np.concatenate(([0.], y, [1.]),)
    return (x, y)


def auc(x, y):
    """
    Computes the area under curve (just a wrapper for trapezoid rule)

    Parameters
    ----------
    x: numpy.ndarray, float
        x-axis
    y: numpy.ndarray, float
        y-axis

    Returns
    -------
    auc: float
        the area under the curve
    """
    i = np.argsort(x)
    auc = trapz(y[i], x[i])
    return auc


def check_auc_grid(grid):
    """
    Checks if a grid for an AUC metric is valid

    Parameters
    ----------
    grid: numpy.ndarray, float or float or int
        array of values between 0 and 1 at which to evaluate AUC or grid spacing or number of grid points

    Returns
    -------
    thresholds_grid: numpy.ndarray, float
        grid of thresholds
    """
    if type(grid) == list or type(grid) == np.ndarray:
        thresholds_grid = np.concatenate((np.zeros(1), np.array(grid), np.ones(1)))
    elif type(grid) == float:
        if grid > 0. and grid < 1.:
            thresholds_grid = np.arange(0., 1., grid)
        else:
            thresholds_grid = None
    elif type(grid) == int:
        if grid > 0:
            thresholds_grid = np.linspace(0., 1., grid)
        else:
            thresholds_grid = None
    try:
        assert thresholds_grid is not None
        return np.sort(thresholds_grid)
    except AssertionError:
        print('Please specify a grid, spacing, or density for this AUC metric.')
        return


def det_to_prob(dets, prediction=None):
    """
    Reformats vector of class assignments into matrix with 1 at true/assigned class and zero elsewhere

    Parameters
    ----------
    dets: numpy.ndarray, int
        vector of classes
    prediction: numpy.ndarray, float, optional
        predicted class probabilities

    Returns
    -------
    probs: numpy.ndarray, float
        matrix with 1 at input classes and 0 elsewhere

    Notes
    -----
    formerly truth_reformatter
    Does not yet handle number of classes in truth not matching number of classes in prediction, i.e. for having "other" class or secret classes not in training set.  The prediction keyword is a kludge to enable this but should be replaced.
    """
    N = len(dets)
    indices = range(N)

    if prediction is None:
        prediction_shape = (N, int(np.max(dets) + 1))
    else:
        prediction, dets = np.asarray(prediction), np.asarray(dets)
        prediction_shape = np.shape(prediction)

    probs = np.zeros(prediction_shape)
    probs[indices, dets] = 1.

    return probs


def prob_to_det(probs, m=None, threshold=None):
    """
    Converts probabilistic classifications to deterministic classifications by assigning the class with highest probability

    Parameters
    ----------
    probs: numpy.ndarray, float
        N * M matrix of class probabilities
    m: int
        class relative to binary decision
    threshold: float, optional
        value between 0 and 1 at which binary decision is made

    Returns
    -------
    dets: numpy.ndarray, int
        maximum probability classes
    """
    if m == None and threshold == None:
        dets = np.argmax(probs, axis=1)
    else:
        try:
            assert(type(m) == int and type(threshold) == np.float64)
        except AssertionError:
            print(str(m)+' is '+str(type(m))+' and must be int; ' +
                  str(threshold)+' is '+str(type(threshold))+' and must be float')
        dets = np.zeros(np.shape(probs)[0]).astype(int)
        dets[probs[:, m] >= threshold] = 1

    return dets


def det_to_cm(dets, truth, per_class_norm=False, vb=False):
    """
    Converts deterministic classifications and truth into confusion matrix

    Parameters
    ----------
    dets: numpy.ndarray, int
        assigned classes
    truth: numpy.ndarray, int
        true classes
    per_class_norm: boolean, optional
        equal weight per class if True, equal weight per object if False
    vb: boolean, optional
        if True, print cm

    Returns
    -------
    cm: numpy.ndarray, int
        confusion matrix

    Notes
    -----
    I need to fix the norm keyword all around to enable more options, like normed output vs. not.
    """
    pred_classes, pred_counts = np.unique(dets, return_counts=True)
    true_classes, true_counts = np.unique(truth, return_counts=True)
    # if vb: print('by request '+str(((pred_classes, pred_counts), (true_classes, true_counts))))

    M = np.int(max(max(pred_classes), max(true_classes)) + 1)

    # if vb: print('by request '+str((np.shape(dets), np.shape(truth)), M))
    cm = np.zeros((M, M), dtype=int)

    coords = np.array(list(zip(dets, truth)))
    indices, index_counts = np.unique(coords, axis=0, return_counts=True)
    if vb:
        print(indices.T, index_counts)
    index_counts = index_counts.astype(int)
    indices = indices.T.astype(int)
    cm[indices[0], indices[1]] = index_counts

    if per_class_norm:
        cm = cm.astype(float) / true_counts[np.newaxis, :].astype(float)

    if vb:
        print('by request '+str(cm))

    return cm

# def prob_to_cm(probs, truth, per_class_norm=True, vb=False):
#     """
#     Turns probabilistic classifications into confusion matrix by taking maximum probability as deterministic class
#
#     Parameters
#     ----------
#     probs: numpy.ndarray, float
#         N * M matrix of class probabilities
#     truth: numpy.ndarray, int
#         N-dimensional vector of true classes
#     per_class_norm: boolean, optional
#         equal weight per class if True, equal weight per object if False
#     vb: boolean, optional
#         if True, print cm
#
#     Returns
#     -------
#     cm: numpy.ndarray, int
#         confusion matrix
#     """
#     dets = prob_to_det(probs)
#
#     cm = det_to_cm(dets, truth, per_class_norm=per_class_norm, vb=vb)
#
#     return cm

# def det_to_rate(dets, truth, per_class_norm=True, vb=False):
#     cm = det_to_cm(dets, truth, per_class_norm=per_class_norm, vb=vb)
#     rates = cm_to_rate(cm, vb=vb)
#     return rates
#
# def prob_to_rate(probs, truth, per_class_norm=True, vb=False):
#     cm = prob_to_cm(probs, truth, per_class_norm=per_class_norm, vb=vb)
#     rates = cm_to_rate(cm, vb=vb)
#     return rates

# def binary_rates(dets, truth, m):
#
# 	tp = np.sum(dets[truth == m])
# 	fp = np.sum(dets[truth != m])
# 	tpr = tp/len(dets[truth == m])
# 	fpr = fp/len(dets[truth != m])
#
# 	return tpr,fpr

# def recall(rates):
#     return 1. - rates.FNR

# def recall(rates):
#     """
#     Calculates recall from rates
#
#     Parameters
#     ----------
#     rates: namedtuple
#         named tuple of 'TPR FPR FNR TNR'
#
#     Returns
#     -------
#     recall: float
#         recall
#     """
#     return 1. - rates.FNR


def precision(TP, FP):
    """
    Calculates precision from rates

    Parameters
    ----------
    TP: float
        number of true positives
    FP: float
        number of false positives

    Returns
    -------
    p: float
        precision
    """
    p = np.asarray(TP / (TP + FP))
    if np.any(np.isnan(p)):
        p[np.isnan(p)] = 0.
    return p
#
# def recall(classifications,truth,class_idx):
#
# 	tp = np.sum(classifications[truth == class_idx])
# 	fp = np.sum(classifications[truth != class_idx])
# 	fn = len(np.where((classifications == 0) & (truth == class_idx))[0])
# 	#import pdb; pdb.set_trace()
# 	#print(fn)
#
# 	return tp/(tp+fn)
