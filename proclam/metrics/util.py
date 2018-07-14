"""
Utility functions for PLAsTiCC metrics
"""

from __future__ import absolute_import, division
__all__ = ['sanitize_predictions',
           'weight_sum', 'averager', 'check_weights',
           'det_to_prob',
           'prob_to_det',
           'det_to_cm', 'prob_to_cm',
           'cm_to_rate', 'det_to_rate', 'prob_to_rate']

import collections
import numpy as np
import sys

RateMatrix = collections.namedtuple('rates', 'TPR FPR FNR TNR')

def sanitize_predictions(predictions, epsilon=sys.float_info.epsilon):
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

def prob_to_det(probs):
    """
    Converts probabilistic classifications to deterministic classifications by assigning the class with highest probability

    Parameters
    ----------
    probs: numpy.ndarray, float
        N * M matrix of class probabilities

    Returns
    -------
    dets: numpy.ndarray, int
        maximum probability classes
    """
    dets = np.argmax(probs, axis=1)

    return dets

def det_to_cm(dets, truth, per_class_norm=True, vb=False):
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
    if vb: print((pred_classes, pred_counts), (true_classes, true_counts))

    M = max(max(pred_classes), max(true_classes)) + 1

    cm = np.zeros((M, M), dtype=float)
    # print((np.shape(dets), np.shape(truth)))
    coords = np.array(list(zip(dets, truth)))
    indices, index_counts = np.unique(coords, axis=0, return_counts=True)
    # if vb: print(indices, index_counts)
    indices = indices.T
    # if vb: print(np.shape(indices))
    cm[indices[0], indices[1]] = index_counts
    if vb: print(cm)

    if per_class_norm:
        # print(type(cm))
        # print(type(true_counts))
        # cm = cm / true_counts
        # cm /= true_counts[:, np.newaxis] #
        cm = cm / true_counts[np.newaxis, :]

    if vb: print(cm)

    return cm

def prob_to_cm(probs, truth, per_class_norm=True, vb=False):
    """
    Turns probabilistic classifications into confusion matrix by taking maximum probability as deterministic class

    Parameters
    ----------
    probs: numpy.ndarray, float
        N * M matrix of class probabilities
    truth: numpy.ndarray, int
        N-dimensional vector of true classes
    per_class_norm: boolean, optional
        equal weight per class if True, equal weight per object if False
    vb: boolean, optional
        if True, print cm

    Returns
    -------
    cm: numpy.ndarray, int
        confusion matrix
    """
    dets = prob_to_det(probs)

    cm = det_to_cm(dets, truth, per_class_norm=per_class_norm, vb=vb)

    return cm

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
    BROKEN!
    This can be done with a mask to weight the classes differently here.
    """
    if vb: print(cm)
    diag = np.diag(cm)
    if vb: print(diag)

    TP = np.sum(diag)
    FN = np.sum(np.sum(cm, axis=0) - diag)
    FP = np.sum(np.sum(cm, axis=1) - diag)
    TN = np.sum(cm) - TP
    if vb: print((TP, FN, FP, TN))

    T = TP + TN
    F = FP + FN
    P = TP + FP
    N = TN + FN
    if vb: print((T, F, P, N))

    TPR = TP / P
    FPR = FP / N
    FNR = FN / P
    TNR = TN / N

    rates = RateMatrix(TPR=TPR, FPR=FPR, FNR=FNR, TNR=TNR)
    if vb: print(rates)

    return rates

def det_to_rate(dets, truth, per_class_norm=True, vb=False):
    cm = det_to_cm(dets, truth, per_class_norm=per_class_norm, vb=vb)
    rates = cm_to_rate(cm, vb=vb)
    return rates

def prob_to_rate(probs, truth, per_class_norm=True, vb=False):
    cm = prob_to_cm(probs, truth, per_class_norm=per_class_norm, vb=vb)
    rates = cm_to_rate(cm, vb=vb)
    return rates

def weight_sum(per_class_metrics, weight_vector, norm=True):
    """
    Calculates the weighted metric

    Parameters
    ----------
    per_class_metrics: numpy.float
        the scores separated by class (a list of arrays)
    weight_vector: numpy.ndarray floar
        The array of weights per class
    norm: boolean, optional

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
            weight[chosen] = 1.
        elif avg_info == 'down':
            weights = np.ones(M)
            weight[chosen] = 1./np.float(M)
    return weights


def averager(per_object_metrics, truth, M):
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
        print((m, how_many_in_class, class_metric[m]))
    return class_metric
