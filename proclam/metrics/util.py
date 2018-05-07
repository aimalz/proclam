"""
Utility functions for PLAsTiCC metrics
"""

from __future__ import absolute_import, division
__all__ = ['det_to_prob',
            'prob_to_det',
            'det_to_cm', 'prob_to_cm',
            'cm_to_rate', 'det_to_rate', 'prob_to_rate']

import collections
import numpy as np

RateMatrix = collections.namedtuple('rates', 'TPR FPR FNR TNR')

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
        prediction_shape = (N, np.max(dets) + 1)
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

def det_to_cm(dets, truth, per_class_norm=True, vb=True):
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

    cm = np.zeros((M, M))
    coords = zip(dets, truth)
    indices, index_counts = np.unique(coords, axis=0, return_counts=True)
    # if vb: print(indices, index_counts)
    indices = indices.T
    # if vb: print(np.shape(indices))
    cm[indices[0], indices[1]] = index_counts
    if vb: print(cm)

    if per_class_norm:
        cm /= true_counts[np.newaxis, :]

    if vb: print(cm)

    return cm

def prob_to_cm(probs, truth, per_class_norm=True, vb=True):
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

def cm_to_rate(cm, vb=True):
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

def det_to_rate(dets, truth, per_class_norm=True, vb=True):
    cm = det_to_cm(dets, truth, per_class_norm=per_class_norm, vb=vb)
    rates = cm_to_rate(cm, vb=vb)
    return rates

def prob_to_rate(probs, truth, per_class_norm=True, vb=True):
    cm = prob_to_cm(probs, truth, per_class_norm=per_class_norm, vb=vb)
    rates = cm_to_rate(cm, vb=vb)
    return rates
