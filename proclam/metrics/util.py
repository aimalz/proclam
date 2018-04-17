"""
Utility functions for PLAsTiCC metrics
"""

from __future__ import absolute_import
__all__ = ['truth_reformatter']

import numpy as np

def truth_reformatter(truth, prediction=None):
    """
    Reformats array of true classes into matrix with 1 at true class and zero elsewhere

    Parameters
    ----------
    truth: numpy.ndarray, int
        true classes
    prediction: numpy.ndarray, float, optional
        predicted class probabilities

    Returns
    -------
    metric: float
        value of the metric

    Notes
    -----
    Does not yet handle number of classes in truth not matching number of classes in prediction, i.e. for having "other" class or secret classes not in training set
    """
    N = len(truth)
    indices = range(N)

    if prediction is None:
        prediction_shape = (N, np.max(truth) + 1)
    else:
        prediction, truth = np.asarray(prediction), np.asarray(truth)
        prediction_shape = np.shape(prediction)

    truth_reformatted = np.zeros(prediction_shape)
    truth_reformatted[indices, truth] = 1.

    return truth_reformatted
