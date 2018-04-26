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


def prob_to_cm(probs, truth):

    N = np.shape(probs)[0]
    N_class = np.shape(probs)[1]

    CM = np.zeros((N_class, N_class))
    
    
    class_type = np.argmax(probs, axis=1)

    for i in range(len(class_type)):      
            CM[int(class_type[i]), int(truth[i])] +=1
            print(CM)
    return CM


import numpy as np
def rates(det_class, truth):
    """
    Returns the the array of rates: [TPR, TNR, FPR, FNR] given a set of 
    deterministic classifications and the true classes.

    Parameters
    ----------
    det_class: numpy.ndarray, float
        The deterministic classification
    truth: numpy.ndarray, int
        true classes
    

    Returns
    -------
    rates: numpy.ndarray, float
        An array with the TPR, TNR, FPR, FNR
    """
    N = len(truth)
    classes = np.unique(det_class) 
    
    TPC = np.sum(det_class == truth)
    TNC = 0
    FPC = 0
    FNC = 0
    for cl in classes:
        FPC += np.sum(det_class[det_class == cl] != truth[det_class == cl])
        FNC += np.sum(truth[truth == cl] != det_class[truth == cl])
        TNC += np.sum(truth[truth != cl] != det_class[truth != cl])
        
    TPR = TPC/(TPC+FNC)
    FPR = FPC/(FPC+TNC)
    TNR = 1 - FPR
    FNR = 1 - TPR

    return np.array([TPR, TNR, FPR, FNR])


