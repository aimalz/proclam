"""
Utility functions for PLAsTiCC metrics
"""

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
    Does not yet handle number of classes in truth not matching number of classes in prediction
    """
    if prediction is None:
        prediction_shape = (len(truth), np.max(truth) + 1)
    else:
        prediction, truth = np.asarray(prediction), np.asarray(truth)
        prediction_shape = np.shape(prediction)

    truth_reformatted = np.zeros(prediction_shape)
    truth_reformatted[:, truth] = 1.

    return truth_reformatted
