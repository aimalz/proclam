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
        for j in range(len(truth)):
            if (int(class_type[i]) == int(truth[j])):
                CM[class_type[i],int(truth[j])] +=1

                

    return CM
