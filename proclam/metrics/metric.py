"""
A superclass for metrics
"""

import numpy as np

class Metric(object):

    def __init__(self, scheme=None):
        """
        An object that evaluates a function of the true classes and class probabilities

        Parameters
        ----------
        scheme: string
            the name of the metric
        """

        self.scheme = scheme

    def evaluate(self, prediction, truth, **kwds):
        """
        Evaluates a function of the truth and prediction

        Parameters
        ----------
        prediction: numpy.ndarray, float
            predicted class probabilities
        truth: numpy.ndarray, int
            true classes

        Returns
        -------
        metric: float
            value of the metric
        """

        print('No metric specified: returning true positive rate based on maximum value')

        mode = np.argmax(prediction, axis=1)
        metric = len(np.where(truth == mode))
        N = len(mode)
        metric /= float(N)

        return metric
