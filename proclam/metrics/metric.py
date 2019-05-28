"""
A superclass for metrics
"""

from __future__ import absolute_import
__all__ = ['Metric']

import numpy as np

from .util import weight_sum
from .util import check_weights

class Metric(object):

    def __init__(self, scheme=None, **kwargs):
        """
        An object that evaluates a function of the true classes and class probabilities

        Parameters
        ----------
        scheme: string
            the name of the metric
        """

        self.scheme = scheme

    def evaluate(self, prediction, truth, weights=None, **kwds):
        """
        Evaluates a function of the truth and prediction

        Parameters
        ----------
        prediction: numpy.ndarray, float
            predicted class probabilities
        truth: numpy.ndarray, int
            true classes
        weights: numpy.ndarray, float
            per-class weights

        Returns
        -------
        metric: float
            value of the metric
        """

        print('No metric specified')
        #
        # mode = np.argmax(prediction, axis=1)
        # metric = len(np.where(truth == mode))
        # N = len(mode)
        # metric /= float(N)

        return # metric
