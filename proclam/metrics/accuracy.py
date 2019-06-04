"""
A class for accuracy
"""

from __future__ import absolute_import
__all__ = ['Accuracy']

import numpy as np

from .util import weight_sum, check_weights
from .util import prob_to_det, det_to_cm, cm_to_rate
from .metric import Metric

class Accuracy(Metric):

    def __init__(self, scheme=None):
        """
        An object that evaluates the accuracy

        Parameters
        ----------
        scheme: string
            the name of the metric
        """
        super(Accuracy, self).__init__(scheme)
        self.scheme = scheme

    def evaluate(self, prediction, truth, averaging='per_class'):
        """
        Evaluates the accuracy

        Parameters
        ----------
        prediction: numpy.ndarray, float
            predicted class probabilities
        truth: numpy.ndarray, int
            true classes
        averaging: string or numpy.ndarray, float
            'per_class' weights classes equally, other keywords possible, vector assumed to be class weights

        Returns
        -------
        accuracy_all: float
            value of the metric
        """
        prediction, truth = np.asarray(prediction), np.asarray(truth)
        (N, M) = np.shape(prediction)

        dets = prob_to_det(prediction)
        cm = det_to_cm(dets, truth)
        rates = cm_to_rate(cm)
        accuracy = rates.TPR

        weights = check_weights(averaging, M, truth=truth)
        accuracy_all = weight_sum(accuracy, weights)

        return accuracy_all