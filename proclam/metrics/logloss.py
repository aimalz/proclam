"""
A metric subclass for the log-loss
"""

from __future__ import absolute_import
import numpy as np
import sys

from .metric import Metric

# would like some shared util functions
# from util import epsilon
# from util import averager

class LogLoss(Metric):

    def __init__(self, scheme=None):
        """
        An object that evaluates a function of the true classes and class probabilities

        Parameters
        ----------
        scheme: string
            the name of the metric
        """

        super(LogLoss, self).__init__(scheme)
        self.scheme = scheme
        self.averaging = averaging

    def evaluate(self, prediction, truth, averaging='per_class'):
        """
        Evaluates a function of the truth and prediction

        Parameters
        ----------
        prediction: numpy.ndarray, float
            predicted class probabilities
        truth: numpy.ndarray, int
            true classes
        averaging: string
            'per_class' weights classes equally, other keywords possible

        Returns
        -------
        logloss: float
            value of the metric

        Notes
        -----
        This uses the natural log.
        """
        prediction, truth = np.asarray(prediction), np.asarray(truth)
        prediction_shape = np.shape(prediction)
        (N, M) = prediction_shape

        truth_reformatted = np.zeros(prediction_shape)
        truth_reformatted += truth_reformatted[:, truth]

        prediction_reformatted = prediction + sys.float.epsilon * np.ones(prediction_shape)
        prediction_reformatted /= np.sum(prediction_reformatted, axis=1)

        log_prob = np.log(prediction_reformatted)
        logloss_each = -1. * np.sum(truth_reformatted * log_prob, axis=1)

        # would like to replace this with general "averager" util function
        if self.averaging == 'per_class':
            class_logloss = np.empty(M)
            for m in range(M):
                true_indices = np.where(truth == m)
                how_many_in_class = len(true_indices)
                per_class_logloss = logloss_each[true_indices]
                class_logloss[m] = np.average(per_class_logloss)
            group_logloss = np.average(class_logloss)
        elif self.averaging == 'per_item':
            group_logloss = logloss_each
        logloss = np.average(group_logloss)

        return logloss
