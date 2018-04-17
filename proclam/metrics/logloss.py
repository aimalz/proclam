"""
A metric subclass for the log-loss
"""

from __future__ import absolute_import
__all__ = ['LogLoss']

import numpy as np
import sys

from .util import truth_reformatter
from .metric import Metric

# would like some shared util functions
# from util import epsilon
# from util import averager

class LogLoss(Metric):

    def __init__(self, scheme=None):
        """
        An object that evaluates the log-loss metric

        Parameters
        ----------
        scheme: string
            the name of the metric
        """

        super(LogLoss, self).__init__(scheme)
        self.scheme = scheme

    def evaluate(self, prediction, truth, averaging='per_class'):
        """
        Evaluates the log-loss

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

        truth = truth_reformatter(truth, prediction)

        # we might also want a util function for normalizing these to be log-friendly along with the right dimensions
        if np.any(prediction == 0.):
            prediction_reformatted = prediction + sys.float_info.epsilon * np.ones(prediction_shape)
            prediction /= np.sum(prediction_reformatted, axis=1)[:, np.newaxis]

        log_prob = np.log(prediction)
        logloss_each = -1. * np.sum(truth * log_prob, axis=1)[:, np.newaxis]

        # would like to replace this with general "averager" util function
        # use a better structure for checking keyword support
        group_logloss = logloss_each
        print('Averaging by '+averaging+'.')
        if averaging == 'per_class':
            class_logloss = np.empty(M)
            for m in range(M):
                true_indices = np.where(truth == m)
                how_many_in_class = len(true_indices)
                per_class_logloss = logloss_each[true_indices]
                class_logloss[m] = np.average(per_class_logloss)
            group_logloss = np.average(class_logloss)
        elif averaging == 'per_item':
            group_logloss = logloss_each
            pass
        else:
            print('Averaging by '+averaging+' not yet supported.')
            return
        logloss = np.average(group_logloss)

        return logloss
