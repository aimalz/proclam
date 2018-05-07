"""
A metric subclass for the log-loss
"""

from __future__ import absolute_import
__all__ = ['LogLoss']

import numpy as np
import sys

from .util import weight_sum
from .util import check_weights
from .util import det_to_prob as truth_reformatter
from .util import averager
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
        averaging: string or numpy.ndarray, float
            'per_class' weights classes equally, other keywords possible
            vector assumed to be class weights

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

        weights = check_weights(averaging, M, truth=truth)
        truth_mask = truth_reformatter(truth, prediction)

        # we might also want a util function for normalizing these to be log-friendly along with the right dimensions
        if np.any(prediction == 0.):
            prediction_reformatted = prediction + sys.float_info.epsilon * np.ones(prediction_shape)
            prediction /= np.sum(prediction_reformatted, axis=1)[:, np.newaxis]

        log_prob = np.log(prediction)
        logloss_each = -1. * np.sum(truth_mask * log_prob, axis=1)[:, np.newaxis]

        # use a better structure for checking keyword support
        class_logloss = averager(logloss_each,truth,M)
        
        logloss = weight_sum(class_logloss, weight_vector=weights)
        # elif averaging == 'per_item':
        #     group_logloss = logloss_each
        #     pass
        # else:
        #     print('Averaging by '+averaging+' not yet supported.')
        #     return
        # logloss = np.average(group_logloss)

        return logloss
