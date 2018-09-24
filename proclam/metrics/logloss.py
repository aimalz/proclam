"""
A metric subclass for the log-loss
"""

from __future__ import absolute_import
__all__ = ['LogLoss', 'LogLossSohier']

import numpy as np
import sys

from .util import weight_sum
from .util import check_weights
from .util import det_to_prob as truth_reformatter
from .util import sanitize_predictions
from .util import averager
from .metric import Metric

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

        prediction = sanitize_predictions(prediction)

        log_prob = np.log(prediction)
        logloss_each = -1. * np.sum(truth_mask * log_prob, axis=1)[:, np.newaxis]

        # use a better structure for checking keyword support
        class_logloss = averager(logloss_each, truth, M)

        logloss = weight_sum(class_logloss, weight_vector=weights)

        assert(~np.isnan(logloss))

        return logloss


class LogLossSohier(Metric):

    def __init__(self, scheme=None):
        """
        An object that evaluates the log-loss metric

        Parameters
        ----------
        scheme: string
            the name of the metric
        """

        super(LogLossSohier, self).__init__(scheme)
        self.scheme = scheme

    def evaluate(self, prediction, truth, averaging=None):
        return self.plasticc_log_loss(truth, prediction, relative_class_weights=averaging)

    def plasticc_log_loss(self, y_true, y_pred, relative_class_weights=None):
        """
        Verbatim copy of Sohier's implementation of weighted log loss
        """
        predictions = y_pred.copy()

        # sanitize predictions
        epsilon = sys.float_info.epsilon  # on my machine this is 2.2*10**-16, not the 10**-15 used by sklearn
        predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
        predictions = predictions / np.sum(predictions, axis=1)[:, np.newaxis]

        predictions = np.log(predictions)
        # multiplying the arrays is equivalent to a truth mask as y_true only contains zeros and ones
        class_logloss = []
        for i in range(predictions.shape[1]):
            # average column wise log loss with truth mask applied
            result = np.average(predictions[:, i][y_true[:, i] == 1])
            class_logloss.append(result)
        return -1 * np.average(class_logloss, weights=relative_class_weights)


