"""
A metric subclass for the Brier score
"""

from __future__ import absolute_import
__all__ = ['Brier']

import numpy as np

from .util import weight_sum
from .util import check_weights
from .util import det_to_prob as truth_reformatter
from .util import averager
from .metric import Metric

class Brier(Metric):

    def __init__(self, scheme=None):
        """
        An object representing the Brier score metric

        Parameters
        ----------
        scheme: string
            the name of the metric instance
        """

        super(Brier, self).__init__(scheme)

    def evaluate(self, prediction, truth, averaging='per_class', vb=False):
        """
        Evaluates the Brier score

        Parameters
        ----------
        prediction: numpy.ndarray, float
            predicted class probabilities
        truth: numpy.ndarray, int
            true classes
        averaging: string, optional
            'per_class' weights classes equally, other keywords possible
            vector assumed to be class weights

        Returns
        -------
        metric: float
            value of the metric

        Notes
        -----
        Uses the [original, multi-class Brier score](https://en.wikipedia.org/wiki/Brier_score#Original_definition_by_Brier).
        """
        prediction, truth = np.asarray(prediction), np.asarray(truth)
        prediction_shape = np.shape(prediction)
        (N, M) = prediction_shape

        weights = check_weights(averaging, M, truth=truth)
        truth_mask = truth_reformatter(truth, prediction)

        q_each = (prediction - truth_mask) ** 2

        class_brier = averager(q_each, truth, M)
        metric = weight_sum(class_brier, weight_vector=weights)

        assert(~np.isnan(metric))

        if vb: return class_brier
        else: return metric
