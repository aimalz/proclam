"""
A class for the F1 score
"""

from __future__ import absolute_import
__all__ = ['F1']

import numpy as np

from .util import weight_sum, check_weights
from .util import prob_to_det, det_to_cm, cm_to_rate
from .util import precision
from .metric import Metric

class F1(Metric):

    def __init__(self, scheme=None):
        """
        An object that evaluates the F1 score

        Parameters
        ----------
        scheme: string
            the name of the metric
        """
        super(F1, self).__init__(scheme)
        self.scheme = scheme

    def evaluate(self, prediction, truth, averaging='per_class'):
        """
        Evaluates the F1 score

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
        f1_all: float
            value of the metric
        """
        prediction, truth = np.asarray(prediction), np.asarray(truth)
        (N, M) = np.shape(prediction)

        for m in range(M):
            if not len(np.where(truth == m)[0]):
                raise RuntimeError('No true values for class %i so F1 is undefined'%m)
        dets = prob_to_det(prediction)
        cm = det_to_cm(dets, truth)
        rates = cm_to_rate(cm)
        r = rates.TPR
        p = precision(rates.TP, rates.FP)
        f1 = 2 * p * r / (p + r)

        weights = check_weights(averaging, M, truth=truth)
        f1_all = weight_sum(f1, weights)

        return f1_all