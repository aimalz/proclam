"""
A class for the Matthews correlation coefficient
"""

from __future__ import absolute_import
__all__ = ['MCC']

import numpy as np

from .util import weight_sum, check_weights
from .util import prob_to_det, det_to_cm, cm_to_rate
from .metric import Metric

class MCC(Metric):

    def __init__(self, scheme=None):
        """
        An object that evaluates the Matthews correlation coefficient

        Parameters
        ----------
        scheme: string
            the name of the metric
        """
        super(MCC, self).__init__(scheme)
        self.scheme = scheme

    def evaluate(self, prediction, truth, averaging='per_class', vb=False):
        """
        Evaluates the Matthews correlation coefficient

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
        mcc_all: float
            value of the metric
        """
        prediction, truth = np.asarray(prediction), np.asarray(truth)
        (N, M) = np.shape(prediction)

        dets = prob_to_det(prediction)
        cm = det_to_cm(dets, truth)
        rates = cm_to_rate(cm)

        mcc = np.empty(M)
        for m in range(M):
            if not len(np.where(truth == m)[0]):
                raise RuntimeError('No true values for class %i so MCC is undefined'%m)
            num = rates.TP[m] * rates.TN[m] - rates.FP[m] * rates.FN[m]
            A = rates.TP[m] + rates.FP[m]
            B = rates.TP[m] + rates.FN[m]
            C = rates.TN[m] + rates.FP[m]
            D = rates.TN[m] + rates.FN[m]
            mcc[m] = num / np.sqrt(A * B * C * D)

        weights = check_weights(averaging, M, truth=truth)
        mcc_all = weight_sum(mcc, weights)

        if vb: return mcc
        else: return mcc_all
