"""
A class for the Receiver Operating Curve
"""

from __future__ import absolute_import
__all__ = ['ROC']

import numpy as np

from .util import weight_sum, check_weights
from .util import prob_to_det, det_to_cm, cm_to_rate
from .util import auc, check_auc_grid, prep_curve
from .metric import Metric


class ROC(Metric):

    def __init__(self, scheme=None):
        """
        An object that evaluates the ROC metric

        Parameters
        ----------
        scheme: string
            the name of the metric
        """
        super(ROC, self).__init__(scheme)
        self.scheme = scheme

    def evaluate(self, prediction, truth, grid, averaging='per_class', vb=False):
        """
        Evaluates the ROC AUC

        Parameters
        ----------
        prediction: numpy.ndarray, float
            predicted class probabilities
        truth: numpy.ndarray, int
            true classes
        grid: numpy.ndarray, float or float or int
            array of values between 0 and 1 at which to evaluate ROC
        averaging: string or numpy.ndarray, float
            'per_class' weights classes equally, other keywords possible, vector assumed to be class weights

        Returns
        -------
        auc_allclass: float
            value of the metric
        """
        thresholds_grid = check_auc_grid(grid)
        n_thresholds = len(thresholds_grid)

        prediction, truth = np.asarray(prediction), np.asarray(truth)
        (N, M) = np.shape(prediction)

        auc_class = np.empty(M)
        curve = np.empty((M, 2, n_thresholds))

        for m in range(M):
            m_truth = (truth == m).astype(int)

            if not len(np.where(truth == m)[0]):
                raise RuntimeError('No true values for class %i so ROC is undefined'%m)

            tpr, fpr = np.empty(n_thresholds), np.empty(n_thresholds)
            for i, t in enumerate(thresholds_grid):
                dets = prob_to_det(prediction, m, threshold=t)
                cm = det_to_cm(dets, m_truth)
                rates = cm_to_rate(cm)
                fpr[i], tpr[i] = rates.FPR[-1], rates.TPR[-1]

            (curve[m][0], curve[m][1]) = (fpr, tpr)
            (fpr, tpr) = prep_curve(fpr, tpr)
            auc_class[m] = auc(fpr, tpr)

        weights = check_weights(averaging, M, truth=truth)
        auc_allclass = weight_sum(auc_class, weights)

        if vb:
            return curve
        else:
            return auc_allclass
