"""
A class for the Precision-Recall Curve
"""

from __future__ import absolute_import
__all__ = ['PRC']

import numpy as np

from .util import weight_sum, check_weights
from .util import prob_to_det, det_to_cm, cm_to_rate, prob_to_rate
from .util import auc, check_auc_grid, precision
from .metric import Metric

class PRC(Metric):

    def __init__(self, scheme=None):
        """
        An object that evaluates the PRC AUC

        Parameters
        ----------
        scheme: string
            the name of the metric
        """
        super(PRC, self).__init__(scheme)
        self.scheme = scheme

    # def _quick_eval(self, per_threshold_rates):
    #     (n_thresholds, M) = np.shape(per_threshold_rates)
    #     auc_class = np.empty(M)
    #     curve = np.empty((M, 2, n_thresholds))
    #
    #     for m in range(M):
    #         precisions, recalls = np.empty(n_thresholds), np.empty(n_thresholds)
    #         precisions = per_threshold_rates[m].TP / (per_threshold_rates[m].TP + per_threshold_rates[m].FP)
    #         recalls = per_threshold_rates[m].TP / (per_threshold_rates[m].TP + per_threshold_rates[m].FN)
    #
    #         (curve[m][0], curve[m][1]) = (recalls, precisions)
    #         auc_class[m] = auc(recalls, precisions)
    #
    #     return auc_class, curve

    def evaluate(self, prediction, truth, grid, averaging='per_class', vb=False):
        """
        Evaluates the area under the PRC

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
                raise RuntimeError('No true values for class %i so PRC is undefined'%m)

        rates = prob_to_rate(prediction, truth, threshold=grid)

            # precisions, recalls = np.empty(n_thresholds), np.empty(n_thresholds)
            # # warning: this is insanely slow!
            # for i, t in enumerate(thresholds_grid):
            #     dets = prob_to_det(prediction, m, threshold=t)
            #     cm = det_to_cm(dets, m_truth)
            #     rates = cm_to_rate(cm)
            #     recalls[i] = rates.TPR[-1]
            #     precisions[i] = precision(rates.TP[-1], rates.FP[-1])

        for m in range(M):
            recalls = rates[m].TPR
            precisions = precision(rates[m].TP, rates[m].FP)
            (curve[m][0], curve[m][1]) = (recalls, precisions)
            auc_class[m] = auc(recalls, precisions)

        if np.any(np.isnan(curve)):
            print('Where did these NaNs come from?')
            return curve

        weights = check_weights(averaging, M, truth=truth)
        auc_allclass = weight_sum(auc_class, weights)

        if vb: return auc_class, curve
        else: return auc_allclass
