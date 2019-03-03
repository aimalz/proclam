"""
A superclass for metrics
"""

from __future__ import absolute_import
__all__ = ['Metric']

import numpy as np

from .util import weight_sum
from .util import check_weights
from .util import prob_to_det_threshold
from .util import auc,fnr,fpr

class Metric(object):

    def __init__(self, scheme=None, **kwargs):
        """
        An object that evaluates a function of the true classes and class probabilities

        Parameters
        ----------
        scheme: string
            the name of the metric
        """

        self.scheme = scheme
        self.debug = False

    def evaluate(self, prediction, truth, gridspace=0.01, weights=None, **kwds):
        """
        Evaluates a function of the truth and prediction

        Parameters
        ----------
        prediction: numpy.ndarray, float
            predicted class probabilities
        truth: numpy.ndarray, int
            true classes
        weights: numpy.ndarray, float
            per-class weights

        Returns
        -------
        metric: float
            value of the metric
        """

        auc_allclass = 0
        n_class = np.shape(prediction)[1]
        if not weights:
            weights = [1./n_class]*n_class
        
        for class_idx in range(n_class):
            if not len(np.where(truth == class_idx)[0]):
                raise RuntimeError('No true values for class %i so DET is undefined'%class_idx)

            thresholds_grid = np.arange(0,1,gridspace)
            n_thresholds = len(thresholds_grid)
        
            fnr_arr,fpr_arr = np.zeros(n_thresholds),np.zeros(n_thresholds)
            for t,i in zip(thresholds_grid,range(n_thresholds)):
                classifications = prob_to_det_threshold(prediction,class_idx=class_idx,threshold=t)

                fnr_arr[i] = fnr(classifications,truth,class_idx)
                fpr_arr[i] = fpr(classifications,truth,class_idx)

            auc_class = auc(fnr_arr,fpr_arr)
                
            if self.debug:
                import pylab as plt
                plt.clf()
                plt.plot(fnr_arr,fpr_arr)
                plt.show()
                # import pdb; pdb.set_trace()

            auc_allclass += auc_class*weights[class_idx]
                
        return auc_allclass
