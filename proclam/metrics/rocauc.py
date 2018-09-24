"""
A metric subclass for the Brier score
"""

from __future__ import absolute_import
__all__ = ['ROCAUC']

import numpy as np
from sklearn.metrics import roc_auc_score
from .metric import Metric

class ROCAUC(Metric):

    def __init__(self, scheme=None):
        """
        An object representing the Brier score metric

        Parameters
        ----------
        scheme: string
            the name of the metric instance
        """

        super(ROCAUC, self).__init__(scheme)

    def evaluate(self, prediction, truth, averaging='per_class'):
        """
        Evaluates the ROC AUC score

        Parameters
        ----------
        prediction: numpy.ndarray, float
            predicted class probabilities
        truth: numpy.ndarray, int
            true classes
        averaging: string, optional
            'per_class' weights classes equally, other keywords possible
            vector assumed to be class weights
            Does absolutely nothing for ROCAUC.

        Returns
        -------
        metric: float
            value of the metric

        Notes
        -----
        """
        metric = roc_auc_score(truth, prediction, average='macro')
        assert(~np.isnan(metric))

        return metric
