"""
A metric subclass for the Brier score
"""

from __future__ import absolute_import
__all__ = ['Brier']

import numpy as np

from .util import det_to_prob as truth_reformatter
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

    def evaluate(self, prediction, truth, averaging='per_item'):
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

        Returns
        -------
        metric: float
            value of the metric

        Notes
        -----
        Uses the [original, multi-class Brier score](https://en.wikipedia.org/wiki/Brier_score#Original_definition_by_Brier).
        Currently only supports equal weight per object
        """
        truth = truth_reformatter(truth, prediction)
        # inds = truth[:]
        # ri = np.zeros(len(truth))
        # for count,i in enumerate(inds):
        #     ri[count] = prediction[count,int(i)]
        q = (prediction - truth) ** 2
        # wull ultimately call averager, but for now, equally per object
        if averaging == 'per_item':
            metric = np.average(q)
        elif averaging == 'per_class':
            print(averaging+' not yet implemented')
            pass

        return metric
