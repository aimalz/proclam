"""
A superclass for metrics
"""

import numpy as np

class BRIER_Metric(object):

    def __init__(self, scheme=None):
        """
        An object that evaluates a function of the true classes and class probabilities

        Parameters
        ----------
        scheme: string
            the name of the metric
        """

        self.scheme = scheme

    def evaluate(self, prediction, truth, **kwds):
        """
        Evaluates a function of the truth and prediction

        Parameters
        ----------
        prediction: numpy.ndarray, float
            predicted class probabilities
        truth: numpy.ndarray, int
            true classes

        Returns
        -------
        metric: float
            value of the metric
        """
        inds = truth[:]
        ri=np.zeros(len(truth))
        for count,i in enumerate(inds):
            ri[count]=prediction[count,int(i)]
        

        q=2*ri-np.sum(prediction**2, axis=1)
        metric = q

        return metric
