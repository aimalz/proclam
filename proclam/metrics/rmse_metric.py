"""
A superclass for metrics
"""
from __future__ import absolute_import
__all__ = ['RMSE_Metric']
import numpy as np

class RMSE_Metric(object):

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

#        print('No metric specified: returning true positive rate based on maximum value')
#        print(np.shape(prediction))
        mode = np.argmax(prediction, axis=1)
        
        metric = np.sum((mode - truth)**2)
        N = len(mode)
        
        metric /= float(N)
        return metric 

