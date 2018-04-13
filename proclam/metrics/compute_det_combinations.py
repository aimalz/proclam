"""
A superclass for metrics
"""

import numpy as np

class Det_Metric(object):

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
            For now we will turn these into classes by some weighting
        truth: numpy.ndarray, int
            true classes

        Returns
        -------
        tpr: float
            true positive rate = TP/Number of true positives P_true
        tnr: float
            true negative rate = TN/Number of true negatives N_true
        fnr: float
            false negative rate = FN/Number of true positives P_true
        fpr: float
            false positive rate = FP/Number of true negatives N_true
        ppr: float
             positive predictive rate = TP/Number of postive predicted P_pred
        npr: float
             negative predictive rate = TN/Number of negative predicted N_pred
        
        """
        inds = truth[:]
        ri=np.zeros(len(truth))
        for count,i in enumerate(inds):
            ri[count]=prediction[count,int(i)]
        

        q=2*ri-np.sum(prediction**2, axis=1)
        metric = q

        return metric
