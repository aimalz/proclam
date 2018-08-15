"""
A subclass for a general classifier based on a perturbed confusion matrix
"""

from __future__ import absolute_import
__all__  = ['FromCMDM']

import numpy as np
import scipy.stats as sps

from .classifier import Classifier

class FromCMDM(Classifier):

    def __init__(self, scheme='CMDM', seed=0):
        """
        An object that simulates predicted classifications from the truth values and and arbitrary confusion matrix, according to the Dirichlet distribution

        Parameters
        ----------
        scheme: string
            the name of the classifier
        seed: int, optional
            the random seed to use, handy for testing
        """

        super(FromCM, self).__init__(scheme, seed)
        np.random.seed(seed=self.seed)

    def classify(self, cm, truth, delta=0.1, other=False):
        """
        Simulates mock classifications by perturbing a given confusion matrix

        Parameters
        ----------
        cm: numpy.ndarray, float
            the confusion matrix, normalized to sum to 1 across rows. Its dimensions need to match the anticipated number of classes.
        truth: numpy.ndarray, int
            array of the true classes of the items
        delta: float, optional
            perturbation factor for confusion matrix
        other: boolean, optional
            include class for other

        Returns
        -------
        prediction: numpy.ndarray, float
            predicted classes

        Notes
        -----
        other keyword doesn't actually work right now
        """

        N = len(truth)
        M = len(cm)
        
        alpha_m = delta*cm
        prediction =  np.zeros(M)

        for m in range(M):
            x[m] =  sts.gamma(alpha_m[truth], 1)

        for i in range(M):
            prediction[m] = x[m]/np.sum(x) 
            
        return prediction
