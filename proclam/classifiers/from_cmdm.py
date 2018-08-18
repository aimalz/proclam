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

        super(FromCMDM, self).__init__(scheme, seed)
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
        Larger delta means larger scatter, smaller delta means smaller scatter
        """

        N = len(truth)
        M = len(cm)

        cm[cm == 0.] = 1.e-8
        alpha = cm / delta
        prediction =  np.empty((N, M))

        for m in range(M):
            func_m = sps.dirichlet(alpha[m])
            inds_m = np.where(truth == m)[0]
            prediction[inds_m] = func_m.rvs(len(inds_m))

        prediction /= np.sum(prediction, axis=1)[:, np.newaxis]

        return prediction
