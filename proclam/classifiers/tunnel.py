"""
A subclass for a general classifier based on a perturbed confusion matrix
"""

from __future__ import absolute_import
__all__  = ['Tunnel']

import numpy as np
import scipy.stats as sps

from .classifier import Classifier

class FromCM(Classifier):

    def __init__(self, scheme='CM', seed=0):
        """
        An object that simulates predicted classifications from the truth values and and arbitrary confusion matrix.

        Parameters
        ----------
        scheme: string
            the name of the classifier
        seed: int, optional
            the random seed to use, handy for testing
        """

        super(FromCM, self).__init__(scheme, seed)
        np.random.seed(seed=self.seed)

    def classify(self, cm, truth, other=False):
        """
        Simulates mock classifications by perturbing a given confusion matrix

        Parameters
        ----------
        cm: numpy.ndarray, float
            the confusion matrix, normalized to sum to 1 across rows. Its dimensions need to match the anticipated number of classes.
        truth: numpy.ndarray, int
            array of the true classes of the items
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
        if other: M += 1
 
        prediction = np.zeros((N,M))
        class_corr = np.random.randint(0, M, size=1) # randomly choose which class to work well on
        
        for i in range(N):
            if cm[truth][i][class_corr] > (1./M):    # take cm[truth] for the rows with correct class
                prediction[i] = cm[truth][i]
            else:
                prediction[i] = np.ones(M) * (1./M)  # assign (1/M) probability to rows of other classes
        
        prediction /= np.sum(prediction, axis=1)[:, np.newaxis] # normalize the rows to sum to 1

        return prediction
