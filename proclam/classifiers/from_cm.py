"""
A subclass for a randomly guessing classifier
"""
from __future__ import absolute_import
__all__  = ['FromCM']
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

    def classify(self, cm, truth, delta=0.1, other=False):
        """
        Simulates mock classifications based on truth

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
        if other: M += 1

        prediction = cm[truth] + delta * sps.halfcauchy.rvs(size=(N, M))
        prediction /= np.sum(prediction, axis=1)[:, np.newaxis]

        return prediction
