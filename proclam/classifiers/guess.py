"""
A subclass for a randomly guessing classifier
"""

from __future__ import absolute_import
__all__ = ['Guess']

import numpy as np

from .classifier import Classifier

class Guess(Classifier):

    def __init__(self, scheme='guess', seed=0):
        """
        An object that simulates predicted classifications that are totally random guesses

        Parameters
        ----------
        scheme: string
            the name of the classifier
        seed: int, optional
            the random seed to use, handy for testing
        """

        super(Guess, self).__init__(scheme, seed)
        np.random.seed(seed=self.seed)

    def classify(self, M, truth, other=False):
        """
        Simulates mock classifications based on truth

        Parameters
        ----------
        M: int
            the number of anticipated classes
        truth: numpy.ndarray, float
            class probabilities for all classes for all items
        other: boolean, optional
            include class for other

        Returns
        -------
        prediction: numpy.ndarray
            true classes
        """

        N = len(truth)
        if other: M += 1
        prediction = np.random.uniform(size=(N, M))
        prediction /= np.sum(prediction, axis=1)[:, np.newaxis]

        return prediction
