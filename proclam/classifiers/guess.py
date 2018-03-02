"""
A subclass for a randomly guessing classifier
"""

import numpy as np

from classifier import Classifier

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

    def classify(self, M, truth, **kwds):
        """
        Simulates mock classifications based on truth

        Parameters
        ----------
        M: int
            the number of anticipated classes
        truth: numpy.ndarray, float
            class probabilities for all classes for all items
        Returns
        -------
        prediction: numpy.ndarray
            true classes
        """

        N = len(truth)
        prediction = np.random.uniform(size=(N, M))
        prediction /= np.sum(prediction, axis=1)[:, np.newaxis]

        return prediction
