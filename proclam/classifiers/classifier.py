"""
A superclass for classifiers
"""

import numpy as np

class Classifier(object):

    def __init__(self, scheme=None, seed=0):
        """
        An object that simulates predicted classifications.

        Parameters
        ----------
        scheme: string
            the name of the classifier
        seed: int, optional
            the random seed to use, handy for testing
        """

        self.scheme = scheme
        self.seed = seed

    def classify(self, M, truth, **kwds):
        """
        Simulates mock classifications based on truth

        Parameters
        ----------
        M: int
            the number of anticipated classes
        truth: numpy.ndarray, int
            class assignments for all items

        Returns
        -------
        prediction: numpy.ndarray, float
            predicted classes
        """

        print('No classification procedure specified: returning truth table')

        N = len(truth)
        indices = range(N)
        prediction = np.zeros((N, M))

        prediction[indices, truth[indices]] += 1.

        return prediction
