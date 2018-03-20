"""
A superclass for simulators
"""

import numpy as np

class Simulator(object):

    def __init__(self, *args, seed=0, **kwds):
        """
        An object that simulates true class assignments.

        Parameters
        ----------
        seed: int, optional
            the random seed to use, handy for testing
        """

        self.seed = np.random.seed(seed=self.seed)

    def simulate(self, M, N, **kwds):
        """
        Simulates the truth table

        Parameters
        ----------
        M: int
            the number of true classes
        N: int
            the number of items

        Returns
        -------
        truth: numpy.ndarray, int
            true classes
        """

        print('No simulation procedure specified: returning zeros')

        truth = np.zeros(N).astype(int)

        return truth
