"""
A superclass for simulators
"""

import numpy as np

class Simulator(object):

    def __init__(self, seed=0, *args, **kwds):
        """
        An object that simulates true class assignments.

        Parameters
        ----------
        seed: int, optional
            the random seed to use, handy for testing
        """

        self.seed = np.random.seed()

    def simulate(self, N, M, **kwds):
        """
        Simulates the truth table

        Parameters
        ----------
        N: int
            the number of items
        M: int
            the number of true classes

        Returns
        -------
        truth: numpy.ndarray, int
            true classes
        """

        print('No simulation procedure specified: returning zeros')

        truth = np.zeros(N).astype(int)

        return truth
