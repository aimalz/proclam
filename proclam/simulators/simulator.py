"""
A superclass for simulators
"""

import numpy as np

class Simulator(object):

    def __init__(self, scheme, seed=0):
        """
        An object that simulates true class assignments.

        Parameters
        ----------
        scheme: string
            the name of the simulator
        seed: int, optional
            the random seed to use, handy for testing
        """

        self.scheme = scheme
        self.seed = seed

    def simulate(self, M, N, **kwds):
        """
        Simulates the truth table

        Parameters
        ----------
        M: int
            the number of classes
        N: int
            the number of items

        Returns
        -------
        truth: numpy.ndarray
            true classes
        """

        print('No simulation procedure specified')

        truth = np.empty(N)

        return truth
