"""
A subclass for a uniform distribution of classes
"""

import numpy as np

from simulator import Simulator

class Uniform(Simulator):

    def __init__(self, scheme='uniform', seed=0):
        """
        An object that simulates true class assignments assuming equal fractions of each class.

        Parameters
        ----------
        scheme: string
            the name of the simulator
        seed: int, optional
            the random seed to use, handy for testing
        """

        super(Uniform, self).__init__(scheme, seed)
        np.random.seed(seed=self.seed)

    def simulate(self, M, N):
        """
        Simulates the truth table as a uniform distribution

        Parameters
        ----------
        M: int
            the number of true classes
        N: int
            the number of items

        Returns
        -------
        truth: numpy.ndarray, int
            array of true class indices
        """

        truth = np.random.choice(M, size=N)

        return truth
