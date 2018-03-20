"""
A subclass for a distribution of classes dictated by a `scipy.stats.rv_discrete` distribution.  See [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/stats.html#discrete-distributions) for usage.
"""

import numpy as np

from simulator import Simulator

class Stats(Simulator):

    def __init__(self, scheme, seed=0):
        """
        An object that simulates true class assignments based on a provided functional distribution

        Parameters
        ----------
        scheme: scipy.stats.rv_discrete object
            the statistical distribution from which classes should be drawn
        seed: int, optional
            the random seed to use, handy for testing
        """

        super(Stats, self).__init__(scheme, seed)
        np.random.seed(seed=self.seed)

        self.scheme = scheme

    def simulate(self, N, M=None):
        """
        Simulates the truth table

        Parameters
        ----------
        N: int
            the number of items
        M: int, optional
            the number of true classes

        Returns
        -------
        truth: numpy.ndarray, int
            array of true class indices
        """
        if M is not None:
            self.scheme.b = M

        truth = self.scheme.rvs(N, seed=self.seed)

        return truth
