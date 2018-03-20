"""
A subclass for a distribution of classes dictated by a `scipy.stats.rv_discrete` distribution.  See [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/stats.html#discrete-distributions) for usage.
"""

import numpy as np

from simulator import Simulator

class SciPyDist(Simulator):

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

        super(SciPyDist, self).__init__(scheme, seed)
        # np.random.seed(seed=self.seed)

        self.scheme = scheme
        self.scheme.seed = self.seed

    def simulate(self, N, M):
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
            array of true class indices
        """
        try:
            if M is not None:
                self.scheme.b = M

        except TypeError:
            print('Cannot use '+str(type(self.scheme))+' because it is unbounded.')
            return

        try:
            truth = self.scheme.rvs(N)

            assert max(truth) < M

        except AssertionError:
            print('Cannot use '+str(type(self.scheme))+' because it is unbounded.')
            return
        return truth
