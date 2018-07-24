"""
A subclass for a log-scaled unbalanced distribution of classes.
"""

from __future__ import absolute_import
__all__ = ['LogUnbalanced']

import numpy as np

from .simulator import Simulator

class LogUnbalanced(Simulator):

    def __init__(self, scheme='log-unbalanced', seed=0):
        """
        An object that simulates unbalanced true class assignments such that the probability of an object being in class 'x' (with 0<=x<=M) is proportional to 10**y, where y is a draw from a uniform distribution U(0,M).

        Parameters
        ----------
        scheme: string
            the name of the simulator
        seed: int, optional
            the random seed to use, handy for testing
        """

        super(LogUnbalanced, self).__init__(scheme, seed)
        np.random.seed(seed=self.seed)

    def simulate(self, M, N):
        """
        Simulates the truth table as an unbalanced distribution

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
        seeds = np.random.uniform(M, size=M)
        # seeds = [max(seeds[i], 10./M) for i in range(M)]
        counts = 10 ** (np.sort(seeds))
        prob_classes = counts / np.sum(counts)
        truth = np.random.choice(M, size=N, p=prob_classes)

        return truth
