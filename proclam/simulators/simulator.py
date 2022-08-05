"""
A superclass for simulators
"""

from __future__ import absolute_import
__all__ = ['Simulator']

import numpy as np


class Simulator(object):

    def __init__(self, scheme=None, seed=0):
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
