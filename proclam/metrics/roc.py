"""
A class for the Receiver Operating Curve
"""

from __future__ import absolute_import
__all__ = ['ROC']

import numpy as np

from .util import weight_sum
from .util import check_weights
from .util import check_auc_grid
from .util import prob_to_det
from .util import det_to_cm
from .util import cm_to_rate
from .util import auc
from .metric import Metric

class ROC(Metric):

	def __init__(self, scheme=None):
		"""
		An object that evaluates the ROC metric

		Parameters
		----------
		scheme: string
			the name of the metric
		"""
		super(ROC, self).__init__(scheme)
		self.scheme = scheme

	def evaluate(self, prediction, truth, grid, averaging='per_class'):
		"""
		Evaluates the area under the ROC curve for a given class_idx

		Parameters
		----------
		prediction: numpy.ndarray, float
			predicted class probabilities
		truth: numpy.ndarray, int
			true classes
		grid: numpy.ndarray, float or float or int
			array of values between 0 and 1 at which to evaluate ROC
		averaging: string or numpy.ndarray, float
			'per_class' weights classes equally, other keywords possible, vector assumed to be class weights

		Returns
		-------
		metric: float
			value of the metric
		"""
		thresholds_grid = check_auc_grid(grid)
		n_thresholds = len(thresholds_grid)

		prediction, truth = np.asarray(prediction), np.asarray(truth)
		(N, M) = np.shape(prediction)

		auc_class = np.empty(M)

		for m in range(M):
			if not len(np.where(truth == m)[0]):
				raise RuntimeError('No true values for class %i so ROC is undefined'%m)

			tpr, fpr = np.zeros(n_thresholds), np.zeros(n_thresholds)
			for i, t in enumerate(thresholds_grid):
				dets = prob_to_det(prediction, m, threshold=t)
				cm = det_to_cm(dets, truth)
				rates = cm_to_rate(cm)
				tpr[i] = rates.TPR[m]
				fpr[i] = rates.FPR[m]

			auc_class[m] = auc(fpr, tpr)

		weights = check_weights(averaging, M, truth=truth)
		auc_allclass = weight_sum(auc_class, weights)

		return auc_allclass
