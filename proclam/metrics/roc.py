"""
A class for the Receiver Operating Curve
"""

from __future__ import absolute_import
__all__ = ['ROC']

import numpy as np
from scipy.integrate import trapz

from .util import weight_sum
from .util import check_weights
from .util import prob_to_det_threshold
from .util import auc, tpr_fpr
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

	def evaluate(self, prediction, truth, grid=None, averaging='per_class'):
		"""
		Evaluates the area under the ROC curve for a given class_idx

		Parameters
		----------
		prediction: numpy.ndarray, float
			predicted class probabilities
		truth: numpy.ndarray, int
			true classes
		grid: numpy.ndarray, float, optional
			array of values between 0 and 1 at which to evaluate ROC
		averaging: string or numpy.ndarray, float
            'per_class' weights classes equally, other keywords possible
            vector assumed to be class weights

		Returns
		-------
		metric: float
			value of the metric
		"""
		if type(grid) == list or type(grid) == numpy.ndarray:
			thresholds)grid = np.array(grid)
		elif type(grid) == float:
			thresholds_grid = np.arange(0., 1., grid)
			n_thresholds = len(thresholds_grid)
		else:
			print('Please specify a grid or spacing for this AUC metric.')
			return

		prediction, truth = np.asarray(prediction), np.asarray(truth)
        prediction_shape = np.shape(prediction)
        (N, M) = prediction_shape
        weights = check_weights(averaging, M, truth=truth)

		auc_allclass = 0

		for m in range(M):
			if not len(np.where(truth == m)[0]):
				raise RuntimeError('No true values for class %i so ROC is undefined'%m)

			thresholds_grid = np.arange(0,1,gridspace)
			n_thresholds = len(thresholds_grid)

			tpr, fpr = np.zeros(n_thresholds), np.zeros(n_thresholds)
			for t,i in zip(thresholds_grid,range(n_thresholds)):
				classifications = prob_to_det_threshold(prediction, m, t)
				tpr_thresh, fpr_thresh = binary_rates(classifications, truth, m)

				#tp = np.sum(classifications[truth == class_idx])
				#fp = np.sum(classifications[truth != class_idx])
				tpr[i] = tpr_thresh #tp/len(classifications[truth == class_idx])
				fpr[i] = fpr_thresh #fp/len(classifications[truth != class_idx])
				#if tpr[i] != tpr[i]: import pdb; pdb.set_trace()

			auc_class = auc(fpr,tpr)

			#fpr = np.concatenate(([0],fpr,[1]),)
			#tpr = np.concatenate(([0],tpr,[1]),)

			#ifpr = np.argsort(fpr)
			#auc = trapz(tpr[ifpr],fpr[ifpr])

			auc_allclass += auc_class*weights[class_idx]

		return auc_allclass
