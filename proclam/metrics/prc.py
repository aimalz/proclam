"""
A superclass for metrics
"""

from __future__ import absolute_import
__all__ = ['Metric']

import numpy as np

from .util import weight_sum
from .util import check_weights
from .util import prob_to_det_threshold
from .util import auc, precision, recall
from scipy.integrate import trapz
from sklearn.metrics import precision_recall_curve

class Metric(object):

	def __init__(self, scheme=None, **kwargs):
		"""
		An object that evaluates the F-score

		Parameters
		----------
		scheme: string
			the name of the metric
		"""

		self.debug = False
		self.scheme = scheme

	def evaluate(self, prediction, truth, gridspace=0.01, weights=None, **kwds):
		"""
		Evaluates the area under the ROC curve for a given class_idx

		Parameters
		----------
		prediction: numpy.ndarray, float
			predicted class probabilities
		truth: numpy.ndarray, int
			true classes
		weights: numpy.ndarray, float
			per-class weights

		Returns
		-------
		metric: float
			value of the metric
		"""
		
		auc_allclass = 0
		n_class = np.shape(prediction)[1]
		if not weights:
			weights = [1./n_class]*n_class
		
		for class_idx in range(n_class):
			if not len(np.where(truth == class_idx)[0]):
				raise RuntimeError('No true values for class %i so ROC is undefined'%class_idx)
			
			truth_bool = np.zeros(len(truth),dtype=bool)
			truth_bool[truth == class_idx] = 1
			P,R,thresholds_grid = precision_recall_curve(truth_bool,prediction[:,class_idx])
				
			auc_class = auc(R,P)

			if self.debug:
				import pylab as plt
				plt.clf()
				plt.plot(R,P)
				plt.show()
				import pdb; pdb.set_trace()

			auc_allclass += auc_class*weights[class_idx]
				
		return auc_allclass
