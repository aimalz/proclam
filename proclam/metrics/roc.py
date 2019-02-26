"""
A superclass for metrics
"""

from __future__ import absolute_import
__all__ = ['Metric']

import numpy as np

from .util import weight_sum
from .util import check_weights
from .util import prob_to_det_threshold
from scipy.integrate import trapz

class Metric(object):

	def __init__(self, scheme=None, **kwargs):
		"""
		An object that evaluates the ROC metric

		Parameters
		----------
		scheme: string
			the name of the metric
		"""

		self.debug = False
		self.scheme = scheme

	def evaluate(self, prediction, truth, class_idx, gridspace=0.01, weights=None, **kwds):
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
		for class_idx in range(n_class):
		
			thresholds_grid = np.arange(0,1,gridspace)
			n_thresholds = len(thresholds_grid)
		
			tpr,fpr = np.zeros(n_thresholds),np.zeros(n_thresholds)
			for t,i in zip(thresholds_grid,range(n_thresholds)):
				classifications = prob_to_det_threshold(prediction,class_idx=class_idx,threshold=t)

				tp = np.sum(classifications[truth == class_idx])
				fp = np.sum(classifications[truth != class_idx])
				tpr[i] = tp/len(classifications[truth == class_idx])
				fpr[i] = fp/len(classifications[truth != class_idx])
				#if tpr[i] != tpr[i]: import pdb; pdb.set_trace()
			
			fpr = np.concatenate(([0],fpr,[1]),)
			tpr = np.concatenate(([0],tpr,[1]),)
		
			ifpr = np.argsort(fpr)
			auc = trapz(tpr[ifpr],fpr[ifpr])

			if self.debug:
				import pylab as plt
				plt.clf()
				plt.plot(fpr[ifpr],tpr[ifpr])
				import pdb; pdb.set_trace()

			if weights: auc_allclass += auc*weights[class_idx]
			else: auc_allclass += auc
				
		return auc_allclass
