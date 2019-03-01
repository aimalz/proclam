"""
A superclass for metrics
"""

from __future__ import absolute_import
__all__ = ['Metric']

import numpy as np

from .util import weight_sum
from .util import check_weights
from .util import prob_to_det_threshold
from .util import auc, tpr_fpr
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
			
			thresholds_grid = np.arange(0,1,gridspace)
			n_thresholds = len(thresholds_grid)
		
			tpr,fpr = np.zeros(n_thresholds),np.zeros(n_thresholds)
			for t,i in zip(thresholds_grid,range(n_thresholds)):
				classifications = prob_to_det_threshold(prediction,class_idx=class_idx,threshold=t)
				
				tpr_thresh,fpr_thresh = tpr_fpr(classifications,truth,class_idx)
				
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

			if self.debug:
				import pylab as plt
				plt.clf()
				plt.plot(fpr[ifpr],tpr[ifpr])
				import pdb; pdb.set_trace()

			auc_allclass += auc_class*weights[class_idx]
				
		return auc_allclass
