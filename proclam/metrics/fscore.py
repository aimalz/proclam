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
from sklearn.metrics import f1_score

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

	def evaluate(self, prediction, truth, **kwds):
		"""
		Evaluates the area under the ROC curve for a given class_idx

		Parameters
		----------
		prediction: numpy.ndarray, float
			predicted class probabilities
		truth: numpy.ndarray, int
			true classes

		Returns
		-------
		metric: float
			value of the metric
		"""
		
		best_class = np.zeros(len(truth))
		for i in range(len(truth)):
			best_class[i] = np.where(prediction[i,:] == np.max(prediction[i,:]))[0]
		fscore = f1_score(truth,best_class,average='macro')
			
		return fscore
