# metric modules

The metrics in this directory should be subclass instances of the `proclam.Metric` class contained in `metric.py`.  Each subclass should be in a separate `.py` file.  The log-loss metric is already implemented in `logloss.py` and can be used as a template for other metrics.

A `proclam.Metric` subclass instance must have an `evaluate()` method that takes as input a length N `numpy.ndarray` of true class indices and an N*M `numpy.ndarray` with classification probabilities, and it outputs a single scalar value.  

Both the initialization and `evaluate()` methods may accept additional keyword parameters.  Likely examples of such parameters are AUC thresholds and a multi-class weighting scheme.

`proclam.Metric` subclass instances, particularly if the metric is AUC-based, may benefit from internal helper functions to evaluate the base metric at on deterministic reductions.

# shared utilities

There will be a utility file in this directory containing functions shared over all metrics.  These are particularly critical for AUC-based metrics and any adaptations of deterministic metrics but also for any metrics that aren't naturally accommodating of multiple classes.

## multi-class combiners

There must be at least one function enabling combination of per-object metrics given the per-object classification probabilities, as metric results can differ based on whether they average over individual objects or over pre-averaged class summaries.

## evaluation of the true/false positive/negative rates

Many metrics are built on these basic quantities.  There should be a set of shared functions that compute these so we don't need to implement them for adaptations of deterministic metrics.

## probability reducer

There must be a function to produce classification point estimates based on the classification probabilities and a probability threshold.

## curve generator

There must be a function to evaluate a deterministic metric at many probability thresholds.

## AUC evaluator

One function must evaluate the AUC given the probability thresholds and the deterministic metric's values at those probability thresholds.
