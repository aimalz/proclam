# metric modules

The metric modules in this directory should each be in their own `.py` file.  They should take as input the dataset of mock classification results and output a single scalar, but they may need additional keyword inputs for the AUC thresholds or multi-class weighting scheme.

# shared utilities

A utility file in this directory containing functions shared over all metrics.

## AUC evaluator

One function must evaluate the AUC given a deterministic metric, a dataset of mock classification results, and the probability thresholds.

## multi-class combiners

There must be a function enabling combination of metrics on a per-object or per-class basis, probably by using a keyword argument.