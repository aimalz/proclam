# classifiers

The "classifiers" in this directory don't take lightcurves as data!  Based on the true classes, they simulate mock classification schemes.  Then they output mock classification results based on that scheme.  The schemes should simulate different anticipated submissions to PLAsTiCC.

To implement a mock classification scheme, make a subclass instance of the `proclam.Classifier` class contained in `classifier.py`.  The `proclam.Classifier.guess` class is an example of one such subclass instance.  Each subclass should be in a separate `.py` file.  

A `proclam.Classifier` subclass instance must have a `classify()` method that takes as input a length N `numpy.ndarray` of true class indices and the number M of true classes (i.e. there can be more classes than appear in the truth table), and it outputs an N*M `numpy.ndarray` with classification probabilities.
