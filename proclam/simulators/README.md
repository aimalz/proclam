# simulators

The "simulators" in this directory don't make lightcurves!  A simulator function produces the truth table given the number of objects, the number of classes, and any keyword arguments necessary to generate the balance of classes.  They should be subclass instances of the `proclam.Simulator` class, each in a separate `.py` file.  The `proclam.Simulator.Uniform` subclass defined in `uniform.py` is provided as an example.

A `proclam.Simulator` subclass instance must have a `simulate()` method that takes as input the number M of true classes, the number N of catalog objects.  A `proclam.Simulator` subclass may also be initialized with keyword parameters impacting the class probabilities.  Additionally, `proclam.Simulator` subclass objects may benefit from internal helper functions to generate such class probabilities.
