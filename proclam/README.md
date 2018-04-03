# `proclam` testing suite

We want to choose a metric whose results are generally in accordance with our intuition for what makes a good or bad classifier.  It must be resilient to pathological effects in the true classes and must be able to handle classification submissions of a variety of forms.  To test all these effects, there are three superclasses: simulators, classifiers, and metrics.  The test cases discussed in the issues should be implemented via subclass instances of these superclasses.  

Members of the PLAsTiCC team, particularly those who are not assisting in the validation efforts, are encouraged to implement some of the test cases, outlined in issues, as subclass instances of these superclasses.  There's currently at least one subclass instance already implemented to use as a template.  

To contribute, **please make a new branch** and _test your code in the `pipeline_sandbox.ipynb` jupyter notebook_ (with any necessary modifications, or with a similar script committed to this repository) before submitting a pull request.
