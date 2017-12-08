# Challenge Question and Motivation

- The actual question
PLASTICC is a challenge designed to engage people with different relevant expertise to work on the problem of photometrically classifying transients from the kind of data LSST would obtain.
A stated goal of PLASTICC is to engage experts in the field of  have to ask a specific question for the challenge and provide a metric to judge success. It is possible that we could ask a handful of such questions rather than a single one, but still the list has to be small. It is thus important that 
1. Such a list of question is un-ambiguous to people, including those without an astronomy background.
2. The list of the questions address the most important questions for us.
3. Metrics can be formulated to rank the answers to these questions and the rankings consistent with our desirbility.

 
## Questions:

There are three forms of questions we seem to want to ask. There are details in these questions that we need to clarify:

### Classification from complete light curves
#### Motivation
The motivation for this question is to understand the performance of a photometric classification algorithm on the sample of light curves at the end of some time (eg. data releases in LSST or the end of LSST).

#### Question : 
How well does your classifier perform according to our metric 'so and so' at creating a sample of transient X ? (For many purposes pertaining to SN Cosmology X will be SNIa)

#### Issues/Details:
- Define complete light curves (and therefore the set of light curves that needs to be in training or test data) For the length of SNIa light curves, a good timeline is seasons, (except for the highest redshift ones which could have higher seasons). Alternatively, one could phrase this as light curves all the way to a certain threshold in SNR. Longer baselines don't give us much identification but can establish the floor and the lack of further activity (say in contrast to an AGN)

### Early Classification Challenge
#### Motivation
The motivation for this question is to understand the point of time when we might expect to have enough to trigger spectroscopic resources or other photometric resources.

#### Question
How well does your classifier perform according to our metric N days before the peak of transient X?
 

#### Issues/Details:
- How should we define early classification ? Should it be days, or numbers of observed epochs or number of 'colors'. Tentavily, propose this should be days because it tells us how many days we have to get other telescopes to point at it. We could further propose that N should be a number between 5 and 3 days. This is based on [Mark Sullivan's Talk on TIDES in the SN group meeting](https://confluence.slac.stanford.edu/download/attachments/227166428/sullivan_descsnwg_oct2017.pdf?api=v2)  where they talked about being able to schedule live follow-up between 3-1 days. This should be updated on the more ideas from other telescopes.
- Even the peak of a SNIa is unclear, as it depends on band. Let us talk of 'r' band peak for specificity. Happy to move to a better motivated suggestion.  

### Anomaly Detection Detection Challenge
#### Motivation
The motivation for this question is the detection of a rare or unexpected transient in the data.

#### Question
Could you find a transient which has properties not represented among any transient in the training sample?

#### Issues/Details:
This is a question I think is hardest to summarize accurately and judge with an objective metric.
- To do anomaly detection, one could find a set of features and then do cluster analysis based on that. Since the training set is smaller than the test set (We will not have a training set of LSST volume data to analyze the LSST data!) it is likely that there will be outliers on the basis of this, which will have wierd properties compared to the ones in the training set. Likely, we will want the algorithm to find the new ones that are thrown in. How can we explain the kind of properties we would be interested in seeing to the participants ?
- How do we define how large their reply set could be? For example a way of talking about this problem would be to say that we get all the anomalously bright objects. This would include known objects, but we could say that is fine, because we can still deal with the size of their sample.
- How do we deal with intrinsic vs extrinsic? Some transients will be very bright because they are very close (This was an example brought up by Ashley Villar in a previous meeting), and that is relatively rare. The kind of transients that we want to find are intrinsically bright (but would likely occur at higher redshift due to volume) or more likely intrinsically dim (but not the huge population of objects that are dim due to distances). Is explaining this issue a way of treating the first point in of the set of details? 
## Thoughts on Metrics associated with questions

- Selection Cuts: It is important to define quality of data for the use of metrics, through a single selection cut to define a population or multiple selection cuts to define multiple sub-populations. The argument here is  that if a metric is the sum of sub-metrics on slices of population, the the sub-metrics ~ metric on sub-population. While we have not defined what the data quality is, it is usual that the number of objects in a bin of data quality falls exponentially with data quality. Thus, if we include everything the metric will be dominated by values at the tail of poor data quality where a small difference may not be indicative of metric performance (and could be dominated by noise). On the other end, if we have perfect data all classifiers may do pretty well and their differences show up in lower quality data.

