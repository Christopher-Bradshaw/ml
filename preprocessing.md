# Preprocessing

See http://scikit-learn.org/stable/modules/preprocessing.html

## Scaling (Linear)

* Consider two vectors x_1, x_2 where x_2 = 2 * x_1. Are these the same data?
    * **NO** We don't scale each vector separately
* However is the relation between x_1 and x_2 the same as v_1 and v_2 if v_2 = 2 * v_1?
    * **Yes** - if we assume our data is linear. This is just a scaling, or re-uniting.
* Therefore, we can scale our data so that each component has mean 0, variance 1. Apparently this makes our algos easier to write.
* Note that if we are working with a training and test set we probably want to apply exactly the same scaling to the training set that we did to the test set (rather than just scaling the test set).
    * I'm not actually sure about why...
    * This shouldn't make much of a difference (assuming our test and training set are large and were randomly separated).
    * My guess - you don't need the 0 mean, unit variance in the test set (that is just for the training algo) more important is that the test set is scaled the same way as the training set.
