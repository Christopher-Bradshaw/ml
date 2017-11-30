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


## Dimensionality reduction

### Curse of dimensionality
* As you have more dimensions, the number of data points you need to ensure a reasonble sampling increases. Thus having a large dimensions to data points ratio makes over fitting more likely
* Another definition appears to be that increasing definitions just increases search space

### Reducing dimensions
* Consider some data in 3d space that hash x = [ <some values> ], y = [ <some values> ], z = x. This is actually just 2d data maskerading as 3d data. The z axis gives us nothing new.
* We can reduce this to 2d data and so speed up our fitting/analysis.
* We can also reduce dimensions by throwing away some information
    * This appears to always make your model perform slightly worse (as expected?)
    * Maybe you should use this to try many other parameters and then run the final with full dimensions?
    * But maybe this helps avoid overfitting and so improves things?
