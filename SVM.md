# Support Vector Machines
* Non probibalistic, linear classifier.


## Basic Application
* Supervised learning technique for classfication.
* Training set is N, p dimensional vectors, Labels: N labels (can have k possible values)
* N=3, p=2: data - [[1,1], [3,2], [0, -1]], labels - [0, 1, 0]
* Question:
    * Can we separate the data (which is in p dimension space) by a hyperplane (which is in p-1 space)
    * So for out p=2 example, we need p=1 which is a line. Can we draw lines separating things with different labels?

### Linearly separable data
* If our data is linearly separable, then the answer to the question is yes (or if we can separate the data by a hyperplane then it is linearly separable).
* In this case there are probably many possible hyperplanes, unless the data is incredibly clustered around the hyperplane. How to pick the hyperplane?
* Pick the hyperplane which has the max distance to any point. This is the "maximum-margin hyperplane" or "perceptron of optimal stability".
* Note that this hyperplane is only defined by those vectors that are close to it! Vectors not near the boundary have no effect on where we draw our hyperplane. The vectors that determine our hyperplane are the "Support Vectors".

### Should we linearly separate data if we can?
* Imagine we have data 2d vectors that need to be divided into 2 classes. 0 and 1.
* The 0s live at y = 0 for 0 < x < n. The 1s live at y = 1 for 0 < x < n.
* We also have a 0 at y = 1, x = n.
* This data is lineraly separable - the hyperplane (line) that goes through (0,0) and (n,1) divides this data perfectly.
* However this hyperplane has a small max distance. Instead we might accept that we get one data point wrong and use the y=0.5 line as our hyperplane. That might be more stable.
* **Trade off: size of margin vs performance on the training set**

### Non linearly separable data
* Add a penalty for each data point on the wrong side of the hyperplane that is proportional to how far wrong it is. Try to minimise this penalty.
* Add a penalty as the size of margin gets small. (with some scaling factor lambda)
* We now trade off getting things on the wrong side with the size of the margin we want.

### Perceptron interpretation
* Our hyperplane is made up of the set of x vectors that satisfy: w dot x - b = 0
* w is the set of weights our perceptron needs to learn

## Kernels

## Papers
* http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.103.1189&rep=rep1&type=pdf
* Estimation of Dependences Based on Empirical Data
