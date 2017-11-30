# Principle Component Analysis

Mostly taken from http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf
Should also look at http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html
[kaggle1] https://www.kaggle.com/arthurtok/interactive-intro-to-dimensionality-reduction


## Basic Stats
* take a data set X which has n elements (each element can have multiple components).

### Defined for a single component
* Mean = X_m = Sum (i = 1 to n) (X_i / n)
* Variance = s^2 = Sum (i = i to n) ((X_i - X_m)^2 / (n-1))
    * note the n-1. For some reason you use n-1 when this is a sample, n if it is a full population.
* Standard dev = s = sqrt(variance)

### Defined for multiple components
* Covariance - a measure of how two components vary with respect to each other, e.g. Hours studied vs grade.
    * cov(x, y) = Sum (i = i to n) ((X_i - X_m)(Y_i - Y-M) / (n-1))
    * positive covariance means that as one increases so does the other.
    * negative covariance means that as one increases the other decreases.
    * 0 covariance means they are independent of each other
    * cov(x, y) = cov(y, x)
    * cov(x, y) = var(x) (multiplied by n-1?)
* Covariance matrix
    * If out data has 3 components we can have cov(x|y|z, x|y|z) or 9 covariances. This is a matrix.
        * C_xx C_xy C_xz
        * C_yx C_yy C_yz
        * C_zx C_zy C_zz
    * Note that this is a symmetric matrix (C_xy == C_yz)


### Eigen[vectors|values]
* Can only be found for square matrixes. Not all square matrixes have them, but if they do they have n of them (where n is their size).
* Eigenvectors are orthogonal to each other (you can use them to make a basis)
* Av = kv (where A is the square matrix, v the eigenvector, k the eigenvalue)
* If v is an eigenvector c*V is also an eigenvector. Usually we will scale to get the unit eigenvector.


## PCA (finally!)

If we have n dimensional data currently represented on (x_1, x_2, ... x_n) axis. Can we instead represent that data using a new basis that is more useful? Can we concentrate the usefulness along a few axis allowing us to drop the other axis, speeding up analysis.

### Method
For a 2 dimensional data set
* Rescale each of our components to a 0 mean (X = X - X_m)
* Build the covariance matrix - 2x2 matrix.
* Calculate the eigenvalues/vectors for this matrix
    * Eigenvectors tell you the orientation of the new axis. (Goes through the origin and this vector)
    * Eigenvalues tell you how significant this new axis is. (How stretched it is?)
        * If you like, could ignore smaller eigenvalues
* Multiply the original data set by the eigenvectors (or possibly reduced eigenvectors)
    * You will want to do this for both the test and training set (if relevant)

## Thoughts
* It appears that by applying PCA you will always get worse performance. However, it may be that the speed increase is worth this - you can test more other params. Though once you have found the best other params, you should probably switch back to the full data set.
* PCA is an unsupervised method. This means that if you have labelled data, you don't take advantage of that. This comes back to my thought experiment of:
    * x = [some vars], y = x + some normally distributed random var. label == y > x then 1 else 0
    * if the random var has less variance than x, PCA would throw away x despite all info being contained in the random var.
    * "PCA is meant to be an unsupervised method and therefore not optimised for separating different class labels" [kaggle1]
