# Linear discriminant analysis

[kaggle1] https://www.kaggle.com/arthurtok/interactive-intro-to-dimensionality-reduction
[raschka] http://sebastianraschka.com/Articles/2014_python_lda.html

"The goal is to project a dataset onto a lower-dimensional space with good class-separability in order avoid overfitting." [raschka]
Project our feature space (set of n_feature vectors) onto a subspace (set of n_important_features) while maintaining the discriminatory ability. This reduces computational costs and can help prevent overfitting.

Sparse data is more likely to be overfit?

## Relation to PCA
* LDA is basically PCA but instead of just looking for axis that maximise variance of the data, we also want axes that maximise the variance in the classes.
* PCA is unsupervised - just trys to split the data in the most effective way. LDA is supervised - tries to split the data in the most effective way WRT the labels
* LDA not always better than PCA for supervised learning! This is a bit counterintuitive. Could also use both.
* In 2d:
    * PCA splits our data into a widely spead axis and a narrowly spread axis
    * LDA splits our data into a good discrimination axis and a bad discrimination axis.
