import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin

# PCA and CSP is the same ??
# good explanation
# https://towardsdatascience.com/a-step-by-step-implementation-of-principal-component-analysis-5520cc6cd598
# for working with multiple channels/ and classes
# https://github.com/mne-tools/mne-python/blob/main/mne/decoding/csp.py
# great code
# https://www.askpython.com/python/examples/principal-component-analysis
# explanation and code
# https://www.youtube.com/watch?v=Rjr62b_h7S4

class CSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components

    def covariance_matrix(self, X, y):
        cov = []
        _ , channels, _ = X.shape

        for class_ in self.classes:
            x_class = X[y == class_]
            x_class = np.transpose(x_class, [1, 0, 2])
            x_class = x_class.reshape(channels, -1)
            x_class -= np.mean(x_class, axis=1)[:, None] # mean-center data
            cov_mat = np.cov(x_class)
            cov.append(cov_mat)
        return np.stack(cov) # join covariance matrixes along axis 0 (2, 13, 13)

    def fit(self, X, y):
        self.classes = np.unique(y)

        # get concatenation of covariance matrixes
        covs = self.covariance_matrix(X, y)

        # Calculating Eigenvalues and Eigenvectors of the covariance matrix
        eigen_values , eigen_vectors = linalg.eigh(covs[0], covs.sum(0))
        # sort the eigenvalues/vector
        sorted_index = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]

        # create list of new axes for projection
        self.pick_filters = sorted_eigenvectors.T[:self.n_components]
        return self

    def transform(self, X):
        # Project data on new axes 
        X = np.asarray([np.dot(self.pick_filters, epoch) for epoch in X])
        
        # compute mean band power
        X = (X ** 2).mean(axis=2)
        
        # z-score normalization
        X -= X.mean()
        X /= X.std()

        return X

    def fit_transform(self, X, y):
        self.fit(X,y)
        ret = self.transform(X)
        return ret
