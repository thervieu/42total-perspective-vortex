import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

class CSP(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=4, tranform_into='average_power'):
    
        if not isinstance(n_components, int):
            raise ValueError('n_components must be an integer.')
        
        self.n_components =  n_components
        self.mean_ = 0
        self.std_ = 0
    
    def _calculate_covariance(self, X):
        n = X.shape[1]
        X -= X.mean(axis=1)[:, None]
        return np.dot(X, X.T.conj()) / float(n)


    def _compute_covariance_matrices(self, X, y):
        _, n_channels, _ = X.shape

        covs = []
        for cur_class in self._classes:
            """Concatenate epochs before computing the covariance."""
            x_class = X[y==cur_class]
            x_class = np.transpose(x_class, [1, 0, 2])
            x_class = x_class.reshape[n_channels, -1]
            cov = self._calculate_covariance(x_class)
            covs.append(cov)
        
        return np.stack(covs)

    def _decompose_covs(sef, covs):
        from scipy import linalg
        n_classes = len(covs)
        if n_classes == 2:
            eigen_values, eigen_vectors = linalg.eigh(covs[0], covs.sum(0))
        else:
            raise Exception('_decompose_covs: more than 2 classes not handled')
        return eigen_vectors, eigen_values
    
    def _order_components(self, eigne_values):
        n_classes = len(self._classes)

        if n_classes == 2:
            ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        else:
            raise Exception('_order_components: more than 2 classes not handled')
        return ix

    def fit(self, X, y):
        """Estimate the CSP decomposition on epochs.
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            The data on which to estimate the CSP.
        y : array, shape (n_epochs,)
            The class for each epoch.
        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        """

        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2.")
        

        covs = self._compute_covariance_matrices(X, y)
        eigen_vectors, eigen_values = self._decompose_covs(covs)
    
        ix = self._order_components(eigen_values)
        eigen_vectors = eigen_vectors[:, ix]

        self.filters_ = eigen_vectors.T
        pick_filters = self.filters_[:self.n_components]

        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])
        # compute features (mean power)
        X = (X ** 2).mean(axis=2)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return self

    def transform(self, X):
        """Estimate epochs sources given the CSP filters.
        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            The data.
        Returns
        -------
        X : ndarray
            If self.transform_into == 'average_power' then returns the power of
            CSP features averaged over time and shape (n_epochs, n_sources)
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)." % type(X))
        if self.filters_ is None:
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean band power)
        if self.transform_into == 'average_power':
            X = (X ** 2).mean(axis=2)
            log = True if self.log is None else self.log
            if log:
                X = np.log(X)
            else:
                X -= self.mean_
                X /= self.std_
        return X
    


    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
