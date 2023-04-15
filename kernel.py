"""The weisfeiler lehman kernel :cite:`shervashidze2011weisfeiler`."""
# Author: Ioannis Siglidis <y.siglidis@gmail.com>
# License: BSD 3 clause
import warnings
from collections import Counter
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import networkx
data_path = "data/"
# Python 2/3 cross-compatibility import
from six import iteritems
from six import itervalues
from six.moves.collections_abc import Iterable
from scipy.sparse import csr_matrix
from sklearn.svm import SVC
from utils import grid_search, results_to_csv, plot_graphs, insert_zeros
import svm

class Kernel(BaseEstimator, TransformerMixin):
    """A general class for graph kernels.
    At default a kernel is considered as pairwise. 
    Parameters
    ----------
    normalize : bool, optional
        Normalize the output of the graph kernel.
    Attributes
    ----------
    X : list of graphs
    _method_calling : int
        An inside enumeration defines which method calls another method.
            - 1 stands for fit
            - 2 stands for fit_transform
            - 3 stands for transform
    """

    X = None
    _graph_format = "dictionary"
    _method_calling = 0

    def __init__(self, normalize=False):
        """`__init__` for `kernel` object."""
        self.normalize = normalize
        self._initialized = dict()

    def fit(self, X, y=None):
        """Fit a dataset, for a transformer.
        Parameters
        ----------
        X : iterable
        Returns
        -------
        self : object
        Returns self.
        """
        self._method_calling = 1

        # Parameter initialization
        self.initialize()
        self.X = self.parse_input(X)    

        # Return the transformer
        return self

    def transform(self, X):
        """Calculate the kernel matrix, between given and fitted dataset.
        Parameters
        ----------
        X : iterable
        Returns
        -------
        K : numpy array, shape = [n_targets, n_input_graphs]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features
        """
        self._method_calling = 3
        # Check is fit had been called
        check_is_fitted(self, ['X'])

        # Input validation and parsing
        if X is None:
            raise ValueError('`transform` input cannot be None')
        else:
            Y = self.parse_input(X)

        # Transform - calculate kernel matrix
        km = self._calculate_kernel_matrix(Y)
        self._Y = Y

        # Self transform must appear before the diagonal call on normilization
        self._is_transformed = True
        if self.normalize:
            X_diag, Y_diag = self.diagonal()
            km /= np.sqrt(np.outer(Y_diag, X_diag))
        return km

    def fit_transform(self, X):
        """Fit and transform, on the same dataset.
        Parameters
        ----------
        X : iterable
            
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        K : numpy array, shape = [n_targets, n_input_graphs]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features
        """
        self._method_calling = 2
        self.fit(X)

        # Transform - calculate kernel matrix
        km = self._calculate_kernel_matrix()

        self._X_diag = np.diagonal(km)
        if self.normalize:
            return km / np.sqrt(np.outer(self._X_diag, self._X_diag))
        else:
            return km

    def initialize(self):
        """Initialize all transformer arguments, needing initialisation."""

    def _calculate_kernel_matrix(self, Y=None):
        """Calculate the kernel matrix given a target_graph and a kernel.
        Each a matrix is calculated between all elements of Y on the rows and
        all elements of X on the columns.
        Parameters
        ----------
        Y : list
        Returns
        -------
        K : numpy array, shape = [n_targets, n_inputs]
            The kernel matrix
        """
        if Y is None:
            K = np.zeros(shape=(len(self.X), len(self.X)))
            cache = list()
            for (i, x) in enumerate(self.X):
                K[i, i] = self.pairwise_operation(x, x)
                for (j, y) in enumerate(cache):
                    K[j, i] = self.pairwise_operation(y, x)
                cache.append(x)
            K = np.triu(K) + np.triu(K, 1).T

        else:
            K = np.zeros(shape=(len(Y), len(self.X)))
            for (j, y) in enumerate(Y):
                for (i, x) in enumerate(self.X):
                    K[j, i] = self.pairwise_operation(y, x)
        return K

    def diagonal(self):
        """Calculate the kernel matrix diagonal of the fit/transformed data.
        Returns
        -------
        X_diag : np.array
            The diagonal of the kernel matrix between the fitted data.
        Y_diag : np.array
            The diagonal of the kernel matrix, of the transform.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X'])
        try:
            check_is_fitted(self, ['_X_diag'])
        except NotFittedError:
            # Calculate diagonal of X
            self._X_diag = np.empty(shape=(len(self.X),))
            for (i, x) in enumerate(self.X):
                self._X_diag[i] = self.pairwise_operation(x, x)

        try:
            # If transform has happened return both diagonals
            check_is_fitted(self, ['_Y'])
            Y_diag = np.empty(shape=(len(self._Y),))
            for (i, y) in enumerate(self._Y):
                Y_diag[i] = self.pairwise_operation(y, y)

            return self._X_diag, Y_diag
        except NotFittedError:
            # Else just return both X_diag
            return self._X_diag

    def set_params(self, **params):
        """Call the parent method."""
        if len(self._initialized):
            # Copy the parameters
            params = copy.deepcopy(params)

            # Iterate over the parameters
            for key, value in iteritems(params):
                key, delim, sub_key = key.partition('__')
                if delim:
                    if sub_key in self._initialized:
                        self._initialized[sub_key] = False
                elif key in self._initialized:
                    self._initialized[key] = False

        # Set parameters
        super(Kernel, self).set_params(**params)


def indexes(n_jobs, nsamples):
    """Distribute samples accross n_jobs."""
    n_jobs = n_jobs

    if n_jobs >= nsamples:
        for i in range(nsamples):
            yield (i, i+1)
    else:
        ns = nsamples/n_jobs
        start = 0
        for i in range(n_jobs-1):
            end = start + ns
            yield (int(start), int(end))
            start = end
        yield (int(start), nsamples)


def assign(data, K, pairwise_operation):
    """Assign list values of an iterable to a numpy array while calculating a pairwise operation."""
    for d in data:
        K[d[0][0], d[0][1]] = pairwise_operation(d[1][0], d[1][1])