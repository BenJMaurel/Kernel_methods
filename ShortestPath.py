import networkx
import numpy as np
from kernel import Kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

class ShortestPath(Kernel):
    """
    Shortest Path Kernel
    """

    _graph_bins = dict()

    def __init__(self,
                 normalize=False, with_labels = True):
        """Initialize a `shortest_path` kernel."""
        super(ShortestPath, self).__init__(normalize=normalize)
        self.with_labels = with_labels
        self._initialized.update({"with_labels": False, "algorithm_type": False})

    def initialize(self):
        """Initialize all transformer arguments, needing initialization."""
        if not self._initialized["with_labels"]:
            if self.with_labels:
                self._lt = "vertex"
                self._lhash = lhash_labels
                self._decompose_input = decompose_input_labels

    def transform(self, X):
        """Calculate the kernel matrix, between given and fitted dataset.
        Parameters
        ----------
        X : iterable of graphs
        Returns
        -------
        K : numpy array, shape = [n_targets, n_input_graphs]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features
        """
        self._method_calling = 3
        # Check is fit had been called
        check_is_fitted(self, ['X', '_nx', '_enum'])

        # Input validation and parsing
        Y = self.pre_calc_matrix(X)
        
        # Transform - calculate kernel matrix
        try:
            check_is_fitted(self, ['_phi_X'])
            phi_x = self._phi_X
        except NotFittedError:
            phi_x = np.zeros(shape=(self._nx, len(self._enum)))
            for i in self.X.keys():
                for j in self.X[i].keys():
                    phi_x[i, j] = self.X[i][j]
            self._phi_X = phi_x

        phi_y = np.zeros(shape=(self._ny, len(self._enum) + len(self._Y_enum)))
        for i in Y.keys():
            for j in Y[i].keys():
                phi_y[i, j] = Y[i][j]

        self._phi_Y = phi_y
        km = np.dot(phi_y[:, :len(self._enum)], phi_x.T)
        self._is_transformed = True
        if self.normalize:
            X_diag, Y_diag = self.diagonal()
            return km / np.sqrt(np.outer(Y_diag, X_diag))
        else:
            return km

    def diagonal(self):
        """Calculate the kernel matrix diagonal for fitted data.
        A funtion called on transform on a seperate dataset to apply
        normalization on the exterior.
        Parameters
        ----------
        None.
        Returns
        -------
        X_diag : np.array
            The diagonal of the kernel matrix, of the fitted data.
            This consists of kernel calculation for each element with itself.
        Y_diag : np.array
            The diagonal of the kernel matrix, of the transformed data.
            This consists of kernel calculation for each element with itself.
        """
        # calculate feature matrices.
        phi_x = np.zeros(shape=(self._nx, len(self._enum)))

        for i in self.X.keys():
            for j in self.X[i].keys():
                phi_x[i, j] = self.X[i][j]
                # Transform - calculate kernel matrix
        self._phi_X = phi_x

        self._X_diag = np.sum(np.square(self._phi_X), axis=1)
        self._X_diag = np.reshape(self._X_diag, (self._X_diag.shape[0], 1))

        try:
            check_is_fitted(self, ['_phi_Y'])
            # Calculate diagonal of Y
            Y_diag = np.sum(np.square(self._phi_Y), axis=1)
            return self._X_diag, Y_diag
        except NotFittedError:
            return self._X_diag

    def fit_transform(self, X, y=None):
        """Fit and transform, on the same dataset.
        Parameters
        ----------
        X : iterable of graphs
        y : Object, default=None
            Ignored argument, added for the pipeline.
        Returns
        -------
        K : numpy array, shape = [n_targets, n_input_graphs]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features
        """
        self._method_calling = 2
        self.fit(X)
        
        # calculate feature matrices.
        phi_x = np.zeros(shape=(self._nx, len(self._enum)))

        for i in self.X.keys():
            for j in self.X[i].keys():
                phi_x[i, j] = self.X[i][j]

        # Transform - calculate kernel matrix
        self._phi_X = phi_x
        km = np.dot(phi_x, phi_x.T)
        self._X_diag = np.diagonal(km)
        if self.normalize:
            return np.divide(km, np.sqrt(np.outer(self._X_diag, self._X_diag)))
        else:
            return km

    def pre_calc_matrix(self, X):
        """Parse and create features for "shortest path" kernel.
        Parameters
        ----------
        X : iterable of graphs
        Returns
        -------
        sp_counts : dict
            A dictionary that for each vertex holds the counts of shortest path
            tuples.
        """
        
        i = -1
        sp_counts = dict()
        if self._method_calling == 1:
            self._enum = dict()
        elif self._method_calling == 3:
            self._Y_enum = dict()
        for (idx, x) in enumerate(iter(X)):
            spm_data = (networkx.floyd_warshall_numpy(x), {key: value[0] for key, value in networkx.get_node_attributes(x, 'labels').items()})

            # Build the shortest path matrix
            i += 1
            
            S, L = self._decompose_input(spm_data)
            sp_counts[i] = dict()
            for u in range(S.shape[0]):
                for v in range(S.shape[1]):
                    if u == v or S[u, v] == float("Inf"):
                        continue
                    label = self._lhash(S, u, v, *L)
                    if label not in self._enum:
                        if self._method_calling == 1:
                            idx = len(self._enum)
                            self._enum[label] = idx
                        elif self._method_calling == 3:
                            if label not in self._Y_enum:
                                idx = len(self._enum) + len(self._Y_enum)
                                self._Y_enum[label] = idx
                            else:
                                idx = self._Y_enum[label]
                    else:
                        idx = self._enum[label]
                    if idx in sp_counts[i]:
                        sp_counts[i][idx] += 1
                    else:
                        sp_counts[i][idx] = 1

        if i == -1:
            raise ValueError('parsed input is empty')

        if self._method_calling == 1:
            self._nx = i+1
        elif self._method_calling == 3:
            self._ny = i+1
        return sp_counts

def lhash_labels(S, u, v, *args):
    return (args[0][u], args[0][v], S[u, v])


def decompose_input_labels(args):
    return (args[0], args[1:])

class ShortestPath2(BaseEstimator, TransformerMixin):
    """
    A class for computing the shortest path kernel on NetworkX graphs.
    """
    
    def __init__(self, normalize=True):
        """
        :param normalize: Whether to normalize the kernel matrix (default: True)
        """
        self.normalize = normalize
    
    def fit_transform(self, X, y=None):
        """
        Fit the kernel on the input graphs and compute the kernel matrix between X and the fitted graphs.
        :param X: A list of NetworkX graph objects
        :param y: Unused (required for compatibility with scikit-learn)
        :return: A kernel matrix between X and the fitted graphs
        """
        
        self.graphs_ = X
        K = np.zeros((len(X), len(self.graphs_)))
        for i, G1 in tqdm(enumerate(X)):
            for j, G2 in enumerate(self.graphs_):
                dist1, _ = nx.single_source_dijkstra(G1, 0)
                dist2, _ = nx.single_source_dijkstra(G2, 0)
                kernel_value = sum([1 / (2 ** dist1[node] + 2 ** dist2[node]) for node in set(dist1) & set(dist2)])
                K[i, j] = kernel_value
        if self.normalize:
            K = pairwise_kernels(K, metric='linear')
        return K
    
    def transform(self, X):
        """
        Compute the kernel matrix between the input graphs and the fitted graphs.
        :param X: A list of NetworkX graph objects
        :return: A kernel matrix between X and the fitted graphs
        """
        K = np.zeros((len(X), len(self.graphs_)))
        for i, G1 in tqdm(enumerate(X)):
            for j, G2 in enumerate(self.graphs_):
                dist1, _ = nx.single_source_dijkstra(G1, 0)
                dist2, _ = nx.single_source_dijkstra(G2, 0)
                kernel_value = sum([1 / (2 ** dist1[node] + 2 ** dist2[node]) for node in set(dist1) & set(dist2)])
                K[i, j] = kernel_value
        if self.normalize:
            K = pairwise_kernels(K, metric='linear')
        return K
