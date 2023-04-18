
import warnings
from collections import Counter
import numpy as np
import pickle
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import networkx
data_path = "data/"
from six import iteritems
from six import itervalues
from scipy.sparse import csr_matrix
from utils import grid_search, results_to_csv, plot_graphs, insert_zeros
from kernel import Kernel
from VertexHistogram import VertexHistogram
from tqdm import tqdm

class WeisfeilerLehman(Kernel):
    """Compute the Weisfeiler Lehman Kernel.
    
    Parameters
    ----------
    n_iter : int, default=5
        The number of iterations.
    """

    def __init__(self,
                 normalize=False, n_iter=5, base_graph_kernel=VertexHistogram):
        """Initialise a `weisfeiler_lehman` kernel."""
        super(WeisfeilerLehman, self).__init__(normalize=normalize)

        self.n_iter = n_iter
        self.base_graph_kernel = base_graph_kernel
        self._initialized.update({"n_iter": False, "base_graph_kernel": False})

    def initialize(self):
        """Initialize all transformer arguments, needing initialization."""
        super(WeisfeilerLehman, self).initialize()
        if not self._initialized["base_graph_kernel"]:
            
            base_graph_kernel, params = VertexHistogram, dict()
            params["normalize"] = False
            self._base_graph_kernel = base_graph_kernel
            self._params = params
            self._initialized["base_graph_kernel"] = True

        if not self._initialized["n_iter"]:
            self._n_iter = self.n_iter + 1
            self._initialized["n_iter"] = True

    def pre_calc_matrix(self, X):
        """Get_ready to compute matrix.
        Parameters
        ----------
        X : iterable of graphs
        Returns
        -------
        base_graph_kernel : object
        Returns base_graph_kernel.
        """
        if hasattr(self, '_X_diag'):
            # Clean _X_diag value
            delattr(self, '_X_diag')

        nx = 0
        Gs_ed, L, distinct_values, extras = dict(), dict(), set(), dict()
        for (idx, x) in enumerate(iter(X)):
            el = networkx.get_edge_attributes(x, 'labels')
            if el is None:
                extra = tuple()
            else:
                extra = (el, )
            # import pdb; pdb.set_trace()
            Gs_ed[nx] = networkx.to_dict_of_dicts(x)
            Gs_ed[nx] = {k1: {k2: v2['labels'][0] for k2, v2 in v1.items()} for k1, v1 in Gs_ed[nx].items()}

            L[nx] = networkx.get_node_attributes(x, 'labels')
            L[nx] = {key: value[0] for key, value in L[nx].items()}
            extras[nx] = extra
            distinct_values |= set(itervalues(L[nx]))
            nx += 1
        if nx == 0:
            raise ValueError('parsed input is empty')

        # Save the number of "fitted" graphs.
        self._nx = nx

        # get all the distinct values of current labels
        WL_labels_inverse = dict()

        # assign a number to each label
        label_count = 0
        for dv in sorted(list(distinct_values)):
            WL_labels_inverse[dv] = label_count
            label_count += 1

        # Initalize an inverse dictionary of labels for all iterations
        self._inv_labels = dict()
        self._inv_labels[0] = WL_labels_inverse

        def generate_graphs(label_count, WL_labels_inverse):
            new_graphs = list()
            for j in range(nx):
                new_labels = dict()
                for k in L[j].keys():
                    new_labels[k] = WL_labels_inverse[L[j][k]]
                L[j] = new_labels
                # add new labels
                new_graphs.append((Gs_ed[j], new_labels) + extras[j])
            yield new_graphs

            for i in range(1, self._n_iter):
                label_set, WL_labels_inverse, L_temp = set(), dict(), dict()
                for j in range(nx):
                    # Find unique labels and sort
                    # them for both graphs
                    # Keep for each node the temporary
                    L_temp[j] = dict()
                    for v in Gs_ed[j].keys():
                        credential = str(L[j][v]) + "," + \
                            str(sorted([L[j][n] for n in Gs_ed[j][v].keys()]))
                        L_temp[j][v] = credential
                        label_set.add(credential)

                label_list = sorted(list(label_set))
                for dv in label_list:
                    WL_labels_inverse[dv] = label_count
                    label_count += 1

                # Recalculate labels
                new_graphs = list()
                for j in range(nx):
                    new_labels = dict()
                    for k in L_temp[j].keys():
                        new_labels[k] = WL_labels_inverse[L_temp[j][k]]
                    L[j] = new_labels
                    # relabel
                    new_graphs.append((Gs_ed[j], new_labels) + extras[j])
                self._inv_labels[i] = WL_labels_inverse
                yield new_graphs
        
        base_graph_kernel = {i: self._base_graph_kernel(**self._params) for i in range(self._n_iter)}
        import pdb; pdb.set_trace()
        if self._method_calling == 1:
            for (i, g) in tqdm(enumerate(generate_graphs(label_count, WL_labels_inverse))):
                base_graph_kernel[i].fit(g)
        elif self._method_calling == 2:
            graphs = generate_graphs(label_count, WL_labels_inverse)
            values = [
                base_graph_kernel[i].fit_transform(g) for (i, g) in enumerate(graphs)
            ]
            K = np.sum(values, axis=0)

        if self._method_calling == 1:
            return base_graph_kernel
        elif self._method_calling == 2:
            return K, base_graph_kernel

    def fit_transform(self, X, y=None):
        """Fit and transform, on the same dataset.
        Parameters
        ----------
        X : graphs iterable

        Returns
        -------
        K : numpy array, shape = [n_targets, n_input_graphs]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features
        """
        self._method_calling = 2
        self._is_transformed = False
        self.initialize()
        km, self.X = self.pre_calc_matrix(X)

        self._X_diag = np.diagonal(km)
        if self.normalize:
            old_settings = np.seterr(divide='ignore')
            km = np.nan_to_num(np.divide(km, np.sqrt(np.outer(self._X_diag, self._X_diag))))
            np.seterr(**old_settings)
        return km

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
        check_is_fitted(self, ['X', '_nx', '_inv_labels'])
        nx = 0
        distinct_values = set()
        Gs_ed, L = dict(), dict()
        for (i, x) in enumerate(iter(X)):
            Gs_ed[nx] = networkx.to_dict_of_dicts(x)
            Gs_ed[nx] = {k1: {k2: v2['labels'][0] for k2, v2 in v1.items()} for k1, v1 in Gs_ed[nx].items()}
            L[nx] = networkx.get_node_attributes(x, 'labels')
            L[nx] = {key: value[0] for key, value in L[nx].items()}
            # Hold all the distinct values
            distinct_values |= set(
                v for v in itervalues(L[nx])
                if v not in self._inv_labels[0])
            nx += 1

        nl = len(self._inv_labels[0])
        WL_labels_inverse = {dv: idx for (idx, dv) in
                             enumerate(sorted(list(distinct_values)), nl)}

        def generate_graphs(WL_labels_inverse, nl):
            # calculate the kernel matrix for the 0 iteration
            new_graphs = list()
            for j in range(nx):
                new_labels = dict()
                for (k, v) in iteritems(L[j]):
                    if v in self._inv_labels[0]:
                        new_labels[k] = self._inv_labels[0][v]
                    else:
                        new_labels[k] = WL_labels_inverse[v]
                L[j] = new_labels
                # produce the new graphs
                new_graphs.append([Gs_ed[j], new_labels])
            yield new_graphs

            for i in range(1, self._n_iter):
                new_graphs = list()
                L_temp, label_set = dict(), set()
                nl += len(self._inv_labels[i])
                for j in range(nx):
                    # Find unique labels and sort them for both graphs
                    # Keep for each node the temporary
                    L_temp[j] = dict()
                    for v in Gs_ed[j].keys():
                        credential = str(L[j][v]) + "," + \
                            str(sorted([L[j][n] for n in Gs_ed[j][v].keys()]))
                        L_temp[j][v] = credential
                        if credential not in self._inv_labels[i]:
                            label_set.add(credential)

                # Calculate the new label_set
                WL_labels_inverse = dict()
                if len(label_set) > 0:
                    for dv in sorted(list(label_set)):
                        idx = len(WL_labels_inverse) + nl
                        WL_labels_inverse[dv] = idx

                # Recalculate labels
                new_graphs = list()
                for j in range(nx):
                    new_labels = dict()
                    for (k, v) in iteritems(L_temp[j]):
                        if v in self._inv_labels[i]:
                            new_labels[k] = self._inv_labels[i][v]
                        else:
                            new_labels[k] = WL_labels_inverse[v]
                    L[j] = new_labels
                    # Create the new graphs with the new labels.
                    new_graphs.append([Gs_ed[j], new_labels])
                yield new_graphs

        graphs = generate_graphs(WL_labels_inverse, nl)
        values = [self.X[i].transform(g) for (i, g) in enumerate(graphs)]
        K = np.sum(values, axis=0)

        self._is_transformed = True
        if self.normalize:
            X_diag, Y_diag = self.diagonal()
            old_settings = np.seterr(divide='ignore')
            K = np.nan_to_num(np.divide(K, np.sqrt(np.outer(Y_diag, X_diag))))
            np.seterr(**old_settings)

        return K

    def diagonal(self):
        """Calculate the kernel matrix diagonal for fitted data.

        Parameters
        ----------
        None.
        Returns
        -------
        X_diag : np.array
        Y_diag : np.array
           
        """
        # Check if fit had been called
        check_is_fitted(self, ['X'])
        try:
            check_is_fitted(self, ['_X_diag'])
            if self._is_transformed:
                Y_diag = self.X[0].diagonal()[1]
                for i in range(1, self._n_iter):
                    Y_diag += self.X[i].diagonal()[1]
        except NotFittedError:
            # Calculate diagonal of X
            if self._is_transformed:
                X_diag, Y_diag = self.X[0].diagonal()
                # X_diag is considered a mutable and should not affect the kernel matrix itself.
                X_diag.flags.writeable = True
                for i in range(1, self._n_iter):
                    x, y = self.X[i].diagonal()
                    X_diag += x
                    Y_diag += y
                self._X_diag = X_diag
            else:
                # case sub kernel is only fitted
                X_diag = self.X[0].diagonal()
                # X_diag is considered a mutable and should not affect the kernel matrix itself.
                X_diag.flags.writeable = True
                for i in range(1, self._n_iter):
                    x = self.X[i].diagonal()
                    X_diag += x
                self._X_diag = X_diag

        if self._is_transformed:
            return self._X_diag, Y_diag
        else:
            return self._X_diag
