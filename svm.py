import numpy as np
from functools import partial
import cvxopt
from sklearn.metrics import make_scorer, f1_score, auc, accuracy_score, roc_curve
from abc import ABC, abstractmethod
import time
from sklearn.utils.class_weight import compute_class_weight
from scipy import optimize

class SVM():
    """
    SVM implementation
    
    Usage:
        svm = SVM(kernel='precomputed', C=1)
        svm.fit(X_train, y_train)
        svm.predict(X_test)
    """

    def __init__(self, kernel, C=1.0, gamma=1e-4):
        """
        kernel: Which kernel to use
        C: float > 0, default=1.0, regularization parameter
        tol_support_vectors: Threshold for alpha value to consider vectors as support vectors
        """
        self.kernel = kernel
        self.gamma = gamma
        self.C = C

    def separating_function(self, K):
        # Input : matrix K of shape N data points times d dimension
        # Output: vector of size N
        return K @ self.beta_sup

    def fit(self, K, y, class_weight=None):
        self.K = K
            
        n_samples = K.shape[0]

        # Define the optimization problem to solve
        dim = K.shape[0]
        assert dim == len(y)
        y_values_sorted = np.sort(np.unique(y))
        if y_values_sorted[0] == 0 and y_values_sorted[1] == 1:
            y = 2 * (y - 0.5)
        self.y = y
        diag_y = np.diag(y)
        idt = np.identity(dim)
        cvxopt.solvers.options['show_progress'] = False

        if class_weight == "balanced":
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            
            class_weight = class_weights[np.searchsorted(np.unique(y), y)]
        else:
            class_weight = np.asarray(class_weight)

        P = self.C /2 * np.dot(diag_y, np.dot(K, diag_y))
        q = - np.ones(dim)
        G = np.concatenate((idt, -idt))
        h = np.concatenate((class_weight / dim, self.C-class_weight / dim))

        A = cvxopt.matrix(y, (1, dim), "d")
        b = cvxopt.matrix(0.0)
        
        ## Solving the quadratic problem
        res = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), cvxopt.matrix(G), cvxopt.matrix(h), A=A, b=b)['x']
        self.alphas = np.dot(diag_y, res)*self.C /2

        alpha = y * np.array(res).reshape(-1)
        tol = 1e-8
        support_vectors = np.where(np.abs(alpha) > tol)[0]
        intercept = 0
        for sv in support_vectors:
            intercept += y[sv]
            intercept -= np.sum(
                alpha[support_vectors] * K[sv, support_vectors])
        if len(support_vectors) > 0:
            intercept /= len(support_vectors)
        self.support_vectors = support_vectors

        # Set to zero non-support vectors
        alpha[np.where(np.abs(alpha) <= tol)[0]] = 0

        self.b = intercept
        self.alphas = alpha
        print(self.b)

    def decision_function(self, K):
        """
        K: array (n_samples, n_features)\\
        Return: float array (n_samples,)
        """
        K = K[:,self.support_vectors]
        preds = np.dot(K, (self.alphas).T[self.support_vectors]) + self.b
        return preds
        

    def predict_classes(self, K, threshold=0):
        """
        X: array (n_samples, n_features)\\
        Return: 0 and 1 array (n_samples,)
        """
        y = np.dot(K, self.alphas)
        return np.where(y > threshold, 1, 0)