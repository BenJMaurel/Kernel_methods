import networkx as nx
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score, auc, accuracy_score, roc_curve
from sklearn.metrics import make_scorer, f1_score
data_path = "data/"
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import lru_cache
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv, norm
import svm

def data_training():
    # Load the list of graphs from the .pkl file
    with open(data_path + 'training_data.pkl', 'rb') as f:
        graphs = pickle.load(f)
    with open(data_path + 'training_labels.pkl', 'rb') as f:
        labels_graph = pickle.load(f)
    G_train = []
    to_delete = []
    for i,graph in enumerate(graphs):
        if len(graph.edges())>0:
            G_train.append(graph)
        else : 
            to_delete.append(i)
    return G_train, np.delete(labels_graph, to_delete)

def data_test():
    with open(data_path + 'test_data.pkl', 'rb') as f:
        graphs = pickle.load(f)
    G_test = []
    to_delete = []
    for i,graph in enumerate(graphs):
        if len(graph.edges())> 0 :
                G_test.append(graph)
        else : 
            to_delete.append(i)
    return G_test, to_delete

def data_training_2(graph = False):
    # Load the list of graphs from the .pkl file
    with open(data_path + 'training_data.pkl', 'rb') as f:
        graphs = pickle.load(f)
    with open(data_path + 'training_labels.pkl', 'rb') as f:
        labels_graph = pickle.load(f)
    graphs_G = []
    to_delete = []
    for i, G in enumerate(graphs):
        
        edges = set([(u, v) for u, v in G.edges()])
        labels_node = {node: G.nodes[node]['labels'][0] for node in G.nodes}
        labels_edge = {edge : G.edges[edge]['labels'][0] for edge in G.edges}
        if len(edges)> 0 :
            graphs_G.append([edges, labels_node, labels_edge])
        else :  
            to_delete.append(i)
    return graphs_G , np.delete(labels_graph, to_delete)

def data_test_2():
    with open(data_path + 'test_data.pkl', 'rb') as f:
        graphs = pickle.load(f)

    G_test = []
    to_delete = []
    for i, G in enumerate(graphs):
        edges = set([(u, v) for u, v in G.edges()])
        labels = {i: G.nodes[i]['labels'][0] for i in G.nodes}
        labels_edge = {edge : G.edges[edge]['labels'][0] for edge in G.edges}
        # graphs_G.append(Graph(edges, node_labels=labels))
        if len(edges)> 0 :
            G_test.append([edges, labels, labels_edge])
        else : 
            to_delete.append(i)
    return G_test, to_delete

def plot_graphs(graphs, y ,nrows = 3, ncols = 3):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6))
    
    for i, G in enumerate(graphs[:nrows*ncols]):
        # Calculate the row and column index of the subplot
        row = i // ncols
        col = i % ncols
        
        # Plot the graph in the appropriate subplot
        ax = axes[row][col]
        nx.draw(G, ax=ax, with_labels=True)
        ax.set_title(f"Graph {i+1}, {y[i]}")
        
        # You can customize the plot options here
        # For example, to change the node color:
        # nx.draw(G, ax=ax, with_labels=True, node_color='red')

    # Adjust the spacing between subplots and show the plot
    fig.tight_layout()
    plt.show()

def grid_search(K_train, y_train, K_test, y_test):
    # use grid search to find the best hyperparameters
    param_grid = [0.1, 1, 2, 5, 10, 20]
    aucs = []
    for C in tqdm(param_grid):
        test = svm.SVM(kernel = "precomputed", C = C)
        test.fit(K_train, y_train, class_weight = "balanced")
        preds = test.decision_function(K_test)
        fpr, tpr, thresholds = roc_curve(y_test, 1/(1 + np.exp(-preds)))
        aucs.append(auc(fpr, tpr))
    C_max = param_grid[np.argmax(aucs)]
    print(aucs)
    best_svm = svm.SVM(kernel = 'precomputed', C = C_max)

    return best_svm

def insert_zeros(binary_list, indices):
    new_list = []
    index_offset = 0
    for i in range(len(binary_list) + len(indices)):
        if i in indices:
            new_list.append(-4)
            index_offset +=1
        else:
            new_list.append(binary_list[i-index_offset])
    return new_list

def results_to_csv(y_pred, to_delete = []):
    if to_delete != []:
        y_pred = insert_zeros(y_pred, to_delete)
    df = {'Id' : np.arange(1, len(y_pred) +1), 'Predicted': y_pred}
    df = pd.DataFrame(df)
    df.to_csv('y_pred.csv', index = False)

def product_graph(G1,G2):
    return nx.cartesian_product(G1,G2)

def walk_kernel(n, G_1, G_2):
    # Compute the graph product of G_1 and G_2
    G_1G_2 = product_graph(G_1, G_2)
    # Take the adjacency matrix
    A = nx.to_numpy_matrix(G_1G_2)
    # Compute the matrix of power n
    A_n = np.linalg.matrix_power(A, n)
    # Return the frobenius norm:
    return np.sum(A)

def matrix_walk_kernel(Gs, n):
    kernel_matrix = np.zeros((len(Gs), len(Gs)))
    for i in tqdm(range(len(Gs))):
        for j in tqdm(range(i, len(Gs))):
            kernel_matrix[i,j] = walk_kernel(n, Gs[i], Gs[j])
            kernel_matrix[j,i] = kernel_matrix[i,j]
    return kernel_matrix

def nth_order_walk_kernel(adj_matrix, n):
    # Step 1: Compute the n-step transition matrix
    transition_matrix = np.linalg.matrix_power(adj_matrix, n)

    # Step 2: Compute the diagonal matrix D
    D = np.reciprocal(np.sum(transition_matrix, axis=1))

    # Step 3: Compute the symmetric matrix L
    L = np.diag(np.sqrt(D)) @ transition_matrix @ np.diag(np.sqrt(D))

    # Step 4: Compute the matrix exponential
    kernel = np.eye(adj_matrix.shape[0])
    for i in range(1, 50):
        kernel += np.linalg.matrix_power(L, i) / np.math.factorial(i)

    return kernel

def compute_nth_all(Gs):
    for i in range(len(Gs)):
        adj_matrix_i = nx.to_numpy_matrix(Gs[i])
        kernel_ij = nth_order_walk_kernel(adj_matrix_i, n)
        for j in range(i, len(Gs)):
            # Compute the adjacency matrix of the graph
            adj_matrix_j = nx.to_numpy_matrix(Gs[j])
            # Compute the kernel for the two graphs
            
            kernel_ji = nth_order_walk_kernel(adj_matrix_j, n)
            # Fill in the corresponding entries in the kernel matrix
            kernel_matrix[i, j] = np.trace(kernel_ij @ kernel_ji)
            kernel_matrix[j, i] = kernel_matrix[i, j]  # Symmetric matrix

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))


def train_test_split(X, Y, test_size=0.2, random_state=None, shuffle=False):
    """
    Split the data into training and test sets.
    :param X: inputs
    :param Y: labels
    :param test_size: size of the test set
    :param random_state: a random state for the permutation
    :param shuffle: if the data should be shuffled
    :return:
    """
    n_samples = X.shape[0]
    n_train = int(n_samples * (1. - test_size))

    # Set random seed if there is one
    if random_state is not None:
        np.random.seed(random_state)

    # Shuffle the data if shuffle is True
    if shuffle:
        permutation = np.random.permutation(n_samples)
    
    X = X[permutation]
    Y = Y[permutation]

    # Split the data
    X_train = X[:n_train]
    X_test = X[n_train:]
    Y_train = Y[:n_train]
    Y_test = Y[n_train:]

    return X_train, X_test, Y_train, Y_test
