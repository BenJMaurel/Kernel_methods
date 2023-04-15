from WL import WeisfeilerLehman
from utils import data_training, data_test, results_to_csv
from sklearn.metrics import make_scorer, f1_score, auc, accuracy_score, roc_curve
import svm
import numpy as np
from ShortestPath import ShortestPath

G_train, y_train = data_training()
G_test, to_delete = data_test()

#gk = ShortestPath(normalize=True)
gk = WeisfeilerLehman(n_iter = 5)
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)



C_Weisfeiler = 20
svm_ = svm.SVM(kernel = "precomputed", C = C_Weisfeiler)
svm_.fit(K_train, y_train, class_weight = "balanced")

preds = svm_.decision_function(K_test)

results_to_csv(preds, to_delete)
