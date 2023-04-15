from WL import WeisfeilerLehman
from utils import data_training, data_test, grid_search
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score, auc, accuracy_score, roc_curve
from sklearn.svm import SVC
import svm
import numpy as np
from ShortestPath import ShortestPath

G_train, y_train = data_training()

G_train, G_test, y_train, y_test = train_test_split(G_train, y_train, test_size=0.2)

gk = WeisfeilerLehman()
#gk = ShortestPath(normalize=True)
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)



K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)
print("done")


best_svm = grid_search(K_train, y_train, K_test, y_test)
best_svm.fit(K_train, y_train, class_weight = "balanced")
print(best_svm)
preds = best_svm.decision_function(K_test)

##Comparaison with svm from scikit SVC
svm2 = SVC(kernel='precomputed', class_weight='balanced',C =5)
svm2.fit(K_train, y_train)
y_pred_prob = svm2.decision_function(K_test)


fpr, tpr, thresholds = roc_curve(y_test, 1/(1 + np.exp(-y_pred_prob)))
roc_auc = auc(fpr, tpr)
print("auc score:", str(round(roc_auc*100, 2)) + "%")
fpr, tpr, thresholds = roc_curve(y_test, 1/(1 + np.exp(-preds)))
roc_auc = auc(fpr, tpr)
print("auc score:", str(round(roc_auc*100, 2)) + "%")
