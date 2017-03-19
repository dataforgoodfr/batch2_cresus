# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

import xgboost as xgb

from pca import reduce_dim_pca

import matplotlib.pyplot as plt
import pylab
pylab.ion()




# ------ Parameters ------

reduce_dim = False
# Classification
RFC = False
XGB = ~RFC
ac = True  # Prédire uniquement sur A & C (B devient A)


#    ---------- Data preparation -------------

data = pd.read_csv("../data/preprocessed_data.csv")
# Useful variables
lab = {2.: 'Accompagnement',
       3.: 'Mediation',
       4.: 'Surendettement'}

# Si on souhaite prédire sur A & C uniquement
if ac:
    data['orientation'] = data.where(
            ~(data.orientation == 3), 2)['orientation']

# Split to training and test set
mask = ~data.columns.isin(['orientation', 'id', 'id_user'])
X = data.loc[:, mask]
if reduce_dim:
    X = reduce_dim_pca(X, 67)
y = ((data.orientation-2)/2).map(int)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=10)


#     ------------ Classification -------------
#     -----------------------------------------

if RFC:
    n_estimators = [10, 20, 100, 200, 500]
    max_depths = [3, 5, 7, 10, 15]
    iterator = [(i, j) for i in n_estimators for j in max_depths]
if XGB:
    # Première optimisation
    etas = [0.01, 0.03, 0.1, 0.2]
    max_depths = [3, 5, 7, 10]
    min_child_weights = [1, 2, 5]
    subsamples = [0.8, 0.9]
    # Rafinement 1
    etas = [0.15, 0.2, 0.25]
    max_depths = [4, 5, 6]
    min_child_weights = [2]
    # Rafinement 2
    max_depths = [4]

    iterator = [(i, j, k, l) for i in etas
                             for j in max_depths
                             for k in min_child_weights
                             for l in subsamples]
best_auc, best_acc = 0, 0
for params in iterator:
    # Random Forest model
    if RFC:
        (n_estimator, max_depth) = params
        rfc = RandomForestClassifier(n_estimators=n_estimator,
                                     max_depth=max_depth)
        feat_imp = dict()
        print('\nRF - n_est:{}, max_depth:{}'.format(n_estimator, max_depth))

    # Xgboost
    if XGB:
        (eta, max_depth, min_child_weight, subsample) = params
        param = {'max_depth': max_depth, 'eta': eta, 'silent': 1,
                 'objective': 'binary:logistic',  # 'num_class': 2,
                 'subsample': subsample, 'min_child_weight': min_child_weight,
                 'n_estimators': 5000}
        num_round = 20
        print('\nXGB - eta:{}, max_depth:{}, numround: {}'.format(eta, max_depth, num_round))


    # Stratified K folds
    n_splits = 5
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=1)
    accuracies, aucs, index = np.zeros(n_splits), np.zeros(n_splits), 0
    conf_mat = []
    ypred_tot = np.zeros(data.shape[0])

    for train_index, test_index in skf.split(X, y):

        Xtrain, Xtest = X.ix[train_index], X.ix[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        if RFC:
            rfc.fit(Xtrain, ytrain)
            ypred = rfc.predict(Xtest)
            ytrainpred = rfc.predict(Xtrain)
            for i, j in zip(Xtrain.columns, rfc.feature_importances_ * 100):
                feat_imp[i] = feat_imp.get(i, 0) + j
            ypred_tot[test_index] = ypred

        if XGB:
            dtrain = xgb.DMatrix(Xtrain, label=ytrain)
            dtest = xgb.DMatrix(Xtest, label=ytest)
            bst = xgb.train(param, dtrain, num_round)
            label_pred = bst.predict(dtest)
            label_trainpred = bst.predict(dtrain)
            ypred = label_pred.round().astype(int)
            ytrainpred = label_trainpred.round().astype(int)
            ypred_tot[test_index] = label_pred

        # --- Accuracy du split---
        accuracies[index] = np.mean(ypred == ytest)
        print('Split n°{} - train: {:0.2f}, test: {:0.2f}'.format(
                index+1, np.mean(ytrainpred == ytrain), accuracies[index]))
        index += 1

    # ------ Métriques cross validées ------
    auc = roc_auc_score(y, ypred_tot)
    accuracy = np.mean(ypred_tot.round() == y)

    # ------- Update des meilleurs paramètres -----
    select_on_acc = False
    if accuracy > best_acc:
        best_acc = accuracy
        if select_on_acc:
            best_param = params
            best_pred = ypred_tot
        print('Mean accuracy improved: {:0.3f}'.format(accuracy))
    if auc > best_auc:
        best_auc = auc
        if ~select_on_acc:
            best_param = params
            best_pred = ypred_tot
        print('Mean AUC improved: {:0.3f}'.format(auc))

print("\nBest cross validated accuracy: %0.4f" % best_acc)
print("\nBest cross validated AUC: %0.4f\n" % best_auc)

# -------- Matrice de confusion du meilleur modèle -------
ylab = y.map(lambda x: lab[x*2+2])
ylab.index = range(0, len(ylab))
ylab.name = 'Real'
ypred_lab = pd.Series(ypred_tot.round(), name='Predicted').map(lambda x: lab[x*2+2])
conf_mat = pd.crosstab(ylab, ypred_lab)
print('\nCross validated Confusion matrix :')
print(conf_mat)

# ------ ROC Curve -----------
fpr, tpr, thresholds = roc_curve(y, ypred_tot)
plt.plot(fpr, tpr, '-', color = 'orange', label = 'AUC = %.3f' %best_auc)
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve: Logistic regression', fontsize=16)
plt.legend(loc="lower right")

if RFC:
    for feat, imp in feat_imp.items():
        feat_imp[feat] = imp / n_splits
    feat_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
    print('Showing 10 biggest feature importances:')
    for (i, j) in feat_imp:
        print('{:>25} | {:3.2f}'.format(i, j / n_splits))


