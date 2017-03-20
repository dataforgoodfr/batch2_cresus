# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from pca import reduce_dim_pca

import matplotlib.pyplot as plt
import pylab
pylab.ion()




# ------ Parameters ------

reduce_dim = False
# Classification

RFC, XGB, SVM, KNN = False, False, False, False
algo = input('Quel algo ? 1:RFC, 2:XGB, 3:SVM, 4:KNN\n')
if algo == '1':
    RFC = True
    print('RFC\n')
elif algo == '2':
    XGB = True
    print('XGB\n')  
elif algo == '3':
    SVM = True
    print('SVM\n')
elif algo == '4':
    KNN = True
    print('KNN\n')

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

# --------- Iterator parameters -------------
if RFC:
    n_estimators = [50, 70, 100, 120, 150]
    max_depths = [3, 5, 7, 10, 15]
    iterator = [(i, j) for i in n_estimators for j in max_depths]
    iterator = [(120, 10)]
if XGB:
    num_rounds = range(10,60, 2)
    # Première optimisation
    etas = [0.01, 0.03, 0.1, 0.2]
    max_depths = [3, 5, 7, 10]
    min_child_weights = [1, 2, 5]
    # Raffinement
    etas = [0.2]
    max_depths = [4, 5, 6]
    min_child_weights = [2, 3, 4]
    # Robustesse
    # max_depths = [4]
    max_depths = [4]
    min_child_weights = [4]
    subsamples = [0.8, 0.9, 1.]
    colsample_bytrees = [0.8, 0.9, 1.]
    gammas = [0., 0.001, 0.005, 0.01]

    iterator = [(num_round, eta, max_depth, min_child_weight,
                subsample, colsample_bytree, gamma)
                for num_round in num_rounds
                for eta in etas
                for max_depth in max_depths
                for min_child_weight in min_child_weights
                for subsample in subsamples
                for colsample_bytree in colsample_bytrees
                for gamma in gammas]
    iterator = [(numround, 0.2, 4, 4, 0.9, 0.8, 0.3)
                for numround in num_rounds]
    iterator = [(58, 0.2, 4, 4, 0.9, 0.8, 0.3)]
if SVM:
    Cs = [.5, 1, 2]
    gammas = [.005, .01, .02, .05]
    iterator = [(c, gamma) for c in Cs for gamma in gammas]
if KNN:
    n_neighbors_ = [50, 75, 100, 125, 150, 200]
    weights_ = ['uniform', 'distance']
    p_ = [1, 2, 3]
    leaf_size_ = [15, 30, 50]
    iterator = [(n_neighbors, weights, p, leaf_size)
                for n_neighbors in n_neighbors_
                for weights in weights_
                for p in p_
                for leaf_size in leaf_size_]
    iterator = [(125, 'distance', 1, 15)]
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
        (num_round, eta, max_depth, min_child_weight, subsample, colsample_bytrees, gamma) = params
        param = {'max_depth': max_depth, 'eta': eta, 'silent': 1,
                 'objective': 'binary:logistic',  # 'num_class': 2,
                 'subsample': subsample, 'min_child_weight': min_child_weight,
                 'colsample_bytrees': colsample_bytrees, 'gamma': gamma,
                 'n_estimators': 5000}
        print('\nXGB - eta:{}, max_depth:{}, numround: {}'.format(eta, max_depth, num_round))

    if SVM:
        (c, gamma) = params
        svc = SVC(C=c, gamma=gamma)
        print('\nSVM - C:{}, gamma:{}'.format(c, gamma))

    if KNN:
        (n_neighbors, weights, p, leaf_size) = params
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights,
                                   p=p, leaf_size=leaf_size)
        print('\nKNN - nn:{}, weights:{}, p: {}, leaf_size:{}'.format(n_neighbors, weights, p, leaf_size))

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
            yprob = rfc.predict_proba(Xtest)[:, 1]
            ytrainprob = rfc.predict_proba(Xtrain)[:, 1]
            for i, j in zip(Xtrain.columns, rfc.feature_importances_ * 100):
                feat_imp[i] = feat_imp.get(i, 0) + j

        if XGB:
            dtrain = xgb.DMatrix(Xtrain, label=ytrain)
            dtest = xgb.DMatrix(Xtest, label=ytest)
            bst = xgb.train(param, dtrain, num_round)
            yprob = bst.predict(dtest)
            ytrainprob = bst.predict(dtrain)

        if SVM:
            svc.fit(Xtrain, ytrain)
            yprob = svc.decision_function(Xtest)
            ytrainprob = svc.decision_function(Xtrain)

        if KNN:
            s = StandardScaler()
            Xtrains = s.fit_transform(Xtrain)
            Xtests = s.transform(Xtest)
            knn.fit(Xtrains, ytrain)
            yprob = knn.predict_proba(Xtests)[:, 1]
            ytrainprob = knn.predict_proba(Xtrains)[:, 1]

        ypred = yprob.round().astype(int)
        ytrainpred = ytrainprob.round().astype(int)
        ypred_tot[test_index] = yprob

        # --- Accuracy du split---
        accuracies[index] = np.mean(ypred == ytest)
        print('Split n°{} - train: {:0.2f}, test: {:0.2f}'.format(
                index+1, np.mean(ytrainpred == ytrain), accuracies[index]))
        index += 1

    # ------ Métriques cross validées ------
    auc = roc_auc_score(y, ypred_tot)
    accuracy = np.mean(ypred_tot.round() == y)

    # ------- Update des meilleurs paramètres -----
    select_on_acc = True
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
color = 'blue' if RFC else 'green' if KNN else 'yellow'
label = 'RFC' if RFC else 'KNN' if KNN else 'XGB'
plt.plot(fpr, tpr, '-', color=color, label='AUC %s = %.3f' % (label, best_auc))d
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve', fontsize=16)
plt.legend(loc="lower right")

if RFC:
    for feat, imp in feat_imp.items():
        feat_imp[feat] = imp / n_splits
    feat_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
    print('Showing 10 biggest feature importances:')
    for (i, j) in feat_imp:
        print('{:>25} | {:3.2f}'.format(i, j / n_splits))


