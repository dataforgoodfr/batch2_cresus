# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

from pca import reduce_dim_pca


# ------ Parameters ------

reduce_dim = False
# Classification
RFC = True
XGB = ~RFC
ac = True  # Prédire uniquement sur A & C (B devient A)


# ---- Data preparation ----

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
mask = ~data.columns.isin(['orientation', 'id'])
X = data.loc[:, mask]
if reduce_dim:
    X = reduce_dim_pca(X, 67)
y = data.orientation
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=10)

#     ------ Classification ------
#     ----------------------------

# Random Forest model
rfc = RandomForestClassifier(n_estimators=200)
feat_imp = dict()

# Xgboost
param = {'max_depth': 6, 'eta': 0.1, 'silent': 1,
         'objective': 'multi:softmax', 'num_class': 3}
num_round = 20

# Stratified K folds
n_splits = 5
skf = StratifiedKFold(n_splits, shuffle=True)
accuracies, index = np.zeros(n_splits), 0
conf_mat = []

for train_index, test_index in skf.split(X, y):

    Xtrain, Xtest = X.ix[train_index], X.ix[test_index]
    ytrain, ytest = y[train_index], y[test_index]

    if RFC:
        rfc.fit(Xtrain, ytrain)
        ypred = rfc.predict(Xtest)
        for i, j in zip(Xtrain.columns, rfc.feature_importances_ * 100):
            feat_imp[i] = feat_imp.get(i, 0) + j

    if XGB:
        label_train = ytrain.map(lambda x: int(x - 2))
        label_test = ytest.map(lambda x: int(x - 2))
        dtrain = xgb.DMatrix(Xtrain, label=label_train)
        dtest = xgb.DMatrix(Xtest, label=label_test)
        bst = xgb.train(param, dtrain, num_round)
        label_pred = bst.predict(dtest)
        ypred = label_pred + 2

    accuracies[index] = np.mean(ypred == ytest)

    ytest_lab = ytest.map(lab)
    ytest_lab.index = range(1, len(ytest_lab) + 1)
    ytest_lab.name = 'Actual'
    ypred_lab = pd.Series(ypred, name='Predicted').map(lab)
    try:
        conf_mat = conf_mat + pd.crosstab(ytest_lab, ypred_lab)
    except:
        conf_mat = pd.crosstab(ytest_lab, ypred_lab)
    index += 1

for feat, imp in feat_imp.items():
    feat_imp[feat] = imp / n_splits
feat_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nCross validated accuracy: %0.2f (+/- %0.2f)\n" %
      (accuracies.mean(), accuracies.std()))
if RFC:
    print('Showing 10 biggest feature importances:')
    for (i, j) in feat_imp:
        print('{:>25} | {:3.2f}'.format(i, j / n_splits))

print('\nCross validated Confusion matrix :')
print(conf_mat)