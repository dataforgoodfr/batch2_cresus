# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pprint import pprint

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

data = pd.read_csv("data/preprocessed_data.csv")

# Useful variables
lab = {2.: 'Accompagnement', 3.: 'Mediation', 4.: 'Surendettement'}


# Split to training and test set
mask = ~data.columns.isin(['orientation', 'id'])
X = data.loc[:, mask]
y = data.orientation
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state = 10)

# --- Classification ----
RFC = False
XGB = True

# ---- Random Forest ----
# -----------------------


# Random Forest model
rfc = RandomForestClassifier(n_estimators = 200)
feat_imp = dict()

# Xgboost
param = {'max_depth':6, 'eta':0.1, 'silent':1, 'objective':'multi:softmax' , 'num_class':3}
num_round = 20

# Stratified K folds
n_splits = 5
skf = StratifiedKFold(n_splits, shuffle = True)
accuracies, index  = np.zeros(n_splits), 0
conf_mat = []

for train_index, test_index in skf.split(X, y):

    Xtrain, Xtest = X.ix[train_index], X.ix[test_index]
    ytrain, ytest = y[train_index], y[test_index]

    if RFC:
        rfc.fit(Xtrain, ytrain)
        ypred = rfc.predict(Xtest)
        for i, j in zip(Xtrain.columns, rfc.feature_importances_*100):
            feat_imp[i] = feat_imp.get(i, 0) + j

    if XGB:
        label_train = ytrain.map(lambda x : int(x-2))
        label_test = ytest.map(lambda x : int(x-2))
        dtrain = xgb.DMatrix(Xtrain, label = label_train)
        dtest = xgb.DMatrix(Xtest, label = label_test)
        bst = xgb.train(param, dtrain, num_round)
        label_pred = bst.predict(dtest)
        ypred = label_pred+2

    accuracies[index] = np.mean(ypred == ytest)

    ytest_lab = ytest.map(lab)
    ytest_lab.index = range(1,len(ytest_lab) + 1)
    ytest_lab.name = 'Actual'
    ypred_lab = pd.Series(ypred, name='Predicted').map(lab)
    try:
        conf_mat = conf_mat + pd.crosstab(ytest_lab, ypred_lab)
    except:
        conf_mat = pd.crosstab(ytest_lab, ypred_lab)
    index+=1

conf_mat /= n_splits
for feat, imp in feat_imp.items():
    feat_imp[feat] = imp/ n_splits
feat_imp = sorted(feat_imp.items(), key = lambda x : x[1], reverse = True)[:10]
print("\nCross validated accuracy: %0.2f (+/- %0.2f)\n" %
      (accuracies.mean(), accuracies.std()))
if RFC :
    print('Showing 10 biggest feature importances:')
    for (i,j) in feat_imp:
        print('{:>25} | {:3.2f}'.format(i,j/n_splits))

print('\nCross validated Confusion matrix :')
print(conf_mat)

# ------ XGBOOST ------
#----------------------
if (XGB & False):
    # Encoding to 1 - n_classes
    label_train = ytrain.map(lambda x : int(x-2))
    label_test = ytest.map(lambda x : int(x-2))
    dtrain = xgb.DMatrix(Xtrain, label = label_train)
    dtest = xgb.DMatrix(Xtest, label = label_test)

    # specify parameters via map
    param = {'max_depth':6, 'eta':0.1, 'silent':1, 'objective':'multi:softmax' , 'num_class':3}
    num_round = 20
    bst = xgb.train(param, dtrain, num_round)
    # make prediction
    label_pred = bst.predict(dtest)
    ypred = label_pred+2
    print('XGboost Accuracy is {}\n'.format(np.mean(ypred == ytest)))


    # Get cross validated accuracy
    scores = xgb.cv(param, dtrain, num_round, nfold=5, metrics={'merror'}, seed = 0)
    print("XGboost Cross validated accuracy: %0.2f (+/- %0.2f)" %
          (1-scores['test-merror-mean'][num_round-1],
           scores['test-merror-std'][num_round-1]))
