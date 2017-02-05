# -*- coding: utf-8 -*-

# Work-around for Atom Script encoding issue
# import sys
# import io
# #
# sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

import pandas as pd
import numpy as np
from pprint import pprint

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

import xgboost as xgb

data = pd.read_csv("data/preprocessed_data.csv")



# Split to training and test set
mask = ~data.columns.isin(['orientation', 'id'])
Xtrain, Xtest, ytrain, ytest = train_test_split(data.loc[:, mask], data.orientation, random_state = 10)

# --- Classification ----
RFC = True
XGB = True

# ---- Random Forest ----
# -----------------------
if (RFC):
    rfc = RandomForestClassifier(n_estimators = 200)
    # Perform a test
    rfc.fit(Xtrain, ytrain)
    ypred = rfc.predict(Xtest)
    print('RF Accuracy is {}\n'.format(np.mean(ypred == ytest)))
    feat_imp = dict()
    for i, j in zip(Xtrain.columns, rfc.feature_importances_*100):
        feat_imp[i]=j
    feat_imp = sorted(feat_imp.items(), key = lambda x : x[1], reverse = True)[:10]
    print('Showing 10 biggest feature importances:')
    for (i,j) in feat_imp:
      print('{:>25} | {:3.2f}'.format(i,j))
    # Get cross validated Accuracy
    scores = cross_val_score(rfc, data.loc[:, mask], data.orientation, cv=5)
    print("Cross validated accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

# ------ XGBOOST ------
#----------------------
if (XGB):
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


print('Confusion matrix :')
lab = {2.: 'Accompagnement', 3.: 'Mediation', 4.: 'Surendettement'}
ytest_lab = ytest.map(lab)
ytest_lab.index = range(1,len(ytest_lab) + 1)
ytest_lab.name = 'Actual'
ypred_lab = pd.Series(ypred, name='Predicted').map(lab)
print(pd.crosstab(ytest_lab, ypred_lab))
