# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pprint import pprint

from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

import xgboost as xgb

import prepare_data

print(dir(prepare_data))

data = prepare_data._main()

# Filtering columns & obs
# This needs to be improved, it's just a MVP to showcase that we can already start applying algorithms.

data = data.loc[:,to_keep]
mask = ~data.columns.isin(['orientation', 'id'])
Xtrain, Xtest, ytrain, ytest = train_test_split(data.loc[:, mask], data.orientation, random_state = 10)
rfc = RandomForestClassifier(random_state = 10, n_estimators = 100)
rfc.fit(Xtrain, ytrain)
ypred = rfc.predict(Xtest)
print('Number of observations kept: {}'.format(len(data)))
print('Accuracy is {}'.format(np.mean(ypred == ytest)))

feat_imp = dict()
for i, j in zip(Xtrain.columns, rfc.feature_importances_*100):
    feat_imp[i]=j
feat_imp = sorted(feat_imp.items(), key = lambda x : x[1], reverse = True)
print('Showing feature importances:')
for (i,j) in feat_imp:
  print('{:>25} | {:3.2f}'.format(i,j))

# ------ XGBOOST ------
#----------------------

model = xgboost.XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(y_test)
print('XGBoost accuracy is {}'.format(np.mean(ypred == ytest)))