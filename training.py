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

data = prepare_data.prepare_all()
print('Number of observations kept: {}'.format(len(data)))

# Split to training and test set
mask = ~data.columns.isin(['orientation', 'id'])
Xtrain, Xtest, ytrain, ytest = train_test_split(data.loc[:, mask], data.orientation, random_state = 10)

# ---- Random Forest ----
# -----------------------
if (False):
	rfc = RandomForestClassifier(random_state = 10, n_estimators = 100)
	rfc.fit(Xtrain, ytrain)
	ypred = rfc.predict(Xtest)


	feat_imp = dict()
	for i, j in zip(Xtrain.columns, rfc.feature_importances_*100):
	    feat_imp[i]=j
	feat_imp = sorted(feat_imp.items(), key = lambda x : x[1], reverse = True)
	print('Showing feature importances:')
	for (i,j) in feat_imp:
	  print('{:>25} | {:3.2f}'.format(i,j))

# ------ XGBOOST ------
#----------------------
if (True):
	# Encoding to 1 - n_classes

	label_train = ytrain.map(lambda x : int(x-2))
	label_test = ytest.map(lambda x : int(x-2))
	dtrain = xgb.DMatrix(Xtrain, label = label_train)
	dtest = xgb.DMatrix(Xtest, label = label_test)

	# specify parameters via map
	param = {'max_depth':6, 'eta':0.1, 'silent':1, 'objective':'multi:softmax' , 'num_class':3}
	num_round = 2
	bst = xgb.train(param, dtrain, num_round)
	# make prediction
	label_pred = bst.predict(dtest)
	ypred = label_pred+2

	print('XGBoost accuracy is {}'.format(np.mean(ypred == ytest)))

print('Accuracy is {}'.format(np.mean(ypred == ytest)))