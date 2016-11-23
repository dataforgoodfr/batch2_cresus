# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

def transform_data(data):
    data = data[to_keep]
    data.loc[:, credit_flat] = data[credit_flat].fillna(0)
    for i, col in enumerate(new_cols):
        data.loc[:, col] = np.sum(data[credit_detail[i]], axis=1)
    data = data[(data.sum_mensualite > 0) & (data.sum_mensualite < 8000)]
    data = data[data.orientation > 1]
    data=data[data.revenus.notnull()] # Elimine ceux pour lesquels on a pas de budget
    le = LabelEncoder()
    for col, dtype in zip(data.columns, data.dtypes):
        if dtype == 'object':
            data[col] = data[col].apply(lambda s: str(s))
            data[col] = le.fit_transform(data[col])
    return data

# Masks:

new_cols = ('sum_mensualite', 'moy_nb_mensualite', 'sum_solde')
credit_detail = []
for text in new_cols:
    credit_detail.append(["{}_{}".format(text, i) for i in range(6)])
credit_flat = [item for sublist in credit_detail for item in sublist]

budget = ['typ', 'revenus', 'allocations',
          'pensions_alim', 'revenus_FL', 'autre1', 'autre2', 'autre3', 'loyer',
          'charges_loc_cop', 'gdf', 'electicite', 'eau', 'tel_fixe', 'tel_port',
          'impots', 'taxe_fonciere', 'taxe_habitation', 'assurance_habitat',
          'assurance_voiture', 'mutuelle', 'autre_assurance', 'epargne_enfant',
          'frais_scolarite', 'transport_enfant', 'autres_charges_enfant',
          'frais_bancaire', 'soins_recurrent', 'frais_justice', 'frais_transport',
          'epargne', 'autres_charges', 'fioul_bois', 'internet', 'abonnement_tv',
          'abonnement_autre', 'autre_charge', 'taxe_ordure', 'autre_impots',
          'assurance_gav', 'assurance_prevoyance', 'assurance_scolaire',
          'pensions_alim_payee', 'internat', 'frais_garde', 'cantine',
          'alim_hyg_hab', 
          #'dat_budget'
          ]

autres_infos = ['id', 'profession', 'logement', 'situation', 'transferable',
                'retard_facture', 'retard_pret',
                 #'nature', 
                 'orientation',
                'personne_charges', 'releve_bancaire']

to_keep = autres_infos + credit_flat + budget



data = pd.read_csv('out.csv', sep='\t')
data = transform_data(data)
# This needs to be improved, it's just a MVP to showcase that we can already start applying algorithms.
data.fillna(0, inplace=True) #TODO: Find which are actually NA.

mask = ~data.columns.isin(['orientation', 'id'])
Xtrain, Xtest, ytrain, ytest = train_test_split(data.loc[:, mask], data.orientation)
rfc = RandomForestClassifier(random_state = 0)
rfc.fit(Xtrain, ytrain)
ypred = rfc.predict(Xtest)
print('Number of observations kept: {}'.format(len(data)))
print('Accuracy is {}'.format(np.mean(ypred == ytest)))

feat_imp = dict()
for i, j in zip(Xtrain.columns, rfc.feature_importances_*100):
    feat_imp[i]=j
feat_imp = sorted(feat_imp.items(), key = lambda x : x[1], reverse = True)
print('Showing feature importances:')
pprint(feat_imp)

