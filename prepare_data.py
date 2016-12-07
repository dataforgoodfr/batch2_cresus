# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

# --- Data import ---

# Import sql extract 'out.csv' as left for future join with age data
left = pd.read_csv('out.csv', sep='\t')

#Import birth and file opening years, and computing difference to get age
naissance = pd.read_csv('annee_naissance.csv',sep=';')
naissance.columns = ['id', 'annee_naissance']
ouverture = pd.read_csv('annee_ouverture.csv',sep=';')
right = pd.merge(ouverture, naissance, on='id')
right['age'] = right['annee_ouverture']-right['annee_naissance']

# left joining age to the extract
data = pd.merge(left, right.loc[:,['id', 'age']], on='id')

# Masks:

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
          'alim_hyg_hab'
          #,'dat_budget'
          ]

autres_infos = ['id','age', 'profession', 'logement', 'situation', #'transferable',
                'retard_facture', 'retard_pret',
                 #'nature', 
                 'orientation',
                'personne_charges', #'releve_bancaire'
                ]

new_cols = ('sum_mensualite',
            #'moy_nb_mensualite', 
            'sum_solde')
credit_detail = []
for text in new_cols:
    credit_detail.append(["{}_{}".format(text, i) for i in range(6)])
credit_flat = [item for sublist in credit_detail for item in sublist] 
credit_flat += list(new_cols)
to_keep = autres_infos + credit_flat + budget


# ----- Local functions -----
# ---------------------------

def age_control(data):
  #Supprime les ages inférieurs à 18 ans et supérieurs à 90
  data = data[(data.age>=18) & (data.age<=90)]
  return data

def recup_orientation_old(data):
  d = {'surendettement' : 4 ,
        'accompagnement' : 2,
        'mediation' : 3,
        'microcredit' : 5,
        'Microcredit' : 5}
  data.orientation_old = data.orientation_old.apply(lambda x : d.get(x,0))

  data.ix[(data.orientation_old > 1) 
        & (data.orientation<=1), 'orientation'] =  data.ix[(data.orientation_old > 1) 
                                                          & (data.orientation<=1), 'orientation_old']
  return data

def aggreg_features(data):
  # Aggrège les différents types de crédits
  for i, col in enumerate(new_cols):
      data.loc[:, col] = np.sum(data[credit_detail[i]], axis=1)
  # Aggrège les différentes catégories du budget
  data.revenus_tot = data.loc[:, ('revenus',
                                      'allocations',
                                      'pensions_alim',
                                      'revenus_FL',
                                      'autre1',
                                      'autre2',
                                      'autre3'
                                      )].apply(sum,1)
  # Aggrège les différents types de charges
  charges = ['loyer','charges_loc_cop','gdf','electicite','eau','tel_fixe','tel_port','impots','taxe_fonciere',
    'taxe_habitation','assurance_habitat','assurance_voiture','mutuelle','autre_assurance',
    'epargne_enfant','frais_scolarite','transport_enfant','autres_charges_enfant','frais_bancaire',
    'soins_recurrent','frais_justice','frais_transport','epargne','autres_charges','fioul_bois',
   'internet','abonnement_tv','abonnement_autre','autre_charge','taxe_ordure','autre_impots',
    'assurance_gav','assurance_prevoyance','assurance_scolaire','pensions_alim_payee',
    'internat','frais_garde','cantine']
  data.charges = data.loc[:, charges].apply(sum,1)
  return data

def transform_data(data):
    ''' Filter columns'''
    data = data.loc[:,to_keep]
    data = data[(data.sum_mensualite > 0) & (data.sum_mensualite < 8000)]
    data = data[(data.orientation > 1)&(data.orientation < 5)]
    data.personne_charges = data.personne_charges.apply(abs)
    data=data[data.revenus.notnull()] # Elimine ceux pour lesquels on a pas de budget
    le = LabelEncoder()
    for col, dtype in zip(data.columns, data.dtypes):
        if dtype == 'object':
            data[col] = data[col].apply(lambda s: str(s))
            data[col] = le.fit_transform(data[col])
    return data

def fill_na(data):
  ''' Fill NA with median of the column
    To be improved
  '''
  # Columns where NAs are filled with zeros
  fill_with_zeros = credit_flat
  for col in fill_with_zeros:
    data[col].fillna(value = 0,
                    inplace = True)  
  # Columns where NAs are filled by the median
  for col in data:
    data[col].fillna(value = data[col].median(),
                    inplace = True)


## ----- Main ------
# ------------------

data = age_control(data)
data = aggreg_features(data)
data = recup_orientation_old(data)
data = transform_data(data)
fill_na(data)
# This needs to be improved, it's just a MVP to showcase that we can already start applying algorithms.

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

