# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pprint import pprint

from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

import xgboost as xgb

def import_data():
  """ 
    Import the dataset from SQL extract, joins additional data
  """

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
  return data

# ------- Local functions -------
# -------------------------------


# Masks:
def create_masks(data) : 
  budget = [#'typ', 
            'revenus', 'allocations',
            'pensions_alim', 'revenus_FL', 'autre1', 'autre2', 'autre3', 'loyer',
            'charges_loc_cop', 'gdf', 'electicite', 'eau', 'tel_fixe', 'tel_port',
            'impots', 'taxe_fonciere', 'taxe_habitation', 'assurance_habitat',
            'assurance_voiture', 'mutuelle', 'autre_assurance', 'epargne_enfant',
            'frais_scolarite', 'transport_enfant', 'autres_charges_enfant',
            'frais_bancaire', 'soins_recurrent', 'frais_justice', 'frais_transport',
            'epargne', 'autres_charges', 'fioul_bois', 'internet', 'abonnement_tv',
            'abonnement_autre', 'autre_charge', 'taxe_ordure', 'autre_impots',
            'assurance_gav', 'assurance_prevoyance', 'assurance_scolaire',
            'pensions_alim_payee', 'internat', 'frais_garde', 'cantine','alim_hyg_hab'
            #,'dat_budget'
            ]
  autres_infos = ['id','age', 'profession', 'logement', 'situation', #'transferable',
                  'retard_facture', 'retard_pret',
                   #'nature', 
                   'orientation',
                  'personne_charges'#, 'releve_bancaire'
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
  return [budget, autres_infos, new_cols, credit_detail, credit_flat, to_keep]

def encode_categ(data):
    # Encode categorical variables
    le = LabelEncoder()
    mapping = dict()
    for col, dtype in zip(data.columns, data.dtypes):
        if dtype == 'object':
            data[col] = data[col].apply(lambda s: str(s))
            # Replace 0 and NaNs with unique label : 'None'
            data[col] = data[col].where(~data[col].isin(['0','nan']), 'None')
            data[col] = le.fit_transform(data[col])
            mapping[col]= dict(zip(le.inverse_transform(data[col].unique()), data[col].unique()))
    return [data,mapping]

def clean_budget(data, mapping):
  '''Identifie les 0 qui devraient être des NAs, et les remplace par des plus proche voisins 
  '''
  locataire = mapping['logement']['locataire']
  n_neighbors = 5
  for col  in ['loyer', 'gdf', 'electicite', 'eau', 'assurance_habitat']:
    regle = (data.logement == locataire) & (data[col] == 0)
    data[col]   = data[col].where(~regle, None)
    if data[col].isnull().sum()>0:
        knn = neighbors.KNeighborsRegressor(n_neighbors)
        data_temp = data.loc[(data.logement== locataire) & (~data[col].isnull()), :]
        mask = ~data.columns.isin([col, 'id'])
        X = data_temp.loc[:, mask]
        null_index = data[col].isnull()
        y_ = knn.fit(X, data_temp[col]).predict(data.loc[null_index,mask])
        data.loc[null_index,col] = y_
  return data

def aggreg(data, to_keep, credit_detail, new_cols) : 
  # Aggrège les différents types de crédits
  for i, col in enumerate(new_cols):
      data.loc[:, col] = np.sum(data[credit_detail[i]], axis=1)

  # Aggrège les différentes catégories du budget
  data['revenus_tot'] = data.loc[:, ('revenus',
                                      'allocations',
                                      'pensions_alim',
                                      'revenus_FL',
                                      'autre1',
                                      'autre2',
                                      'autre3'
                                      )].sum(1)

  # Aggrège les différents types de charges
  which = ['loyer','charges_loc_cop','gdf','electicite','eau','tel_fixe','tel_port','impots','taxe_fonciere',
    'taxe_habitation','assurance_habitat','assurance_voiture','mutuelle','autre_assurance',
    'epargne_enfant','frais_scolarite','transport_enfant','autres_charges_enfant','frais_bancaire',
    'soins_recurrent','frais_justice','frais_transport','epargne','autres_charges','fioul_bois',
   'internet','abonnement_tv','abonnement_autre','autre_charge','taxe_ordure','autre_impots',
    'assurance_gav','assurance_prevoyance','assurance_scolaire','pensions_alim_payee',
    'internat','frais_garde','cantine']
  data['charges'] = data.loc[:, which].sum(1, skipna = True)

  to_keep = to_keep + ['charges', 'revenus_tot']
  return [data, to_keep]


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


def age_control(data):
  #Supprime les ages inférieurs à 18 ans et supérieurs à 90
  data = data[(data.age>=18) & (data.age<=90)]
  return data

def filter_data(data):
    ''' Filter observations'''
    data = data[(data.orientation > 1)&(data.orientation < 5)]
    data = data[(data.sum_mensualite > 0) & (data.sum_mensualite < 8000)]
    data = data.loc[data.charges>0,:]
    data.personne_charges = data.personne_charges.apply(abs)
    data=data[data.revenus.notnull()] # Elimine ceux pour lesquels on a pas de budget
    return data


def fill_na(data,mapping, credit_flat):
  ''' Fill NA with median of the column
    To be improved
  '''
  # Variables catégorielles
  for col in mapping:
    if 'None' in mapping[col]:
      data[col] = data[col].where(
                      data[col].isin([mapping[col]['None']]),
                      # Valeur de remplacement des None : -- à améliorer --
                      data[col].value_counts().idxmax()
                      )

  # Les None dans les colonnes du crédit correspondent à des 0 (absence de crédit)
  fill_with_zeros = credit_flat
  for col in fill_with_zeros:
    data[col].fillna(value = 0,
                    inplace = True)  

  # Columns where NAs are filled by the median -- à améliorer -- mais bon ça concerne quasi personne --
  for col in data:
    med = data[col].median()
    data[col].fillna(value = 0 if (not(med) or np.isnan(med))  else med,
                    inplace = True)
  return data


def prepare_all():
  ''' Ici on fait tout'''
  data = import_data()
  [budget, autres_infos, new_cols, credit_detail, credit_flat, to_keep] = create_masks(data)
  [data, mapping] = encode_categ(data)
  data = age_control(data)
  [data, to_keep] = aggreg(data, to_keep, credit_detail, new_cols)
  data = filter_data(data)
  data = fill_na(data,mapping, credit_flat)
  data = clean_budget(data, mapping)
  data = recup_orientation_old(data)
  data = data.loc[:,to_keep]
  return data











