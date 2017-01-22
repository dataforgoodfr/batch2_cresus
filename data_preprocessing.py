# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder


# ------- Local functions -------
# -------------------------------

# Masks:
def create_masks(data) :
    ''' Créée les masques qui serviront à séléctionner les variables pertinentes.
        Divise les variables budgétaires en sous parties.
        Les variables inutiles sont exclues de ces sous parties.
    '''
    budget = ['revenus', 'allocations',
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
            #,'dat_budget', 'typ'
            ]
    autres_infos = ['id','age', 'profession', 'logement', 'situation', 'retard_facture',
                 'retard_pret','orientation','personne_charges'
                  #, 'releve_bancaire' ,'transferable', 'nature',
                  ]
    new_cols = ['sum_mensualite',
              #'moy_nb_mensualite',
              'sum_solde']
    credit_detail = []
    for text in new_cols:
        credit_detail.append(["{}_{}".format(text, i) for i in range(6)])
    credit_flat = [item for sublist in credit_detail for item in sublist]

    to_keep = autres_infos + credit_flat + new_cols + budget

    print("\ncreate_masks ------------------------------------------\n")
    return [budget, autres_infos, new_cols, credit_detail, to_keep]

def create_features(data, to_keep, credit_detail) :
    '''
        1. Aggrège les mensualités et soldes
        2. Aggrège les revenus et les charges
        3. Récupère les informations contenues dans 'orientation_old' si besoin
    '''
    # Aggrège les différents types de crédits
    for i, col in enumerate(['sum_mensualite', 'sum_solde']):
        data.loc[:, col] = np.sum(data[credit_detail[i]], axis=1)

    # Les None dans les colonnes du crédit correspondent à des 0 (absence de crédit)
    fill_with_zeros = [item for sublist in credit_detail for item in sublist]
    for col in fill_with_zeros:
        data[col].fillna(value = 0,
                    inplace = True)

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

    # Récupère les orientations contenues dans 'orientation_old' si (orientation in [0,1])
    d = {'surendettement'   : 4 ,
        'accompagnement'    : 2,
        'mediation'         : 3,
        'microcredit'       : 5,
        'Microcredit'       : 5}
    data.orientation_old = data.orientation_old.apply(lambda x : d.get(x,0))

    data.ix[(data.orientation_old > 1)
          & (data.orientation<=1), 'orientation'] =  data.ix[(data.orientation_old > 1)
                                                          & (data.orientation<=1), 'orientation_old']
    print("\ncreate_features ---------------------------------------\n")
    return [data, to_keep]

def filter_data(data):
    ''' Filtre les dossiers qui
    - n'ont pas reçu d'orientation,
    - dont l'orientation est inconnue
    - dont l'orientation est inconnue correspond plutôt à des mesures professionnelles (microcrédit)
    - dont on a pas de données budgétaires'''

    print("\nfilter_data -------------------------------------------")
    n_tot = data.shape[0]

    orientation_check = data[(data.orientation < 2)&(data.orientation > 4)].shape[0]
    print("Orientations : %i valeurs non retenues soit %i%% du jeu." %(orientation_check, 100*orientation_check/n_tot))

    charges_check = data.loc[data.charges<=0,:].shape[0]
    print("Charges : %i valeurs négatives ou nulles soit %i%% du jeu." %(charges_check, 100*charges_check/n_tot))

    revenus_check = data[data.revenus_tot == 0].shape[0]
    print("Revenus : %i valeurs non retenues soit %i%% du jeu." %(revenus_check, 100*revenus_check/n_tot))

    data = data[(data.orientation > 1)&(data.orientation < 5)]
    data = data.loc[data.charges>0,:]
    data=data[data.revenus_tot.notnull()] # Elimine ceux pour lesquels on a pas de budget

    print("Nombre de dossiers par plateforme d'origine après filtrage :")
    for e in ["CRESUS", "social", "bancaire"]:
        print('{:>15} | {:3.0f}'.format(e,data[data.plateforme==e].shape[0]))
    print("\n")
    return data



def encode_categ(data):
    """Encode les variables catégorielles
        Renvoie le jeu de données modifié et un mapping"""
    le = LabelEncoder()
    mapping = dict()
    for col, dtype in zip(data.columns, data.dtypes):
        if dtype == 'object':
            data[col] = data[col].apply(lambda s: str(s))
            # Replace 0 and NaNs with unique label : 'None'
            data[col] = data[col].where(~data[col].isin(['0','nan']), 'None')
            data[col] = le.fit_transform(data[col])
            mapping[col]= dict(zip(le.inverse_transform(data[col].unique()), data[col].unique()))

    print("\nencode_categ ------------------------------------------\n")
    return [data,mapping]


def detect_na(data):
    '''  Détecter les 0 qui sont en réalité des NAs et les valeurs aberrantes'''
    n_tot = data.shape[0]

    print("detect_na ---------------------------------------------")

    mens_check = data[(data.sum_mensualite <= 0) & (data.sum_mensualite > 8000)].shape[0]
    print("Mensualiés : %i valeurs aberrantes soit %i%% du jeu. " %(mens_check, 100*mens_check/n_tot))

    age_check = data[(data.age<18) & (data.age>90)].shape[0]
    print("Age : %i valeurs aberrantes soit %i%% du jeu. " %(age_check, 100*age_check/n_tot))

    pac_check = data.personne_charges[(data.personne_charges<0) & (data.personne_charges>10)].shape[0]
    print("Personnes à charge : %i valeurs aberrantes soit %i%% du jeu. \n" %(pac_check, 100*pac_check/n_tot))
    return data

def fill_na(data, mapping):
    '''Identifie les 0 qui devraient être des NAs, et les remplace par des plus proche voisins
    '''

    locataire = mapping['logement']['locataire']
    n_neighbors = 5
    for col in ['loyer', 'gdf', 'electicite', 'eau', 'assurance_habitat']:
        regle = (data.logement == locataire) & (data[col] == 0)
        print("%s : %i valeurs sont effacées et recomplétées" %(col,data[regle][col].shape[0]))
        data[col] = data[col].where(~regle, None)
        if data[col].isnull().sum()>0:
            knn = neighbors.KNeighborsRegressor(n_neighbors)
            data_temp = data.loc[(data.logement== locataire) & (~data[col].isnull()), :]
            mask = ~data.columns.isin([col, 'id'])
            X = data_temp.loc[:, mask]
            null_index = data[col].isnull()
            y_ = knn.fit(X, data_temp[col]).predict(data.loc[null_index,mask])
            data.loc[null_index,col] = y_
            data[col] = data[col].astype(float)

    print("\nfill_data ---------------------------------------------\n")
    return data