# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import pylab
pylab.ion()


# ------- Parameters ------------

ac = True  # Ne prédire que sur les classes A et C (B devient A)
toggle_plot = False

# ------- Local Functions -------


def error_rate(seuil):
    data['heur_pred'] = 4
    data['heur_pred'] = data['heur_pred'].where(~(data.rav >= seuil), 2)
    acc = np.mean(data.heur_pred == data.orientation)
    return 1 - acc


def heuristic(data, seuil_opti):
    """Prédiction de l'orientation en fonction 
    des seuils 100 et 400 euros de reste à vivre"""
    data['heur_pred'] = 4
    data['heur_pred'] = data['heur_pred'].where(~(data.rav >= seuil_opti), 2)
    # data['heur_pred'] = data['heur_pred'].where(~(data['rav'] <= 100), 4)
    if not(ac):
        data['heur_pred'] = data['heur_pred'].where(
            ~((data['imc'] <= 1 / 17) & (data['imc'] >= 1 / 23)), 3)
    return(data)


# ------------ Main ---------------
# ---------------------------------

# Import du mapping
with open('../data/mapping.p', 'rb') as fp:
    mapping = pickle.load(fp)

# Import data
data = pd.read_csv("../data/preprocessed_data.csv")
# # masks (pas besoin a priori)
# mask = ~data.columns.isin(['orientation', 'id', 'id_user'])
# data = data.loc[:, mask]

# Si on souhaite prédire sur A & C uniquement
if ac:
    data['orientation'] = data.where(
        ~(data.orientation == 3), 2)['orientation']

# Optimisation de l'heuristique
seuil_opti = minimize_scalar(error_rate, bracket=[100, 200]).x  # 170
data = heuristic(data, seuil_opti)

print('accuracy : %f' % np.mean(data['heur_pred'] == data['orientation']))
print(pd.crosstab(data['heur_pred'], data['orientation']))


# --- En optimisant le seuil pour chaque conseiller ---

# Fusionner en un seul méta conseiller ceux qui n'ont pas assez de dossiers
value_count = data.id_user.value_counts() < 20
to_merge = value_count[value_count == True].index.values
data.id_user[data.id_user.isin(to_merge)] = 10000


def error_rate_per_user(seuil):
    subdata = data.loc[data.id_user == user].copy()
    subdata['heur_pred'] = 4
    regle = subdata.rav >= seuil
    subdata['heur_pred'] = subdata['heur_pred'].where(~regle, 2)
    acc = np.mean(subdata.heur_pred == subdata.orientation)
    return 1 - acc


# Obtenir les seuils de chacuns des conseillers
id_users = data.id_user.unique()
seuil_optis = {}
for user in id_users:
    seuil_optis[user] = minimize_scalar(error_rate_per_user,
                                        bracket=[50, 200]).x

# Accuracy de cette configuration
data['heur_pred'] = 4
for user in id_users:
    regle = data[data.id_user == user].rav >= seuil_optis[user]
    data.ix[data.id_user == user,
            'heur_pred'] = data.ix[data.id_user == user,
                                   'heur_pred'].where(~regle, 2)
print('accuracy : %f' % np.mean(data['heur_pred'] == data['orientation']))


# ---------- Plots ---------------
if toggle_plot:
    A = data[(data.orientation == 2)]
    B = data[(data.orientation == 3)]
    C = data[(data.orientation == 4)]

    A_pred = data[(data.heur_pred == 2)]
    B_pred = data[(data.heur_pred == 3)]
    C_pred = data[(data.heur_pred == 4)]

    # print(A.shape[0] + B.shape[0] + C.shape[0])

    fig = plt.figure(figsize=(12, 18))
    fig.add_subplot(311)
    h = plt.hist([A['rav'], B['rav'], C['rav']], 
                 color=['green', 'orange', 'red'],
                 bins=30, range=[-1000, 2000], stacked=True, normed=True)
    plt.title('Reste à vivre', fontsize=10)
    fig.add_subplot(312)  # 2 x 2 grid, 2nd subplot

    h = plt.hist([A_pred['rav'], B_pred['rav'], C_pred['rav']],
                 color=['green', 'orange', 'red'],
                 bins=30, range=[-1000, 2000], stacked=True, normed=True)
    plt.title('Reste à vivre, classes prédites', fontsize=10)

    fig.add_subplot(313)
    s = plt.scatter(data['rav'], data['imc'])
    plt.title('IMC vs RAV', fontsize=10)
