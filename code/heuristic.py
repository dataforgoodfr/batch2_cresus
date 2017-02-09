# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab
pylab.ion()
import pickle


def unite_consommation(data):
    """Ajoute la colonne unite de consommation ('u_c') à partir des données de situation et
    personnes à charge.
    L'échelle actuellement la plus utilisée (dite de l'OCDE) retient la pondération suivante :
    - 1 UC pour le premier adulte du ménage ;
    - 0,5 UC pour les autres personnes de 14 ans ou plus ;
    - 0,3 UC pour les enfants de moins de 14 ans.
    """

    data['u_c'] = 1

    # On identifie les bénéficiaires en couple
    situation_couple = ['concubinage', 'marie', 'pacs']
    couple = []
    for e in situation_couple:
        couple.append(mapping['situation'][e])

    # On ajuste l'unité de consommation en fonction
    data['u_c'] = data['u_c'].where(~data['situation'].isin(couple), 1.5)

    # On ajoute la pondération des personnes à charge
    data['u_c'] = data['u_c'] + data['personne_charges']*0.3
    return(data)

def rav(data):
    """Ajoute la colonne reste-à-vivre ('rav') calculée comme la différence des revenus totaux
    moins les charges totales, divisée par l'unitité de consommation
    Requiert donc d'avoir calculé l'unité de consommation.
    """
    data['rav'] = (data['revenus_tot'] - data['charges'] - data['sum_mensualite'])/data['u_c']
    return data

def max_filter(data):
    """Filtre Max"""
    data = data[(data.charges > 400) & (data.revenus_tot > 400)]
    data['orientation'] = data.where(~(data.orientation == 3), 2)['orientation']
    print('Nombre de lignes retenures %i' %data.shape[0])
    return data

def heuristic(data):
    """Prédiction de l'orientation en fonction des seuils 100 et 400 euros de reste à vivre"""
    data['rav_pred'] = 4
    data['rav_pred'] = data['rav_pred'].where(~(data['rav'] >=145), 2)
    #data['rav_pred'] = data['rav_pred'].where(~(data['rav'] <= 100), 4)
    return(data)

#------------ Main -----------------------------

# Import du mapping
with open('../data/mapping.p', 'rb') as fp:
    mapping = pickle.load(fp)

# Calcul de l'unité de conso, du rav et de la prédiction
data = pd.read_csv("../data/preprocessed_data.csv")
data = max_filter(data)
data = unite_consommation(data)
data = rav(data)
data = heuristic(data)

print('accuracy : %f' %np.mean(data['rav_pred'] == data['orientation']))
print(pd.crosstab(data['rav_pred'], data['orientation']))


# Plots

A = data[(data.orientation == 2)]
B = data[(data.orientation == 3)]
C = data[(data.orientation == 4)]

print(A.shape[0]+B.shape[0]+C.shape[0])


fig = plt.figure(figsize=(12, 12))
fig.add_subplot(211)
h = plt.hist([A['rav'],B['rav'],C['rav']], color = ['green', 'orange', 'red'],bins= 30,
            range=[-1000, 2000],
            stacked=True, normed = True)
plt.title('Reste à vivre', fontsize=10)
fig.add_subplot(212) # 2 x 2 grid, 2nd subplot

h = plt.hist([A['sum_solde'],B['sum_solde'],C['sum_solde']], color = ['green', 'orange', 'red'],bins= 30,
            range=[0, 400000],
            stacked=True, normed = True)
plt.title('Capital restant dû', fontsize=10)
