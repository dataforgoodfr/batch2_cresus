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
    data['rav'] = (data['revenus_tot'] - data['charges'])/data['u_c']
    return data


#------------ Main -----------------------------

# Import du mapping
with open('../data/mapping.p', 'rb') as fp:
    mapping = pickle.load(fp)

data = pd.read_csv("../data/preprocessed_data.csv")
data = unite_consommation(data)
data = rav(data)

# Plots

A = data[data.orientation == 2]
B = data[data.orientation == 3]
C = data[data.orientation == 4]


fig = plt.figure(figsize=(12, 8))
fig.add_subplot(211)
h = plt.hist([A['rav'],B['rav'],C['rav']], color = ['green', 'orange', 'red'],bins= 30,
            range=[-1000, 4000],
            stacked=True, normed = True)
plt.title('Reste à vivre', fontsize=16)
fig.add_subplot(212) # 2 x 2 grid, 2nd subplot

h = plt.hist([A['sum_solde'],B['sum_solde'],C['sum_solde']], color = ['green', 'orange', 'red'],bins= 30,
            range=[0, 400000],
            stacked=True, normed = True)
plt.title('Capital restant dû', fontsize=16)
