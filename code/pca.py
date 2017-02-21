# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pylab

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
pylab.ion()


data = pd.read_csv("../data/preprocessed_data.csv")
data_n = scale(data)
# Etude de la décroissance des valeurs propres
pca = PCA()
pca.fit(data_n)
plt.clf()
plt.plot(pca.explained_variance_ratio_)
#
plt.clf()
plt.plot(np.cumsum(pca.explained_variance_ratio_))


# Plot sur les deux premières dimensions
def plot_pca(data, x_comp, y_comp):
    ''' data : preprocessed data
        x_comp : index of the component used as x axis
        y_comp : index of the component used as y axis
        Plots the data on the hyperplan formed by the two eig vec
        chosen from PCA components
    '''
    plt.clf()
    pca = PCA()
    data2D = pca.fit_transform(data_n)[:, [x_comp, y_comp]]
    col = {2.: "red", 3.: "green", 4.: "blue"}
    lab = {2.: 'Accompagnement', 3.: 'Mediation', 4.: 'Surendettement'}
    for orient in data.orientation.unique():
        to_plot = pd.DataFrame(data2D)[np.array(data.orientation == orient)]
        plt.plot(
            to_plot[0], to_plot[1], 'ro', c=col[orient], label=lab[orient])
    plt.legend(loc='upper left')


def reduce_dim_pca(data, var_exp):
    '''
    Réduit la dimension du data set, en gardant var_exp % de variance expliquée
    data : preprocessed data
    var_exp : % de variance expliquée que l'on souhaite garder
    output : data (shape changed)
    '''
    pca = PCA()
    data = scale(data)
    new_data = pca.fit_transform(data)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    thresh = 2. / 3
    k = len(cum_var[cum_var < thresh])
    new_data = new_data[:, range(k)]
    return pd.DataFrame(new_data)


plot_pca(data, 0, 1)
