# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import pylab
pylab.ion()

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from scipy.cluster.hierarchy import dendrogram, linkage

data = pd.read_csv("../data/preprocessed_data.csv")

mask = ~data.columns.isin(['orientation', 'id'])
X = data.loc[:, mask]

X = scale(X)

# Kmeans : réduire à 200 groupes
kmeans_200 = KMeans(n_clusters=200, random_state=0).fit(X)
X_200 = kmeans_200.cluster_centers_


# Générer la matrice des liens
Z = linkage(X_200, method='ward', metric='euclidean')
# Affichage du dendrogramme
plt.title("CAH")
dendrogram(Z, orientation='left', color_threshold=0)
plt.show()