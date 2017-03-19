# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from matplotlib import pyplot as plt
import pylab
pylab.ion()

#      ----------- Data import -----------
#      -----------------------------------

np.random.seed(1)


data = pd.read_csv("../data/preprocessed_data.csv")

mask = ~data.columns.isin(['orientation', 'id'])
X = data.loc[:, mask]

s = StandardScaler()
Xs = s.fit_transform(X)

#     ---------- K means ------------
#     -------------------------------

# Kmeans : réduire à 1000 groupes
kmeans_1k = KMeans(n_clusters=1000, random_state=0).fit(Xs)
X_1k = kmeans_1k.cluster_centers_


#     ---------- Classification ascendante hierarchique ------------
#     --------------------------------------------------------------

# Matrice de distance
Z = linkage(X_1k, method='ward', metric='euclidean')
# Affichage du dendrogramme
plt.title("CAH")
dendrogram(Z, orientation='left', color_threshold=0)
plt.show()  # on choisit t=90

# CAH
T = fcluster(Z, t=75, criterion='distance')
repartition = pd.Series(T)

# Aggréger les petites classes ensemble
small_classes = repartition.value_counts()[
                                        repartition.value_counts() < 30].index
repartition[repartition.isin(small_classes)] = 0


# Calculer les centres des clusters
lens = {}
centroids = {}
for idx, clno in enumerate(T):
    centroids[clno] = centroids.get(clno,
                                    np.zeros(X_1k.shape[1])) + X_1k[idx, :]
    lens[clno] = lens.get(clno, 0) + 1
# Divide by number of observations in each cluster to get the centroid
for clno in centroids:
    centroids[clno] /= float(lens[clno])
centers = np.vstack(list(centroids.values()))


#     ---------- K means ------------
#     -------------------------------

# Encore un petit coup de kmeans pour lisser les résultats:
kmeans_final = KMeans(n_clusters=repartition.nunique(),
                      random_state=0,
                      init=centers).fit_predict(X_1k)


# A faire : décrire les classes selon les variables
# qui les caractérisent le mieux

# Output : to R
X_1kn = s.inverse_transform(X_1k)
to_R = pd.DataFrame(X_1kn, columns=data.loc[:, mask].columns.values)
to_R['cluster'] = kmeans_final
to_R.to_csv("../data/export_to_R.csv")