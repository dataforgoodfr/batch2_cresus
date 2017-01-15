# -*- coding: utf-8 -*-

import pandas as pd

def import_data():
  """ 
    Importe les données extraites de la base de données anonymisée de CRESUS,
    ajoute l'age et filtre les données issues des antennes de la fédération autre que celle de Strasbourg.
  """

  # Import de l'extrait de la base anonymisée
  raw = pd.read_csv('out.csv', sep='\t')

  # Import de la table relationnelle partenaires, et sélection des ids pertinents
  origine =  pd.read_csv('partenaires.csv', sep=';')
  relevant_idx = origine.id_partenaire[origine.plateforme.isin(["bancaire", "social", "CRESUS"])]
  
  # Suppression des lignes issues de platformes hors Alsace
  left = raw[raw.id_group.isin(relevant_idx)]
  print("%i lignes et %i colonnes ont été importées après suppression de %i lignes issues de la féfération Crésus hors Alsace"  %(left.shape[0], left.shape[1], raw.shape[0]-left.shape[0]))

  # Import des années de naissance et d'ouverture, calcul de la différence pour obtenir l'age
  naissance = pd.read_csv('annee_naissance.csv',sep=';')
  naissance.columns = ['id', 'annee_naissance']
  ouverture = pd.read_csv('annee_ouverture.csv',sep=';')
  right = pd.merge(ouverture, naissance, on='id')
  right['age'] = right['annee_ouverture']-right['annee_naissance']

  # Jointure des données
  data = pd.merge(left, right.loc[:,['id', 'age']], on='id')
  return data