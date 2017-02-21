# -*- coding: utf-8 -*-

import pandas as pd
import os

def import_data(folder):
    """
        Importe les données extraites de la base de données anonymisée de CRESUS,
        ajoute l'age et filtre les données issues des antennes de la fédération autre que celle de Strasbourg.
    """

    # Import de l'extrait de la base anonymisée
    data = pd.read_csv(os.path.join(folder,'out.csv'), sep='\t')
    print("\nimport_data -------------------------------------------")
    print("%i lignes et %i colonnes ont été importées."    %(data.shape[0], data.shape[1]))
    print("\nNombre de dossiers par plateforme d'origine :")
    for e in ["CRESUS", "social", "bancaire"]:
        print('{:>15} | {:3.0f}'.format(e,data[data.plateforme==e].shape[0]))
    print("\n")
    return data
