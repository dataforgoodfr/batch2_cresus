import pandas as pd


def consistency_check(data, print_every_n=10000):
    """
    Check that columns that should not vary do not vary and remove faulty items.
    """
    uids_seen = {}
    to_drop = []
    l = len(data)
    for i, dup in data.loc[:, no_changers].duplicated().items():
        if not dup:
            uid = data.iloc[i].id
            if uid in uids_seen:
                print("L'id {} a des changements dans des colonnes qui ne doivent pas changer.".format(uid))
                print('Suppression de la ligne erronnée {}'.format(i))
                to_drop.append(i)
            else:
                uids_seen[uid] = None
        if not i % print_every_n:
            print("{}/{} lignes traitées.".format(i, l))
    print('On a supprimé {} lignes défectueuses.'.format(len(to_drop)))
    data.drop(to_drop, axis=0, inplace=True)
    return data

# Column indexes to be able to use masks.
data = pd.read_csv('extract_open.csv', encoding='cp1252', sep=',', low_memory=False)
credit = ['id_credit', 'type_credit', 'val_credit', 'mensualite', 'nb_mensualite', 'solde']
budget = ["id_budget","typ","revenus","allocations","pensions_alim","revenus_FL","autre1","autre2","autre3",
          "loyer","charges_loc_cop","gdf","electicite","eau","tel_fixe","tel_port","impots","taxe_fonciere",
          "taxe_habitation","assurance_habitat","assurance_voiture","mutuelle","autre_assurance","epargne_enfant",
          "frais_scolarite","transport_enfant","autres_charges_enfant","frais_bancaire","soins_recurrent",
          "frais_justice","frais_transport","epargne","autres_charges","fioul_bois","internet","abonnement_tv",
          "abonnement_autre","autre_charge","taxe_ordure","autre_impots","assurance_gav","assurance_prevoyance",
          "assurance_scolaire", "pensions_alim_payee","internat","frais_garde","cantine","alim_hyg_hab","dat_budget"]
action = ['id_action', 'object_full'] + ['Unnamed:{}'.format(i) for i in range(87, 193)]
changers = data.columns.isin(credit + budget + action)
no_changers = ~changers



# Call the treatment functions and report on progress.
print("Pré-traitement : {} lignes, {} colonnes".format(*data.shape))
print("On devrait obtenir {} entrées uniques.".format(len(data.id.unique())))
print("Nombre de lignes initiales: {}".format(len(data)))
data = consistency_check(data)
print("Nombre de lignes après vérification de la cohérence: {}".format(len(data)))

