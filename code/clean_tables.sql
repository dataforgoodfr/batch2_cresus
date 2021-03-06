USE cresus;

# Les données de dates de naissances ont été envoyées séparément, il faut les réintégrer
DROP TABLE IF EXISTS naissance;
CREATE TABLE naissance (
  id          INT UNSIGNED NOT NULL,
  annee_naissance     INT UNSIGNED)
ENGINE = MyISAM;
LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 5.7/Uploads/annee_naissance.txt' INTO TABLE naissance;

# Mise à jour de dossier avec l'age
DROP TABLE IF EXISTS dossier_maj;
CREATE TABLE dossier_maj
	SELECT dossier.*, naissance.annee_naissance, YEAR(dossier.date_ouverture)-naissance.annee_naissance AS age
    FROM dossier, naissance
    WHERE dossier.id = naissance.id;



## Group credits by [id_dossier, type_credit]
DROP TABLE IF EXISTS credits_bytype;
CREATE TABLE credits_bytype
	SELECT id_dossier,
			type_credit,
			SUM(mensualite) AS sum_mensualite,
			AVG(nb_mensualite) moy_nb_mensualite,
			SUM(solde) AS sum_solde
	FROM credits
	GROUP BY id_dossier, type_credit;


# Etale en colonnes les différentes catégories
DROP TABLE IF EXISTS credits_extended;
CREATE TABLE credits_extended
	SELECT credits_bytype.*,
		CASE WHEN type_credit = 0
			THEN sum_mensualite END AS sum_mensualite_0,
		CASE WHEN type_credit = 1
			THEN sum_mensualite END AS sum_mensualite_1,
		CASE WHEN type_credit = 2
			THEN sum_mensualite END AS sum_mensualite_2,
		CASE WHEN type_credit = 3
			THEN sum_mensualite END AS sum_mensualite_3,
		CASE WHEN type_credit = 4
			THEN sum_mensualite END AS sum_mensualite_4,
		CASE WHEN type_credit = 5
			THEN sum_mensualite END AS sum_mensualite_5,
		CASE WHEN type_credit = 0
			THEN moy_nb_mensualite END AS moy_nb_mensualite_0,
		CASE WHEN type_credit = 1
			THEN moy_nb_mensualite END AS moy_nb_mensualite_1,
		CASE WHEN type_credit = 2
			THEN moy_nb_mensualite END AS moy_nb_mensualite_2,
		CASE WHEN type_credit = 3
			THEN moy_nb_mensualite END AS moy_nb_mensualite_3,
		CASE WHEN type_credit = 4
			THEN moy_nb_mensualite END AS moy_nb_mensualite_4,
		CASE WHEN type_credit = 5
			THEN moy_nb_mensualite END AS moy_nb_mensualite_5,
		CASE WHEN type_credit = 0
			THEN sum_solde END AS sum_solde_0,
		CASE WHEN type_credit = 1
			THEN sum_solde END AS sum_solde_1,
		CASE WHEN type_credit = 2
			THEN sum_solde END AS sum_solde_2,
		CASE WHEN type_credit = 3
			THEN sum_solde END AS sum_solde_3,
		CASE WHEN type_credit = 4
			THEN sum_solde END AS sum_solde_4,
		CASE WHEN type_credit = 5
			THEN sum_solde END AS sum_solde_5
	FROM credits_bytype;

# Aggrège par id_dossier
DROP TABLE IF EXISTS credits_grouped;
CREATE TABLE credits_grouped
	SELECT id_dossier,
			COUNT(id_dossier) AS nombre_credits,
			SUM(sum_mensualite_0) AS sum_mensualite_0,
			SUM(sum_mensualite_1) AS sum_mensualite_1,
			SUM(sum_mensualite_2) AS sum_mensualite_2,
			SUM(sum_mensualite_3) AS sum_mensualite_3,
			SUM(sum_mensualite_4) AS sum_mensualite_4,
			SUM(sum_mensualite_5) AS sum_mensualite_5,
			SUM(moy_nb_mensualite_0) AS moy_nb_mensualite_0,
			SUM(moy_nb_mensualite_1) AS moy_nb_mensualite_1,
			SUM(moy_nb_mensualite_2) AS moy_nb_mensualite_2,
			SUM(moy_nb_mensualite_3) AS moy_nb_mensualite_3,
			SUM(moy_nb_mensualite_4) AS moy_nb_mensualite_4,
			SUM(moy_nb_mensualite_5) AS moy_nb_mensualite_5,
			SUM(sum_solde_0) AS sum_solde_0,
			SUM(sum_solde_1) AS sum_solde_1,
			SUM(sum_solde_2) AS sum_solde_2,
			SUM(sum_solde_3) AS sum_solde_3,
			SUM(sum_solde_4) AS sum_solde_4,
			SUM(sum_solde_5) AS sum_solde_5
	FROM credits_extended
	GROUP BY id_dossier;


-- # Vérification de la diminution de lignes
-- SELECT COUNT(*) FROM credits_grouped;
-- SELECT COUNT(*) FROM credits;


## Remove duplicates from action
DROP TABLE IF EXISTS action_unique;
CREATE TABLE action_unique
	SELECT id_dossier, objectfull
	FROM action
	WHERE id_action IN
		(SELECT min(a2.id_action)
		FROM action a2
		GROUP BY a2.id_dossier);

-- # Vérification de l'unicité des actions par dossier
-- SELECT COUNT(*) FROM action_unique;
-- SELECT COUNT(DISTINCT id_dossier) FROM action;

## Ne garder que le dernier budget en date pour chaque id_dossier
DROP TABLE IF EXISTS budget_dat_max;
CREATE TABLE budget_dat_max
	SELECT *
    FROM (
		SELECT *
		FROM budget
		WHERE typ NOT LIKE 'Impaye') as b
    WHERE b.dat_budget = (
		SELECT MAX(dat_budget)
		FROM (
			SELECT *
            FROM budget
            WHERE typ NOT LIKE 'Impaye') as b2
		WHERE b.id_dossier = b2.id_dossier);

## Sélectioner à date égale l id le plus élevé
DROP TABLE IF EXISTS budget_unique;
CREATE TABLE budget_unique
	SELECT *
	FROM budget_dat_max
	WHERE id_budget IN
		(SELECT max(b2.id_budget)
		FROM budget_dat_max b2
		GROUP BY b2.id_dossier);

-- # Vérification de la diminution de ligne
-- SELECT COUNT(*) FROM budget_dat_max;
-- SELECT COUNT(*) FROM budget_unique;
-- SELECT COUNT(DISTINCT id_dossier) FROM budget;


# JOIN dossier, credits_grouped, action, budget_uniquified
DROP TABLE IF EXISTS extract;
CREATE TABLE extract
	SELECT
		  `id`,
          `etat`,
          `id_group`,
          `plateforme`,
          `id_user`,
          `charte`,
          `duree`,
          `profession`,
          `logement`,
          `situation`,
          `transferable`,
          `retard_facture`,
          `retard_pret`,
          `nature`,
          `orientation`,
          `personne_charges`,
          `vip`,
          `indicateur_suivi`,
          `releve_bancaire`,
          `aide_sociale`,
          `reactive`,
          `memo`,
          `etat_old`,
          `orientation_old`,
          `indicateur_suivi_old`,
          `transfert`,
          `plan_bdf`,
          `effacement_dette`,
          `gain_mediation`,
          `PCB` ,
          `mensu_bdf`,
          `age`,
          #`id_credit`,
          #`type_credit`,
          #`val_credit`,
          #`mensualite`,
          #`nb_mensualite`,
          #`solde`,
          #`id_budget`,
          `nombre_credits`,
            `sum_mensualite_0`,
	    	`sum_mensualite_1`,
	    	`sum_mensualite_2`,
	    	`sum_mensualite_3`,
	    	`sum_mensualite_4`,
	    	`sum_mensualite_5`,
	    	`moy_nb_mensualite_0`,
	    	`moy_nb_mensualite_1`,
	    	`moy_nb_mensualite_2`,
	    	`moy_nb_mensualite_3`,
	    	`moy_nb_mensualite_4`,
	    	`moy_nb_mensualite_5`,
	    	`sum_solde_0`,
	    	`sum_solde_1`,
	    	`sum_solde_2`,
	    	`sum_solde_3`,
	    	`sum_solde_4`,
	    	`sum_solde_5`,
          `typ`,
          `revenus`,
          `allocations`,
          `pensions_alim`,
          `revenus_FL`,
          `autre1`,
          `autre2`,
          `autre3`,
          `loyer`,
          `charges_loc_cop`,
          `gdf`,
          `electicite`,
          `eau`,
           budget_unique.`tel_fixe` as tel_fixe,
          `tel_port`,
          `impots`,
          `taxe_fonciere`,
          `taxe_habitation`,
          `assurance_habitat`,
          `assurance_voiture`,
          `mutuelle`,
          `autre_assurance`,
          `epargne_enfant`,
          `frais_scolarite`,
          `transport_enfant`,
          `autres_charges_enfant`,
          `frais_bancaire`,
          `soins_recurrent`,
          `frais_justice`,
          `frais_transport`,
          `epargne`,
          `autres_charges`,
          `fioul_bois`,
          `internet`,
          `abonnement_tv`,
          `abonnement_autre`,
          `autre_charge`,
          `taxe_ordure`,
          `autre_impots`,
          `assurance_gav`,
          `assurance_prevoyance`,
          `assurance_scolaire`,
          `pensions_alim_payee`,
          `internat`,
          `frais_garde`,
          `cantine`,
          `alim_hyg_hab`,
          `dat_budget`,
          #`id_action`,
          `objectfull`
	FROM dossier_maj
    INNER JOIN (SELECT `id_partenaire`, `plateforme`
				FROM partenaire
                WHERE `plateforme` IN ('bancaire', 'social', 'CRESUS')
                ) AS p ON
                dossier_maj.`id_group`=p.`id_partenaire`
	LEFT JOIN credits_grouped ON
		dossier_maj.`id`=credits_grouped.`id_dossier`
	LEFT JOIN budget_unique ON
		dossier_maj.`id`=budget_unique.`id_dossier`
	LEFT JOIN action_unique ON
		dossier_maj.`id`=action_unique.`id_dossier`
;

-- # Vérification du nombre final de lignes
-- SELECT COUNT(*) FROM extract;
-- SELECT COUNT(DISTINCT id) FROM dossier;
