CREATE TABLE extract
	SELECT etat,
		id_group,
		id_user,
		charte,
		duree,
		profession,
		logement,
		situation,
		transferable,
		retard_facture,
		retard_pret,
		nature,
		/*debutpret, finpret, datetime	,montantpret,media*/
		orientation,
		personne_charges,
		vip,
		indicateur_suivi,
		releve_bancaire,	
		aide_sociale,	
		reactive,
		memo,
		etat_old,	
		orientation_old,	
		indicateur_suivi_old,	
		transfert,
		plan_bdf,	
		effacement_dette,
		gain_mediation,	
		PCB,	
		mensu_bdf
		  
	FROM dossier
	LEFT JOIN credits ON
		dossier.`id`=credits.`id_dossier` 
	LEFT JOIN budget ON
		dossier.`id`=budget.`id_dossier` 
	LEFT JOIN action ON
		dossier.`id`=action.`id_dossier` ;
