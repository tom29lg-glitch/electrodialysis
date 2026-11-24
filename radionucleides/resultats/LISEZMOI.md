# Résultats retenus (radionucléides)

Hypothèses de référence : géométrie et essais de Xie et al. 2022, membranes CEM 6200 / AEM 9000 mol/m3
(capacité d'échange, taux d'eau 0,3 g/g), rendement de courant 0,82, vitesse effective 2,23 x 0,82^3 x v,
arrêt au courant limite.

| Dossier | Contenu |
|---|---|
| validation_xie_REFERENCE_membranes_xie | Les 9 essais de Xie, hypothèses de référence (écart moyen 4,9 %) |
| validation_xie_comparaison_cmem1000 | Même chose avec cMem = 1000 (valeur COMSOL) : écart 3,5 % |
| validation_xie_comparaison_sans_rendement | Rendement 1 : biais de +14 % |
| validation_xie_sensibilite_A2 | Essai A2 : partage K, cMem 500/2000, membranes épaisses, rendement 0,87 |
| optimisation_*_REFERENCE | Balayages : courant et mode de pilotage (frac), vitesse (v), canaux (wch), membranes (cmem), volumes (VR) |
| courbe_iv_geometrie_xie_calee | Courbe courant-tension dans la géométrie de Xie, vitesse effective : i_lim ~ 68-72 A/m2 |
| recirculation_exemple_geometrie_comsol | Premier essai de recirculation (cellule COMSOL, NaCl 34,2 mol/m3) |
| figures_diapo | Figures du diapo d'avancement |
| _archives | Calculs remplacés (cMem 1000, canal vide, mode plafond) et calculs interrompus : peut être supprimé |

Pour recalculer, les scripts créent de nouveaux dossiers (voir ../README.md) :

    python validation_xie.py --tag REFERENCE_membranes_xie --rendement 0.82 --cmem 6200 --cmem-aem 9000 --rapide
    python optimisation.py --balayage frac --ratio-aem 1.46 --base cmem=6200 --suffixe _REFERENCE --objectif 0.999
