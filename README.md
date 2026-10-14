# Simulation d'électrodialyse 2D

## Objectif

Développer un solveur Python d'électrodialyse 2D
et comparer ses résultats au modèle COMSOL 6.4
"Desalination in an Electrodialysis Cell".

## Modèle numérique

- Méthode des volumes finis
- Transport de Na+ et Cl- par Nernst-Planck
- Électroneutralité
- Membranes CEM et AEM
- Conditions de Donnan couplées
- Écoulement de Poiseuille dans les canaux
- Périodicité du potentiel en x
- Conditions d'entrée de Danckwerts
- Résolution non linéaire par Newton creux

## Scripts principaux

electrodialyse_membranes_v28.py
    Solveur principal.

validation_physique_v26.py
    Vérification des bilans ioniques, de l'électroneutralité
    et des conditions de Donnan.

continuation_tension_v25.py
    Calculs successifs à différentes tensions.

convergence_maillage_v27.py
    Étude de sensibilité au maillage.

visualisation_electrodialyse_v24.py
    Visualisation des champs calculés.

comparaison_comsol_v28.py
    Export des résultats Python et comparaison avec COMSOL.

audit_comsol_v28.py
    Inspection des paramètres du fichier COMSOL.

## Résultats de référence V28

Tension : 1.5 V

Maillage     Courant moyen
45 x 18      -1740.27 A/m2
60 x 24      -1739.04 A/m2
75 x 30      -1739.30 A/m2

Solution de référence :
resultats_v28/essai_75x30/solution.npz

## Validation

- Newton converge sur les trois maillages.
- Bilans ioniques discrets satisfaits.
- Électroneutralité respectée.
- Conditions de Donnan respectées.
- Courant moyen peu sensible au raffinement.

Limites :
- Convergence locale des concentrations non démontrée.
- Comparaison quantitative avec COMSOL non réalisée.
- Compatibilité fonctionnelle des scripts historiques
  de continuation et de convergence avec V28 à confirmer.

## Archives

Le dossier archives/ contient les anciens scripts,
maillages, solutions et documents historiques.
