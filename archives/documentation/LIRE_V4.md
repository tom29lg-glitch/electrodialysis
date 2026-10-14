# Étape v4 — interfaces membranaires

Fichier ajouté : `solver_comsol_membranes_v4.py`. Il conserve les originaux.

Modification : le flux numérique aux quatre interfaces membranaires inclut désormais le flux Nernst–Planck moyen et une pénalisation du saut de potentiel électrochimique. Dans la version précédente, il ne restait que la pénalisation, donc le flux s’annulait artificiellement à l’équilibre de Donnan même si le gradient transportait du sel.

**Statut : expérimental et non validé.** Le terme adjoint et la stabilité de la formulation DG restent à examiner. Il manque les flux périodiques gauche/droite et le traitement correct de la contrainte de courant. Newton ne doit pas être lancé sur ce modèle pour interpréter des résultats COMSOL.

Contrôle statique : `python -m py_compile solver_comsol_membranes_v4.py`

Diagnostic d’assemblage : `python solver_comsol_membranes_v4.py` dans l’environnement DOLFINx 0.11.0.
