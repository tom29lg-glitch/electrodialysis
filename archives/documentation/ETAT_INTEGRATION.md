# État du projet d'électrodialyse (inspection du ZIP)

## Vérifié
- Archive contenant 49 entrées, incluant les XDMF et HDF5 des domaines et frontières.
- `solver_comsol_exact.py` contient les champs DG1 Na+, Cl-, potentiel, la charge fixe, les diffusivités, une initialisation Donnan, la convection, un résidu et son jacobien UFL.
- `periodic_dofs.npy` contient 960 correspondances périodiques vérifiées précédemment sur le Mac.
- `periodic_affine.py` ajouté : construit la réduction algébrique `w=Tq+g`, avec décalage +1.5 V du potentiel droit. Test algébrique exécuté dans l'environnement de travail.

## Pas encore résolu : bloquants scientifiques
1. La périodicité DG exige des flux numériques entre facettes gauches et droites. La réduction des DOFs seule ne suffit pas.
2. Le terme membranaire actuel pénalise le saut de potentiel électrochimique mais n'est pas un flux de Nernst–Planck complet et consistant hors équilibre.
3. L'équation d'électroneutralité testée localement ne garantit pas à elle seule la conservation du courant à toutes les interfaces ; la formulation discrète doit être contrôlée.
4. Le solveur Newton n'est pas encore connecté ; aucune convergence à 1.5 V n'a été obtenue ni revendiquée.
5. La jauge du potentiel et les diagnostics de conservation sont à fixer avant validation.

## Exécution
Depuis le répertoire `electrodialysis` : `python periodic_affine.py`.
Ce test vérifie uniquement la réduction affine, pas le modèle physique.

## Ajout du 13 octobre : moteur Newton affine
`newton_affine.py` implémente la réduction des résidus et jacobiens PETSc
par `T.T @ R` et `T.T @ J @ T`, avec amortissement et contrôle de positivité.
Ce module n'est **pas branché** sur `solver_comsol_exact.py` : les flux DG
périodiques, la formulation membranaire conservative et la jauge restent
à résoudre. L'utiliser maintenant sur le résidu actuel ne validerait pas
le modèle COMSOL.
