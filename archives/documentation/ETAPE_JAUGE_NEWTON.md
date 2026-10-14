# Étape : référence du potentiel dans Newton affine

Modifications :
- `newton_affine.py` accepte `gauge_dof` et `gauge_value` ; élimine une colonne de T et l'équation correspondante afin de fixer le potentiel absolu.
- `run_newton_experimental.py` choisit un DOF de potentiel indépendant et le fixe à 0 V.
- `test_newton_gauge_synthetique.py` vérifie la réduction affine, la jauge et le saut 1,5 V sur un système synthétique sans DOLFINx.

Test exécuté dans l'environnement de préparation : `python test_newton_gauge_synthetique.py` -> OK.

## Limites
- Aucun essai DOLFINx 0.11 sur le Mac utilisateur n'a été exécuté.
- Le résidu DG reste incomplet (flux inter-facettes périodiques et membranes), et `run_newton_experimental.py` conserve son garde-fou.
- Le test synthétique ne valide pas la formulation physique ni la convergence de l'électrodialyse.
