# V6 – flux périodiques expérimentaux

Ajouts : `periodic_flux_v6.py` et `audit_periodic_assembly_v6.py`.

Le module ajoute au résidu et au jacobien PETSc des termes de flux Nernst–Planck appariés pour les 160 couples de facettes, avec une différence affine de potentiel de 1,5 V et un terme de pénalisation en potentiel électrochimique. Chaque flux est appliqué avec des signes opposés aux deux cellules pour la conservation algébrique.

**Limites :** quadrature au point milieu, pénalisation heuristique, sans termes adjoints/symétriques SIPG, sans preuve de stabilité ou de convergence. Les flux membranaires v4 restent expérimentaux. Le module ne lance pas Newton et n'est pas une reproduction COMSOL validée. Le jacobien est une dérivée analytique de l'opérateur de facette proposé, pas une validation de l'ensemble du modèle.

Dans l'environnement Miniforge `electrodialysis`, depuis le dossier du projet :

```sh
python audit_periodic_assembly_v6.py
```

L'exécution DOLFINx n'a pas pu être testée ici. Une erreur de version/API doit être corrigée avant toute utilisation. Conserver le projet original en sauvegarde.
