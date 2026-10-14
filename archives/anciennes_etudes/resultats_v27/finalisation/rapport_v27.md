# V27 — Synthèse de convergence en maillage

Calculs à 1,5 V. Les concentrations en sortie sont **pondérées par le débit** de Poiseuille.

| Maillage | Résidu normalisé | Courant (A/m²) | Sel sortie (mol/m³) | Dessalement (%) | Na+ min. diluat (mol/m³) |
|---|---:|---:|---:|---:|---:|
| 45×18 | 9.474e-08 | -1956.866955 | 479.357892 | 4.128422 | 189.091598 |
| 60×24 | 1.892e-07 | -1953.103264 | 479.378259 | 4.124348 | 161.784871 |
| 75×30 | 3.127e-07 | -1952.956278 | 479.416779 | 4.116644 | 145.914868 |

## Écarts relatifs successifs (dénominateur : maillage le plus fin de la paire)

| Grandeur | 1 → 2 | 2 → 3 |
|---|---:|---:|
| Courant | 0.1927 % | 0.0075264 % |
| Sel sortie | 0.0042486 % | 0.0080347 % |
| Dessalement | 0.098764 % | 0.18714 % |
| Na+ minimum | 16.878 % | 10.876 % |

## Interprétation

- Les bilans de conservation et les équilibres de Donnan ont été vérifiés séparément par V26 sur les trois maillages.
- Les valeurs de résidu ont été obtenues avec des tolérances Newton différentes (1e-7, 2e-7, 4e-7) : les erreurs algébriques ne sont donc pas strictement comparables.
- La stabilité des grandeurs globales ne prouve pas la convergence locale. Examiner les profils Na+/Cl- et les minima.
- Le profil vertical de la première cellule du diluat change de position avec le raffinement ; il est qualitatif, pas une comparaison à abscisse physique identique.
- Une convergence asymptotique et un ordre de précision ne sont pas établis par ces seules données.
- Le modèle reste un prototype cartésien : aucune équivalence quantitative à COMSOL n’est démontrée.
