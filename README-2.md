# Simulation d'électrodialyse 2D

Ce projet simule le transport de Na⁺ et Cl⁻ dans une cellule d'électrodialyse à trois canaux et deux membranes (CEM et AEM). Le modèle utilise une discrétisation par volumes finis, les équations de Nernst–Planck, l'électroneutralité et des conditions d'interface de Donnan.

## Démarrage rapide

1. Installer Python et les dépendances :

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Modifier les paramètres dans `config.json`.
3. Lancer la simulation depuis le dossier du projet :

   ```bash
   python main.py
   ```

Les résultats sont enregistrés dans un nouveau sous-dossier de `resultats/`, nommé d'après la simulation et sa date d'exécution.

## Paramètres principaux (`config.json`)

| Paramètre | Signification | Exemple |
| --- | --- | --- |
| `simulation_name` | Nom de la simulation | `test_1p5V` |
| `voltage` | Tension cible (V) | `1.5` |
| `ions.Na+.diffusion` | Diffusion de Na⁺ (m²/s) | `2.5e-9` |
| `ions.Cl-.diffusion` | Diffusion de Cl⁻ (m²/s) | `2.0e-9` |
| `concentration_inlet` | Concentration d'entrée (mol/m³) | `500` |
| `temperature` | Température (K) | `298.15` |
| `velocity` | Vitesse moyenne (m/s) | `0.005` |
| `membrane.porosity` | Porosité des membranes | `0.15` |
| `membrane.charge_cem` | Charge fixe CEM (selon la convention du modèle) | `-1000` |
| `membrane.charge_aem` | Charge fixe AEM (selon la convention du modèle) | `1000` |
| `mesh.nx_regions` | Cellules dans les cinq régions selon x | `[9,6,15,6,9]` |
| `mesh.ny` | Cellules selon y | `18` |
| `newton.tolerance` | Tolérance du solveur | `4e-7` |
| `newton.max_iterations` | Limite d'itérations | `60` |
| `initial_solution` | Chemin facultatif vers une solution `.npz` compatible, ou `null` | `null` |
| `generate_plots` | Générer les figures | `true` |
| `run_validation` | Lancer les vérifications numériques | `true` |

Ordre des cinq régions selon x : canal 1, CEM, canal 2, AEM, canal 3.

Exemples de maillage :

- Rapide : `nx_regions = [3,2,5,2,3]`, `ny = 6` → 15 × 6.
- Intermédiaire : `nx_regions = [9,6,15,6,9]`, `ny = 18` → 45 × 18.
- Plus fin : `nx_regions = [15,10,25,10,15]`, `ny = 30` → 75 × 30.

Ces appellations sont relatives : une étude de convergence en maillage reste nécessaire pour les résultats quantitatifs.

## Organisation

```text
electrodialysis/
├── main.py                 # Point d'entrée
├── config.json             # Paramètres utilisateur
├── solver.py               # Solveur physique et numérique
├── continuation.py         # Script de continuation en tension
├── visualization.py        # Figures et diagnostics
├── validation.py           # Vérifications numériques
├── requirements.txt        # Dépendances Python
├── resultats/              # Sorties des simulations
├── references/             # Références, notamment COMSOL
├── archives/               # Anciens scripts et figures
├── AVANCEMENT_ET_LIMITES.md
└── README.md
```

## Résultats et validation

Pour chaque simulation convergée, le programme peut enregistrer `solution.npz`, des figures PNG et un diagnostic dans le dossier de la simulation. Les vérifications portent notamment sur les bilans de conservation, l'électroneutralité et les conditions de Donnan.

**Important :** un faible résidu et la satisfaction des bilans discrets ne démontrent ni la convergence en maillage ni la validité expérimentale du modèle.

## Limites actuelles

Le solveur est actuellement construit pour deux espèces monovalentes, Na⁺ et Cl⁻. Modifier simplement les noms des ions dans `config.json` ne suffit pas pour modéliser d'autres ions, notamment des espèces multivalentes ou radioactives. Une extension du modèle physique et du code serait nécessaire.

## Conseils de partage

Conserver les fichiers source et `requirements.txt` dans le dépôt partagé. Éviter d'y ajouter les gros dossiers de résultats, les caches Python et les fichiers temporaires, sauf si les résultats sont nécessaires à la reproduction d'une étude.
