# Simulation d'électrodialyse 2D

> **Réorganisation (novembre 2026)** : les scripts sont maintenant dans `reference_comsol/` (modèle COMSOL figé) et `radionucleides/` (modèle qui évolue). Pour suivre ce guide, se placer d'abord dans l'un de ces dossiers (`cd radionucleides`). Voir `README.md`.

## 1. À quoi sert ce projet ?

Ce projet permet de simuler le fonctionnement d'une cellule d'électrodialyse avec Python.

L'électrodialyse est une technique qui utilise un courant électrique et des membranes pour déplacer des ions dans de l'eau.

Dans notre simulation, nous étudions deux ions :
- **Na⁺** : ion sodium.
- **Cl⁻** : ion chlorure.

Notre cellule contient **trois canaux et deux membranes** :

- **CEM** : membrane qui laisse principalement passer les ions positifs.
- **AEM** : membrane qui laisse principalement passer les ions négatifs.

Le programme calcule notamment :
- La concentration des ions dans la cellule.
- Le potentiel électrique.
- Le déplacement des ions à travers les membranes.
- Le courant électrique.

Le but est de comprendre comment différents paramètres, comme la tension électrique ou les propriétés des membranes, influencent le fonctionnement de la cellule.

## 2. Comment installer le projet ?

### Étape 1 — Télécharger le projet

Ouvrir le dépôt GitHub et cliquer sur :

**Code → Download ZIP**

Décompresser ensuite le fichier ZIP sur l'ordinateur.

Il est également possible de récupérer le projet avec Git :

```bash
git clone https://github.com/tom29lg-glitch/electrodialysis.git
```

### Étape 2 — Installer Python

Il faut avoir Python installé sur son ordinateur.

Pour vérifier, ouvrir un terminal et écrire :

```bash
python --version
```

Si Python n'est pas installé, le télécharger depuis https://www.python.org/downloads/.

Sur certains ordinateurs, notamment les Mac, la commande est `python3` au lieu de `python`.

### Étape 3 — Installer les bibliothèques nécessaires

Ouvrir le dossier du projet dans VS Code, puis ouvrir un terminal.

Exécuter :

```bash
python -m pip install -r requirements.txt
```

Cette commande installe les bibliothèques nécessaires au fonctionnement du programme, notamment NumPy, SciPy et Matplotlib.

Cette installation est normalement nécessaire une seule fois.

## 3. Comment lancer une simulation ?

Il suffit de suivre trois étapes.

**Étape 1 — Ouvrir `config.json`**

Ce fichier contient tous les paramètres que l'on peut modifier.

Par exemple :

```json
"simulation_name": "test_1p5V",
"voltage": 1.5,
"concentration_inlet": 500,
"temperature": 298.15,
"velocity": 0.005
```

Ces paramètres correspondent à :
- `simulation_name` : nom donné à la simulation.
- `voltage` : tension électrique appliquée, en volts.
- `concentration_inlet` : concentration des ions à l'entrée.
- `temperature` : température en kelvins.
- `velocity` : vitesse moyenne de l'eau dans les canaux.

Pour une première simulation, il est conseillé de conserver les paramètres déjà présents dans le fichier.

**Étape 2 — Lancer le programme**

Dans le terminal de VS Code, depuis le dossier du projet :

```bash
python main.py
```

**Étape 3 — Attendre la fin du calcul**

Le programme effectue automatiquement les calculs nécessaires.

Pour faciliter la convergence, il peut commencer avec une faible tension, puis l'augmenter progressivement jusqu'à atteindre la tension demandée.

Lorsque la simulation est terminée, le programme demande :

```text
Voulez-vous sauvegarder les résultats ? [O/n] :
```

- Appuyer sur **Entrée** ou écrire `o` pour conserver les résultats.
- Écrire `n` pour supprimer les résultats de cette simulation.

## 4. Où trouver les résultats ?

Les résultats sont enregistrés automatiquement dans le dossier `resultats/`.

Chaque simulation possède son propre sous-dossier.

Par exemple :

```text
resultats/
└── test_1p5V_45x18_2026-10-15_14h30m00s/
    ├── config.json
    ├── solution.npz
    ├── export.csv
    ├── calcul.log
    └── figures/
```

**À quoi servent ces fichiers ?**

| Fichier | Utilité |
|---|---|
| `config.json` | Conserve les paramètres utilisés pour la simulation. |
| `solution.npz` | Contient les résultats numériques complets. |
| `export.csv` | Permet de consulter les résultats dans Excel ou Python. |
| `calcul.log` | Contient des informations sur le déroulement du calcul. |
| `figures/` | Contient les graphiques générés automatiquement. |

Les figures permettent notamment de visualiser les concentrations de Na⁺ et Cl⁻, le potentiel électrique et les profils dans les canaux.

## 5. Quels paramètres peut-on modifier ?

Tous les paramètres principaux se trouvent dans `config.json`.

| Paramètre | À quoi sert-il ? |
|---|---|
| `voltage` | Modifier la tension appliquée. |
| `ions.Na+.diffusion` | Modifier le coefficient de diffusion du sodium. |
| `ions.Cl-.diffusion` | Modifier le coefficient de diffusion du chlorure. |
| `concentration_inlet` | Modifier la concentration à l'entrée. |
| `temperature` | Modifier la température. |
| `velocity` | Modifier la vitesse moyenne de l'eau. |
| `membrane.porosity` | Modifier la porosité des membranes. |
| `membrane.charge_cem` | Modifier la charge fixe de la membrane CEM. |
| `membrane.charge_aem` | Modifier la charge fixe de la membrane AEM. |
| `mesh` | Modifier la précision du maillage. |
| `generate_plots` | Activer ou désactiver les graphiques. |
| `generate_csv` | Activer ou désactiver l'export CSV. |
| `run_validation` | Activer ou désactiver les vérifications numériques. |

Pour les paramètres physiques, il faut respecter les unités indiquées dans le fichier de configuration.

**Attention :** le modèle actuel est conçu pour Na⁺ et Cl⁻. Il n'est pas encore possible de simuler correctement d'autres ions en changeant uniquement leur nom.

## 6. Comment choisir le maillage ?

Le maillage correspond au nombre de petites cellules utilisées pour effectuer les calculs.

Plus le maillage est fin, plus la simulation peut être précise, mais plus elle demande de temps de calcul.

Dans `config.json`, les paramètres sont :

```json
"nx_regions": [9, 6, 15, 6, 9],
"ny": 18
```

Les cinq nombres de `nx_regions` correspondent aux cinq régions du modèle :

**Canal 1 → CEM → Canal 2 → AEM → Canal 3**

Voici trois exemples :

| Maillage | Paramètres | Utilisation |
|---|---|---|
| 15 × 6 | `[3,2,5,2,3]`, `ny=6` | Tests très rapides |
| 45 × 18 | `[9,6,15,6,9]`, `ny=18` | Simulations courantes |
| 75 × 30 | `[15,10,25,10,15]`, `ny=30` | Calculs plus détaillés |

**Pour commencer, utiliser le maillage 45 × 18.**

Un maillage plus fin ne garantit pas automatiquement un résultat exact : il faut comparer plusieurs maillages pour vérifier la convergence.

## 7. À quoi servent les différents scripts ?

| Fichier | Rôle |
|---|---|
| `main.py` | Lance et organise toute la simulation. |
| `config.json` | Contient les paramètres à modifier. |
| `solver.py` | Effectue les calculs physiques et numériques. |
| `visualization.py` | Génère les graphiques. |
| `validation.py` | Vérifie certains résultats numériques. |
| `export_csv.py` | Exporte les résultats dans un fichier CSV. |
| `continuation.py` | Ancien script de continuation en tension, conservé séparément. |
| `requirements.txt` | Liste les bibliothèques Python nécessaires. |

**Pour une utilisation normale, il suffit de modifier `config.json` et de lancer `main.py`.** Il n'est pas nécessaire de modifier les autres scripts.

## 8. Comment travailler à plusieurs avec GitHub ?

Pour récupérer les dernières modifications du projet :

```bash
git pull
```

Pour envoyer ses propres modifications, il est conseillé de créer une branche et de proposer une Pull Request sur GitHub.

Les fichiers de résultats sont normalement conservés sur chaque ordinateur et ne sont pas envoyés automatiquement sur GitHub.

## 9. Quelles sont les limites du modèle ?

Le programme utilise une méthode numérique appelée **volumes finis** pour résoudre les équations de transport des ions.

Il prend en compte notamment la diffusion, la migration électrique, le transport lié à l'écoulement et les propriétés des membranes.

Cependant :
- Le modèle est actuellement limité à deux ions monovalents : Na⁺ et Cl⁻.
- Il n'est pas encore adapté à la simulation d'ions radioactifs ou multivalents.
- Les résultats doivent être vérifiés, notamment en comparant plusieurs maillages.
- La convergence numérique ne garantit pas à elle seule que le modèle représente parfaitement une expérience réelle.

## 10. Évolutions V29 : géométrie COMSOL, radionucléides, validation

### Géométrie COMSOL et radionucléides en traces

**Géométrie alignée sur `references/electrodialysis.mph`** (L = 10 mm, W_ch = 1 mm, W_m = 0,25 mm) :

- largeurs `[W_ch/2, W_m, W_ch, W_m, W_ch/2]` : les deux canaux extérieurs sont des demi-canaux qui forment un seul canal de concentrat à travers la frontière périodique ;
- `velocity_profile = "comsol"` : vel = 3·v_avg·(1 − (t/(W_ch/2))²), comme COMSOL (moyenne réelle 2·v_avg = 10 mm/s, maximum sur la frontière périodique dans les demi-canaux). `"poiseuille"` redonne l'ancien profil V28 ;
- `mesh.grading` : rapport grande/petite maille par région (« element ratio » COMSOL), resserré vers les membranes. COMSOL : `[10, 2, 10, 2, 10]` avec `nx_regions = [20, 20, 100, 20, 20]`, `ny = 100`.

Avec `ED_WIDTHS=0.001,0.0002,0.001,0.0002,0.001`, `ED_VELOCITY_PROFILE=poiseuille` et `grading = 1`, le solveur redonne exactement les résultats V28.

**Radionucléide en traces (`tracer.py`)** : Nernst–Planck pour une espèce en traces (Cs⁺, Sr²⁺, Co²⁺…) dans le potentiel déjà calculé, même discrétisation que `solver.py`, Donnan généralisé avec un coefficient de partage K par membrane (K = 1 : Donnan idéal).

```bash
python tracer.py resultats/<dossier>/solution.npz --species Cs+
python tracer.py resultats/<dossier>/solution.npz --species Sr2+ --c-in 0.0376 --K-cem 1
python tracer.py resultats/<dossier>/solution.npz --z 2 --D 0.79e-9   # espèce quelconque
```

Sorties : `tracer_<espece>.npz` et `.json` (élimination en un passage, rapport élimination traceur / sel, nombre de transport dans la CEM, facteur de séparation par rapport à Na⁺).

Test de cohérence : `--species Na+ --D 2.5e-9 --c-in 500` redonne c_Na du solveur (écart relatif ≈ 2·10⁻¹⁰).

### Validation contre COMSOL (`comparaison_comsol.py`)

Les figures 5 à 9 du PDF COMSOL sont numérisées dans `references/comsol_figures_numerisees.npz` (script : `references/numerisation_figures_comsol.py`). `comparaison_comsol.py` trace les résultats Python sur les mêmes coupes et écrit un tableau d'écarts.

1. Dans `config.json` : `nx_regions = [20, 20, 100, 20, 20]`, `ny = 60`, `grading = [10, 2, 10, 2, 10]`, D de COMSOL (2,5e-9 et 2e-9), tension 1,5 V.
2. Lancer `main.py` trois fois, avec `charge_cem` / `charge_aem` = −1000 / 1000, −500 / 500 et 0 / 0.
3. Comparer :

```bash
python comparaison_comsol.py --c1000 resultats/<dossier_1000>/solution.npz \
    --c500 resultats/<dossier_500>/solution.npz --c0 resultats/<dossier_0>/solution.npz
```

Résultat sur le maillage 180 × 60 (`resultats/comparaison_comsol_180x60/`) : potentiel, concentrations, flux et courant le long de l'AEM à moins de 0,2 % des valeurs COMSOL lues sur les figures (précision de lecture ≈ 0,3 % de l'axe).

### Limites de la V29

Le solveur principal reste construit pour deux espèces monovalentes, Na⁺ et Cl⁻ (sel porteur). Les radionucléides sont traités en traces par `tracer.py` : valable tant que leur concentration reste très inférieure à celle du sel. Un sel porteur multi-ions (Na₂SO₄, Ca(NO₃)₂) ou un radionucléide majoritaire demanderait une extension du solveur. Le fonctionnement en recirculation (évolution temporelle des réservoirs) n'est pas encore modélisé.
