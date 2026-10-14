# Simulation d'électrodialyse 2D

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
