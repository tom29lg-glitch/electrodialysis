# Électrodialyse des radionucléides : modèle Python

Le projet contient deux modèles, côte à côte :

| Dossier | Rôle | Règle |
|---|---|---|
| `reference_comsol/` | Adaptation Python du modèle COMSOL « Desalination in an Electrodialysis Cell » (NaCl 0,5 M). Sert à la comparaison avec COMSOL. | **Figé** : on ne modifie que pour corriger un bug, et on relance alors `validation_comsol.py`. |
| `radionucleides/` | Copie de la référence qui évolue vers les effluents radioactifs : conditions réalistes, traceur Cs⁺/Sr²⁺, recirculation, optimisation. | Libre : toute évolution se fait ici. |
| `references/` | Fichier COMSOL `.mph`, courbes du PDF COMSOL numérisées. | Partagé. |
| `archives/` | Anciennes versions et études. | Lecture seule. |

Chaque dossier se lance depuis lui-même et écrit ses résultats dans son propre `resultats/` :

```bash
cd reference_comsol && python validation_comsol.py --rapide
cd radionucleides && python main.py
```

Le guide pas à pas d'installation et d'utilisation est dans `README-2.md`. Les scripts y sont décrits à la racine : depuis la réorganisation, ils se trouvent dans l'un des deux dossiers ci-dessus.

Installation : `python -m pip install -r requirements.txt` (NumPy, SciPy, Matplotlib).
