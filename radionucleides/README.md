# Modèle en évolution : effluents radioactifs

Copie de `../reference_comsol/` (V29) qui évolue vers les conditions des effluents radioactifs.

Ajouts par rapport à la référence :
- `tracer.py` : radionucléide en traces (Cs⁺, Sr²⁺, Co²⁺…) dans le potentiel du sel porteur.

Plan de travail :
1. Conditions réalistes : D physiques, NaCl dilué (≈ 17 et 34 mol/m³), courbe courant-tension, courant limite, mode à courant imposé.
2. Couplage aux réservoirs (recirculation) : C_sel(t), C_Cs(t).
3. Validation : Xie et al. 2022, puis Sugimoto 1978 ; validation croisée de `tracer.py` avec COMSOL (3e espèce).
4. Études de sensibilité et optimisation : courant, débit, rapport de volumes, charge des membranes, épaisseur des canaux.
