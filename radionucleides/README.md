# Modèle en évolution : effluents radioactifs

Copie de `../reference_comsol/` (V29) qui évolue vers les conditions des effluents radioactifs.

Ajouts par rapport à la référence :
- `tracer.py` : radionucléide en traces (Cs⁺, Sr²⁺, Co²⁺…) dans le potentiel du sel porteur.
- `solver.py` : mode à courant imposé (`ED_CURRENT` en A/m²) ; la tension devient une inconnue. Sans `ED_CURRENT`, comportement identique à la référence.
- `courbe_iv.py` : courbe courant-tension et comparaison au courant limite de Lévêque. `--dossier` refait les figures sans recalculer.
- `courant_impose.py` : calcul à courant imposé (`--i` en A/m² ou `--frac` du courant limite), avec `--tracer Cs+` en option.
- `recirculation.py` : fonctionnement en recirculation (deux réservoirs, empilement de N paires) : C_sel(t), C_Cs(t), tension, énergie. Modes `courant` (constant, comme Xie), `frac` (fraction du courant limite du moment) et `tension` (constante, comme Sugimoto). Entrées de diluat et de concentrat séparées (`ED_C_INLET_CONC` dans `solver.py`, `c_in_conc` dans `tracer.py`).

Par défaut : D physiques (D_Na = 1,334e-9, D_Cl = 2,032e-9 m²/s), NaCl 17,1 mol/m³ (1000 ppm), géométrie de la cellule COMSOL.

```bash
python courbe_iv.py --c0 17.1                  # ~3 min (180 x 60), --rapide pour un essai
python courant_impose.py --frac 0.6 --tracer Cs+
python recirculation.py --rapide --mode frac --frac 0.6   # conditions de Xie, ~1 min
```

Premiers résultats (cellule COMSOL, cMem = 1000 mol/m³) :
- NaCl 17,1 mol/m³ : courant limite ≈ 51 A/m² (Lévêque : 55), atteint vers 0,6 à 0,8 V par cellule ; 34,2 mol/m³ : ≈ 104 A/m² (Lévêque : 110).
- À 0,6 i_lim (33 A/m², 17,1 mol/m³) : 0,42 V par cellule, 2,0 % du sel éliminé en un passage, Cs⁺ éliminé 1,41 fois plus vite que le sel.

Plan de travail :
1. Conditions réalistes : D physiques, NaCl dilué (≈ 17 et 34 mol/m³), courbe courant-tension, courant limite, mode à courant imposé.
2. Couplage aux réservoirs (recirculation) : C_sel(t), C_Cs(t).
3. Validation : Xie et al. 2022, puis Sugimoto 1978 ; validation croisée de `tracer.py` avec COMSOL (3e espèce).
4. Études de sensibilité et optimisation : courant, débit, rapport de volumes, charge des membranes, épaisseur des canaux.

Recirculation, conditions de Xie (10 paires, 108,5 cm², 5 L / 0,5 L, NaCl 34,2 mol/m³, Cs 3,76e-2 mol/m³, maillage 45 x 18) :
- courant constant 0,6 × i_lim initial : arrêt au courant limite après 15 min, 49 % du Cs retiré ;
- courant à 0,6 × i_lim(t) : saturation à 87,4 % du Cs retiré après ≈ 2,6 h, concentrat enrichi 9,7 fois, 0,46 kWh/m³ (empilement seul) ;
- matière conservée à ~1e-10 près dans tous les cas.
