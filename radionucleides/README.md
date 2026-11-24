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

## Validation contre Xie et al. 2022 (validation_xie.py)

Choix retenu : **le modele s'arrete au courant limite** (`--au-dela arret`, valeur par defaut). Au-dela, Xie continue
jusqu'a 25 V en regime de surlimite (dissociation de l'eau, electroconvection), que le modele electroneutre ne represente pas.

Calcul de reference :

    python validation_xie.py --tag rend082 --rendement 0.82 --rapide

Hypotheses et parametres :
- geometrie de Xie : canal 0,95 mm, membranes 0,30 mm (tableau 1), L = 0,9 m, 10 paires, 108,5 cm2 ;
- sel porteur NaCl + Na2SO4 (2000 mg/L) modelise en NaCl equivalent, 31,2 mol/m3 (meme nombre de transport, 0,39 / 0,40) ;
- 15 H2O transportees par ion (valeur de Xie) ;
- **rendement de courant 0,82** (`--rendement`) : part du courant de l'alimentation qui traverse les cellules
  (courants de fuite de l'empilement, defauts des membranes), calee sur le flux moyen de Cs ;
- **vitesse effective = 2,23 x rendement^3 x v** (calcul automatique) : represente l'espaceur et redonne le courant
  limite mesure (0,78 A), vu par les cellules.

Resultat : 8 essais sur 9 a +-7 % du flux de Cs mesure, ecart moyen 3,5 % (dispersion experimentale ~13 %).
Le partage K du Cs (`--Kcem`), cMem et l'epaisseur des membranes n'expliquent pas le biais : le Cs suit le sel.

L'option `--au-dela plafond` (courant utile au sel plafonne a 0,85 i_lim au-dela de la limite) existe pour essai,
mais n'est pas retenue : elle sous-estime l'elimination en fin d'essai (-8 % en moyenne).

## Charge fixe des membranes de Xie (cMem)

cMem est la charge fixe par volume d'eau de la membrane : cMem = capacite d'echange x rho_eau / taux d'eau
(Elozeiri et al., eq. 1a). Xie (tableau 1) : 1,85 (CEM) et 2,70 (AEM), unite probablement meq/g sec ; taux d'eau
non donne, pris a 0,3 g/g (valeur typique de membranes homogenes, a verifier) :
CEM ~ 6200 mol/m3, AEM ~ 9000 mol/m3 (fourchette 0,2-0,4 g/g : 4600-9300 et 6800-13500).

    python validation_xie.py --tag cmem_iec --rendement 0.82 --cmem 6200 --cmem-aem 9000 --rapide
    python optimisation.py --balayage VR --valeurs 5,10,20,40 --modes frac --ratio-aem 1.46 --base cmem=6200 --suffixe _membranes_xie --objectif 0.999
