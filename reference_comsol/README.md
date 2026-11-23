# Modèle de référence : électrodialyse de l'eau salée (COMSOL)

Version V29 figée. Elle reproduit le modèle COMSOL « Desalination in an Electrodialysis Cell » (NaCl 0,5 M, 1,5 V, cMem = 0 à 1000 mol/m³) à moins de 0,2 % sur les figures 5 à 9 du PDF.

**Ne pas modifier**, sauf correction de bug. Après toute correction, relancer :

```bash
python validation_comsol.py            # 180 x 60, environ 1 min
python validation_comsol.py --rapide   # 45 x 18, environ 15 s
```

et vérifier que les écarts au PDF COMSOL restent sous 0,2 % (maillage 180 x 60).

Paramètres COMSOL (fichier `../references/electrodialysis.mph`) : L = 10 mm, W_ch = 1 mm, W_m = 0,25 mm, v_avg = 5 mm/s, D_Na = 2,5e-9 et D_Cl = 2e-9 m²/s, c0 = 500 mol/m³, T = 298,15 K, porosité 0,15 (Bruggeman).

Prochaine étape de ce dossier : comparaison directe avec les résultats exportés de COMSOL (mêmes coupes, champs complets, courbe courant-tension, balayages de paramètres, convergence en maillage des deux côtés).
