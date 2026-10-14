"""
Paramètres du modèle d'électrodialyse COMSOL.
Toutes les grandeurs sont exprimées en unités SI.
"""

# Géométrie (m)
L = 0.01
W_ch = 1.0e-3
W_m = 0.25e-3

# Potentiel appliqué (V)
Vtot = 1.5

# Diffusivités (m²/s)
DNa = 2.5e-9
DCl = 2.0e-9

# Conditions physiques
T = 298.15
v_avg = 0.005
cCl_0 = 500.0
cMem_default = 1000.0
epsilon_mem = 0.15

# Constantes physiques
F = 96485.33212
R = 8.314462618

# Charges des espèces
zNa = +1
zCl = -1

# Étude paramétrique de la charge fixe des membranes (mol/m³)
cMem_values = [0.0, 500.0, 1000.0]

# Permittivite electrique (F/m)
epsilon_0 = 8.8541878128e-12

# Valeurs provisoires pour le modele PNP
epsilon_r_solution = 78.3
epsilon_r_CEM = 78.3
epsilon_r_AEM = 78.3

if __name__ == "__main__":
    for name, value in [
        ("L", L),
        ("W_ch", W_ch),
        ("W_m", W_m),
        ("Vtot", Vtot),
        ("DNa", DNa),
        ("DCl", DCl),
        ("T", T),
        ("v_avg", v_avg),
        ("cCl_0", cCl_0),
        ("cMem_default", cMem_default),
        ("epsilon_mem", epsilon_mem),
    ]:
        print(f"{name:15s} = {value:g}")

    print("\nParamètres chargés correctement.")
