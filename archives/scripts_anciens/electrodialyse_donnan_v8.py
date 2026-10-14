"""V8: equilibre de Donnan ideal aux quatre interfaces solution/membrane.

Modele analytique independant de DOLFINx. Ce n'est PAS un solveur de transport.
Conventions: X = charge fixe signee en mol/m^3 d'equivalents;
CEM: X<0, AEM: X>0; Delta_phi = phi_membrane - phi_solution.
Hypotheses: solution NaCl ideale, ions monovalents, activites=concentrations,
coefficient de partage standard egal a 1, electroneutralite locale.

Exemple:
    ED_C_SALT=500 ED_X_CEM=-1000 ED_X_AEM=1000 python electrodialyse_donnan_v8.py
"""
import os
from pathlib import Path
import numpy as np

F = 96485.33212  # C/mol
R = 8.314462618  # J/(mol K)


def finite_float(name, default):
    val = float(os.getenv(name, str(default)))
    if not np.isfinite(val):
        raise ValueError(f"{name} doit etre fini")
    return val


def donnan_equilibrium(c_s, X, temperature):
    """Retourne c_Na_mem, c_Cl_mem, phi_mem - phi_sol, en SI."""
    if c_s <= 0 or temperature <= 0:
        raise ValueError("Concentration et temperature doivent etre positives")
    # asinh evite la soustraction de nombres presque egaux pour les forts X.
    psi = np.arcsinh(X / (2.0 * c_s))  # F*(phi_mem-phi_sol)/(R*T)
    c_na = c_s * np.exp(-psi)
    c_cl = c_s * np.exp(psi)
    delta_phi = (R * temperature / F) * psi
    return c_na, c_cl, delta_phi


def main():
    c_s = finite_float("ED_C_SALT", 500.0)
    X_cem = finite_float("ED_X_CEM", -1000.0)
    X_aem = finite_float("ED_X_AEM", 1000.0)
    T = finite_float("ED_TEMPERATURE", 298.15)
    if c_s <= 0 or T <= 0:
        raise ValueError("ED_C_SALT et ED_TEMPERATURE doivent etre > 0")
    if X_cem >= 0 or X_aem <= 0:
        raise ValueError("CEM: ED_X_CEM < 0 ; AEM: ED_X_AEM > 0")

    interfaces = [
        ("Canal gauche / CEM", X_cem),
        ("CEM / Canal central", X_cem),
        ("Canal central / AEM", X_aem),
        ("AEM / Canal droit", X_aem),
    ]
    names, c_na_values, c_cl_values, jumps = [], [], [], []
    print("\n=== V8 : EQUILIBRES DE DONNAN AUX 4 INTERFACES ===")
    print(f"T = {T:.2f} K ; c_Na,sol = c_Cl,sol = {c_s:.6g} mol/m3")
    print("Convention: Delta_phi = phi_membrane - phi_solution")
    print("Hypotheses : equilibre ideal local, sans flux ni courant")
    for name, X in interfaces:
        c_na, c_cl, delta_phi = donnan_equilibrium(c_s, X, T)
        electroneutrality_error = c_na - c_cl + X
        product_error = c_na * c_cl / (c_s * c_s) - 1.0
        print(f"\n{name}")
        print(f"  X fixe signe       : {X:+.6f} mol/m3")
        print(f"  c_Na membrane      : {c_na:.8f} mol/m3")
        print(f"  c_Cl membrane      : {c_cl:.8f} mol/m3")
        print(f"  Delta_phi Donnan   : {delta_phi * 1e3:+.8f} mV")
        print(f"  residu neutralite  : {electroneutrality_error:.3e} mol/m3")
        print(f"  residu produit     : {product_error:.3e}")
        names.append(name)
        c_na_values.append(c_na)
        c_cl_values.append(c_cl)
        jumps.append(delta_phi)
        if abs(electroneutrality_error) > 1e-8 * max(1, c_s, abs(X)):
            raise RuntimeError("Echec du controle d'electroneutralite")
        if abs(product_error) > 1e-10:
            raise RuntimeError("Echec du controle du produit de Donnan")

    out = Path(__file__).resolve().parent / "equilibres_donnan_v8.npz"
    np.savez(out, interface=np.array(names), X=np.array([X for _, X in interfaces]),
             c_na_mem=np.array(c_na_values), c_cl_mem=np.array(c_cl_values),
             delta_phi_mem_minus_sol=np.array(jumps), c_s=c_s, T=T)
    print(f"\nFichier : {out.name}")
    print("ATTENTION: sauts locaux d'equilibre, PAS profil global de potentiel.")
    print("Les deux faces d'une membrane peuvent avoir des solutions differentes en regime de transport.")
    print("Ne pas additionner ces quatre valeurs comme une tension de cellule.")


if __name__ == "__main__":
    main()
