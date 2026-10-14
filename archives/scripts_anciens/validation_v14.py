#!/usr/bin/env python3
"""V14: tests de convergence et de bilans du prototype V13.

Execute V13 sur plusieurs grilles, sans changer sa formulation.
La continuite de Donnan n'est PAS prouvee par un ecart entre centres
de cellules : un test rigoureux exigerait des traces aux interfaces.
"""
import os
import sys
import subprocess
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SOLVER = HERE / "electrodialyse_2d_electroneutre_v13.py"
if not SOLVER.exists():
    raise FileNotFoundError(f"Placer V13 a cote de V14 : {SOLVER}")

def parse_grid(s):
    groups, ny = s.split(":")
    nx = np.fromstring(groups, dtype=int, sep=",")
    if len(nx) != 5 or np.any(nx <= 0) or int(ny) < 2:
        raise ValueError(f"Grille incorrecte : {s}")
    return nx, int(ny)

grids = os.getenv("ED_V14_GRIDS", "3,2,5,2,3:6;6,4,10,4,6:12").split(";")
out = HERE / "electrodialyse_electroneutre_v13.npz"
records = []
for k, spec in enumerate(grids, 1):
    nx, ny = parse_grid(spec)
    env = os.environ.copy()
    env["ED_NX"] = ",".join(map(str, nx))
    env["ED_NY"] = str(ny)
    print(f"\n=== Grille {k}/{len(grids)} : Nx={sum(nx)}, Ny={ny} ===", flush=True)
    cp = subprocess.run([sys.executable, str(SOLVER)], cwd=HERE,
                        env=env, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
    print(cp.stdout, flush=True)
    if cp.returncode:
        raise RuntimeError(f"V13 a echoue (code {cp.returncode})")
    with np.load(out) as data:
        x = data["x"].copy()
        y = data["y"].copy()
        region = data["region"].copy()
        na = data["c_na"].copy()
        cl = data["c_cl"].copy()
        phi = data["phi"].copy()
        ix = data["current_x"].copy()
        X = data["X"].copy()
        residual = float(data["residual_max"])
    charge = float(np.max(np.abs(na-cl+X[None, :])))
    # Comparaison a des positions physiques fixes, independantes du maillage.
    interfaces = np.where(region[:-1] != region[1:])[0]
    interface_mean = np.array([ix[:, j].mean() for j in interfaces])
    interface_span = np.array([np.ptp(ix[:, j]) for j in interfaces])
    # Profil a mi-hauteur des faces inter-domaines.
    profile_mid = np.array([np.interp(0.005, y, ix[:, j]) for j in interfaces])
    # Ecart des potentiels electrochimiques aux centres adjacents.
    # NE PAS assimiler a un saut de Donnan a l'interface.
    beta = 96485.33212 / (8.31446261815324 * float(env.get("ED_T", 298.15)))
    mu_na = np.log(na) + beta * phi
    mu_cl = np.log(cl) - beta * phi
    jumps_na = np.array([np.max(np.abs(mu_na[:, j+1]-mu_na[:, j]))
                         for j in interfaces])
    jumps_cl = np.array([np.max(np.abs(mu_cl[:, j+1]-mu_cl[:, j]))
                         for j in interfaces])
    print("i_x moyen aux quatre faces inter-domaines:", interface_mean)
    print("i_x a y=5 mm:", profile_mid)
    print("Max |neutralite|:", charge, "mol/m3")
    print("Residu local normalise maximal:", residual)
    print("Ecart max de mu entre centres voisins (pas traces):",
          "Na", jumps_na, "Cl", jumps_cl)
    records.append((int(sum(nx)),ny,charge,residual,
                    interface_mean,interface_span,profile_mid))
    saved = HERE / f"electrodialyse_v14_grille_{k}.npz"
    np.savez(saved,x=x,y=y,region=region,c_na=na,c_cl=cl,phi=phi,
             current_x=ix,X=X,residual_max=residual,
             interface_current_mean=interface_mean,
             interface_current_mid=profile_mid,
             mu_center_jump_na=jumps_na,mu_center_jump_cl=jumps_cl)
    print("Sauvegarde:", saved.name)

print("\n=== COMPARAISON DES GRILLES ===")
for row in records:
    print(f"Nx={row[0]:3d} Ny={row[1]:3d}: "
          f"neutralite={row[2]:.3e}, residu={row[3]:.3e}, "
          f"courants interfaces={np.array2string(row[4],precision=5)}")
if len(records) >= 2:
    a,b=records[-2:]
    rel=np.abs(b[4]-a[4])/np.maximum(np.abs(b[4]),1e-12)
    print("Variation relative des courants moyens (deux dernieres grilles):",
          np.array2string(100*rel,precision=3),"%")
    if np.max(rel) > 0.05:
        print("ATTENTION: ecart >5%; etude de convergence non terminee.")
    else:
        print("Ecart <5% sur cette grandeur; autres grilles et bilans requis.")
print("V14 ne valide PAS a elle seule les conditions exactes de Donnan,")
print("ni les bilans de frontiere, ni le maillage Gmsh, ni COMSOL.")
