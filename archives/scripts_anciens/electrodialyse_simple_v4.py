"""V4 : diagnostic du courant sur la solution V3.

A lancer depuis le dossier contenant electrodialyse_simple_v3.py.
La V3 est executee puis ses objets DOLFINx sont reutilises.
"""
from pathlib import Path
import runpy
import numpy as np
import ufl
from dolfinx import fem

root = Path(__file__).resolve().parent
p = runpy.run_path(str(root / "electrodialyse_simple_v3.py"))
domain = p["domain"]
phi = p["phi"]
sigma = p["sigma"]
cell_tags = p["cell_tags"]
sigmas = p["sigmas"]
voltage = p["voltage"]

if domain.comm.size != 1:
    raise RuntimeError("Diagnostic V4 prevu pour un seul processus MPI.")

# Courant ohmique J = -sigma grad(phi). On mesure Jx moyen dans
# chacun des cinq sous-domaines physiques du maillage.
Jx = -sigma * ufl.grad(phi)[0]
dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)

def integrate(expr, marker=None):
    measure = dx if marker is None else dx(marker)
    return float(np.real(fem.assemble_scalar(fem.form(expr * measure))))

names = ("Canal gauche", "CEM", "Canal central", "AEM", "Canal droit")
means = []
areas = []
print("\n=== V4 : DIAGNOSTIC DU COURANT ===")
print("Jx = -sigma * d(phi)/dx ; signe negatif attendu pour tension croissante vers +x")
for marker, name in enumerate(names, start=1):
    area = integrate(1.0, marker)
    if area <= 0:
        raise RuntimeError(f"Aire nulle pour la region {marker}")
    jmean = integrate(Jx, marker) / area
    areas.append(area)
    means.append(jmean)
    print(f"ID {marker} {name:14s} : aire={area:.6e} m2, "
          f"<Jx>={jmean:.8e} A/m2, sigma={sigmas[marker-1]:.3g} S/m")

means = np.asarray(means)
reference = float(np.mean(means))
spread = (float(np.ptp(means)) / max(abs(reference), 1e-30)) * 100
print(f"Ecart relatif max-min des courants moyens : {spread:.4f} %")

# Bilan energetique : P par unite de profondeur (W/m) pour un domaine 2D.
power_per_depth = integrate(sigma * ufl.inner(ufl.grad(phi), ufl.grad(phi)))
print(f"Dissipation Joule integree (par metre de profondeur) : {power_per_depth:.8e} W/m")
print("Note : <Jx> est une moyenne volumique, pas une integration de flux")
print("sur une coupe. L'ecart entre regions est un diagnostic, pas une preuve")
print("de conservation locale. Un controle de flux sur des coupes suivra.")

out = root / "diagnostic_courant_v4.npz"
np.savez(out, region_ids=np.arange(1, 6), region_names=np.array(names),
         mean_Jx_A_m2=means, region_area_m2=np.asarray(areas),
         spread_percent=spread, joule_power_per_depth_W_m=power_per_depth,
         voltage_V=voltage)
print(f"Fichier : {out.name}")
