import numpy as np
from pathlib import Path

path = Path("resultats_v25_1p5V/steps/v_1.50000000.npz")

with np.load(path, allow_pickle=False) as data:
    c_na = data["c_na"]
    c_cl = data["c_cl"]
    X = data["X"]
    region = data["region"]
    current = data["current_x"]
    flux_na = data["flux_na_x"]
    flux_cl = data["flux_cl_x"]

    voltage = float(data["voltage"])
    residual = float(data["residual_max"])
    converged = bool(data["converged"])

print("\n=== V26 : DIAGNOSTICS À 1,5 V ===")

print("\n1. Convergence")
print(f"Tension : {voltage:.6f} V")
print(f"Convergé : {converged}")
print(f"Résidu : {residual:.3e}")

print("\n2. Concentrations")
for name, c in [("Na+", c_na), ("Cl-", c_cl)]:
    print(
        f"{name} : min={c.min():.6f}, "
        f"max={c.max():.6f} mol/m3"
    )

print("\n3. Électroneutralité")
charge = c_na - c_cl + X[None, :]

print(
    "Erreur absolue maximale :",
    f"{np.max(np.abs(charge)):.6e} mol/m3"
)

for r in np.unique(region):
    mask = region == r
    err = np.max(np.abs(charge[:, mask]))
    print(f"Région {r} : {err:.6e} mol/m3")

print("\n4. Conservation du courant transversal")
mean_current = np.mean(current, axis=0)

print(
    "Courant moyen :",
    f"{mean_current.mean():.6f} A/m2"
)
print(
    "Minimum des moyennes :",
    f"{mean_current.min():.6f} A/m2"
)
print(
    "Maximum des moyennes :",
    f"{mean_current.max():.6f} A/m2"
)

reference = max(np.max(np.abs(mean_current)), 1e-30)
variation = (
    (mean_current.max() - mean_current.min())
    / reference * 100
)

print(f"Variation relative : {variation:.4f} %")

print("\n5. Flux ioniques transversaux")
for name, flux in [
    ("Na+", flux_na),
    ("Cl-", flux_cl),
]:
    print(
        f"{name} : min={np.min(flux):.6e}, "
        f"max={np.max(flux):.6e} mol/(m2.s)"
    )

print("\nDiagnostic terminé.")
