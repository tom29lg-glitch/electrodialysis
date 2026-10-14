import runpy
import numpy as np
from pathlib import Path
from petsc4py import PETSc

p = runpy.run_path(str(Path("newton_periodique_v8.py").resolve()))

print("\n=== DIAGNOSTIC MATRICE ===")
print("Le solveur v8 a été exécuté.")
print("Pour inspecter la matrice réellement utilisée par Newton,")
print("il faut récupérer A juste avant ksp.solve().")
