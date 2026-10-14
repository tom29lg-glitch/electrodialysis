"""Audit hors DOLFINx : contraintes DG1, maillage et géométrie."""
import numpy as np
from pathlib import Path
from periodic_affine import build_affine_map

base = Path(__file__).resolve().parent
pairs = np.load(base / 'periodic_dofs.npy')
faces = np.load(base / 'periodic_pairs.npz')
assert pairs.shape == (960, 3), pairs.shape
assert len(faces['facet_left']) == len(faces['facet_right']) == 160
assert len(np.unique(faces['facet_left'])) == 160
assert len(np.unique(faces['facet_right'])) == 160
for comp in range(3):
    p = pairs[pairs[:, 0] == comp]
    assert p.shape == (320, 3)
    assert len(np.unique(p[:, 1])) == len(np.unique(p[:, 2])) == 320
assert not np.any(pairs[:, 1] == pairs[:, 2])
n = int(pairs[:, 1:].max()) + 1
T, g, ind = build_affine_map(n, str(base / 'periodic_dofs.npy'), 1.5)
q = np.linspace(-0.3, 0.3, len(ind))
w = T @ q + g
errors = w[pairs[:, 2]] - w[pairs[:, 1]] - np.where(pairs[:, 0] == 2, 1.5, 0.)
assert np.max(np.abs(errors)) < 1e-12
print('AUDIT PERIODICITE : OK')
print('Facettes : 160 paires ; DOFs : 960 paires')
print('Erreur max relations affines :', np.max(np.abs(errors)))
print('T :', T.shape, 'nonzeros :', T.nnz)
print('ATTENTION : cet audit ne vérifie pas les flux DG.')
