"""Contraintes affines périodiques pour les 3 composantes DG1.

Ce module prépare la réduction algébrique, MAIS ne construit pas les flux DG
entre facettes périodiques. Ne pas considérer sa seule utilisation comme
une discrétisation périodique complète.
"""
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix


def build_affine_map(num_dofs, pair_file='periodic_dofs.npy', voltage=1.5):
    """Retourne T, g, independent pour w=T@q+g.

    Convention : colonne 1 = DOF gauche, colonne 2 = DOF droit.
    La relation est w[droit]=w[gauche]+voltage pour phi seulement.
    """
    pairs = np.load(pair_file)
    if pairs.ndim != 2 or pairs.shape[1] != 3:
        raise ValueError('Tableau attendu : (composante, dof_gauche, dof_droit)')
    pairs = np.asarray(pairs, dtype=np.int64)
    if np.any((pairs[:, 0] < 0) | (pairs[:, 0] > 2)):
        raise ValueError('Composantes attendues : 0, 1, 2')
    if np.any(pairs[:, 1:] < 0) or np.any(pairs[:, 1:] >= num_dofs):
        raise ValueError('DOF hors plage')
    left, right = pairs[:, 1], pairs[:, 2]
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise ValueError('DOFs périodiques dupliqués')
    if set(left) & set(right):
        raise ValueError('Chaîne de contraintes non prise en charge')
    if len(set(zip(pairs[:, 0], left, right))) != len(pairs):
        raise ValueError('Paires dupliquées')
    independent = np.setdiff1d(np.arange(num_dofs), right, assume_unique=True)
    col = np.full(num_dofs, -1, dtype=np.int64)
    col[independent] = np.arange(len(independent))
    col[right] = col[left]
    if np.any(col < 0):
        raise RuntimeError('Correspondance incomplète')
    T = coo_matrix((np.ones(num_dofs), (np.arange(num_dofs), col)),
                   shape=(num_dofs, len(independent))).tocsr()
    g = np.zeros(num_dofs)
    g[right[pairs[:, 0] == 2]] = voltage
    return T, g, independent


def reduce_residual(full_residual, T):
    return T.T @ full_residual


def reduce_jacobian(full_jacobian, T):
    """Accepte une matrice scipy.sparse (pas directement PETSc)."""
    return (T.T @ full_jacobian @ T).tocsr()


if __name__ == '__main__':
    pairs = np.load('periodic_dofs.npy')
    n = int(pairs[:, 1:].max()) + 1
    T, g, independent = build_affine_map(n)
    assert T.shape == (n, n-len(pairs))
    q = np.zeros(len(independent))
    w = T @ q + g
    assert np.allclose(w[pairs[:, 2]]-w[pairs[:, 1]],
                       np.where(pairs[:, 0] == 2, 1.5, 0.0))
    print('Réduction affine : OK')
    print('DOFs représentés :', n)
    print('Contraintes :', len(pairs))
    print('Inconnues indépendantes :', len(independent))
    print('ATTENTION : flux DG périodiques et Newton non implémentés.')
