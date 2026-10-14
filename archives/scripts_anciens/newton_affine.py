"""Newton amorti sur contraintes affines w=Tq+g (mono-processus).

ATTENTION : ne construit PAS les flux DG périodiques ni les flux de Donnan.
Ce module est un moteur algébrique, non un solveur électrodialyse validé.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from periodic_affine import build_affine_map


def petsc_to_csr(A):
    """Copie une matrice PETSc assemblée vers SciPy CSR (mono-processus)."""
    indptr, indices, values = A.getValuesCSR()
    return csr_matrix((values.copy(), indices.copy(), indptr.copy()), shape=A.getSize())


def solve_affine(w, assemble_residual, assemble_jacobian, *, voltage=1.5,
                 pair_file='periodic_dofs.npy', concentration_dofs=None,
                 max_iterations=35, tolerance=1e-9, verbose=True,
                 gauge_dof=None, gauge_value=0.0):
    """Résout T.T R(Tq+g)=0 avec Newton amorti.

    w : dolfinx.fem.Function (état initial modifié sur place).
    assemble_residual() : retourne le vecteur PETSc du résidu COMPLET.
    assemble_jacobian() : retourne la matrice PETSc du jacobien COMPLET.
    Les deux fonctions doivent assembler à partir de l'état courant w.
    concentration_dofs : indices globaux des deux concentrations, pour
        refuser les essais avec valeurs <=0 (logarithmes de Donnan).
    """
    if w.function_space.mesh.comm.size != 1:
        raise RuntimeError('Ce moteur exige un seul processus MPI')
    n = len(w.x.array)
    T, g, independent = build_affine_map(n, pair_file, voltage)
    # Remove the additive-potential nullspace by fixing one independent phi DOF.
    # The corresponding test equation is removed as well (square reduced system).
    if gauge_dof is None:
        raise ValueError('gauge_dof requis : choisir un DOF indépendant du potentiel')
    where = np.flatnonzero(independent == int(gauge_dof))
    if len(where) != 1:
        raise ValueError('gauge_dof doit être un DOF indépendant (non esclave)')
    gauge_col = int(where[0])
    keep = np.ones(len(independent), dtype=bool)
    keep[gauge_col] = False
    T_full = T
    T = T_full[:, keep].tocsr()
    g = g + np.asarray(T_full[:, gauge_col].toarray()).ravel() * gauge_value
    q = np.asarray(w.x.array[independent[keep]], dtype=float).copy()
    cdofs = (np.asarray(concentration_dofs, dtype=np.int64)
             if concentration_dofs is not None else np.empty(0, dtype=np.int64))

    def set_state(z):
        w.x.array[:] = T @ z + g
        w.x.scatter_forward()

    def reduced_residual(z):
        set_state(z)
        r = assemble_residual()
        arr = r.getArray(readonly=True).copy()
        if len(arr) != n:
            raise RuntimeError(f'Résidu de longueur {len(arr)}, attendu {n}')
        return np.asarray(T.T @ arr).reshape(-1)

    r = reduced_residual(q)
    initial = max(np.linalg.norm(r), 1e-30)
    if verbose:
        print(f'Newton affine : {n} DOFs, {len(q)} inconnues après jauge, '
              f'{n-len(q)} contraintes')
    for iteration in range(max_iterations + 1):
        norm = np.linalg.norm(r)
        if verbose:
            print(f'  iteration {iteration:02d}: ||T^T R|| = {norm:.6e}')
        if norm <= tolerance * max(1.0, initial):
            set_state(q)
            return {'converged': True, 'iterations': iteration,
                    'residual_norm': norm}
        if iteration == max_iterations:
            break
        J = petsc_to_csr(assemble_jacobian())
        if J.shape != (n, n):
            raise RuntimeError(f'Jacobian shape {J.shape}, attendu {(n,n)}')
        Jr = (T.T @ J @ T).tocsr()
        step = np.asarray(spsolve(Jr, -r)).reshape(-1)
        if not np.all(np.isfinite(step)):
            raise RuntimeError('Pas Newton non fini : vérifier la jauge et le jacobien')
        alpha = 1.0
        accepted = False
        while alpha >= 2**-20:
            trial = q + alpha * step
            state = T @ trial + g
            if cdofs.size and np.min(state[cdofs]) <= 0:
                alpha *= 0.5
                continue
            trial_r = reduced_residual(trial)
            if np.all(np.isfinite(trial_r)) and (np.linalg.norm(trial_r)
                    < (1 - 1e-4*alpha)*norm):
                q, r = trial, trial_r
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            set_state(q)
            raise RuntimeError('Recherche linéaire échouée : vérifier la formulation')
    set_state(q)
    raise RuntimeError('Newton affine non convergé')
