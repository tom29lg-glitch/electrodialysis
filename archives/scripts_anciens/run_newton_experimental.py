"""Essai Newton sur le résidu DG EXISTANT (NON validé physiquement).

IMPORTANT : les facettes périodiques ne sont pas couplées par flux DG.
Ce script sert uniquement à déboguer l'assemblage PETSc et les contraintes.
Il n'est PAS une reproduction COMSOL et ses résultats ne doivent pas être interprétés.
"""
import os
import numpy as np
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector, assemble_matrix

# Le module existant initialise w, residual, jacobian et affiche son diagnostic.
import solver_comsol_exact as model
from newton_affine import solve_affine

F_form = fem.form(model.residual)
J_form = fem.form(model.jacobian)
W = model.W
w = model.w

# Indices des concentrations, dans la numérotation du champ mixte.
concentration_dofs = np.unique(np.concatenate([
    W.sub(0).dofmap.list.flatten(),
    W.sub(1).dofmap.list.flatten(),
])).astype(np.int64)


def assemble_residual():
    b = assemble_vector(F_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    return b


def assemble_jacobian():
    A = assemble_matrix(J_form)
    A.assemble()
    return A


if __name__ == '__main__':
    if os.environ.get('ALLOW_INCOMPLETE_DG_NEWTON') != '1':
        raise SystemExit(
            'ARRET DE SECURITE : flux DG periodiques et flux membranaires incomplets.\n'
            'Pour un essai de debogage uniquement :\n'
            '  ALLOW_INCOMPLETE_DG_NEWTON=1 python run_newton_experimental.py\n'
            'NE PAS interpreter les resultats comme une simulation COMSOL.'
        )
    print('ATTENTION : test numerique incomplet, pas de resultat physique valide.')
    result = solve_affine(w, assemble_residual, assemble_jacobian,
                          voltage=1.5,
                          concentration_dofs=concentration_dofs,
                          gauge_dof=int(np.setdiff1d(
                              W.sub(2).dofmap.list.flatten(),
                              np.load('periodic_dofs.npy')[:, 2]
                          )[0]))
    print(result)
