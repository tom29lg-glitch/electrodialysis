"""Assemble (but DO NOT solve) the experimental paired-face DG operator."""
import runpy
from pathlib import Path
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_matrix
from parameters import DNa, DCl, F, R, T
from periodic_flux_v6 import PeriodicNPFlux

p = runpy.run_path(str(Path(__file__).with_name('solver_comsol_membranes_v4.py')))
W, w, mesh = p['W'], p['w'], p['mesh']
F_form = fem.form(p['residual'])
J_form = fem.form(p['jacobian'])

b = assemble_vector(F_form)
J = create_matrix(J_form)
# Cross-boundary DG entries are not in the UFL sparsity pattern.
J.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
J.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, True)
assemble_matrix(J, J_form)
J.assemble()

operator = PeriodicNPFlux(mesh, W, 'periodic_pairs.npz',
                          p['DNa_field'], p['DCl_field'],
                          beta=F/(R*T), voltage=1.5)
print(f'Periodic paired faces: {len(operator.faces)}')
operator.add(w, b, J)
b.assemblyBegin(); b.assemblyEnd()
J.assemblyBegin(); J.assemblyEnd()
print('Residual norm with periodic DG facet terms:', b.norm())
print('Jacobian dimensions:', J.getSize(), 'nonzeros:', J.getInfo()['nz_used'])
print('ASSEMBLAGE V6 EXPERIMENTAL : OK')
print('NO NEWTON RUN. Not validated for physical COMSOL comparison.')
