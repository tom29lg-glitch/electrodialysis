from parameters import v_avg
import numpy as np
import basix.ufl
import ufl

from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import NonlinearProblem

from parameters import (
    DNa, DCl, T, F, R, cCl_0,
    cMem_default
)
from solver_comsol import build_dg_measures

comm = MPI.COMM_WORLD

if comm.size != 1:
    raise RuntimeError("Cette version est mono-processus.")

# --------------------------------------------------
# Maillage
# --------------------------------------------------

with XDMFFile(comm, "electrodialysis_domains.xdmf", "r") as f:
    mesh = f.read_mesh(name="mesh")
    cell_tags = f.read_meshtags(mesh, name="cell_tags")

mesh.topology.create_entities(mesh.topology.dim - 1)

with XDMFFile(comm, "electrodialysis_boundaries.xdmf", "r") as f:
    facet_tags = f.read_meshtags(mesh, name="mesh_tags")

dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
dS = build_dg_measures(mesh, cell_tags, facet_tags)

# --------------------------------------------------
# Espaces
# --------------------------------------------------

element = basix.ufl.element("DG", mesh.basix_cell(), 1)

W = fem.functionspace(
    mesh,
    basix.ufl.mixed_element([element] * 3)
)

w = fem.Function(W, name="solution")
cNa, cCl, phi = ufl.split(w)

vNa, vCl, vPhi = ufl.TestFunctions(W)

Q = fem.functionspace(mesh, ("DG", 0))
X = fem.Function(Q)
DNa_field = fem.Function(Q)
DCl_field = fem.Function(Q)

for cell, marker in zip(cell_tags.indices, cell_tags.values):
    marker = int(marker)
    dofs = Q.dofmap.cell_dofs(int(cell))

    if marker == 2:
        X.x.array[dofs] = -cMem_default
    elif marker == 4:
        X.x.array[dofs] = cMem_default
    else:
        X.x.array[dofs] = 0.0

    factor = 0.15 if marker in (2, 4) else 1.0
    DNa_field.x.array[dofs] = factor * DNa
    DCl_field.x.array[dofs] = factor * DCl

for field in (X, DNa_field, DCl_field):
    field.x.scatter_forward()

# --------------------------------------------------
# Initialisation de Donnan
# --------------------------------------------------

for cell, marker in zip(cell_tags.indices, cell_tags.values):
    marker = int(marker)

    if marker == 2:
        charge = -cMem_default
    elif marker == 4:
        charge = cMem_default
    else:
        charge = 0.0

    cna = (-charge + np.sqrt(charge**2 + 4*cCl_0**2)) / 2
    ccl = ( charge + np.sqrt(charge**2 + 4*cCl_0**2)) / 2
    potential = -(R*T/F) * np.log(cna/cCl_0)

    # Affectation par sous-espace DG1
    for component, value in enumerate((cna, ccl, potential)):
        dofs = W.sub(component).dofmap.cell_dofs(int(cell))
        w.x.array[dofs] = value

w.x.scatter_forward()

# --------------------------------------------------
# Residus : equations volumiques uniquement
# --------------------------------------------------


# CONVECTION ELECTRODIALYSE
V0 = fem.functionspace(mesh, ("DG", 0))
uy = fem.Function(V0, name="velocity_y")

# Centres et demi-largeurs des canaux
channel_geometry = {
    1: (-0.001, 0.00025),
    3: ( 0.000, 0.00050),
    5: ( 0.001, 0.00025),
}

for cell, marker in zip(cell_tags.indices, cell_tags.values):
    cell = int(cell)
    marker = int(marker)
    dofs = V0.dofmap.cell_dofs(cell)

    if marker in channel_geometry:
        center, half_width = channel_geometry[marker]
        points = mesh.geometry.x[mesh.geometry.dofmap[cell]]
        x = float(np.mean(points[:, 0]))
        xi = (x - center) / half_width
        uy.x.array[dofs] = (
            1.5 * v_avg * max(0.0, 1.0 - xi**2)
        )
    else:
        uy.x.array[dofs] = 0.0

uy.x.scatter_forward()
velocity = ufl.as_vector((0.0, uy))

NNa = -DNa_field * ufl.grad(cNa) - DNa_field*F/(R*T)*cNa*ufl.grad(phi) + velocity*cNa
NCl = -DCl_field * ufl.grad(cCl) + DCl_field*F/(R*T)*cCl*ufl.grad(phi) + velocity*cCl

residual = (
    -ufl.dot(NNa, ufl.grad(vNa))*dx
    -ufl.dot(NCl, ufl.grad(vCl))*dx
    + (cNa-cCl+X)*vPhi*dx
)

# --------------------------------------------------
# Flux numeriques DG aux interfaces membranaires
# --------------------------------------------------

beta = F / (R*T)
n = ufl.FacetNormal(mesh)
h = ufl.CellDiameter(mesh)
gamma = 10.0

def donnan_interface(c, v, D, z, marker):
    """Flux NP conservatif sur interface, avec penalisation de Donnan.

    Le saut de potentiel electrochimique est nul a l'equilibre.
    Le terme moyen fournit la consistance lorsque le flux traverse
    une membrane hors equilibre. Les traces des deux cotes partagent
    un seul flux numerique, assurant la conservation locale.
    """
    psi = ufl.ln(c / cCl_0) + z * beta * phi
    N = -D * ufl.grad(c) - z * beta * D * c * ufl.grad(phi)
    nplus = n("+")
    normal_flux = ufl.dot(ufl.avg(N), nplus)
    # Flux dissipatif sur le saut electrochimique (unites mol/m2/s).
    conductance = gamma * ufl.avg(D * c) / ufl.avg(h)
    normal_flux -= conductance * ufl.jump(psi)
    return normal_flux * ufl.jump(v) * dS(marker)

for marker in (5, 6, 7, 8):
    residual += donnan_interface(
        cNa, vNa, DNa_field, +1, marker
    )
    residual += donnan_interface(
        cCl, vCl, DCl_field, -1, marker
    )

# --------------------------------------------------
# Flux DG ordinaires : diffusion et migration
# --------------------------------------------------

def ordinary_dg(c, v, D, z):

    # Flux Nernst-Planck sans convection
    N = (
        -D * ufl.grad(c)
        -z * beta * D * c * ufl.grad(phi)
    )

    # Flux physique moyen
    consistency = (
        ufl.dot(ufl.avg(N), n("+"))
        * ufl.jump(v)
        * dS(10)
    )

    # Terme adjoint de diffusion
    symmetry = (
        -ufl.avg(D)
        * ufl.dot(
            ufl.avg(ufl.grad(v)),
            ufl.jump(c, n)
        )
        * dS(10)
    )

    # Penalisation SIPG
    penalty = (
        gamma * ufl.avg(D) / ufl.avg(h)
        * ufl.dot(
            ufl.jump(c, n),
            ufl.jump(v, n)
        )
        * dS(10)
    )


    # Flux convectif upwind sur les facettes ordinaires
    un = ufl.avg(uy) * n("+")[1]

    cup = ufl.conditional(
        ufl.ge(un, 0.0),
        c("+"),
        c("-")
    )

    convection = un * cup * ufl.jump(v) * dS(10)

    return consistency + symmetry + penalty + convection


residual += ordinary_dg(cNa, vNa, DNa_field, +1)
residual += ordinary_dg(cCl, vCl, DCl_field, -1)



# --------------------------------------------------
# Conditions convectives : entree et sortie
# --------------------------------------------------

ds = ufl.Measure(
    "ds",
    domain=mesh,
    subdomain_data=facet_tags
)

for c, v in ((cNa, vNa), (cCl, vCl)):
    # Entree : concentration imposee dans le flux entrant
    residual += -uy * cCl_0 * v * ds(1)

    # Sortie : flux convectif sortant
    residual += uy * c * v * ds(2)

jacobian = ufl.derivative(
    residual, w, ufl.TrialFunction(W)
)

# --------------------------------------------------
# Verification uniquement
# --------------------------------------------------

fem.form(residual)
fem.form(jacobian)

print("\nSOLVEUR NON LINEAIRE : STRUCTURE PREPAREE")
print("DOFs :", W.dofmap.index_map.size_global * W.dofmap.index_map_bs)
print("Initialisation Donnan : OK")
print("Equations volumiques : OK")
print("Jacobien : OK")
print("Flux membrane : moyenne NP + penalisation Donnan (experimental)")
print("Flux periodiques : NON ASSEMBLES ; ne pas interpreter physiquement")
print("Aucune resolution effectuee.")

# --------------------------------------------------
# Diagnostic de l'etat initial
# --------------------------------------------------

print("\nVERIFICATION DE L'ETAT INITIAL")
print("=" * 55)

for component, name in enumerate(("Na+", "Cl-", "Potentiel")):
    values = w.x.array[
        W.sub(component).dofmap.list.flatten()
    ]

    print(
        f"{name:12s} : "
        f"min = {np.min(values):.8f}, "
        f"max = {np.max(values):.8f}"
    )

print("=" * 55)

# --------------------------------------------------
# Assemblage du residu initial
# --------------------------------------------------

from dolfinx.fem.petsc import assemble_vector

print("\nASSEMBLAGE DU RESIDU INITIAL")
print("=" * 55)

F_form = fem.form(residual)

b = assemble_vector(F_form)
b.ghostUpdate(
    addv=__import__("petsc4py").PETSc.InsertMode.ADD_VALUES,
    mode=__import__("petsc4py").PETSc.ScatterMode.REVERSE
)

print(f"Norme L2 du residu : {b.norm():.8e}")
print(f"Nombre de DOFs : {W.dofmap.index_map.size_global * W.dofmap.index_map_bs}")
print("=" * 55)
print("Residuel incomplet : flux periodiques absents.")
