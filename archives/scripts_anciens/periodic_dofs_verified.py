import numpy as np
import basix.ufl

from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import XDMFFile

comm = MPI.COMM_WORLD

if comm.size != 1:
    raise RuntimeError("Version mono-processus uniquement")

with XDMFFile(comm, "electrodialysis_domains.xdmf", "r") as f:
    mesh = f.read_mesh(name="mesh")

P = basix.ufl.element("DG", mesh.basix_cell(), 1)
W = fem.functionspace(mesh, basix.ufl.mixed_element([P] * 3))

pairs = np.load("periodic_pairs.npz")

mesh.topology.create_connectivity(1, 2)
mesh.topology.create_connectivity(1, 0)

f2c = mesh.topology.connectivity(1, 2)
f2v = mesh.topology.connectivity(1, 0)

all_pairs = []

for component in range(3):

    Vc, mapping = W.sub(component).collapse()
    coords = Vc.tabulate_dof_coordinates()

    for fl, fr in zip(pairs["facet_left"], pairs["facet_right"]):

        cl = int(f2c.links(int(fl))[0])
        cr = int(f2c.links(int(fr))[0])

        vl = f2v.links(int(fl))
        vr = f2v.links(int(fr))

        yl = np.sort(mesh.geometry.x[vl, 1])
        yr = np.sort(mesh.geometry.x[vr, 1])

        dl_scalar = Vc.dofmap.cell_dofs(cl)
        dr_scalar = Vc.dofmap.cell_dofs(cr)

        dl_mixed = W.sub(component).dofmap.cell_dofs(cl)
        dr_mixed = W.sub(component).dofmap.cell_dofs(cr)

        if len(dl_scalar) != len(dl_mixed):
            raise RuntimeError("Incompatibilite des DOFs locaux")

        if len(dr_scalar) != len(dr_mixed):
            raise RuntimeError("Incompatibilite des DOFs locaux")

        for y in yl:

            il = [
                k for k, d in enumerate(dl_scalar)
                if abs(coords[d, 1] - y) < 1e-12
                and abs(coords[d, 0] + 0.00125) < 1e-12
            ]

            ir = [
                k for k, d in enumerate(dr_scalar)
                if abs(coords[d, 1] - y) < 1e-12
                and abs(coords[d, 0] - 0.00125) < 1e-12
            ]

            if len(il) != 1 or len(ir) != 1:
                raise RuntimeError(
                    f"Correspondance introuvable : composante={component}, y={y}"
                )

            all_pairs.append((
                component,
                int(dl_mixed[il[0]]),
                int(dr_mixed[ir[0]])
            ))

all_pairs = np.unique(
    np.asarray(all_pairs, dtype=np.int64),
    axis=0
)

for component, name in enumerate(("Na+", "Cl-", "Potentiel")):
    subset = all_pairs[all_pairs[:, 0] == component]
    print(f"{name:12s} : {len(subset)} paires")

assert len(all_pairs) == 960

for component in range(3):
    subset = all_pairs[all_pairs[:, 0] == component]

    assert len(np.unique(subset[:, 1])) == len(subset)
    assert len(np.unique(subset[:, 2])) == len(subset)

np.save("periodic_dofs.npy", all_pairs)

print("\nPERIODICITE DG1 VERIFIEE")
print("=" * 55)
print("Total : 960 paires")
print("Correspondances construites avec les DOFs locaux")
print("Fichier periodic_dofs.npy mis a jour")
print("=" * 55)
