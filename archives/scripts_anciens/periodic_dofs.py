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

element = basix.ufl.element("DG", mesh.basix_cell(), 1)

W = fem.functionspace(
    mesh,
    basix.ufl.mixed_element([element] * 3)
)

pairs = np.load("periodic_pairs.npz")

left_facets = pairs["facet_left"]
right_facets = pairs["facet_right"]

mesh.topology.create_connectivity(1, 0)
mesh.topology.create_connectivity(2, 0)
mesh.topology.create_connectivity(2, 1)

f2v = mesh.topology.connectivity(1, 0)
c2v = mesh.topology.connectivity(2, 0)

# DG1 Lagrange : DOFs associes aux sommets locaux.
# On verifie explicitement cette hypothese.
if mesh.topology.cell_type.name != "triangle":
    raise RuntimeError("Ce script attend un maillage triangulaire")

dof_pairs = []

for fl, fr in zip(left_facets, right_facets):

    vl = f2v.links(int(fl))
    vr = f2v.links(int(fr))

    yl = mesh.geometry.x[vl, 1]
    yr = mesh.geometry.x[vr, 1]

    vl = vl[np.argsort(yl)]
    vr = vr[np.argsort(yr)]

    for vertex_l, vertex_r in zip(vl, vr):

        # Cellule adjacente de chaque facette
        mesh.topology.create_connectivity(1, 2)
        f2c = mesh.topology.connectivity(1, 2)

        cl = int(f2c.links(int(fl))[0])
        cr = int(f2c.links(int(fr))[0])

        local_vertices_l = c2v.links(cl)
        local_vertices_r = c2v.links(cr)

        il = int(np.where(local_vertices_l == vertex_l)[0][0])
        ir = int(np.where(local_vertices_r == vertex_r)[0][0])

        # Indices locaux des DOFs dans les sous-espaces mixtes
        for component in range(3):
            dl = W.sub(component).dofmap.cell_dofs(cl)[il]
            dr = W.sub(component).dofmap.cell_dofs(cr)[ir]

            dof_pairs.append((component, dl, dr))

dof_pairs = np.asarray(dof_pairs, dtype=np.int64)

# Supprimer les doublons (sommets partages entre facettes)
dof_pairs = np.unique(dof_pairs, axis=0)

np.save("periodic_dofs.npy", dof_pairs)

print("\nCORRESPONDANCES DG1 PERIODIQUES")
print("=" * 55)

for component, name in enumerate(("Na+", "Cl-", "Potentiel")):
    subset = dof_pairs[dof_pairs[:, 0] == component]

    print(f"{name:12s} : {len(subset)} paires de DOFs")

print("Total :", len(dof_pairs))
print("Fichier : periodic_dofs.npy")
print("=" * 55)
