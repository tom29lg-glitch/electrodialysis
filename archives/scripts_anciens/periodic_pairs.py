import numpy as np
from mpi4py import MPI
from dolfinx.io import XDMFFile

comm = MPI.COMM_WORLD

if comm.size != 1:
    raise RuntimeError("Version mono-processus uniquement")

with XDMFFile(comm, "electrodialysis_domains.xdmf", "r") as f:
    mesh = f.read_mesh(name="mesh")

mesh.topology.create_connectivity(1, 0)
mesh.topology.create_connectivity(1, 2)
mesh.topology.create_entities(1)

with XDMFFile(comm, "electrodialysis_boundaries.xdmf", "r") as f:
    tags = f.read_meshtags(mesh, name="mesh_tags")

left = tags.find(3)
right = tags.find(4)

f2v = mesh.topology.connectivity(1, 0)
f2c = mesh.topology.connectivity(1, 2)

def facet_data(facet):
    vertices = f2v.links(int(facet))
    coords = mesh.geometry.x[vertices, :2]
    y = np.sort(coords[:, 1])
    cells = f2c.links(int(facet))

    if len(cells) != 1:
        raise RuntimeError("Facette exterieure non conforme")

    return y, int(cells[0])

left_data = []
right_data = []

for f in left:
    y, cell = facet_data(f)
    left_data.append((float(np.mean(y)), int(f), cell, y))

for f in right:
    y, cell = facet_data(f)
    right_data.append((float(np.mean(y)), int(f), cell, y))

left_data.sort(key=lambda x: x[0])
right_data.sort(key=lambda x: x[0])

if len(left_data) != len(right_data):
    raise RuntimeError("Nombre de facettes periodiques different")

pairs = []
max_error = 0.0

for a, b in zip(left_data, right_data):
    yl, fl, cl, vertices_l = a
    yr, fr, cr, vertices_r = b

    error = float(np.max(np.abs(vertices_l - vertices_r)))
    max_error = max(max_error, error)

    if error > 1e-10:
        raise RuntimeError(
            f"Facettes incompatibles : {fl} et {fr}, erreur={error}"
        )

    pairs.append((fl, fr, cl, cr))

pairs = np.asarray(pairs, dtype=np.int64)

np.savez(
    "periodic_pairs.npz",
    facet_left=pairs[:, 0],
    facet_right=pairs[:, 1],
    cell_left=pairs[:, 2],
    cell_right=pairs[:, 3]
)

print("\nPERIODICITE GEOMETRIQUE COMSOL")
print("=" * 55)
print("Facettes gauche :", len(left))
print("Facettes droite :", len(right))
print("Paires :", len(pairs))
print("Erreur maximale en y :", f"{max_error:.3e}", "m")
print("Correspondances sauvegardees : periodic_pairs.npz")
print("=" * 55)
