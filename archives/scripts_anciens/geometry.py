
"""
geometry.py
-----------
Géométrie 2D de la cellule d'électrodialyse.

Domaines :
    1 : canal gauche
    2 : membrane cationique (CEM)
    3 : canal central
    4 : membrane anionique (AEM)
    5 : canal droit

Frontières :
    1 : entrée (y = 0)
    2 : sortie (y = L)
    3 : frontière extérieure gauche
    4 : frontière extérieure droite
    5 : canal gauche / CEM
    6 : CEM / canal central
    7 : canal central / AEM
    8 : AEM / canal droit
"""

import gmsh
import numpy as np
from mpi4py import MPI

from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as gmshio
from dolfinx.mesh import (
    locate_entities_boundary,
    meshtags,
)

from parameters import L, W_ch, W_m


def create_geometry():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # ---------------------------------------------------------
    # 1. Dimensions et limites des domaines
    # ---------------------------------------------------------

    widths = [
        W_ch / 2,
        W_m,
        W_ch,
        W_m,
        W_ch / 2,
    ]

    x_min = -(W_ch + W_m)
    x_max = +(W_ch + W_m)

    x_edges = [x_min]
    for width in widths:
        x_edges.append(x_edges[-1] + width)

    domain_names = [
        "canal_gauche",
        "membrane_cationique",
        "canal_central",
        "membrane_anionique",
        "canal_droit",
    ]

    # Positions des interfaces internes
    interface_x = {
        5: x_edges[1],
        6: x_edges[2],
        7: x_edges[3],
        8: x_edges[4],
    }

    # ---------------------------------------------------------
    # 2. Géométrie et maillage Gmsh
    # ---------------------------------------------------------

    gmsh.initialize()
    gmsh.model.add("electrodialysis")

    try:
        gmsh.option.setNumber("General.Terminal", 1)

        surfaces = []

        for i, width in enumerate(widths):
            tag = gmsh.model.occ.addRectangle(
                x_edges[i],
                0.0,
                0.0,
                width,
                L,
            )
            surfaces.append(tag)

        # Fusionner les surfaces adjacentes en conservant
        # leurs interfaces communes.
        gmsh.model.occ.fragment(
            [(2, tag) for tag in surfaces],
            [],
        )
        gmsh.model.occ.synchronize()

        # Retrouver les cinq surfaces et les classer
        # de gauche à droite.
        surface_info = []

        for dim, tag in gmsh.model.getEntities(2):
            bbox = gmsh.model.getBoundingBox(dim, tag)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox
            center_x = (xmin + xmax) / 2
            surface_info.append((center_x, tag))

        surface_info.sort(key=lambda item: item[0])

        if len(surface_info) != 5:
            raise RuntimeError(
                f"5 domaines attendus, "
                f"{len(surface_info)} détectés."
            )

        for physical_id, ((_, surface_tag), name) in enumerate(
            zip(surface_info, domain_names),
            start=1,
        ):
            group = gmsh.model.addPhysicalGroup(
                2, [surface_tag], physical_id
            )
            gmsh.model.setPhysicalName(2, group, name)

        # Taille des éléments
        mesh_size = min(widths) / 4

        gmsh.option.setNumber(
            "Mesh.MeshSizeMin", mesh_size / 2
        )
        gmsh.option.setNumber(
            "Mesh.MeshSizeMax", mesh_size
        )

        gmsh.model.mesh.generate(2)

        mesh_data = gmshio.model_to_mesh(
            gmsh.model,
            comm,
            0,
            gdim=2,
        )

        domain = mesh_data.mesh
        cell_tags = mesh_data.cell_tags

    finally:
        gmsh.finalize()

    # ---------------------------------------------------------
    # 3. Connectivité du maillage
    # ---------------------------------------------------------

    tdim = domain.topology.dim
    fdim = tdim - 1

    domain.topology.create_entities(fdim)
    domain.topology.create_connectivity(fdim, tdim)

    # Pour identifier les interfaces, il faut aussi connaître
    # les cellules situées de chaque côté de chaque arête.
    domain.topology.create_connectivity(fdim, tdim)

    # ---------------------------------------------------------
    # 4. Frontières extérieures
    # ---------------------------------------------------------

    tol = 1e-10

    inlets = locate_entities_boundary(
        domain,
        fdim,
        lambda x: np.isclose(x[1], 0.0, atol=tol),
    )

    outlets = locate_entities_boundary(
        domain,
        fdim,
        lambda x: np.isclose(x[1], L, atol=tol),
    )

    periodic_left = locate_entities_boundary(
        domain,
        fdim,
        lambda x: np.isclose(x[0], x_min, atol=tol),
    )

    periodic_right = locate_entities_boundary(
        domain,
        fdim,
        lambda x: np.isclose(x[0], x_max, atol=tol),
    )

    # ---------------------------------------------------------
    # 5. Interfaces internes entre matériaux
    # ---------------------------------------------------------
    #
    # Une interface interne n'est PAS une frontière extérieure.
    # On identifie donc les arêtes qui :
    #   - sont verticales ;
    #   - sont situées à une abscisse d'interface ;
    #   - possèdent deux cellules adjacentes.
    #
    # Les arêtes sont ensuite associées à leurs deux domaines.
    # ---------------------------------------------------------

    domain.topology.create_connectivity(fdim, tdim)
    facet_to_cell = domain.topology.connectivity(fdim, tdim)

    # Connectivité cellule -> domaine physique
    cell_to_marker = {}

    if cell_tags is not None:
        for cell, marker in zip(
            cell_tags.indices,
            cell_tags.values,
        ):
            cell_to_marker[int(cell)] = int(marker)

    # Coordonnées des sommets de chaque arête
    domain.topology.create_connectivity(fdim, 0)
    facet_to_vertex = domain.topology.connectivity(fdim, 0)

    interface_facets = {marker: [] for marker in interface_x}

    geometry_dofmap = domain.geometry.dofmaps[0]

    for facet in range(
        domain.topology.index_map(fdim).size_local
    ):
        adjacent_cells = facet_to_cell.links(facet)

        # Une interface interne doit avoir deux cellules
        if len(adjacent_cells) != 2:
            continue

        cell_a = int(adjacent_cells[0])
        cell_b = int(adjacent_cells[1])

        marker_a = cell_to_marker.get(cell_a)
        marker_b = cell_to_marker.get(cell_b)

        if marker_a is None or marker_b is None:
            continue

        vertices = facet_to_vertex.links(facet)
        if len(vertices) < 2:
            continue

        # Coordonnées géométriques des deux sommets
        coords = domain.geometry.x[vertices]

        # Une interface entre les rectangles est verticale :
        # les deux sommets ont la même abscisse.
        if not np.isclose(
            coords[0, 0], coords[1, 0], atol=tol
        ):
            continue

        x_facet = float(np.mean(coords[:, 0]))

        for interface_marker, x_interface in interface_x.items():
            if not np.isclose(
                x_facet, x_interface, atol=tol
            ):
                continue

            # Vérifier que les deux domaines adjacents
            # correspondent à l'interface attendue.
            expected_pairs = {
                5: {1, 2},
                6: {2, 3},
                7: {3, 4},
                8: {4, 5},
            }

            if {marker_a, marker_b} == expected_pairs[
                interface_marker
            ]:
                interface_facets[interface_marker].append(
                    facet
                )
                break

    # ---------------------------------------------------------
    # 6. Combiner les marqueurs extérieurs et internes
    # ---------------------------------------------------------

    boundary_groups = [
        (inlets, 1),
        (outlets, 2),
        (periodic_left, 3),
        (periodic_right, 4),
    ]

    boundary_groups.extend(
        (np.asarray(facets, dtype=np.int32), marker)
        for marker, facets in interface_facets.items()
    )

    facets_all = np.concatenate(
        [
            np.asarray(facets, dtype=np.int32)
            for facets, _ in boundary_groups
            if len(facets) > 0
        ]
    )

    values_all = np.concatenate(
        [
            np.full(len(facets), marker, dtype=np.int32)
            for facets, marker in boundary_groups
            if len(facets) > 0
        ]
    )

    # Les marqueurs doivent être triés et uniques.
    order = np.argsort(facets_all)
    facets_all = facets_all[order]
    values_all = values_all[order]

    if len(np.unique(facets_all)) != len(facets_all):
        raise RuntimeError(
            "Des arêtes ont reçu plusieurs marqueurs. "
            "Vérifier la sélection des frontières."
        )

    boundary_tags = meshtags(
        domain,
        fdim,
        facets_all,
        values_all,
    )

    # ---------------------------------------------------------
    # 7. Enregistrer les fichiers XDMF
    # ---------------------------------------------------------

    with XDMFFile(
        comm, "electrodialysis_mesh.xdmf", "w"
    ) as xdmf:
        xdmf.write_mesh(domain)

    if cell_tags is not None:
        with XDMFFile(
            comm, "electrodialysis_domains.xdmf", "w"
        ) as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_meshtags(
                cell_tags, domain.geometry
            )

    with XDMFFile(
        comm, "electrodialysis_boundaries.xdmf", "w"
    ) as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(
            boundary_tags, domain.geometry
        )

    # ---------------------------------------------------------
    # 8. Résumé
    # ---------------------------------------------------------

    if rank == 0:
        print("\nMaillage généré avec succès.")
        print(f"Longueur L : {L:g} m")
        print(f"Largeur totale : {x_max - x_min:g} m")
        print(f"Largeurs des domaines : {widths}")

        print("\nDomaines physiques :")
        for marker, name in enumerate(domain_names, start=1):
            print(f"  {marker} : {name}")

        print("\nFrontières extérieures :")
        print(f"  1 — Entrée : {len(inlets)} arêtes locales")
        print(f"  2 — Sortie : {len(outlets)} arêtes locales")
        print(
            f"  3 — Gauche : {len(periodic_left)} arêtes locales"
        )
        print(
            f"  4 — Droite : {len(periodic_right)} arêtes locales"
        )

        print("\nInterfaces internes :")
        interface_names = {
            5: "Canal gauche / CEM",
            6: "CEM / canal central",
            7: "Canal central / AEM",
            8: "AEM / canal droit",
        }

        for marker, name in interface_names.items():
            print(
                f"  {marker} — {name} : "
                f"{len(interface_facets[marker])} arêtes locales"
            )

        print(
            "\nNombre global de cellules :",
            domain.topology.index_map(tdim).size_global,
        )

        print("\nFichiers créés :")
        print("  electrodialysis_mesh.xdmf")
        print("  electrodialysis_domains.xdmf")
        print("  electrodialysis_boundaries.xdmf")

    return domain, cell_tags, boundary_tags


if __name__ == "__main__":
    create_geometry()
