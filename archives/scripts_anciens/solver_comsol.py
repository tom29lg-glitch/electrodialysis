"""Solveur de validation Nernst-Planck pour une cellule d'electrodialyse.

Ce fichier prepare les champs necessaires au futur couplage electroneutre
Nernst-Planck. Il ne resout PAS encore le potentiel : il est impose lineairement.
Aucune electroneutralite locale ni condition de Donnan n'est imposee.

Les mesures DG dS(10) (facettes ordinaires) et dS(5..8) (interfaces)
sont construites et controlees, mais pas encore utilisees dans le solveur CG1.
"""

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import LinearProblem

from parameters import (
    L, W_ch, W_m, DNa, DCl, T, v_avg, cCl_0,
    cMem_default, epsilon_mem, Vtot, F, R, zNa, zCl,
)

# Permittivites relatives provisoires (validation uniquement).
EPSILON_0 = 8.8541878128e-12  # F/m
EPSILON_R_SOLUTION = 78.3
EPSILON_R_CEM = 78.3
EPSILON_R_AEM = 78.3

DOMAIN_NAMES = {
    1: "Canal gauche",
    2: "Membrane cationique",
    3: "Canal central",
    4: "Membrane anionique",
    5: "Canal droit",
}


def build_dg_measures(domain, cell_tags, facet_tags):
    """Cree dS(10) pour les facettes ordinaires et dS(5..8) pour les membranes.

    Cette construction est validee ici pour une execution mono-processus.
    """
    if domain.comm.size != 1:
        raise NotImplementedError(
            "La classification DG doit etre adaptee aux facettes partagees "
            "avant une execution MPI multi-processus."
        )

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_entities(fdim)
    domain.topology.create_connectivity(fdim, tdim)
    facet_to_cell = domain.topology.connectivity(fdim, tdim)

    cell_markers = {
        int(cell): int(marker)
        for cell, marker in zip(cell_tags.indices, cell_tags.values)
    }
    interface_markers = {
        int(facet): int(marker)
        for facet, marker in zip(facet_tags.indices, facet_tags.values)
        if int(marker) in (5, 6, 7, 8)
    }

    indices = []
    values = []
    num_facets = domain.topology.index_map(fdim).size_local

    for facet in range(num_facets):
        cells = facet_to_cell.links(facet)
        if len(cells) != 2:
            continue

        a, b = (cell_markers[int(cell)] for cell in cells)
        if facet in interface_markers:
            marker = interface_markers[facet]
            if a == b:
                raise RuntimeError(
                    f"Facette {facet} marquee comme interface dans un seul domaine"
                )
        elif a == b:
            marker = 10
        else:
            raise RuntimeError(
                f"Interface non identifiee : facette {facet}, domaines {a}, {b}"
            )

        indices.append(facet)
        values.append(marker)

    indices = np.asarray(indices, dtype=np.int32)
    values = np.asarray(values, dtype=np.int32)
    order = np.argsort(indices)
    dg_tags = dmesh.meshtags(domain, fdim, indices[order], values[order])
    dS = ufl.Measure("dS", domain=domain, subdomain_data=dg_tags)
    return dS


def main():
    comm = MPI.COMM_WORLD

    with XDMFFile(comm, "electrodialysis_domains.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="mesh")
        cell_tags = xdmf.read_meshtags(domain, name="cell_tags")

    fdim = domain.topology.dim - 1
    domain.topology.create_entities(fdim)
    with XDMFFile(comm, "electrodialysis_boundaries.xdmf", "r") as xdmf:
        facet_tags = xdmf.read_meshtags(domain, name="mesh_tags")
    domain.topology.create_connectivity(fdim, domain.topology.dim)

    # Mesures DG disponibles pour le futur solveur couple.
    dS = build_dg_measures(domain, cell_tags, facet_tags)
    if comm.rank == 0:
        print("\nVerification des mesures DG :")
    for marker in (10, 5, 6, 7, 8):
        local_length = fem.assemble_scalar(fem.form(1.0 * dS(marker)))
        length = comm.allreduce(local_length, op=MPI.SUM)
        if comm.rank == 0:
            print(f"  dS({marker:2d}) : {length:.8e} m")
        if marker in (5, 6, 7, 8) and not np.isclose(length, L, rtol=1e-8):
            raise RuntimeError(f"Longueur incorrecte pour l'interface {marker}")

    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    V = fem.functionspace(domain, ("Lagrange", 1))
    Q = fem.functionspace(domain, ("DG", 0))

    DNa_field = fem.Function(Q, name="Diffusivite_Na")
    DCl_field = fem.Function(Q, name="Diffusivite_Cl")
    fixed_charge = fem.Function(Q, name="Charge_fixe")
    permittivity = fem.Function(Q, name="Permittivite_electrique")
    velocity_y = fem.Function(Q, name="Vitesse_y")
    phi = fem.Function(V, name="Potentiel_electrique")

    geometry_dofmap = domain.geometry.dofmaps[0]
    for cell, marker in zip(cell_tags.indices, cell_tags.values):
        cell = int(cell)
        marker = int(marker)
        dofs = Q.dofmap.cell_dofs(cell)

        if marker in (1, 3, 5):
            d_na, d_cl, charge = DNa, DCl, 0.0
            eps_r = EPSILON_R_SOLUTION
        elif marker == 2:
            d_na, d_cl, charge = epsilon_mem * DNa, epsilon_mem * DCl, -cMem_default
            eps_r = EPSILON_R_CEM
        elif marker == 4:
            d_na, d_cl, charge = epsilon_mem * DNa, epsilon_mem * DCl, cMem_default
            eps_r = EPSILON_R_AEM
        else:
            raise ValueError(f"Marqueur de domaine inconnu : {marker}")

        DNa_field.x.array[dofs] = d_na
        DCl_field.x.array[dofs] = d_cl
        fixed_charge.x.array[dofs] = charge
        permittivity.x.array[dofs] = EPSILON_0 * eps_r

        geometry_dofs = geometry_dofmap[cell]
        points = domain.geometry.x[geometry_dofs]
        x_cell = float(np.mean(points[:, 0]))

        centers = {
            1: -W_ch,
            3: 0.0,
            5: +W_ch,
        }

        half_widths = {
            1: W_ch / 4.0,
            3: W_ch / 2.0,
            5: W_ch / 4.0,
        }

        if marker in (1, 3, 5):
            relative_x = (
                x_cell - centers[marker]
            ) / half_widths[marker]

            velocity_y.x.array[dofs] = (
                1.5 * v_avg
                * (1.0 - relative_x**2)
            )
        else:
            velocity_y.x.array[dofs] = 0.0

    for field in (DNa_field, DCl_field, fixed_charge, permittivity, velocity_y):
        field.x.scatter_forward()

    velocity = ufl.as_vector((0.0, velocity_y))

    # Potentiel de validation : il n'est pas calcule par conservation du courant.
    x_min = -(W_ch + W_m)
    x_max = +(W_ch + W_m)
    phi.interpolate(lambda x: Vtot * (x[0] - x_min) / (x_max - x_min))
    phi.x.scatter_forward()

    inlet_facets = facet_tags.find(1)
    inlet_dofs = fem.locate_dofs_topological(V, fdim, inlet_facets)
    c_in = fem.Constant(domain, float(cCl_0))
    bc = fem.dirichletbc(c_in, inlet_dofs, V)

    c_trial = ufl.TrialFunction(V)
    test = ufl.TestFunction(V)

    def solve_species(diffusivity, valence, name):
        # Validation simplifiee : CG1, potentiel impose, pas de Donnan.
        a = (
            diffusivity * ufl.dot(ufl.grad(c_trial), ufl.grad(test))
            + valence * diffusivity * F / (R * T) * c_trial
            * ufl.dot(ufl.grad(phi), ufl.grad(test))
            + ufl.dot(velocity, ufl.grad(c_trial)) * test
        ) * dx
        Lform = fem.Constant(domain, 0.0) * test * dx
        problem = LinearProblem(
            a, Lform, bcs=[bc],
            petsc_options_prefix=f"{name}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        solution = problem.solve()
        solution.name = name
        solution.x.scatter_forward()
        return solution

    if comm.rank == 0:
        print("\nValidation Nernst-Planck avec potentiel impose")
        print("Conservation du courant et conditions de Donnan : NON resolues")
        print(f"Tension de validation : {Vtot:.6g} V")

    cNa = solve_species(DNa_field, zNa, "Concentration_Na")
    cCl = solve_species(DCl_field, zCl, "Concentration_Cl")

    def global_min_max(function):
        index_map = function.function_space.dofmap.index_map
        bs = function.function_space.dofmap.index_map_bs
        n_owned = index_map.size_local * bs
        values = np.real(function.x.array[:n_owned])
        local_min = float(np.min(values)) if values.size else np.inf
        local_max = float(np.max(values)) if values.size else -np.inf
        return (
            comm.allreduce(local_min, op=MPI.MIN),
            comm.allreduce(local_max, op=MPI.MAX),
        )

    na_min, na_max = global_min_max(cNa)
    cl_min, cl_max = global_min_max(cCl)
    phi_min, phi_max = global_min_max(phi)
    eps_min, eps_max = global_min_max(permittivity)

    areas = {}
    for marker in DOMAIN_NAMES:
        local_area = fem.assemble_scalar(fem.form(1.0 * dx(marker)))
        areas[marker] = comm.allreduce(local_area, op=MPI.SUM)
    total_area = sum(areas.values())
    expected_area = 2.0 * (W_ch + W_m) * L
    relative_error = abs(total_area - expected_area) / expected_area

    with XDMFFile(comm, "electrodialysis_pnp_preparation.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi)
        xdmf.write_function(cNa)
        xdmf.write_function(cCl)
        xdmf.write_function(fixed_charge)
        xdmf.write_function(permittivity)

    if comm.rank == 0:
        print("\n" + "=" * 66)
        print("VALIDATION DES CHAMPS POUR LE FUTUR MODELE ELECTRONEUTRE")
        print("=" * 66)
        print("\nAires des domaines :")
        for marker, name in DOMAIN_NAMES.items():
            print(f"  {marker} - {name:<25} {areas[marker]:.6e} m²")
        print(f"\nAire totale     : {total_area:.6e} m²")
        print(f"Aire attendue   : {expected_area:.6e} m²")
        print(f"Erreur relative : {relative_error:.3e}")
        print("\nParametres :")
        print(f"  Temperature             : {T:.2f} K")
        print(f"  Tension de validation   : {Vtot:.6g} V")
        print(f"  Concentration d'entree  : {cCl_0:.6g} mol/m³")
        print(f"  Charge fixe membranaire : {cMem_default:.6g} mol/m³")
        print(f"  Facteur transport mem.  : {epsilon_mem:.6g}")
        print(f"  Vitesse moyenne ref.    : {v_avg:.6e} m/s")
        print(f"  Permittivite min/max    : {eps_min:.6e} / {eps_max:.6e} F/m")
        print(f"\nPotentiel : min = {phi_min:.8g}, max = {phi_max:.8g} V")
        print(f"Na+ : min = {na_min:.8g}, max = {na_max:.8g} mol/m³")
        print(f"Cl- : min = {cl_min:.8g}, max = {cl_max:.8g} mol/m³")
        if na_min < 0 or cl_min < 0:
            print("ATTENTION : concentration negative; discretisation a stabiliser.")
        print("\nSortie : electrodialysis_pnp_preparation.xdmf")
        print("\nLimites : potentiel impose; electroneutralite et Donnan absents;")
        print("         conditions aux limites et convection simplifiees.")
        print("=" * 66)


if __name__ == "__main__":
    main()
