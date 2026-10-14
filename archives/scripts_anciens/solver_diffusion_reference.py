
"""
solver.py
---------
Premier solveur stationnaire d'electrodialyse avec DOLFINx.

Domaines :
    1 : canal gauche
    2 : membrane CEM
    3 : canal central
    4 : membrane AEM
    5 : canal droit

Etape 1 :
    Transport stationnaire par diffusion et convection.
    Migration electrique et conditions de Donnan desactivees
    temporairement pour valider le solveur.
"""

import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import LinearProblem

from parameters import (
    L,
    W_ch,
    W_m,
    DNa,
    DCl,
    T,
    v_avg,
    cCl_0,
    cMem_default,
    epsilon_mem,
    Vtot,
    F,
    R,
    zNa,
    zCl,
)


DOMAIN_NAMES = {
    1: "Canal gauche",
    2: "Membrane cationique",
    3: "Canal central",
    4: "Membrane anionique",
    5: "Canal droit",
}


def main():

    comm = MPI.COMM_WORLD

    # --------------------------------------------------------
    # 1. Chargement du maillage et des marqueurs
    # --------------------------------------------------------

    with XDMFFile(
        comm, "electrodialysis_domains.xdmf", "r"
    ) as xdmf:
        domain = xdmf.read_mesh(name="mesh")
        cell_tags = xdmf.read_meshtags(
            domain, name="cell_tags"
        )

    with XDMFFile(
        comm, "electrodialysis_boundaries.xdmf", "r"
    ) as xdmf:
        domain.topology.create_entities(domain.topology.dim - 1)
        facet_tags = xdmf.read_meshtags(
            domain, name="mesh_tags"
        )

    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)

    dx = ufl.Measure(
        "dx", domain=domain, subdomain_data=cell_tags
    )

    # --------------------------------------------------------
    # 2. Espaces fonctionnels
    # --------------------------------------------------------

    V = fem.functionspace(domain, ("Lagrange", 1))
    Q = fem.functionspace(domain, ("DG", 0))

    cNa = fem.Function(V, name="Concentration_Na")
    cCl = fem.Function(V, name="Concentration_Cl")

    DNa_field = fem.Function(Q, name="Diffusivite_Na")
    DCl_field = fem.Function(Q, name="Diffusivite_Cl")
    fixed_charge = fem.Function(Q, name="Charge_fixe")
    velocity_y = fem.Function(Q, name="Vitesse_y")

    geometry_dofmap = domain.geometry.dofmaps[0]

    # --------------------------------------------------------
    # 3. Proprietes des domaines
    # --------------------------------------------------------

    for cell, marker in zip(cell_tags.indices, cell_tags.values):

        cell = int(cell)
        marker = int(marker)

        dofs = Q.dofmap.cell_dofs(cell)

        if marker in (1, 3, 5):
            d_na = DNa
            d_cl = DCl
            charge = 0.0

        elif marker == 2:
            d_na = epsilon_mem * DNa
            d_cl = epsilon_mem * DCl
            charge = -cMem_default

        elif marker == 4:
            d_na = epsilon_mem * DNa
            d_cl = epsilon_mem * DCl
            charge = +cMem_default

        else:
            raise ValueError(
                f"Marqueur de domaine inconnu : {marker}"
            )

        DNa_field.x.array[dofs] = d_na
        DCl_field.x.array[dofs] = d_cl
        fixed_charge.x.array[dofs] = charge

        # Centre de la cellule
        geometry_dofs = geometry_dofmap[cell]
        points = domain.geometry.x[geometry_dofs]
        x_cell = np.mean(points[:, 0])

        # Coordonnée transversale dans chaque canal
        if marker == 1:
            t = x_cell + W_ch + W_m
        elif marker == 3:
            t = x_cell
        elif marker == 5:
            t = x_cell - W_ch - W_m
        else:
            velocity_y.x.array[dofs] = 0.0
            continue

        # Profil parabolique de vitesse
        u_local = 3.0 * v_avg * (
            1.0 - (t / (W_ch / 2.0)) ** 2
        )

        velocity_y.x.array[dofs] = u_local

    for field in (
        DNa_field, DCl_field, fixed_charge, velocity_y
    ):
        field.x.scatter_forward()

    velocity = ufl.as_vector((0.0, velocity_y))

    # --------------------------------------------------------
    # 4. Conditions aux limites
    # --------------------------------------------------------
    # Marqueur 1 : entree y = 0
    # Marqueur 2 : sortie y = L
    #
    # Concentrations imposees a l'entree.
    # A la sortie, le terme convectif est traite sous forme
    # u.grad(c), avec flux diffusif naturel.
    # --------------------------------------------------------

    inlet_facets = facet_tags.find(1)

    inlet_dofs = fem.locate_dofs_topological(
        V, fdim, inlet_facets
    )

    c_in = fem.Constant(domain, float(cCl_0))
    bc = fem.dirichletbc(c_in, inlet_dofs, V)

    # --------------------------------------------------------
    # 5. Formulation faible stationnaire
    # --------------------------------------------------------
    # -div(D grad(c)) + u.grad(c) = 0
    #
    # La migration electrique et Donnan ne sont pas encore
    # inclus. Cette etape sert a valider le solveur.
    # --------------------------------------------------------

    c_trial = ufl.TrialFunction(V)
    test = ufl.TestFunction(V)

    def solve_species(diffusivity, name):

        a = (
            diffusivity * ufl.dot(
                ufl.grad(c_trial), ufl.grad(test)
            )
            + ufl.dot(velocity, ufl.grad(c_trial)) * test
        ) * dx

        Lform = fem.Constant(domain, 0.0) * test * dx

        problem = LinearProblem(
            a,
            Lform,
            bcs=[bc],
            petsc_options_prefix=f"{name}_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )

        solution = problem.solve()
        solution.name = name
        solution.x.scatter_forward()

        return solution

    if comm.rank == 0:
        print("\nResolution stationnaire en cours...")

    cNa = solve_species(DNa_field, "Concentration_Na")
    cCl = solve_species(DCl_field, "Concentration_Cl")

    # --------------------------------------------------------
    # 6. Statistiques des concentrations
    # --------------------------------------------------------

    def global_min_max(function):

        index_map = function.function_space.dofmap.index_map
        n_owned = index_map.size_local * (
            function.function_space.dofmap.index_map_bs
        )

        values = np.real(function.x.array[:n_owned])

        local_min = float(np.min(values)) if values.size else np.inf
        local_max = float(np.max(values)) if values.size else -np.inf

        global_min = comm.allreduce(local_min, op=MPI.MIN)
        global_max = comm.allreduce(local_max, op=MPI.MAX)

        return global_min, global_max

    na_min, na_max = global_min_max(cNa)
    cl_min, cl_max = global_min_max(cCl)

    # --------------------------------------------------------
    # 7. Verification des aires
    # --------------------------------------------------------

    areas = {}

    for marker in DOMAIN_NAMES:
        local_area = fem.assemble_scalar(
            fem.form(1.0 * dx(marker))
        )
        areas[marker] = comm.allreduce(
            local_area, op=MPI.SUM
        )

    total_area = sum(areas.values())
    expected_area = 2.0 * (W_ch + W_m) * L

    relative_error = abs(
        total_area - expected_area
    ) / expected_area

    # --------------------------------------------------------
    # 8. Sauvegarde des resultats
    # --------------------------------------------------------

    with XDMFFile(comm, "electrodialysis_results.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(cNa)
        xdmf.write_function(cCl)

    # --------------------------------------------------------
    # 9. Bilan
    # --------------------------------------------------------

    if comm.rank == 0:

        print("\n" + "=" * 64)
        print("PREMIER CALCUL STATIONNAIRE — ELECTRODIALYSE")
        print("=" * 64)

        print("\nAires des domaines :")
        for marker, name in DOMAIN_NAMES.items():
            print(
                f"  {marker} - {name:<25} "
                f"{areas[marker]:.6e} m²"
            )

        print(f"\nAire totale : {total_area:.6e} m²")
        print(f"Aire attendue : {expected_area:.6e} m²")
        print(f"Erreur relative : {relative_error:.3e}")

        print("\nParametres :")
        print(f"  Temperature : {T:.2f} K")
        print(f"  Tension cible pour les prochaines etapes : {Vtot:.3f} V")
        print(f"  Concentration d'entree : {cCl_0:.6f} mol/m³")
        print(f"  Charge fixe membranaire : {cMem_default:.1f} mol/m³")
        print(f"  Porosite membranaire : {epsilon_mem:.3f}")
        print(f"  Vitesse moyenne de reference : {v_avg:.6e} m/s")

        print("\nConcentrations calculees :")
        print(f"  Na+ : min = {na_min:.8g}, max = {na_max:.8g} mol/m³")
        print(f"  Cl- : min = {cl_min:.8g}, max = {cl_max:.8g} mol/m³")

        if relative_error < 1.0e-8:
            print("\nVerification geometrique : OK")
        else:
            print("\nATTENTION : erreur sur l'aire totale")

        print("\nSolveur stationnaire : termine.")
        print("Resultats sauvegardes dans electrodialysis_results.xdmf")
        print("Attention : migration electrique et Donnan desactives.")
        print("=" * 64)


if __name__ == "__main__":
    main()
