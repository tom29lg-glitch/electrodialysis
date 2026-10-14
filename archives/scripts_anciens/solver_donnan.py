"""
solver.py
---------

Solveur stationnaire provisoire d'electrodialyse avec DOLFINx.

Domaines :
    1 : canal gauche
    2 : membrane CEM
    3 : canal central
    4 : membrane AEM
    5 : canal droit

Modele actuel :
    - Diffusion
    - Convection
    - Migration electrique de Nernst-Planck
    - Potentiel electrique impose lineairement suivant x
    - Charge fixe des membranes stockee pour les prochaines etapes

Simplifications :
    - Pas de resolution auto-coherente du potentiel electrique
    - Pas de conditions de Donnan aux interfaces
    - Pas de couplage d'electroneutralite
    - Conditions d'entree simplifiees
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

    # Creer les entites de facette avant de lire les marqueurs
    domain.topology.create_entities(domain.topology.dim - 1)

    with XDMFFile(
        comm, "electrodialysis_boundaries.xdmf", "r"
    ) as xdmf:
        facet_tags = xdmf.read_meshtags(
            domain, name="mesh_tags"
        )

    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)

    dx = ufl.Measure(
        "dx", domain=domain, subdomain_data=cell_tags
    )

    # --------------------------------------------------------
    # 2. Espaces fonctionnels et champs
    # --------------------------------------------------------

    V = fem.functionspace(domain, ("Lagrange", 1))
    Q = fem.functionspace(domain, ("DG", 0))

    DNa_field = fem.Function(Q, name="Diffusivite_Na")
    DCl_field = fem.Function(Q, name="Diffusivite_Cl")
    fixed_charge = fem.Function(Q, name="Charge_fixe")
    velocity_y = fem.Function(Q, name="Vitesse_y")

    # Potentiel et concentrations
    phi = fem.Function(V, name="Potentiel_electrique")

    # --------------------------------------------------------
    # 3. Proprietes des domaines
    # --------------------------------------------------------

    geometry_dofmap = domain.geometry.dofmaps[0]

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

        # Centre geometrique approximatif de la cellule
        geometry_dofs = geometry_dofmap[cell]
        points = domain.geometry.x[geometry_dofs]
        x_cell = np.mean(points[:, 0])

        # Coordonnees transversales dans les canaux
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
        DNa_field,
        DCl_field,
        fixed_charge,
        velocity_y,
    ):
        field.x.scatter_forward()

    velocity = ufl.as_vector((0.0, velocity_y))

    # --------------------------------------------------------
    # 4. Potentiel electrique impose
    # --------------------------------------------------------
    # Profil provisoire lineaire entre les bords x_min et x_max.
    # Ce n'est pas encore une resolution auto-coherente de phi.
    # --------------------------------------------------------

    x_min = -(W_ch + W_m)
    x_max = +(W_ch + W_m)

    phi.interpolate(
        lambda x: Vtot * (x[0] - x_min) / (x_max - x_min)
    )
    phi.x.scatter_forward()

    # --------------------------------------------------------
    # 5. Conditions aux limites des concentrations
    # --------------------------------------------------------
    # Marqueur 1 : entree y = 0.
    # La meme concentration d'entree est imposee aux deux ions
    # pour ce calcul de validation.
    # --------------------------------------------------------

    inlet_facets = facet_tags.find(1)

    inlet_dofs = fem.locate_dofs_topological(
        V, fdim, inlet_facets
    )

    c_in = fem.Constant(domain, float(cCl_0))
    bc = fem.dirichletbc(c_in, inlet_dofs, V)

    # --------------------------------------------------------
    # 6. Formulation faible de Nernst-Planck
    # --------------------------------------------------------
    # Flux :
    # N_i = -D_i grad(c_i)
    #       - z_i D_i F/(R T) c_i grad(phi)
    #       + u c_i
    #
    # Equation stationnaire provisoire :
    # -div(D grad(c) + z D F/(R T) c grad(phi))
    # + u.grad(c) = 0
    #
    # La convection est ecrite sous forme non conservative,
    # avec un traitement simplifie de la sortie.
    # --------------------------------------------------------

    c_trial = ufl.TrialFunction(V)
    test = ufl.TestFunction(V)

    def solve_species(diffusivity, valence, name):
        a = (
            diffusivity
            * ufl.dot(
                ufl.grad(c_trial),
                ufl.grad(test),
            )
            + valence
            * diffusivity
            * F
            / (R * T)
            * c_trial
            * ufl.dot(
                ufl.grad(phi),
                ufl.grad(test),
            )
            + ufl.dot(
                velocity,
                ufl.grad(c_trial),
            )
            * test
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
        print("\nResolution stationnaire de Nernst-Planck...")
        print("Migration electrique activee.")
        print(f"Tension imposee : {Vtot:.6g} V")

    cNa = solve_species(
        DNa_field, zNa, "Concentration_Na"
    )
    cCl = solve_species(
        DCl_field, zCl, "Concentration_Cl"
    )

    # --------------------------------------------------------
    # 7. Statistiques globales
    # --------------------------------------------------------

    def global_min_max(function):
        index_map = function.function_space.dofmap.index_map
        bs = function.function_space.dofmap.index_map_bs
        n_owned = index_map.size_local * bs

        values = np.real(function.x.array[:n_owned])

        local_min = (
            float(np.min(values))
            if values.size
            else np.inf
        )
        local_max = (
            float(np.max(values))
            if values.size
            else -np.inf
        )

        global_min = comm.allreduce(local_min, op=MPI.MIN)
        global_max = comm.allreduce(local_max, op=MPI.MAX)

        return global_min, global_max

    na_min, na_max = global_min_max(cNa)
    cl_min, cl_max = global_min_max(cCl)
    phi_min, phi_max = global_min_max(phi)

    # --------------------------------------------------------
    # 8. Verification geometrique
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

    relative_error = (
        abs(total_area - expected_area) / expected_area
    )

    # --------------------------------------------------------
    # 9. Sauvegarde des resultats
    # --------------------------------------------------------

    with XDMFFile(
        comm, "electrodialysis_results.xdmf", "w"
    ) as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi)
        xdmf.write_function(cNa)
        xdmf.write_function(cCl)

    # --------------------------------------------------------
    # 10. Bilan
    # --------------------------------------------------------

    if comm.rank == 0:
        print("\n" + "=" * 66)
        print("CALCUL STATIONNAIRE — ELECTRODIALYSE")
        print("=" * 66)

        print("\nAires des domaines :")
        for marker, name in DOMAIN_NAMES.items():
            print(
                f"  {marker} - {name:<25} "
                f"{areas[marker]:.6e} m²"
            )

        print(f"\nAire totale     : {total_area:.6e} m²")
        print(f"Aire attendue   : {expected_area:.6e} m²")
        print(f"Erreur relative : {relative_error:.3e}")

        print("\nParametres :")
        print(f"  Temperature             : {T:.2f} K")
        print(f"  Tension imposee         : {Vtot:.6g} V")
        print(f"  Concentration d'entree  : {cCl_0:.6g} mol/m³")
        print(f"  Charge fixe membranaire : {cMem_default:.6g} mol/m³")
        print(f"  Porosite membranaire    : {epsilon_mem:.6g}")
        print(f"  Vitesse moyenne ref.    : {v_avg:.6e} m/s")

        print("\nPotentiel electrique :")
        print(f"  Min : {phi_min:.8g} V")
        print(f"  Max : {phi_max:.8g} V")

        print("\nConcentrations calculees :")
        print(
            f"  Na+ : min = {na_min:.8g}, "
            f"max = {na_max:.8g} mol/m³"
        )
        print(
            f"  Cl- : min = {cl_min:.8g}, "
            f"max = {cl_max:.8g} mol/m³"
        )

        if na_min < 0.0 or cl_min < 0.0:
            print(
                "\nATTENTION : une concentration est negative. "
                "Le modele ou la discretisation doit etre stabilise."
            )

        if relative_error < 1.0e-8:
            print("\nVerification geometrique : OK")
        else:
            print("\nATTENTION : erreur sur l'aire totale")

        print("\nResultats sauvegardes dans :")
        print("  electrodialysis_results.xdmf")
        print("\nLimites actuelles :")
        print("  - potentiel impose, non resolu")
        print("  - conditions de Donnan absentes")
        print("  - electroneutralite non imposee")
        print("  - conditions aux limites simplifiees")
        print("=" * 66)


if __name__ == "__main__":
    main()
