from pathlib import Path

p = Path("solver_comsol_nonlinear.py")
s = p.read_text()

# 1. Ajouter les parametres manquants
s = s.replace(
    "from solver_comsol import build_dg_measures",
    """from solver_comsol import build_dg_measures
from parameters import W_ch, v_avg"""
)

# 2. Construire une vitesse DG0 dans chaque cellule
anchor = "jacobian = ufl.derivative(residual,w,ufl.TrialFunction(W))"

velocity_code = """
# --------------------------------------------------
# Vitesse d'ecoulement : profil parabolique
# --------------------------------------------------

V0 = fem.functionspace(mesh, ("DG", 0))
uy = fem.Function(V0, name="velocity_y")

centers = {1: -W_ch, 3: 0.0, 5: W_ch}
half_widths = {
    1: W_ch / 4,
    3: W_ch / 2,
    5: W_ch / 4
}

tdim = mesh.topology.dim
geometry_dofs = mesh.geometry.dofmap

for cell, marker in zip(cell_tags.indices, cell_tags.values):
    cell = int(cell)
    marker = int(marker)

    dofs = V0.dofmap.cell_dofs(cell)

    if marker in centers:
        points = mesh.geometry.x[geometry_dofs[cell]]
        x = float(np.mean(points[:, 0]))

        xi = (x - centers[marker]) / half_widths[marker]

        uy.x.array[dofs] = (
            1.5 * v_avg * max(0.0, 1.0 - xi**2)
        )
    else:
        uy.x.array[dofs] = 0.0

uy.x.scatter_forward()
"""

if anchor not in s:
    raise RuntimeError("Point d'insertion de la vitesse introuvable")

s = s.replace(anchor, velocity_code + "\n" + anchor, 1)

# 3. Ajouter la convection aux flux volumiques
s = s.replace(
    "NNa = -DNa_field * ufl.grad(cNa) - DNa_field*F/(R*T)*cNa*ufl.grad(phi)",
    "NNa = -DNa_field * ufl.grad(cNa) - DNa_field*F/(R*T)*cNa*ufl.grad(phi) + ufl.as_vector((0.0, uy))*cNa"
)

s = s.replace(
    "NCl = -DCl_field * ufl.grad(cCl) + DCl_field*F/(R*T)*cCl*ufl.grad(phi)",
    "NCl = -DCl_field * ufl.grad(cCl) + DCl_field*F/(R*T)*cCl*ufl.grad(phi) + ufl.as_vector((0.0, uy))*cCl"
)

# 4. Corriger la formulation DG ordinaire
start = s.find("def ordinary_dg(")
end = s.find("residual += ordinary_dg(", start)

if start < 0 or end < 0:
    raise RuntimeError("Fonction ordinary_dg introuvable")

new_dg = """def ordinary_dg(c, v, D, z):

    # Flux diffusif + migratoire
    Ndiff = (
        -D * ufl.grad(c)
        -z * beta * D * c * ufl.grad(phi)
    )

    # Terme conservatif sur les facettes
    consistency = (
        ufl.dot(ufl.avg(Ndiff), n("+"))
        * ufl.jump(v)
        * dS(10)
    )

    # Stabilisation symetrique de diffusion
    symmetry = (
        ufl.avg(D * ufl.grad(v))
        * ufl.jump(c, n)
        * dS(10)
    )

    # Penalisation
    penalty = (
        gamma * ufl.avg(D) / ufl.avg(h)
        * ufl.dot(
            ufl.jump(c, n),
            ufl.jump(v, n)
        )
        * dS(10)
    )

    # Convection upwind
    un = ufl.avg(uy) * n("+")[1]

    cup = ufl.conditional(
        ufl.ge(un, 0.0),
        c("+"),
        c("-")
    )

    convection = (
        un * cup * ufl.jump(v)
        * dS(10)
    )

    return consistency + symmetry + penalty + convection

"""

# Le terme de symetrie sera traite separement :
# il doit etre contracte avec la normale et utilise
# avec le bon signe dans la formulation finale.
new_dg = new_dg.replace(
    """ufl.avg(D * ufl.grad(v))
        * ufl.jump(c, n)""",
    """ufl.dot(
            ufl.avg(D * ufl.grad(v)),
            ufl.jump(c, n)
        )"""
)

s = s[:start] + new_dg + s[end:]

# 5. Ajouter l'entree et la sortie
anchor = "residual += ordinary_dg(cCl, vCl, DCl_field, -1)"

boundary_code = """

# --------------------------------------------------
# Entree et sortie des canaux
# --------------------------------------------------

ds = ufl.Measure(
    "ds",
    domain=mesh,
    subdomain_data=facet_tags
)

# Entree : flux convectif impose, c_in = 500 mol/m3
# Sortie : flux convectif sortant
# Les termes diffusifs de bord sont supposes nuls ici.

for c, v in ((cNa, vNa), (cCl, vCl)):
    residual += (
        -uy * cCl_0 * v * ds(1)
        +uy * c * v * ds(2)
    )
"""

if anchor not in s:
    raise RuntimeError("Point d'insertion des frontieres introuvable")

s = s.replace(anchor, anchor + boundary_code)

p.write_text(s)
print("Convection, flux upwind et entree/sortie ajoutes.")
