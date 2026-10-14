from pathlib import Path

p = Path("solver_comsol_nonlinear.py")
s = p.read_text()

if "# CONVECTION ELECTRODIALYSE" in s:
    raise RuntimeError("La convection a deja ete ajoutee.")

lines = s.splitlines()

# Localiser les elements existants
i_na = next(i for i, line in enumerate(lines)
            if line.startswith("NNa ="))
i_cl = next(i for i, line in enumerate(lines)
            if line.startswith("NCl ="))

if not any(line.startswith("def ordinary_dg(") for line in lines):
    raise RuntimeError("Fonction ordinary_dg introuvable")

# Profil parabolique, nul dans les membranes
velocity_code = '''
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
'''

# Ajouter la vitesse avant les flux volumiques
lines.insert(i_na, velocity_code)

s = "\n".join(lines) + "\n"

# Verifier les imports
if "import numpy as np" not in s:
    s = "import numpy as np\n" + s

if "from parameters import v_avg" not in s:
    s = "from parameters import v_avg\n" + s

# Ajouter la convection aux flux volumiques
lines = s.splitlines()

for i, line in enumerate(lines):
    if line.startswith("NNa ="):
        lines[i] = line + " + velocity*cNa"
    elif line.startswith("NCl ="):
        lines[i] = line + " + velocity*cCl"

s = "\n".join(lines) + "\n"

# Ajouter le flux upwind aux facettes ordinaires
anchor = "    return consistency + symmetry + penalty"

if anchor not in s:
    raise RuntimeError("Retour de ordinary_dg introuvable")

replacement = '''
    # Flux convectif upwind sur les facettes ordinaires
    un = ufl.avg(uy) * n("+")[1]

    cup = ufl.conditional(
        ufl.ge(un, 0.0),
        c("+"),
        c("-")
    )

    convection = un * cup * ufl.jump(v) * dS(10)

    return consistency + symmetry + penalty + convection
'''

s = s.replace(anchor, replacement, 1)

p.write_text(s)

print("Convection ajoutee dans les trois canaux.")
print("Flux volumiques et flux DG upwind modifies.")
