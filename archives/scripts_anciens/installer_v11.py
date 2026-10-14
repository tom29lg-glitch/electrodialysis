"""Genere newton_periodique_v11.py depuis le solveur local v10/v8.
Ne modifie jamais le fichier source.
"""
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
# Ce script est destine a etre place dans le dossier electrodialysis.
sources = [root / 'newton_periodique_v10.py', root / 'newton_periodique_v8.py']
source = next((p for p in sources if p.exists()), None)
if source is None:
    sys.exit('ERREUR : placer installer_v11.py dans ~/Desktop/electrodialysis avec v10 ou v8.')
s = source.read_text()
anchor = 'ksp.setOperators(A)'
if s.count(anchor) != 1:
    sys.exit(f'ERREUR : attendu un seul {anchor!r}, trouve {s.count(anchor)}. Aucun fichier cree.')
if 'def normaliser_lignes_v11(' in s:
    sys.exit('ERREUR : normalisation v11 deja presente.')
if 'import numpy as np' not in s:
    sys.exit('ERREUR : import numpy manquant.')
helper = '''\n\ndef normaliser_lignes_v11(matrix, rhs):
    """Normalisation gauche des lignes PETSc (MPI mono-rang seulement).

    Chaque ligne i est divisee par max_j |A_ij|. Le vecteur residu
    est divise par la meme quantite. Ne change pas le systeme exact.
    """
    start, end = matrix.getOwnershipRange()
    nlocal = end - start
    scales = np.empty(nlocal, dtype=PETSc.ScalarType)
    zero_rows = []
    nonfinite_rows = []
    for local, row in enumerate(range(start, end)):
        cols, vals = matrix.getRow(row)
        values = np.asarray(vals)
        if values.size == 0:
            zero_rows.append(row)
            scales[local] = 1.0
            continue
        if not np.all(np.isfinite(values)):
            nonfinite_rows.append(row)
            scales[local] = 1.0
            continue
        magnitude = float(np.max(np.abs(values)))
        if magnitude == 0.0:
            zero_rows.append(row)
            scales[local] = 1.0
        else:
            scales[local] = 1.0 / magnitude
    if zero_rows or nonfinite_rows:
        raise RuntimeError(
            f'Lignes nulles: {zero_rows[:20]}; non finies: {nonfinite_rows[:20]}')
    scale = matrix.createVecLeft()
    scale.array[:] = scales
    scale.assemblyBegin()
    scale.assemblyEnd()
    matrix.diagonalScale(L=scale)
    rhs.pointwiseMult(rhs, scale)
    matrix.assemble()
    print('  V11 normalisation: lignes =', nlocal,
          '| facteur min =', f'{float(np.min(scales)):.3e}',
          '| facteur max =', f'{float(np.max(scales)):.3e}', flush=True)
    scale.destroy()
\n'''
# Insert helper before the first top-level function, or before the main section.
idx = s.find('\ndef enforce_state(')
if idx == -1:
    sys.exit('ERREUR : fonction enforce_state introuvable. Aucun fichier cree.')
s = s[:idx] + helper + s[idx:]
# Determine indentation from actual line rather than assuming 8 spaces.
lines = s.splitlines(keepends=True)
for i, line in enumerate(lines):
    if line.strip() == anchor:
        indent = line[:len(line) - len(line.lstrip())]
        lines.insert(i, indent + '# Normaliser A et le residu avant la resolution lineaire.\n' + indent + 'normaliser_lignes_v11(A, b)\n')
        break
s = ''.join(lines)
s = s.replace('Newton v10 diagnostic:', 'Newton v11 normalise:')
s = s.replace('Newton v9 experimental:', 'Newton v11 normalise:')
output = root / 'newton_periodique_v11.py'
if output.exists():
    sys.exit('ERREUR : v11 existe deja. Aucun ecrasement.')
compile(s, str(output), 'exec')
output.write_text(s)
print('Cree :', output.name)
print('Source conservee :', source.name)
