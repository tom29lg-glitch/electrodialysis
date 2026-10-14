"""Cree v12 depuis v11 : PETSc FieldSplit Schur ions/potentiel, sans ecraser les sources."""
from pathlib import Path
root = Path(__file__).resolve().parent
source = root / 'newton_periodique_v11.py'
out = root / 'newton_periodique_v12.py'
if not source.is_file():
    raise SystemExit('Fichier newton_periodique_v11.py introuvable dans ce dossier.')
if out.exists():
    raise SystemExit('newton_periodique_v12.py existe deja : aucun ecrasement.')
s = source.read_text()
old = "ksp.getPC().setType('jacobi')"
assert s.count(old) == 1, 'Configuration Jacobi introuvable ou multiple'
s = s.replace(old, "ksp.getPC().setType('fieldsplit')", 1)
anchor = 'ksp.setOperators(A)'
assert s.count(anchor) == 1, 'Point de configuration KSP introuvable'
helper = '''\n\ndef configure_schur_v12():
    """Deux blocs : ions (Na, Cl) et potentiel. Un seul processus MPI."""
    ions = np.unique(np.concatenate([
        W.sub(0).dofmap.list.flatten(), W.sub(1).dofmap.list.flatten()
    ])).astype(PETSc.IntType)
    potential = np.unique(W.sub(2).dofmap.list.flatten()).astype(PETSc.IntType)
    if len(ions) + len(potential) != len(w.x.array):
        raise RuntimeError('Partition des DOFs incomplete ou redondante')
    is_ions = PETSc.IS().createGeneral(ions, comm=mesh.comm)
    is_phi = PETSc.IS().createGeneral(potential, comm=mesh.comm)
    pc = ksp.getPC()
    pc.setType('fieldsplit')
    pc.setFieldSplitIS(('ions', is_ions), ('potential', is_phi))
    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)
    pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.SELFP)
    # PETSc options for the two inner solvers. Schur SELF P is approximate.
    options = PETSc.Options()
    options['fieldsplit_ions_ksp_type'] = 'preonly'
    options['fieldsplit_ions_pc_type'] = 'ilu'
    options['fieldsplit_potential_ksp_type'] = 'gmres'
    options['fieldsplit_potential_ksp_max_it'] = str(int(os.getenv('ED_SCHUR_MAX', '20')))
    options['fieldsplit_potential_ksp_rtol'] = '1e-2'
    options['fieldsplit_potential_pc_type'] = 'none'
    ksp.setFromOptions()
    print('V12 FieldSplit Schur: ions=', len(ions), 'potentiel=', len(potential), flush=True)
    is_ions.destroy(); is_phi.destroy()
\n'''
anchor2 = '\ndef enforce_state('
assert s.count(anchor2) == 1
s = s.replace(anchor2, helper + anchor2, 1)
# Configure once before any Newton iterations, after forms and W have been created.
anchor3 = "if __name__ == '__main__':"
assert s.count(anchor3) == 1
s = s.replace(anchor3, anchor3 + "\n    configure_schur_v12()", 1)
s = s.replace('Newton v11 normalise:', 'Newton v12 Schur:')
compile(s, str(out), 'exec')
out.write_text(s)
print('Cree :', out.name)
print('Source intacte :', source.name)
print('Essai experimental : ne prouve ni inversibilite ni validite physique.')
