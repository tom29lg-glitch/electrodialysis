"""Experimental conservative DG1 periodic Nernst--Planck facet fluxes.

Single-rank triangular DG1 meshes only.  Add the residual and exact local
Jacobian contributions to already assembled PETSc objects.  The normal is
from the left outer facet (-x), and the potential is shifted by V on the
right for the electrochemical-potential jump.  This module does NOT establish
correctness of the remaining membrane or bulk formulation.
"""
import numpy as np


def _basis_at(coords, point):
    """P1 nodal basis and Cartesian gradients at a point in a triangle."""
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (3, 2):
        raise ValueError("DG1 triangle: exactly three scalar DOFs per cell required")
    A = np.column_stack((np.ones(3), coords))
    C = np.linalg.inv(A)  # columns are nodal basis coefficients
    return np.array([1., *point]) @ C, C[1:3, :].T


class PeriodicNPFlux:
    """Paired-face flux and Jacobian, evaluated with midpoint quadrature.

    The PETSc vector/matrix passed to add() must be preassembled, writable,
    and not yet finally assembled; call PETSc assemblyBegin/End afterwards.
    """
    def __init__(self, mesh, W, pair_file, Dna, Dcl, *, beta, voltage=1.5,
                 gamma=10., c_ref=500.):
        if mesh.comm.size != 1:
            raise RuntimeError("Only single MPI rank supported")
        self.mesh, self.W = mesh, W
        self.beta, self.voltage = float(beta), float(voltage)
        self.gamma, self.c_ref = float(gamma), float(c_ref)
        self.D = (Dna, Dcl)
        pairs = np.load(pair_file)
        mesh.topology.create_connectivity(1, 2)
        mesh.topology.create_connectivity(1, 0)
        f2c = mesh.topology.connectivity(1, 2)
        f2v = mesh.topology.connectivity(1, 0)
        V, _ = W.sub(0).collapse()
        xyz = V.tabulate_dof_coordinates()
        self.faces = []
        for fl, fr in zip(pairs['facet_left'], pairs['facet_right']):
            cl, cr = int(f2c.links(int(fl))[0]), int(f2c.links(int(fr))[0])
            vertices_l = mesh.geometry.x[f2v.links(int(fl)), :2]
            vertices_r = mesh.geometry.x[f2v.links(int(fr)), :2]
            length = float(np.linalg.norm(vertices_l[1] - vertices_l[0]))
            if abs(length - np.linalg.norm(vertices_r[1]-vertices_r[0])) > 1e-11:
                raise ValueError("Periodic facet lengths do not match")
            points = (vertices_l.mean(axis=0), vertices_r.mean(axis=0))
            cells = (cl, cr)
            local = []
            for cell, point in zip(cells, points):
                sdofs = V.dofmap.cell_dofs(cell)
                B, G = _basis_at(xyz[sdofs, :2], point)
                ids = [np.asarray(W.sub(k).dofmap.cell_dofs(cell), dtype=np.int64)
                       for k in range(3)]
                if any(len(x) != 3 for x in ids):
                    raise ValueError("Expected three DG1 DOFs per field and cell")
                local.append((B, G, ids))
            self.faces.append((cells, local, length))

    def add(self, w, residual, jacobian=None):
        """Add paired-face contributions to PETSc Vec/Mat in-place."""
        state = w.x.array
        for (cl, cr), local, length in self.faces:
            for species, charge in ((0, 1.), (1, -1.)):
                # 18 local variables: [L Na3, L Cl3, L phi3, R Na3, R Cl3, R phi3]
                ids = np.concatenate([local[s][2][k] for s in range(2) for k in range(3)])
                a = state[ids]
                data = []
                for side in range(2):
                    B, G, _ = local[side]
                    cvec = a[side*9+species*3:side*9+species*3+3]
                    pvec = a[side*9+6:side*9+9]
                    c = float(B @ cvec)
                    if c <= 0:
                        raise ValueError("Nonpositive concentration on periodic facet")
                    dcx = float(G[:, 0] @ cvec)
                    dpx = float(G[:, 0] @ pvec)
                    D = float(self.D[species].x.array[
                        self.D[species].function_space.dofmap.cell_dofs((cl, cr)[side])[0]])
                    data.append((B, G, c, dcx, dpx, D))
                BL, GL, cL, gcL, gpL, DL = data[0]
                BR, GR, cR, gcR, gpR, DR = data[1]
                # Outward normal on left boundary n=(-1,0)
                # N_x = -D*(dc/dx + z*beta*c*dphi/dx)
                flux = .5*(DL*(gcL+charge*self.beta*cL*gpL)
                           + DR*(gcR+charge*self.beta*cR*gpR))
                Dbar = .5*(DL+DR)
                cbar = .5*(cL+cR)
                # h = facet length; simple stabilization (not a proven SIPG scheme)
                K = self.gamma*Dbar/length
                jump = np.log(cL/cR) + charge*self.beta*(
                    (BL @ a[6:9]) - (BR @ a[15:18]) + self.voltage)
                flux += K*cbar*jump
                dflux = np.zeros(18)
                for side, (B,G,c,gc,gp,D) in enumerate(data):
                    sign = 1. if side == 0 else -1.
                    off = side*9
                    # derivative of mean physical flux
                    dflux[off+species*3:off+species*3+3] += .5*D*(
                        G[:, 0]+charge*self.beta*gp*B)
                    dflux[off+6:off+9] += .5*D*charge*self.beta*c*G[:, 0]
                    # derivative of electrochemical jump and cbar
                    dflux[off+species*3:off+species*3+3] += K*(
                        .5*B*jump + cbar*sign*B/c)
                    dflux[off+6:off+9] += K*cbar*sign*charge*self.beta*B
                for side, (B, _, _, _, _, _) in enumerate(data):
                    sign = 1. if side == 0 else -1.
                    rows = local[side][2][species]
                    residual.setValues(rows, sign*length*B*flux, addv=True)
                    if jacobian is not None:
                        jacobian.setValues(rows, ids, sign*length*np.outer(B, dflux), addv=True)
