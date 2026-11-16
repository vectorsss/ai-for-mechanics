"""Task A data: random fibre layouts and their FEM stress maps.

Plate: the unit square, a soft matrix (E_m = 1) with stiff circular fibres
(E_f = 10), Poisson's ratio 0.3, linear elasticity, plane stress.
Boundary conditions: left edge u_x = 0, right edge u_x = eps_bar (1 % of the
width), bottom-left corner u_y = 0; top and bottom edges are traction-free.

Mesh: n x n bilinear square elements (Q1). An element belongs to a fibre if
its centre lies inside one, so fibre boundaries are stepped.

Input  x = log10 E in each element (0 in the matrix, 1 in a fibre), (n, n)
Output s = von Mises stress at each element centre / (E_m * eps_bar), (n, n)
       s = 1 is the stress of a plate without fibres.

Arrays are indexed [row, col] = [y, x] with row 0 at the bottom of the plate
(plot them with origin="lower").

    python fem.py      # 1000 layouts on a 32 x 32 mesh -> data/taskA.npz
"""
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

E_M, E_F, NU, EPS_BAR = 1.0, 10.0, 0.3, 0.01


# ---------------------------------------------------------------- geometry
def random_fibres(rng):
    """N_f ~ U{3,...,8} discs, centres ~ U(0,1)^2, radii ~ U(0.06, 0.15).
    Discs may overlap or be cut by the edge of the plate."""
    nf = rng.integers(3, 9)
    return rng.uniform(0.0, 1.0, (nf, 2)), rng.uniform(0.06, 0.15, nf)


def rasterize(fibres, n):
    """Young's modulus of each element of an n x n mesh, shape (n, n)."""
    centres, radii = fibres
    c = (np.arange(n) + 0.5) / n                 # element centres
    X, Y = np.meshgrid(c, c)                     # X[j, i] = x_i, Y[j, i] = y_j
    inside = np.zeros((n, n), dtype=bool)
    for (px, py), r in zip(centres, radii):
        inside |= (X - px) ** 2 + (Y - py) ** 2 < r ** 2
    return np.where(inside, E_F, E_M)


# ----------------------------------------------------------- Q1 element, E = 1
# Local nodes in natural coordinates (xi, eta) in [-1, 1]^2, counter-clockwise.
_NODES = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
_D1 = np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]]) / (1 - NU ** 2)


def _B(xi, eta, h):
    """Strain-displacement matrix (3 x 8) of a square element of side h.
    Dof order: [ux0, uy0, ux1, uy1, ux2, uy2, ux3, uy3]."""
    dN_dxi = _NODES[:, 0] * (1 + eta * _NODES[:, 1]) / 4
    dN_deta = _NODES[:, 1] * (1 + xi * _NODES[:, 0]) / 4
    dN_dx, dN_dy = dN_dxi * 2 / h, dN_deta * 2 / h
    B = np.zeros((3, 8))
    B[0, 0::2] = dN_dx
    B[1, 1::2] = dN_dy
    B[2, 0::2] = dN_dy
    B[2, 1::2] = dN_dx
    return B


def _reference_matrices(h):
    """K_ref: element stiffness for E = 1 (2 x 2 Gauss points).
    S_ref: maps element displacements to the stress at the centre, E = 1."""
    g = 1 / np.sqrt(3)
    K = sum(_B(xi, eta, h).T @ _D1 @ _B(xi, eta, h) * h * h / 4
            for xi in (-g, g) for eta in (-g, g))
    return K, _D1 @ _B(0.0, 0.0, h)


class Mesh:
    """Everything about an n x n mesh that does not depend on the layout."""

    def __init__(self, n):
        self.n = n
        h = 1.0 / n
        self.K_ref, self.S_ref = _reference_matrices(h)

        node = lambda i, j: j * (n + 1) + i        # i: x index, j: y index
        ii, jj = np.meshgrid(np.arange(n), np.arange(n))      # element (row j, col i)
        ii, jj = ii.ravel(), jj.ravel()
        conn = np.stack([node(ii, jj), node(ii + 1, jj),
                         node(ii + 1, jj + 1), node(ii, jj + 1)], axis=1)
        self.edofs = np.repeat(2 * conn, 2, axis=1) + np.tile([0, 1], 4)  # (n^2, 8)
        self.rows = np.repeat(self.edofs, 8, axis=1).ravel()
        self.cols = np.tile(self.edofs, (1, 8)).ravel()

        ndof = 2 * (n + 1) ** 2
        j = np.arange(n + 1)
        left, right = 2 * node(0, j), 2 * node(n, j)          # u_x dofs
        corner = np.array([2 * node(0, 0) + 1])               # u_y dof
        self.p = np.concatenate([left, right, corner])        # prescribed dofs
        self.u_p = np.concatenate([np.zeros(n + 1), np.full(n + 1, EPS_BAR * 1.0), [0.0]])
        self.f = np.setdiff1d(np.arange(ndof), self.p)        # free dofs
        self.ndof = ndof


def solve(E, mesh):
    """FEM stress map s for a modulus map E of shape (n, n)."""
    m = mesh
    vals = (E.ravel()[:, None, None] * m.K_ref).ravel()
    K = sp.csr_matrix((vals, (m.rows, m.cols)), shape=(m.ndof, m.ndof))
    # Minimising the strain energy under the constraints: K_ff u_f = -K_fp u_p
    u = np.zeros(m.ndof)
    u[m.p] = m.u_p
    u[m.f] = spla.spsolve(K[m.f][:, m.f].tocsc(), -K[m.f][:, m.p] @ m.u_p)
    sig = E.ravel()[:, None] * (u[m.edofs] @ m.S_ref.T)     # (n^2, 3) at centres
    sxx, syy, sxy = sig.T
    vm = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3 * sxy ** 2)
    return (vm / (E_M * EPS_BAR)).reshape(m.n, m.n)


# ------------------------------------------------------------------- dataset
def make_dataset(num=1000, n=32, seed=0):
    rng = np.random.default_rng(seed)
    mesh = Mesh(n)
    X = np.empty((num, n, n), np.float32)
    Y = np.empty((num, n, n), np.float32)
    for k in range(num):
        E = rasterize(random_fibres(rng), n)
        X[k], Y[k] = np.log10(E), solve(E, mesh)
    return X, Y


def main():
    n = 32
    # Sanity check: a plate without fibres is in uniform uniaxial tension, s = 1.
    s0 = solve(np.full((n, n), E_M), Mesh(n))
    assert np.allclose(s0, 1.0), "homogeneous plate should give s = 1"

    t = time.perf_counter()
    X, Y = make_dataset(num=1000, n=n, seed=0)
    t = time.perf_counter() - t

    out = Path("data")
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "taskA.npz", x_train=X[:800], y_train=Y[:800],
                        x_test=X[800:], y_test=Y[800:])
    print(f"1000 layouts on a {n} x {n} mesh in {t:.1f} s -> data/taskA.npz")
    print(f"fibre fraction {np.mean(X):.2f}, s: mean {Y.mean():.2f}, max {Y.max():.2f}")


if __name__ == "__main__":
    main()
