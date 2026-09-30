"""1-D PDE solver for degenerate and non-degenerate parabolic equations.

Solves forward-in-time problems of the form:
    du/dt = 0.5 * a(x) * d^2u/dx^2 + b(x) * du/dx + c(x) * u
    u(0, x) = f(x)

using implicit Euler time-stepping and tridiagonal sparse linear algebra.
For HJB equations uses explicit Euler with monotone upwind for the |p| term.

Custom — Context7 found no library equivalent for degenerate-diffusion PDE
branch selection (paper methodology, Section 4).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import spsolve


def build_diffusion_matrix(
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray | None = None,
    c: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the second-order finite-difference matrix L for operator
    L u = 0.5 * a(x) * u'' + b(x) * u' + c(x) * u.

    Uses central differences for the diffusion term and centered for drift.
    Returns a (N x N) dense matrix; sparse callers should convert if needed.

    Parameters
    ----------
    x  : 1-D grid array of length N
    a  : diffusion coefficients at each grid point
    b  : drift coefficients (optional, default zero)
    c  : zeroth-order (killing-rate) coefficients (optional, default zero)
    """
    N = len(x)
    dx = x[1] - x[0]  # uniform grid assumed
    if b is None:
        b = np.zeros(N)
    if c is None:
        c = np.zeros(N)

    # Diffusion: 0.5 * a_j * (u_{j+1} - 2*u_j + u_{j-1}) / dx^2
    d_main = -a / dx ** 2 + c          # diagonal
    d_upper = 0.5 * a[:-1] / dx ** 2 + b[:-1] / (2.0 * dx)   # super-diagonal
    d_lower = 0.5 * a[1:] / dx ** 2  - b[1:] / (2.0 * dx)    # sub-diagonal

    L = sp_diags([d_lower, d_main, d_upper], [-1, 0, 1], shape=(N, N),
                 format="csr")
    return L


def pde_implicit_euler(
    f: np.ndarray,
    x: np.ndarray,
    a: np.ndarray,
    t_end: float,
    n_steps: int,
    b: np.ndarray | None = None,
    c: np.ndarray | None = None,
    absorb_idx: int | None = None,
) -> np.ndarray:
    """Solve du/dt = L u backward/forward to time t_end via implicit Euler.

    Parameters
    ----------
    f         : initial condition u(0, ·), shape (N,)
    x         : spatial grid of length N
    a         : local variance coefficients a(x), shape (N,)
    t_end     : final time T
    n_steps   : number of time steps
    b         : drift coefficients (optional)
    c         : zeroth-order coefficients (optional)
    absorb_idx: grid index of absorbing point (Dirichlet u=f[absorb_idx] fixed)

    Returns
    -------
    u : solution at time t_end, shape (N,)
    """
    dt = t_end / n_steps
    N = len(x)
    L = build_diffusion_matrix(x, a, b, c)

    # Identity - dt*L (implicit Euler system matrix)
    from scipy.sparse import eye as sp_eye
    I = sp_eye(N, format="csr")
    A_sys = I - dt * L

    u = f.copy().astype(float)

    for _ in range(n_steps):
        if absorb_idx is not None:
            # Freeze the absorbing point: enforce u[absorb_idx] = f[absorb_idx]
            rhs = u.copy()
            rhs[absorb_idx] = f[absorb_idx]  # RHS fixed
            u_new = spsolve(A_sys, rhs)
            u_new[absorb_idx] = f[absorb_idx]
        else:
            u_new = spsolve(A_sys, u)
        u = u_new

    return u


def pde_implicit_euler_snapshots(
    f: np.ndarray,
    x: np.ndarray,
    a: np.ndarray,
    t_grid: np.ndarray,
    n_steps_per_unit: int = 200,
    b: np.ndarray | None = None,
    c: np.ndarray | None = None,
    absorb_idx: int | None = None,
) -> dict[float, np.ndarray]:
    """Solve PDE at each time in t_grid and return snapshots.

    Parameters
    ----------
    t_grid            : sorted array of output times (must start > 0)
    n_steps_per_unit  : time steps per unit time (controls accuracy)

    Returns
    -------
    dict mapping t -> u(t, ·)
    """
    from scipy.sparse import eye as sp_eye

    results: dict[float, np.ndarray] = {}
    N = len(x)
    L = build_diffusion_matrix(x, a, b, c)
    I = sp_eye(N, format="csr")

    u = f.copy().astype(float)
    t_current = 0.0
    t_sorted = np.sort(t_grid)

    for t_target in t_sorted:
        if t_target <= t_current:
            results[t_target] = u.copy()
            continue
        dt_seg = t_target - t_current
        n_seg = max(1, int(np.round(dt_seg * n_steps_per_unit)))
        dt = dt_seg / n_seg
        A_sys = I - dt * L

        for _ in range(n_seg):
            if absorb_idx is not None:
                rhs = u.copy()
                rhs[absorb_idx] = f[absorb_idx]
                u_new = spsolve(A_sys, rhs)
                u_new[absorb_idx] = f[absorb_idx]
            else:
                u_new = spsolve(A_sys, u)
            u = u_new

        t_current = t_target
        results[t_target] = u.copy()

    return results


def hjb_pde_1d(
    f: np.ndarray,
    x: np.ndarray,
    sigma: float,
    mu: float,
    t_end: float,
    n_steps: int,
) -> np.ndarray:
    """Solve HJB PDE: du/dt = mu*|du/dx| + 0.5*sigma^2*d^2u/dx^2.

    Model A: G(x,U) = mu*|p| + 0.5*sigma^2*H.
    Uses explicit upwind for the |p| term, implicit for diffusion.

    Parameters
    ----------
    f      : initial payoff, shape (N,)
    x      : grid of length N
    sigma  : diffusion coefficient
    mu     : drift magnitude (robust supremum over {-mu, +mu})
    t_end  : horizon
    n_steps: number of time steps

    Returns
    -------
    u : solution at t_end, shape (N,)
    """
    dx = x[1] - x[0]
    dt = t_end / n_steps
    N = len(x)

    # Stability check for explicit upwind (CFL)
    cfl_adv = mu * dt / dx
    cfl_diff = 0.5 * sigma ** 2 * dt / dx ** 2
    if cfl_adv + cfl_diff > 1.0:
        # Use sub-stepping
        sub = int(np.ceil((cfl_adv + cfl_diff) * 2))
        dt = dt / sub
        n_steps = n_steps * sub

    a_arr = np.full(N, sigma ** 2)
    from scipy.sparse import eye as sp_eye
    L_diff = build_diffusion_matrix(x, a_arr)
    I = sp_eye(N, format="csr")
    A_imp = I - dt * L_diff

    u = f.copy().astype(float)

    for _ in range(n_steps):
        # Upwind approximation of mu * |u'|
        du_fw = np.roll(u, -1) - u   # u_{j+1} - u_j  (forward difference)
        du_bk = u - np.roll(u, 1)    # u_j - u_{j-1}  (backward difference)
        du_fw[-1] = du_fw[-2]
        du_bk[0] = du_bk[1]
        abs_p_upwind = np.maximum(du_fw, 0.0) / dx - np.minimum(du_bk, 0.0) / dx
        # Godunov scheme for H(p) = mu |p|: max_u(p*u - L*(u)) with |u|<=mu
        # = mu * max(|p+|, |p-|) approx
        p_plus = np.maximum(du_fw / dx, 0.0)
        p_minus = np.minimum(du_bk / dx, 0.0)
        god = mu * np.maximum(np.abs(p_plus), np.abs(p_minus))

        rhs = u + dt * god   # explicit Hamiltonian step
        u = spsolve(A_imp, rhs)

    return u
