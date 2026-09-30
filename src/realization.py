"""Appendix-C realization construction validation (Experiment 5).

Implements a computational version of the proof pipeline:
  1. Continuous outer approximations G_n >= G
  2. Sup-convolution regularization of a viscosity subsolution u
  3. Ellipticity floor lambda_j -> 0
  4. Continuous coefficient selector zeta_j
  5. Realizing laws P_{n,j} for linearized coefficients
  6. Stopped realization inequality

Custom — Context7 found no library equivalent for the Appendix-C approximation
pipeline (paper Appendix C / realization theorem proof).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from .pde_solver import pde_implicit_euler_snapshots


# ---------------------------------------------------------------------------
# Outer approximations
# ---------------------------------------------------------------------------

def outer_approx_G_n(
    x: np.ndarray,
    a_base: np.ndarray,
    n: int,
    H: np.ndarray,
) -> np.ndarray:
    """Outer approximation: G_n(x, H) = 0.5 * (a(x) + 1/n) * H.

    For H > 0 (positive definite direction), G_n >= G.

    Parameters
    ----------
    x      : spatial grid
    a_base : base local variance a(x)
    n      : approximation index (n -> infty gives G)
    H      : Hessian value (scalar or array)

    Returns
    -------
    G_n values
    """
    a_n = a_base + 1.0 / n
    return 0.5 * a_n * H


def outer_approx_q_n(
    q_base: np.ndarray,
    n: int,
) -> np.ndarray:
    """Source term outer approximation: q_n = q + 1/n >= q."""
    return q_base + 1.0 / n


# ---------------------------------------------------------------------------
# Sup-convolution regularization of a subsolution u
# ---------------------------------------------------------------------------

def sup_convolution(
    u: np.ndarray,
    x: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Sup-convolution: u^eps(x) = sup_{y} {u(y) - |x-y|^2 / (2*eps)}.

    Approximates u from above and improves semiconvexity.

    Parameters
    ----------
    u   : subsolution values on grid x
    x   : spatial grid
    eps : regularization parameter > 0

    Returns
    -------
    u_eps : regularized values, shape same as u
    """
    N = len(x)
    u_eps = np.zeros(N)
    for i in range(N):
        candidates = u - (x - x[i]) ** 2 / (2.0 * eps)
        u_eps[i] = np.max(candidates)
    return u_eps


def mollify(
    u: np.ndarray,
    x: np.ndarray,
    sigma_mol: float,
) -> np.ndarray:
    """Gaussian mollification of u to produce a smooth approximant.

    Parameters
    ----------
    u         : values on uniform grid
    x         : grid
    sigma_mol : mollification radius (grid units)

    Returns
    -------
    v : smoothed version of u
    """
    dx = x[1] - x[0]
    sigma_pts = sigma_mol / dx
    return gaussian_filter1d(u, sigma=sigma_pts)


# ---------------------------------------------------------------------------
# Ellipticity floor and coefficient selectors
# ---------------------------------------------------------------------------

def add_ellipticity_floor(
    a: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Return a + lambda to enforce uniform ellipticity with floor lambda."""
    return a + lam


def build_selector_zeta(
    v: np.ndarray,
    x: np.ndarray,
    a_set: tuple[float, float],
) -> np.ndarray:
    """Select diffusion coefficient zeta_j(x) from support A_G(x) = [a_min, a_max].

    For finite-control HJB, the selector is the maximizing coefficient given
    the sign of v'' at each point. For a >= 0 set: select a_max where v'' > 0
    (to match the sup), otherwise a_min. Both are valid since the operator
    is linear in a.

    Parameters
    ----------
    v      : smoothed subsolution
    x      : grid
    a_set  : (a_min, a_max) admissible diffusion range

    Returns
    -------
    zeta : selected diffusion values on grid
    """
    dx = x[1] - x[0]
    # Finite-difference second derivative
    v_xx = np.gradient(np.gradient(v, dx), dx)
    a_min, a_max = a_set
    # Maximizing selector: sup over a in [a_min, a_max] of 0.5 * a * v_xx
    zeta = np.where(v_xx >= 0.0, a_max, a_min)
    return zeta.astype(float)


# ---------------------------------------------------------------------------
# Realizing law via PDE with selected coefficients
# ---------------------------------------------------------------------------

def construct_realizing_law_pde(
    f: np.ndarray,
    x: np.ndarray,
    zeta: np.ndarray,
    t_grid: np.ndarray,
    n_steps_per_unit: int = 200,
) -> dict[float, np.ndarray]:
    """Compute the semigroup T_t^{zeta} f by solving du/dt = 0.5 zeta(x) u_xx.

    This approximates E^{P_{n,j}}[f(X_t)] for the linear diffusion with
    state-dependent coefficient zeta_j(x).

    Parameters
    ----------
    f              : initial condition
    x              : grid
    zeta           : selected diffusion coefficient array
    t_grid         : output times
    n_steps_per_unit : resolution

    Returns
    -------
    dict t -> u(t, ·)
    """
    return pde_implicit_euler_snapshots(
        f=f,
        x=x,
        a=zeta,
        t_grid=t_grid,
        n_steps_per_unit=n_steps_per_unit,
    )


# ---------------------------------------------------------------------------
# Stopped realization inequality
# ---------------------------------------------------------------------------

def check_stopped_inequality(
    u: np.ndarray,
    T_zeta_f: np.ndarray,
    x: np.ndarray,
    compact_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Verify the stopped realization inequality: u(x) <= T_t^{zeta} f(x).

    Returns diagnostics including max violation and fraction satisfied.

    Parameters
    ----------
    u           : subsolution values at time 0 (or appropriate time)
    T_zeta_f    : realizing law semigroup value at horizon T
    x           : grid
    compact_mask: boolean mask for compact subset

    Returns
    -------
    dict with 'max_violation', 'mean_error', 'fraction_satisfied'
    """
    if compact_mask is None:
        compact_mask = np.ones(len(x), dtype=bool)

    diff = u[compact_mask] - T_zeta_f[compact_mask]  # should be <= 0
    max_viol = float(np.max(diff))
    mean_err = float(np.mean(diff))
    frac_ok = float(np.mean(diff <= 1e-10))

    return {
        "max_violation": max_viol,
        "mean_error": mean_err,
        "fraction_satisfied": frac_ok,
    }


# ---------------------------------------------------------------------------
# Full pipeline for one benchmark
# ---------------------------------------------------------------------------

def run_realization_pipeline(
    u_subsol: np.ndarray,
    a_base: np.ndarray,
    x: np.ndarray,
    q_base: np.ndarray,
    t: float,
    n_seq: list[int],
    eps_seq: list[float],
    lam_seq: list[float],
    sigma_mol_seq: list[float],
    a_set: tuple[float, float],
    n_steps_per_unit: int = 200,
) -> list[dict]:
    """Run the full Appendix-C approximation pipeline.

    For each (n, eps, lam, sigma_mol) combination (diagonal sequence),
    build v_{n,j} and check the stopped realization inequality.

    Returns
    -------
    list of diagnostic dicts for each diagonal step
    """
    results = []
    compact_mask = (x > x[0] + 0.2 * (x[-1] - x[0])) & (x < x[-1] - 0.2 * (x[-1] - x[0]))

    for step_idx, (n, eps, lam, sigma_mol) in enumerate(zip(n_seq, eps_seq, lam_seq, sigma_mol_seq)):
        # Step 1: outer approximation coefficient
        a_n = a_base + 1.0 / n

        # Step 2: sup-convolution regularization of u
        u_eps = sup_convolution(u_subsol, x, eps)

        # Step 3: mollify
        v_nj = mollify(u_eps, x, sigma_mol)

        # Step 4: ellipticity floor
        a_nj = add_ellipticity_floor(a_n, lam)

        # Step 5: coefficient selector
        zeta = build_selector_zeta(v_nj, x, a_set)
        zeta = np.maximum(zeta, lam)  # enforce floor

        # Step 6: realizing law PDE
        law_snaps = construct_realizing_law_pde(
            v_nj, x, zeta, np.array([t]), n_steps_per_unit
        )
        T_zeta_f = law_snaps[t]

        # Step 7: check stopped inequality
        diag = check_stopped_inequality(v_nj, T_zeta_f, x, compact_mask)

        # Residual: how much v_nj is NOT a subsolution of the PDE
        target = 0.5 * a_nj * np.gradient(np.gradient(v_nj, x[1] - x[0]), x[1] - x[0])
        v_nj_time_deriv_approx = (T_zeta_f - v_nj) / t
        residual = v_nj_time_deriv_approx - target
        diag.update({
            "step": step_idx,
            "n": n,
            "eps": eps,
            "lam": lam,
            "sigma_mol": sigma_mol,
            "max_residual": float(np.max(np.abs(residual[compact_mask]))),
        })
        results.append(diag)

    return results
