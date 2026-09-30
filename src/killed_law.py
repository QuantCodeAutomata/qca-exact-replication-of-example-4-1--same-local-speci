"""Killed-law / discounted-pair equivalence (Theorem 6.4).

Implements the map Phi: (A, Q) -> P, where:
  - Q is a continuous-path law (no killing)
  - A_t = integral_0^t k_s ds is a cumulative discount process
  - P = Phi(A, Q) is the induced killed law

Validates: E^P[f(X_T) 1_{T < tau_infty}] = E^Q[e^{-A_T} f(X_T)]

Custom — Context7 found no library equivalent for killed-law / discounted-pair
transformation (paper Theorem 6.4).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# SDE simulation under Q (no killing)
# ---------------------------------------------------------------------------

def simulate_ou(
    x0: float,
    n_paths: int,
    n_steps: int,
    dt: float,
    theta: float = 1.0,
    mu: float = 0.0,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Euler-Maruyama simulation of Ornstein-Uhlenbeck: dX = -theta*(X-mu)*dt + sigma*dW.

    Parameters
    ----------
    x0      : initial state (scalar, same for all paths)
    n_paths : number of Monte Carlo paths
    n_steps : number of time steps
    dt      : time step size
    theta   : mean-reversion speed
    mu      : long-run mean
    sigma   : diffusion coefficient
    rng     : numpy random Generator (for reproducibility)

    Returns
    -------
    X : paths array, shape (n_paths, n_steps+1)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = x0

    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    for j in range(n_steps):
        X[:, j + 1] = (
            X[:, j]
            + (-theta * (X[:, j] - mu)) * dt
            + sigma * dW[:, j]
        )
    return X


def simulate_bm_drift(
    x0: float,
    n_paths: int,
    n_steps: int,
    dt: float,
    b: float = 0.0,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Euler-Maruyama simulation of dX = b*dt + sigma*dW (drifted BM).

    Returns
    -------
    X : shape (n_paths, n_steps+1)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = x0
    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    for j in range(n_steps):
        X[:, j + 1] = X[:, j] + b * dt + sigma * dW[:, j]
    return X


# ---------------------------------------------------------------------------
# Cumulative discount and killing
# ---------------------------------------------------------------------------

def cumulative_discount(
    X: np.ndarray,
    k_func,
    dt: float,
) -> np.ndarray:
    """Compute A_t = integral_0^t k(X_s) ds along each path via Riemann sum.

    Parameters
    ----------
    X      : paths array, shape (n_paths, n_steps+1)
    k_func : callable(x) -> nonneg rate, applied element-wise
    dt     : time step

    Returns
    -------
    A : cumulative discount, shape (n_paths, n_steps+1), A[:,0] = 0
    """
    rates = k_func(X)                 # shape (n_paths, n_steps+1)
    A = np.cumsum(rates * dt, axis=1)
    A = np.hstack([np.zeros((A.shape[0], 1)), A[:, :-1]])
    return A


def apply_killing(
    X: np.ndarray,
    A: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate killing times tau via the exponential-clock construction.

    For each path i: sample E_i ~ Exp(1), set tau_i = inf{t: A_t >= E_i}.
    The path is alive at step j iff A_{j*dt} < E_i.

    Parameters
    ----------
    X   : paths array, shape (n_paths, n_steps+1)
    A   : cumulative discount, shape (n_paths, n_steps+1)
    rng : random generator

    Returns
    -------
    alive : boolean alive-flag, shape (n_paths, n_steps+1)
    tau_idx : index of killing step for each path (-1 if not killed)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_paths, n_steps1 = A.shape
    E = rng.exponential(scale=1.0, size=n_paths)  # Exp(1) clocks

    alive = A < E[:, np.newaxis]   # shape (n_paths, n_steps+1)

    # Find first step where alive becomes False
    tau_idx = np.full(n_paths, n_steps1 - 1, dtype=int)  # default: not killed
    for i in range(n_paths):
        killed_steps = np.where(~alive[i])[0]
        if len(killed_steps) > 0:
            tau_idx[i] = killed_steps[0] - 1   # last alive step index

    return alive, tau_idx


# ---------------------------------------------------------------------------
# Validation: killed expectation vs discounted expectation
# ---------------------------------------------------------------------------

def killed_expectation(
    X: np.ndarray,
    alive: np.ndarray,
    f_func,
    T_idx: int,
) -> float:
    """E^P[f(X_T) * 1_{T < tau_infty}].

    Parameters
    ----------
    X      : paths, shape (n_paths, n_steps+1)
    alive  : boolean alive-flag
    f_func : payoff callable
    T_idx  : time index corresponding to horizon T

    Returns
    -------
    scalar Monte Carlo estimate
    """
    survivors = alive[:, T_idx]
    vals = f_func(X[:, T_idx])
    return float(np.mean(vals * survivors))


def discounted_expectation(
    X: np.ndarray,
    A: np.ndarray,
    f_func,
    T_idx: int,
) -> float:
    """E^Q[exp(-A_T) * f(X_T)].

    Parameters
    ----------
    X      : paths under Q (no killing), shape (n_paths, n_steps+1)
    A      : cumulative discount, shape (n_paths, n_steps+1)
    f_func : payoff callable
    T_idx  : time index corresponding to horizon T

    Returns
    -------
    scalar Monte Carlo estimate
    """
    A_T = A[:, T_idx]
    vals = f_func(X[:, T_idx])
    return float(np.mean(np.exp(-A_T) * vals))


def validate_theorem_6_4(
    x0: float,
    n_paths: int,
    n_steps: int,
    T: float,
    k_func,
    f_func,
    process: str = "ou",
    **process_kwargs,
) -> dict[str, float]:
    """End-to-end validation of E^P[...] = E^Q[e^{-A_T} ...].

    Parameters
    ----------
    x0            : initial state
    n_paths       : Monte Carlo sample size
    n_steps       : time steps (T / n_steps = dt)
    T             : horizon
    k_func        : discount rate function k(x)
    f_func        : payoff f(x)
    process       : 'ou' or 'bm'
    **process_kwargs : passed to the simulator

    Returns
    -------
    dict with keys 'killed_exp', 'disc_exp', 'abs_error', 'rel_error'
    """
    dt = T / n_steps
    rng = np.random.default_rng(process_kwargs.pop("seed", 2024))

    if process == "ou":
        X = simulate_ou(x0=x0, n_paths=n_paths, n_steps=n_steps, dt=dt,
                        rng=rng, **process_kwargs)
    else:
        X = simulate_bm_drift(x0=x0, n_paths=n_paths, n_steps=n_steps, dt=dt,
                              rng=rng, **process_kwargs)

    A = cumulative_discount(X, k_func, dt)
    rng2 = np.random.default_rng(process_kwargs.pop("kill_seed", 2025) if False else 7777)
    alive, _ = apply_killing(X, A, rng=rng2)

    T_idx = n_steps  # last column corresponds to time T

    ke = killed_expectation(X, alive, f_func, T_idx)
    de = discounted_expectation(X, A, f_func, T_idx)

    abs_err = abs(ke - de)
    denom = max(abs(de), 1e-12)

    return {
        "killed_exp": ke,
        "disc_exp": de,
        "abs_error": abs_err,
        "rel_error": abs_err / denom,
    }


# ---------------------------------------------------------------------------
# Recovery of A and Q from a killed law
# ---------------------------------------------------------------------------

def recover_discount_process(
    alive: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Estimate A_t = -log S_t from empirical survival probabilities.

    S_t = P(tau > t) estimated by fraction of alive paths at each time.

    Parameters
    ----------
    alive : boolean alive-flag, shape (n_paths, n_steps+1)
    dt    : time step

    Returns
    -------
    A_hat : estimated cumulative discount, shape (n_steps+1,)
    """
    S = np.mean(alive.astype(float), axis=0)  # survival fractions
    S = np.maximum(S, 1e-12)
    A_hat = -np.log(S)
    return A_hat
