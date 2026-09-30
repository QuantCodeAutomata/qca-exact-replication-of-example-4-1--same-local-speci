"""Generator estimation via difference quotients.

For a valuation semigroup T, the upper generator is defined by:
    G_T f(x) = limsup_{h downarrow 0}  Delta_h^T f(x)
where
    Delta_h^T f(x) = (T_h f(x) - f(x)) / h

This module provides tools to estimate the generator numerically and compare
it against the theoretical local specification G_alpha(x,H) = 0.5*a_alpha(x)*H.

Custom — Context7 found no library equivalent for semigroup generator estimation
via small-h difference quotients (paper Section 4).
"""

from __future__ import annotations

import numpy as np

from .local_spec import a_alpha


def difference_quotient_semigroup(
    semigroup_eval,
    f_vals: np.ndarray,
    x: np.ndarray,
    h: float,
) -> np.ndarray:
    """Compute Delta_h^T f(x) = (T_h f(x) - f(x)) / h.

    Parameters
    ----------
    semigroup_eval : callable(f_vals, t_grid) -> dict{t: u}
                     Evaluates the semigroup at a single time h.
    f_vals         : f evaluated on x, shape (N,)
    x              : spatial grid, shape (N,)
    h              : small positive time step

    Returns
    -------
    Delta : difference quotient array, shape (N,)
    """
    snaps = semigroup_eval(f_vals, np.array([h]))
    T_h_f = snaps[h]
    return (T_h_f - f_vals) / h


def generator_residual(
    semigroup_eval,
    f_vals: np.ndarray,
    f_double_prime: np.ndarray,
    x: np.ndarray,
    alpha: float,
    h: float,
) -> np.ndarray:
    """Compute Delta_h^T f(x) - 0.5*a_alpha(x)*f''(x) (generator residual).

    A small residual confirms the semigroup has local specification G_alpha.

    Parameters
    ----------
    semigroup_eval  : semigroup evaluator callable
    f_vals          : f on grid x
    f_double_prime  : f''(x), analytically computed
    x               : spatial grid
    alpha           : volatility exponent
    h               : difference-quotient step

    Returns
    -------
    residual : array of pointwise errors, shape (N,)
    """
    dq = difference_quotient_semigroup(semigroup_eval, f_vals, x, h)
    target = 0.5 * a_alpha(x, alpha) * f_double_prime
    return dq - target


def convergence_study(
    semigroup_eval,
    f_vals: np.ndarray,
    f_double_prime: np.ndarray,
    x: np.ndarray,
    alpha: float,
    h_seq: np.ndarray,
    compact_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run generator convergence study over a sequence of h values.

    For each h in h_seq, compute max_x |Delta_h f(x) - 0.5 a_alpha(x) f''(x)|
    on the compact subset defined by compact_mask.

    Parameters
    ----------
    semigroup_eval : callable
    f_vals         : payoff on grid
    f_double_prime : f'' on grid
    x              : grid
    alpha          : exponent
    h_seq          : decreasing sequence of h values
    compact_mask   : boolean mask for x in compact subset

    Returns
    -------
    dict with keys 'h', 'max_residual', 'mean_residual'
    """
    if compact_mask is None:
        compact_mask = np.ones(len(x), dtype=bool)

    max_res = []
    mean_res = []

    for h in h_seq:
        res = generator_residual(semigroup_eval, f_vals, f_double_prime, x, alpha, h)
        max_res.append(float(np.max(np.abs(res[compact_mask]))))
        mean_res.append(float(np.mean(np.abs(res[compact_mask]))))

    return {
        "h": np.array(h_seq),
        "max_residual": np.array(max_res),
        "mean_residual": np.array(mean_res),
    }


def compute_valuation_gap(
    abs_val_at_zero: dict[float, float],
    zero_val_at_zero: dict[float, float],
) -> dict[float, float]:
    """Compute T^0_t f(0) - T^{abs}_t f(0) for each t.

    The key success criterion: this gap is strictly positive for some t_0 > 0.
    """
    return {
        t: zero_val_at_zero[t] - abs_val_at_zero[t]
        for t in sorted(abs_val_at_zero.keys())
        if t in zero_val_at_zero
    }
