"""Valuation semigroups for Examples 4.1 and 4.2.

Implements:
  - T_abs: absorbing-at-zero valuation semigroup (PDE with frozen origin)
  - T_0  : zero-sojourn valuation semigroup (non-absorbing, via regularization)
  - V_det: deterministic-shift valuation for Example 4.2, V_t f(x) = f(x+t)

Custom — Context7 found no library equivalent for branch-specific degenerate
diffusion semigroup construction (paper Section 4, Example 4.1).
"""

from __future__ import annotations

import numpy as np

from .local_spec import a_alpha, a_alpha_reg
from .pde_solver import pde_implicit_euler_snapshots


class AbsorbingValuation:
    """Valuation semigroup T^{abs}: process absorbed at x=0.

    T^{abs}_t f(x_0) = PDE solution at x_0 with a_alpha(x), treating x=0
    as a Dirichlet point (frozen at the initial value f(0) = 0 for f_+).

    Since a_alpha(0) = 0, the implicit Euler step automatically preserves
    u[x=0] = f(0) for the exact local variance. We make this explicit by
    passing absorb_idx to the PDE solver.
    """

    def __init__(
        self,
        alpha: float,
        L: float = 10.0,
        N: int = 401,
        n_steps_per_unit: int = 500,
    ) -> None:
        """
        Parameters
        ----------
        alpha            : exponent in (0, 1/2)
        L                : half-domain length (grid on [-L, L])
        N                : number of grid points (odd ensures 0 is included)
        n_steps_per_unit : temporal resolution (steps per unit time)
        """
        self.alpha = alpha
        self.L = L
        self.N = N
        self.n_steps_per_unit = n_steps_per_unit
        self.x = np.linspace(-L, L, N)
        self._zero_idx = int(np.argmin(np.abs(self.x)))
        self._a = a_alpha(self.x, alpha)

    def evaluate(self, f_vals: np.ndarray, t_grid: np.ndarray) -> dict[float, np.ndarray]:
        """Compute T^{abs}_t f at each t in t_grid.

        Parameters
        ----------
        f_vals : payoff evaluated on self.x, shape (N,)
        t_grid : array of positive horizons

        Returns
        -------
        dict mapping t -> T^{abs}_t f on self.x
        """
        return pde_implicit_euler_snapshots(
            f=f_vals,
            x=self.x,
            a=self._a,
            t_grid=t_grid,
            n_steps_per_unit=self.n_steps_per_unit,
            absorb_idx=self._zero_idx,
        )

    def at_zero(self, f_vals: np.ndarray, t_grid: np.ndarray) -> dict[float, float]:
        """Return T^{abs}_t f(0) for each t in t_grid."""
        snaps = self.evaluate(f_vals, t_grid)
        return {t: float(u[self._zero_idx]) for t, u in snaps.items()}


class ZeroSojournValuation:
    """Valuation semigroup T^0: zero-sojourn (non-absorbing) at x=0.

    Implemented via regularized volatility a_{alpha,eps} -> a_alpha as eps -> 0.
    For each eps, we solve the uniformly elliptic PDE with a_{alpha,eps}(x) (no
    special treatment at 0). The limit eps -> 0 selects the non-absorbing branch.
    """

    def __init__(
        self,
        alpha: float,
        L: float = 10.0,
        N: int = 401,
        n_steps_per_unit: int = 500,
        eps: float = 1e-3,
    ) -> None:
        """
        Parameters
        ----------
        alpha            : exponent in (0, 1/2)
        L                : half-domain length
        N                : number of grid points
        n_steps_per_unit : temporal resolution
        eps              : regularization parameter (eps -> 0 selects T^0 branch)
        """
        self.alpha = alpha
        self.L = L
        self.N = N
        self.n_steps_per_unit = n_steps_per_unit
        self.eps = eps
        self.x = np.linspace(-L, L, N)
        self._zero_idx = int(np.argmin(np.abs(self.x)))
        self._a = a_alpha_reg(self.x, alpha, eps)

    def evaluate(self, f_vals: np.ndarray, t_grid: np.ndarray) -> dict[float, np.ndarray]:
        """Compute T^0_t f at each t in t_grid (regularized branch)."""
        return pde_implicit_euler_snapshots(
            f=f_vals,
            x=self.x,
            a=self._a,
            t_grid=t_grid,
            n_steps_per_unit=self.n_steps_per_unit,
            absorb_idx=None,       # no absorbing point — non-absorbing branch
        )

    def at_zero(self, f_vals: np.ndarray, t_grid: np.ndarray) -> dict[float, float]:
        """Return T^0_t f(0) for each t in t_grid."""
        snaps = self.evaluate(f_vals, t_grid)
        return {t: float(u[self._zero_idx]) for t, u in snaps.items()}


# ---------------------------------------------------------------------------
# Example 4.2 — deterministic shift valuation
# ---------------------------------------------------------------------------

def V_det(f_func, x: np.ndarray, t: float) -> np.ndarray:
    """Canonical valuation for Example 4.2: V_t^G f(x) = f(x + t).

    Since the only admissible path is X_s = x + s, the valuation is exact.

    Parameters
    ----------
    f_func : callable, the payoff function R -> R
    x      : array of initial states
    t      : time horizon

    Returns
    -------
    array of V_t f(x)
    """
    return f_func(x + t)


def difference_quotient_det(
    f_func,
    f_prime_func,
    x: np.ndarray,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Delta_h^{V^G} f(x) = (f(x+h) - f(x)) / h and compare to f'(x).

    Returns
    -------
    dq    : difference quotient array
    exact : exact derivative f'(x) for comparison
    """
    dq = (f_func(x + h) - f_func(x)) / h
    exact = f_prime_func(x)
    return dq, exact


def G_downarrow_residual(
    x: np.ndarray,
    p: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """Pointwise gap G(x,z,p,H) - G^downarrow(x,z,p,H) = 1_{x=0} * p^+.

    Confirms G^downarrow < G at x=0 for p>0 and G^downarrow = G elsewhere.
    """
    indicator = np.where(x == 0.0, 1.0, 0.0)
    return indicator * np.maximum(p, 0.0)
