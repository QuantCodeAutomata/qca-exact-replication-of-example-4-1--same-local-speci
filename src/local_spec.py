"""Local specifications G_alpha (Example 4.1) and G_example2 (Example 4.2).

All formulas follow the paper's notation exactly:
  G_alpha(x, z, p, H) = 0.5 * a_alpha(x) * H
  a_alpha(x)          = sigma_alpha(x)^2
  sigma_alpha(x)      = |x|^alpha / (1 + |x|^alpha)

Example 4.2:
  G(x, z, p, H) = p + 1_{x=0} * p^+
  G_downarrow(x, z, p, H) = p
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Example 4.1 — degenerate local volatility
# ---------------------------------------------------------------------------

def sigma_alpha(x: np.ndarray, alpha: float) -> np.ndarray:
    """Local volatility sigma_alpha(x) = |x|^alpha / (1 + |x|^alpha).

    Parameters
    ----------
    x : array-like state values
    alpha : exponent in (0, 1/2)

    Returns
    -------
    sigma values in [0, 1)
    """
    ax = np.abs(x)
    ax_a = ax ** alpha
    return ax_a / (1.0 + ax_a)


def sigma_alpha_reg(x: np.ndarray, alpha: float, eps: float = 1e-3) -> np.ndarray:
    """Regularized volatility: replaces |x| by sqrt(x^2 + eps^2).

    sigma_{alpha,eps}(x) = (x^2 + eps^2)^{alpha/2} / (1 + (x^2 + eps^2)^{alpha/2})

    Used to select the non-absorbing (zero-sojourn) branch as eps -> 0.
    """
    r = (x ** 2 + eps ** 2) ** (alpha / 2.0)
    return r / (1.0 + r)


def a_alpha(x: np.ndarray, alpha: float) -> np.ndarray:
    """Local variance a_alpha(x) = sigma_alpha(x)^2."""
    s = sigma_alpha(x, alpha)
    return s ** 2


def a_alpha_reg(x: np.ndarray, alpha: float, eps: float = 1e-3) -> np.ndarray:
    """Regularized local variance a_{alpha,eps}(x) = sigma_{alpha,eps}(x)^2."""
    s = sigma_alpha_reg(x, alpha, eps)
    return s ** 2


def G_alpha(x: np.ndarray, z: np.ndarray, p: np.ndarray, H: np.ndarray,
            alpha: float) -> np.ndarray:
    """Upper generating function of Example 4.1.

    G_alpha(x, z, p, H) = 0.5 * a_alpha(x) * H

    No killing (z-term absent), no drift (p-term absent), pure diffusion.
    """
    return 0.5 * a_alpha(x, alpha) * H


# ---------------------------------------------------------------------------
# Example 4.2 — single-point drift inflation
# ---------------------------------------------------------------------------

def G_example2(x: np.ndarray, z: np.ndarray, p: np.ndarray,
               H: np.ndarray) -> np.ndarray:
    """Local specification of Example 4.2.

    G(x, z, p, H) = p + 1_{x=0} * p^+

    At x = 0 the drift envelopes p in [1, 2] (for p > 0); everywhere else
    the drift is exactly p = 1 (for unit-speed shifts).
    """
    indicator = np.where(x == 0.0, 1.0, 0.0)
    p_plus = np.maximum(p, 0.0)
    return p + indicator * p_plus


def G_downarrow(x: np.ndarray, z: np.ndarray, p: np.ndarray,
                H: np.ndarray) -> np.ndarray:
    """Effective local specification G^downarrow = G_{V^G}.

    G^downarrow(x, z, p, H) = p

    Recovered from the valuation V_t^G f(x) = f(x + t) via difference
    quotients: (f(x+h) - f(x)) / h -> f'(x), confirming purely deterministic
    unit drift.
    """
    return p


def support_G_example2_drift(x: float) -> tuple[float, float]:
    """Return the admissible drift interval [b_min, b_max] at state x.

    Away from zero: {1}. At zero: [1, 2] (because p^+ adds up to p itself).
    """
    if x == 0.0:
        return (1.0, 2.0)
    return (1.0, 1.0)
