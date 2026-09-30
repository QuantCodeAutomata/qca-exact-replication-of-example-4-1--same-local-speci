"""Utility functions: smooth bounded test payoffs and common helpers."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Smooth bounded test functions on R (all in C_b^infty(R))
# ---------------------------------------------------------------------------

def f_tanh(x: np.ndarray) -> np.ndarray:
    """tanh(x) — odd, bounded in (-1,1), smooth."""
    return np.tanh(x)


def f_tanh_prime(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.cosh(x) ** 2


def f_tanh_double_prime(x: np.ndarray) -> np.ndarray:
    return -2.0 * np.tanh(x) / np.cosh(x) ** 2


def f_gaussian(x: np.ndarray) -> np.ndarray:
    """exp(-x^2) — even, bounded in (0,1], smooth."""
    return np.exp(-(x ** 2))


def f_gaussian_prime(x: np.ndarray) -> np.ndarray:
    return -2.0 * x * np.exp(-(x ** 2))


def f_gaussian_double_prime(x: np.ndarray) -> np.ndarray:
    return (4.0 * x ** 2 - 2.0) * np.exp(-(x ** 2))


def f_rational(x: np.ndarray) -> np.ndarray:
    """x / (1 + x^2) — odd, bounded, smooth."""
    return x / (1.0 + x ** 2)


def f_rational_prime(x: np.ndarray) -> np.ndarray:
    return (1.0 - x ** 2) / (1.0 + x ** 2) ** 2


def f_rational_double_prime(x: np.ndarray) -> np.ndarray:
    return 2.0 * x * (x ** 2 - 3.0) / (1.0 + x ** 2) ** 3


def f_sin_rational(x: np.ndarray) -> np.ndarray:
    """sin(x) / (1 + x^2) — bounded, smooth."""
    return np.sin(x) / (1.0 + x ** 2)


def f_sin_rational_prime(x: np.ndarray) -> np.ndarray:
    return np.cos(x) / (1.0 + x ** 2) - 2.0 * x * np.sin(x) / (1.0 + x ** 2) ** 2


def f_sin_rational_double_prime(x: np.ndarray) -> np.ndarray:
    t = 1.0 + x ** 2
    return (-np.sin(x) / t
            - 4.0 * x * np.cos(x) / t ** 2
            + (8.0 * x ** 2 - 2.0) * np.sin(x) / t ** 3)


def f_payoff_plus(x: np.ndarray) -> np.ndarray:
    """f_+(x) = x^2 / (1 + x^2) — bounded in [0,1), zero at 0, smooth."""
    return x ** 2 / (1.0 + x ** 2)


def f_payoff_plus_prime(x: np.ndarray) -> np.ndarray:
    return 2.0 * x / (1.0 + x ** 2) ** 2


def f_payoff_plus_double_prime(x: np.ndarray) -> np.ndarray:
    return (2.0 - 6.0 * x ** 2) / (1.0 + x ** 2) ** 3


# Registry for experiment loops
TEST_FUNCTIONS = {
    "tanh": (f_tanh, f_tanh_prime, f_tanh_double_prime),
    "gaussian": (f_gaussian, f_gaussian_prime, f_gaussian_double_prime),
    "rational": (f_rational, f_rational_prime, f_rational_double_prime),
    "sin_rational": (f_sin_rational, f_sin_rational_prime, f_sin_rational_double_prime),
}


def lyapunov(x: np.ndarray) -> np.ndarray:
    """Lyapunov function phi(x) = 1 + x^2."""
    return 1.0 + x ** 2


def make_grid(L: float, N: int) -> np.ndarray:
    """Uniform grid on [-L, L] with N interior points (endpoints at ±L)."""
    return np.linspace(-L, L, N)


def relative_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Element-wise relative error |a - b| / (|b| + eps)."""
    return np.abs(a - b) / (np.abs(b) + eps)
