"""Tests for src/local_spec.py: local specifications G_alpha and G_example2."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.local_spec import (
    G_alpha,
    G_downarrow,
    G_example2,
    a_alpha,
    a_alpha_reg,
    sigma_alpha,
    sigma_alpha_reg,
    support_G_example2_drift,
)


# ---------------------------------------------------------------------------
# sigma_alpha
# ---------------------------------------------------------------------------

def test_sigma_alpha_at_zero():
    """sigma_alpha(0, alpha) must equal 0 for any alpha in (0, 1/2)."""
    for alpha in [0.1, 0.2, 0.3, 0.4]:
        val = float(sigma_alpha(np.array([0.0]), alpha)[0])
        assert val == 0.0, f"sigma_alpha(0, {alpha}) = {val}, expected 0"


def test_sigma_alpha_range():
    """sigma_alpha must lie in [0, 1) for all x and valid alpha."""
    x = np.linspace(-5, 5, 200)
    for alpha in [0.1, 0.2, 0.4, 0.49]:
        s = sigma_alpha(x, alpha)
        assert np.all(s >= 0.0), "sigma_alpha must be non-negative"
        assert np.all(s < 1.0), "sigma_alpha must be strictly less than 1"


def test_sigma_alpha_even():
    """sigma_alpha(x) must be an even function: sigma(-x) = sigma(x)."""
    x = np.array([0.5, 1.0, 2.0, 3.0])
    for alpha in [0.2, 0.4]:
        assert np.allclose(sigma_alpha(x, alpha), sigma_alpha(-x, alpha))


def test_sigma_alpha_reg_positive():
    """Regularized sigma must be strictly positive everywhere."""
    x = np.linspace(-3, 3, 50)
    for alpha in [0.2, 0.4]:
        for eps in [0.1, 0.01, 0.001]:
            s = sigma_alpha_reg(x, alpha, eps)
            assert np.all(s > 0.0), "Regularized sigma must be positive"


def test_sigma_alpha_reg_converges():
    """sigma_{alpha,eps}(x) -> sigma_alpha(x) as eps -> 0 for x != 0."""
    x = np.array([0.5, 1.0, 2.0])
    alpha = 0.2
    s_exact = sigma_alpha(x, alpha)
    for eps in [1e-3, 1e-4, 1e-5]:
        s_reg = sigma_alpha_reg(x, alpha, eps)
        assert np.allclose(s_reg, s_exact, atol=max(eps, 1e-4))


# ---------------------------------------------------------------------------
# a_alpha
# ---------------------------------------------------------------------------

def test_a_alpha_at_zero():
    """a_alpha(0) = 0 for all valid alpha."""
    for alpha in [0.1, 0.2, 0.3, 0.4]:
        val = float(a_alpha(np.array([0.0]), alpha)[0])
        assert val == 0.0


def test_a_alpha_non_negative():
    """a_alpha(x) >= 0 everywhere."""
    x = np.linspace(-5, 5, 100)
    for alpha in [0.1, 0.2, 0.4]:
        assert np.all(a_alpha(x, alpha) >= 0.0)


def test_a_alpha_less_than_sigma_squared():
    """a_alpha = sigma_alpha^2 <= sigma_alpha < 1 => a_alpha < 1."""
    x = np.linspace(-5, 5, 100)
    for alpha in [0.2, 0.4]:
        a = a_alpha(x, alpha)
        s = sigma_alpha(x, alpha)
        assert np.allclose(a, s ** 2)
        assert np.all(a < 1.0)


def test_a_alpha_reg_strictly_positive():
    """Regularized local variance is strictly positive."""
    x = np.linspace(-3, 3, 50)
    a = a_alpha_reg(x, 0.2, eps=0.01)
    assert np.all(a > 0.0)


# ---------------------------------------------------------------------------
# G_alpha
# ---------------------------------------------------------------------------

def test_G_alpha_no_killing_no_drift():
    """G_alpha should be independent of z and p."""
    x = np.array([1.0, 2.0])
    H = np.array([1.0, 1.0])
    alpha = 0.2
    z1 = np.array([0.0, 0.0]); z2 = np.array([5.0, -3.0])
    p1 = np.array([0.0, 0.0]); p2 = np.array([1.0, -1.0])
    g1 = G_alpha(x, z1, p1, H, alpha)
    g2 = G_alpha(x, z2, p2, H, alpha)
    assert np.allclose(g1, g2), "G_alpha should not depend on z or p"


def test_G_alpha_at_zero():
    """G_alpha(0, z, p, H) = 0 for any H (degenerate at 0)."""
    alpha = 0.2
    x = np.array([0.0])
    H = np.array([10.0])
    val = float(G_alpha(x, np.array([0.0]), np.array([1.0]), H, alpha)[0])
    assert val == 0.0, f"G_alpha at x=0 must be 0, got {val}"


def test_G_alpha_linear_in_H():
    """G_alpha must scale linearly with H."""
    x = np.array([1.0])
    alpha = 0.3
    H1, H2 = np.array([1.0]), np.array([2.0])
    z, p = np.zeros(1), np.zeros(1)
    g1 = G_alpha(x, z, p, H1, alpha)
    g2 = G_alpha(x, z, p, H2, alpha)
    assert np.allclose(2 * g1, g2)


# ---------------------------------------------------------------------------
# Example 4.2 specifications
# ---------------------------------------------------------------------------

def test_G_example2_away_from_zero():
    """Away from x=0, G_example2 = p."""
    x = np.array([-2.0, -1.0, 1.0, 2.0])
    p = np.array([1.0, 1.0, 1.0, 1.0])
    z = np.zeros(4); H = np.zeros(4)
    g = G_example2(x, z, p, H)
    assert np.allclose(g, p)


def test_G_example2_at_zero_p_positive():
    """At x=0 with p > 0: G_example2 = p + p = 2p."""
    x = np.array([0.0])
    p = np.array([1.0])
    g = G_example2(x, np.zeros(1), p, np.zeros(1))
    assert np.allclose(g, 2.0 * p)


def test_G_example2_at_zero_p_negative():
    """At x=0 with p <= 0: G_example2 = p (p^+ = 0)."""
    x = np.array([0.0])
    p = np.array([-0.5])
    g = G_example2(x, np.zeros(1), p, np.zeros(1))
    assert np.allclose(g, p)


def test_G_downarrow_equals_p():
    """G_downarrow = p everywhere."""
    x = np.array([-2.0, 0.0, 1.0])
    p = np.array([0.5, 0.5, 0.5])
    g = G_downarrow(x, np.zeros(3), p, np.zeros(3))
    assert np.allclose(g, p)


def test_G_downarrow_strictly_less_than_G_at_zero():
    """G^downarrow(0,z,p,0) < G_example2(0,z,p,0) for p > 0."""
    x = np.array([0.0])
    z = np.zeros(1); H = np.zeros(1)
    for p_val in [0.1, 0.5, 1.0, 2.0]:
        p = np.array([p_val])
        g = G_example2(x, z, p, H)
        gd = G_downarrow(x, z, p, H)
        assert float(gd[0]) < float(g[0]), f"G_down must be < G at x=0 for p={p_val}"


def test_support_G_example2_drift():
    """Admissible drift at x=0 is [1,2]; elsewhere [1,1]."""
    assert support_G_example2_drift(0.0) == (1.0, 2.0)
    assert support_G_example2_drift(1.0) == (1.0, 1.0)
    assert support_G_example2_drift(-1.5) == (1.0, 1.0)
