"""Tests for src/valuation_semigroup.py."""
import sys, os
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.valuation_semigroup import (
    AbsorbingValuation, ZeroSojournValuation, V_det,
    difference_quotient_det, G_downarrow_residual
)
from src.utils import f_payoff_plus, f_tanh, f_tanh_prime


def test_absorbing_at_zero_payoff_plus():
    """T^abs_t f_+(0) = f_+(0) = 0 for all t > 0."""
    T_abs = AbsorbingValuation(alpha=0.2, L=6.0, N=201, n_steps_per_unit=200)
    f_vals = f_payoff_plus(T_abs.x)
    t_grid = np.array([0.05, 0.1, 0.2])
    result = T_abs.at_zero(f_vals, t_grid)
    for t, v in result.items():
        assert abs(v) < 1e-10, f"T^abs_t f_+(0) = {v} at t={t}, expected 0"


def test_absorbing_values_finite():
    """T^abs output must be finite and in [0, 1] for f_+."""
    T_abs = AbsorbingValuation(alpha=0.2, L=5.0, N=101, n_steps_per_unit=100)
    f_vals = f_payoff_plus(T_abs.x)
    snaps = T_abs.evaluate(f_vals, np.array([0.05]))
    u = snaps[0.05]
    assert np.all(np.isfinite(u))
    assert np.all(u >= -1e-8)
    assert np.all(u <= 1.0 + 1e-8)


def test_zero_sojourn_strictly_positive_at_zero():
    """T^0_t f_+(0) > 0 for t > 0 (non-absorbing branch)."""
    T0 = ZeroSojournValuation(alpha=0.2, L=6.0, N=201, n_steps_per_unit=200, eps=1e-3)
    f_vals = f_payoff_plus(T0.x)
    result = T0.at_zero(f_vals, np.array([0.1, 0.2]))
    for t, v in result.items():
        assert v > 1e-6, f"T^0_t f_+(0) = {v} at t={t}, expected > 0"


def test_zero_sojourn_finite_and_bounded():
    """T^0 output must be finite and bounded by 1 for f_+."""
    T0 = ZeroSojournValuation(alpha=0.2, L=5.0, N=101, n_steps_per_unit=100, eps=1e-2)
    f_vals = f_payoff_plus(T0.x)
    snaps = T0.evaluate(f_vals, np.array([0.1]))
    u = snaps[0.1]
    assert np.all(np.isfinite(u))
    assert np.all(u >= -1e-8)
    assert np.all(u <= 1.0 + 1e-8)


def test_valuation_gap_strictly_positive():
    """T^0_t f_+(0) > T^abs_t f_+(0) for t > 0 (strict separation)."""
    alpha = 0.2
    T_abs = AbsorbingValuation(alpha=alpha, L=6.0, N=201, n_steps_per_unit=200)
    T0 = ZeroSojournValuation(alpha=alpha, L=6.0, N=201, n_steps_per_unit=200, eps=1e-3)
    f_vals = f_payoff_plus(T_abs.x)
    t_grid = np.array([0.1])
    abs_v = T_abs.at_zero(f_vals, t_grid)[0.1]
    zero_v = T0.at_zero(f_vals, t_grid)[0.1]
    assert zero_v > abs_v + 1e-6, f"Gap too small: {zero_v - abs_v}"


def test_V_det_exact():
    """V_t f(x) = f(x + t) exactly."""
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    t = 0.5
    val = V_det(f_tanh, x, t)
    expected = f_tanh(x + t)
    assert np.allclose(val, expected)


def test_difference_quotient_det_convergence():
    """Delta_h V_t f(x) -> f'(x) as h->0."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    errors = []
    for h in [0.1, 0.01, 0.001, 0.0001]:
        dq, exact = difference_quotient_det(f_tanh, f_tanh_prime, x, h)
        errors.append(float(np.max(np.abs(dq - exact))))
    # Errors should decrease monotonically
    for i in range(len(errors) - 1):
        assert errors[i+1] < errors[i] + 1e-10
    assert errors[-1] < 1e-3


def test_G_downarrow_residual_at_zero_positive_p():
    """G - G^down = p^+ at x=0."""
    x = np.array([0.0, 0.0, 0.0])
    p = np.array([0.0, 1.0, 2.0])
    H = np.zeros(3)
    res = G_downarrow_residual(x, p, H)
    expected = np.maximum(p, 0.0)
    assert np.allclose(res, expected)


def test_G_downarrow_residual_away_from_zero():
    """G = G^down away from x=0, so residual is 0."""
    x = np.array([-1.0, 1.0, 2.0])
    p = np.array([1.0, 1.0, 1.0])
    H = np.zeros(3)
    res = G_downarrow_residual(x, p, H)
    assert np.allclose(res, 0.0)
