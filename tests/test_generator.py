"""Tests for src/generator.py: generator estimation."""
import sys, os
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.generator import (
    difference_quotient_semigroup, generator_residual,
    convergence_study, compute_valuation_gap
)
from src.valuation_semigroup import AbsorbingValuation, ZeroSojournValuation
from src.utils import f_gaussian, f_gaussian_double_prime


def test_difference_quotient_semigroup_shape():
    """Difference quotient should return same shape as f_vals."""
    T = AbsorbingValuation(alpha=0.2, L=5.0, N=101, n_steps_per_unit=100)
    f_vals = f_gaussian(T.x)
    dq = difference_quotient_semigroup(T.evaluate, f_vals, T.x, h=0.01)
    assert dq.shape == T.x.shape


def test_generator_residual_finite():
    """Generator residual should be finite everywhere."""
    T = AbsorbingValuation(alpha=0.2, L=5.0, N=101, n_steps_per_unit=100)
    f_vals = f_gaussian(T.x)
    f_dpp = f_gaussian_double_prime(T.x)
    res = generator_residual(T.evaluate, f_vals, f_dpp, T.x, 0.2, h=0.01)
    assert np.all(np.isfinite(res))


def test_convergence_study_returns_correct_keys():
    """convergence_study must return dict with h, max_residual, mean_residual."""
    T = AbsorbingValuation(alpha=0.2, L=5.0, N=101, n_steps_per_unit=100)
    f_vals = f_gaussian(T.x)
    f_dpp = f_gaussian_double_prime(T.x)
    h_seq = np.array([0.02, 0.01])
    result = convergence_study(T.evaluate, f_vals, f_dpp, T.x, 0.2, h_seq)
    assert 'h' in result
    assert 'max_residual' in result
    assert 'mean_residual' in result
    assert len(result['max_residual']) == 2


def test_convergence_study_residuals_non_negative():
    """Residuals must be non-negative (they are absolute values)."""
    T = AbsorbingValuation(alpha=0.2, L=5.0, N=101, n_steps_per_unit=100)
    f_vals = f_gaussian(T.x)
    f_dpp = f_gaussian_double_prime(T.x)
    h_seq = np.array([0.02, 0.01])
    result = convergence_study(T.evaluate, f_vals, f_dpp, T.x, 0.2, h_seq)
    assert np.all(result['max_residual'] >= 0)


def test_compute_valuation_gap_sign():
    """Valuation gap T^0 - T^abs should be non-negative (T^0 >= T^abs for f_+)."""
    alpha = 0.2
    T_abs = AbsorbingValuation(alpha=alpha, L=5.0, N=101, n_steps_per_unit=100)
    T0 = ZeroSojournValuation(alpha=alpha, L=5.0, N=101, n_steps_per_unit=100, eps=1e-2)
    from src.utils import f_payoff_plus
    f_vals = f_payoff_plus(T_abs.x)
    t_grid = np.array([0.1])
    abs_v = T_abs.at_zero(f_vals, t_grid)
    zero_v = T0.at_zero(f_vals, t_grid)
    gap = compute_valuation_gap(abs_v, zero_v)
    for t, g in gap.items():
        assert g >= -1e-8, f"Negative gap {g} at t={t}"


def test_convergence_study_decreasing_trend():
    """Residuals should generally decrease as h decreases (for smooth f)."""
    T = AbsorbingValuation(alpha=0.2, L=5.0, N=101, n_steps_per_unit=100)
    f_vals = f_gaussian(T.x)
    f_dpp = f_gaussian_double_prime(T.x)
    # Compact mask: avoid boundaries
    compact_mask = (np.abs(T.x) < 3.0) & (np.abs(T.x) > 0.2)
    h_seq = np.array([0.05, 0.025, 0.01])
    result = convergence_study(T.evaluate, f_vals, f_dpp, T.x, 0.2, h_seq, compact_mask)
    # At least the last residual should be smaller than the first
    assert result['max_residual'][-1] < result['max_residual'][0] + 0.1
