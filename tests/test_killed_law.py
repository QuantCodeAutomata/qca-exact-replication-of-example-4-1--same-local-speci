"""Tests for src/killed_law.py: killed-law / discounted-pair equivalence."""
import sys, os
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.killed_law import (
    simulate_ou, simulate_bm_drift, cumulative_discount,
    apply_killing, killed_expectation, discounted_expectation,
    validate_theorem_6_4, recover_discount_process
)
from src.utils import f_gaussian, f_payoff_plus


def test_simulate_ou_shape():
    """OU paths must have shape (n_paths, n_steps+1)."""
    X = simulate_ou(x0=0.0, n_paths=50, n_steps=100, dt=0.01)
    assert X.shape == (50, 101)


def test_simulate_ou_initial_state():
    """All paths start at x0."""
    X = simulate_ou(x0=1.5, n_paths=20, n_steps=10, dt=0.01)
    assert np.allclose(X[:, 0], 1.5)


def test_simulate_bm_drift_shape():
    X = simulate_bm_drift(x0=0.0, n_paths=30, n_steps=50, dt=0.02)
    assert X.shape == (30, 51)


def test_cumulative_discount_starts_at_zero():
    """A[:,0] = 0 (no discount at time 0)."""
    X = simulate_ou(x0=0.0, n_paths=20, n_steps=100, dt=0.01)
    k_func = lambda x: np.full_like(x, 0.5)
    A = cumulative_discount(X, k_func, dt=0.01)
    assert np.allclose(A[:, 0], 0.0)


def test_cumulative_discount_non_decreasing():
    """A must be non-decreasing in time for non-negative k."""
    X = simulate_ou(x0=0.0, n_paths=20, n_steps=100, dt=0.01)
    k_func = lambda x: 0.3 * x**2
    A = cumulative_discount(X, k_func, dt=0.01)
    diffs = np.diff(A, axis=1)
    assert np.all(diffs >= -1e-12)


def test_apply_killing_returns_correct_shape():
    X = simulate_ou(x0=0.0, n_paths=30, n_steps=50, dt=0.02)
    A = cumulative_discount(X, lambda x: np.full_like(x, 0.5), dt=0.02)
    alive, tau_idx = apply_killing(X, A)
    assert alive.shape == X.shape
    assert tau_idx.shape == (30,)


def test_apply_killing_alive_initially():
    """All paths alive at t=0 (A[:,0] = 0 < E ~ Exp(1))."""
    X = simulate_ou(x0=0.0, n_paths=100, n_steps=50, dt=0.02)
    A = cumulative_discount(X, lambda x: np.full_like(x, 1.0), dt=0.02)
    alive, _ = apply_killing(X, A)
    assert np.all(alive[:, 0])


def test_killed_vs_discounted_match_constant_rate():
    """With constant rate k, killed and discounted expectations match within MC error."""
    result = validate_theorem_6_4(
        x0=0.0, n_paths=50000, n_steps=100, T=0.5,
        k_func=lambda x: np.full_like(x, 0.5),
        f_func=f_gaussian,
        process='ou', theta=1.0, sigma=0.5
    )
    assert result['abs_error'] < 0.02, f"Abs error too large: {result['abs_error']}"


def test_recover_discount_process_shape():
    """Recovered A_hat should have length n_steps+1."""
    X = simulate_ou(x0=0.0, n_paths=500, n_steps=100, dt=0.01)
    k_func = lambda x: np.full_like(x, 0.5)
    A = cumulative_discount(X, k_func, dt=0.01)
    alive, _ = apply_killing(X, A)
    A_hat = recover_discount_process(alive, dt=0.01)
    assert A_hat.shape == (101,)


def test_recover_discount_starts_near_zero():
    """A_hat[0] should be near 0 (almost all paths alive at t=0)."""
    X = simulate_ou(x0=0.0, n_paths=1000, n_steps=50, dt=0.01)
    A = cumulative_discount(X, lambda x: np.full_like(x, 0.5), dt=0.01)
    alive, _ = apply_killing(X, A)
    A_hat = recover_discount_process(alive, dt=0.01)
    assert abs(A_hat[0]) < 0.05


def test_recover_discount_non_decreasing():
    """Recovered A_hat must be non-decreasing (survival probability non-increasing)."""
    X = simulate_ou(x0=0.0, n_paths=2000, n_steps=50, dt=0.01)
    A = cumulative_discount(X, lambda x: np.full_like(x, 0.5), dt=0.01)
    alive, _ = apply_killing(X, A)
    A_hat = recover_discount_process(alive, dt=0.01)
    diffs = np.diff(A_hat)
    assert np.all(diffs >= -0.1)  # allow small numerical noise
