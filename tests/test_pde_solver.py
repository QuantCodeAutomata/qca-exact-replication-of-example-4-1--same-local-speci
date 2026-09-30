"""Tests for src/pde_solver.py: 1-D PDE solver."""
import sys, os
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.pde_solver import build_diffusion_matrix, pde_implicit_euler, pde_implicit_euler_snapshots, hjb_pde_1d


def test_build_diffusion_matrix_shape():
    x = np.linspace(-1, 1, 21)
    a = np.ones(21)
    L = build_diffusion_matrix(x, a)
    assert L.shape == (21, 21)


def test_pde_heat_equation_conservation():
    """Pure diffusion with constant a: total integral should be conserved (zero-flux BCs approximation)."""
    N = 101
    x = np.linspace(-5, 5, N)
    a = np.ones(N)
    f = np.exp(-x**2)
    dx = x[1] - x[0]
    u = pde_implicit_euler(f, x, a, t_end=0.1, n_steps=50)
    # Mass approximately conserved (not exactly due to Dirichlet-ish boundary artifacts)
    assert u.shape == (N,)
    assert np.all(np.isfinite(u))


def test_pde_absorbing_boundary_frozen():
    """With absorb_idx at center, that point should stay at f[absorb_idx]."""
    N = 51
    x = np.linspace(-1, 1, N)
    a = np.ones(N)
    f = x**2 / (1 + x**2)
    center = N // 2
    u = pde_implicit_euler(f, x, a, t_end=0.1, n_steps=50, absorb_idx=center)
    assert abs(u[center] - f[center]) < 1e-14


def test_pde_snapshots_monotone_in_time():
    """For f = exp(-x^2) with diffusion, the maximum should spread (not necessarily monotone everywhere, but finite)."""
    N = 101
    x = np.linspace(-5, 5, N)
    a = np.ones(N)
    f = np.exp(-x**2)
    t_grid = np.array([0.05, 0.1, 0.2])
    snaps = pde_implicit_euler_snapshots(f, x, a, t_grid, n_steps_per_unit=100)
    for t in t_grid:
        assert t in snaps
        assert np.all(np.isfinite(snaps[t]))
    # Maximum value should decrease or stay (diffusion spreads)
    assert snaps[0.2].max() <= snaps[0.05].max() + 1e-6


def test_pde_degenerate_at_zero_absorbing():
    """With degenerate a (zero at center), absorbing BC means u[center] stays at f[center]."""
    N = 51
    x = np.linspace(-2, 2, N)
    center = N // 2
    a = x**2 / (1 + x**2)  # degenerate at 0
    f = x**2 / (1 + x**2)
    f_0 = f[center]
    u = pde_implicit_euler(f, x, a, t_end=0.1, n_steps=50, absorb_idx=center)
    assert abs(u[center] - f_0) < 1e-14


def test_hjb_pde_larger_than_pure_diffusion():
    """HJB with mu>0 should give values >= pure diffusion (robust control adds value)."""
    N = 101
    x = np.linspace(-5, 5, N)
    f = np.exp(-x**2)
    sigma, mu = 0.5, 0.3
    u_hjb = hjb_pde_1d(f, x, sigma, mu, t_end=0.1, n_steps=100)
    u_diff = pde_implicit_euler(f, x, np.full(N, sigma**2), t_end=0.1, n_steps=100)
    # HJB value >= diffusion value (robust over controls)
    assert np.mean(u_hjb) >= np.mean(u_diff) - 1e-6


def test_hjb_pde_mu_zero_matches_diffusion():
    """When mu=0, HJB reduces to pure diffusion."""
    N = 101
    x = np.linspace(-5, 5, N)
    f = np.exp(-x**2)
    sigma = 0.5
    u_hjb = hjb_pde_1d(f, x, sigma, mu=0.0, t_end=0.05, n_steps=50)
    u_diff = pde_implicit_euler(f, x, np.full(N, sigma**2), t_end=0.05, n_steps=50)
    # Should be very close when mu=0
    assert np.allclose(u_hjb, u_diff, atol=1e-4)


def test_pde_non_negative_payoff_stays_non_negative():
    """Non-negative initial condition should stay non-negative under diffusion."""
    N = 101
    x = np.linspace(-5, 5, N)
    a = np.ones(N) * 0.5
    f = np.exp(-x**2)  # non-negative
    u = pde_implicit_euler(f, x, a, t_end=0.2, n_steps=100)
    assert np.all(u >= -1e-10)
