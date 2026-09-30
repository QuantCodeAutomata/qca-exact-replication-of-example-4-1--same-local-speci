"""Tests for src/realization.py: Appendix-C realization pipeline."""
import sys, os
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.realization import (
    outer_approx_G_n, outer_approx_q_n, sup_convolution, mollify,
    add_ellipticity_floor, build_selector_zeta, construct_realizing_law_pde,
    check_stopped_inequality, run_realization_pipeline
)
from src.local_spec import a_alpha
from src.utils import make_grid, f_gaussian


def test_outer_approx_G_n_dominates_G():
    """G_n = 0.5*(a + 1/n)*H >= 0.5*a*H = G for H > 0."""
    x = np.linspace(-2, 2, 21)
    a_base = a_alpha(x, 0.2)
    H = np.ones(21)
    for n in [1, 5, 10, 100]:
        G_n = outer_approx_G_n(x, a_base, n, H)
        G = 0.5 * a_base * H
        assert np.all(G_n >= G - 1e-15)


def test_outer_approx_G_n_converges():
    """G_n -> G as n -> infty."""
    x = np.linspace(-2, 2, 21)
    a_base = a_alpha(x, 0.2)
    H = np.ones(21)
    G = 0.5 * a_base * H
    G_n = outer_approx_G_n(x, a_base, 1000, H)
    assert np.allclose(G_n, G, atol=0.001)


def test_outer_approx_q_n_dominates():
    q = np.array([0.1, 0.5, 1.0])
    for n in [1, 10, 100]:
        q_n = outer_approx_q_n(q, n)
        assert np.all(q_n >= q - 1e-15)


def test_sup_convolution_dominates():
    """Sup-convolution u^eps(x) >= u(x)."""
    x = np.linspace(-3, 3, 61)
    u = np.exp(-x**2)
    for eps in [0.5, 0.1, 0.01]:
        u_eps = sup_convolution(u, x, eps)
        assert np.all(u_eps >= u - 1e-12)


def test_sup_convolution_converges_to_u():
    """u^eps -> u pointwise as eps -> 0 for smooth u."""
    x = np.linspace(-3, 3, 61)
    u = np.exp(-x**2)
    u_eps_small = sup_convolution(u, x, eps=1e-4)
    assert np.allclose(u_eps_small, u, atol=0.01)


def test_mollify_smooth():
    """Mollification returns same-shape finite array."""
    x = np.linspace(-3, 3, 61)
    u = np.exp(-x**2)
    v = mollify(u, x, sigma_mol=0.1)
    assert v.shape == u.shape
    assert np.all(np.isfinite(v))


def test_add_ellipticity_floor_increases_a():
    a = np.array([0.0, 0.1, 0.5])
    lam = 0.05
    a_floor = add_ellipticity_floor(a, lam)
    assert np.all(a_floor >= a)
    assert np.allclose(a_floor - a, lam)


def test_build_selector_zeta_range():
    """Selector must output values in [a_min, a_max]."""
    x = np.linspace(-2, 2, 41)
    v = np.exp(-x**2)
    a_set = (0.1, 1.0)
    zeta = build_selector_zeta(v, x, a_set)
    assert np.all(zeta >= a_set[0] - 1e-12)
    assert np.all(zeta <= a_set[1] + 1e-12)


def test_check_stopped_inequality_satisfied_when_u_leq_Tf():
    """Inequality satisfied when u <= T_zeta f."""
    x = np.linspace(-2, 2, 21)
    u = np.zeros(21)
    T_f = np.ones(21)
    result = check_stopped_inequality(u, T_f, x)
    assert result['fraction_satisfied'] == 1.0
    assert result['max_violation'] <= 0.0


def test_check_stopped_inequality_violated_when_u_gt_Tf():
    """Violation detected when u > T_zeta f."""
    x = np.linspace(-2, 2, 21)
    u = np.ones(21)
    T_f = np.zeros(21)
    result = check_stopped_inequality(u, T_f, x)
    assert result['max_violation'] > 0.0
    assert result['fraction_satisfied'] < 1.0


def test_run_realization_pipeline_returns_list():
    """Pipeline should return a list of diagnostics."""
    x = make_grid(3.0, 61)
    a_base = a_alpha(x, 0.2)
    u = f_gaussian(x)
    q_base = np.zeros_like(x)
    result = run_realization_pipeline(
        u_subsol=u, a_base=a_base, x=x, q_base=q_base,
        t=0.1, n_seq=[5], eps_seq=[0.1], lam_seq=[0.05],
        sigma_mol_seq=[0.1], a_set=(0.0, 1.0), n_steps_per_unit=50
    )
    assert isinstance(result, list)
    assert len(result) == 1
    assert 'max_violation' in result[0]
    assert 'max_residual' in result[0]
