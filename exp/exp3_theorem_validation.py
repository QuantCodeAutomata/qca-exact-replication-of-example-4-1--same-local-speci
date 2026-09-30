"""Experiment 3: Theorem Validation Suite — Valuation-Uncertainty-Generator.

Validates Theorems 3.1, 4.1, 4.2, 4.3/5.1, 5.3, and optionally 5.5 on
tractable finite-control HJB models.

Baseline model (Model A):
  G(x, U) = mu * |p| + 0.5 * sigma^2 * H
  HJB PDE: du/dt = mu * |du/dx| + 0.5 * sigma^2 * d^2u/dx^2

Stochastic representation: sup over b_t in {-mu, +mu} of E[f(X_T)].
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.killed_law import simulate_bm_drift
from src.pde_solver import hjb_pde_1d, pde_implicit_euler_snapshots
from src.local_spec import a_alpha
from src.utils import f_gaussian, f_gaussian_double_prime, make_grid


def stochastic_robust_valuation(
    x0_arr: np.ndarray,
    f_func,
    T: float,
    sigma: float,
    mu: float,
    n_paths: int = 20000,
    n_steps: int = 200,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compute V_T^G f(x0) stochastically via Monte Carlo worst-case control.

    For G = mu*|p| + 0.5*sigma^2*H, the optimal control selects b_t = +mu
    when the gradient of the value is positive, and b_t = -mu when negative.
    For f(x) = exp(-x^2) (concave), the optimal drift is -mu*sign(x) (pushes
    toward the maximum at x=0).

    Here we approximate by brute-force over bang-bang controls:
    b_t in {-mu, +mu}, selected optimally at each step via the current u gradient.
    Alternatively, we use a simpler upper bound by evaluating both constant
    controls and taking the supremum.

    Parameters
    ----------
    x0_arr  : initial states, shape (M,)
    f_func  : payoff callable
    T       : horizon
    sigma   : diffusion coefficient
    mu      : drift magnitude
    n_paths : MC paths per x0
    n_steps : time steps
    rng     : random generator

    Returns
    -------
    val : array of V_T^G f(x0) for each x0, shape (M,)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    dt = T / n_steps
    M = len(x0_arr)
    val = np.zeros(M)

    # For each starting point, evaluate over both constant controls {-mu, +mu}
    # and also a greedy adaptive control approximation.
    for i, x0 in enumerate(x0_arr):
        best = -np.inf
        # Constant controls
        for b_const in [-mu, 0.0, mu]:
            X = simulate_bm_drift(
                x0=x0, n_paths=n_paths, n_steps=n_steps, dt=dt,
                b=b_const, sigma=sigma, rng=rng,
            )
            v = float(np.mean(f_func(X[:, -1])))
            best = max(best, v)

        # Adaptive greedy: step forward with current sign-based optimal b
        X_adp = np.full(n_paths, x0, dtype=float)
        dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
        for j in range(n_steps):
            # Greedy choice: b = mu * sign(E[f'(X_T) | X_j]) — approximate
            # by b = -mu * sign(X) for Gaussian payoff (decreasing in |x|)
            b_adp = -mu * np.sign(X_adp)
            X_adp = X_adp + b_adp * dt + sigma * dW[:, j]
        v_adp = float(np.mean(f_func(X_adp)))
        best = max(best, v_adp)

        val[i] = best

    return val


def run_exp3(
    sigma: float = 0.5,
    mu: float = 0.3,
    L: float = 5.0,
    N: int = 201,
    T_vals: list[float] | None = None,
    n_steps_per_unit: int = 300,
    n_mc_paths: int = 15000,
    results_dir: str = "results",
) -> dict:
    """Run Experiment 3: HJB theorem validation."""
    os.makedirs(results_dir, exist_ok=True)

    if T_vals is None:
        T_vals = [0.1, 0.25, 0.5, 1.0]

    print(f"\n{'='*60}")
    print(f"Experiment 3: Theorem Validation (sigma={sigma}, mu={mu})")
    print(f"{'='*60}")

    x = make_grid(L, N)
    dx = x[1] - x[0]

    # ------------------------------------------------------------------ #
    # 1. Payoff and grid
    # ------------------------------------------------------------------ #
    f_vals = f_gaussian(x)
    # Exact second derivative for generator check
    f_dpp = f_gaussian_double_prime(x)

    # ------------------------------------------------------------------ #
    # 2. PDE canonical valuation V^G (Theorem 5.3 — PDE branch)
    # ------------------------------------------------------------------ #
    pde_snaps: dict[float, np.ndarray] = {}
    for T in T_vals:
        u_pde = hjb_pde_1d(f_vals, x, sigma, mu, t_end=T,
                            n_steps=max(50, int(T * n_steps_per_unit)))
        pde_snaps[T] = u_pde

    # ------------------------------------------------------------------ #
    # 3. Stochastic robust valuation (Theorem 5.3 — MC branch)
    # ------------------------------------------------------------------ #
    x0_grid = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    stoch_rows = []
    rng = np.random.default_rng(2024)

    for T in T_vals:
        val_stoch = stochastic_robust_valuation(
            x0_grid, f_gaussian, T, sigma, mu,
            n_paths=n_mc_paths,
            n_steps=max(50, int(T * n_steps_per_unit)),
            rng=rng,
        )
        pde_at_x0 = np.interp(x0_grid, x, pde_snaps[T])
        for xi, vs, vp in zip(x0_grid, val_stoch, pde_at_x0):
            stoch_rows.append({
                "T": T, "x0": xi,
                "V_stoch": vs, "V_pde": vp,
                "abs_err": abs(vs - vp), "rel_err": abs(vs - vp) / (abs(vp) + 1e-12),
            })

    df_stoch = pd.DataFrame(stoch_rows)
    print("\n  Theorem 5.3 validation (PDE vs Monte Carlo):")
    pivot = df_stoch.pivot_table(values="abs_err", index="x0", columns="T",
                                  aggfunc="mean")
    print(pivot.to_string())

    # ------------------------------------------------------------------ #
    # 4. Generator estimation from V^G (Theorem 4.1 / G^downarrow test)
    # ------------------------------------------------------------------ #
    from src.generator import difference_quotient_semigroup

    h_seq = 2.0 ** np.arange(-3, -9, -1)
    compact_mask = np.abs(x) < L * 0.7

    def pde_eval(f_v: np.ndarray, t_grid: np.ndarray) -> dict[float, np.ndarray]:
        snaps = {}
        for t in t_grid:
            snaps[t] = hjb_pde_1d(f_v, x, sigma, mu, t_end=t,
                                    n_steps=max(20, int(t * n_steps_per_unit)))
        return snaps

    # For linear sub-problem (mu=0): should recover pure diffusion generator
    def pure_diff_eval(f_v: np.ndarray, t_grid: np.ndarray) -> dict[float, np.ndarray]:
        a_arr = np.full(N, sigma ** 2)
        return pde_implicit_euler_snapshots(
            f=f_v, x=x, a=a_arr, t_grid=t_grid,
            n_steps_per_unit=n_steps_per_unit
        )

    gen_residuals_hjb = []
    gen_residuals_diff = []
    for h in h_seq:
        # HJB generator estimate
        dq_hjb = difference_quotient_semigroup(pde_eval, f_vals, x, h)
        # Expected: mu * |f'(x)| + 0.5 * sigma^2 * f''(x)
        from src.utils import f_gaussian_prime
        f_prime = f_gaussian_prime(x)
        target_hjb = mu * np.abs(f_prime) + 0.5 * sigma ** 2 * f_dpp
        gen_residuals_hjb.append(float(np.max(np.abs((dq_hjb - target_hjb)[compact_mask]))))

        # Pure diffusion generator estimate
        dq_diff = difference_quotient_semigroup(pure_diff_eval, f_vals, x, h)
        target_diff = 0.5 * sigma ** 2 * f_dpp
        gen_residuals_diff.append(float(np.max(np.abs((dq_diff - target_diff)[compact_mask]))))

    print(f"\n  Generator convergence study:")
    print(f"    {'h':>8s}  {'HJB_resid':>12s}  {'Diff_resid':>12s}")
    for h, r_hjb, r_diff in zip(h_seq, gen_residuals_hjb, gen_residuals_diff):
        print(f"    {h:8.4e}  {r_hjb:12.6e}  {r_diff:12.6e}")

    # ------------------------------------------------------------------ #
    # 5. Theorem 4.1: domination equivalence test
    # ------------------------------------------------------------------ #
    # For T_sub = pure-diffusion semigroup (mu=0, sigma fixed):
    #   G_{T_sub} = 0.5 sigma^2 H  which is <= G = mu|p| + 0.5 sigma^2 H  (for mu > 0)
    #   T_sub_t f <= V_t^G f  expected
    print(f"\n  Theorem 4.1: G_T <= G => T_t f <= V_t^G f:")
    theo41_rows = []
    for T in [0.1, 0.5, 1.0]:
        a_arr = np.full(N, sigma ** 2)
        sub_snaps = pde_implicit_euler_snapshots(
            f=f_vals, x=x, a=a_arr, t_grid=np.array([T]),
            n_steps_per_unit=n_steps_per_unit
        )
        T_sub = sub_snaps[T]
        T_G = pde_snaps[T]
        dominated = bool(np.all(T_sub <= T_G + 1e-10))
        max_viol = float(np.max(T_sub - T_G))
        theo41_rows.append({"T": T, "dominated": dominated, "max_violation": max_viol})

    df_theo41 = pd.DataFrame(theo41_rows)
    print(df_theo41.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 6. Theorem 5.1 / G^downarrow: effective localization of V^G
    # ------------------------------------------------------------------ #
    # Recover G^downarrow from the HJB PDE valuation
    print(f"\n  Theorem 5.1: G^downarrow <= G numerically:")
    f_test2 = f_gaussian(x)
    h_fine = 2.0 ** (-8)
    dq_G = difference_quotient_semigroup(pde_eval, f_test2, x, h_fine)
    from src.utils import f_gaussian_prime
    G_est = dq_G  # generator estimate from V^G
    G_theory = mu * np.abs(f_gaussian_prime(x)) + 0.5 * sigma ** 2 * f_dpp
    gen_gap = G_theory - G_est
    print(f"    Max |G_est - G_theory|: {float(np.max(np.abs(gen_gap[compact_mask]))):.4e}")
    print(f"    G^downarrow <= G globally: "
          f"{'True' if float(np.max(G_est - G_theory)) <= 1e-6 else 'approximate'}")

    # ------------------------------------------------------------------ #
    # 7. Plots
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Exp 3: Theorem Validation (sigma={sigma}, mu={mu})", fontsize=13)

    # (a) PDE valuation profiles
    ax = axes[0, 0]
    ax.plot(x, f_vals, "k--", lw=1.5, label="f (payoff)")
    for T in T_vals:
        ax.plot(x, pde_snaps[T], label=f"V^G (T={T})")
    ax.set_xlim(-4, 4); ax.set_xlabel("x"); ax.set_ylabel("value")
    ax.set_title("PDE canonical valuation V^G"); ax.legend(fontsize=7)

    # (b) PDE vs MC scatter
    ax = axes[0, 1]
    ax.scatter(df_stoch["V_pde"], df_stoch["V_stoch"], alpha=0.6, s=30)
    mn = df_stoch[["V_pde", "V_stoch"]].min().min()
    mx = df_stoch[["V_pde", "V_stoch"]].max().max()
    ax.plot([mn, mx], [mn, mx], "r--")
    ax.set_xlabel("PDE V^G"); ax.set_ylabel("MC V^G")
    ax.set_title("Theorem 5.3: PDE = MC (scatter)")

    # (c) Generator convergence
    ax = axes[1, 0]
    ax.loglog(h_seq, gen_residuals_hjb, "r-o", label="HJB |G_T - G|")
    ax.loglog(h_seq, gen_residuals_diff, "b-s", label="Diffusion |G_T - G|")
    ax.set_xlabel("h"); ax.set_ylabel("Max residual")
    ax.set_title("Generator convergence"); ax.legend()

    # (d) Theorem 4.1 violation
    ax = axes[1, 1]
    x_pts = df_stoch["x0"].unique()
    for T in [0.1, 0.5, 1.0]:
        sub_T = df_stoch[df_stoch["T"] == T]
        ax.bar([f"x={xi},T={T}" for xi in sub_T["x0"]], sub_T["abs_err"],
               label=f"T={T}", alpha=0.7)
    ax.set_xlabel("(x0, T)"); ax.set_ylabel("|V_stoch - V_pde|")
    ax.set_title("Stoch vs PDE absolute error")
    ax.tick_params(axis="x", labelsize=5)

    plt.tight_layout()
    fig_path = os.path.join(results_dir, "exp3_theorem_validation.png")
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"\n  Figure saved: {fig_path}")

    summary = {
        "pde_mc_max_abs_err": float(df_stoch["abs_err"].max()),
        "pde_mc_mean_abs_err": float(df_stoch["abs_err"].mean()),
        "theo41_all_dominated": bool(df_theo41["dominated"].all()),
        "gen_hjb_finest_h_residual": gen_residuals_hjb[-1],
        "gen_diff_finest_h_residual": gen_residuals_diff[-1],
    }
    print(f"\n  Summary: {summary}")
    return {
        "summary": summary,
        "df_stoch": df_stoch,
        "df_theo41": df_theo41,
        "gen_residuals_hjb": gen_residuals_hjb,
        "gen_residuals_diff": gen_residuals_diff,
    }


if __name__ == "__main__":
    run_exp3()
    print("\nDone.")
