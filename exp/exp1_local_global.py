"""Experiment 1: Example 4.1 — Same Local Specification, Different Global Valuations.

Validates the paper's claim that two distinct valuation semigroups T^{abs} and T^0
share the same local specification G_alpha yet differ globally: for the payoff
f_+(x) = x^2 / (1 + x^2),

    T^{abs}_{t0} f_+(0) = 0  while  T^0_{t0} f_+(0) > 0.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.generator import compute_valuation_gap, convergence_study
from src.local_spec import a_alpha, sigma_alpha
from src.utils import (
    TEST_FUNCTIONS,
    f_gaussian_double_prime,
    f_gaussian,
    f_payoff_plus,
    f_tanh_double_prime,
    f_tanh,
    make_grid,
)
from src.valuation_semigroup import AbsorbingValuation, ZeroSojournValuation


def run_exp1(
    alpha: float = 0.2,
    L: float = 8.0,
    N: int = 401,
    n_steps_per_unit: int = 400,
    eps_values: list[float] | None = None,
    t_grid: np.ndarray | None = None,
    h_seq: np.ndarray | None = None,
    results_dir: str = "results",
) -> dict:
    """Run Experiment 1 for a single alpha value.

    Returns a dict with all key metrics.
    """
    os.makedirs(results_dir, exist_ok=True)

    if eps_values is None:
        eps_values = [1e-1, 1e-2, 1e-3, 5e-4]
    if t_grid is None:
        t_grid = np.array([0.01, 0.02, 0.05, 0.1, 0.2])
    if h_seq is None:
        h_seq = 2.0 ** np.arange(-3, -10, -1)

    print(f"\n{'='*60}")
    print(f"Experiment 1: Example 4.1 (alpha={alpha})")
    print(f"{'='*60}")

    # ------------------------------------------------------------------ #
    # 1. Grid and local variance
    # ------------------------------------------------------------------ #
    x = make_grid(L, N)
    zero_idx = int(np.argmin(np.abs(x)))
    a_vals = a_alpha(x, alpha)
    print(f"  a_alpha(0) = {a_alpha(np.array([0.0]), alpha)[0]:.6e}")

    # ------------------------------------------------------------------ #
    # 2. Absorbing semigroup
    # ------------------------------------------------------------------ #
    T_abs = AbsorbingValuation(alpha=alpha, L=L, N=N, n_steps_per_unit=n_steps_per_unit)

    # ------------------------------------------------------------------ #
    # 3. Zero-sojourn semigroup (multiple eps, check convergence)
    # ------------------------------------------------------------------ #
    T0_semigroups = {
        eps: ZeroSojournValuation(alpha=alpha, L=L, N=N,
                                   n_steps_per_unit=n_steps_per_unit, eps=eps)
        for eps in eps_values
    }

    # ------------------------------------------------------------------ #
    # 4. Key valuation at x=0 for payoff f_+
    # ------------------------------------------------------------------ #
    f_plus_vals = f_payoff_plus(x)
    print(f"  f_+(0) = {f_plus_vals[zero_idx]:.6e}")

    abs_at_zero = T_abs.at_zero(f_plus_vals, t_grid)
    print(f"\n  T^{{abs}}_t f_+(0):")
    for t, v in sorted(abs_at_zero.items()):
        print(f"    t={t:.3f}: {v:.6e}")

    print(f"\n  T^0_t f_+(0) by eps:")
    zero_at_zero_by_eps: dict[float, dict[float, float]] = {}
    for eps in eps_values:
        T0 = T0_semigroups[eps]
        z_vals = T0.at_zero(f_plus_vals, t_grid)
        zero_at_zero_by_eps[eps] = z_vals
        vals_str = "  ".join(f"t={t:.3f}: {v:.4f}" for t, v in sorted(z_vals.items()))
        print(f"    eps={eps:.1e}:  {vals_str}")

    # Use the finest eps as the T^0 representative
    eps_best = min(eps_values)
    T0_best = T0_semigroups[eps_best]
    zero_at_zero_best = zero_at_zero_by_eps[eps_best]

    gap = compute_valuation_gap(abs_at_zero, zero_at_zero_best)
    print(f"\n  Valuation gap T^0_t f_+(0) - T^{{abs}}_t f_+(0):")
    for t, g in sorted(gap.items()):
        sign = "POSITIVE ✓" if g > 1e-8 else "zero"
        print(f"    t={t:.3f}: gap={g:.6f}  [{sign}]")

    # ------------------------------------------------------------------ #
    # 5. Generator consistency check
    # ------------------------------------------------------------------ #
    print(f"\n  Generator convergence study (T^{{abs}}):")
    compact_mask = (np.abs(x) < L * 0.5) & (np.abs(x) > 0.05)

    # Use tanh as test function (smooth, bounded)
    f_test_vals = f_tanh(x)
    f_test_dpp = f_tanh_double_prime(x)

    conv_abs = convergence_study(
        T_abs.evaluate, f_test_vals, f_test_dpp,
        x, alpha, h_seq, compact_mask
    )
    print(f"    {'h':>10s}  {'max_residual':>14s}")
    for h, mr in zip(conv_abs["h"], conv_abs["max_residual"]):
        print(f"    {h:10.4e}  {mr:14.6e}")

    print(f"\n  Generator convergence study (T^0, eps={eps_best:.1e}):")
    conv_zero = convergence_study(
        T0_best.evaluate, f_test_vals, f_test_dpp,
        x, alpha, h_seq, compact_mask
    )
    print(f"    {'h':>10s}  {'max_residual':>14s}")
    for h, mr in zip(conv_zero["h"], conv_zero["max_residual"]):
        print(f"    {h:10.4e}  {mr:14.6e}")

    # ------------------------------------------------------------------ #
    # 6. Robustness over alpha grid
    # ------------------------------------------------------------------ #
    alpha_grid = [0.1, 0.2, 0.3, 0.4]
    t0 = 0.1
    robustness_table = []
    for a_val in alpha_grid:
        T_a = AbsorbingValuation(alpha=a_val, L=L, N=N, n_steps_per_unit=n_steps_per_unit)
        T0_a = ZeroSojournValuation(alpha=a_val, L=L, N=N,
                                     n_steps_per_unit=n_steps_per_unit, eps=1e-3)
        fp = f_payoff_plus(T_a.x)
        abs_v = T_a.at_zero(fp, np.array([t0]))[t0]
        zero_v = T0_a.at_zero(fp, np.array([t0]))[t0]
        robustness_table.append({
            "alpha": a_val, "T_abs_t0_f+(0)": abs_v,
            "T0_t0_f+(0)": zero_v, "gap": zero_v - abs_v
        })

    df_robust = pd.DataFrame(robustness_table)
    print(f"\n  Robustness table at t0={t0}:")
    print(df_robust.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 7. Plots
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Experiment 1: Example 4.1 — alpha={alpha}", fontsize=13)

    # (a) Local variance
    ax = axes[0, 0]
    ax.plot(x, a_vals, label=r"$a_\alpha(x)$", lw=2)
    ax.set_xlabel("x"); ax.set_ylabel(r"$a_\alpha(x)$")
    ax.set_title("Local variance")
    ax.legend()

    # (b) Semigroup values over time
    ax = axes[0, 1]
    t_vals_sorted = sorted(abs_at_zero.keys())
    abs_vals = [abs_at_zero[t] for t in t_vals_sorted]
    zero_vals = [zero_at_zero_best[t] for t in t_vals_sorted]
    ax.plot(t_vals_sorted, abs_vals, "b-o", label=r"$T^{\rm abs}_t f_+(0)$")
    ax.plot(t_vals_sorted, zero_vals, "r-s", label=r"$T^0_t f_+(0)$")
    ax.set_xlabel("t"); ax.set_ylabel("value at x=0")
    ax.set_title(r"Valuations at $x=0$"); ax.legend()

    # (c) Generator convergence
    ax = axes[1, 0]
    ax.loglog(conv_abs["h"], conv_abs["max_residual"], "b-o", label=r"$T^{\rm abs}$")
    ax.loglog(conv_zero["h"], conv_zero["max_residual"], "r-s", label=r"$T^0$")
    ax.set_xlabel("h"); ax.set_ylabel("Max generator residual")
    ax.set_title("Generator convergence"); ax.legend()

    # (d) Robustness over alpha
    ax = axes[1, 1]
    ax.bar([str(a) for a in df_robust["alpha"]], df_robust["gap"], color="steelblue")
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("Gap")
    ax.set_title(r"Valuation gap $T^0 - T^{\rm abs}$ at $t_0$=" + str(t0))

    plt.tight_layout()
    fig_path = os.path.join(results_dir, f"exp1_alpha{alpha:.1f}.png")
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"\n  Figure saved: {fig_path}")

    # ------------------------------------------------------------------ #
    # 8. eps-convergence plot
    # ------------------------------------------------------------------ #
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for eps in eps_values:
        z_by_t = zero_at_zero_by_eps[eps]
        ts = sorted(z_by_t.keys())
        ax2.plot(ts, [z_by_t[t] for t in ts], marker="o",
                 label=f"eps={eps:.1e}")
    ax2.plot(sorted(abs_at_zero.keys()),
             [abs_at_zero[t] for t in sorted(abs_at_zero.keys())],
             "k--", lw=2, label=r"$T^{\rm abs}$ (eps=0 limit)")
    ax2.set_xlabel("t"); ax2.set_ylabel(r"$T^0_t f_+(0)$")
    ax2.set_title(r"$T^0_t f_+(0)$ convergence as $\varepsilon \to 0$ (alpha=" + str(alpha) + ")")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    fig2_path = os.path.join(results_dir, f"exp1_eps_convergence_alpha{alpha:.1f}.png")
    plt.savefig(fig2_path, dpi=120)
    plt.close()
    print(f"  eps-convergence figure saved: {fig2_path}")

    # ------------------------------------------------------------------ #
    # 9. Collect summary metrics
    # ------------------------------------------------------------------ #
    t0_key = 0.1
    summary = {
        "alpha": alpha,
        "T_abs_t0_f_plus_0": abs_at_zero.get(t0_key, float("nan")),
        "T0_t0_f_plus_0": zero_at_zero_best.get(t0_key, float("nan")),
        "gap_t0": gap.get(t0_key, float("nan")),
        "gen_abs_max_res_finest_h": float(conv_abs["max_residual"][-1]),
        "gen_zero_max_res_finest_h": float(conv_zero["max_residual"][-1]),
        "strict_separation": bool(gap.get(t0_key, 0.0) > 1e-8),
    }
    return {
        "summary": summary,
        "gap": gap,
        "conv_abs": conv_abs,
        "conv_zero": conv_zero,
        "robustness": df_robust,
        "zero_at_zero_by_eps": zero_at_zero_by_eps,
    }


if __name__ == "__main__":
    results = {}
    for alpha in [0.2, 0.4]:
        results[alpha] = run_exp1(alpha=alpha)
    print("\nDone. All results saved to results/")
