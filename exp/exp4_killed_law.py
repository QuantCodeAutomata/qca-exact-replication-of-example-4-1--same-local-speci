"""Experiment 4: Discounted-Pair to Killed-Law Equivalence (Theorem 6.4).

Validates:
  E^P[ f(X_T) 1_{T < tau_infty} ] = E^Q[ e^{-A_T} f(X_T) ]

for multiple payoffs, horizons, discount rates, and process types.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.killed_law import (
    cumulative_discount,
    discounted_expectation,
    killed_expectation,
    apply_killing,
    recover_discount_process,
    simulate_ou,
    simulate_bm_drift,
    validate_theorem_6_4,
)
from src.utils import f_gaussian, f_payoff_plus, f_rational


# ---- Discount rate functions -----------------------------------------------

def k_constant(x: np.ndarray, k: float = 0.5) -> np.ndarray:
    """Constant discount rate k(x) = k."""
    return np.full_like(x, k, dtype=float)


def k_quadratic(x: np.ndarray, scale: float = 0.3) -> np.ndarray:
    """State-dependent rate k(x) = scale * x^2."""
    return scale * x ** 2


def k_sigmoid(x: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Sigmoid-shaped rate k(x) = scale / (1 + exp(-x))."""
    return scale / (1.0 + np.exp(-x))


def run_exp4(
    n_paths: int = 30000,
    n_steps: int = 300,
    T_vals: list[float] | None = None,
    x0_vals: list[float] | None = None,
    results_dir: str = "results",
) -> dict:
    """Run Experiment 4: Killed law / discounting equivalence."""
    os.makedirs(results_dir, exist_ok=True)

    if T_vals is None:
        T_vals = [0.1, 0.5, 1.0, 2.0]
    if x0_vals is None:
        x0_vals = [-1.0, 0.0, 1.0]

    print(f"\n{'='*60}")
    print("Experiment 4: Discounted-Pair to Killed-Law (Theorem 6.4)")
    print(f"{'='*60}")

    payoffs = {
        "gaussian": f_gaussian,
        "payoff_plus": f_payoff_plus,
        "rational": f_rational,
    }

    discount_rates = {
        "constant_k05": lambda x: k_constant(x, k=0.5),
        "quadratic": lambda x: k_quadratic(x, scale=0.3),
        "sigmoid": lambda x: k_sigmoid(x, scale=0.5),
    }

    processes = {
        "ou": dict(theta=1.0, mu=0.0, sigma=1.0),
        "bm": dict(b=0.0, sigma=0.5),
    }

    rows = []
    rng = np.random.default_rng(42)

    for proc_name, proc_kwargs in processes.items():
        for k_name, k_func in discount_rates.items():
            for fname, f_func in payoffs.items():
                for T in T_vals:
                    for x0 in x0_vals:
                        dt = T / n_steps
                        rng_sim = np.random.default_rng(int(abs(x0 * 100 + T * 17 + 1000)))

                        if proc_name == "ou":
                            X = simulate_ou(
                                x0=x0, n_paths=n_paths, n_steps=n_steps, dt=dt,
                                rng=rng_sim, **proc_kwargs
                            )
                        else:
                            X = simulate_bm_drift(
                                x0=x0, n_paths=n_paths, n_steps=n_steps, dt=dt,
                                rng=rng_sim, **proc_kwargs
                            )

                        A = cumulative_discount(X, k_func, dt)
                        rng_kill = np.random.default_rng(int(abs(x0 * 53 + T * 31 + 2000)))
                        alive, _ = apply_killing(X, A, rng=rng_kill)

                        T_idx = n_steps
                        ke = killed_expectation(X, alive, f_func, T_idx)
                        de = discounted_expectation(X, A, f_func, T_idx)

                        abs_err = abs(ke - de)
                        rel_err = abs_err / (abs(de) + 1e-12)

                        rows.append({
                            "process": proc_name, "discount": k_name,
                            "payoff": fname, "T": T, "x0": x0,
                            "killed_exp": ke, "disc_exp": de,
                            "abs_error": abs_err, "rel_error": rel_err,
                        })

    df = pd.DataFrame(rows)

    print(f"\n  Overall statistics:")
    print(f"    N_experiments: {len(df)}")
    print(f"    Max abs error:  {df['abs_error'].max():.4e}")
    print(f"    Mean abs error: {df['abs_error'].mean():.4e}")
    print(f"    Max rel error:  {df['rel_error'].max():.4e}")
    print(f"    Mean rel error: {df['rel_error'].mean():.4e}")

    # Per-configuration table
    summary_by_config = df.groupby(["process", "discount"])["abs_error"].agg(
        ["mean", "max"]
    ).reset_index()
    summary_by_config.columns = ["process", "discount", "mean_abs_err", "max_abs_err"]
    print(f"\n  Per-configuration summary:")
    print(summary_by_config.to_string(index=False))

    # ------------------------------------------------------------------ #
    # Recovery validation
    # ------------------------------------------------------------------ #
    print(f"\n  Recovery of A from killed law:")
    T_rec = 1.0
    dt_rec = T_rec / n_steps
    rng_rec = np.random.default_rng(1000)
    X_rec = simulate_ou(x0=0.0, n_paths=n_paths, n_steps=n_steps, dt=dt_rec,
                         rng=rng_rec, theta=1.0, mu=0.0, sigma=1.0)
    k_ref = lambda x: k_constant(x, k=0.5)
    A_ref = cumulative_discount(X_rec, k_ref, dt_rec)
    alive_rec, _ = apply_killing(X_rec, A_ref, rng=np.random.default_rng(2000))

    A_hat = recover_discount_process(alive_rec, dt_rec)
    # True A_t = k * t (constant rate, deterministic)
    t_axis = np.arange(n_steps + 1) * dt_rec
    A_true = 0.5 * t_axis
    recovery_err = np.abs(A_hat - A_true)
    print(f"    Max recovery error: {float(np.max(recovery_err)):.4e}")
    print(f"    Mean recovery error: {float(np.mean(recovery_err)):.4e}")

    # ------------------------------------------------------------------ #
    # Injectivity test: two distinct (A, Q) -> different killed expectations
    # ------------------------------------------------------------------ #
    print(f"\n  Injectivity test (distinct (A,Q) -> distinct P):")
    T_inj = 0.5
    dt_inj = T_inj / n_steps
    rng_inj = np.random.default_rng(9999)

    # Pair 1: OU with k=0.5
    X1 = simulate_ou(x0=0.0, n_paths=n_paths, n_steps=n_steps, dt=dt_inj,
                      rng=rng_inj, theta=1.0, mu=0.0, sigma=1.0)
    A1 = cumulative_discount(X1, lambda x: k_constant(x, 0.5), dt_inj)
    alive1, _ = apply_killing(X1, A1, rng=np.random.default_rng(11))
    ke1 = killed_expectation(X1, alive1, f_gaussian, n_steps)

    # Pair 2: OU with k=1.0 (different discount -> different P)
    rng_inj2 = np.random.default_rng(9999)  # same paths, different killing
    X2 = simulate_ou(x0=0.0, n_paths=n_paths, n_steps=n_steps, dt=dt_inj,
                      rng=rng_inj2, theta=1.0, mu=0.0, sigma=1.0)
    A2 = cumulative_discount(X2, lambda x: k_constant(x, 1.0), dt_inj)
    alive2, _ = apply_killing(X2, A2, rng=np.random.default_rng(22))
    ke2 = killed_expectation(X2, alive2, f_gaussian, n_steps)

    print(f"    E^P1[f] = {ke1:.6f}  (k=0.5)")
    print(f"    E^P2[f] = {ke2:.6f}  (k=1.0)")
    print(f"    Distinct: {abs(ke1 - ke2) > 1e-4}")

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Experiment 4: Killed Law ↔ Discounted Expectation (Thm 6.4)", fontsize=12)

    # (a) Error distribution
    ax = axes[0]
    ax.hist(df["abs_error"], bins=40, color="steelblue", edgecolor="k", alpha=0.8)
    ax.set_xlabel("Absolute error |E^P - E^Q[e^{-A}]|")
    ax.set_ylabel("Count")
    ax.set_title("Error distribution (all configs)")

    # (b) Scatter: killed vs discounted
    ax = axes[1]
    ax.scatter(df["disc_exp"], df["killed_exp"], alpha=0.3, s=10)
    mn = df[["disc_exp", "killed_exp"]].min().min()
    mx = df[["disc_exp", "killed_exp"]].max().max()
    ax.plot([mn, mx], [mn, mx], "r--")
    ax.set_xlabel(r"$E^Q[e^{-A_T} f(X_T)]$")
    ax.set_ylabel(r"$E^P[f(X_T) \mathbf{1}_{T < \tau}]$")
    ax.set_title("Killed vs Discounted expectation")

    # (c) Recovery of A
    ax = axes[2]
    ax.plot(t_axis, A_true, "b-", lw=2, label=r"$A_t$ true (0.5t)")
    ax.plot(t_axis, A_hat, "r--", lw=1.5, label=r"$\hat{A}_t$ recovered")
    ax.set_xlabel("t"); ax.set_ylabel("A_t")
    ax.set_title("Recovery of discount process A_t")
    ax.legend()

    plt.tight_layout()
    fig_path = os.path.join(results_dir, "exp4_killed_law.png")
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"\n  Figure saved: {fig_path}")

    summary = {
        "max_abs_error": float(df["abs_error"].max()),
        "mean_abs_error": float(df["abs_error"].mean()),
        "max_rel_error": float(df["rel_error"].max()),
        "recovery_max_error": float(np.max(recovery_err)),
        "injectivity_distinct": bool(abs(ke1 - ke2) > 1e-4),
        "ke1_k05": ke1, "ke2_k10": ke2,
    }
    print(f"\n  Summary: {summary}")
    return {"summary": summary, "df": df, "A_hat": A_hat, "A_true": A_true}


if __name__ == "__main__":
    run_exp4()
    print("\nDone.")
