"""Experiment 5: Appendix-C Realization Construction Validation.

Implements the computational version of the paper's proof pipeline for the
subsolution realization theorem in 1D:

  1. Continuous outer approximations G_n
  2. Sup-convolution regularization of a viscosity subsolution u
  3. Ellipticity floor lambda_j
  4. Continuous coefficient selectors zeta_j
  5. Realizing laws P_{n,j} via PDE with selected coefficients
  6. Stopped realization inequality
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.local_spec import a_alpha
from src.pde_solver import pde_implicit_euler_snapshots
from src.realization import (
    outer_approx_G_n,
    outer_approx_q_n,
    sup_convolution,
    mollify,
    add_ellipticity_floor,
    build_selector_zeta,
    construct_realizing_law_pde,
    check_stopped_inequality,
    run_realization_pipeline,
)
from src.utils import make_grid


def make_subsolution(
    x: np.ndarray,
    a_base: np.ndarray,
    T: float,
    n_steps: int = 500,
) -> np.ndarray:
    """Produce a viscosity subsolution u by solving the PDE backward from a payoff.

    For the benchmark operator L u = 0.5 * a(x) * u_xx, a subsolution satisfies:
      du/dt - L u <= 0
    We obtain an exact solution by solving du/dt = L u with initial condition
    f(x) = exp(-x^2) and evaluating at T. The solution is a classical subsolution
    (equality holds), which is the simplest valid example.

    Returns
    -------
    u : subsolution values on x, shape (N,)
    """
    from src.utils import f_gaussian
    f = f_gaussian(x)
    snaps = pde_implicit_euler_snapshots(
        f=f, x=x, a=a_base, t_grid=np.array([T]),
        n_steps_per_unit=n_steps
    )
    return snaps[T]


def run_exp5(
    alpha: float = 0.2,
    L: float = 5.0,
    N: int = 201,
    T: float = 0.5,
    n_diag_steps: int = 5,
    results_dir: str = "results",
) -> dict:
    """Run Experiment 5: Appendix-C realization pipeline."""
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment 5: Appendix-C Realization Construction (alpha={alpha})")
    print(f"{'='*60}")

    x = make_grid(L, N)
    a_base = a_alpha(x, alpha)

    # ------------------------------------------------------------------ #
    # 1. Construct benchmark subsolution u
    # ------------------------------------------------------------------ #
    u_subsol = make_subsolution(x, a_base, T, n_steps=300)
    print(f"  Subsolution range: [{u_subsol.min():.4f}, {u_subsol.max():.4f}]")

    # ------------------------------------------------------------------ #
    # 2. Source term q (for this benchmark, q = 0)
    # ------------------------------------------------------------------ #
    q_base = np.zeros(N)

    # ------------------------------------------------------------------ #
    # 3. Diagonal sequence parameters
    # ------------------------------------------------------------------ #
    n_seq = [2, 4, 8, 16, 32][:n_diag_steps]
    eps_seq = [0.5, 0.25, 0.1, 0.05, 0.02][:n_diag_steps]
    lam_seq = [0.2, 0.1, 0.05, 0.02, 0.01][:n_diag_steps]
    sigma_mol_seq = [0.5, 0.3, 0.2, 0.15, 0.1][:n_diag_steps]
    # a_set: admissible diffusion range (finite-control HJB)
    a_max = float(a_base.max()) + 0.5
    a_set = (0.0, a_max)

    # ------------------------------------------------------------------ #
    # 4. Run pipeline
    # ------------------------------------------------------------------ #
    pipeline_results = run_realization_pipeline(
        u_subsol=u_subsol,
        a_base=a_base,
        x=x,
        q_base=q_base,
        t=T,
        n_seq=n_seq,
        eps_seq=eps_seq,
        lam_seq=lam_seq,
        sigma_mol_seq=sigma_mol_seq,
        a_set=a_set,
        n_steps_per_unit=200,
    )

    df_pipeline = pd.DataFrame(pipeline_results)
    print("\n  Pipeline results:")
    print(df_pipeline[["step", "n", "eps", "lam", "max_violation",
                         "fraction_satisfied", "max_residual"]].to_string(index=False))

    # ------------------------------------------------------------------ #
    # 5. Detailed check of final diagonal step
    # ------------------------------------------------------------------ #
    final = pipeline_results[-1]
    print(f"\n  Final step diagnostics:")
    print(f"    Max violation (u - T_zeta_f):  {final['max_violation']:.4e}")
    print(f"    Fraction inequality satisfied: {final['fraction_satisfied']:.4f}")
    print(f"    Max PDE residual:              {final['max_residual']:.4e}")

    # ------------------------------------------------------------------ #
    # 6. Approximation quality: sup-convolution vs mollification
    # ------------------------------------------------------------------ #
    eps_test = 0.1
    u_eps = sup_convolution(u_subsol, x, eps_test)
    v_smooth = mollify(u_eps, x, sigma_mol=0.2)
    print(f"\n  Approximation quality:")
    print(f"    ||u_eps - u||_inf:        {float(np.max(np.abs(u_eps - u_subsol))):.4e}")
    print(f"    max(u_eps - u) [dominates]: {float(np.max(u_eps - u_subsol)):.4e}")
    print(f"    min(u_eps - u):              {float(np.min(u_eps - u_subsol)):.4e}")
    print(f"    ||v_smooth - u_eps||_inf: {float(np.max(np.abs(v_smooth - u_eps))):.4e}")

    # ------------------------------------------------------------------ #
    # 7. Plots
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Experiment 5: Appendix-C Realization (alpha={alpha})", fontsize=13)

    # (a) Subsolution and approximations
    ax = axes[0, 0]
    ax.plot(x, u_subsol, "b-", lw=2, label="u (subsolution)")
    ax.plot(x, u_eps, "r--", lw=1.5, label=r"$u^\varepsilon$ (sup-conv)")
    ax.plot(x, v_smooth, "g:", lw=1.5, label=r"$v$ (mollified)")
    ax.set_xlabel("x"); ax.set_ylabel("value")
    ax.set_title("Subsolution approximation chain")
    ax.legend(fontsize=8)

    # (b) Pipeline: max violation vs step
    ax = axes[0, 1]
    ax.plot(df_pipeline["step"], df_pipeline["max_violation"], "ro-", label="max violation")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Diagonal step"); ax.set_ylabel("max(u - T_zeta f)")
    ax.set_title("Stopped inequality violation"); ax.legend()

    # (c) Pipeline: max residual vs step
    ax = axes[1, 0]
    ax.semilogy(df_pipeline["step"], df_pipeline["max_residual"], "bs-")
    ax.set_xlabel("Diagonal step"); ax.set_ylabel("Max PDE residual")
    ax.set_title("PDE residual convergence")

    # (d) Selector function zeta_j (final step)
    n_last = n_diag_steps - 1
    a_n_last = a_base + 1.0 / n_seq[n_last]
    u_eps_last = sup_convolution(u_subsol, x, eps_seq[n_last])
    v_last = mollify(u_eps_last, x, sigma_mol_seq[n_last])
    zeta_last = build_selector_zeta(v_last, x, a_set)

    ax = axes[1, 1]
    ax.plot(x, a_base, "b-", lw=1.5, label=r"$a_\alpha(x)$")
    ax.plot(x, zeta_last, "r--", lw=1.5, label=r"$\zeta_j(x)$ selector")
    ax.set_xlabel("x"); ax.set_ylabel("coefficient")
    ax.set_title("Coefficient selector vs base diffusion")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(results_dir, "exp5_realization.png")
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"\n  Figure saved: {fig_path}")

    summary = {
        "final_max_violation": final["max_violation"],
        "final_fraction_satisfied": final["fraction_satisfied"],
        "final_max_residual": final["max_residual"],
        "sup_conv_dominates": bool(float(np.min(u_eps - u_subsol)) >= -1e-10),
        "pipeline_converges": bool(df_pipeline["max_residual"].iloc[-1]
                                    < df_pipeline["max_residual"].iloc[0]),
    }
    print(f"\n  Summary: {summary}")
    return {"summary": summary, "df_pipeline": df_pipeline}


if __name__ == "__main__":
    run_exp5()
    print("\nDone.")
