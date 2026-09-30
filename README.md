# QCA: Exact Replication — Example 4.1 and 4.2 (Valuation Semigroups)

This repository implements numerical experiments from the paper on **Valuation Semigroups, Local Specifications, and Uncertainty Structures**, validating the key examples and theorems about the relationship between local and global valuation rules.

## Repository Structure

```
repo/
├── data/                  # Placeholder (no market data — synthetic models only)
├── src/                   # Core implementation modules
│   ├── local_spec.py      # Local specifications G_alpha, G_example2, G_downarrow
│   ├── valuation_semigroup.py  # T^abs, T^0, V_det semigroups
│   ├── pde_solver.py      # 1-D PDE solver (implicit Euler, HJB)
│   ├── generator.py       # Generator estimation via difference quotients
│   ├── killed_law.py      # Killed-law / discounted-pair equivalence (Theorem 6.4)
│   ├── realization.py     # Appendix-C realization construction pipeline
│   └── utils.py           # Smooth bounded test functions
├── exp/                   # Experiment scripts
│   ├── exp1_local_global.py        # Example 4.1: T^abs vs T^0
│   ├── exp2_effective_reduction.py # Example 4.2: G^downarrow
│   ├── exp3_theorem_validation.py  # HJB theorem validation suite
│   ├── exp4_killed_law.py          # Theorem 6.4: discounting-to-killing
│   └── exp5_realization.py         # Appendix-C realization pipeline
├── tests/                 # pytest test suite
├── results/               # Figures, metrics, RESULTS.md
├── run_all_experiments.py # Master runner
└── README.md
```

## Experiments

### Experiment 1: Example 4.1 — Same Local Spec, Different Global Valuations
Constructs two valuation semigroups T^{abs} (absorbing at 0) and T^0 (zero-sojourn) sharing the same local specification G_alpha(x,H) = 0.5*a_alpha(x)*H. Verifies T^{abs}_t f_+(0) = 0 < T^0_t f_+(0) for t > 0.

### Experiment 2: Example 4.2 — Effective Reduction G^downarrow
Demonstrates that G(x,z,p,H) = p + 1_{x=0}*p^+ collapses to V_t f(x) = f(x+t), and the effective specification G^downarrow = p is strictly smaller at x=0 yet generates the same admissible laws.

### Experiment 3: Theorem Validation Suite
Validates Theorems 3.1, 4.1, 4.2, 4.3/5.1, 5.3 on tractable HJB models (Model A: G = mu|p| + 0.5*sigma^2*H). Compares stochastic robust valuations against HJB PDE solutions.

### Experiment 4: Discounted-Pair to Killed-Law Equivalence (Theorem 6.4)
Verifies E^P[f(X_T) 1_{T<tau}] = E^Q[e^{-A_T} f(X_T)] via Monte Carlo simulation of OU and drifted BM processes.

### Experiment 5: Appendix-C Realization Construction
Implements the constructive proof pipeline: outer approximations, sup-convolution, mollification, ellipticity floor, coefficient selectors, and stopped realization inequality.

## Setup

```bash
pip install numpy scipy matplotlib pandas pytest
```

## Running

```bash
# Run all experiments
python run_all_experiments.py

# Run individual experiment
python exp/exp1_local_global.py

# Run tests
pytest tests/ -v
```

## Key Results

See [results/RESULTS.md](results/RESULTS.md) for metrics and figures after running.

## Mathematical Background

- `sigma_alpha(x) = |x|^alpha / (1 + |x|^alpha)`, alpha in (0, 1/2)
- `a_alpha(x) = sigma_alpha(x)^2` (local variance, degenerate at 0)
- `G_alpha(x,z,p,H) = 0.5 * a_alpha(x) * H` (local specification, no drift/killing)
- T^{abs}: process absorbed at 0 → T^{abs}_t f_+(0) = 0
- T^0: non-absorbing branch via regularized volatility (eps -> 0)
- Both have the same local generator yet differ globally
