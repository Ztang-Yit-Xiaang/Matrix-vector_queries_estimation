# Adaptive Randomized Trace Estimation and Risk-Certification Research


> **Current recovery record — 2026-09-14:** See [[CURRENT_STATE]] and [[recovery_audit_20260914]]. The 1,860-row Haar and 480-row Hessian datasets have been regenerated and validated; 217 maintained tests pass. Phase 2C proof/boundary corrections are recorded in [[proof_structural_paired_difference_confidence]] and [[structural_paired_difference_confidence_phase2c]]. Gated results remain exploratory, including the reproduced coordinate-aligned failure. Older entries below retain their historical context; they are superseded where the recovery audit corrects them.

[![Tests](https://img.shields.io/badge/tests-152%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-purple.svg)]()

This repository contains the research codebase for **adaptive randomized trace
estimation and realized-risk certification studies**, developed as part of
UROP Summer 2026 under the mentorship of Swati Padmanabhan. No online
confidence-certified allocator has yet been proved.

The project investigates adaptive query allocation in **Hutch++**, proves
fundamental diagnostic and certification limits, and retains a local
two-stage Ritz-gap estimator as an exploratory prototype rather than a
certified final method.

---

## 🌟 Key Research Breakthroughs

```
                               ┌────────────────────────────────────────────────────────┐
                               │       Total Query Budget Conservation: q + r + ℓ = m   │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                ┌──────────────────────────┴──────────────────────────┐
                                │                                                     │
                                ▼                                                     ▼
                  Stage 1: Zero-Cost Screening                         Stage 2: Gated Adaptive Subspace
             (Pilot Ritz gap γ_gap on b_0 queries)                 (Resolves knee: q_target = r_knee + p)
                                │                                                     │
                 ┌──────────────┴──────────────┐                                      ▼
                 ▼                             ▼                     Tail-Risk Insurance Oversampling
          γ_gap < τ_gap                 γ_gap ≥ τ_gap                     (Insures against square sketch
       (Benign spectrum)              (Step knee detected)                    ill-conditioning)
                 │                             │                                      │
                 ▼                             └──────────────────────────────────────┘
      Seamless Fallback to q_0
      (pilot reused inside q_0)
```

1. **In-Sample Pilot Error Explosion (RQ1)**: Proved and demonstrated that extrapolating low-rank Ritz values into residual spectrum predictions explodes tail estimation variance by up to $10^{10}\times$.
2. **Subspace Horizon Threshold Law (RQ2)**: Proved that pilot sketch width must satisfy $b \ge 1.33 r_\star$ before spectral knees can be resolved from noise.
3. **Zero-Oversampling Fragility Audit**: Established an exact graph-subspace identity and empirically found that rare square-boundary failures can move the mean-risk minimizer by $+1$ on the frozen step experiment. The exact shift is not universal.
4. **Phase 1A–1D Risk Certification Program**:
   - *Phase 1A (Feasibility)*: Empirically observed a 0.0134% false-safe rate for ordinary sample variance over the frozen 960,000-row artifact; this is not a theorem-level guarantee.
   - *Phase 1B (Budget Emulation)*: Formally proved construction costs $c_{\text{pre}} = \max\{q_0+r_0, q_a+r_a\}$ and residual capacity $\ell_{\text{paid}} = m - c_{\text{pre}} - s$.
   - *Phase 1C (Chebyshev Bounds)*: Derived exact sample variance variance and proved Chebyshev bounds are budget-vacuous ($\varepsilon > 1$) for $s \le 32$.
   - *Phase 1D (Linear Tail Analytic No-Go)*: Applied Cortinovis–Kressner Theorem 2 and proved that truncation--Bernstein on the linear Hoeffding component satisfies $D_{s, \kappa} > e^{-0.2} \approx 0.819 \gg \delta_{\text{linear}}$ (`STRONG LINEAR NO-GO`).
5. **Exploratory Two-Stage Gate**: A local prototype reuses an eight-column pilot inside the Standard-Hutch++ fallback allocation and shows mixed results on a small development benchmark. It does not use or solve the risk-certification problem.

---

## 📊 Summary of Estimators

All estimators strictly route matrix-vector products through `MatVecOracle` and satisfy the **3-way conservation law**:

$$\boxed{q + r_{\text{actual}} + \ell = m}$$

| Estimator | Function Name | Strategy | Complexity / Rate |
| :--- | :--- | :--- | :--- |
| **Classical Hutchinson** | `Hutchinson` | Uniform random probing | $O(1/\varepsilon^2)$ |
| **Standard Hutch++** | `Hutch_pplus` | Fixed 3-way split ($q = m/3, \ell = m/3$) | $O(1/\varepsilon)$ |
| **NA-Hutch++** | `NA_Hutch_pplus` | Non-adaptive single-batch generation | $O(1/\varepsilon)$ |
| **Gaussian Hutch++** | `Gaussian_Hutch_pplus` | Standard Gaussian probing | $O(1/\varepsilon)$ |
| **Sequential Pilot Hutch++** | `Adaptive_Hutch_pplus_SequentialPilot` | Multi-stage pilot expansion with trust-region guards | Adaptive |
| **Marginal Risk Hutch++** | `Adaptive_Hutch_pplus_MarginalRisk` | Marginal energy-drop stopping | Adaptive |
| **Two-Stage Gated (exploratory)** | `Adaptive_Hutch_pplus_TwoStageGated` | Reused-pilot screening + gated knee allocation | Heuristic prototype |

---

## 📈 Benchmark Results

### 1. Exploratory synthetic benchmark ($m=60, d=100$)

This is a five-spectrum, one-orientation, 50-trial development check with no
confidence intervals. It is not a held-out or certification result.

| Spectrum Type | Hutchinson Median Rel Err | Standard Hutch++ Median Rel Err | TwoStageGated (Ours) Median Rel Err | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Step Spectrum ($r_\star=5, \eta=10^{-3}$)** | $0.059309$ | $0.000128$ | **$0.000070$** | **$1.83\times$ better (45.3% reduction)** |
| **Step Spectrum ($r_\star=15, \eta=10^{-3}$)** | $0.021888$ | **$0.000038$** | $0.000048$ | 26.7% worse median than Hutch++ |
| **Flat Power-Law ($c=0.5$)** | $0.008304$ | $0.011627$ | **$0.009480$** | **18.5% better than Hutch++** |
| **Moderate Power-Law ($c=1.0$)** | $0.017421$ | $0.007738$ | **$0.007352$** | Median slightly lower; mean higher |
| **Steep Power-Law ($c=2.0$)** | $0.113232$ | **$0.000930$** | $0.001023$ | Median higher; mean lower |

### 2. Real-World Datasets
* **YearPredictionMSD** ($d=90, n=50,000, r_{\text{eff}}=28.70$): Crossover occurs at $m \approx 30 \approx r_{\text{eff}}$, yielding $3.5\times$ variance reduction over Hutchinson.
* **Wiki-Vote Triangle Network** ($d=7,115, \text{edges}=103,689, r_{\text{eff}}=64.42$): Crossover occurs at $m \approx 65 \approx r_{\text{eff}}$, yielding $4.2\times$ variance reduction.

---

## 📂 Codebase Structure

```
.
├── src/                                   # Core estimator implementations & theory
│   ├── trace_baseline.py                  # MatVecOracle, Hutch++, TwoStageGated estimators
│   ├── data_loaders.py                    # Real-world data loaders (Wiki-Vote, YearPredictionMSD)
│   ├── rademacher_sample_variance_confidence.py # Phase 1C Chebyshev confidence module
│   ├── rademacher_linear_projection_no_go.py    # Phase 1D linear lower-tail audit module
│   └── theorem15_certification.py         # HMT Gaussian knee certification bounds
│
├── experiments/                           # Production experimental & diagnostic scripts
│   ├── run_twostage_gated_benchmark.py    # TwoStageGated comparative evaluation
│   ├── postprocess_rademacher_linear_projection_no_go_phase1d.py # Phase 1D artifact generator
│   ├── run_q_rank_vs_realized_risk_mechanism.py # Zero-oversampling fragility audit
│   ├── run_exact_risk_bridge.py           # 24-spectrum exact conditional risk bridge
│   └── generate_recent_progress_figures.py # Publication-quality progress plots
│
├── tests/                                 # Automated unit & regression test suite (217 tests)
│   ├── test_twostage_gated_estimator.py   # TwoStageGated budget & trigger tests
│   ├── test_rademacher_linear_projection_no_go_phase1d.py # Phase 1D mathematical tests
│   ├── test_new_estimators.py             # Query accounting and unbiasedness tests
│   └── test_probe_identical_equivalence.py# Mathematical equivalence tests
│
├── docs/                                  # Formal mathematical proof notes
│   ├── proof_notation.md                  # Standardized notation ledger
│   ├── proof_sequential_unbiasedness.md   # Theorem 1: Martingale pilot unbiasedness
│   ├── proof_rank_aware_risk.md           # Theorems 2-3: Exact conditional risk
│   ├── proof_step_ritz_gap.md             # Theorems 9-10: Ritz gap & principal angles
│   ├── proof_near_oracle_regret.md        # Theorems 4, 11-14: Regret & marginal energy
│   ├── proof_budget_aware_certification.md# Phase 1B: Budget accounting proof
│   └── proof_rademacher_sample_variance_confidence.md # Phase 1C/1D: Sample variance theory
│
├── reports/                               # Standalone technical reports & audits
│   ├── rademacher_linear_projection_no_go_phase1d.md # Phase 1D analytic no-go report
│   ├── direct_rademacher_risk_certification_phase1b_budget.md # Phase 1B budget report
│   ├── q_rank_vs_realized_risk_mechanism.md # Zero-oversampling fragility report
│   └── rank_deficient_bridge_analysis.md  # Multi-budget risk bridge analysis
│
└── results/                               # Validated production CSV artifacts & manifests
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/username/Matrix-vector_queries_estimation.git
cd Matrix-vector_queries_estimation

# Install dependencies via conda or uv
conda create -n randnla python=3.12 numpy scipy matplotlib pandas pytest
conda activate randnla
```

### 2. Running the Test Suite
```bash
pytest tests/ -v
# Output: 217 passed in the September 14 recovery run
```

### 3. Running the TwoStageGated Estimator
```python
import numpy as np
from src.trace_baseline import MatVecOracle, Adaptive_Hutch_pplus_TwoStageGated

# Construct sample matrix
d = 500
A = np.random.randn(d, d)
A = A.T @ A

# Initialize exact query tracking oracle
oracle = MatVecOracle(A)
m_budget = 120

# Run Two-Stage Gated Estimator
tr_est, diag = Adaptive_Hutch_pplus_TwoStageGated(
    oracle,
    m=m_budget,
    d=d,
    b_0=8,
    tau_gap=1.5,
    p_oversample=2,
    return_diagnostics=True
)

print(f"Exact Trace:     {np.trace(A):.4f}")
print(f"Estimated Trace: {tr_est:.4f}")
print(f"Queries Used:    {oracle.query_count} (Budget: {m_budget})")
print(f"Knee Triggered:  {diag['is_gated_trigger']}")
```

---

## 📜 Mathematical Notation Standards

* $S \in \mathbb{R}^{d \times q}$: Random sketching matrix ($S_1 = U_1^T S$, $S_2 = U_2^T S$).
* $Q \in \mathbb{R}^{d \times r}$: Orthonormal basis spanning range($AS$).
* $R = I - QQ^T$: Symmetric orthogonal projector onto $({\text{range}}(Q))^\perp$.
* $E_G(Q) = \|RAR\|_F^2$: Realized Gaussian residual energy.
* $E_R(Q) = \sum_{i \ne j} (RAR)_{ij}^2$: Realized Rademacher residual energy.
* $\ell = m - q - r$: Number of fresh residual queries.

---

## 📄 License & Attribution

Developed under MIT License for the UROP 2026 RandNLA project.
For theoretical notes and research inquiries, refer to `docs/` and `reports/`.
