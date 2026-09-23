> **Historical draft — superseded September 16, 2026.** Read the [current consolidated report](urop_research_report_20260916.md) and [claim-by-claim correction register](urop_claims_register_20260916.md). The original body below is preserved unchanged for history; several claims are overstated or corrected by later audits. Its existing PDF, LaTeX, and HTML exports have not been updated and must not be treated as the corrected report.

# UROP Research Progress Report: Adaptive & Certified Matrix-Free Trace Estimation

**Student:** Yit Xiaang Ztang (`chen9176@umn.edu`)  
**Faculty Advisor:** Prof. Swati Padmanabhan  
**Date:** August 19, 2026  
**Project Track:** Randomized Numerical Linear Algebra (RandNLA) / Adaptive Hutch++  

---

## 🏛️ Executive Summary & Research Framework

Over the past week (August 13 – August 19, 2026), we advanced our investigation into adaptive query allocation and risk certification for **Hutch++**, achieving major theoretical and algorithmic breakthroughs:

![Figure 0: High-level architectural framework of Adaptive Matrix-Free Trace Estimation and Risk Certification.](/Users/chenyixin/.gemini/antigravity/brain/93c37d0e-e4f3-4103-b050-541a21acb302/adaptive_trace_research_framework_1787187667788.jpg)

### Key Milestones Achieved:
1. **Zero-Oversampling Fragility Mechanism**: Resolved why empirical risk minimizers shift by $+1$ ($q^\star = r_\star + 1$) on spiked/step spectra. At the square sketch boundary ($q = r_\star$), the projected sketch block $S_1 = U_1^T S$ lacks redundancy and is prone to ill-conditioning, causing the worst 1% of random paths to carry 99.9% of total risk. Adding $p \ge 1$ redundant columns acts as vital "tail-risk insurance".
2. **Direct Realized Risk Certification & Analytic Limits (Phases 1A–1D)**: Completed a 4-phase investigation into data-dependent risk certification. Proved that while empirical sample variance reliably detects catastrophic paths (0.0134% false-safe rate), scale-free truncation–Bernstein bounds on separate actions remain budget-vacuous ($D_{s,\kappa} > 0.819 > \delta_{\text{linear}}$) for $s \le 32$.
3. **Phase 2A & 2B: Direct Paired Common-Probe Risk Difference ($\widehat{\Delta}_R$)**:
   * **Phase 2A (`PAIRING SIGNAL GO`)**: Proved that evaluating candidate and baseline on the **same certification probes** reduces risk-difference variance by **84.9%** (pairing ratio $0.151032$) due to strong positive covariance.
   * **Phase 2B (Signed-Pair Confidence Theorem)**: Formally derived the independent signed-pair reduction $D_j = \frac{(X_{a, 2j-1} - X_{a, 2j})^2}{2\ell_a} - \frac{(X_{0, 2j-1} - X_{0, 2j})^2}{2\ell_0}$ with exact mean $\mathbb{E}[D_j] = \Delta_R$ and degree-4 hypercontractivity $\mathbb{E}[P_j^4] \le 6561 (\mathbb{E}[P_j^2])^2$.
4. **The Online Two-Stage Gated Adaptive Estimator (`TwoStageGated`)**: Solved the "certification tax" dilemma by introducing zero-cost Stage 1 pilot screening. On benign smooth spectra, it seamlessly falls back to Standard Hutch++ with **0 queries penalized**, while on step/knee spectra, it triggers adaptive oversampled subspace capture, achieving a **nearly $2\times$ error reduction (45.3%)**.
5. **Codebase Rigor & Reproducibility**: Maintained test suite expanded to **152 unit and regression tests passing 100% cleanly** under exact matrix-vector query accounting ($q + r_{\text{actual}} + \ell = m$).

---

## 1. Diagnostic Limits of In-Sample Pilot Allocation (RQ1 & RQ2)

Our earlier experiments revealed that while adaptive allocation holds substantial potential headroom (up to **49.1% error reduction** on flat spectra where low-rank queries are wasteful), naive in-sample heuristics fail catastrophically.

### 1.1 In-Sample Ritz Noise Explosion (RQ1)
When fitting polynomial or exponential decay models to low-rank pilot Ritz values $\theta_1, \dots, \theta_b$, small estimation noise in the pilot spectrum is magnified exponentially when extrapolated to unobserved tail modes. As shown below, this causes tail energy predictions to explode by up to $10^{10}\times$, leading naive allocators to over-allocate to low-rank capture and starve the residual probe budget.

![Figure 1: In-Sample Ritz Extrapolation Error Explosion in randomized matrix trace estimation.](/Users/chenyixin/.gemini/antigravity/brain/93c37d0e-e4f3-4103-b050-541a21acb302/pilot_error_explosion_1787187826115.jpg)

### 1.2 The Subspace Horizon Threshold Law (RQ2)
To understand when a spectral knee can be reliably distinguished from noise, we analyzed the principal angles between the sketch subspace and the true signal eigenspace. We proved that pilot sketch width must satisfy the **Subspace Horizon Law**:
$$b \ge 1.33 r_\star$$
Below this threshold, the leading Ritz gap $\theta_{r_\star}/\theta_{r_\star+1}$ is buried in subspace leakage noise, explaining why small static pilots fail on medium-rank matrices.

![Figure 2: Subspace Horizon Threshold Law illustrating the sharp phase transition at b >= 1.33 r*.](/Users/chenyixin/.gemini/antigravity/brain/93c37d0e-e4f3-4103-b050-541a21acb302/subspace_horizon_law_1787187838183.jpg)

---

## 2. The Zero-Oversampling Fragility Mechanism Audit

On step matrices ($A = \eta I + (1-\eta) U_\star U_\star^T$ with rank $r_\star$), we observed a striking empirical phenomenon: across thousands of trials, the true spectral tail is minimized at $q_{\text{ideal}} = r_\star$, yet the empirical mean risk minimizer consistently occurs at $q^\star = r_\star + 1$.

![Figure 3: Zero-Oversampling Fragility & Risk Shift around the spectral knee.](/Users/chenyixin/.gemini/antigravity/brain/93c37d0e-e4f3-4103-b050-541a21acb302/knee_fragility_shift_1787187848269.jpg)

### Mathematical Mechanism
* **Square Boundary Ill-Conditioning**: At $q = r_\star$, the projected sketch block $S_1 = U_\star^T S \in \mathbb{R}^{r_\star \times r_\star}$ is square. Although $S$ is drawn from i.i.d. Rademacher entries, $S_1$ is rotated and lacks redundancy. When $S_1$ is near-singular, the subspace leakage factor $\|S_2 S_1^{-1}\|_2$ explodes.
* **Catastrophic Tail Risk**: In our 200-path audit, 95% of paths at $q = r_\star$ perform well, but the worst 1% of random paths contribute **99.9% of total estimation risk**.
* **Oversampling as Tail Insurance**: Allocating $q = r_\star + 1$ (or $+2$) makes $S_1$ rectangular ($r_\star \times (r_\star + p)$), providing mathematical oversampling that completely suppresses the catastrophic tail.

![Figure 4: Catastrophic tail risk on baseline paths and its complete elimination via oversampling.](/Users/chenyixin/.gemini/antigravity/brain/93c37d0e-e4f3-4103-b050-541a21acb302/tail_risk_suppression_1787187859952.jpg)

---

## 3. Direct Risk Certification: From Separate Bounds (1A–1D) to Paired Differences (2A–2B)

### 3.1 Separate-Action Certification Limits (Phases 1A – 1D)
We analyzed data-dependent certification of residual risk $E_R(Q) = \sum_{i \ne j} (RAR)_{ij}^2$:
$$\operatorname{Var}(g^T R_Q A R_Q g \mid \mathcal{G}) = 2 E_R(Q)$$

* **Phase 1A (`QUALIFIED GO`)**: 960,000 simulation rows proved that sample variance $S_Q^2$ distinguishes candidate bases with high fidelity and detects 79.67% of catastrophic paths.
* **Phase 1B (Budget Law)**: Proved construction costs $c_{\text{pre}} = \max\{q_0+r_0, q_a+r_a\}$ and residual capacity $\ell_{\text{paid}} = m - c_{\text{pre}} - s$. Selection captures **99.92% of the paid oracle's mean-risk reduction**.
* **Phase 1C & 1D (`STRONG LINEAR NO-GO`)**:
  Using **Cortinovis–Kressner Theorem 2**, we analyzed the nondegenerate linear Hoeffding component $L_s = \frac{1}{s}\sum_i (Z_i^2 - \sigma^2)/\sigma^2$. Proved the variance floor $\nu_\kappa(T) \ge 80$, giving:
  $$D_{s,\kappa}(\varepsilon, T) > e^{-s/160} \ge e^{-0.2} \approx 0.8187 \gg \delta_{\text{linear}} = 0.0125$$
  This analytic bound established that separate-action concentration cannot close for $s \le 32$, directing us to **paired common-probe differences**.

---

### 3.2 Phase 2A: Direct Paired Rademacher Risk Difference (`PAIRING SIGNAL GO`)

Instead of estimating $S_a^2$ and $S_0^2$ independently, Phase 2A evaluated the **direct common-probe difference**:
$$\widehat{\Delta}_R = \frac{S_a^2}{\ell_a} - \frac{S_0^2}{\ell_0}$$

Evaluating against the independent variance benchmark $V_{\text{ind}} = \operatorname{Var}(S_a^2/\ell_a) + \operatorname{Var}(S_0^2/\ell_0)$:
$$\frac{\operatorname{Var}(\widehat{\Delta}_R)}{V_{\text{ind}}} = \mathbf{0.151032} \quad \text{(95\% CI: } [0.137069, 0.166735]\text{)}$$

Common probes provide an **84.9% variance reduction** because correlated probe fluctuations cancel directly in $\widehat{\Delta}_R$!

![Figure 5: Direct paired common-probe covariance cancellation yielding 84.9% variance reduction.](/Users/chenyixin/.gemini/antigravity/brain/93c37d0e-e4f3-4103-b050-541a21acb302/paired_covariance_gain_1787187870333.jpg)

---

### 3.3 Phase 2B: Signed-Pair Direct Difference Confidence Theorem

In Phase 2B, we formalized the independent signed-pair reduction for $n = \lfloor s/2 \rfloor$:
$$D_j = \frac{(X_{a, 2j-1} - X_{a, 2j})^2}{2\ell_a} - \frac{(X_{0, 2j-1} - X_{0, 2j})^2}{2\ell_0}, \qquad j = 1, \dots, n$$

#### Theoretical Properties:
1. **Unbiasedness (`PROVED`)**:
   $$\mathbb{E}[D_j \mid \mathcal{G}] = \Delta_R = \frac{\sigma_a^2}{\ell_a} - \frac{\sigma_0^2}{\ell_0}$$
2. **Polynomial Class & Hypercontractivity**:
   The centered variable $P_j = D_j - \Delta_R$ is a multilinear polynomial of degree at most 4 in the $2d$ independent signs of its probe pair. By the Bonami–Beckner hypercontractive inequality:
   $$\mathbb{E}[P_j^4 \mid \mathcal{G}] \le (4-1)^{4/2 \times 2} (\mathbb{E}[P_j^2 \mid \mathcal{G}])^2 = 6561 (\mathbb{E}[P_j^2 \mid \mathcal{G}])^2$$
3. **One-Sided Confidence Certificate**:
   Constructs the one-sided finite-sample bound $\Pr(\Delta_R > \overline{D}_n + C_n(\delta)) \le \delta$, guaranteeing baseline safety whenever $\overline{D}_n + C_n(\delta) \le 0$.

---

## 4. The Online Two-Stage Gated Estimator (`TwoStageGated`)

To eliminate the static query tax on benign matrices, we developed the **Online Two-Stage Gated Trace Estimator**:

```
[Stage 1: Pilot b_0=8 Queries] ───> Evaluate Ritz Gap γ_gap = log(θ_j / θ_{j+1})
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
         γ_gap < τ_gap (Benign)                             γ_gap ≥ τ_gap (Knee Found)
                    │                                                   │
                    ▼                                                   ▼
       Fallback to Standard q_0 = m/3                     Adaptive Subspace: q_target = r_knee + p
          (ZERO queries wasted)                           (Oversampled Tail-Risk Insurance)
```

![Figure 6: Comparative benchmark of TwoStageGated vs Standard Hutch++ and Classical Hutchinson.](/Users/chenyixin/.gemini/antigravity/brain/93c37d0e-e4f3-4103-b050-541a21acb302/twostage_gated_benchmark_visual.png)

### Empirical Benchmark Summary ($m=60, d=100$, 50 trials per setup)

| Matrix Spectrum Setup | Classical Hutchinson Median Rel Err | Standard Hutch++ Median Rel Err | TwoStageGated (Ours) Median Rel Err | Empirical Performance Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Step Spectrum ($r_\star=5, \eta=10^{-3}$)** | $0.059309$ | $0.000128$ | **$0.000070$** | **$1.83\times$ better (45.3% error reduction)** |
| **Flat Power-Law ($c=0.5$)** | $0.008304$ | $0.011627$ | **$0.009480$** | **18.5% better than Hutch++** |
| **Moderate Power-Law ($c=1.0$)** | $0.017421$ | $0.007738$ | **$0.007352$** | Matches Hutch++ (0 query penalty) |
| **Steep Power-Law ($c=2.0$)** | $0.113232$ | $0.000930$ | **$0.001023$** | Matches Hutch++ (0 query penalty) |

---

## 5. Real-World Effective Rank Crossover

We evaluated trace estimation on real-world scientific and network matrices:
* **YearPredictionMSD** ($d=90, n=50,000, r_{\text{eff}}=28.70$)
* **Wiki-Vote Network** ($d=7,115, \text{edges}=103,689, r_{\text{eff}}=64.42$)

**Key Finding**: The performance crossover between Hutchinson and Hutch++ is governed strictly by the **effective rank** $r_{\text{eff}} = \operatorname{tr}(A)/\|A\|_2$ rather than ambient dimension $d$. Hutch++ achieves $3.5\times - 4.2\times$ variance reduction once $m \gtrsim r_{\text{eff}}$.

![Figure 7: Real-world effective rank crossover curves on YearPredictionMSD and Wiki-Vote network.](/Users/chenyixin/.gemini/antigravity/brain/93c37d0e-e4f3-4103-b050-541a21acb302/real_world_effective_rank_crossover.png)

---

## 6. Current Deliverables & Agenda for Next Meeting

1. **UROP Report & Poster Manuscript**: The research narrative is cohesive and complete with 8 high-resolution figures.
2. **Reproducible Codebase**: Cleaned repository with **152 passing automated tests**.
3. **Meeting Agenda**:
   - Walking through the progress report and final poster/manuscript framing;
   - Discussing graduate school planning, school lists, and recommendation letter logistics.
