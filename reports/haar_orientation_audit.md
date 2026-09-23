# Haar Orientation Robustness Audit: Investigating Coordinate Invariance in Adaptive Hutch++

**Author**: Yit Xiang Zhang (`chen9176@umn.edu`)  
**Faculty Advisor**: Prof. Swati Padmanabhan  
**Date**: September 7, 2026  
**Artifacts**: `results/haar_orientation_audit_trials.csv` (1,860 trials), `results/haar_orientation_audit_summary.csv`

---

**Recovery provenance (2026-09-14):** 1,860 rows reconstructed with the original default protocol and seed 2026 after a smoke test overwrote the untracked CSVs. The original CSV hashes were unavailable, so this is a validated rerun, not a claim of byte-identical recovery. The preserved earlier report is in `results/recovery_20260914/before/`.

## 1. Executive Summary & Core Question

A fundamental question for any randomized matrix algorithm operating with discrete probes (such as Rademacher $\pm 1$ vectors) is:
\[
\boxed{\text{Does the adaptive policy detect true spectral structure, or is it biased by coordinate orientation?}}
\]
Because Gaussian vectors are rotationally invariant while Rademacher vectors live on the discrete Boolean cube $\{-1, 1\}^d$, trace estimation variance depends on the off-diagonal residual energy:
\[
\operatorname{Var}(g^T H g) = 2 \sum_{i \ne j} H_{ij}^2 = 2 \left( \|H\|_F^2 - \sum_{i=1}^d H_{ii}^2 \right).
\]
If an adaptive allocator were sensitive to coordinate rotations, its gating decisions and convergence rates might vary wildly depending on how the eigenbasis $U$ is aligned with the standard coordinate axes.

In this experiment, we freeze six representative spectra $\Lambda$ and evaluate `TwoStageGated` against Standard Hutch++ across:
1. **$N = 30$ independent Haar-random orthogonal orientations** $U_j \sim \text{Haar}(O(d))$ (1,800 trials);
2. **The extreme coordinate-aligned basis** $U = I_d$ (60 trials).

---

## 2. Key Empirical Findings

### Finding 1: Observed Trigger Rates Across the Tested Orientations
Across all 1,860 trials, the Stage 1 pilot screening trigger ($\gamma_{\text{gap}} \ge \tau_{\text{gap}}$ on $b_0=8$ queries) had the following empirical trigger rates:
* **Step Spectrum ($r_\star = 5, \eta = 10^{-3}$)**: Triggered in **100% of trials** on $U = I_d$ and in **100% of trials (300/300)** across all 30 Haar-random orientations, consistently selecting $q_{\text{target}} = 8$.
* **Smooth Spectra (Power-laws $c \in \{0.5, 1.0, 2.0\}$ and Exponential $\alpha=0.08$)**: Triggered in **0% of trials** on $U = I_d$ and **0% of trials (1,200/1,200)** across all Haar orientations, seamlessly falling back to $q_0 = 20$ with zero query penalty.
* **Intermediate Step ($r_\star = 15, \eta = 10^{-2}$)**: Because pilot $b_0 = 8 < r_\star = 15$, the knee is beyond the pilot horizon . The gate triggered in **0% of trials**, selecting the Standard allocation $q_0 = 20$.

**Scientific conclusion, corrected 2026-09-14**: Trigger outcomes were stable on this finite orientation grid. Coordinate-Rademacher range sketches are not rotationally invariant, and these observations do not establish invariance of the gate or universal detection. The recovery rerun reproduces the reported means; no uncertainty interval was preregistered for these comparisons.

---

### Finding 2: Haar Performance Gains (`5.9x Error Reduction on Knees`)
Under general Haar-random rotations (representing realistic scientific matrices where eigenvectors are delocalized):
* On the step spectrum ($r_\star = 5$), `TwoStageGated` reduced MSE from $1.67 \times 10^{-6}$ (Standard Hutch++) to $2.84 \times 10^{-7}$, achieving a **$5.89\times$ error reduction (83.0% reduction, MSE ratio 0.1697)**.
* On smooth power laws, `TwoStageGated` had mean MSE ratios close to Standard Hutch++ on the tested power laws:
  * $c = 0.5$: MSE ratio **$1.007$** (0 query overhead)
  * $c = 1.0$: MSE ratio **$0.949$** (0 query overhead)
  * $c = 2.0$: MSE ratio **$1.070$** (0 query overhead)
  * Exponential: MSE ratio **$0.783$** (0 query overhead)

---

### Finding 3: The Rademacher Diagonal Anomaly on $U = I$
When $U=I_d$, $A$ is diagonal, so classical Rademacher Hutchinson applied directly to $A$ has zero variance. The recorded `diag_concentration` uses the **ideal top-$q_0$ eigenspace residual** $H_{\mathrm{ideal}}$, which is also diagonal. It does not measure the actual randomized Hutch++ residual $R_QAR_Q$, which generally has off-diagonal entries even when $A$ is diagonal.
* Under $U = I_d$, the diagonal concentration $\sum_i H_{ii}^2 / \|H\|_F^2$ is exactly $1.0$.
* Under Haar-random rotations, delocalization spreads energy across the off-diagonals, and diagonal concentration drops to $0.26 - 0.80$.
* This diagnostic illustrates orientation dependence of an ideal residual. Haar averaging is one orientation-averaged benchmark, not a universal coordinate-free performance guarantee. In the coordinate rank-5 case, actual Standard Hutch++ MSE is $7.209998\times10^{-7}$, while gated MSE is $0.0033039424$; neither is zero. The ratio $4582.45$ is a reproduced failure of the gated method on these trials, not a zero-variance denominator artifact.

---

## 3. Quantitative Summary Table ($d=100, m=60, 1{,}860\text{ trials}$)

| Spectrum Case | Orientation Type | Gate Trigger Rate | Mean $q_{\text{target}}$ | Median Rel Err (Hutch++) | Median Rel Err (TwoStageGated) | MSE Ratio (Gated / Hutch++) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Step ($r_\star=5, \eta=10^{-3}$)** | **Haar-Random** | **$100.0\%$** | **$8.0$** | **$0.0182\%$** | **$0.0069\%$** | **$0.1697$ ($5.89\times$ better)** |
| Step ($r_\star=5, \eta=10^{-3}$) | Coordinate ($U=I$) | $100.0\%$ | $8.0$ | $0.0101\%$ | $0.0071\%$ | $4.58 \times 10^3$ (observed gated failure) |
| **PowerLaw ($c=0.5$ - Flat)** | **Haar-Random** | **$0.0\%$** | **$20.0$** | **$0.930\%$** | **$0.941\%$** | **$1.007$ (match)** |
| PowerLaw ($c=0.5$ - Flat) | Coordinate ($U=I$) | $0.0\%$ | $20.0$ | $0.960\%$ | $1.218\%$ | $2.254$ |
| **PowerLaw ($c=1.0$ - Moderate)** | **Haar-Random** | **$0.0\%$** | **$20.0$** | **$0.724\%$** | **$0.730\%$** | **$0.949$ (match)** |
| PowerLaw ($c=1.0$ - Moderate) | Coordinate ($U=I$) | $0.0\%$ | $20.0$ | $0.180\%$ | $0.942\%$ | $4.800$ |
| **PowerLaw ($c=2.0$ - Steep)** | **Haar-Random** | **$0.0\%$** | **$20.0$** | **$0.095\%$** | **$0.103\%$** | **$1.070$ (match)** |
| PowerLaw ($c=2.0$ - Steep) | Coordinate ($U=I$) | $0.0\%$ | $20.0$ | $0.038\%$ | $0.105\%$ | $2.476$ |
| **Exponential ($\alpha=0.08$)** | **Haar-Random** | **$0.0\%$** | **$20.0$** | **$1.358\%$** | **$1.167\%$** | **$0.783$ (better)** |
| Exponential ($\alpha=0.08$) | Coordinate ($U=I$) | $0.0\%$ | $20.0$ | $0.901\%$ | $1.236\%$ | $1.518$ |
