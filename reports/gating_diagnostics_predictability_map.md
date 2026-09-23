# Pilot Feature Gating Diagnostics & Predictability Map: When, Why, and How Hutch++ Should Adapt

**Author**: Yit Xiang Zhang (`chen9176@umn.edu`)  
**Faculty Advisor**: Prof. Swati Padmanabhan  
**Date**: September 8, 2026  
**Artifacts**: `results/gating_diagnostics_summary.csv` (23 configurations), `results/gating_diagnostics_pilot_features.csv`, `results/gating_diagnostics_q_curves.csv`

---

**Audit status (2026-09-14):** Exploratory development data; threshold selection and evaluation use the same spectral grid. Reported MSE ratios have no accompanying uncertainty assessment. No statement below establishes universal safety or a necessary-and-sufficient adaptation rule.

## 1. Executive Summary & Core Research Question

Standard Hutch++ hard-codes an invariant query allocation:
\[
q_{\text{standard}} = \frac{m}{3}, \quad \ell_{\text{standard}} = \frac{m}{3}.
\]
While this $1/3 : 2/3$ split is minimax rate-optimal for worst-case matrices, it is **sub-optimal for structured linear algebra**:
\[
\boxed{\text{“When, why, and how should Hutch++ adapt its query allocation?”}}
\]

In this investigation, we construct a comprehensive **Predictability Map** across 23 configurationsl families (step heights, step locations, power laws, exponentials, and neural Hessians). We demonstrate that:
1. **The Allocation Duality**: The optimal subspace allocation $q^*(A, m)$ bifurcates into two distinct physical regimes:
   - **Subspace Isolation Regime ($q^* \ll m/3$)**: Occurs when a low-rank signal space is separated by a sharp cliff from a flat noise floor. Truncating at $q \approx r$ frees budget for residual probes, reducing MSE by up to **$7.4\times$**.
   - **Continuous Curvature Regime ($q^* \ge m/3$)**: Occurs when eigenvalues decay smoothly and continuously. Extending the subspace dimension drives the residual Frobenius norm $\|A_{\text{res}}\|_F \to 0$, suppressing variance faster than residual averaging.
2. **The Isolation Contrast Discovery**: A single Ritz gap $\log(\theta_1 / \theta_2)$ cannot distinguish an isolated step cliff from a steep continuous slope. By measuring the **Gap Isolation Contrast** $\mathcal{C} = \Delta_{\max} / \operatorname{median}_{k \ne \max} \Delta_k$, the development examples suggest that a pilot can distinguish step cliffs ($\mathcal{C} \ge 30 - 60{,}000$) from continuous decay ($\mathcal{C} \le 3.5$) in the recorded development sample.

---

## 2. Mathematical Mechanisms of Query Allocation

### Mechanism 1: Why Isolated Knees Demand Small $q$ and Large $\ell$
Consider a matrix with $r$ signal eigenvalues $\lambda_1 = \dots = \lambda_r = 1$ and $d - r$ noise eigenvalues $\lambda_{r+1} = \dots = \lambda_d = \eta \ll 1$.
In the full-rank case only, $\ell=m-2q$. For the actual residual $H=RAR$, Gaussian conditional variance is $2\|H\|_F^2/\ell$, while Rademacher conditional variance is $2\sum_{i\ne j}H_{ij}^2/\ell$. The following is an ideal-capture Gaussian surrogate:
\[
\mathcal R_{\mathrm{ideal},G}(q)=\frac{2T(q)}{m-2q}.
\]
- Under the additional ideal-subspace assumption, $q=r$ captures all $r$ signal directions. A randomized basis need not satisfy this; the frozen bridge demonstrated the resulting leakage. The residual matrix contains only noise:
  \[
  \|(I - Q Q^T) A\|_F^2 \approx (d - r) \eta^2.
  \]
  Residual probes $\ell = m - 2r$ average out this isotropic noise floor.
- If an algorithm spends further queries to increase $q > r$ (such as Standard Hutch++ setting $q = m/3 \gg r$), each extra dimension captures noise-level eigenvalues $\eta$. The residual Frobenius norm shrinks negligibly (from $(d-r)\eta^2$ to $(d-q)\eta^2$), but **residual probes $\ell$ are starved** by $2$ probes per extra dimension!
- **Heuristic motivated by this surrogate**: Truncate at $q^* \approx r + p_{\text{oversample}}$ and allocate all surplus queries to $\ell$.

### Mechanism 2: Why Continuous Curvature Demands Large $q$
When eigenvalues decay continuously without a plateau (e.g. $\lambda_i = i^{-\alpha}$ with $\alpha \ge 2$):
\[
\|(I - Q Q^T) A\|_F^2 \approx \sum_{j > q} j^{-2\alpha} \approx \frac{q^{1 - 2\alpha}}{2\alpha - 1}.
\]
For $\alpha = 2$, increasing $q$ from $8$ to $24$ suppresses the residual Frobenius norm by a factor of $(24/8)^3 = 27\times$!
Because variance scales with $\|A_{\text{res}}\|_F^2$, shrinking the residual norm by $27\times$ is far more powerful than increasing residual probes $\ell$ by $2\times$.
- **Surrogate interpretation**: Steep decay can favor larger sketches. Actual superiority depends on realized capture and the residual denominator; this is not a universal strict ordering.

### Mechanism 3: The Gap Isolation Contrast Filter
How can an online pilot of $b_0$ queries distinguish between an isolated knee and steep continuous decay?
Let $\Delta_j = \log(\theta_j / \theta_{j+1})$ be adjacent log Ritz gaps.
1. **Isolated Knee Spectrum**:
   $\Delta_{\text{knee}} \ge 1.5$, while $\Delta_k \approx 0$ for all $k \ne \text{knee}$.
   \[
   \mathcal{C} = \frac{\max_j \Delta_j}{\operatorname{median}_{k \ne j} \Delta_k + \epsilon} \ge 30 - 69{,}000.
   \]
2. **Continuous Power Law / Exponential**:
   For an ideal power law the gaps are $\alpha\log((j+1)/j)$; for an ideal exponential they are constant. Pilot Ritz gaps may differ. All gaps are non-zero:
   \[
   \mathcal{C} \le 3.5.
   \]
By enforcing the dual gating rule:
\[
\boxed{\text{Trigger Gated Policy} \iff \Delta_{\max} \ge \tau_{\text{gap}} \quad \text{AND} \quad \mathcal{C} \ge \tau_{\text{contrast}}}
\]
with $\tau_{\text{gap}} = 1.2$ and $\tau_{\text{contrast}} = 5.0$, the recorded development examples show these trigger patterns for visible sharp knees and smooth controls. The threshold was evaluated on the same grid, and the patterns are not a guarantee for new matrices.

---

## 3. Empirical Predictability Map & Decision Boundary ($d=100, m=60$)

The following table summarizes the 23 evaluated configurations, mapping pilot features to empirical fixed-$q$ grid minima and achieved allocations. The table's legacy “Oracle” labels mean Monte Carlo grid minima, not exact risk oracles; independently estimated gains may exceed these noisy reference gains.

| Spectral Family & Case | Mean Pilot Gap $\Delta_{\max}$ | Mean Contrast $\mathcal{C}$ | Oracle $q^*$ | Oracle Gain | Gated Trigger | Mean $q_{\text{gated}}$ | Achieved Gain | Regime Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Step ($r=4, \text{drop}=1.5$)** | $0.06$ | $12.3$ | $2$ | $20.1\times$ | $0.0\%$ | $20.0$ | $1.74\times$ | Benign Baseline (Protected) |
| **Step ($r=4, \text{drop}=2.0$)** | $0.15$ | $13.5$ | $2$ | $11.3\times$ | $0.0\%$ | $20.0$ | $0.48\times$ | Benign Baseline (Protected) |
| **Step ($r=4, \text{drop}=5.0$)** | $0.80$ | $30.1$ | $10$ | $2.61\times$ | $0.0\%$ | $20.0$ | $1.35\times$ | Benign Baseline (Protected) |
| **Step ($r=4, \text{drop}=10.0$)** | **$1.85$** | **$206.6$** | **$12$** | **$2.54\times$** | **$100.0\%$** | **$8.0$** | **$1.79\times$** | **Obvious Adapt (Captured)** |
| **Step ($r=4, \text{drop}=50.0$)** | **$3.87$** | **$6{,}221$** | **$10$** | **$4.43\times$** | **$100.0\%$** | **$8.0$** | **$3.25\times$** | **Obvious Adapt (Captured)** |
| **Step ($r=4, \text{drop}=100.0$)**| **$4.60$** | **$22{,}764$**| **$8$** | **$4.80\times$** | **$100.0\%$** | **$8.0$** | **$2.60\times$** | **Obvious Adapt (Captured)** |
| **Step ($r=4, \text{drop}=500.0$)**| **$6.21$** | **$62{,}143$**| **$6$** | **$8.65\times$** | **$100.0\%$** | **$8.0$** | **$3.97\times$** | **Obvious Adapt (Captured)** |
| **Step ($r=4, \text{drop}=1000$)** | **$6.91$** | **$69{,}077$**| **$6$** | **$5.90\times$** | **$100.0\%$** | **$8.0$** | **$5.69\times$** | **Obvious Adapt (Captured)** |
| **Step ($r=2, \text{drop}=100$)**  | **$4.60$** | **$46{,}026$**| **$6$** | **$7.40\times$** | **$100.0\%$** | **$8.0$** | **$4.44\times$** | **Obvious Adapt (Captured)** |
| **Step ($r=4, \text{drop}=100$)**  | **$4.60$** | **$24{,}087$**| **$8$** | **$3.83\times$** | **$100.0\%$** | **$8.0$** | **$7.40\times$** | **Obvious Adapt (Captured)** |
| **Step ($r=6, \text{drop}=100$)**  | **$4.57$** | **$7{,}273$** | **$10$** | **$3.60\times$** | **$100.0\%$** | **$8.0$** | **$1.20\times$** | **Obvious Adapt (Captured)** |
| PowerLaw ($\alpha=0.0$ - Flat)      | $0.00$ | $0.0$ | $2$ | $36.1\times$ | $0.0\%$ | $20.0$ | $0.95\times$ | Benign Baseline (Protected) |
| PowerLaw ($\alpha=0.5$ - Slow)      | $0.49$ | $3.4$ | $8$ | $1.51\times$ | $0.0\%$ | $20.0$ | $0.55\times$ | Benign Baseline (Protected) |
| PowerLaw ($\alpha=1.0$ - Harmonic)  | $0.76$ | $2.4$ | $18$ | $1.31\times$ | $0.0\%$ | $20.0$ | $1.79\times$ | Benign Baseline (Protected) |
| PowerLaw ($\alpha=1.5$ - Moderate)  | $1.06$ | $2.4$ | $20$ | $1.00\times$ | $0.0\%$ | $20.0$ | $1.25\times$ | Benign Baseline (Protected) |
| PowerLaw ($\alpha=2.0$ - Steep)     | $1.39$ | $2.5$ | $22$ | $1.59\times$ | $0.0\%$ | $20.0$ | $0.84\times$ | Benign Baseline (Protected) |
| PowerLaw ($\alpha=2.5$ - Rapid)     | $1.73$ | $2.7$ | $26$ | $1.34\times$ | $0.0\%$ | $20.0$ | $1.23\times$ | Benign Baseline (Protected) |
| PowerLaw ($\alpha=3.0$ - Extreme)   | $2.08$ | $2.9$ | $24$ | $2.21\times$ | $0.0\%$ | $20.0$ | $1.58\times$ | Benign Baseline (Protected) |
| Exponential ($\beta=0.03$)          | $0.15$ | $2.6$ | $4$ | $2.36\times$ | $0.0\%$ | $20.0$ | $1.04\times$ | Benign Baseline (Protected) |
| Exponential ($\beta=0.08$)          | $0.25$ | $2.2$ | $24$ | $1.04\times$ | $0.0\%$ | $20.0$ | $0.71\times$ | Benign Baseline (Protected) |
| Exponential ($\beta=0.15$)          | $0.41$ | $2.5$ | $28$ | $2.15\times$ | $0.0\%$ | $20.0$ | $1.32\times$ | Benign Baseline (Protected) |
| Exponential ($\beta=0.30$)          | $0.63$ | $1.8$ | $28$ | $19.3\times$ | $0.0\%$ | $20.0$ | $1.09\times$ | Benign Baseline (Protected) |

---

## 4. Exploratory Interpretation (Corrected 2026-09-14)

This development-grid diagnostic suggests a hypothesis requiring held-out evaluation:

1. **When should Hutch++ adapt?**
   The proposed heuristic adapts when the pilot detects an **isolated eigenspace** ($\Delta_{\max} \ge 1.2$ and contrast $\mathcal{C} \ge 5.0$).
2. **Why does it adapt?**
   Because spending queries on isotropic noise eigenvalues wastes budget that is desperately needed for residual probes $\ell$. Truncating at $q \approx r$ and expanding $\ell$ yields up to **$7.4\times$ variance reduction**.
3. **What safeguards does the heuristic attempt?**
   By coupling:
   - **Zero-Cost Pilot Screening**: The first $b_0$ queries are reused directly in the final estimate, imposing zero query overhead.
   - **Isolation Contrast Filtering**: The threshold $\mathcal{C}\ge5.0$ separated the reported development examples. This is not a guarantee against false triggers on other spectra, seeds, or orientations. Selecting $q_0$ preserves the allocation budget, not a theorem of identical pathwise risk.
   - **Tail-Risk Insurance**: Extra sketch columns can reduce poor capture associated with zero oversampling. One column is not a universal guarantee, and conditioning can matter even without numerical rank deficiency.
