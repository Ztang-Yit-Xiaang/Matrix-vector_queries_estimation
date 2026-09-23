# Phase 2C Proof Note: Structural Priors for Direct Paired Rademacher Risk Difference

**Author**: Yit Xiang Zhang (`chen9176@umn.edu`)  
**Faculty Advisor**: Prof. Swati Padmanabhan  
**Date**: September 7, 2026  
**Classification**: `PROVED UNDER PRIOR AND NESTING` (Theorem 16), `PROVED` (corrected Theorem 17), and `PROVED ROUTE-SPECIFIC BOUNDARY` (corrected Theorem 18).

**Recovery audit, 2026-09-14:** Theorem 16 is conditional on the supplied prior and nested bases; a finite radius need not permit acceptance. Theorem 17's fourth-moment expansion below is corrected by direct combinatorial counting. Theorem 18 is a route-specific threshold, not an impossibility theorem for empirical certification. Earlier versions are retained in `results/recovery_20260914/before/`.

---

## 1. Context and Problem Statement

In **Phase 2B**, we proved that disjoint common-probe pairs yield conditionally i.i.d. signed observations:
\[
D_j = \frac{(X_{a,2j-1} - X_{a,2j})^2}{2\ell_a} - \frac{(X_{0,2j-1} - X_{0,2j})^2}{2\ell_0}, \qquad j = 1, \dots, n = \lfloor s/2 \rfloor
\]
with exact conditional expectation:
\[
\mathbb{E}[D_j \mid \mathcal{G}] = \Delta_R = \frac{\sigma_a^2}{\ell_a} - \frac{\sigma_0^2}{\ell_0}.
\]
Under the strict data-only contract, $v_D = \operatorname{Var}(D_j \mid \mathcal{G})$ had to be bounded using only the worst-case degree-4 Boolean hypercontractive factor $\mathbb{E}[P_j^4] \le 6561 v_D^2$. This caused the relative scale estimation radius $\varepsilon_v$ to exceed 90 on all frozen probe budgets $s \in \{4, 8, 16, 32\}$, yielding the verdict `PROVED BUT BUDGET-VACUOUS`.

**Phase 2C Objective**: Study finite, computable one-sided confidence radii for $\Delta_R$ under explicit structural priors. Finiteness alone does not establish useful acceptance or resolve the query cost of certification.

---

## 2. Setting and Notation

Condition on the pre-certification $\sigma$-algebra $\mathcal{G}$, which contains the symmetric matrix $A \in \mathbb{R}^{d \times d}$, orthonormal bases $Q_a \in \mathbb{R}^{d \times r_a}, Q_0 \in \mathbb{R}^{d \times r_0}$, residual projectors $R_x = I - Q_x Q_x^T$, and positive residual probe denominators $\ell_a, \ell_0 > 0$.

Define the residual matrices and their off-diagonal components:
\[
H_x = R_x A R_x, \qquad C_x = H_x - \operatorname{diag}(H_x), \qquad x \in \{a, 0\}.
\]
Let $g \in \{\pm 1\}^d$ be an i.i.d. coordinate-Rademacher probe. Then:
\[
X_x = g^T H_x g = \operatorname{tr}(H_x) + g^T C_x g.
\]
The centered variable $Z_x = X_x - \mathbb{E}[X_x] = g^T C_x g$ is a degree-2 multilinear Rademacher chaos with exact variance:
\[
\sigma_x^2 = \operatorname{Var}(X_x \mid \mathcal{G}) = 2 \|C_x\|_F^2 = 2 \sum_{i \ne j} (H_x)_{ij}^2 \le 2 \|H_x\|_F^2.
\]

---

## 3. Theorem 16: Norm-Envelope Structural Certificate (`PROVED`)

### Statement
Assume the **Norm-Envelope Prior**: the baseline residual Frobenius norm is bounded by a known constant $M_0 < \infty$:
\[
\|H_0\|_F \le M_0.
\]
Assume candidate $Q_a$ is nested with or extends $Q_0$ (i.e., $R_a \preceq R_0$ on the PSD cone). Then:
1. $\sigma_a^2 \le 2 M_0^2$ and $\sigma_0^2 \le 2 M_0^2$.
2. Under these assumptions the conditional variance of $D_j$ satisfies:
   \[
   \boxed{\operatorname{Var}(D_j \mid \mathcal{G}) \le \bar{v}_D(M_0) \triangleq 164 M_0^4 \left(\frac{1}{\ell_a} + \frac{1}{\ell_0}\right)^2.}
   \]
3. For any target failure probability $\delta \in (0, 1)$ and sample size $n = \lfloor s/2 \rfloor \ge 1$, the one-sided confidence certificate:
   \[
   \boxed{U_n^{\text{norm}}(\delta) = \overline{D}_n + C_n^{\text{norm}}(\delta; M_0), \qquad C_n^{\text{norm}}(\delta; M_0) \triangleq \sqrt{\frac{\bar{v}_D(M_0)}{n} \cdot \frac{1-\delta}{\delta}}}
   \]
   satisfies:
   \[
   \boxed{\Pr\left(\Delta_R \le U_n^{\text{norm}}(\delta) \mid \mathcal{G}\right) \ge 1 - \delta.}
   \]
   In particular, the radius is finite for all $n\ge1$. Whether the upper confidence bound is nonpositive, and hence useful for acceptance, is a separate question. The API relies on the caller to justify the prior and nesting; observations alone cannot verify them.

### Proof
Nesting implies $R_aR_0=R_a$ and $H_a=R_aH_0R_a$, so $\|H_a\|_F\le\|H_0\|_F\le M_0$. This contraction only requires symmetry of $A$, not positivity. If $M_0=0$, both residual matrices vanish and the asserted event is deterministic; handle this before dividing by variance.

For independent probes $g_1, g_2$, let $Y_x = \frac{(X_{x,1} - X_{x,2})^2}{2} = \frac{(Z_{x,1} - Z_{x,2})^2}{2}$.  
Then $\mathbb{E}[Y_x \mid \mathcal{G}] = \sigma_x^2$.  
Computing the second moment:
\[
Y_x^2 = \frac{(Z_{x,1} - Z_{x,2})^4}{4} = \frac{Z_{x,1}^4 - 4 Z_{x,1}^3 Z_{x,2} + 6 Z_{x,1}^2 Z_{x,2}^2 - 4 Z_{x,1} Z_{x,2}^3 + Z_{x,2}^4}{4}.
\]
Because $Z_{x,1}$ and $Z_{x,2}$ are independent and centered:
\[
\mathbb{E}[Y_x^2 \mid \mathcal{G}] = \frac{2 \mathbb{E}[Z_x^4 \mid \mathcal{G}] + 6 (\mathbb{E}[Z_x^2 \mid \mathcal{G}])^2}{4} = \frac{1}{2} \mathbb{E}[Z_x^4 \mid \mathcal{G}] + \frac{3}{2} \sigma_x^4.
\]
Subtracting $(\mathbb{E}[Y_x])^2 = \sigma_x^4$:
\[
\operatorname{Var}(Y_x \mid \mathcal{G}) = \frac{1}{2} \mathbb{E}[Z_x^4 \mid \mathcal{G}] + \frac{1}{2} \sigma_x^4.
\]
By Bonami-Beckner hypercontractivity for degree-2 Rademacher polynomials, $\mathbb{E}[Z_x^4 \mid \mathcal{G}] \le 81 \sigma_x^4$. Hence:
\[
\operatorname{Var}(Y_x \mid \mathcal{G}) \le \frac{81 + 1}{2} \sigma_x^4 = 41 \sigma_x^4.
\]
Since $D_j = \frac{Y_a}{\ell_a} - \frac{Y_0}{\ell_0}$, applying the triangle inequality on standard deviations:
\[
\sqrt{\operatorname{Var}(D_j \mid \mathcal{G})} \le \frac{\sqrt{\operatorname{Var}(Y_a)}}{\ell_a} + \frac{\sqrt{\operatorname{Var}(Y_0)}}{\ell_0} \le \sqrt{41}\left(\frac{\sigma_a^2}{\ell_a} + \frac{\sigma_0^2}{\ell_0}\right).
\]
Squaring both sides and using $\sigma_x^2 \le 2 \|H_x\|_F^2 \le 2 M_0^2$:
\[
\operatorname{Var}(D_j \mid \mathcal{G}) \le 41 \cdot (2 M_0^2)^2 \left(\frac{1}{\ell_a} + \frac{1}{\ell_0}\right)^2 = 164 M_0^4 \left(\frac{1}{\ell_a} + \frac{1}{\ell_0}\right)^2 \equiv \bar{v}_D(M_0).
\]
Now consider the sample mean $\overline{D}_n = \frac{1}{n} \sum_{j=1}^n D_j$. Since the $D_j$'s are conditionally i.i.d. with mean $\Delta_R$ and variance $v_D \le \bar{v}_D(M_0)$:
\[
\operatorname{Var}(\overline{D}_n \mid \mathcal{G}) = \frac{v_D}{n} \le \frac{\bar{v}_D(M_0)}{n}.
\]
Applying Cantelli's one-sided inequality: for any $t > 0$,
\[
\Pr\left(\Delta_R - \overline{D}_n > t \mid \mathcal{G}\right) \le \frac{\operatorname{Var}(\overline{D}_n)}{\operatorname{Var}(\overline{D}_n) + t^2} \le \frac{\bar{v}_D(M_0)/n}{\bar{v}_D(M_0)/n + t^2}.
\]
Setting this upper bound to $\delta$:
\[
\frac{\bar{v}_D(M_0)/n}{\bar{v}_D(M_0)/n + t^2} = \delta \iff t^2 = \frac{\bar{v}_D(M_0)}{n} \cdot \frac{1-\delta}{\delta} \iff t = \sqrt{\frac{\bar{v}_D(M_0)}{n} \cdot \frac{1-\delta}{\delta}} \equiv C_n^{\text{norm}}(\delta; M_0).
\]
Thus, $\Pr(\Delta_R \le \overline{D}_n + C_n^{\text{norm}}(\delta; M_0)) \ge 1 - \delta$. $\blacksquare$

---

## 4. Theorem 17: Structural Chaos Kurtosis Refinement (`PROVED`)

### Statement
Let $C \in \mathbb{R}^{d \times d}$ be nonzero, symmetric, and zero diagonal. Define $\kappa = \|C\|_2/\|C\|_F \in (0,1]$. For $C=0$, the fourth moment is zero and no ratio is defined.
The fourth moment of the Rademacher chaos $Z = g^T C g$ satisfies:
\[
\boxed{\mathbb{E}[Z^4] \le (3 + 12 \kappa^2) \sigma^4 \le 15 \sigma^4,}
\]
where $\sigma^2 = 2 \|C\|_F^2$.

In particular:
1. As effective rank $r_{\text{eff}}(C) = 1/\kappa^2 \to \infty$ ($\kappa \to 0$), the kurtosis ratio $\frac{\mathbb{E}[Z^4]}{\sigma^4} \to 3$, matching standard Gaussian chaos.
2. The universal envelope is at most $15$. The endpoint $\kappa=1$ is a conservative envelope endpoint, not a realizable nonzero rank-one symmetric zero-diagonal matrix.

### Proof
Expand $Z = \sum_{i \ne j} C_{ij} g_i g_j$. Because $C$ is symmetric with zero diagonal:
\[
Z^2 = \sum_{i \ne j} \sum_{k \ne l} C_{ij} C_{kl} g_i g_j g_k g_l.
\]
Write $Z=\sum_{i<j}a_{ij}g_ig_j$ with $a_{ij}=2C_{ij}$. In the fourth-power expansion a product survives expectation exactly when every vertex has even degree. The possible edge multisets are one edge repeated four times, two distinct edges each repeated twice, or a simple four-cycle. Their permutation counts are respectively $1$, $6$, and $24$. Thus, with each undirected four-cycle counted once,
\[
\mathbb E Z^4=\sum_e a_e^4+6\sum_{e<f}a_e^2a_f^2+24\sum_{\text{four-cycles}}\prod_{e\text{ in cycle}}a_e.
\]
Expanding $\operatorname{tr}(C^4)$ into closed four-step walks (a single edge, two-edge wedges, and four-cycles) gives the exact identity
\[
\mathbb{E}[Z^4] = 3 \sigma^4 + 48 \operatorname{tr}(C^4) - 96 \sum_i (C^2)_{ii}^2 + 32 \sum_{i,j} C_{ij}^4.
\]
Put $F^2=\|C\|_F^2$, $R=\sum_i(\sum_j C_{ij}^2)^2$, and $J=\sum_{ij}C_{ij}^4$. Since $J\le R$, the correction $-96R+32J$ is nonpositive. Also
\[
\operatorname{tr}(C^4) \le \|C\|_2^2 \|C\|_F^2 = \kappa^2 \|C\|_F^4 = \frac{1}{4} \kappa^2 \sigma^4.
\]
Consequently,
\[
\mathbb E Z^4\le 3\sigma^4+48\operatorname{tr}(C^4)\le(3+12\kappa^2)\sigma^4.
\]
For the asserted limit, $R\le\|C\|_2^2F^2=\kappa^2F^4$ also yields $\mathbb E Z^4/\sigma^4\ge3-24\kappa^2$. Squeezing proves convergence to 3 as $\kappa\to0$. This theorem concerns degree-two $Z$; it does not supply the kurtosis of the degree-four signed-pair variable $D_j-\Delta_R$. $\blacksquare$

---

## 5. Theorem 18: The Scale Estimation Feasibility Boundary (`PROVED ANALYTIC LIMIT`)

### Statement
Suppose an observable scale confidence bound for $v_D$ is constructed from sample variance $S_D^2$ via Chebyshev's inequality under a kurtosis bound $K_D \triangleq \frac{\mathbb{E}[P_j^4]}{(\mathbb{E}[P_j^2])^2}$:
\[
\frac{\operatorname{Var}(S_D^2 \mid \mathcal{G})}{v_D^2} \le A_n \triangleq \frac{K_D - (n-3)/(n-1)}{n}.
\]
Let $\delta_{\text{scale}} \in (0, 1)$ be the allocated scale failure probability.  
Then the relative confidence radius $\varepsilon_v = \sqrt{A_n / \delta_{\text{scale}}}$ is strictly non-vacuous ($\varepsilon_v < 1$) if and only if the sample size satisfies:
\[
\boxed{\delta_{\text{scale}}n(n-1)>(K_D-1)(n-1)+2,\qquad n\ge2.}
\]

This follows by multiplying $A_n<\delta_{\text{scale}}$ by the positive quantity $n(n-1)$. Equivalently $n$ exceeds the larger root of
\[
\delta_{\text{scale}}n^2-(\delta_{\text{scale}}+K_D-1)n+K_D-3=0.
\]
The implementation finds the smallest integer satisfying the strict polynomial inequality using exact decimal rational arithmetic. Since $A_n=(K_D-1)/n+2/[n(n-1)]$ decreases strictly for $K_D\ge1$, binary search returns the unique minimum. The earlier expression $(K_D-1)/\delta_{\text{scale}}+1$ was not the exact boundary.

### Consequence for Small Probe Budgets ($s \le 32$)
1. **Under Phase 2B's worst-case bound ($K_D = 6561$)**:
   For $\delta_{\text{joint}} = 0.10 \implies \delta_{\text{scale}} = 0.05$:
   \[
   n_{\min}(6561, 0.05)=131{,}201.
   \]
   Since $s \le 32 \implies n \le 16$, the data-only route misses feasibility by four orders of magnitude ($16 \ll 131{,}201$).
2. **Illustrative assumed kurtosis bound $K_D=3$**:
   This is an illustrative input, not a consequence of Theorem 17 for signed pairs:
   \[
   A_{41}=1/20=0.05,\qquad n_{\min}(3,0.05)=42,\qquad s_{\min}=84.
   \]
   At $n = 16$ ($s = 32$), $A_{16} = (3 - 13/15)/16 \approx 0.133$, yielding $\varepsilon_v = \sqrt{0.133 / 0.05} \approx 1.63 > 1$.
3. **Route-specific conclusion**:
   For the declared bound $K_D\ge3$ at $\delta_{\text{scale}}=0.05$, this sample-variance/Chebyshev radius exceeds one for $n\le16$. Nondegenerate random variables need not have kurtosis at least three (a centered Rademacher variable has kurtosis one). Neither this calculation nor Theorem 17 proves that all empirical certification is impossible or that a norm prior is necessary. A norm envelope is one sufficient additional assumption; useful acceptance and the cost of obtaining that envelope remain separate questions.

---

## 6. Summary of Classification

| Theorem | Description | Mathematical Status | Operating Regime |
| :--- | :--- | :--- | :--- |
| **Theorem 16** | Norm-Envelope Cantelli Certificate | `PROVED UNDER PRIOR AND NESTING` | Finite radius for $s\ge2$; useful acceptance not established |
| **Theorem 17** | Chaos Kurtosis vs. Effective Rank $\kappa$ | `PROVED` | Refines 81 down to $3 + 12\kappa^2$ |
| **Theorem 18** | Chebyshev Feasibility Threshold | `PROVED ROUTE-SPECIFIC BOUNDARY` | Exact strict inequality above; no universal impossibility claim |
