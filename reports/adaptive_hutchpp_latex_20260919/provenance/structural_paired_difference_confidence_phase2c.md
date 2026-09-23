# Phase 2C: Structural Priors for Paired Risk-Difference Confidence

**Author:** Yit Xiang Zhang  
**Faculty advisor:** Prof. Swati Padmanabhan  
**Original date:** September 7, 2026  
**Recovery audit:** September 14, 2026

The structural-prior route supplies a finite one-sided radius under a valid external norm envelope and nested bases. Its ability to accept useful candidates at a small budget has not been established. The earlier report overstated this as resolution of the certification tax and a proof that data-only certification is impossible. That version is preserved in `results/recovery_20260914/before/`.

## 1. Exact target and assumptions

Condition on the pre-certification information \(\mathcal G\). With fresh common Rademacher probes, fixed positive denominators, and residuals \(H_x=R_xAR_x\), define
\[
D_j=\frac{(X_{a,2j-1}-X_{a,2j})^2}{2\ell_a}
-\frac{(X_{0,2j-1}-X_{0,2j})^2}{2\ell_0}.
\]
The disjoint pairs are conditionally iid and
\[
\mathbb E[D_j\mid\mathcal G]=\Delta_R
=\frac{\sigma_a^2}{\ell_a}-\frac{\sigma_0^2}{\ell_0}.
\]
The denominators determine the comparison: net improvement requires the paid candidate denominator and the original baseline denominator. An acceptance statement does not make an entire policy safe after abstention.

## 2. Theorem 16 — proved under the supplied prior and nesting

Assume \(\|H_0\|_F\le M_0\) is known before certification and \(R_aR_0=R_a\). Then \(H_a=R_aH_0R_a\) and \(\|H_a\|_F\le M_0\). The degree-two fourth-moment envelope 81 gives
\[
\operatorname{Var}(D_j\mid\mathcal G)
\le164M_0^4(1/\ell_a+1/\ell_0)^2=\bar v_D.
\]
For \(n=s/2\) disjoint pairs, Cantelli yields
\[
\Pr\!\left(\Delta_R\le\overline D_n+
\sqrt{\frac{\bar v_D(1-\delta)}{n\delta}}\;\middle|\;\mathcal G\right)\ge1-\delta.
\]
A finite radius is not a guarantee that the upper bound is nonpositive. No experiment here validates a practical acceptance rate or supplies a computable \(M_0\). The API cannot infer nesting or verify the prior from its observation arrays. Its legacy `nonvacuous` field means only finite radius; `certified_safe` records the observed acceptance inequality.

At \(M_0=0\), nested residual matrices vanish and the conclusion is deterministic. The preserved factor 164 does not rely on the sharpened Theorem 17.

## 3. Theorem 17 — corrected proof, same upper bound

For nonzero symmetric zero-diagonal \(C\), let \(Z=g^\top Cg\), \(\sigma^2=2\|C\|_F^2\), and \(\kappa=\|C\|_2/\|C\|_F\). Direct even-degree edge counting gives
\[
\mathbb E Z^4
=3\sigma^4+48\operatorname{tr}(C^4)
-96\sum_i(C^2)_{ii}^2+32\sum_{ij}C_{ij}^4.
\]
The previous displayed identity had incorrect coefficients. The corrected identity is checked by exhaustive Rademacher enumeration and derived in the proof note. Since \(\sum_{ij}C_{ij}^4\le\sum_i(C^2)_{ii}^2\),
\[
\mathbb E Z^4\le3\sigma^4+48\|C\|_2^2\|C\|_F^2
=(3+12\kappa^2)\sigma^4\le15\sigma^4.
\]
A lower bound \(3-24\kappa^2\) on normalized kurtosis establishes convergence to 3 as \(\kappa\to0\). The endpoint \(\kappa=1\) is only an envelope endpoint: a nonzero rank-one symmetric matrix cannot have zero diagonal.

This controls degree-two \(Z\). It does **not** establish a \(3+12\kappa^2\) kurtosis bound for the centered degree-four signed-pair variable \(D_j-\Delta_R\). The structural-kurtosis CSV is explicitly an illustrative calculation with an assumed \(K\), not a certified improvement of the Phase 2B signed-pair theorem.

## 4. Theorem 18 — exact boundary for this Chebyshev route

For an unbiased sample variance based on \(n\ge2\) observations and an assumed kurtosis bound \(K\ge1\),
\[
A_n=\frac{K-(n-3)/(n-1)}n
=\frac{K-1}{n}+\frac2{n(n-1)}.
\]
The radius \(\sqrt{A_n/\delta_{\rm scale}}\) is strictly below one exactly when
\[
\delta_{\rm scale}n(n-1)>(K-1)(n-1)+2.
\]
The code finds the smallest feasible integer using exact decimal arithmetic. At \(K=3,\delta_{\rm scale}=0.05\), \(n=41\) is equality and \(n_{\min}=42\), requiring \(s=84\) probes for disjoint pairs.

| Assumed kurtosis bound \(K\) | \(\delta_{\rm scale}\) | Minimum \(n\) | Minimum \(s=2n\) |
|---|---:|---:|---:|
| 3 | 0.05 | 42 | 84 |
| 3.12 | 0.05 | 44 | 88 |
| 6 | 0.05 | 101 | 202 |
| 15 | 0.05 | 281 | 562 |
| 6561 | 0.05 | 131201 | 262402 |

These results establish vacuity of this bound on the declared \(s\le32\) grid. Nondegenerate variables need not have kurtosis at least three. No universal impossibility theorem or necessity of a norm prior follows.

## 5. Corrected numerical examples

The earlier hand-entered radius table did not agree with its stated denominators. Direct evaluation at \(\delta=0.05,\ell_a=70,\ell_0=100\) gives:

| \(M_0\) | \(s=4\) | \(s=8\) | \(s=16\) | \(s=32\) |
|---:|---:|---:|---:|---:|
| 0.01 | 0.0000958594 | 0.0000677828 | 0.0000479297 | 0.0000338914 |
| 0.05 | 0.00239648 | 0.00169457 | 0.00119824 | 0.000847285 |
| 0.10 | 0.00958594 | 0.00677828 | 0.00479297 | 0.00338914 |
| 1.00 | 0.958594 | 0.677828 | 0.479297 | 0.338914 |

These are hypothetical scales, not evidence that \(M_0\le0.05\) is typical. The full 252-row radius grid uses its original three denominator pairs \((70,80),(80,100),(100,100)\). The 36-row kurtosis grid now uses the corrected strict threshold and records its illustrative scope.

## 6. Recovery and evidence status

- **PROVED UNDER PRIOR:** Theorem 16 with nesting, fresh independent probe pairs, and a valid supplied norm envelope.
- **PROVED:** The corrected degree-two fourth-moment identity and its upper bound.
- **PROVED ROUTE-SPECIFIC BOUNDARY:** The exact sample-variance/Chebyshev inequality above.
- **OPEN:** obtaining useful observable scale information and useful acceptance under the actual total query budget.
- Earlier numerical grids and this report are archived before regeneration. The recovery manifest records changed files, test results, and preserved historical checksums.
