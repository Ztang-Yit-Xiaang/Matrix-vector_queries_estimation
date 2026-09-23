# Phase 1D: Rademacher Linear Projection Lower-Tail Analytic No-Go Report

**Status:** `PROVED` analytic no-go for the worst-case scale-free truncation--Bernstein route on the nondegenerate Hoeffding linear component at $s \le 32$.

---

## 1. Executive Summary & Research Question

Phase 1C proved that the full sample variance $S_s^2$ of centered Rademacher quadratic forms has fourth-moment-plus-Chebyshev confidence radii that are budget-vacuous ($\varepsilon_{\text{Ch}} > 1$) for $s \le 32$.

Phase 1D investigates whether isolating the **nondegenerate linear component** of the Hoeffding decomposition and applying a state-of-the-art large-deviation inequality for quadratic Rademacher chaos can yield a non-vacuous candidate lower-tail bound:

$$
\boxed{
\text{Can a large-deviation bound for quadratic Rademacher chaos control the candidate-underestimation tail at } s \le 32\text{?}
}
$$

**Analytical Verdict:**

$$
\boxed{\texttt{STRONG LINEAR NO-GO}}
$$

For the frozen truncation--Bernstein family, the variance envelope of the capped squared chaos is uniformly bounded below by $\nu_\kappa(T) \ge 80$. Consequently, the one-sided Bernstein candidate lower-tail probability is strictly bounded below by:

$$
D_{s,\kappa}(\varepsilon, T) > e^{-s/160} \ge e^{-32/160} = e^{-0.2} \approx 0.81873
$$

Since the maximum declared linear component failure probability is $\delta_{\text{linear}} \le 0.05$, the analytic lower bound ($0.8187$) exceeds the failure threshold by more than $16\times$. This mathematically rules out the truncation--Bernstein route for all $s \le 32$, regardless of the structural ratio $\kappa = \|C\|_2 / \|C\|_F$.

---

## 2. Mathematical Framework & Normalization

Condition on the pre-certification $\sigma$-algebra $\mathcal{G}$. Let $H = RAR$ with $R = I - QQ^T$. Define the zero-diagonal symmetric matrix:

$$
C = H - \operatorname{diag}(H).
$$

For a fresh coordinate-Rademacher vector $g \sim \{-1, +1\}^d$, the centered quadratic form is $Z = g^T C g$. The exact conditional variance is:

$$
\sigma^2 = \operatorname{Var}(g^T H g \mid \mathcal{G}) = 2 \|C\|_F^2.
$$

### Imported Theorem (Cortinovis & Kressner 2022, Theorem 2, Eq. 8)
For any nonzero symmetric matrix $C$ with zero diagonal:

$$
\Pr(|g^T C g| \ge t \mid \mathcal{G}) \le 2 \exp\left( -\frac{t^2}{8 \|C\|_F^2 + 8 t \|C\|_2} \right), \qquad t > 0.
$$

Defining the dimensionless structural ratio $\kappa = \frac{\|C\|_2}{\|C\|_F} \in (0, 1]$ and substituting $t = u \sigma$ with $\sigma = \sqrt{2}\|C\|_F$ yields the scale-free tail bound for $V = Z^2 / \sigma^2$:

$$
\Pr(V \ge v \mid \mathcal{G}) \le p_\kappa(v) \equiv \min\left\{ 1, 2 \exp\left( -\frac{v}{4 + 4\sqrt{2v}\kappa} \right) \right\}.
$$

---

## 3. Hoeffding Linear Component & The Candidate Lower Tail

The U-statistic Hoeffding decomposition of sample variance $S_s^2$ is:

$$
S_s^2 - \sigma^2 = \frac{2}{s} \sum_{i=1}^s h_1(X_i) + \binom{s}{2}^{-1} \sum_{i < j} h_2(X_i, X_j),
$$

where $h_1(X) = \frac{Z^2 - \sigma^2}{2}$. Normalizing by $\sigma^2$ gives the average normalized squared-chaos error:

$$
L_s \equiv \frac{(2/s)\sum_i h_1(X_i)}{\sigma^2} = \frac{1}{s} \sum_{i=1}^s \left( \frac{Z_i^2}{\sigma^2} - 1 \right) = \frac{1}{s} \sum_{i=1}^s W_i.
$$

For candidate safety, the catastrophic failure event is **sample-variance underestimation** ($S_a^2 \le (1 - \varepsilon)\sigma_a^2$), corresponding to the lower tail:

$$
L_s \le -\varepsilon.
$$

---

## 4. The Truncation-Bernstein Obstruction

To apply Bernstein's inequality, large values of $V = Z^2/\sigma^2$ are capped at $T \ge 64$: $V^{(T)} = \min(V, T)$.

1. **Truncation Bias**:
   $$b_T = \mathbb{E}[(V - T)_+] \le \beta_\kappa(T) = 4 e^{-a_{\kappa,T}\sqrt{T}} \left( \frac{\sqrt{T}}{a_{\kappa,T}} + \frac{1}{a_{\kappa,T}^2} \right), \quad a_{\kappa,T} = \frac{1}{4\sqrt{2}\kappa + 4/\sqrt{T}}.$$
2. **Variance Envelope Floor**:
   Combining Bonami's hypercontractive fourth moment $\mathbb{E}[V^2] \le 81$ with the truncation bound yields:
   $$\nu_\kappa(T) = \min\left\{ 81, \frac{T^2}{4}, 81 - \max(0, 1 - \beta_\kappa(T))^2 \right\}.$$
   Since $T \ge 64 \implies T^2/4 \ge 1024$ and $\beta_\kappa(T) \ge 0$, we have:
   $$\boxed{80 \le \nu_\kappa(T) \le 81 \quad \forall T \in \mathcal{T}, \kappa \in (0, 1].}$$
3. **One-Sided Bernstein Lower-Tail Bound**:
   For any $\varepsilon \in (0, 1)$ with $\beta_\kappa(T) < \varepsilon$, setting $x = \varepsilon - \beta_\kappa(T) \in (0, 1)$:
   $$D_{s,\kappa}(\varepsilon, T) = \exp\left( -\frac{s x^2}{2(\nu_\kappa(T) + x/3)} \right) > \exp\left( -\frac{s (1)^2}{2(80 + 1/3)} \right) > \exp\left( -\frac{s}{160} \right).$$

For maximum budget $s = 32$:
$$D_{32,\kappa}(\varepsilon, T) > e^{-32/160} = e^{-0.2} \approx 0.81873.$$

---

## 5. Artifact & Theorem Summary

All 4 Phase 1D output CSV files have been generated and validated:

| File | Rows | Description |
| :--- | :---: | :--- |
| `rademacher_linear_projection_no_go_phase1d_manifest.csv` | 1 | Complete metadata, constants, DOI references, and SHA-256 hashes. |
| `rademacher_linear_projection_no_go_phase1d_theorem_grid.csv` | 24 | Complete $(s, \delta_{\text{joint}}, \text{allocation})$ grid; 100% `STRONG LINEAR NO-GO`. |
| `rademacher_linear_projection_no_go_phase1d_cap_regression.csv` | 900 | Numerical sweep over $(s, T, \kappa, \varepsilon)$ verifying $80 \le \nu \le 81$ and floor dominance. |
| `rademacher_linear_projection_no_go_phase1d_verdict.csv` | 1 | High-level verdict summary (`new_matvec_queries = 0`). |

---

## 6. Strategic Takeaway for Future Theory

1. **Why Phase 1D is a High-Value Result**:
   * It prevents researchers from wasting effort trying to prove concentration on the canonical $h_2$ remainder, because the necessary linear component $h_1$ already fails to close under truncation--Bernstein.
2. **The Deeper Probabilistic Lesson**:
   * Large-deviation theorems bound the upper tail ($\Pr(Z \text{ is huge})$), but candidate risk safety requires a **small-ball / lower-tail bound** on the average of squared chaos ($\Pr(\frac{1}{s}\sum Z_i^2 \ll \mathbb{E}Z^2)$).
3. **The Preferred Continuation Route**:
   * Rather than constructing separate multiplicative intervals on $S_a^2$ and $S_0^2$, future certification should directly analyze the **paired common-probe difference**:
     $$\widehat{\Delta}_R = \binom{s}{2}^{-1} \sum_{i < j} \left[ \frac{(X_{a,i} - X_{a,j})^2}{2\ell_a} - \frac{(X_{0,i} - X_{0,j})^2}{2\ell_0} \right],$$
     exploiting common-probe covariance cancellation.
