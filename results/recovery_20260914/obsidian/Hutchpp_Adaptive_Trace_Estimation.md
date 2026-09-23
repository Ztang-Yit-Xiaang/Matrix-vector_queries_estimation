# Hutch++ Track: Authoritative 3-Way Query Allocation & Sequential Pilot Theory


> **Current recovery record — 2026-09-14:** See [[CURRENT_STATE]] and [[recovery_audit_20260914]]. The 1,860-row Haar and 480-row Hessian datasets have been regenerated and validated; 217 maintained tests pass. Phase 2C proof/boundary corrections are recorded in [[proof_structural_paired_difference_confidence]] and [[structural_paired_difference_confidence_phase2c]]. Gated results remain exploratory, including the reproduced coordinate-aligned failure. Older entries below retain their historical context; they are superseded where the recovery audit corrects them.

Last updated: 2026-09-14
Parent Index: [[memory]]

---

## 🎯 Overarching Research Framework
> **How should a matrix-free trace estimator spend queries between learning, low-rank capture, and residual estimation?**
>
> Matrix-free trace estimation is governed by exact rank-aware query accounting under total query budget $m$:
> $$\boxed{q + r + \ell = m \implies \ell = m - q - r}$$
> where $q = q_{\text{target}}$ is the range-finding sketch width, $r = r_{\text{actual}} \le q$ is the realized orthonormal basis rank, and $\ell$ is the residual probe count.
>
> The pilot size $b$ is a commitment lower bound constraint ($b \le q$), **not an additive query cost block**.
>
> **Notation:** $S$ is reserved for the sketch matrix; $J$ is the random stage index; $r=r_{\mathrm{actual}}$ is realized basis rank; $r_\star$ is true knee rank; $p=b-r_\star$ is sketch oversampling; $p_Y=r_Y-r_\star$ counts realized post-knee Ritz positions; and $S_1=U^T S$, $S_2=U_\perp^T S$. See [[proof_notation]].

---

## 📐 Strict Theoretical Categorization

### `PROVED`
1. **Sequential-Pilot Adaptive Unbiasedness Theorem** (`docs/proof_sequential_unbiasedness.md`):
   $$\mathbb{E}[\widehat{t}] = \operatorname{tr}(A)$$
   holds unconditionally for any random pilot stopping time $B \in \{b_1, \dots, b_{N_{\mathrm{stg}}}\}$ when the final basis and allocation are $\mathcal{G}$-measurable and fresh residual probes are conditionally isotropic given the full pre-residual $\sigma$-algebra $\mathcal{G}$.
2. **Exact Rank-Aware Gaussian Variance** (`docs/proof_rank_aware_risk.md`):
   $$\operatorname{Var}(\widehat{t} \mid \mathcal{G}) = \frac{2 \|R A R\|_F^2}{m - q - r}, \qquad R_{\text{rank}}(q, r) = \frac{2 T(r)}{m - q - r}$$
3. **The $2\varepsilon_R$ Near-Oracle Regret Lemma** (`docs/proof_near_oracle_regret.md`):
   $$R(\widehat{a}) - R(a^*) \le 2 \varepsilon_R = 4 \max_{(q, r) \in \mathcal{A}_b} \frac{U_b(r, \delta)}{m - q - r}$$
4. **Boundary-Anchored Exponential Tail Sensitivity Ratio** (`docs/proof_exponential_sensitivity.md`):
   $$\log \frac{\widehat{T}(q)}{T(q)} = -2 \Delta_\alpha (q + 1 - b) + \log \frac{1 - e^{-2\alpha}}{1 - e^{-2(\alpha + \Delta_\alpha)}} = -2 \Delta_\alpha (q - b) + O(\Delta_\alpha)$$
5. **Deterministic Step-Spectrum Ritz Structure** (`docs/proof_step_ritz_gap.md`):
   For $A = \eta I_d + (1-\eta) U U^T$ and $b = r_\star + p$ ($p \ge 1$), the post-knee Ritz values are **exactly $\theta_{r_\star+1} = \dots = \theta_{r_\star+p} = \eta$**.

---

### `PROVED UNDER EXPLICIT ASSUMPTIONS`
1. **Safe Extrapolation Bound**:
   If $q - b \le D_{\max}$ and $|\Delta_\alpha| \le \Delta_0$, relative tail error satisfies $\left| \frac{\widehat{T}(q)}{T(q)} - 1 \right| \le e^{M |\Delta_\alpha|} - 1$ where $M = 2(D_{\max}+1) + C_{\alpha, \Delta_0}$.
2. **Soft Safety Layer Regret Bound**:
   Under $L$-Lipschitz risk, $R(q_{\text{final}}) - R(q^*) \le 2\varepsilon_R + L \left[ (1-\gamma_{\mathrm{shrink}})|q_0 - \widehat{q}| + \frac{1}{2} \right]$.
3. **High-Probability Knee Detection**:
   Requires $r_Y=\operatorname{rank}(Y)\ge r_\star+1$ so $\theta_{r_\star+1}$ exists, plus an explicit probabilistic bound on $\|S_2S_1^\dagger\|_2$. Full row rank of $S_1$ alone is insufficient.
4. **Explicit Standard-Gaussian Specialization (Theorem 15A)**:
   For $p\ge4$, Halko--Martinsson--Tropp Proposition 10.4 and Gaussian operator-norm concentration give a computable conservative constant $K^{\mathrm G}_{r_\star,d,p,\delta}$. The resulting sufficient condition is
   $$K^{\mathrm G}_{r_\star,d,p,\delta}<\sqrt{\frac{1-\eta\tau_{\mathrm R}}{\eta^3(\tau_{\mathrm R}-1)}},\qquad 1<\tau_{\mathrm R}<1/\eta.$$

---

### `EMPIRICALLY OBSERVED`
1. In the clean 50-trial, $d=500$, $m=160$ rerun, Step ($r_\star=20,\eta=0.01$) used $b=36$, found the exact knee in 50/50 trials, and had MSE $2.139\times10^{-4}$ versus $2.101\times10^{-4}$ for Standard Hutch++ (ratio $1.018$; paired-bootstrap 95% interval $[0.638,1.667]$). This supports parity, not superiority.
2. Step ($r_\star=10,\eta=0.01$) used $b=24$ and found the exact knee in 50/50 trials, but had MSE $3.142\times10^{-4}$ versus $1.553\times10^{-4}$ for Standard Hutch++ (ratio $2.023$; interval $[1.303,3.372]$). Detection and risk improvement are therefore distinct.
3. On steep Power-Law ($c=2.0$), Sequential Pilot achieved MSE $1.155\times10^{-7}$ versus $1.828\times10^{-7}$ (ratio $0.632$), but its interval $[0.371,1.049]$ includes parity. On the exponential spectrum, the ratio $3.844$ with interval $[2.516,6.021]$ is a resolved disadvantage in this experiment.
4. The sequential implementation uses Rademacher pilot sketches, so its observed step detections do not constitute a Gaussian Theorem 15A certificate or a Rademacher theorem.
5. Logarithmic tail error $|\log \widehat{T}(q) - \log T(q)|$ grows strictly linearly with extrapolation distance $(q - b)$.

#### Fixed Trust-Region Follow-Up

- The opt-in $s=4$ guard preserved conditional unbiasedness and exact budget accounting but did not pass its empirical robustness criterion: the exponential guarded/Standard MSE ratio remained $2.111$ with paired-bootstrap interval $[1.112,3.830]$.
- The implementation audit corrected a rank-deficient accounting bug by distinguishing stopped pilot sketch queries `b_final` from realized pilot rank `r_pilot_actual`. A rank-two counterexample now uses exactly its budget, and all 26 tests pass.
- A post-hoc exact-anchor diagnostic $q=q_0=53$ restored parity on all five spectra. Exact exponential oracle-risk ratios $R(49)/R(53)=1.299$ and $R(41)/R(53)=2.299$ support the same direction: under-allocation is especially costly for exponential tails. The next candidate should be asymmetric or sensitivity-aware and must be evaluated on new or held-out spectra.

---

### `HEURISTIC`
1. **Horizon-Resolution Stopping Rule**: Logs $g_j = \log(\theta_j / \theta_{j+1})$. Stops when peak gap $g_{\widehat r} \ge \gamma_{\mathrm{gap}}$ (stored by code parameter `tau_gap`) and post-gap count $b - \widehat r \ge p_{\min}$ under allocation stability and $D_{\max}$ guard.

---

### `CONJECTURE / OPEN`
1. Construction of a finite-sample valid simultaneous confidence radius $U_b(r, \delta)$ for heterogeneous unknown spectra.
2. Derivation of optimal risk-certified pilot stopping time $B^*$.
3. Sharpening the conservative Gaussian constant and obtaining explicit numerical constants for the isotropic-subgaussian theorem that covers the Rademacher sketches used by the current sequential implementation.

> [!important] Horizon versus resolution (audit clarification, 2026-08-13)
> The condition $b\ge r_\star+1$ is necessary for the ordered ratio $\theta_{r_\star}/\theta_{r_\star+1}$ to be defined, but it is not sufficient for detection. The pilot must also capture the leading eigenspace strongly enough to create a visible gap. Likewise, the near-oracle regret result is conditional on a simultaneous-confidence event; a computable finite-sample radius $U_b(r,\delta)$ has not yet been proved.

> [!important] Threshold notation and the withdrawn formula
> Use $\tau_{\mathrm R}$ for the Ritz-ratio threshold and $\gamma_{\mathrm{gap}}=\log\tau_{\mathrm R}$ for the implementation's log-gap threshold stored by `tau_gap`. Thus `tau_gap=1.5` means $\tau_{\mathrm R}=e^{1.5}\approx4.48$. The withdrawn formula with denominator $\log(1/\eta)$ is not an oversampling theorem for $p=b-r_\star$; that logarithmic denominator arises naturally from power/subspace iteration, where power depth $h$ creates geometric suppression $\eta^h$.

---

## Baseline-Safe Allocation Direction (2026-08-14)

### Theorem 11 — `PROVED UNDER AN EXPLICIT SIMULTANEOUS-CONFIDENCE EVENT`

For a finite feasible set containing $q_0$, suppose
$$
\forall q:\quad
|\widehat{\mathcal R}_b(q)-\mathcal R(q)|\le e_b(q).
$$
With $U_b^R(q)=\widehat{\mathcal R}_b(q)+e_b(q)$ and $L_b^R(q)=\widehat{\mathcal R}_b(q)-e_b(q)$, accept a nonbaseline action only if
$$
U_b^R(q)\le L_b^R(q_0).
$$
Then
$$
\mathcal R(q)\le U_b^R(q)\le L_b^R(q_0)\le\mathcal R(q_0).
$$
Simultaneous validity is required because $q$ is data-dependent. The same proof applies to risk differences, to rank-aware actions $a=(q,r)$ versus $a_0=(q_0,r_0)$, and to a random stopping stage after a finite union bound. A computable finite-sample radius for the actual risk is still `OPEN`; an oracle-surrogate certificate is not automatically an MSE certificate.

### Held-out asymmetric-guard verdict — `EMPIRICALLY OBSERVED`

- The isolated frozen benchmark used 24 spectra, five methods, 200 trials per setup, $d=500$, $m=160$, and 20,000 bootstrap resamples per comparison. All 24,000 trials passed exact query and allocation checks; all 47 tests pass.
- The $[0,+4]$ heuristic produced resolved gains at exponential $\alpha=0.08,0.10,0.15$ (MSE ratios $0.568,0.473,0.431$), but resolved disadvantages at exponential $\alpha=0.02$ ($1.418$) and steps $(5,0.001)$ ($1.927$), $(5,0.05)$ ($1.500$), $(15,0.001)$ ($1.408$), and $(30,0.001)$ ($1.451$). It fails the frozen robustness criterion.
- Large positive raw shifts were harmful on every $\eta=0.001$ step; a negative shift was harmful on the power/exponential mixture. The asymmetric floor repaired the mixture, while its $+4$ allowance remained too large for several low-noise steps.
- The 24 intervals are exploratory and have no familywise error control. Paired trial indices do not imply probe-identical randomness across methods, so unresolved differences are not equivalence and the forced-baseline control does not prove a distinct sequential-basis effect.

### Next step

The active research problem is a computable simultaneous certificate for
$$
\Delta(q)=\mathcal R(q)-\mathcal R(q_0),
\qquad
\widehat\Delta_b(q)+C_b(q)\le0.
$$
Do not tune another fixed guard on these held-out outcomes. Theorem 15A remains useful for knee detection, but certified allocation movement is now the limiting theory.

---

## Marginal-Risk Repair and Exact Conditional-Risk Bridge (2026-08-14)

### Corrected theory

- The exact full-rank marginal identity is Theorem 12 on the domain where both $q$ and $q+1$ are feasible:
  $$
  \mathcal R(q+1)<\mathcal R(q)
  \iff
  M(q)=(m-2q)\lambda_{q+1}^2-2T(q)>0.
  $$
- Lemma 12.1 proves single crossing because
  $$
  M(q+1)-M(q)=(m-2q-2)(\lambda_{q+2}^2-\lambda_{q+1}^2)\le0.
  $$
- The corrected step result is Theorem 13: for $0<\eta<1$ and feasible $r_\star$, the unique ideal minimizer is $q^*=r_\star$ when
  $$
  2r_\star+2(d-r_\star)\eta^2<m<2d.
  $$
  The earlier condition $m<2d$ alone was insufficient. Boundary equalities create plateaus.
- The three-state marginal decision is certified only when supplied with valid simultaneous $C_b^M(q)$. Constructing such a computable radius remains `OPEN`.

### Implementation status

- `Adaptive_Hutch_pplus_MarginalRisk` is retained as an explicitly `HEURISTIC` Ritz-model prototype. It separates acquired pilot sketch columns from numerical rank, validates configurations, supports the baseline-feasibility cap, and records its raw proposal and intervention.
- A resolved knee or nonpositive fitted marginal at the current pilot floor now stops at the committed sketch width. A resolved-knee fixture initialized at $B=12$ records the rejected power-law proposal $q=46$ but uses $q=12$. With the default $B=8$ start, the same step example currently stops earlier at $q=8$ because its fitted marginal is already nonpositive; this is heuristic behavior, not a certified or empirically validated improvement.
- Rank-two and zero-matrix accounting tests, certificate-state tests, step-theorem tests, and corrected probe-identical tests all pass. The full suite contains 75 passing tests.

### Frozen exact-risk result — `EMPIRICALLY OBSERVED`

- The bridge uses all 24 frozen spectra, 200 shared nested sketch trials per setup, $q=0,\ldots,76$, and exact conditional risks
  $$
  \mathcal R_G(Q_q)=\frac{2\|H_q\|_F^2}{\ell_q},
  \qquad
  \mathcal R_R(Q_q)=\frac{2\sum_{i\ne j}(H_q)_{ij}^2}{\ell_q}.
  $$
- The validated outputs contain 369,600 trial rows, 1,848 mean-curve rows, and 72 minimizer rows.
- On the 12 non-step setups, choosing the ideal spectral-tail minimizer costs at most $4.2\%$ extra Gaussian risk and $3.9\%$ extra Rademacher risk. The ideal target is therefore a useful approximation there even when its exact minimizing integer differs.
- All 12 step cases favor oversampling beyond $r_\star$: $+3$ to $+16$ for Gaussian risk and $+4$ to $+24$ for Rademacher risk. At $\eta=0.001$, $q=r_\star$ costs $103\times$--$196\times$ the minimum Gaussian risk and $2{,}073\times$--$5{,}087\times$ the minimum Rademacher risk because ideal $T(q)$ omits randomized range-capture leakage.
- Rademacher step optima are systematically larger than Gaussian optima; using the Gaussian-optimal allocation costs up to $36.9\%$ additional Rademacher risk.

### Active conclusion

Future confidence bounds should target realized residual energy, not $T(q)$ alone, and should keep Gaussian and Rademacher probe theory separate. The evidence is conditional on one frozen orientation per setup and uses exploratory, non-familywise bootstrap intervals.

### Rank-deficient bridge revision before implementation

The controlled comparison now fixes one signal orientation $U_\star$ for each $r_\star$ and reuses it across $\eta\in\{0,10^{-14},10^{-10},10^{-6}\}$. The common basis seeds and Rademacher prefixes are also reused, so changing $\eta$ does not simultaneously change the signal-subspace orientation. This is essential because exact Rademacher conditional risk depends on coordinate-basis off-diagonal energy.

The bridge will report

$$
\operatorname{rank\_efficiency}(q)=\frac{r_q}{q},
\qquad q>0,
$$

the rejected count $q-r_q$, and cumulative exact rejected-transition penalties. Under nested increments $r_j-r_{j-1}\in\{0,1\}$, the cumulative rejected count is exactly $q-r_q$ and is not duplicated as a second column.

For $E_X(Q)>0$, the realized successful-direction rule is equivalently

$$
\frac{E_X(Q)-E_X(Q')}{E_X(Q)}>\frac{2}{D}.
$$

Thus the fractional residual energy removed must exceed the fractional loss of residual-probe capacity. The $E_X(Q)=0$ boundary remains undivided. Cross-budget comparisons use regret and allocation diagnostics, not raw risk. Commit `8de09a7` contains the revised design; experiment implementation remains pending final review.

### Final rank-deficient bridge result (2026-08-15)

The bridge is now implemented and validated on 554,400 allocation rows. Numerical rank loss is activated at $\eta=0$ and mostly at $10^{-14}$, while $10^{-10}$ and $10^{-6}$ remain full rank. Rejected queries have the exact nonnegative cost predicted by Corollary 14.1 when positive tail risk remains and zero cost after exact residual risk reaches zero.

The decisive result is not the rejected-query count. At $\eta=10^{-6}$, all paths satisfy $r_q=q$, yet the rank/full oracle chooses $q=r_\star$ and both realized risks choose $q=r_\star+1$. At $m=160$, the Gaussian mean-risk ratios at the oracle choice versus the realized minimum are approximately $9.83$, $1602.68$, and $1.62$ for $r_\star=5,15,30$. The extra direction helps only a small fraction of individual trials but protects against rare catastrophic randomized range-capture failures.

Therefore the final target decision is

$$
\boxed{
\text{certify realized probe-specific conditional risk differences, not }T(r)\text{ alone}.
}
$$

For the practical Rademacher estimator, the target is $\mathcal R_R(Q;q,r)=2E_R(Q)/(m-q-r)$ or its action difference/marginal. Gaussian $E_G(Q)$ remains a separate theoretical/control target. The UROP experimental scope is now frozen and the project moves to report writing.

#### Why one extra direction wins here

The observed $q_G^\star=r_\star+1$ is not universal. Write $S_1=U_1^TS$ and $S_2=U_2^TS$. At $q=r_\star$, $S_1$ is square, so there is no oversampling redundancy. Full rank of $AS$ does not imply good dominant-subspace recovery. Although $S$ is coordinate Rademacher, the rotated block $U_1^TS$ is generally not iid Rademacher, so iid square-Rademacher singularity results do not apply directly. When $S_1$ has full row rank, leakage is amplified through $\Lambda_2S_2S_1^\dagger\Lambda_1^{-1}$.

For a realized path with positive old energy, the extra direction wins exactly when

$$
\frac{E_G(Q_{r_\star+1})}{E_G(Q_{r_\star})}
<
\frac{m-2r_\star-2}{m-2r_\star}.
$$

At $(m,r_\star)=(160,15)$ this requires only a 1.54% reduction. Yet only 5.5%--7.5% of individual Gaussian paths improve: the empirical mean prefers the extra column because it protects against rare catastrophic losses. The right interpretation is instance-specific oversampling insurance, not a universal $+1$ rule.

#### Focused mechanism-audit result

The seed-exact reconstruction found no rank, risk, or query-accounting bug. At $q=k=r_\star$, the exact step-model graph factor $F=\eta S_2S_1^{-1}$ determines the dominant-subspace angles, and

$$
E_G(Q)=\eta^2(d-q)+2\eta(1-\eta)\|Z\|_F^2+(1-\eta)^2\|Z^TZ\|_F^2,
\qquad Z=(I-QQ^T)U_1,
$$

maps the leakage to Gaussian risk exactly. In the frozen 200-path experiment, the worst 1% contribute 37%--99.94% of Gaussian and 84.97%--99.997% of Rademacher risk at $q=k$. The median path prefers $k$, whereas the empirical mean prefers $k+1$ because the nested extra column removes the catastrophic tail. A new 216-row sensitivity grid finds optimal shifts of 0, 1, 2, and larger, so the scientific conclusion is positive-oversampling tail insurance—not a universal $+1$ theorem. See [[q_rank_vs_realized_risk_mechanism]].

---

## Approved Rank-Aware Multi-Budget Bridge Design (2026-08-14)

- The committed design `cac0b21` preserves the frozen 24-spectrum bridge and adds a separate 12-spectrum rank-deficient diagnostic.
- The primary budget is $m=160$ with sensitivity budgets $80$ and $240$. Derived worst-case-feasible endpoints are $36$, $76$, and $116$, giving 554,400 trial-allocation rows at 200 paired basis trials.
- The proof package distinguishes the ideal next-eigenvector marginal from realized Gaussian/Rademacher energy-drop marginals and includes the nonnegative failed-rank penalty.
- Risk summaries use additive regret and $\mathcal R(0)$-normalized additive regret universally; multiplicative regret remains undefined at exact zero minima.
- The numerical design uses the matrix-free identity $A=\eta I+(1-\eta)U_\star U_\star^T$ and a stable nonnegative residual-energy formula, avoiding catastrophic cancellation for $\eta=10^{-14}$.
- The complete adjustable argument surface is frozen in the specification. Nondefault configurations must use separate outputs. Implementation has not begun.
- Commit `4bd9aee` rewrites the full specification in renderer-independent Markdown. Every theorem, configuration, output, test, and limitation remains, but equations now display as Unicode/plain-text formula blocks without a math plugin.
- Current format is commit `1f3db32`: full LaTeX equations restored inside the same Markdown file, no PDF or extra artifact, with all display delimiters moved flush-left for better renderer compatibility.
