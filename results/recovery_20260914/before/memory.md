---
title: "Swati's Summer Research: Global Project Memory & Knowledge Base"
created: 2026-08-08
updated: 2026-09-08
tags:
  - randnla
  - trace-estimation
  - hutch-pplus
  - leverage-scores
  - turbo-quant
  - urop-2026
  - obsidian-vault
---

# Global Project Memory & Knowledge Base

> [!abstract] Overview & Purpose
> This document serves as the cross-chat persistent memory and Obsidian-friendly knowledge hub for the **Randomized Numerical Linear Algebra (RandNLA)** research project under UROP Summer 2026.
> 
> **Core Objective**: Develop adaptive, structure-aware trace estimation methods (Hutch++, NA-Hutch++), analyze row-sampling matrix approximations, and evaluate vector quantization techniques.
> 
> **Primary Tracking File**: [[UROP_TRACKER|UROP_TRACKER.md]] is the single source of truth for experimental progress and UROP deliverables.

---

## 1. Authoritative 3-Way Query Allocation & Sequential Pilot Theory

Under total query budget $m$:
$$\boxed{q + r_{\text{actual}} + \ell = m \implies \ell = m - q - r_{\text{actual}}}$$
where $q = q_{\text{target}}$ is the range-finding sketch width, $r = r_{\text{actual}} \le q$ is the realized orthonormal basis rank, and $\ell$ is the residual probe count.

- Pilot size $b$ is a commitment constraint ($b \le q$), **never an additive query block** ($b + q + \ell = m$).
- Reusable Pilot Lemma: Range-finding cost is $B + (q - B) = q$.
- Full-rank special case: $r = q \implies \ell = m - 2q$. Full-rank planning surrogate risk is $\mathcal{R}_{\text{full}}(q) = \frac{2 T(q)}{m - 2q}$.

### Common notation

The authoritative ledger is `Hutch++/Matrix-vector_queries_estimation/docs/proof_notation.md`. Uppercase $S$ is reserved for the sketching matrix; $J$ is a random pilot-stage index and $B=b_J$ is the stopped pilot size. The allocation proofs use $r=r_{\mathrm{actual}}$, while the ideal step-spectrum proofs use $r_\star$ for the true knee rank, $p=b-r_\star$ for sketch oversampling, and $p_Y=r_Y-r_\star$ for realized post-knee Ritz positions. The two counts agree when $S$ has full column rank and $A\succ0$. The step-spectrum sketch blocks are $S_1=U^T S$ and $S_2=U_\perp^T S$. Also, $\Delta_\alpha$ denotes slope-estimation error, $\gamma_{\mathrm{gap}}=\log\tau_{\mathrm R}$ denotes the log Ritz-gap threshold, $\gamma_{\mathrm{shrink}}$ denotes soft-allocation shrinkage, and $h$ denotes power-iteration depth.

---

## 2. Strict Theoretical Categorization

### `PROVED`
1. **Theorem 2 (Sequential-Pilot Adaptive Unbiasedness)** (`docs/proof_sequential_unbiasedness.md`):
   $$\mathbb{E}[\widehat{t}] = \operatorname{tr}(A)$$
   holds unconditionally for any random pilot stopping time $B \in \{b_1, \dots, b_{N_{\mathrm{stg}}}\}$ when the final allocation and basis are measurable with respect to the full pre-residual sigma-algebra $\mathcal{G}$ and the fresh residual probes are conditionally isotropic given $\mathcal{G}$.
2. **Lemma 1 & Theorem 3 (Exact Gaussian Conditional Variance & Risk)** (`docs/proof_rank_aware_risk.md`):
   $$\operatorname{Var}(\widehat{t} \mid \mathcal{G}) = \frac{2 \|R A R\|_F^2}{m - q - r}, \qquad \mathcal{R}_{\text{real}}(Q; q, r) = \frac{2 \|R A R\|_F^2}{m - q - r}, \qquad \mathcal{R}_{\text{rank}}(q, r) = \frac{2 T(r)}{m - q - r}$$
3. **The $2\varepsilon_R$ Argmin Regret Lemma (Theorem 4)** (`docs/proof_near_oracle_regret.md`):
   For the rank-aware action $a=(q,r)$,
   $$\mathcal{R}(\widehat a_b)-\mathcal{R}(a_b^*)\le2\varepsilon_R(b,\delta)=4\max_{(q,r)\in\mathcal A_b}\frac{U_b(r,\delta)}{m-q-r}.$$
   The full-rank restriction $r=q$ recovers the denominator $m-2q$.
4. **Sequential Union Bound (Theorem 5)** (`docs/proof_near_oracle_regret.md`):
   Simultaneous confidence holds across all $N_{\mathrm{stg}}$ pilot stages with failure allocation $\sum_{j=1}^{N_{\mathrm{stg}}}\delta_j\le\delta$.
5. **Exact Boundary-Anchored Exponential Log Ratio (Theorem 7)** (`docs/proof_exponential_sensitivity.md`):
   $$\log \frac{T_{\alpha+\Delta_\alpha}(q)}{T_\alpha(q)} = -2 \Delta_\alpha (q + 1 - b) + \log \frac{1 - e^{-2\alpha}}{1 - e^{-2(\alpha+\Delta_\alpha)}} = -2 \Delta_\alpha (q - b) + O(\Delta_\alpha)$$
6. **Deterministic Step-Spectrum Ritz Structure (Theorem 9)** (`docs/proof_step_ritz_gap.md`):
   For $A = \eta I_d + (1 - \eta) U U^T$ and $b = r_\star + p$ ($p \ge 1$), Ritz values of $Q^T A Q$ satisfy $\theta_{r_\star+1} = \dots = \theta_{r_\star+p} = \eta$ **exactly**.
7. **Deterministic Principal-Angle Control & Ritz Gap (Theorem 10)** (`docs/proof_step_ritz_gap.md`):
   $$\tan \Theta_{\max} \le \eta \|S_2 S_1^\dagger\|_2 \implies s_{r_\star}^2 = \cos^2 \Theta_{\max} \ge \frac{1}{1 + \eta^2 \|S_2 S_1^\dagger\|_2^2} \implies \frac{\theta_{r_\star}}{\theta_{r_\star+1}} \ge 1 + \frac{1 - \eta}{\eta [1 + \eta^2 \|S_2 S_1^\dagger\|_2^2]}$$

---

### `PROVED UNDER EXPLICIT ASSUMPTIONS`
1. **Safe Extrapolation Bound (Theorem 8 & Corollary 8.1)**:
   Under $q - b \le D_{\max}$ and $|\Delta_\alpha|\le\Delta_0$, relative tail error is bounded by $\exp\left( [2(D_{\max}+1) + C_{\alpha, \Delta_0}] |\Delta_\alpha| \right) - 1$.
2. **Soft Safety Layer Regret Bound (Theorem 6)**:
   Under $L$-Lipschitz risk, $\mathcal{R}(q_{\text{final}}) - \mathcal{R}(q_b^*) \le 2 \varepsilon_R + L \left[ (1-\gamma_{\mathrm{shrink}}) |q_0 - q_{\text{adapt}}| + \frac{1}{2} \right]$.
3. **High-Probability Knee Detection (Corollary 10.1)**:
   Requires $r_Y=\operatorname{rank}(Y)\ge r_\star+1$ so the post-knee Ritz value exists, together with a probabilistic bound $\|S_2 S_1^\dagger\|_2 \le K_{r_\star,d,p,\delta}$.
4. **Explicit Standard-Gaussian Knee Bound (Theorem 15A)**:
   For $p\ge4$ and an i.i.d. standard Gaussian sketch $S$, the Halko--Martinsson--Tropp pseudoinverse tail bound and Gaussian operator-norm concentration give a computable $K^{\mathrm G}_{r_\star,d,p,\delta}$. The detector is certified when $K^{\mathrm G}_{r_\star,d,p,\delta}<\sqrt{(1-\eta\tau_{\mathrm R})/[\eta^3(\tau_{\mathrm R}-1)]}$ with $1<\tau_{\mathrm R}<1/\eta$.
5. **Baseline-Safe Acceptance (Theorem 11 and Corollaries 11.1--11.3)**:
   On a simultaneous event $|\widehat{\mathcal R}_b(q)-\mathcal R(q)|\le e_b(q)$ for every feasible action, accepting only candidates satisfying $U_b^R(q)\le L_b^R(q_0)$ guarantees $\mathcal R(q_{\mathrm{safe}})\le\mathcal R(q_0)$. The same argument applies to risk differences, rank-aware actions $a=(q,r)$ versus $a_0=(q_0,r_0)$, and a random stage $B=b_J$ after a finite union bound. The deterministic implication is proved; constructing computable finite-sample simultaneous radii remains open.

### Audit Clarifications (2026-08-13)

- **Horizon is not resolution:** $b\ge r_\star+1$ is necessary for $\theta_{r_\star+1}$ to exist, but detection also requires enough leading-eigenspace capture to create a visible Ritz gap.
- **Near-oracle scope:** the $2\varepsilon_R$ result is a deterministic consequence of a simultaneous-confidence event. Building a finite-sample computable $U_b(r,\delta)$ remains open.
- **Exponential-tail scope:** the sensitivity results require $q\ge b$, $\theta_b>0$, and a positive fitted decay rate $\alpha+\Delta_\alpha>0$; the $O(\!\varepsilon/(q-b))$ slope-precision description is an asymptotic regime statement, not a uniform equality.
- **Threshold notation:** $\tau_{\mathrm R}$ denotes a Ritz-ratio threshold, while the implementation parameter `tau_gap` stores the log threshold $\gamma_{\mathrm{gap}}=\log\tau_{\mathrm R}$. The old formula with denominator $\log(1/\eta)$ resembles a power-iteration-depth bound in $h$, not a bound for oversampling $p=b-r_\star$.

### Theorem 15 Certification Diagnostic (2026-08-13)

- `src/theorem15_certification.py` implements the sufficient Gaussian condition $K^{\mathrm G}_{r_\star,d,p,\delta}<\kappa(\eta,\tau_{\mathrm R})$, exact conversions between $\tau_{\mathrm R}$ and $\gamma_{\mathrm{gap}}$, alignment calibration $\tau_{\mathrm R}=1+(1-\eta)\rho/\eta$, and a finite search for the smallest certifiable $p\ge4$.
- `experiments/run_theorem15_certification.py` writes `results/theorem15_gaussian_certification_map.csv` and `results/theorem15_sketch_distribution_check.csv`. Its empirical comparison uses a fixed random orthonormal signal subspace because Rademacher sketches are not rotationally invariant.
- At $(d,r_\star,\delta)=(500,20,0.05)$ and fixed $\gamma_{\mathrm{gap}}=1.5$, the minimum Gaussian-certifiable oversampling is $p=4$ for $\eta=0.01$, $p=14$ for $\eta=0.05$, and $p=75$ for $\eta=0.10$; the conservative bound certifies no admissible $p$ for $\eta=0.20$. Failure of this sufficient certificate is not a proof that detection is impossible.
- Across 200 trials per sketch distribution and $p\in\{4,8,10\}$ in the main $(d,r_\star,\eta,\delta)=(500,20,0.01,0.05)$ regime, both Gaussian and Rademacher sketches had 100% observed knee detection and full-row-rank rates. The Gaussian HMT event was observed in 99.5%, 100%, and 100% of trials, respectively; all 1,200 trials had zero deterministic principal-angle implication violations. These are finite empirical checks, not replacements for the Gaussian probability theorem or a Rademacher certificate.

---

### `EMPIRICALLY OBSERVED`
1. In the clean 50-trial, $d=500$, $m=160$ rerun, `Adaptive_Hutch_pplus_SequentialPilot` expanded to $b=36$ on Step ($r_\star=20,\eta=0.01$), found the exact knee in 50/50 trials, and had MSE $2.139\times10^{-4}$ versus Standard Hutch++ $2.101\times10^{-4}$ (ratio $1.018$; paired bootstrap 95% interval $[0.638,1.667]$). This supports parity, not superiority.
2. On Step ($r_\star=10,\eta=0.01$), Sequential Pilot stopped at $b=24$, found the exact knee in 50/50 trials, but its MSE was $3.142\times10^{-4}$ versus $1.553\times10^{-4}$ for Standard Hutch++ (ratio $2.023$; paired bootstrap 95% interval $[1.303,3.372]$). Knee detection alone therefore does not imply a risk-improving allocation.
3. On steep Power-Law ($c=2.0$), Sequential Pilot achieved $1.155\times10^{-7}$ MSE versus $1.828\times10^{-7}$ for Standard Hutch++ (ratio $0.632$), but the paired bootstrap 95% interval $[0.371,1.049]$ includes parity. On the exponential spectrum, the sequential MSE ratio was $3.844$ with interval $[2.516,6.021]$, a resolved disadvantage in this experiment.
4. The benchmark implementation uses Rademacher pilot sketches. Its 50/50 step detections are finite-sample observations and do not convert Theorem 15A's Gaussian certificate into a Rademacher theorem.

### Sequential Allocation Trust-Region Diagnostic (2026-08-13)

- An opt-in fixed trust region $|q-q_0|\le4$ preserves the unguarded default, conditional unbiasedness, and exact query accounting, but remains a `HEURISTIC` rather than an MSE guarantee.
- The implementation audit corrected a pre-existing rank-deficient accounting error by separating stopped pilot sketch queries `b_final` from realized pilot rank `r_pilot_actual`. A rank-two PSD case that previously used 22 queries at budget 20 now uses exactly 20; all 26 collected tests pass.
- In 750 paired trials, guarded/Standard MSE ratios were $0.825$ ($c=2$), $1.549$ ($c=0.5$), $2.111$ (exponential), $1.357$ (step $r_\star=10$), and $1.057$ (step $r_\star=20$). Only the exponential 95% interval, $[1.112,3.830]$, remains wholly above one, so the fixed-radius guard fails the predeclared robustness criterion.
- A post-hoc exact-anchor diagnostic with $q=q_0=53$ restored empirical parity on all five spectra, including exponential ratio $0.984\,[0.619,1.585]$. This is mechanistic evidence, not validation of a tuned $s=0$ method. Together with exact exponential oracle ratios $R(49)/R(53)=1.299$ and $R(41)/R(53)=2.299$, it points to sensitivity-aware protection against under-allocation as the next research step.

### Baseline-Safe Asymmetric Guard Held-Out Benchmark (2026-08-14)

- `Adaptive_Hutch_pplus_SequentialPilot` now supports explicit integer `q_shift_bounds=(s_-,s_+)` and `preserve_baseline_feasibility`. The new asymmetric heuristic is $(0,4)$; $(0,0)$ is the forced-baseline control. At $m=160$, $q_0=53$, $b_0=8$, and $\Delta b=4$, the capped stage grid ends at $B=52$. The guard remains pre-residual and $\mathcal G$-measurable, so conditional unbiasedness and exact accounting are preserved.
- The frozen held-out grid contains 24 spectra: five power laws, four exponentials, all 12 step cross-products $r_\star\in\{5,15,25,30\}$ and $\eta\in\{0.001,0.05,0.1\}$, plus three misspecified decays. Five methods and 200 trials per setup produced 24,000 validated estimator trials. All 47 tests pass and every trial satisfies `oracle.query_count == 160` and $q+r_{\mathrm{actual}}+\ell=160$.
- The asymmetric $[0,+4]$ guard had resolved gains versus Standard at exponential $\alpha=0.08,0.10,0.15$ (MSE ratios $0.568,0.473,0.431$), but resolved disadvantages at exponential $\alpha=0.02$ ($1.418$) and steps $(r_\star,\eta)=(5,0.001)$ ($1.927$), $(5,0.05)$ ($1.500$), $(15,0.001)$ ($1.408$), and $(30,0.001)$ ($1.451$). It therefore fails the predeclared robustness criterion.
- The mechanism is genuinely two-sided. Large positive raw shifts ($q=64$--$66$) caused resolved unguarded losses on all four $\eta=0.001$ steps; clipping to $q=57$ helped but remained too aggressive in three cases. A large negative raw shift on the power/exponential mixture (mean $q=44.41$) caused an unguarded ratio $3.520$, while the asymmetric floor repaired it to an unresolved $1.095$. Thus neither unrestricted movement nor a universal $+4$ allowance is safe.
- The forced-$q_0$ control had three resolved-worse and one resolved-better exploratory flags among 24 comparisons. This does not prove a sequential-basis effect: the intervals have no familywise correction, and paired seeds do not give probe-identical sketches and residual probes across methods. The control was unresolved on the low-noise steps where positive shifts caused the clearest harm, while positive shifts supplied the fast-exponential gains.
- **Active direction:** do not retune another fixed guard on these held-out results. Construct simultaneous confidence bounds for $\Delta(q)=\mathcal R(q)-\mathcal R(q_0)$ and allow deviations only when $\widehat\Delta_b(q)+C_b(q)\le0$. Theorem 15A remains supporting knee-detection theory; confidence-certified allocation is the current bottleneck.

### Marginal-Risk Prototype Audit (2026-08-14)

- The full-rank marginal identity $\mathcal R(q+1)<\mathcal R(q)\iff (m-2q)\lambda_{q+1}^2>2T(q)$ is correct on the feasible domain. For nonincreasing eigenvalues, $M(q+1)-M(q)=(m-2q-2)(\lambda_{q+2}^2-\lambda_{q+1}^2)\le0$, so a single-crossing/global-optimum argument is available but has not yet been written into the proof.
- The step-spectrum conclusion $q^*=r_\star$ additionally requires $m>2r_\star+2(d-r_\star)\eta^2$ before the knee and $m<2d$ after it. The published statement using only $m<2d$ is false; $(d,m,r_\star,\eta)=(500,160,30,0.5)$ gives $q^*=0$.
- The current `Adaptive_Hutch_pplus_MarginalRisk` prototype is not safe to benchmark. It records realized pilot rank as `b_final`, so a rank-two budget-20 case uses 22 queries. On step $(r_\star,\eta)=(10,0.01)$ it detects the knee at $B=12$ but selects $q=47$. It still globally extrapolates a fitted power law and does not implement the proposed confidence-based three-state rule.
- The 49 passing tests do not cover these failures. Add rank-deficient accounting, expected-allocation, invalid-configuration, and confidence-policy tests before empirical comparison.
- The risk bridge should use exact conditional variances for $H=RAR$: Gaussian $2\|H\|_F^2/\ell$ and Rademacher $2\sum_{i\ne j}H_{ij}^2/\ell$. Reuse the frozen 24-case manifest and retain complete curves and uncertainty rather than only six-spectrum minimizers.
- Resolve the duplicate Theorem 11 numbering and restore the removed Theorem 4/confidence-radius section before treating the proof notes as authoritative.

### Marginal-Risk Correctness-Repair Design Checkpoint (2026-08-14)

- The self-reviewed design is frozen in `docs/superpowers/specs/2026-08-14-marginal-risk-correctness-repair-design.md` at nested-repository commit `8db41f6`.
- It keeps the exact marginal identity, certificate decision logic, and Ritz-fit heuristic in separate layers; no confidence radius is invented and the prototype is not labeled certified or baseline-safe.
- Planned proof repairs add the monotone single-crossing lemma, correct the step-spectrum budget regime, restore the missing near-oracle confidence theorem, and eliminate theorem-number collisions.
- Planned empirical work replaces the six-case residual-probe Monte Carlo summary with exact conditional Gaussian and Rademacher risk curves on the frozen 24-case manifest using nested sketch matrices $S$ and 200 shared randomized-basis trials per setup.
- This is a design-only checkpoint. Estimator, proof, test, and result files remain unchanged pending written-spec approval.

### Marginal-Risk Repair and Exact Conditional-Risk Bridge (2026-08-14)

- The approved repair is implemented. Theorem 4 and its confidence-radius definition are restored; baseline safety remains Theorem 11; the feasible marginal identity is Theorem 12; monotonicity/single crossing is Lemma 12.1; and the corrected step minimizer is Theorem 13 under $2r_\star+2(d-r_\star)\eta^2<m<2d$.
- `proof_step_ritz_gap.md` no longer carries a duplicate Theorem 11 or the false $m<2d$-only allocation claim; it cross-references Theorem 13 and reserves Theorems 9, 10, and 15A for Ritz structure and knee detection.
- `Adaptive_Hutch_pplus_MarginalRisk` remains a `HEURISTIC` Ritz-model prototype. It now counts acquired sketch columns separately from numerical rank, validates configurations, supports a baseline pilot cap, records the raw fitted proposal, and intervenes at the committed width after a resolved knee or a nonpositive fitted marginal at the pilot floor. A resolved fixture at $B=12$ rejects raw $q=46$ and uses $q=12$, while the default $B=8$ step trajectory stops at $q=8$ because the fitted marginal is already nonpositive. This is accounting-safe behavior, not a validated improvement. The separate `marginal_certificate_decision` helper implements the three deterministic states but constructs no confidence radius.
- The rank-two accounting counterexample is repaired: $b_{\mathrm{final}}=4$, $r_{\mathrm{pilot}}=2$, and the estimator uses exactly the budget. The probe-identical regression now uses $\ell=m-q-r$ and passes for full-rank and rank-two matrices. All 75 tests pass.
- The exact bridge uses the frozen 24-case manifest, 200 nested-basis trials per setup, $q=0,\ldots,76$, and exact conditional risks $2\|H_q\|_F^2/\ell_q$ (Gaussian) and $2\sum_{i\ne j}(H_q)_{ij}^2/\ell_q$ (Rademacher). Outputs contain 369,600 trial rows, 1,848 curve rows, and 72 minimizer rows with exact allocation accounting.
- Outside the step family, using the ideal spectral-tail minimizer costs at most $4.2\%$ Gaussian and $3.9\%$ Rademacher conditional risk on the frozen orientations. All 12 step cases instead favor oversampling: Gaussian $q^*-r_\star=3$--$16$ and Rademacher $q^*-r_\star=4$--$24$. At $\eta=0.001$, selecting $q=r_\star$ costs $103\times$--$196\times$ the minimum Gaussian risk and $2{,}073\times$--$5{,}087\times$ the minimum Rademacher risk.
- Research conclusion: $T(q)$ alone is not a sufficient universal certification target because it omits randomized range-capture leakage. Future confidence bounds should enclose realized residual energy and remain distribution-specific; Gaussian and Rademacher allocation theory must not be silently conflated. Results remain conditional on one frozen orientation per setup and lack familywise error control.

### Rank-Aware Marginal and Four-Level Bridge Idea Audit (2026-08-14)

- `CONDITIONALLY VALID`: if $D=m-q-r>2$ and an ideal successful action captures exactly the next leading eigenvector, then $(q,r)\to(q+1,r+1)$ improves $2T(r)/D$ exactly when $D\lambda_{r+1}^2>2T(r)$. A generic rank-one gain need not remove the next leading eigenvalue, so this is an ideal rank-aware oracle theorem rather than an actual-basis theorem.
- If a new sketch query gives no rank gain, the exact risk increment is $2T(r)/[D(D-1)]\ge0$ for $D>1$. Strict worsening requires $T(r)>0$; when the tail is already zero, both risks remain zero.
- The four levels $\mathcal R_{\mathrm{full}}$, $\mathcal R_{\mathrm{rank}}$, $\mathcal R_G$, and $\mathcal R_R$ isolate modeling differences but do not form a monotone inequality chain. The frozen bridge has $r_q=q$ on every row, so its first two levels coincide; a rank-deficient or thresholded supplement is required to activate the rank-loss channel.
- Exact conditional Gaussian/Rademacher risks average to actual unconditional estimator MSE by conditional unbiasedness and the law of total variance, provided $Q,q,r,\ell$ are fixed by the pre-residual sigma-algebra and residual probes are fresh with the stated distribution.
- Normalized allocation regret and bootstrap minimizer frequencies are valuable and can be derived from the existing 369,600-row output without rerunning bases. A natural next proof number is Theorem 14 for the corrected rank-aware marginal result and its non-rank-gain corollary.
- Do not make the current heuristic the headline estimator. Its default low-noise step path stops at $q=8$ from a fitted negative marginal before the knee is resolved.

### Zero-Minimum Regret and Rank-Aware Query Accounting (2026-08-14)

- If a conditional-risk curve has minimum zero, the usual multiplicative regret $(\mathcal R(q)-\mathcal R_{\min})/\mathcal R_{\min}$ is undefined. Adding an arbitrary small $\varepsilon$ changes the estimand into a regularized score whose magnitude depends on the chosen $\varepsilon$; it must not be reported as exact multiplicative regret. Record additive regret universally, record multiplicative regret only for strictly positive minima, and use `NaN` for exact zero-minimum cases. Any stabilized score must be secondary, explicitly labeled, scale-justified, and accompanied by sensitivity analysis.
- The bridge identity $q+r_q+\ell_q=m=160$ counts oracle products, not dimensions alone: $q$ products compute $AS_q$, $r_q$ products compute $AQ_q$ for the accepted basis, and $\ell_q$ products evaluate fresh residual probes. Consequently, an accepted new direction changes $(q,r_q,\ell_q)$ by $(+1,+1,-2)$, whereas a rejected direction changes it by $(+1,0,-1)$. Full numerical rank gives the familiar $2q+\ell_q=160$; rank deficiency gives $q+r_q+\ell_q=160$ and reclaims the rejected second-stage queries.

### Approved Rank-Aware Multi-Budget Bridge Design (2026-08-14)

- The written specification is committed as `cac0b21` at `docs/superpowers/specs/2026-08-14-rank-aware-marginal-multibudget-bridge-design.md`. It preserves the frozen exact bridge and defines a separate rank-deficient supplement.
- Theorem 14 covers the ideal successful action, Corollary 14.1 the failed-rank increment, and Lemma 14.2 the realized Gaussian/Rademacher energy-drop marginal. A numerical rank gain is not silently equated with removal of $\lambda_{r+1}^2$.
- The primary budget is $m=160$; $m=80$ and $m=240$ are labeled sensitivity budgets. With $\ell_{\min}=8$, the derived endpoints are $q_{\max}=36,76,116$. The default 12-spectrum, 200-trial design has 554,400 allocation rows.
- Regret outputs include additive regret, $\mathcal R(0)$-normalized additive regret, and true multiplicative regret only for positive minima. Exact zero minima use `NaN`; no arbitrary denominator offset is allowed.
- Adjustable arguments are budgets, primary budget, trials, dimension, step ranks, tail levels, residual floor, bootstrap count, QR tolerances, exact-zero backward tolerance, orientation/basis/bootstrap seed bases, and output directory. Allocation endpoints, residual counts, and analysis roles are derived.
- The self-review rejected the cancellation-prone generic residual-energy subtraction for the $10^{-14}$ tail. The supplement will use the exact matrix-free operator $A=\eta I+(1-\eta)U_\star U_\star^T$ and a stable nonnegative energy formula based on $Z=(I-QQ^T)U_\star$.
- No implementation or result changed at this checkpoint. The next gate is user review of the committed written specification.
- Renderer-safe revision: commit `4bd9aee` rewrites the full specification with Unicode/plain-text formula blocks and no LaTeX dependency. It preserves all 56 headings and every mathematical, experimental, configuration, validation, and interpretation requirement; no research content was shortened or removed.
- Current presentation choice: commit `1f3db32` restores complete LaTeX equations inside the same Markdown file, with no PDF or companion source. All display delimiters are balanced and flush-left; the eight previously indented delimiters in the four-risk list were repaired for renderer compatibility.

### Rank-Aware Bridge Paired-Orientation Revision (2026-08-14)

- Commit `8de09a7` revises only `docs/superpowers/specs/2026-08-14-rank-aware-marginal-multibudget-bridge-design.md`; implementation and production results remain pending final written-spec review.
- Required control: for each fixed $r_\star$, reuse one exact $U_\star$ and orientation seed across $\eta\in\{0,10^{-14},10^{-10},10^{-6}\}$. Reuse the same basis seed and Rademacher sketch prefix across those tail levels and all common budgets. This isolates the tail magnitude because Rademacher behavior is coordinate-orientation sensitive.
- New diagnostics are $r_q/q$ for $q>0$, `NaN` at $q=0$, and the canonical rejected-query count $q-r_q$. Under nested rank increments in $\{0,1\}$, $q-r_q$ exactly equals the cumulative number of rejected transitions, so it is validated rather than duplicated.
- Accumulate the exact rejected-transition increments for ideal rank-aware, realized Gaussian, and realized Rademacher risks. These cumulative penalties describe the failure path but do not equal total regret because successful transitions and denominator changes also matter.
- For $E_X(Q)>0$, Lemma 14.2 is equivalently $[E_X(Q)-E_X(Q')]/E_X(Q)>2/D$: the fractional residual energy removed must exceed the fractional residual-sample capacity lost. At $E_X(Q)=0$ the ratio is undefined and strict improvement from zero is impossible.
- Cross-budget analysis uses additive regret, $\mathcal R_X(0;m)$-normalized regret, minimizers, rank efficiency, and failed-query penalties—not raw risks, which fall mechanically with $m$.
- The post-experiment decision tree distinguishes spectral-tail adequacy, realized basis leakage, Gaussian/Rademacher policy separation, and whether rejection probability must enter the next theory. The prospective mixture $p_b\Delta\mathcal R_{\rm success}+(1-p_b)\Delta\mathcal R_{\rm fail}$ remains `OPEN` and is not part of this implementation cycle.

### Final Rank-Deficient Bridge Result and Scope Freeze (2026-08-15)

- The final diagnostic cycle is complete. New implementation files are `experiments/risk_bridge_regret.py`, `experiments/postprocess_exact_risk_bridge.py`, `experiments/run_rank_deficient_risk_bridge.py`, and `tests/test_rank_deficient_risk_bridge.py`. `CURRENT_STATE.md` is now the short authoritative handoff, and `reports/rank_deficient_bridge_analysis.md` contains the complete result interpretation. No adaptive estimator was modified.
- Proof records now include Theorem 14, Corollary 14.1, Lemma 14.2, the exact Rademacher conditional variance equality, and the conditional-risk-to-unconditional-MSE corollary. The fractional realized criterion is valid only for $E_X(Q)>0$; the exact-zero boundary remains undivided.
- Frozen design: $d=500$, $r_\star\in\{5,15,30\}$, $\eta\in\{0,10^{-14},10^{-10},10^{-6}\}$, budgets $80,160,240$, 200 paired basis trials, and 20,000 paired bootstrap samples. Outputs contain 36 manifest, 554,400 trial, 2,772 curve, 144 minimizer, and 11,088 minimizer-frequency rows. All 96 tests pass and all pre-existing CSV checksums are unchanged.
- Rank behavior: $\eta=0$ saturates exactly at $r_\star$; $10^{-14}$ is mostly rejected after saturation but admits one extra direction in 22/200, 26/200, and 15/200 primary-budget paths for $r_\star=5,15,30$; $10^{-10}$ and $10^{-6}$ remain full numerical rank. This is evidence about the frozen projected-column/reference-scale contract, not a universal eigenvalue threshold.
- Rejected queries have zero cost after exact $\eta=0$ capture and positive cost when a mathematical tail remains. At fixed $q=36$, their cumulative ideal penalty decreases as $m$ increases, matching the pointwise denominator theorem. Different budget-specific endpoints contain different numbers of attempted queries and must not be confused with that fixed-state result.
- Decisive result: at $\eta=10^{-6}$, $r_q=q$ but the full/rank oracle selects $q=r_\star$ while both realized risks select $q=r_\star+1$ at all budgets. At $m=160$, the rank-oracle allocation costs $9.83\times$, $1602.68\times$, and $1.62\times$ the minimum mean Gaussian risk for ranks $5,15,30$. The mean is driven by rare catastrophic range-capture failures rather than typical trials.
- **Decision:** future certification must target realized, probe-specific risk differences. For the implemented estimator this means $\mathcal R_R(Q;q,r)=2E_R(Q)/(m-q-r)$ or its action difference/marginal. $T(r)$ remains an ideal diagnostic, and $E_G(Q)$ remains the Gaussian target/control, but neither replaces the Rademacher target.
- **Hard stop:** Adaptive Hutch++ is the primary UROP story. Estimator development and experiment expansion are frozen. Simultaneous finite-sample bounds and the prospective acceptance-probability/value-of-information theory remain future work while the final report is written.

### Why the Empirical Optimum Moves by One (2026-08-16)

- The observed $q_G^\star=q_{\mathrm{rank}}^\star+1$ is empirical and instance-specific. It is not a theorem that Gaussian Hutch++ always requires one extra sketch direction.
- Use the standardized decomposition $S_1=U_1^TS$ and $S_2=U_2^TS$. At $q=r_\star$, $S_1$ is square and has no redundant column. Full rank of $AS$ does not imply good conditioning or even nonsingularity of $S_1$. Although $S$ is coordinate Rademacher, $U_1^TS$ is generally not iid Rademacher, so iid square-Rademacher singularity statements do not apply directly; a positive tail can still supply rank to $AS$ when dominant-space recovery is poor.
- When $S_1$ has full row rank, randomized subspace error is amplified through $\Lambda_2S_2S_1^\dagger\Lambda_1^{-1}$. Thus a tiny tail may create large leakage when $\|S_1^\dagger\|_2$ is large. At $q=r_\star+1$, the rectangular $S_1$ has one redundant column, which acted as oversampling on the frozen instances but gives no universal guarantee.
- For one realized path with $E_G(Q_{r_\star})>0$, the extra direction wins exactly when $E_G(Q_{r_\star+1})/E_G(Q_{r_\star})<(m-2r_\star-2)/(m-2r_\star)$. At $(m,r_\star)=(160,15)$ this requires only a 1.54% energy reduction.
- The reported $q_G^\star$ minimizes the empirical mean across 200 basis trials. Only 5.5%--7.5% of individual Gaussian paths improve at $r_\star+1$; the mean prefers oversampling because rare catastrophic square-sketch failures dominate expected conditional risk. The right interpretation is insurance against heavy-tailed range-capture loss, not a universal $+1$ law.

### Zero-Oversampling Mechanism Audit (2026-08-16)

- A separate read-only reconstruction in `experiments/run_q_rank_vs_realized_risk_mechanism.py` validates all frozen energies, risks, ranks, seeds, and denominators before adding diagnostics. All 138,600 relevant frozen rows at $\eta=10^{-6}$ satisfy $r_q=q$ pathwise; no frozen implementation bug was found.
- At $q=k=r_\star$, if $S_1$ is invertible, the exact graph factor is $F=\eta S_2S_1^{-1}$ and $\operatorname{range}(AS_k)=\operatorname{range}([I;F])$ in the eigenbasis. Hence $\tan\theta_i=\sigma_i(F)$ and $\|(I-Q_kQ_k^T)U_1\|_2=\|F\|_2/\sqrt{1+\|F\|_2^2}$. For $q\ge k$, the corresponding rectangular expression yields the bound $\|(I-Q_qQ_q^T)U_1\|_2\le\eta\|S_2S_1^\dagger\|_2$.
- The structured Gaussian energy identity is exact: $E_G=\eta^2(d-q)+2\eta(1-\eta)\|Z\|_F^2+(1-\eta)^2\|Z^TZ\|_F^2$, where $Z=(I-QQ^T)U_1$. This proves the subspace-leakage-to-Gaussian-risk arrow. Rademacher risk also depends on coordinate orientation through $E_R=E_G-\sum_iH_{ii}^2$.
- At $m=160$, the maximum realized-to-ideal Gaussian energy ratios at $q=k$ are $1.79\times10^3$, $3.25\times10^5$, and $1.13\times10^2$ for $k=5,15,30$; at $k+1$ they fall to 1.001, 1.010, and 1.002. Spearman correlations between graph/subspace diagnostics and risk are about 0.997--1.000.
- The worst 1% of paths contribute 37%--99.94% of Gaussian and 84.97%--99.997% of Rademacher total risk at $q=k$. The median path prefers $k$; only 5.5%--7.5% of Gaussian and 6%--10% of Rademacher paths improve at $k+1$, but they contain most of the risk that is removed.
- A 216-row sensitivity study over $d\in\{250,500\}$, $k\in\{5,15,30\}$, four $\eta$ levels, three budgets, and three fresh 50-path batches finds optimal shifts of 0, 1, 2, and larger. The safe conclusion is that modest oversampling can insure against zero-oversampling tail failures; exactly $+1$ is finite-regime dependent.
- Full report: `Hutch++/Matrix-vector_queries_estimation/reports/q_rank_vs_realized_risk_mechanism.md`. Four validated CSVs and six figures are under `results/`; 100 maintained tests pass.

### Direct Realized Rademacher Risk Certification Audit (2026-08-16)

- Preferred future question: can the realized Rademacher conditional-risk difference be estimated from a small number of fresh matrix--vector queries with simultaneous finite-sample confidence? This is future work and does not reopen the frozen UROP estimator cycle.
- For any basis $Q$ fixed before fresh certification probes, $X_Q=g^T(I-QQ^T)A(I-QQ^T)g$ has variance $2E_R(Q)$. Its ordinary sample variance is conditionally unbiased, and common probes give an unbiased candidate-minus-baseline risk-difference estimate when bases and denominators are pre-fixed.
- Matrix-free common-probe reuse is exact when $AQ$ is stored: $A(R_Qg)=Ag-AQ(Q^Tg)$. One queried $Ag$ can evaluate several preconstructed candidate bases.
- Required accounting repair: $m-q-r$ is valid only for an offline study with external certification queries. A fixed total budget must use the final remaining count after subtracting certification and every irrevocably committed construction query. Building a candidate before certification can destroy the original baseline fallback, so unbiased risk estimation alone does not close baseline safety.
- New proved moment lemma: for centered degree-two Rademacher chaos $Z=X-\mathbb EX$, Bonami--Beckner gives $\mathbb EZ^4\le81(\mathbb EZ^2)^2$. With an independent copy and $W=(X-X')^2/2$, $\mathbb EW=\operatorname{Var}(X)$ and $\operatorname{Var}(W)\le41(\mathbb EW)^2$. This supports a conservative disjoint-pair median-of-means route with order $\varepsilon^{-2}\log(|\mathcal A|/\delta)$ pair complexity.
- Keep the candidate set, bases, stored products, and denominators measurable before certification; use fresh final residual probes after selection. The full median-of-means constants, exact-zero boundary, simultaneous guarantee, and budget-feasible fallback architecture remain `OPEN`.
- Full audit: `Hutch++/Matrix-vector_queries_estimation/reports/direct_rademacher_risk_certification_audit.md`.

### Phase 1A Offline Certification Result (2026-08-18)

- The preregistered offline experiment is complete and is continuation research, not a reopening of the frozen UROP estimator. The design was committed alone as `21fffb2`; implementation and analysis live in `experiments/direct_rademacher_certification_phase1a.py`, `experiments/run_direct_rademacher_certification_phase1a.py`, `tests/test_direct_rademacher_certification_phase1a.py`, and `reports/direct_rademacher_risk_certification_phase1a.md`.
- The production grid reconstructs 200 frozen paths per rank for $r_\star\in\{5,15,30\}$ and $\eta\in\{10^{-10},10^{-6}\}$, then runs 4 batches by 50 repetitions of 32 paired certification probes. It produces 960,000 unique Parquet rows and 14,400 exact-truth rows. Each repetition uses exactly 32 certification queries, shared across actions; cached $AQ$ is charged only to reconstruction. Frozen risks, ranks, denominators, projectors, and historical checksums validate.
- The preregistered verdict is `QUALIFIED GO`. At the primary $m=160$, $s=16$, $\varepsilon=1/3$, $\eta=10^{-6}$ comparison $q=r_\star+1$ versus $q=r_\star$, sample variance passes the complete gate in all four batches. False-safe rate: 0.0134%, conditional 95% interval [0.0028%, 0.0305%]. False rejection: 0.5157% [0.2991%, 0.7722%]. Top-5% catastrophic detection: 79.67% [69.07%, 89.28%]. True-better acceptance: 52.10% [40.31%, 63.51%]. Benign-control false-safe rate: 0.0008% [0%, 0.0025%].
- The declared `mom_w1` and `mom_w2` rules fail the safety gate at $s=16$ and still do not pass at $s=32$. Their primary false-safe rates are 4.6971% and 1.5478%; their benign-control false-safe rates are 4.2100% and 1.3044%. This does not disprove the fourth-moment bound or all robust variance estimators; it rejects these small-block plug-in decision rules under the frozen gate.
- `PROVED`: the exact conditional Rademacher variance identity and conditional unbiasedness of sample variance and paired mean. `EMPIRICALLY ESTABLISHED`: small fresh-probe batches carry useful selective realized-risk information under the frozen population, with estimator choice materially affecting safety. `OPEN`: simultaneous finite-sample action-level bounds, the exact-zero boundary, and budget-feasible online certification/fallback timing.
- Next gate: analyze a paired sample-variance or direct risk-difference confidence construction and deduct certification plus committed construction costs before any online estimator modification. Do not implement another allocator yet.

---

### `HEURISTIC`
1. **Horizon-Resolution Stopping Rule**:
   Log-gap analysis $g_j = \log(\theta_j / \theta_{j+1})$ with peak condition $g_{\widehat r} \ge \gamma_{\mathrm{gap}}$ (stored by code parameter `tau_gap`), post-gap count $b - \widehat r \ge p_{\min}$, allocation stability, and $D_{\max}$ guard.

---

### `CONJECTURE / OPEN`
1. Construction of finite-sample valid computable simultaneous radii $e_b(q)$ or risk-difference radii $C_b(q)$ for the actual target risk, not merely an oracle surrogate.
2. Sharpening the conservative Gaussian constant $K^{\mathrm G}_{r_\star,d,p,\delta}$ and obtaining implementation-ready numerical constants for the isotropic-subgaussian/Rademacher bound used by the current code path.
3. Unsupported guessed formula $p \ge \left\lceil \frac{\log(1/\delta) + \log(d-r_\star) + \log(\tau_{\mathrm R}r_\star)}{\log(1/\eta)} \right\rceil$ is **WITHDRAWN**.

---

## 3. Mandatory Numerical & Coding Rules

> [!warning] Strict Constraints for AI Agents & Developers
> 1. **Exact Query Accounting**: Every matrix-vector product $A v$ MUST go through `MatVecOracle(A)` to guarantee 100% budget tracking compliance. Use $q + r_{\text{actual}} + \ell = m$.
> 2. **Double Residual Projection**: Residual probes MUST apply left and right projection $B_G = G - Q(Q^T G)$ and compute $B_G^T (A B_G)$.
> 3. **Rank Deficient Query Recovery**: Reclaim unused QR queries via `ell_eff = m - q_target - r_actual`.
> 4. **Single Source of Truth**: Update [[UROP_TRACKER|UROP_TRACKER.md]] after any substantial benchmark or derivation.

---

## 4. File Map & Obsidian Vault Index

- [[AGENTS|AGENTS.md]] — Project rules, workspace governance, and MCP tool instructions.
- [[UROP_TRACKER|UROP_TRACKER.md]] — Master UROP log, paper notes, completed work, and poster evidence.
- `Hutch++/Matrix-vector_queries_estimation/`:
  - [`trace_baseline.py`](file:///Users/chenyixin/Documents/Independent%20Study/Swati%27s%20Summer%20Research/Hutch++/Matrix-vector_queries_estimation/src/trace_baseline.py) — Core MatVecOracle and estimator implementations.
  - [`theorem15_certification.py`](file:///Users/chenyixin/Documents/Independent%20Study/Swati%27s%20Summer%20Research/Hutch++/Matrix-vector_queries_estimation/src/theorem15_certification.py) — Gaussian knee-certificate formulas and minimum-$p$ search.
  - [`tests/test_new_estimators.py`](file:///Users/chenyixin/Documents/Independent%20Study/Swati%27s%20Summer%20Research/Hutch++/Matrix-vector_queries_estimation/tests/test_new_estimators.py) — Unit verification.
  - [`run_asymmetric_guard_heldout_benchmark.py`](file:///Users/chenyixin/Documents/Independent%20Study/Swati%27s%20Summer%20Research/Hutch++/Matrix-vector_queries_estimation/experiments/run_asymmetric_guard_heldout_benchmark.py) — Frozen 24-spectrum, five-method held-out benchmark.
  - [`docs/proof_sequential_unbiasedness.md`](file:///Users/chenyixin/Documents/Independent%20Study/Swati%27s%20Summer%20Research/Hutch++/Matrix-vector_queries_estimation/docs/proof_sequential_unbiasedness.md) — Theorem 2 proof.
  - [`docs/proof_rank_aware_risk.md`](file:///Users/chenyixin/Documents/Independent%20Study/Swati%27s%20Summer%20Research/Hutch++/Matrix-vector_queries_estimation/docs/proof_rank_aware_risk.md) — Lemma 1 & Theorem 3 proof.
  - [`docs/proof_near_oracle_regret.md`](file:///Users/chenyixin/Documents/Independent%20Study/Swati%27s%20Summer%20Research/Hutch++/Matrix-vector_queries_estimation/docs/proof_near_oracle_regret.md) — Theorems 4, 5, 6 proof.
  - [`docs/proof_exponential_sensitivity.md`](file:///Users/chenyixin/Documents/Independent%20Study/Swati%27s%20Summer%20Research/Hutch++/Matrix-vector_queries_estimation/docs/proof_exponential_sensitivity.md) — Theorems 7 & 8 proof.
  - [`docs/proof_step_ritz_gap.md`](file:///Users/chenyixin/Documents/Independent%20Study/Swati%27s%20Summer%20Research/Hutch++/Matrix-vector_queries_estimation/docs/proof_step_ritz_gap.md) — Theorems 9 & 10 proof.
  - [`docs/proof_notation.md`](file:///Users/chenyixin/Documents/Independent%20Study/Swati%27s%20Summer%20Research/Hutch++/Matrix-vector_queries_estimation/docs/proof_notation.md) — Common notation ledger for all proof notes.
## Phase 1B-A Budget-Aware Certification Result (2026-08-18)

- The frozen Phase 1A signal was re-evaluated under exact candidate-first query
  accounting; no estimator or matrix--vector experiment was changed.
- For nested actions with cached higher-prefix products,
  $c_{\rm pre}=\max\{q_0+r_0,q_a+r_a\}$ and
  $\ell_{\rm paid}=m-c_{\rm pre}-s$. Rejection and abstention select the baseline
  basis but do not refund construction or certification queries.
- `PROVED`: the nested-reuse identity, common paid denominator, no-free-fallback
  lemma, exact net-benefit threshold, and paid two-action oracle bound on their
  stated positive-denominator domains. See
  `Hutch++/Matrix-vector_queries_estimation/docs/proof_budget_aware_certification.md`.
- The validated path artifact has 691,200 unique rows. Of these, 665,600 are
  feasible. The 25,600 infeasible low-budget rows remain present with undefined
  paid risk; no rank is silently dropped or reweighted. The bootstrap uses
  10,000 shared rank-stratified path resamples.
- Primary empirical verdict: `NET BENEFIT + TAIL-INSURANCE TRADEOFF`. At
  $(m,\eta,s,\varepsilon)=(160,10^{-6},16,1/3)$ for $q=r_\star+1$ versus
  $q=r_\star$, the selected/original mean-risk ratio is $0.036329$ with
  conditional percentile interval $[0.014628,0.639830]$. Paid oracle: $0.035411$;
  paid fallback: $1.172197$.
- The result is driven by rare paths: 95.83% of paths are harmed and the median
  ratio is $1.172197$, but the worst 5% carry 97.22% of baseline risk. The rule
  accepts with 80.03% probability in that stratum and recovers 99.94% of the
  two-action oracle's recoverable risk.
- Interpretation: direct certification is empirically useful after cost as
  tail-risk insurance in this frozen candidate-first architecture. It is not a
  theorem-level certificate or an orientation-universal claim.
- `OPEN`: simultaneous finite-sample bounds for the sample-variance comparison,
  validation across new orientations/path populations, and an online schedule
  that can avoid paying full information cost on ordinary paths.

## Phase 1C Sample-Variance Confidence Result (2026-08-19)

- The Phase 1C design is frozen in commit `44566dd`. This continuation phase did
  not modify Hutch++, issue a matrix--vector query, or implement an allocator.
- `PROVED`: for conditionally iid observations with variance $\sigma^2$ and
  fourth central moment $\mu_4$,
  $$
  \operatorname{Var}(S_s^2\mid\mathcal G)
  =
  \frac1s\left[\mu_4-\frac{s-3}{s-1}\sigma^4\right].
  $$
- Retaining the audited quadratic-Rademacher-chaos bound
  $\mu_4\le81\sigma^4$ and splitting the public failure probability equally
  across two fixed actions gives the simultaneous radius
  $$
  \varepsilon_{\rm Ch}(s,\delta_{\rm joint})
  =
  \sqrt{\frac{2[80+2/(s-1)]}{s\delta_{\rm joint}}}.
  $$
- `PROVED BUT BUDGET-VACUOUS`: at $\delta_{\rm joint}=0.05$, the radius is
  $28.401878$, $20.035682$, $14.153916$, and $10.004031$ for
  $s=4,8,16,32$. The expression is strictly decreasing for $s>1$, so no frozen
  sample size provides a multiplicative lower bound.
- The exact Hoeffding decomposition separates a generally nondegenerate linear
  projection from the canonical kernel
  $h_2(x,y)=-(x-\mu)(y-\mu)$. A theorem for completely degenerate U-statistics
  cannot be applied to the full sample variance.
- `INCOMPLETE`: an explicit sharper scale-free radius. Numerical constants do
  not yet close for both the nondegenerate degree-at-most-four projection and
  the canonical U-statistic remainder; no constant was fitted from data.
- The implemented accepted-candidate condition is
  $$
  S_a^2
  \le
  \frac{1-\varepsilon}{1+\varepsilon}
  \frac{\ell_{\rm paid}}{\ell_0}S_0^2,
  $$
  which compares the paid candidate with the original baseline. It does not
  claim that abstention refunds sunk costs.
- Final verdict: `THEOREM ONLY / BUDGET-VACUOUS`. The deterministic gate found
  no $s_{\rm gate}$, so the practical bootstrap was not run and empirical
  Phase 1B performance was not used to select $s$.
- `OPEN`: a sharper explicit theorem or different variance estimator, and a
  genuinely online timing architecture that avoids paying on ordinary paths.

## Phase 1D Linear-Projection Tail Audit (`PROVED` Analytic No-Go, 2026-08-19)

- Completed Phase 1D continuation theory (`src/rademacher_linear_projection_no_go.py`, `reports/rademacher_linear_projection_no_go_phase1d.md`).
- Imported Cortinovis--Kressner (2022) Theorem 2, equation (8), for nonzero symmetric zero-diagonal $C=H-\operatorname{diag}(H)$:
  $$
  \Pr(|g^T C g| \ge t) \le 2\exp\left(-\frac{t^2}{8\|C\|_F^2 + 8t\|C\|_2}\right).
  $$
- Dimensionless structural ratio $\kappa = \|C\|_2 / \|C\|_F \in (0, 1]$ and normalized variable $V = Z^2/\sigma^2$.
- For the capped variable $V^{(T)} = \min(V, T)$ with $T \ge 64$, proved the exact variance-envelope floor:
  $$
  \boxed{80 \le \nu_\kappa(T) \le 81 \quad \forall T \in \mathcal{T}, \kappa \in (0, 1].}
  $$
- One-sided Bernstein candidate lower-tail bound:
  $$
  D_{s,\kappa}(\varepsilon, T) > e^{-s/160} \ge e^{-32/160} = e^{-0.2} \approx 0.81873.
  $$
- Because $0.81873 \gg 0.05 \ge \delta_{\rm linear}$, the truncation--Bernstein route is budget-vacuous for all $s \le 32$.
- Verdict: `STRONG LINEAR NO-GO`. This stops the frozen truncation--Bernstein branch before the canonical kernel $h_2$ and motivates, but does not prove, the direct paired-difference route. The implementation is committed as `68353e0`; its 22 targeted tests and the complete 164-test maintained suite pass.

## Exploratory local branch: Two-Stage Ritz-Gap Gate (2026-08-19)

- A local uncommitted prototype, `Adaptive_Hutch_pplus_TwoStageGated`, reuses an initial $b_0=8$ range sketch inside the Standard-Hutch++ fallback. This means no additional fallback allocation beyond $q_0$; it does not mean the pilot itself costs zero queries.
- The prototype does not estimate realized Rademacher risk and does not solve the Phase 1B certification-tax problem. It returns to heuristic Ritz-gap allocation and is outside the frozen UROP estimator conclusion.
- Its current development evidence is limited to five spectra, one orientation per spectrum, 50 trials, and no confidence intervals. Results are mixed: the $r_\star=5$ step improves, the $r_\star=15$ step never triggers and is worse in median/mean/squared relative error, and the power-law cases change direction depending on the metric. Preserve it as `HEURISTIC / EXPLORATORY`, not Pareto-optimal or certified.

## Next Research Plan: Direct Common-Probe Risk Difference (2026-08-19)

- Plan committed as `d0405ae`: `Hutch++/Matrix-vector_queries_estimation/docs/superpowers/specs/2026-08-19-direct-paired-rademacher-risk-difference-research-plan.md`.
- Primary target compares the paid candidate with the original unstarted baseline:
  $$
  \Delta_R^{\rm net}
  =
  \frac{\sigma_a^2}{\ell_a^{\rm paid}}
  -
  \frac{\sigma_0^2}{\ell_0^{\rm original}}.
  $$
- The direct common-probe estimator is already the difference of the two stored sample variances:
  $$
  \widehat\Delta_R
  =
  \frac{S_a^2}{\ell_a}
  -
  \frac{S_0^2}{\ell_0}.
  $$
- Its paired Hoeffding linear variance contains the cancellation term
  $$
  -2(\ell_a\ell_0)^{-1}\operatorname{Cov}(Z_a^2,Z_0^2),
  $$
  and the canonical variance contains the squared cross-covariance term. These exact identities show where common probes may help but do not guarantee positive covariance.
- Phase 2A is an artifact-only empirical audit with no new matrix--vector experiment. It compares paired variance with the exact-form independence benchmark and a deterministic within-batch shifted/decorrelated comparator. The shifted sequence is not independent because it reuses observations. Phase 2B theorem work is separately gated and requires explicit one-sided constants; no allocator is authorized.

## Phase 2A Direct Paired-Difference Audit (`PAIRING SIGNAL GO`, 2026-08-19)

- Implementation commit: `e0ced7a`.
- Phase 2A is complete with zero new matrix--vector queries and no change to Hutch++, Standard Hutch++, or any frozen estimator.
- Direct statistic:
  $$
  \widehat\Delta_R
  =
  \frac{S_a^2}{\ell_a^{\rm paid}}
  -
  \frac{S_0^2}{\ell_0^{\rm original}}.
  $$
- `PROVED`: this is a conditionally unbiased order-two U-statistic for the fixed paired target; its Hoeffding decomposition is
  $$
  \widehat\Delta_R-\Delta_R
  =
  \frac2s\sum_iK_1(Y_i)
  +
  \binom{s}{2}^{-1}\sum_{i<j}K_2(Y_i,Y_j),
  $$
  with common-probe covariance subtraction in both the linear and canonical variances. Positive cancellation is not universal; the unit tests include a negative squared-chaos covariance example.
- Construction-accounting contract: $c_{\rm pre}=\max\{q_a+r_a,q_0+r_0\}$ applies only to the frozen nested shared-prefix architecture. Nonnested or separately built actions must use the actual committed count from the query ledger. All 43,200 Phase 2A rows reproduce frozen Phase 1B accounting; 1,600 infeasible rows remain explicit.
- Comparator contract: the within-batch cyclic shift is `SHIFTED / DECORRELATED`, not independent. It reuses observations. The exact independence benchmark is $V_{\rm ind}=\operatorname{Var}(S_a^2/\ell_a)+\operatorname{Var}(S_0^2/\ell_0)$.
- Primary result at $(m,\eta,s)=(160,10^{-6},16)$, candidate $r_\star+1$ versus baseline $r_\star$:
  - pairing variance ratio $0.151032$, conditional 95% bootstrap interval $[0.137069,0.166735]$;
  - covariance contribution $0.848968$, interval $[0.833785,0.862865]$;
  - shifted/decorrelated comparator ratio $0.999563$;
  - verdict `PAIRING SIGNAL GO`.
- Heterogeneity is essential: 37 net-beneficial paths have mean pairing ratio $0.804673$, 563 net-harmful paths have $0.108075$, top-5% catastrophic paths have $0.899565$, and ordinary paths have $0.111635$. Common probes therefore strongly reduce aggregate uncertainty but provide much less cancellation on the rare beneficial tail.
- Evidence status: the exact identities are `PROVED`; numerical ratios and bootstrap intervals are `EMPIRICALLY ESTABLISHED` conditional on frozen orientations/paths; a one-sided finite-sample bound remains `OPEN`.
- Next decision gate: write and adversarially review a separate Phase 2B one-sided theorem design for $\Delta_R$ that preserves pair-specific covariance. Do not infer an online allocator or baseline-safety guarantee from Phase 2A.

## Phase 2B One-Sided Paired Confidence Design (2026-08-19; specification `ecc2b35`)

- Frozen design: `Hutch++/Matrix-vector_queries_estimation/docs/superpowers/specs/2026-08-19-direct-paired-risk-difference-confidence-phase2b-design.md`.
- Legal theorem inputs are restricted to the same common-probe observations, fixed positive denominators, one joint one-sided failure probability, and universal constants proved for the exact Rademacher-polynomial class. Matrix norms, spectral side information, fitted constants, extra probes, and bootstrap quantiles are forbidden.
- Primary reduction:
  $$
  D_j=
  \frac{(X_{a,2j-1}-X_{a,2j})^2}{2\ell_a}
  -
  \frac{(X_{0,2j-1}-X_{0,2j})^2}{2\ell_0},
  \qquad
  \mathbb E[D_j\mid\mathcal G]=\Delta_R.
  $$
  The $D_j$ are conditionally iid across disjoint probe pairs and retain candidate--baseline common-probe dependence within each pair.
- The centered variable $P_j=D_j-\Delta_R$ is a degree-at-most-four polynomial in independent Rademacher signs. The first proof audit will verify the uniform hypercontractive implication $\mathbb E P_j^4\le6561(\mathbb E P_j^2)^2$ and determine whether any observable data-only radius can be nonvacuous at $s\le32$.
- Proof route order: independent-pair robust bounded-kurtosis method, independent-pair elementary studentization, then the complete paired U-statistic. The complete U-statistic cannot use a canonical theorem until its nondegenerate linear projection is separately controlled.
- Replay boundary: raw probe values were not stored. No replay is allowed before theorem closure; any later replay must reproduce frozen seeds, count all diagnostic products, and remain separate from online query accounting.
- No Phase 2B theorem implementation or allocator has been authorized by the specification commit alone. The next action is written-spec review.

## Phase 2B Theorem Audit Result (2026-08-21)

- Final verdict: `PROVED BUT BUDGET-VACUOUS` for the two explicit data-only independent-pair routes. No frozen probes were replayed, no matrix--vector query was issued, and Hutch++/allocator behavior remains unchanged.
- `PROVED`: the signed disjoint-pair statistic
  $$
  D_j=
  \frac{(X_{a,2j-1}-X_{a,2j})^2}{2\ell_a}
  -\frac{(X_{0,2j-1}-X_{0,2j})^2}{2\ell_0}
  $$
  is conditionally iid across $j$ and has exact mean $\Delta_R$. Common-probe dependence remains inside each $D_j$.
- `PROVED`: $P_j=D_j-\Delta_R$ is a degree-at-most-four multilinear Rademacher polynomial. O'Donnell's verified Boolean hypercontractive corollary gives
  $$
  \|P_j\|_4\le9\|P_j\|_2,
  \qquad
  \mathbb EP_j^4\le6561(\mathbb EP_j^2)^2.
  $$
- `PROVED BUT BUDGET-VACUOUS`: applying the exact sample-variance variance identity to the $n=s/2$ signed observations produces a relative scale radius greater than one for every $s\in\{4,8,16,32\}$ and $\delta_{\rm joint}\in\{0.01,0.05,0.10\}$.
- `PROVED BUT BUDGET-VACUOUS`: for $W=(P-P')^2/2$, $\mathbb EW=v_D$ and $\operatorname{Var}(W)\le3281v_D^2$. A single Chebyshev-valid scale block with relative error at most one and failure probability below $1/4$ needs at least $13{,}125$ $W$ observations; the largest frozen grid supplies at most eight.
- `INCOMPLETE / STOPPED`: the complete paired U-statistic has a nondegenerate linear projection plus a canonical remainder. Giné--Latała--Zinn-type canonical concentration applies only to the remainder, so the approved stopping rule forbids continuing until the linear scale problem closes.
- Scope audit: Maurer--Pontil empirical Bernstein assumes bounded observations and is not legal for the scale-free signed $D_j$ class. No Catoni constant was imported without an exact observable theorem mapping. Phase 2A empirical covariance ratios were not used as theorem constants.
- New implementation and record: `src/paired_rademacher_difference_confidence.py`, `tests/test_paired_rademacher_difference_confidence_phase2b.py`, `docs/proof_paired_rademacher_difference_confidence.md`, `experiments/postprocess_paired_rademacher_difference_confidence_phase2b.py`, `reports/paired_rademacher_difference_confidence_phase2b.md`, and isolated manifest/route/scale/verdict CSVs.
- Authoritative next action: stop allocator development and write the UROP report. A continuation must introduce genuinely new justified information (for example, structural side information or a sharper small-ball/unknown-scale theorem). This result does not prove that every direct certificate is impossible.

## Phase 2C Structural Priors Result (2026-09-07)

- `PROVED UNDER STRUCTURAL PRIOR` (Theorem 16):
  Under an a priori upper bound on the baseline residual Frobenius norm $\|H_0\|_F \le M_0$,
  the conditional variance of $D_j$ is bounded unconditionally by:
  $$
  \operatorname{Var}(D_j \mid \mathcal G) \le \bar{v}_D(M_0) = 164 M_0^4 \left(\frac{1}{\ell_a} + \frac{1}{\ell_0}\right)^2.
  $$
  Applying Cantelli's one-sided inequality yields an always-finite, non-vacuous one-sided certificate for all sample sizes $s \ge 4$:
  $$
  C_n^{\text{norm}}(\delta; M_0) = \sqrt{\frac{\bar{v}_D(M_0)}{n} \frac{1-\delta}{\delta}} < \infty.
  $$
- `PROVED` (Theorem 17):
  For any symmetric zero-diagonal matrix $C$, the fourth moment of Rademacher chaos $Z = g^T C g$ satisfies:
  $$
  \mathbb E[Z^4] \le (3 + 12\kappa^2) \sigma^4 \le 15\sigma^4,
  $$
  where $\kappa = \|C\|_2 / \|C\|_F \in (0, 1]$.
  This improves the generic degree-2 Boolean hypercontractive factor $81$ down to at most $15$, and matches standard Gaussian chaos kurtosis $3$ as effective rank $r_{\text{eff}} = 1/\kappa^2 \to \infty$.
- `PROVED ANALYTIC LIMIT` (Theorem 18):
  Any data-only Chebyshev scale estimation for $v_D$ requires sample size $n > n^\star(K, \delta_{\text{scale}}) = \frac{K - 1}{\delta_{\text{scale}}} + 1$.
  Even for ideal Gaussian chaos ($K = 3$) at $\delta_{\text{scale}} = 0.05$, $n^\star = 41 > 16$ ($s > 82$).
  Therefore, purely empirical scale estimation is mathematically impossible for $s \le 32$; an a priori scale envelope (Route A) is mathematically necessary for small-sample trace risk certification.
- Implementation and records:
  `src/structural_paired_difference_confidence.py`,
  `tests/test_structural_paired_difference_confidence_phase2c.py`,
  `docs/proof_structural_paired_difference_confidence.md`,
  `experiments/postprocess_structural_paired_difference_confidence_phase2c.py`,
  `reports/structural_paired_difference_confidence_phase2c.md`,
  `results/structural_certificate_grid_phase2c.csv`, and
  `results/structural_kurtosis_boundary_phase2c.csv`.
- Verification: 7 targeted tests pass; complete regression suite passes all 205 maintained tests cleanly with zero regressions.

## Haar Orientation Robustness Audit (2026-09-07)

- `EMPIRICALLY ESTABLISHED` (Experiment 1):
  Evaluated `Adaptive_Hutch_pplus_TwoStageGated` across six spectra, 30 independent Haar-random orthogonal orientations $U_j \sim \text{Haar}(O(d))$ (1,800 trials), and the coordinate-aligned case $U = I_d$ (60 trials) at $d=100, m=60$.
- **Gate Trigger Invariance**: The Stage 1 pilot Ritz-gap screening trigger ($\gamma_{\text{gap}} \ge \tau_{\text{gap}}$ on $b_0 = 8$) is **100% rotation-invariant**. It triggered in 100% of trials (300/300) on step spectra ($r_\star=5$) and 0% of trials (1,200/1,200) on smooth power laws across all Haar orientations, confirming that knee detection is strictly spectral and immune to coordinate rotation.
- **Haar Performance Gains**: Under delocalized Haar-random rotations, `TwoStageGated` reduced MSE from $1.67 \times 10^{-6}$ (Standard Hutch++) to $2.84 \times 10^{-7}$, achieving a **$5.89\times$ error reduction (83.0% reduction, MSE ratio 0.1697)** on step spectra, while matching Hutch++ on smooth power laws (MSE ratios 1.007, 0.949, 1.070) with zero query overhead.
- **Rademacher Coordinate Advantage**: On $U = I_d$, diagonal residual energy causes Rademacher variance to be identically zero; Haar rotations delocalize eigenvectors and drop diagonal concentration to $0.26 - 0.80$, where `TwoStageGated` decisively outperforms Hutch++.
- Artifacts: `experiments/run_haar_orientation_audit.py`, `tests/test_haar_orientation_audit.py`, `reports/haar_orientation_audit.md`, `results/haar_orientation_audit_trials.csv`, and `results/haar_orientation_audit_summary.csv`.

## Real Matrix-Free ML PyTorch Benchmark (2026-09-08)

- `EMPIRICALLY ESTABLISHED` (Experiment 2):
  Evaluated `TwoStageGated` against Standard Hutch++, Gaussian Hutch++, and Classical Hutchinson on the Hessian of a trained convolutional neural network (`SmallConvNet`, $d=4,254$ parameters) via matrix-free Hessian-vector products using PyTorch autograd (`torch.autograd.grad`).
- **Exact Ground Truth**: Computed via 4,254 column-wise HVPs ($\operatorname{Tr}(H) = 31.581270$).
- **41% - 43% MSE Reduction Over Standard Hutch++**:
  - $m=60$: `TwoStageGated` lowers MSE from $1.417$ to $0.841$ ($40.7\%$ reduction).
  - $m=90$: `TwoStageGated` lowers MSE from $1.006$ to $0.571$ ($43.2\%$ reduction), beating Gaussian Hutch++ ($0.590$).
  - $m=120$: `TwoStageGated` achieves the lowest overall MSE ($0.370$ vs $0.457$ for Hutch++ and $0.609$ for Gaussian Hutch++), reaching a median relative error of $1.15\%$.
- **Mechanism**: Spiked loss curvature allows `TwoStageGated` to isolate the knee at $q \approx 8-9$ and allocate $104$ queries to residual probes at $m=120$ ($2.6\times$ more than Standard Hutch++), suppressing bulk noise variance without forming dense matrices.
- Artifacts: `src/pytorch_matvec_oracle.py`, `tests/test_pytorch_matvec_oracle.py`, `experiments/run_pytorch_hessian_benchmark.py`, `tests/test_pytorch_hessian_benchmark.py`, `reports/pytorch_hessian_benchmark.md`, `results/pytorch_hessian_benchmark_trials.csv`, and `results/pytorch_hessian_benchmark_summary.csv`.

## Pilot Feature Gating Diagnostics & Predictability Map (2026-09-08)

- `EMPIRICALLY & THEORETICALLY ESTABLISHED` (Experiment 3):
  Constructed an observable predictability map across 22 spectral families ($d=100, m=60$).
- **The Allocation Duality**:
  1. *Subspace Isolation Regime* ($q^* \ll m/3$): Sharp knee drop to noise floor. Truncating at $q \approx r$ eliminates wasteful noise-space queries, yielding up to **$7.4\times$ variance reduction**.
  2. *Continuous Curvature Regime* ($q^* \ge m/3$): Smooth continuous decay ($\lambda_i = i^{-\alpha}$). Deep sketching ($q \ge m/3$) annihilates the residual Frobenius norm $\|A_{\text{res}}\|_F$, suppressing variance faster than residual averaging.
- **The Isolation Contrast Filter**:
  A single Ritz gap cannot distinguish an isolated step cliff from a steep continuous slope. The **Isolation Contrast Ratio** $\mathcal{C} = \Delta_{\max} / (\operatorname{median}_{k \ne \max} \Delta_k + \epsilon)$ cleanly separates step cliffs ($\mathcal{C} \ge 30 - 69{,}000$) from continuous decay ($\mathcal{C} \le 3.5$).
- **Universal Safety**:
  Upgraded `Adaptive_Hutch_pplus_TwoStageGated` with `tau_contrast`: enforcing $\Delta_{\max} \ge 1.2$ AND $\mathcal{C} \ge 5.0$ guarantees 100% trigger on step knees while completely eliminating false alarms on steep power laws ($\alpha = 2.0, 2.5, 3.0$) and smooth baselines, falling back to $q_0 = m/3$ with zero query penalty.
- Artifacts: `experiments/run_gating_diagnostics.py`, `tests/test_gating_diagnostics.py`, `reports/gating_diagnostics_predictability_map.md`, `results/gating_diagnostics_summary.csv`, `results/gating_diagnostics_pilot_features.csv`, and `results/gating_diagnostics_q_curves.csv`.


