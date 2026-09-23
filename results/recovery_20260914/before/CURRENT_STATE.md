# Current State: Adaptive Hutch++ UROP

**Last updated:** 2026-08-21

## Current research question

What realized risk information is sufficient to justify spending another matrix--vector query on low-rank capture rather than residual estimation?

## Authoritative accounting and notation

The exact allocation identity is

$$
q+r_{\mathrm{actual}}+\ell=m.
$$

Uppercase $S$ is the sketch matrix, $Q\in\mathbb R^{d\times r}$ is the accepted basis, and $R=I-QQ^T$. The formula $m-2q$ is used only in the full-rank special case $r=q$.

## Proved mathematical layers

1. Sequential allocation and stopping preserve unbiasedness when the action is fixed before fresh independent residual probes.
2. Exact conditional Gaussian risk is $2\|RAR\|_F^2/(m-q-r)$.
3. Exact conditional Rademacher risk is $2\sum_{i\ne j}(RAR)_{ij}^2/(m-q-r)$.
4. Exact conditional risk averages to unconditional estimator MSE under conditional unbiasedness.
5. The ideal rank-aware successful marginal is beneficial exactly when $(m-q-r)\lambda_{r+1}^2>2T(r)$.
6. A rejected sketch direction has nonnegative exact increment $2T(r)/[D(D-1)]$; equality is possible when the residual risk is already zero.
7. For the realized basis, a successful direction is beneficial exactly when

$$
\frac{E_X(Q)-E_X(Q')}{E_X(Q)}>\frac{2}{D},
\qquad E_X(Q)>0.
$$

8. Baseline-safe acceptance is proved conditional on valid simultaneous risk bounds. Constructing those computable finite-sample bounds remains open.

## Main empirical conclusion

The ideal spectral tail $T(q)$ is not a universal allocation target. On the frozen step spectra, randomized range-capture leakage shifts realized Gaussian and Rademacher optima beyond the ideal knee and can make the ideal allocation orders of magnitude worse than the realized optimum.

The strongest final-project claim is therefore not that adaptive Hutch++ uniformly beats Standard Hutch++. It is that adaptive allocation is limited by distinct mechanisms: pilot commitment, allocation decision error, randomized subspace capture, probe-distribution-specific residual risk, and potentially numerical rank loss.

## Completed final experiment

The frozen rank-deficient multi-budget four-risk bridge is complete:

$$
\mathcal R_{\mathrm{full}},\qquad
\mathcal R_{\mathrm{rank}},\qquad
\mathcal R_G,\qquad
\mathcal R_R.
$$

It uses $r_\star\in\{5,15,30\}$, $\eta\in\{0,10^{-14},10^{-10},10^{-6}\}$, and $m\in\{80,160,240\}$. At fixed $r_\star$, the same $U_\star$, basis seeds, and Rademacher sketch prefixes are reused across every $\eta$.

The validated production outputs contain 554,400 trial-allocation rows. Exact and below-tolerance cases activate numerical rank loss. The $\eta=10^{-6}$ controls remain numerically full rank but still show that rare randomized range-capture failures can make $q=r_\star$ much worse in mean conditional risk than $q=r_\star+1$.

The observed $q_G^\star=q_{\mathrm{rank}}^\star+1$ is instance-specific. At $q=r_\star$, the projected dominant block $S_1=U_1^TS\in\mathbb R^{r_\star\times r_\star}$ can be poorly conditioned even though the complete sampled range has rank $r_\star$. Although $S$ is coordinate Rademacher, $S_1$ is generally not iid Rademacher. The extra column makes $S_1$ rectangular and acts as oversampling against rare range-capture failures; it does not establish a universal $+1$ rule.

No adaptive estimator was retuned or modified during this cycle.

## Zero-oversampling mechanism audit

The focused reconstruction found no frozen implementation or accounting bug. At $q=k=r_\star$, the exact graph factor for the step model is

$$
F=\eta S_2S_1^{-1},
$$

and it determines the principal-angle error exactly when $S_1$ is invertible. The Gaussian residual energy then satisfies

$$
E_G(Q)=\eta^2(d-q)+2\eta(1-\eta)\|Z\|_F^2+(1-\eta)^2\|Z^TZ\|_F^2,
\qquad Z=(I-QQ^T)U_1.
$$

In the frozen 200-path experiment, the graph factor, subspace error, and risk are almost perfectly rank-correlated. The worst 1% of paths contribute 37%--99.94% of Gaussian and 84.97%--99.997% of Rademacher total risk at $q=k$. The median path prefers $k$; the empirical mean prefers $k+1$ because the nested extra column suppresses those extreme paths.

A broader sensitivity grid finds optimal shifts of 0, 1, 2, and larger. The authoritative interpretation is therefore: **positive oversampling can provide tail-risk insurance at a zero-oversampling boundary, but the useful amount is not universally one.** See `reports/q_rank_vs_realized_risk_mechanism.md`.

## Decision reached

The rank-aware oracle and realized conditional risks differ materially in the $\eta=10^{-6}$ controls even though $r_q=q$. Therefore $T(r)$ is not a sufficient certification target. Future confidence work should target realized, probe-specific risk differences. For the practical Rademacher estimator, the primary object is

$$
\mathcal R_R(Q;q,r)=\frac{2E_R(Q)}{m-q-r}.
$$

$E_G(Q)$ remains the Gaussian target and a useful control. A later theory may model the next-direction acceptance probability $p_b$, but it is not part of the UROP implementation scope.

## Hard scope boundary

After this bridge, freeze estimator development and write the UROP report. Finite-sample simultaneous confidence construction and expected value-of-information are future work unless a separate project explicitly reopens them.

## Phase 2B direct paired-confidence result

The approved data-only theorem audit is complete with verdict
`PROVED BUT BUDGET-VACUOUS`.  For disjoint common-probe pairs,

$$
D_j=
\frac{(X_{a,2j-1}-X_{a,2j})^2}{2\ell_a}
-\frac{(X_{0,2j-1}-X_{0,2j})^2}{2\ell_0}
$$

is conditionally iid across $j$ and satisfies
$\mathbb E[D_j\mid\mathcal G]=\Delta_R$.  Its centered version is a
degree-at-most-four Rademacher polynomial.  The verified Boolean
hypercontractive theorem gives the uniform moment factor

$$
\mathbb E(D_j-\Delta_R)^4
\le6561\,\operatorname{Var}(D_j)^2.
$$

This worst-case factor makes both explicit unknown-scale routes vacuous on
$s\in\{4,8,16,32\}$.  The elementary sample-variance scale radius remains
greater than one everywhere.  For the self-contained pairwise-scale MoM
baseline, a single scale block with relative error at most one and failure
probability below $1/4$ requires at least $13{,}125$ scale observations; the
largest frozen budget provides at most eight.

The complete paired U-statistic was stopped before canonical-kernel work:
canonical U-statistic concentration controls only its degenerate term, while
the necessary nondegenerate linear projection still lacks an observable
small-sample scale bound.  No frozen probes were replayed, no matrix--vector
queries were issued, and no allocator or estimator was modified.

This is route-specific, not an impossibility theorem.  The current UROP action
is to write the final report.  Any continuation would need genuinely new
mathematical information—such as justified structural side information or a
sharper small-ball/unknown-scale theorem—rather than another allocator or a
retuned empirical threshold.  See
`reports/paired_rademacher_difference_confidence_phase2b.md`.

## Phase 1A continuation result

The separate, preregistered offline certification feasibility study is complete. It did not modify the frozen estimator or deduct its external diagnostic probes from the frozen risk denominator. For each already-constructed action, it estimates

$$
\sigma_Q^2=\operatorname{Var}_g(g^TR_QAR_Qg\mid Q)
$$

from fresh Rademacher probes and compares $\widehat{\sigma}_Q^2/(m-q-r)$ across adjacent frozen actions using common probes.

The production artifact contains exactly 960,000 certification rows and 14,400 exact-truth rows. Each repetition uses exactly 32 new $Ag$ products, reused across every action; all $AQ$ products are cached during reconstruction. Historical bridge artifacts remain unchanged.

The preregistered verdict is **`QUALIFIED GO`**. At $m=160$, $s=16$, $\varepsilon=1/3$, $\eta=10^{-6}$, and candidate $q=r_\star+1$ versus baseline $q=r_\star$, ordinary sample variance passes every safety, power, catastrophic-detection, benign-control, confidence-interval, and batch-stability gate. Its false-safe rate is 0.0134% with a conditional 95% percentile interval [0.0028%, 0.0305%]; false rejection is 0.5157% [0.2991%, 0.7722%]; catastrophic top-5% detection is 79.67% [69.07%, 89.28%]; and true-better acceptance is 52.10% [40.31%, 63.51%].

The declared small-block `mom_w1` and `mom_w2` plug-in rules do not pass: their primary false-safe rates are 4.6971% and 1.5478%, and both fail the benign-control safety gate. Thus fresh probes contain useful realized-risk information under the frozen population, but the empirical estimator/decision construction matters.

This result is `EMPIRICALLY ESTABLISHED`, not a theorem-level certificate. Phase 1B should investigate a simultaneous finite-sample bound for a paired sample-variance or risk-difference statistic and then charge certification and committed construction queries in a physically feasible online architecture. No new allocator should be implemented before those two issues are resolved. See `reports/direct_rademacher_risk_certification_phase1a.md`.

## Phase 1B-A budget-aware emulation result

The approved candidate-first budget emulation is complete. It reuses the frozen
Phase 1A paths and certification observations, but now charges every committed
construction query and all $s$ certification queries before assigning the final
residual budget. For nested adjacent prefixes with cached $AQ$, the proved
accounting is

$$
c_{\rm pre}=\max\{q_0+r_0,q_a+r_a\},
\qquad
\ell_{\rm paid}=m-c_{\rm pre}-s.
$$

The no-free-fallback theorem shows that returning to the baseline after paying
cannot restore its original denominator. On the positive-risk domain, the paid
candidate beats the original baseline exactly when

$$
\frac{\sigma_a^2}{\sigma_0^2}
<
\frac{\ell_{\rm paid}}{\ell_0}.
$$

The production postprocessor validates 691,200 unique path/configuration rows
and 10,000 shared rank-stratified cluster-bootstrap replicates. It retains
25,600 low-budget rows with nonpositive paid capacity as explicitly infeasible;
they are not regularized, deleted, or used to renormalize rank averages.

At the primary $m=160$, $\eta=10^{-6}$, $s=16$, $\varepsilon=1/3$ comparison,
the conditional empirical verdict is **`NET BENEFIT + TAIL-INSURANCE
TRADEOFF`**. The selected/original mean-risk ratio is $0.036329$ with percentile
interval $[0.014628,0.639830]$; the paid two-action oracle is $0.035411$, while
always falling back after paying is $1.172197$. The selector captures 99.92% of
the oracle's available mean-risk improvement.

This is not a typical-path gain. The median path ratio is $1.172197$, and 95.83%
of frozen paths are harmed after paying. The worst 5% of baseline paths carry
97.22% of original mean risk; within that stratum the rule accepts with 80.03%
probability and recovers 99.94% of the risk recoverable by the paid oracle. The
approved interpretation is therefore: **direct realized-risk certification can
pay for itself as insurance against rare catastrophic range-capture paths in
this frozen candidate-first architecture, while imposing a real cost on most
ordinary paths.**

The result is conditional on the frozen orientations and paths and is not a
finite-sample safety certificate. The next mathematical target is a simultaneous
confidence theorem for the successful sample-variance comparison. The next
architectural target is a genuinely online schedule that avoids paying full
candidate/certification cost on ordinary paths. Do not implement another
allocator before those two questions are resolved. See
`reports/direct_rademacher_risk_certification_phase1b_budget.md` and
`docs/proof_budget_aware_certification.md`.

## Narrow future-project proposal and Phase 1B gate

The preferred continuation is direct certification of the realized Rademacher conditional-risk difference from fresh quadratic forms. For a pre-certification basis $Q$, the ordinary sample variance of $g^T(I-QQ^T)A(I-QQ^T)g$ is conditionally unbiased for $2E_R(Q)$. A paired candidate-versus-baseline estimate using common fresh probes is therefore conditionally unbiased for the corresponding risk difference.

This does not yet close baseline safety. If certification probes are charged to the same total budget, the final residual count must deduct them and every irrevocably committed construction query; constructing a candidate before certifying it may also destroy the original fallback allocation. The first approved research step is therefore an offline feasibility study, followed by a separately specified budget-aware timing design.

A new fourth-moment calculation gives a possible theorem route. For the centered degree-two Rademacher chaos $Z=X-\mathbb EX$, hypercontractivity implies $\mathbb EZ^4\le81(\mathbb EZ^2)^2$. For the independent-pair statistic $W=(X-X')^2/2$, this yields $\mathbb EW=\operatorname{Var}(X)$ and $\operatorname{Var}(W)\le41(\mathbb EW)^2$. Phase 1A shows, however, that the two declared very-small-block MoM plug-in rules are empirically too unsafe, whereas ordinary sample variance passes. The next theorem audit must therefore compare a paired sample-variance concentration route against more conservative robust constructions rather than assuming the initial MoM plug-in is adequate. Full constants, the exact-zero boundary, simultaneous action control, a feasible action filtration, and end-to-end query accounting remain open. See `reports/direct_rademacher_risk_certification_audit.md` and `reports/direct_rademacher_risk_certification_phase1a.md`.

## Phase 1C sample-variance theorem result

Phase 1C is complete as continuation research. It adds a scale-free simultaneous theorem for the ordinary sample variances of exactly one fixed candidate--baseline pair, with public failure probability interpreted jointly over the two actions.

For conditionally iid quadratic forms with variance $\sigma^2$ and fourth central moment $\mu_4$, the exact identity is

$$
\operatorname{Var}(S_s^2\mid\mathcal G)
=
\frac1s
\left[
\mu_4-
\frac{s-3}{s-1}\sigma^4
\right].
$$

The retained degree-two Rademacher hypercontractive bound $\mu_4\le81\sigma^4$, Chebyshev's inequality, and an equal union-bound split across the two actions give

$$
\varepsilon_{\mathrm{Ch}}(s,\delta_{\mathrm{joint}})
=
\sqrt{
\frac{2[80+2/(s-1)]}{s\delta_{\mathrm{joint}}}
}.
$$

At the primary joint failure probability $0.05$, the radius decreases from $28.401878$ at $s=4$ to $10.004031$ at $s=32$. The squared expression is strictly decreasing for $s>1$, so the certificate is analytically vacuous for every frozen sample size $s\le32$. Its status is `PROVED BUT BUDGET-VACUOUS`.

The Hoeffding decomposition was audited before importing any degenerate U-statistic theorem:

$$
S_s^2-\sigma^2
=
\frac2s\sum_i\frac{(X_i-\mu)^2-\sigma^2}{2}
+
\binom{s}{2}^{-1}\sum_{i<j}-(X_i-\mu)(X_j-\mu).
$$

Only the second term is canonical. The first term is generally nondegenerate, so an Adamczak-type completely degenerate theorem cannot be applied to the full sample variance. A sharper explicit radius remains `INCOMPLETE` because numerical constants have not been closed for both terms.

The implemented net-safe implication compares the paid candidate with the original baseline and therefore retains the denominator penalty:

$$
S_a^2
\le
\frac{1-\varepsilon}{1+\varepsilon}
\frac{\ell_{\mathrm{paid}}}{\ell_0}S_0^2.
$$

No proved radius is below one, so the deterministic sample-size gate is empty and the preregistered verdict is **`THEOREM ONLY / BUDGET-VACUOUS`**. The practical bootstrap gate was not run, no empirical result selected $s$, no new matrix--vector query was issued, and no online allocator was implemented. See `docs/proof_rademacher_sample_variance_confidence.md` and `reports/rademacher_sample_variance_confidence_phase1c.md`.

## Continuation Phase 1D: Linear Lower-Tail Analytic No-Go

Phase 1D isolated the nondegenerate Hoeffding linear component $L_s = \frac{1}{s}\sum_i \frac{Z_i^2 - \sigma^2}{\sigma^2}$ using Cortinovis–Kressner (2022) Theorem 2 and the frozen truncation--Bernstein family:
- Exact variance floor across all caps $T \ge 64$ and structural ratios $\kappa = \|C\|_2 / \|C\|_F \in (0, 1]$:
  $$\boxed{80 \le \nu_\kappa(T) \le 81.}$$
- One-sided Bernstein candidate lower-tail bound:
  $$D_{s,\kappa}(\varepsilon, T) > e^{-s/160} \ge e^{-32/160} = e^{-0.2} \approx 0.81873.$$
- Because $0.81873 \gg 0.05$ (the maximum declared component failure probability), the truncation--Bernstein route is budget-vacuous for all $s \le 32$.
- Verdict: **`STRONG LINEAR NO-GO`**.
- Artifacts: 4 validated CSVs in `results/`, 10 targeted tests passing, and `reports/rademacher_linear_projection_no_go_phase1d.md`.

## Exploratory local branch: two-stage Ritz-gap gate

An uncommitted local prototype, `Adaptive_Hutch_pplus_TwoStageGated`, reuses an
initial $b_0=8$ range sketch as part of the Standard-Hutch++ allocation when no
pilot Ritz gap is detected. "No fallback penalty" means those pilot columns
are reused inside $q_0$; it does not mean that the pilot itself costs zero
queries. The prototype bypasses realized-risk certification and is therefore
not a consequence of Phases 1A--1D.

Its five-spectrum, one-orientation, 50-trial development benchmark is mixed:
it improves the reported median error for the $r_\star=5$ step and $c=0.5$
power law, does not trigger for the $r_\star=15$ step, and is not uniformly
better under median, mean, or squared relative error. It has no held-out
confidence analysis. Preserve this branch as `HEURISTIC / EXPLORATORY`; do
not describe it as certified, Pareto-optimal, or as solving the Phase 1B
certification-tax problem.

## Phase 2A direct paired-risk-difference result

Phase 2A is complete as an artifact-only continuation study. It issued zero
new matrix--vector queries, changed no Hutch++ estimator, and reused the frozen
Phase 1A sample-variance observations and Phase 1B paid/original denominators.
The direct common-probe statistic is

$$
\widehat\Delta_R
=
\frac{S_a^2}{\ell_a^{\rm paid}}
-
\frac{S_0^2}{\ell_0^{\rm original}}.
$$

Its exact order-two U-statistic representation is conditionally unbiased. The
paired Hoeffding decomposition has linear covariance reduction

$$
-2(\ell_a\ell_0)^{-1}\operatorname{Cov}(Z_a^2,Z_0^2)
$$

and canonical reduction through
$-2(\ell_a\ell_0)^{-1}\mathbb E[Z_aZ_0]^2$. These identities locate where
common-probe cancellation can occur; a tested counterexample confirms that
positive covariance is not universal.

For the frozen nested shared-prefix architecture only,

$$
c_{\rm pre}=\max\{q_a+r_a,q_0+r_0\}.
$$

Every nonnested or separately constructed architecture must instead use the
actual committed construction-query count from its oracle ledger. All 43,200
Phase 2A accounting rows were cross-checked against the frozen Phase 1B table.
There are 1,600 explicitly infeasible rows; none was deleted or regularized.

The preregistered verdict is **`PAIRING SIGNAL GO`**. At the primary
$(m,\eta,s)=(160,10^{-6},16)$ comparison $q=r_\star+1$ versus $q=r_\star$,
the equal-rank mean pairing variance ratio is $0.151032$ with conditional
path-bootstrap interval $[0.137069,0.166735]$. The covariance contribution is
$0.848968$ with interval $[0.833785,0.862865]$. The within-batch cyclic-shift
comparator has mean ratio $0.999563$; it is a shifted/decorrelated diagnostic,
not an independent sequence. The mathematically exact independence benchmark
remains

$$
V_{\rm ind}
=
\operatorname{Var}(S_a^2/\ell_a)
+
\operatorname{Var}(S_0^2/\ell_0).
$$

The aggregate gain is strongly heterogeneous. The 37 net-beneficial paths
have mean pairing ratio $0.804673$, the 563 net-harmful paths have ratio
$0.108075$, and the top-5% catastrophic paths have ratio $0.899565$ versus
$0.111635$ on ordinary paths. Thus pairing substantially reduces global
uncertainty, but cancellation is weakest on the rare beneficial tail paths
that motivate certification.

This is `EMPIRICALLY ESTABLISHED`, conditional on the frozen orientations and
path population. It is not a finite-sample confidence theorem. The next
research step is a separately reviewed Phase 2B attempt at a one-sided bound
for the direct paired difference. No online allocator is authorized.
