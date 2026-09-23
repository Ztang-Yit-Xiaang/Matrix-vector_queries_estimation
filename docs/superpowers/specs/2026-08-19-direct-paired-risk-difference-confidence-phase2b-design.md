# Phase 2B Design: One-Sided Direct Paired Rademacher Risk-Difference Confidence

**Status:** approved design; specification-only freeze before theorem implementation

**Date:** 2026-08-19

## 1. Executive decision

Phase 2A established that common certification probes materially reduce the
empirical variance of the direct risk difference under the frozen path
population. It did not provide a finite-sample confidence theorem. Phase 2B
will now ask the narrower mathematical question

\[
\boxed{
\text{Can the same common-probe observations alone produce an explicit,
one-sided finite-sample bound for }\Delta_R?
}
\]

The target is one fixed, preconstructed candidate--baseline pair. The theorem
may use only:

1. the common-probe quadratic-form observations already generated for that
   pair;
2. the fixed positive residual denominators;
3. the declared failure probability;
4. universal constants proved for the exact Rademacher-polynomial class.

It may not use matrix norms, eigenvalue estimates, cached-matrix structural
quantities, fitted constants, external pilot data, additional certification
probes, or empirical bootstrap quantiles.

The primary proof route is an independent signed-pair reduction. The complete
paired U-statistic is a secondary efficiency route and may not bypass its
nondegenerate Hoeffding projection. No allocator is part of Phase 2B.

## 2. Motivation from the completed phases

The relevant completed conclusions are:

- Phase 1A: ordinary sample variance contains useful empirical information
  about realized Rademacher risk.
- Phase 1B: after charging construction and certification, the frozen
  candidate-first architecture behaves as tail-risk insurance and harms most
  ordinary paths.
- Phase 1C: separate-action Chebyshev confidence is proved but
  budget-vacuous for every frozen sample size.
- Phase 1D: the frozen truncation--Bernstein route fails on the necessary
  candidate-underestimation component before the canonical term matters.
- Phase 2A: for the direct common-probe difference, the primary equal-rank
  empirical variance ratio is \(0.151032\), but it rises to \(0.804673\) on
  net-beneficial paths and \(0.899565\) on top-5% catastrophic paths.

Therefore Phase 2B must preserve the paired dependence while also recognizing
that the strongest aggregate cancellation does not occur on the paths that
matter most for a useful certificate.

## 3. Research boundary

### Included

- a one-sided confidence theorem attempt for one fixed pair;
- an independent signed-pair reduction with an exact conditional mean;
- an explicit Rademacher-polynomial degree and moment audit;
- elementary and robust bounded-kurtosis feasibility routes;
- a secondary complete-U-statistic route only after the linear projection is
  controlled;
- an analytic budget-vacuity check before any frozen-probe replay;
- exhaustive small-dimensional verification of every proposed theorem;
- an optional isolated replay only if a theorem with explicit constants closes;
- exact paid-candidate/original-baseline accounting.

### Excluded

- changes to Hutch++, Adaptive Hutch++, Standard Hutch++, or the two-stage
  exploratory prototype;
- construction of a new candidate or allocation rule;
- matrix-structural side information, including \(\|H_x\|\), \(\|C_x\|\),
  eigenvalues, stable ranks, and pair-specific polynomial coefficients;
- constants estimated from Phase 1A, Phase 1B, or Phase 2A;
- a bootstrap interval presented as a theorem;
- treating cyclically shifted observations as independent;
- applying a completely degenerate theorem to a nondegenerate statistic;
- silent use of additional matrix--vector products;
- reuse of certification probes as final Hutchinson probes;
- claims of complete-policy baseline safety.

## 4. Assumption and measurability ledger

Condition on the pre-certification sigma-algebra \(\mathcal G\). It contains:

- the symmetric matrix \(A\in\mathbb R^{d\times d}\);
- fixed orthonormal bases \(Q_a\) and \(Q_0\);
- attempted and accepted ranks;
- the fixed candidate--baseline pair;
- the committed construction-query count;
- the fixed positive residual denominators \(\ell_a\) and \(\ell_0\).

Define

\[
R_x=I-Q_xQ_x^T,
\qquad
H_x=R_xAR_x,
\qquad x\in\{a,0\}.
\]

The certification probes satisfy

\[
g_1,\ldots,g_s
\overset{\mathrm{iid}}{\sim}
\operatorname{Rad}(\pm1)^d,
\qquad
(g_1,\ldots,g_s)\perp\!\!\!\perp\mathcal G.
\]

For each action and probe, put

\[
X_{x,i}=g_i^TH_xg_i,
\qquad
\mu_x=\mathbb E[X_{x,i}\mid\mathcal G],
\qquad
Z_{x,i}=X_{x,i}-\mu_x.
\]

Then

\[
\sigma_x^2
=
\operatorname{Var}(X_{x,i}\mid\mathcal G)
=
2\sum_{u\ne v}(H_x)_{uv}^2.
\]

The fixed target is

\[
\boxed{
\Delta_R
=
\frac{\sigma_a^2}{\ell_a}
-
\frac{\sigma_0^2}{\ell_0}.
}
\]

The public \(\delta\in(0,1)\) is the total failure probability for this one
fixed pair and this one one-sided claim. It is not divided as if two separate
action intervals were being constructed. Any internal split must be named and
must sum to at most \(\delta\).

## 5. Construction and denominator contract

The primary target compares a paid candidate with the original unstarted
baseline. For the frozen nested shared-prefix architecture only,

\[
c_{\mathrm{pre}}
=
\max\{q_a+r_a,q_0+r_0\},
\]

and

\[
\ell_a^{\mathrm{paid}}
=
m-c_{\mathrm{pre}}-s,
\qquad
\ell_0^{\mathrm{original}}
=
m-q_0-r_0.
\]

For nonnested or separately constructed actions, \(c_{\mathrm{pre}}\) is the
actual committed construction-query count from the oracle/query ledger. The
maximum formula is not used.

Every theorem requires positive denominators. Nonpositive paid capacity is an
infeasible action, not a zero-risk or regularized case.

## 6. Primary independent signed-pair reduction

Let

\[
n=\left\lfloor\frac{s}{2}\right\rfloor.
\]

For \(j=1,\ldots,n\), define

\[
\boxed{
D_j
=
\frac{(X_{a,2j-1}-X_{a,2j})^2}{2\ell_a}
-
\frac{(X_{0,2j-1}-X_{0,2j})^2}{2\ell_0}.
}
\]

The candidate and baseline terms inside one \(D_j\) use the same two probes.
This preserves the common-probe dependence. Different \(D_j\)'s use disjoint
probe pairs and are therefore conditionally iid given \(\mathcal G\).

### Proposition 6.1: exact conditional mean

For each action,

\[
\mathbb E\left[
\frac{(X_{x,1}-X_{x,2})^2}{2}
\mid\mathcal G
\right]
=
\sigma_x^2.
\]

Consequently,

\[
\boxed{
\mathbb E[D_j\mid\mathcal G]=\Delta_R.
}
\]

The primary point estimator is

\[
\overline D_n=\frac1n\sum_{j=1}^nD_j.
\]

This estimator uses fewer pair combinations than the complete U-statistic, but
it converts the confidence problem into the mean of conditionally iid signed
observations. It is the proof baseline, not automatically the final efficient
estimator.

## 7. Exact polynomial-class audit

For fixed \(H_x\), the centered quadratic form \(Z_x(g)\) is a degree-two
multilinear polynomial in the independent Rademacher coordinates of \(g\).
Each \(D_j\) is therefore a polynomial of degree at most four in the
\(2d\) independent signs of its probe pair.

Let

\[
P_j=D_j-\Delta_R.
\]

Then \(P_j\) is centered and has degree at most four. The design will verify
the Bonami--Beckner hypercontractive implication

\[
\|P_j\|_4
\le
(4-1)^{4/2}\|P_j\|_2
=
9\|P_j\|_2,
\]

and hence

\[
\boxed{
\mathbb E[P_j^4\mid\mathcal G]
\le
6561\,
\mathbb E[P_j^2\mid\mathcal G]^2.
}
\]

The constant \(6561\) is retained only after the theorem statement and degree
hypotheses are independently verified. A smaller constant may replace it only
if proved uniformly for this exact signed-pair class and adversarially audited.
No empirical kurtosis estimate may replace the universal constant inside a
theorem.

The zero-variance branch is separate. If
\(\operatorname{Var}(D_j\mid\mathcal G)=0\), then \(D_j=\Delta_R\) almost
surely under the finite Rademacher law, and the confidence radius is exactly
zero. No division by a variance occurs in that branch.

## 8. Candidate theorem routes

### Route A: elementary paired studentization

Let

\[
v_D=\operatorname{Var}(D_j\mid\mathcal G)
\]

and let \(S_D^2\) be the ordinary sample variance of
\(D_1,\ldots,D_n\). Use the exact sample-variance identity

\[
\operatorname{Var}(S_D^2\mid\mathcal G)
=
\frac1n
\left[
\mu_{4,D}
-
\frac{n-3}{n-1}v_D^2
\right]
\]

together with the verified fourth-moment factor. Combine an observable upper
confidence bound for \(v_D\) with a one-sided Cantelli or Chebyshev bound for
\(\Delta_R-\overline D_n\).

This route must produce a closed numerical expression using only
\((D_1,\ldots,D_n)\), \(n\), and \(\delta\). Any internal failure split is
fixed before evaluation. Because the worst-case fourth-moment factor is large,
an analytic vacuity test precedes implementation.

### Route B: robust bounded-kurtosis mean and scale

Audit an explicit robust-mean theorem under a known kurtosis bound, such as a
Catoni-type or median-of-means construction. The theorem must provide an
observable one-sided radius without a prior numerical variance bound. If the
method estimates scale and mean from disjoint subsets, the split is frozen and
its effective sample sizes are reported.

The imported result must be checked for:

- whether it assumes known variance rather than only known kurtosis;
- whether its interval is actually observable;
- whether its constants are explicit;
- whether it permits signed observations;
- whether it is one-sided or two-sided;
- whether it remains valid at zero variance;
- whether the same observations are reused legally.

If a theorem states that adaptation to unknown scale is possible only through
a nonobservable interval, it does not satisfy the Phase 2B API.

### Route C: complete paired U-statistic

The complete statistic is

\[
\widehat\Delta_U
=
\binom{s}{2}^{-1}
\sum_{i<j}
\left[
\frac{(X_{a,i}-X_{a,j})^2}{2\ell_a}
-
\frac{(X_{0,i}-X_{0,j})^2}{2\ell_0}
\right].
\]

Its Hoeffding decomposition is

\[
\widehat\Delta_U-\Delta_R
=
\frac2s\sum_iK_1(Y_i)
+
\binom{s}{2}^{-1}\sum_{i<j}K_2(Y_i,Y_j),
\]

where

\[
K_1(Y)
=
\frac{Z_a^2-\sigma_a^2}{2\ell_a}
-
\frac{Z_0^2-\sigma_0^2}{2\ell_0}
\]

and

\[
K_2(Y,Y')
=
-\frac{Z_aZ_a'}{\ell_a}
+
\frac{Z_0Z_0'}{\ell_0}.
\]

Route C is secondary. A theorem for canonical or completely degenerate
U-statistics may be applied only to \(K_2\). The linear projection \(K_1\)
must receive its own explicit, observable, one-sided control. If that necessary
component is budget-vacuous, canonical-kernel work stops unless a different
linear theorem is independently justified.

## 9. Imported-result audit

The initial literature audit includes:

- Giné, Latała, and Zinn, *Exponential and moment inequalities for
  U-statistics*, for canonical U-statistic concentration:
  <https://arxiv.org/abs/math/0003228>;
- Adamczak, *Moment inequalities for U-statistics*, for completely degenerate
  U-statistics:
  <https://arxiv.org/abs/math/0506026>;
- Maurer and Pontil, *Empirical Bernstein Bounds and Sample Variance
  Penalization*, whose empirical Bernstein results require their declared
  boundedness framework:
  <https://arxiv.org/abs/0907.3740>;
- Catoni, *High confidence estimates of the mean of heavy-tailed real random
  variables*, for bounded-kurtosis robust mean/variance constructions and the
  distinction between adaptive estimators and observable confidence intervals:
  <https://arxiv.org/abs/0909.5366>.

No theorem is imported by title alone. Implementation requires recording its
exact statement, theorem number, hypotheses, conclusion, constants, and a
line-by-line mapping to the present variables.

If the exact source statement cannot be verified, the route is `UNVERIFIED` or
`INCOMPLETE`, not `PROVED`.

## 10. Analytic early-stop audit

Before replaying frozen probes, evaluate every closed theorem on

\[
s\in\{4,8,16,32\},
\qquad
n=\lfloor s/2\rfloor,
\qquad
\delta\in\{0.01,0.05,0.10\}.
\]

The primary values are

\[
s=16,
\qquad n=8,
\qquad \delta=0.05.
\]

The theorem route is analytically budget-vacuous if its prerequisites cannot
hold or its observable radius is necessarily infinite/undefined for every
admissible sample on the frozen grid. A scale-free theorem may also be labeled
practically vacuous if its acceptance inequality cannot be satisfied outside
the exact-zero branch; that claim requires proof, not an empirical guess.

No empirical path may choose the theorem method or sample size. Among methods
classified `PROVED`, the fixed priority is:

1. independent-pair robust bounded-kurtosis method;
2. independent-pair elementary method;
3. complete-U-statistic method.

Within one method, the primary sample size remains \(s=16\). The other sample
sizes are sensitivity checks and cannot replace it after results are seen.

If no method yields an observable finite one-sided radius, Phase 2B stops with
`INCOMPLETE` or `PROVED BUT BUDGET-VACUOUS`. It does not replay probes.

## 11. Public theorem API

If a theorem closes, expose a pure constructor such as

```python
paired_risk_difference_certificate(
    paired_observations,
    *,
    joint_delta: float,
    method: str,
) -> PairedDifferenceCertificate
```

Here `paired_observations` means either the independent signed-pair values
\(D_j\) or the complete paired observations required by a proved full-U
method. The returned record includes:

- `joint_delta` for the one fixed pair and one-sided event;
- theorem name, source, and proof status;
- every internal failure allocation;
- sample size and effective independent-block count;
- universal moment constants;
- point estimate \(\widehat\Delta\);
- observable one-sided radius \(C_s\);
- zero-variance branch status;
- the precise guaranteed event;
- whether the result is theorem-valid but budget/practically vacuous.

No public parameter named merely `delta` is allowed. Internal helpers use names
such as `pointwise_delta`, `scale_delta`, or `mean_delta`.

The decision API is

```python
direct_paired_net_safe_decision(
    certificate: PairedDifferenceCertificate,
) -> "accept" | "abstain"
```

It returns `accept` only when

\[
\boxed{
\widehat\Delta+C_s\le0.
}
\]

It returns `abstain` when the theorem is incomplete, the radius is nonfinite,
the denominators are invalid, the zero-variance branch is unresolved, or the
inequality fails.

## 12. Optional frozen-probe replay

Replay is authorized only after at least one method is `PROVED` with an
observable radius. The raw \(X_{x,i}\) values were not retained in the Phase
1A wide artifact, so an independent-pair or jackknife audit requires exact
reconstruction from the frozen orientation, basis, and certification seeds.

The replay contract is:

- use the same frozen Rademacher probes and actions;
- reconstruct the exact same \(X_{x,i}\), not fresh replacements;
- use `MatVecOracle` or an explicitly equivalent diagnostic counter;
- record every replayed \(Ag_i\) product as a diagnostic reconstruction query;
- never describe replay as zero-query online evidence;
- write only isolated Phase 2B artifacts;
- preserve all Phase 1A, Phase 1B, Phase 1C, Phase 1D, and Phase 2A checksums.

The replay does not modify the estimator and does not authorize reusing those
probes in a final trace estimate.

## 13. Preregistered practical audit after theorem closure

The practical audit uses the theorem-selected method, not the empirically best
method. Its primary setting is

\[
m=160,
\qquad
\eta=10^{-6},
\qquad
s=16,
\qquad
\delta=0.05,
\qquad
q_a=r_\star+1,
\qquad
q_0=r_\star.
\]

Report:

- acceptance probability on truly net-beneficial paths;
- false-safe acceptance on truly net-harmful paths as an empirical diagnostic;
- top-1%, top-5%, and top-10% catastrophic detection;
- selected/original mean-risk ratio;
- abstention rate;
- per-rank and equal-rank aggregation;
- 10,000 conditional rank-stratified path-bootstrap replicates.

The bootstrap describes the frozen path population. It does not validate the
finite-sample theorem.

The practical gate is:

1. theorem status `PROVED`;
2. an observable finite radius at the primary setting;
3. top-5% net-beneficial catastrophic detection at least \(75\%\);
4. its conditional bootstrap lower endpoint at least \(60\%\);
5. selected/original mean-risk bootstrap upper endpoint below one;
6. no observed contradiction of the theorem event in exhaustive tiny tests;
7. all frozen checksums and accounting identities pass.

An empirical false-safe observation does not by itself refute a theorem unless
the test enumerates the theorem's full probability space or otherwise provides
a valid contradiction. It is reported as a diagnostic.

## 14. Verdicts

- `THEOREM + PRACTICAL GO`: an explicit theorem closes and the preregistered
  practical gate passes at \(s=16\).
- `PROVED BUT PRACTICALLY VACUOUS`: a valid observable theorem closes but the
  primary practical gate fails or it never accepts outside degenerate cases.
- `PROVED BUT BUDGET-VACUOUS`: a valid route cannot yield a usable radius for
  any frozen \(s\le32\).
- `INCOMPLETE`: a required observable scale bound, theorem hypothesis, or
  explicit numerical constant remains unresolved.
- `REFUTED`: a valid counterexample defeats a proposed theorem.
- `UNVERIFIED`: the exact imported theorem statement or hypothesis mapping
  cannot be responsibly confirmed.

The final Phase 2B verdict is determined by the fixed method priority and may
not be changed by secondary empirical results.

## 15. Mathematical verification

Tests and proof audits must cover:

1. exact conditional unbiasedness of \(D_j\);
2. conditional iid structure across disjoint probe pairs;
3. preservation of common-probe dependence within one \(D_j\);
4. exact equality between the complete U-statistic and the difference of
   common-probe sample variances;
5. the complete Hoeffding decomposition term by term;
6. degeneracy of \(K_2\) and nondegeneracy of \(K_1\);
7. exact polynomial degree of \(P_j\);
8. the imported hypercontractive constant and fourth-moment implication;
9. exact-zero and one-zero-risk branches;
10. negative and zero covariance counterexamples;
11. signed \(D_j\) values;
12. positive-denominator requirements;
13. nested maximum accounting versus nonnested actual-ledger accounting;
14. every internal failure-probability sum;
15. deterministic method and sample-size selection;
16. rejection of a degenerate theorem applied to the full statistic;
17. rejection of a bounded empirical-Bernstein theorem without a proved bound;
18. rejection of a nonobservable interval from the public API;
19. theorem behavior under scaling of \(A\);
20. no use of Phase 2A bootstrap values as theorem constants.

For dimensions small enough to enumerate all Rademacher probes, compare the
declared one-sided failure probability with exact coverage over the complete
finite sample space. Numerical simulation alone cannot classify a theorem as
proved.

## 16. Implementation sequence after written-spec approval

1. Add the approved specification and commit it alone.
2. Verify the exact source statements and constants for every imported theorem.
3. Write the assumption ledger and elementary signed-pair lemmas.
4. Implement pure symbolic/numerical theorem helpers without reading frozen
   empirical outputs.
5. Run the analytic vacuity grid.
6. Stop immediately if no observable route closes.
7. Only after a `PROVED` status, implement an isolated frozen-probe replay.
8. Run the practical audit without retuning method, \(s\), \(\delta\), or gate.
9. Update the proof note, report, tracker, memory, current state, and Obsidian
   records.
10. Do not implement an allocator in Phase 2B.

## 17. Planned files after implementation approval

The theorem stage may add:

- `src/paired_rademacher_difference_confidence.py`;
- `tests/test_paired_rademacher_difference_confidence_phase2b.py`;
- `docs/proof_paired_rademacher_difference_confidence.md`;
- `experiments/postprocess_paired_rademacher_difference_confidence_phase2b.py`;
- `reports/paired_rademacher_difference_confidence_phase2b.md`;
- isolated `results/paired_rademacher_difference_confidence_phase2b_*`
  artifacts.

Replay-specific files are added only if the analytic theorem gate permits
replay. Frozen estimator and historical result files remain unchanged.

## 18. Interpretation rules

A proved one-sided pair certificate would establish:

> For one pair fixed before fresh certification probes, acceptance implies that
> the paid candidate's realized Rademacher conditional risk is no larger than
> the original baseline's risk on the declared confidence event.

It would not establish:

- that abstention refunds sunk costs;
- that a candidate-first policy is baseline-safe end to end;
- that several adaptively searched candidates are simultaneously covered;
- that Gaussian residual risk is controlled;
- that the same probes may be reused in the final estimator;
- that the empirical Phase 2A covariance persists for every orientation;
- that an online allocator has been designed.

The narrow negative conclusion, if all routes fail, is:

> Under the strict data-only contract and tested explicit theorem families,
> \(s\le32\) does not yield a useful observable one-sided radius.

That is not an impossibility theorem for all direct-risk certification methods.

## 19. Fixed decisions

- Only the same common-probe observations and known denominators are legal
  theorem inputs.
- The theorem concerns one fixed pair and one one-sided event.
- Primary \((s,\delta)=(16,0.05)\); secondary
  \(s\in\{4,8,16,32\}\) and \(\delta\in\{0.01,0.05,0.10\}\).
- The independent signed-pair route is the first proof route.
- The full U-statistic is secondary and cannot skip its linear projection.
- Universal constants are proved or imported with exact hypotheses; they are
  never fitted.
- Frozen-probe replay occurs only after theorem closure and is charged as
  diagnostic reconstruction.
- Phase 2A's shifted comparator remains shifted/decorrelated, not independent.
- No online allocator is part of this design.
