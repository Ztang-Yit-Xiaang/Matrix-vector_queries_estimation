# Next Research Plan: Direct Common-Probe Rademacher Risk-Difference Certification

**Status:** written research plan for review; no Phase 2 implementation is
authorized by this document alone.

**Date:** 2026-08-19

## 1. Executive decision

The next continuation study will not construct another Hutch++ allocator and
will not continue the failed separate-action confidence route.  It will study
the quantity the decision actually needs:

\[
\boxed{
\Delta_R(a,0)
=
\frac{\sigma_a^2}{\ell_a}
-
\frac{\sigma_0^2}{\ell_0}.
}
\]

The central question is

\[
\boxed{
\text{Do common Rademacher probes reduce the uncertainty of the direct risk
difference enough to support a useful one-sided confidence bound?}
}
\]

The study begins with a no-new-query audit of the frozen Phase 1A and Phase 1B
artifacts.  A theorem attempt begins only if the preregistered empirical audit
shows reproducible covariance cancellation.  The two-stage Ritz-gap prototype
is outside this plan.

## 2. Why this is the correct continuation

Phases 1A--1D established the following sequence.

1. The exact realized Rademacher risk for a fixed basis is the correct
   probe-specific target.
2. Ordinary sample variance contains useful information about that risk in the
   frozen offline experiment.
3. Charging construction and certification costs turns the method into
   tail-risk insurance: it helps empirical mean risk through rare catastrophic
   paths while harming most ordinary paths.
4. The valid separate-action Chebyshev radius is budget-vacuous for
   \(s\le32\).
5. The sharper frozen truncation--Bernstein route also fails on the necessary
   candidate-underestimation component.

Both failed theorem routes separately bounded candidate and baseline risks and
then combined them by a union bound.  That discards the dependence created by
using the same certification probes.  The next study keeps that dependence.

This is a narrower target than proving two simultaneous multiplicative
intervals.  It needs only a one-sided statement

\[
\boxed{
\Pr\!\left(
\Delta_R>
\widehat\Delta_R+C_s
\mid\mathcal G
\right)
\le\delta.
}
\]

Then

\[
\widehat\Delta_R+C_s\le0
\quad\Longrightarrow\quad
\Delta_R\le0
\]

on the confidence event.

## 3. Research boundary

### Included

- a conditional-unbiasedness proof for the direct paired U-statistic;
- its exact Hoeffding decomposition;
- exact variance identities showing where common-probe cancellation enters;
- an artifact-only empirical covariance-cancellation audit;
- a preregistered comparison with an independence/no-pairing variance proxy;
- a decision gate for whether theorem work is warranted;
- if the gate passes, a separate explicit one-sided theorem feasibility study;
- exact budget denominators inherited from Phase 1B;
- theorem-versus-empirical classification throughout.

### Excluded

- changes to Hutch++, Standard Hutch++, or the frozen adaptive estimators;
- changes to the exploratory two-stage Ritz-gap prototype;
- a new allocation rule;
- new matrix--vector experiments in the first audit stage;
- reuse of certification probes as final trace-estimation probes;
- fitted theorem constants;
- post-hoc choice of \(s\), \(\delta\), action pairs, or success thresholds;
- claims of online baseline safety;
- claims that direct pairing must help for every matrix or pair of bases.

## 4. Assumption ledger

Condition on a pre-certification sigma-algebra \(\mathcal G\).  It contains the
matrix, all constructed bases and cached products, the attempted and accepted
ranks, the selected fixed candidate--baseline pair, and every residual
denominator.  These objects must be measurable before certification probes are
drawn.

Let

\[
A=A^T\in\mathbb R^{d\times d},
\]

and let \(Q_a,Q_0\) have orthonormal columns.  Define

\[
R_x=I-Q_xQ_x^T,
\qquad
H_x=R_xAR_x,
\qquad x\in\{a,0\}.
\]

For conditionally iid coordinate-Rademacher probes

\[
g_1,\ldots,g_s
\overset{\mathrm{iid}}{\sim}
\operatorname{Rad}(\pm1)^d,
\qquad
(g_1,\ldots,g_s)\perp\!\!\!\perp\mathcal G,
\]

put

\[
X_{x,j}=g_j^TH_xg_j,
\qquad
\mu_x=\mathbb E[X_{x,j}\mid\mathcal G],
\qquad
Z_{x,j}=X_{x,j}-\mu_x,
\]

and

\[
\sigma_x^2
=
\operatorname{Var}(X_{x,j}\mid\mathcal G)
=
2\sum_{u\ne v}(H_x)_{uv}^2.
\]

Within one probe index \(j\), \(X_{a,j}\) and \(X_{0,j}\) are generally
dependent because they use the same \(g_j\).  Across different probe indices,
the pairs

\[
Y_j=(X_{a,j},X_{0,j})
\]

are conditionally iid.

All denominators must be positive.  Exact-zero risks remain valid mathematical
boundary cases and must not be regularized by an arbitrary positive floor.

## 5. Primary net-risk target and denominator contract

The primary target compares the paid candidate with the original, unstarted
baseline. For the nested shared-prefix primary architecture used by the frozen
adjacent actions,

\[
c_{\mathrm{pre}}
=
\max\{q_a+r_a,q_0+r_0\},
\]

because constructing the longer prefix makes the shorter nested prefix
available without paying for it a second time. This maximum formula is not a
universal accounting identity. For nonnested actions, separately constructed
bases, or any architecture without the frozen shared-prefix reuse contract,
replace it by the actual committed construction-query count recorded by the
oracle/query ledger.

\[
\ell_a^{\mathrm{paid}}(m,s)
=
m-c_{\mathrm{pre}}-s,
\]

and

\[
\ell_0^{\mathrm{original}}(m)
=
m-q_0-r_0.
\]

The primary difference is

\[
\boxed{
\Delta_R^{\mathrm{net}}
=
\frac{\sigma_a^2}{\ell_a^{\mathrm{paid}}}
-
\frac{\sigma_0^2}{\ell_0^{\mathrm{original}}}.
}
\]

This target certifies an accepted candidate against the original baseline.  It
does not make an abstaining candidate-first policy baseline-safe, because the
policy has already incurred construction and certification costs.

Secondary analyses may use the intrinsic frozen denominators

\[
\ell_x^{\mathrm{intrinsic}}=m-q_x-r_x,
\]

but they must be labeled separately and cannot determine the primary verdict.

## 6. Direct paired U-statistic

For any fixed positive denominators \(\ell_a,\ell_0\), define

\[
K(Y_i,Y_j)
=
\frac{(X_{a,i}-X_{a,j})^2}{2\ell_a}
-
\frac{(X_{0,i}-X_{0,j})^2}{2\ell_0}.
\]

The direct estimator is

\[
\boxed{
\widehat\Delta_R
=
\binom{s}{2}^{-1}
\sum_{i<j}K(Y_i,Y_j).
}
\]

Equivalently,

\[
\widehat\Delta_R
=
\frac{S_a^2}{\ell_a}
-
\frac{S_0^2}{\ell_0},
\]

where both sample variances use the same probes.  Thus the point estimator
already exists in the Phase 1A artifacts.  The new problem is its joint
one-sided concentration, not a new numerical estimator formula.

### Proposition 6.1: conditional unbiasedness

For independent indices \(i\ne j\),

\[
\mathbb E\!\left[
\frac{(X_{x,i}-X_{x,j})^2}{2}
\mid\mathcal G
\right]
=
\sigma_x^2.
\]

Therefore

\[
\boxed{
\mathbb E[\widehat\Delta_R\mid\mathcal G]
=
\Delta_R.
}
\]

This statement is `PROVED` once the measurability, conditional independence,
finite second moments, and positive-denominator assumptions are verified.

## 7. Exact Hoeffding decomposition

Write

\[
\alpha=\ell_a^{-1},
\qquad
\beta=\ell_0^{-1}.
\]

The kernel mean is

\[
\theta
=
\alpha\sigma_a^2-
\beta\sigma_0^2
=
\Delta_R.
\]

The first projection is

\[
\boxed{
K_1(Y)
=
\frac{\alpha}{2}(Z_a^2-\sigma_a^2)
-
\frac{\beta}{2}(Z_0^2-\sigma_0^2).
}
\]

The canonical remainder is

\[
\boxed{
K_2(Y,Y')
=
-\alpha Z_aZ_a'
+
\beta Z_0Z_0'.
}
\]

Hence

\[
\boxed{
\widehat\Delta_R-\Delta_R
=
\frac2s\sum_{i=1}^sK_1(Y_i)
+
\binom{s}{2}^{-1}\sum_{i<j}K_2(Y_i,Y_j).
}
\]

The plan must verify term by term that

\[
\mathbb E[K_1(Y)\mid\mathcal G]=0
\]

and

\[
\mathbb E[K_2(Y,Y')\mid Y,\mathcal G]=0.
\]

A theorem for completely degenerate U-statistics may be applied only to
\(K_2\), never to the complete estimator.

## 8. Exact location of common-probe cancellation

The linear contribution obeys

\[
4\operatorname{Var}(K_1\mid\mathcal G)
=
\operatorname{Var}\!\left(
\alpha Z_a^2-
\beta Z_0^2
\mid\mathcal G
\right).
\]

Expanding gives

\[
\boxed{
4\operatorname{Var}(K_1\mid\mathcal G)
=
\alpha^2\operatorname{Var}(Z_a^2)
+
\beta^2\operatorname{Var}(Z_0^2)
-
2\alpha\beta\operatorname{Cov}(Z_a^2,Z_0^2).
}
\]

Thus common probes reduce the linear variance exactly when the weighted squared
chaoses have positive covariance.

For the canonical component, conditional independence of \(Y\) and \(Y'\)
gives

\[
\boxed{
\mathbb E[K_2(Y,Y')^2\mid\mathcal G]
=
\alpha^2\sigma_a^4
+
\beta^2\sigma_0^4
-
2\alpha\beta
\mathbb E[Z_aZ_0\mid\mathcal G]^2.
}
\]

Therefore common probes can also reduce the canonical variance when
\(Z_a\) and \(Z_0\) are correlated.

The exact order-two U-statistic variance is

\[
\boxed{
\operatorname{Var}(\widehat\Delta_R\mid\mathcal G)
=
\frac4s\operatorname{Var}(K_1\mid\mathcal G)
+
\frac{2}{s(s-1)}
\mathbb E[K_2^2\mid\mathcal G].
}
\]

These identities explain why direct pairing might help.  They do not prove
that either covariance is positive for every matrix or pair of actions.

## 9. Competing approaches considered

### Approach A: artifact-first paired-difference audit — recommended

Use the frozen sample-variance estimates to measure covariance cancellation,
one-sided error, and decision behavior before attempting a theorem.  This is
cheap, preserves the UROP estimator, and directly tests the premise on which a
paired theorem depends.

### Approach B: immediate worst-case quartic-chaos theorem

Start by bounding the degree-four linear projection and the canonical kernel.
This is mathematically ambitious, but Phase 1D shows that generic worst-case
constants can be vacuous before empirical structure is understood.  It is not
the recommended first step.

### Approach C: abandon pairing and seek a new small-ball theorem

Develop anti-concentration for the individual squared chaos.  This remains a
valid alternative, but it ignores covariance already present in the common
probe design and solves a stronger problem than the decision requires.

The project will use Approach A.  Approaches B and C remain future alternatives
if the paired signal is absent or theoretically inaccessible.

## 10. Phase 2A: no-new-query empirical audit

### Frozen data sources

Use, without modification:

- the Phase 1A 960,000-row certification artifact;
- the Phase 1A exact truth table;
- the Phase 1B path-level paid-risk artifact;
- the Phase 1A and Phase 1B manifests and checksums.

Because

\[
\widehat\Delta_R
=
S_a^2/\ell_a-S_0^2/\ell_0,
\]

the primary paired statistic can be computed from stored sample-variance
columns.  The first audit issues no new matrix--vector query and does not need
raw per-probe \(X_j\) values.

An optional later replay may reconstruct raw probe-level cross moments from the
frozen seeds.  It must be specified separately, use immutable output paths,
and count replayed products as diagnostic reconstruction work rather than
silently calling them zero-cost online queries.

### Frozen primary setting

\[
m=160,
\qquad
\eta=10^{-6},
\qquad
s=16,
\qquad
q_a=r_\star+1,
\qquad
q_0=r_\star.
\]

The primary target uses

\[
\ell_a=\ell_a^{\mathrm{paid}}(m,s),
\qquad
\ell_0=\ell_0^{\mathrm{original}}(m).
\]

Secondary sensitivity uses

\[
s\in\{4,8,16,32\},
\quad
m\in\{80,160,240\},
\quad
\eta\in\{10^{-10},10^{-6}\},
\]

and the three frozen adjacent action pairs.  Secondary results cannot change
the primary verdict.

### Primary empirical quantities

For each frozen path, estimate across its 200 certification repetitions:

1. paired difference variance;
2. the independence proxy

   \[
   V_{\mathrm{ind}}
   =
   \operatorname{Var}(S_a^2/\ell_a)
   +
   \operatorname{Var}(S_0^2/\ell_0);
   \]

3. covariance contribution

   \[
   G_{\mathrm{cov}}
   =
   \frac{
   2\operatorname{Cov}(S_a^2/\ell_a,S_0^2/\ell_0)
   }{V_{\mathrm{ind}}};
   \]

4. pairing variance ratio

   \[
   G_{\mathrm{pair}}
   =
   \frac{
   \operatorname{Var}(\widehat\Delta_R)
   }{V_{\mathrm{ind}}}
   =1-G_{\mathrm{cov}};
   \]

5. a deterministic shifted/decorrelated comparator formed within each frozen path and
   certification batch by replacing the baseline repetition index \(j\) with
   \((j+1)\bmod 50\). This breaks the original same-probe alignment while
   preserving the candidate and baseline marginals and batch structure. It is
   not called an independent sample sequence because the cyclic construction
   reuses observations across shifted comparisons. The mathematically exact
   independence benchmark remains \(V_{\mathrm{ind}}\), not the shifted
   comparator;

6. dangerous one-sided error

   \[
   E^+
   =
   \Delta_R-\widehat\Delta_R,
   \]

   because a positive value means the estimate was too optimistic;
7. empirical 90th, 95th, 97.5th, and 99th percentiles of \(E^+\);
8. sign-classification error and abstention under predeclared additive radii;
9. behavior within ordinary, top-10%, top-5%, and top-1% baseline-risk paths.

If \(V_{\mathrm{ind}}=0\), report an exact degenerate case rather than divide by
zero.  Negative estimated covariance is allowed and must not be clipped.

### Aggregation

Certification repetitions are repeated measurements, not independent
scientific paths.  Compute statistics within frozen path first, then average
paths within rank, then average ranks equally over

\[
r_\star\in\{5,15,30\}.
\]

Use 10,000 rank-stratified frozen-path bootstrap replicates for descriptive
percentile intervals.  These intervals are conditional on the frozen
orientations and path population and are not theorem-level confidence bounds.
Use master bootstrap seed `93000` with fixed integer components for metric,
sample size, budget, tail level, action pair, and batch scope.

### Preregistered Phase 2A gate

Assign `PAIRING SIGNAL GO` only if the primary setting satisfies all of:

1. the equal-rank mean covariance contribution is positive;
2. the 95% bootstrap upper endpoint of \(G_{\mathrm{pair}}\) is below one;
3. the point estimate of \(G_{\mathrm{pair}}\) is at most \(0.80\), meaning at
   least a 20% variance reduction relative to the independence proxy;
4. all three ranks have nonempty true-better and true-worse path sets;
5. the top-5% true-better catastrophic set is nonempty in every rank;
6. no historical checksum, truth label, denominator, or path key changes.

Assign `WEAK / MIXED PAIRING SIGNAL` if covariance is positive but the 20%
reduction or interval criterion fails.  Assign `NO PAIRING ADVANTAGE` if the
equal-rank covariance contribution is nonpositive or pairing variance is not
smaller.  Assign `INCONCLUSIVE` if a required eligibility population is empty
or an artifact cannot be reproduced.

The thresholds are frozen before the Phase 2A outputs are inspected.  No
pointwise choice of \(s\) is allowed.

## 11. Phase 2B: theorem feasibility, conditional on Phase 2A

Phase 2B starts only after `PAIRING SIGNAL GO`.  Its theorem target is an
explicit computable radius

\[
C_s(\delta,\text{observed certification data},\mathcal G)
\]

such that

\[
\Pr\!\left(
\Delta_R>widehat\Delta_R+C_s
\mid\mathcal G
\right)
\le\delta.
\]

The public \(\delta\) is the failure probability for this single fixed pair
and one-sided statement.  It is not silently divided as though two separate
action intervals were being constructed.  If several candidate pairs are
considered, a simultaneous finite-action correction must be explicit.

### Candidate theorem routes

1. **Paired linear-projection route.** Bound

   \[
   \alpha(Z_a^2-\sigma_a^2)
   -
   \beta(Z_0^2-\sigma_0^2)
   \]

   while retaining cross moments rather than replacing them by two separate
   fourth-moment bounds.
2. **Paired canonical-kernel route.** Analyze

   \[
   -\alpha Z_aZ_a'+\beta Z_0Z_0'
   \]

   only after the linear term is controlled with explicit constants.
3. **Robust direct-difference route.** Form independent pair or block
   observations of the signed risk difference and use a one-sided robust mean
   estimator.  Negative observations are allowed; positivity assumptions used
   for variance numerators do not transfer automatically.
4. **Matrix-structural route.** Derive computable pair-specific bounds from the
   nested residual geometry.  Any required norm must be available without an
   uncharged dense matrix or extra oracle access.

No imported theorem may be applied until its symmetry, degeneracy,
independence, moment, boundedness, and measurability hypotheses are checked.
Unspecified universal constants lead to `INCOMPLETE`, not fitted values.

### Theorem statuses

- `PROVED`: every assumption and numerical constant closes;
- `PROVED BUT BUDGET-VACUOUS`: valid radius, but no frozen \(s\le32\) yields a
  usable one-sided decision;
- `INCOMPLETE`: a necessary bound or explicit constant is unresolved;
- `REFUTED`: a valid counterexample defeats the proposed theorem;
- `NOT ATTEMPTED`: route excluded by an earlier stopping gate.

## 12. Decision rule if a theorem eventually closes

For the fixed paid-candidate/original-baseline target, accept only if

\[
\boxed{
\widehat\Delta_R+C_s\le0.
}
\]

Otherwise abstain.  This establishes only accepted-candidate safety on the
confidence event.  It does not refund sunk costs and therefore does not by
itself make the entire candidate-first policy baseline-safe.

Certification probes remain separate from fresh final residual probes.  An
online allocator is outside Phase 2A and Phase 2B.

## 13. Planned implementation files after approval

Phase 2A would add:

- `src/paired_rademacher_risk_difference.py` for pure identities and validated
  aggregation helpers;
- `experiments/postprocess_paired_rademacher_risk_difference_phase2a.py`;
- `tests/test_paired_rademacher_risk_difference_phase2a.py`;
- `reports/paired_rademacher_risk_difference_phase2a.md`;
- isolated `results/paired_rademacher_risk_difference_phase2a_*` artifacts;
- figures under an isolated Phase 2A figure directory.

No file in the frozen Phase 1A, Phase 1B, bridge, or estimator implementation
is overwritten.

Phase 2B receives a separate design and approval cycle.  It is not implicitly
authorized by a Phase 2A implementation.

## 14. Phase 2A output artifacts

Create only after all validations pass:

1. manifest with frozen-input and output checksums;
2. path-level covariance and pairing-gain table;
3. equal-rank summary table;
4. one-sided error-quantile table;
5. catastrophic-stratum table;
6. bootstrap interval table;
7. gate-evaluation table;
8. final Phase 2A verdict table;
9. report and publication-quality figures.

Every artifact records that no new matrix--vector experiment was run and that
all statistical intervals are conditional empirical bootstrap summaries.
The complete secondary path table has

\[
3\times200\times2\times3\times4\times3
=
\boxed{43{,}200}
\]

unique rows keyed by

\[
(r_\star,\texttt{basis\_trial},\eta,m,s,\texttt{action\_pair}).
\]

Infeasible paid-capacity rows remain present with an explicit feasibility flag
and undefined paid-risk fields; they are never silently deleted or used to
renormalize rank weights.

## 15. Figures

At minimum:

1. paired versus independence-proxy variance by rank and \(s\);
2. distribution of pathwise covariance contribution;
3. dangerous one-sided error distributions for the paired statistic and the
   deterministic within-batch shifted/decorrelated comparator, with
   \(V_{\mathrm{ind}}\) shown separately as the exact independence benchmark;
4. pairing gain versus exact risk ratio;
5. ordinary versus catastrophic-path pairing gain;
6. batch-stability panel;
7. secondary budget and tail-level sensitivity.

Use logarithmic axes only for positive quantities.  Signed covariance and
signed error plots use linear or symmetric-log scales with the zero line shown.

## 16. Mathematical and software tests

### Exact identities

- exhaustive tiny-dimensional verification of the Rademacher variance formula;
- conditional unbiasedness of the paired U-statistic;
- equality with the difference of two common-probe sample variances;
- the complete Hoeffding decomposition term by term;
- conditional degeneracy of \(K_2\);
- nondegeneracy of \(K_1\) in a counterexample;
- exact linear and canonical variance identities;
- exact order-two U-statistic variance formula by enumeration.

### Boundary cases

- \(Q_a=Q_0\) and equal denominators, giving exact zero difference;
- identical bases with unequal denominators;
- one or both Rademacher risks equal to zero;
- nonpositive paid capacity;
- truth ties;
- negative covariance;
- nested and nonnested bases;
- rank-deficient actions with \(r_x\ne q_x\).

### Artifact and aggregation tests

- immutable Phase 1A/1B checksums;
- complete unique keys and declared grids;
- path-first and equal-rank weighting;
- exact 200-repetition cluster preservation;
- deterministic bootstrap seeds;
- no silent deletion of infeasible or zero-risk rows;
- no empirical selection of \(s\);
- primary versus secondary verdict isolation;
- `INCONCLUSIVE` behavior for empty eligibility sets;
- zero new matrix--vector experiment count.

### Adversarial theorem audit

- show by example that common probes need not produce positive covariance;
- show that unbiasedness does not imply a useful one-sided radius;
- reject any proof that applies a degenerate theorem to the full kernel;
- reject replacement of paid/original denominators by a common denominator;
- reject multiplicative normalization when \(\Delta_R=0\) or changes sign;
- reject theorem constants fitted from frozen empirical quantiles.

## 17. Verification sequence

After written-plan approval:

1. checksum and schema-audit all frozen Phase 1A/1B inputs;
2. implement and exhaustively test the exact paired identities;
3. run a small read-only artifact smoke analysis into `/private/tmp`;
4. validate path-first aggregation and pairing calculations;
5. run the complete maintained suite;
6. execute the full artifact-only Phase 2A audit;
7. validate keys, eligibility, bootstrap intervals, and verdict;
8. generate figures and report only after numerical validation;
9. rerun the complete maintained suite;
10. run `git diff --check`;
11. audit every report claim as `PROVED`, `EMPIRICALLY ESTABLISHED`,
    `STRONGLY SUPPORTED MECHANISM`, or `OPEN`;
12. update the tracker, memory, current state, and Obsidian records.

## 18. Arguable variables frozen for review

These are the variables a future reviewer may deliberately change before
implementation.  They must not be tuned after Phase 2A results are inspected.

| Variable | Frozen value | Role |
|---|---:|---|
| Primary budget | \(m=160\) | Matches Phase 1A/1B primary analysis |
| Primary tail | \(\eta=10^{-6}\) | Catastrophic zero-oversampling regime |
| Primary sample size | \(s=16\) | Phase 1A passing sample size |
| Primary pair | \(r_\star+1\) vs. \(r_\star\) | Existing tail-insurance comparison |
| Bootstrap replicates | 10,000 | Conditional path-level uncertainty |
| Pairing-gain threshold | 20% | Prevents a negligible reduction from passing |
| Bootstrap criterion | upper 95% endpoint below 1 | Requires resolved variance reduction |
| Rank weighting | equal over 5, 15, 30 | Prevents one rank from dominating |
| Secondary \(s\) | 4, 8, 16, 32 | Sensitivity only |
| Secondary budgets | 80, 160, 240 | Sensitivity only |
| Secondary tails | \(10^{-10},10^{-6}\) | Benign/catastrophic contrast |

## 19. Interpretation rules

A `PAIRING SIGNAL GO` means only:

> Under the frozen basis-path population, common probes materially reduce the
> empirical uncertainty of the direct risk difference relative to an
> independence proxy.

It does not establish a finite-sample confidence theorem.

A `NO PAIRING ADVANTAGE` means only that this frozen paired sample-variance
architecture did not reveal useful cancellation.  It does not disprove all
direct-difference estimators.

If Phase 2A passes but Phase 2B is budget-vacuous, the correct conclusion is
again that empirical information exists but the tested worst-case proof route
cannot exploit it at \(s\le32\).

## 20. Final stopping rules

- Do not implement Phase 2B if Phase 2A is `NO PAIRING ADVANTAGE`.
- Do not analyze a canonical term before controlling the paired linear term.
- Do not implement an online allocator from an empirical bootstrap interval.
- Do not describe the two-stage Ritz-gap prototype as a consequence of this
  research.
- Stop theorem work when the first necessary component is proved
  budget-vacuous.
- Preserve all correct negative results; do not delete them when a new route is
  attempted.

The intended next action after review is Phase 2A only: a read-only,
artifact-first paired-difference feasibility audit.
