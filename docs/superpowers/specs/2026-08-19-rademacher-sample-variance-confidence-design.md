# Phase 1C: Sample-Variance Confidence Theorem and Usefulness Audit

**Status:** frozen design for implementation on 2026-08-19.

## 1. Purpose and research boundary

Phase 1C develops a scale-free finite-sample certificate for ordinary sample variance, the only Phase 1A estimator that passed its preregistered empirical gates, for one fixed and preconstructed candidate--baseline pair.

For actions $x\in\{a,0\}$, the theorem targets net improvement over the original, unstarted baseline,

\[
\frac{\sigma_a^2}{\ell_{\mathrm{paid}}}
\le
\frac{\sigma_0^2}{\ell_0},
\]

rather than merely candidate superiority after both actions have paid the same sunk costs.

This phase will:

- prove and implement the exact sample-variance moment identity;
- prove and implement the conservative $81+$Chebyshev two-action certificate;
- prove analytically that this baseline certificate is vacuous for all $s\le32$ at the primary joint failure probability;
- perform the Hoeffding decomposition before considering a degenerate U-statistic theorem;
- attempt a sharper scale-free theorem while preserving every imported constant and hypothesis;
- expose an unambiguous joint-confidence API;
- reuse the frozen Phase 1A/1B artifacts for a deterministic usefulness audit without issuing new matrix--vector queries;
- generate isolated Phase 1C proof, report, and result artifacts.

This phase will not:

- modify Hutch++ or introduce another allocator;
- assume an external spectral-norm, range, or scale bound;
- fit an unspecified theorem constant from Phase 1A data;
- charge or issue new certification probes;
- claim that abstention restores the original baseline;
- claim a complete candidate-first policy is baseline-safe;
- overwrite any Phase 1A, Phase 1B, bridge, or historical artifact;
- implement an online allocator.

## 2. Assumption ledger and estimand

Condition on the pre-certification sigma-algebra $\mathcal G$. For each action $x\in\{a,0\}$, let $Q_x\in\mathbb R^{d\times r_x}$ have orthonormal columns, put

\[
R_x=I-Q_xQ_x^T,
\]

and define

\[
X_{x,j}=g_j^TR_xAR_xg_j.
\]

The matrix $A\in\mathbb R^{d\times d}$ is symmetric. The probes $g_1,\ldots,g_s$ are conditionally independent and identically distributed coordinate-Rademacher vectors and are independent of $\mathcal G$. Define

\[
\sigma_x^2=\operatorname{Var}(X_{x,j}\mid\mathcal G).
\]

The unbiased sample variance is

\[
S_{x,s}^2
=
\frac1{s-1}\sum_{j=1}^s(X_{x,j}-\overline X_x)^2,
\qquad s\ge2.
\]

The paid candidate and original baseline risks are

\[
\mathcal R_{a,\mathrm{paid}}=\frac{\sigma_a^2}{\ell_{\mathrm{paid}}},
\qquad
\mathcal R_0^{\mathrm{original}}=\frac{\sigma_0^2}{\ell_0},
\]

where both denominators must be strictly positive.

## 3. Exact sample-variance baseline

Prove conditionally on $\mathcal G$ that

\[
\mathbb E[S_{x,s}^2\mid\mathcal G]=\sigma_x^2
\]

and

\[
\operatorname{Var}(S_{x,s}^2\mid\mathcal G)
=
\frac1s
\left[
\mu_{4,x}
-
\frac{s-3}{s-1}\sigma_x^4
\right],
\]

where

\[
\mu_{4,x}
=
\mathbb E[(X_{x,j}-\mathbb EX_{x,j})^4\mid\mathcal G].
\]

For the centered degree-two Rademacher chaos $Z_x=X_x-\mathbb EX_x$, retain the proved hypercontractive bound

\[
\mu_{4,x}\le81\sigma_x^4.
\]

No smaller uniform constant may replace (81) unless it is independently proved and adversarially audited. Consequently,

\[
\frac{\operatorname{Var}(S_{x,s}^2\mid\mathcal G)}{\sigma_x^4}
\le
\frac1s\left(80+\frac2{s-1}\right)
\]

on the positive-variance domain. The zero-variance case is handled directly, not by division.

## 4. Analytic budget-vacuity theorem

Chebyshev's inequality gives the pointwise statement

\[
\Pr\left(
|S_{x,s}^2-\sigma_x^2|
\ge\varepsilon\sigma_x^2
\mid\mathcal G
\right)
\le
\frac{80+2/(s-1)}{s\varepsilon^2}.
\]

The public failure probability is joint over both actions. The conservative certificate allocates

\[
\delta_{\mathrm{point}}=\frac{\delta_{\mathrm{joint}}}{2}
\]

to each action. Thus the two-action relative radius is

\[
\boxed{
\varepsilon_{\mathrm{Ch}}(s,\delta_{\mathrm{joint}})
=
\sqrt{
\frac{2[80+2/(s-1)]}{s\delta_{\mathrm{joint}}}
}.
}
\]

At $\delta_{\mathrm{joint}}=0.05$,

\[
\varepsilon_{\mathrm{Ch}}(32,0.05)
=
\sqrt{\frac{2(80+2/31)}{32(0.05)}}
>10.
\]

The squared radius

\[
f(s)=\frac{2}{\delta_{\mathrm{joint}}}
\left(\frac{80}{s}+\frac{2}{s(s-1)}\right)
\]

strictly decreases for every real (s>1), since both terms strictly decrease. Hence the radius also decreases, and

\[
\boxed{
\varepsilon_{\mathrm{Ch}}(s,0.05)>1
\quad\text{for every integer }2\le s\le32.
}
\]

This result is classified `PROVED BUT BUDGET-VACUOUS`: it is a valid simultaneous theorem, but it cannot produce a finite multiplicative lower bound anywhere on the frozen Phase 1C sample-size grid.

## 5. Hoeffding decomposition requirement

Write sample variance as

\[
S_s^2
=
\binom{s}{2}^{-1}\sum_{i<j}h(X_i,X_j),
\qquad
h(x,y)=\frac{(x-y)^2}{2}.
\]

With $\mu=\mathbb EX$, $\sigma^2=\operatorname{Var}(X)$, and an independent copy $X'$, prove

\[
h_1(x)
=
\mathbb E[h(x,X')]-\sigma^2
=
\frac{(x-\mu)^2-\sigma^2}{2},
\]

and

\[
h_2(x,y)
=
h(x,y)-\sigma^2-h_1(x)-h_1(y)
=
-(x-\mu)(y-\mu).
\]

It follows that

\[
S_s^2-\sigma^2
=
\frac2s\sum_{i=1}^s h_1(X_i)
+
\binom{s}{2}^{-1}\sum_{i<j}h_2(X_i,X_j).
\]

The first projection is generally nondegenerate, while (h_2) is canonical because

\[
\mathbb E[h_2(x,X')]=0
\]

for every fixed (x). Therefore Adamczak-type results for completely degenerate U-statistics may be applied only to the (h_2) term. The code and proof tests must reject any attempt to label the complete sample-variance kernel as degenerate.

For a sharper two-action attempt, use the fixed four-way allocation

\[
\delta_{x,\mathrm{linear}}
=
\delta_{x,\mathrm{degenerate}}
=
\frac{\delta_{\mathrm{joint}}}{4},
\qquad x\in\{a,0\}.
\]

These four probabilities sum exactly to \(\delta_{\mathrm{joint}}\).

## 6. Sharper scale-free theorem attempt

For the linear component, derive moment growth from the degree-two chaos $Z=X-\mathbb EX$. The variable

\[
h_1(X)=\frac{Z^2-\sigma^2}{2}
\]

is a degree-at-most-four Rademacher polynomial. An implementation-ready bound must use explicit, verified hypercontractive or Rosenthal constants.

For the degenerate component, verify every hypothesis of the chosen theorem and substitute

\[
h_2(X_i,X_j)=-Z_iZ_j.
\]

Every dimension factor, moment, operator norm, and numerical constant must remain explicit. The two components may be recombined only after each has a numerical bound.

If a cited result contains an unspecified universal constant, or if a needed scale-free reduction cannot be closed, classify this route `INCOMPLETE`. Do not calibrate the constant from empirical data. The theorem hierarchy is:

- `PROVED BUT BUDGET-VACUOUS`: exact (81+)Chebyshev theorem;
- `PROVED`: a sharper explicit scale-free theorem, only if all constants and assumptions close;
- `INCOMPLETE`: a route with unresolved constants or hypotheses;
- `REFUTED`: a counterexample violates a proposed bound.

## 7. Joint-confidence API

Implement:

```python
two_action_sample_variance_certificate(
    sample_size: int,
    joint_delta: float,
    method: str = "sharpest_proved_scale_free",
) -> TwoActionCertificate
```

The argument `joint_delta` always means the total simultaneous failure probability for both actions. It never means a pointwise probability.

The returned immutable certificate records:

- `sample_size`;
- `joint_delta`;
- `action_count = 2`;
- `per_action_delta = joint_delta / 2`;
- the internal Hoeffding-component split, if applicable;
- theorem name, method, and proof status;
- all numerical constants;
- the relative radius \(\varepsilon\);
- whether \(\varepsilon<1\);
- the exact simultaneous event.

Internal one-action helpers must call their input `pointwise_delta`. No public API may use an ambiguous argument named only `delta`.

The default method deterministically selects the smallest analytic radius among methods whose status is `PROVED` or `PROVED BUT BUDGET-VACUOUS`, with a fixed priority order recorded in code. An `INCOMPLETE` route is never selected as a certificate.

## 8. Net-safe decision

Implement:

```python
net_safe_decision(
    candidate_variance_hat: float,
    baseline_variance_hat: float,
    ell_paid: int,
    ell_original: int,
    certificate: TwoActionCertificate,
) -> "accept" | "abstain"
```

On the simultaneous event and when \(0\le\varepsilon<1\),

\[
\frac{S_x^2}{1+\varepsilon}
\le
\sigma_x^2
\le
\frac{S_x^2}{1-\varepsilon}.
\]

Therefore accept only if

\[
\boxed{
S_a^2
\le
\frac{1-\varepsilon}{1+\varepsilon}
\frac{\ell_{\mathrm{paid}}}{\ell_0}
S_0^2.
}
\]

The denominator factor is mandatory because the comparator is the original unstarted baseline, not the paid fallback.

Return `abstain` if:

- \(\varepsilon\ge1\);
- either denominator is nonpositive;
- (S_0^2=0);
- either estimate is negative or nonfinite;
- the certificate does not have a proved status;
- the net-safe inequality fails.

This certifies only an accepted candidate relative to the original baseline. Abstention after sunk construction and certification costs does not recover the original action.

## 9. Preregistered artifact audit

Reuse, without modifying:

- `results/direct_rademacher_certification_phase1a_trials.parquet`;
- `results/direct_rademacher_certification_phase1a_manifest.csv`;
- `results/direct_rademacher_certification_phase1b_budget_paths.parquet`;
- `results/direct_rademacher_certification_phase1b_budget_manifest.csv`.

No matrix--vector queries, new range sketches, or new certification probes are permitted.

Evaluate

\[
s\in\{4,8,16,32\},
\qquad
\delta_{\mathrm{joint}}\in\{0.01,0.05,0.10\},
\]

with primary $\delta_{\mathrm{joint}}=0.05$.

### Deterministic method and sample-size selection

1. For each $s$, select the proved method with the smallest analytic $\varepsilon(s,0.05)$. Resolve exact ties by the frozen method priority.
2. Define

   \[
   s_{\mathrm{gate}}
   =
   \min\{s\in\{4,8,16,32\}:\varepsilon(s,0.05)<1\}.
   \]

3. If the set is empty, issue `THEOREM ONLY / BUDGET-VACUOUS`. Do not inspect empirical performance to choose a sample size.
4. Evaluate the practical gate only at $s_{\mathrm{gate}}$.
5. Every other sample size is descriptive and cannot change the verdict.

### Practical gate

A method receives `THEOREM + PRACTICAL GO` only if the preregistered gate size has:

- a proved radius \(\varepsilon<1\);
- top-5% net-beneficial catastrophic detection at least (75\%\);
- a 10,000-replicate shared rank-stratified path-bootstrap lower endpoint at least (60\%\);
- a selected/original mean-risk bootstrap upper endpoint below (1\).

The bootstrap is conditional on the frozen path population. If no nonvacuous proved sample size exists, the practical gate is not evaluated and no empirical best-size search is performed.

Allowed overall verdicts are:

- `THEOREM + PRACTICAL GO`;
- `THEOREM ONLY / BUDGET-VACUOUS`;
- `INCOMPLETE`;
- `REFUTED`.

## 10. Implementation layout and isolated outputs

Add:

- `src/rademacher_sample_variance_confidence.py` for pure theorem and decision APIs;
- `experiments/postprocess_rademacher_sample_variance_confidence_phase1c.py` for the read-only artifact audit;
- `tests/test_rademacher_sample_variance_confidence_phase1c.py` for mathematical, API, artifact, and selection tests;
- `docs/proof_rademacher_sample_variance_confidence.md` for the detailed proof and theorem-status ledger;
- `reports/rademacher_sample_variance_confidence_phase1c.md` for the empirical/theoretical audit.

Create isolated outputs:

- `results/rademacher_sample_variance_confidence_phase1c_manifest.csv`;
- `results/rademacher_sample_variance_confidence_phase1c_certificate_grid.csv`;
- `results/rademacher_sample_variance_confidence_phase1c_gate.csv`;
- `results/rademacher_sample_variance_confidence_phase1c_verdict.csv`.

If a nonvacuous proved radius exists, also create a populated bootstrap artifact. Otherwise create an explicit zero-row bootstrap CSV with a stable schema and record `gate_evaluable=False`.

The manifest records source checksums, output checksums, software versions, theorem classifications, the method priority, the joint-failure contract, the deterministic sample-size rule, and timestamps. All final files are written only after validation succeeds.

## 11. Verification contract

Tests must cover:

- exact sample-variance expectation and variance by exhaustive enumeration;
- the (81)-constant moment implication on enumerated small Rademacher chaoses;
- analytic monotonicity and $s\le32$ vacuity at joint failure $0.05$;
- the Hoeffding decomposition term by term;
- degeneracy of (h_2) and a concrete nondegenerate (h_1);
- rejection of attempts to apply a degenerate theorem to the full kernel;
- exact joint, per-action, and component failure-probability sums;
- exhaustive simultaneous two-action coverage in tiny examples where practical;
- zero variance, zero estimates, \(\varepsilon\ge1\), invalid denominators, negative values, and nonfinite values;
- the algebraic distinction between paid-order and original-baseline acceptance;
- deterministic \(s_{\mathrm{gate}}\) selection independent of empirical outcomes;
- source checksums and historical-artifact immutability;
- compilation, targeted tests, the maintained suite, and `git diff --check`.

The proof audit must type every random object, state every conditioning and independence assumption, avoid division on the zero-variance boundary, and label each claim as `PROVED`, `PROVED BUT BUDGET-VACUOUS`, `INCOMPLETE`, `REFUTED`, or empirical.

## 12. Documentation and research record

After implementation and validation:

- update `UROP_TRACKER.md` with touched files, theorem statuses, artifact counts, verdict, and remaining Phase 1C/online gaps;
- update workspace `memory.md`;
- update `CURRENT_STATE.md`;
- synchronize the tracker, memory, proof note, report, and current-state record into `/Users/chenyixin/Documents/Obsidian_Vault/Swati_Summer_Research`;
- synchronize the tracker date with its newest running-log entry;
- preserve the completed UROP estimator boundary and describe Phase 1C as continuation research.

## 13. Frozen interpretation limits

- The conservative (81+)Chebyshev result may be mathematically valid and practically vacuous at the same time.
- A theorem for the canonical Hoeffding remainder does not control the linear projection.
- An unresolved universal constant is not numerical evidence and is not fitted from prior experiments.
- The public failure probability is always joint over exactly two fixed actions.
- The certificate does not cover data-dependent action search beyond the fixed pair.
- The net-safe acceptance inequality compares the paid candidate with the original baseline; paid fallback is a different object.
- Abstention does not undo sunk cost.
- Conditional bootstrap intervals do not become theorem-level probabilities.
- No empirical result may choose $s_{\mathrm{gate}}$.
- No allocator is implemented in this phase.
