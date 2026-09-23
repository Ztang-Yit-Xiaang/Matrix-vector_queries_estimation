# Baseline-Safe Asymmetric Adaptive Hutch++ Design

**Date:** 2026-08-14

**Status:** Approved and frozen before held-out evaluation

**Primary classification:** the fixed asymmetric guard is `HEURISTIC`; Theorem 11 is `PROVED UNDER AN EXPLICIT SIMULTANEOUS-CONFIDENCE EVENT`; held-out comparisons are `EMPIRICALLY OBSERVED`.

## 1. Research question

The development experiments showed that perfect step-knee detection does not guarantee a good allocation. The unguarded sequential estimator found the exact $r_\star=10$ knee in 50/50 trials but selected approximately $(q,\ell)=(62,36)$ and had about twice the Standard-Hutch++ MSE. A symmetric $|q-q_0|\le4$ guard reduced this failure but left a resolved exponential disadvantage. A post-hoc forced-$q_0$ diagnostic restored empirical parity.

The next question is therefore:

$$
\boxed{\text{Can Sequential Adaptive Hutch++ make only justified deviations from }q_0\text{ while retaining genuine upside?}}
$$

This stage combines a conservative asymmetric heuristic, a baseline-feasibility pilot cap, a conditional baseline-safety theorem, and a fully pre-specified held-out benchmark.

## 2. Frozen estimator design

Extend `Adaptive_Hutch_pplus_SequentialPilot` with:

```python
q_shift_bounds=None
preserve_baseline_feasibility=False
```

Backward compatibility is mandatory:

- no guard parameters preserves the unguarded path;
- `max_q_shift=s` continues to mean bounds $(-s,s)$;
- `q_shift_bounds=(-4,4)` is the explicit symmetric guard;
- `q_shift_bounds=(0,4)` is the asymmetric guard;
- `q_shift_bounds=(0,0)` forces the baseline;
- supplying both guard interfaces raises `ValueError`.

The explicit bounds must be a length-two tuple of signed integers $(s_-,s_+)$ satisfying $s_-\le s_+$. Booleans, floats, NaNs, infinities, malformed values, and reversed bounds are invalid.

When baseline feasibility is enabled, use

$$
q_0=\min\left\{q_{\max},\left\lfloor\frac m3\right\rfloor\right\},
$$

require $b_0\le q_0$, and cap the effective pilot horizon at $q_0$. For the held-out experiment, $(m,q_0,b_0,\Delta b)=(160,53,8,4)$ and requested `b_max=53`, so the attainable stages end at $B=52$.

For bounds $(s_-,s_+)$, define

$$
L=\max\{B,q_0+s_-\},\qquad U=\min\{q_{\max},q_0+s_+\},
$$

and

$$
q_{\mathrm{target}}=\operatorname{clip}(q_{\mathrm{adapt,raw}},L,U).
$$

The guard is applied before fresh residual probes are drawn. It is therefore measurable with respect to the pre-residual sigma-algebra $\mathcal G$ and does not disturb the existing conditional-unbiasedness proof.

Add diagnostics for requested/effective guard bounds, raw/safe shifts from $q_0$, requested/effective pilot caps, pilot-cap activation, and the baseline-feasibility flag. Retain `b_final` as pilot sketch-query count and `r_pilot_actual` as realized pilot rank.

## 3. Theorem 11: baseline-safe decisions

At a fixed pilot stage, let the finite nonempty feasible set $\mathcal Q_b$ contain $q_0$. Suppose nonnegative radii $e_b(q)$ satisfy the simultaneous event

$$
E_b=\left\{\forall q\in\mathcal Q_b:\left|\widehat{\mathcal R}_b(q)-\mathcal R(q)\right|\le e_b(q)\right\}.
$$

Define

$$
U_b^R(q)=\widehat{\mathcal R}_b(q)+e_b(q),\qquad
L_b^R(q)=\widehat{\mathcal R}_b(q)-e_b(q),
$$

and the certified set

$$
\mathcal C_b=\{q\in\mathcal Q_b:U_b^R(q)\le L_b^R(q_0)\}.
$$

Use $q_0$ if $\mathcal C_b$ is empty; otherwise choose a measurable member, canonically a minimizer of $U_b^R$. On $E_b$,

$$
\mathcal R(q_{\mathrm{safe}})
\le U_b^R(q_{\mathrm{safe}})
\le L_b^R(q_0)
\le \mathcal R(q_0).
$$

Thus $\Pr(E_b)\ge1-\delta$ implies baseline non-inferiority for the risk quantity enclosed by the bounds with probability at least $1-\delta$. Simultaneous validity is essential because the candidate is selected from the data.

The proof note will also include:

- a risk-difference corollary: $|\widehat\Delta_b(q)-\Delta(q)|\le C_b(q)$ and $\widehat\Delta_b(q)+C_b(q)\le0$ imply $\mathcal R(q)\le\mathcal R(q_0)$;
- a rank-aware corollary comparing actions $a=(q,r)$ and $a_0=(q_0,r_0)$ under $2T(r)/(m-q-r)$;
- a sequential-stage corollary using per-stage failure levels whose sum is at most $\delta$.

The result is a deterministic implication on a confidence event. No computable $e_b$ or $C_b$ is supplied here. A certificate for an oracle surrogate is not automatically a certificate for actual estimator MSE.

## 4. Frozen held-out suite

Use $d=500$, $m=160$, 200 trials, 20,000 bootstrap resamples, trial seeds `50000 + trial`, and orientation seeds `42000 + setup_index`. One fixed random orientation is used per setup, so results are conditional on those orientations.

Compare five methods:

1. Standard Hutch++;
2. Sequential forced baseline $(0,0)$;
3. Sequential unguarded with $B\le q_0$;
4. Sequential symmetric guard $(-4,4)$;
5. Sequential asymmetric guard $(0,4)$.

All sequential methods use identical pilot parameters and requested `b_max=q_0`; only the mapping from raw proposal to final $q$ differs.

Use 24 spectra:

- power laws $\lambda_i=i^{-c}$ for $c\in\{0.3,0.7,1.2,1.7,2.5\}$;
- exponentials $\lambda_i=e^{-\alpha i}$ for $\alpha\in\{0.02,0.08,0.10,0.15\}$;
- all 12 steps from $r_\star\in\{5,15,25,30\}$ and $\eta\in\{0.001,0.05,0.1\}$;
- smooth elbow $\lambda_i=(1+(i/20)^4)^{-1/2}$ normalized by $\lambda_1$;
- mixture $\lambda_i=\tfrac12i^{-1.2}+\tfrac12e^{-0.08(i-1)}$ normalized by $\lambda_1$;
- monotone log-normal shape $\lambda_i=e^{-0.35(\log i)^2}$.

Every spectrum must be finite, positive, and nonincreasing. Construct $A=(Q\operatorname{diag}(\lambda))Q^T$ with deterministic Gaussian QR and use $\sum_i\lambda_i$ as the exact trace.

The experiment totals 24,000 estimator trials. The five development spectra are not reused as held-out parameter settings.

## 5. Benchmark interface and artifacts

The isolated benchmark accepts `--trials`, `--dimension`, `--budget`, `--bootstrap-samples`, and `--output-dir`. Defaults run the complete experiment; smoke tests write only to a temporary directory.

Write four CSVs only after all trials succeed:

- `asymmetric_guard_heldout_manifest.csv`;
- `asymmetric_guard_heldout_trials.csv`;
- `asymmetric_guard_heldout_summary.csv`;
- `asymmetric_guard_delta_q_strata.csv`.

Trial rows contain spectrum metadata, matrix/trial seeds, estimate and errors, $B$, pilot rank, $q_0$, raw and final $q$, raw and safe $\Delta q$, realized rank, residual count, intervention flags, stopping/gap diagnostics, and query count.

Summary rows contain MSE, median relative error, allocation summaries, intervention rates, MSE ratio versus Standard, and a paired-bootstrap 95% interval. Intervals are per-comparison exploratory intervals without familywise error control.

Stratify sequential trials by raw $\Delta q<0$, $=0$, and $>0$. This is descriptive evidence, not a causal analysis.

## 6. Required assertions and tests

Unit tests cover backward compatibility, equivalence of old and explicit symmetric APIs, conflicting/invalid bounds, forced baseline, asymmetric bounds, baseline pilot caps, the $B=52$ stage-grid endpoint, infeasible $b_0$, rank-deficient query accounting, spectrum validity, and exactly 24 manifest entries.

Every benchmark trial must satisfy

$$
q+r_{\mathrm{actual}}+\ell=m
$$

and `oracle.query_count == m`. Method-specific guard bounds and $B\le q_0$ are asserted. All five methods must have exactly the requested number of aligned trial indices per setup, with no nonfinite estimates or errors.

Verification order: compile, targeted tests, full test suite, temporary smoke run, complete held-out run, CSV schema/count validation, `git diff --check`, and line-by-line proof audit.

## 7. Interpretation and recordkeeping

The primary heuristic question is whether the asymmetric guard avoids resolved held-out disadvantages while preserving upward gains. The forced-baseline control diagnoses sequential basis construction separately from allocation. Raw-shift strata test whether negative proposals align with exponential failures and large positive proposals align with low-rank step failures.

Failure to resolve a difference is not equivalence. Multiple per-setup intervals do not yield a familywise claim. No hyperparameter or spectrum is changed after results are observed.

After completion, update `UROP_TRACKER.md`, workspace `memory.md`, and the corresponding Obsidian proof and research notes. The active theory direction becomes confidence-certified improvement over Standard Hutch++, while Theorem 15 remains supporting knee-detection theory.
