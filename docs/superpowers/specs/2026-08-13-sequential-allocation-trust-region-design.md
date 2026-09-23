# Sequential Allocation Trust-Region Guard Design

**Date:** 2026-08-13

**Status:** Implemented and evaluated; fixed radius did not pass the empirical robustness criterion

**Classification:** `HEURISTIC` robustness safeguard, not a proved MSE guarantee

## 1. Motivation

The repaired 50-trial sequential-pilot benchmark separates two questions that the current implementation had conflated:

1. whether the pilot detects a spectral knee; and
2. whether the fitted tail model converts that information into a good query allocation.

For both step spectra, the sequential pilot found the exact knee in all 50 trials. Nevertheless, on Step $(r_\star=10,\eta=0.01)$ it selected mean allocation $(q,r,\ell)=(62,62,36)$ and produced an MSE ratio of $2.023$ relative to Standard Hutch++. On the exponential spectrum it selected mean $q\approx40.98$ and produced an MSE ratio of $3.844$. The immediate problem is therefore allocation robustness, not knee visibility.

The first intervention should be conservative, reversible, and auditable. It must preserve the original estimator as an unguarded comparison and must not be presented as a theorem before a finite-sample tail-risk confidence radius is available.

## 2. Goals and Non-Goals

### Goals

- Bound how far the sequential proposal may move from the Standard Hutch++ allocation.
- Preserve the original sequential stopping, Ritz-gap calculation, and tail-model proposal.
- Preserve exact query accounting $q+r_{\mathrm{actual}}+\ell=m$.
- Preserve conditional unbiasedness by making the guarded allocation before drawing fresh residual probes.
- Retain both the raw and guarded allocations in diagnostics.
- Compare Standard Hutch++, unguarded Sequential Pilot, and guarded Sequential Pilot using identical matrices and paired trial seeds.
- Save trial-level output so uncertainty intervals do not require reconstructing trials.

### Non-Goals

- Prove that the guard dominates Standard Hutch++ in MSE.
- Replace or strengthen Theorem 15A.
- Tune the guard radius on the same five benchmark spectra.
- Change the sketch distribution, knee threshold, stopping rule, fitted tail models, or residual estimator.
- Remove or silently alter the behavior of the existing unguarded estimator.

## 3. Guard Definition

Let

$$
q_0=\min\!\left\{q_{\max},\max\!\left(b_0,\left\lfloor\frac m3\right\rfloor\right)\right\}
$$

be the existing Standard-Hutch++ anchor, and let $q_{\mathrm{adapt}}$ be the allocation proposed by the current sequential tail-risk heuristic after stopping. For a nonnegative integer radius $s$, define

$$
L=\max\{b_{\mathrm{final}},q_0-s\},
\qquad
U=\min\{q_{\max},q_0+s\},
$$

and

$$
\boxed{
q_{\mathrm{safe}}=\operatorname{clip}(q_{\mathrm{adapt}},L,U).
}
$$

The first experiment fixes $s=\Delta b=4$, one sequential pilot increment. For $m=160$, $b_0=8$, and $\Delta b=4$, this restricts $q_{\mathrm{safe}}$ to $[49,57]$ whenever the pilot floor does not tighten the interval.

If a general parameter combination makes $L>U$, feasibility takes precedence over the trust region: set $q_{\mathrm{safe}}=\min\{q_{\max},b_{\mathrm{final}}\}$ and record that the pilot floor forced relaxation. The implementation must never reduce $q$ below the already acquired reusable pilot rank.

## 4. API and Data Flow

Extend `Adaptive_Hutch_pplus_SequentialPilot` with the optional keyword

```python
max_q_shift=None
```

- `None` preserves the current unguarded behavior exactly.
- A nonnegative integer activates the trust-region guard.
- Negative, non-integral, NaN, or infinite values raise `ValueError`.

The guard is applied after the sequential loop determines the raw proposal and before basis extension. No additional matrix-vector products are used.

Diagnostics will retain the existing keys and add:

- `q_0`: Standard-Hutch++ anchor;
- `q_adapt_raw`: unguarded proposal;
- `q_target`: final guarded or unguarded action;
- `max_q_shift`: configured radius or `None`;
- `guard_applied`: whether clipping changed the proposal;
- `guard_relaxed_for_pilot_floor`: whether feasibility overrode the nominal interval.

## 5. Correctness Ledger

### Exact query budget

The guard changes only `q_target`. Basis extension still uses $q_{\mathrm{target}}-b_{\mathrm{final}}$ new sketch directions, the realized basis rank remains $r_{\mathrm{actual}}$, and the residual count remains

$$
\ell=m-q_{\mathrm{target}}-r_{\mathrm{actual}}.
$$

The existing runtime assertion must continue to enforce exactly $m$ oracle queries.

### Unbiasedness

The guarded allocation is a deterministic function of pilot-stage data and fixed configuration, hence is measurable with respect to the pre-residual sigma-algebra $\mathcal G$. Fresh conditionally isotropic residual probes are drawn only after the guard is applied. Therefore the existing sequential-pilot unbiasedness proof continues to apply; the guard does not itself establish a variance improvement.

### Theoretical status

The trust region is `HEURISTIC`. The proved soft-shrinkage regret theorem requires a valid risk approximation event and a Lipschitz risk assumption. Neither supplies a numerical MSE guarantee for this fixed radius. Empirical non-inferiority on five spectra must not be promoted to a universal theorem.

## 6. Implementation Scope

Minimal source changes:

1. `src/trace_baseline.py`
   - add and validate `max_q_shift`;
   - retain `q_adapt_raw` before clipping;
   - apply the feasibility-aware trust region;
   - expose the new diagnostic fields.
2. `tests/test_new_estimators.py`
   - verify `None` preserves the current output and diagnostics under an identical seed;
   - verify the guarded action lies in the declared trust region;
   - verify invalid radii fail clearly;
   - verify exact query and residual-budget identities.
3. `experiments/run_sequential_pilot_guard_benchmark.py`
   - reuse the five existing deterministic matrix generators;
   - compare Standard Hutch++, unguarded Sequential Pilot, and guarded Sequential Pilot with $s=4$;
   - use the same 50 trial seeds for paired comparisons;
   - assert exact query count on every trial.
4. New result files, leaving the clean historical benchmark untouched:
   - `results/sequential_pilot_guard_trials.csv`;
   - `results/sequential_pilot_guard_summary.csv`.

No proof theorem text is changed unless the implementation audit reveals a missing assumption. The heuristic and empirical records will be updated after the benchmark.

## 7. Trial and Summary Outputs

The trial CSV will contain at least:

- setup, algorithm, trial seed, exact trace, estimate;
- squared error and absolute relative error;
- $b_{\mathrm{final}}$, $q_0$, $q_{\mathrm{adapt,raw}}$, $q_{\mathrm{target}}$, $r_{\mathrm{actual}}$, and $\ell$;
- whether the guard applied;
- maximum Ritz log gap, detected gap location, and stopping reason.

The summary CSV will report MSE, median relative error, mean allocations, guard activation rate, and paired-bootstrap 95% intervals for guarded/Standard and unguarded/Standard MSE ratios. Bootstrap resampling will use a fixed documented seed.

## 8. Evaluation Criteria

The guard passes its implementation audit only if:

1. every trial uses exactly $m=160$ oracle queries;
2. all guarded allocations satisfy feasibility and the trust-region rule unless a recorded pilot-floor relaxation is necessary;
3. the unguarded estimator reproduces the clean historical results up to floating-point serialization;
4. the default `max_q_shift=None` behavior is unchanged under identical randomness.

The empirical diagnostic is considered encouraging if the guarded method has no setup whose paired-bootstrap 95% MSE-ratio interval lies wholly above $1$, especially the exponential and Step $r_\star=10$ cases. Failure to resolve a difference is not proof of non-inferiority. The steep-power-law point estimate and interval will be reported as a secondary measure of how much adaptive upside the conservative guard retains.

## 9. Expected Interpretation

Three outcomes are informative:

- If the two resolved disadvantages disappear without creating a new one, the trust region is a useful conservative baseline for later confidence-calibrated allocation.
- If disadvantages persist even near $q_0$, sequential basis construction or pilot reuse—not only the scalar allocation—requires investigation.
- If robustness improves only by eliminating the steep-spectrum gain, the next step is a principled data-dependent radius based on a validated tail-risk uncertainty estimate rather than tuning a fixed radius on this benchmark.

## 10. Implementation Audit and Outcome

The implementation audit found and repaired a pre-existing rank-deficient accounting error: the code had treated realized pilot rank as the number of pilot sketch queries. These quantities agree on the full-rank benchmark but need not agree in general. The implementation now records stopped pilot query count as `b_final`, records realized rank separately as `r_pilot_actual`, and uses the former when computing basis-extension cost. A rank-two PSD counterexample that previously used 22 queries at budget 20 now satisfies the exact budget. All 26 collected tests pass.

The predeclared $s=4$ guard activated in every guarded trial. Guarded/Standard MSE ratios and paired-bootstrap 95% intervals were $0.825\,[0.483,1.405]$ for power law $c=2$, $1.549\,[0.985,2.407]$ for $c=0.5$, $2.111\,[1.112,3.830]$ for the exponential spectrum, $1.357\,[0.716,2.578]$ for Step $r_\star=10$, and $1.057\,[0.681,1.644]$ for Step $r_\star=20$. The guard therefore did not pass the empirical criterion because the exponential interval remains wholly above one.

A post-hoc exact-anchor diagnostic with $s=0$ gave ratios $0.900$, $1.303$, $0.984$, $0.979$, and $0.642$ in the same setup order, with every interval containing one. This does not validate $s=0$ as a tuned method; it indicates that the exponential failure is driven primarily by moving below the Standard anchor rather than by sequential basis construction alone. The exact exponential oracle surrogate supports the direction: $R(49)/R(53)=1.299$, whereas $R(41)/R(53)=2.299$. The next design should therefore use asymmetric or sensitivity-aware protection against under-allocation, evaluated on new or held-out spectra rather than tuned on these five cases.
