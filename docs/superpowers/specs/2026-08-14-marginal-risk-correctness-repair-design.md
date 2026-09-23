# Marginal-Risk Correctness Repair and Exact Risk-Bridge Design

**Date:** 2026-08-14

**Status:** Proposed after correctness audit; implementation requires written-spec approval

## 1. Objective and classification

The marginal-risk direction asks whether one additional range-finding direction is worth the residual probes it consumes. The exact full-rank oracle quantity is

$$
M(q)=(m-2q)\lambda_{q+1}^2-2T(q),
\qquad
T(q)=\sum_{i=q+1}^d\lambda_i^2.
$$

The project will preserve three distinct classifications:

- the algebraic marginal identity and its single-crossing consequence are `PROVED` on an explicit feasible domain;
- a decision rule supplied with valid simultaneous confidence radii is `PROVED UNDER AN EXPLICIT SIMULTANEOUS-CONFIDENCE EVENT`;
- the current Ritz-fit allocator remains `HEURISTIC` because no computable confidence radius is available and it extrapolates an estimated spectral model.

No fixed padding, guard radius, or confidence radius will be tuned from the previous held-out outcomes.

## 2. Chosen architecture

The implementation will use three isolated layers.

### 2.1 Exact mathematical layer

Pure helpers will evaluate the ideal full-rank marginal quantity and its sign on known spectra. These helpers support theorem regression tests and offline diagnostics; they do not claim that unknown eigenvalues are observable online.

### 2.2 Certificate decision layer

A pure three-state decision function will accept

```python
marginal_hat
radius
```

and return one of:

```text
increase   if marginal_hat - radius > 0
stop       if marginal_hat + radius < 0
learn_more otherwise
```

The function will validate finite inputs and a nonnegative radius. It implements only the deterministic implication on a stated confidence event. It will not construct or fit the radius.

### 2.3 Heuristic estimator layer

`Adaptive_Hutch_pplus_MarginalRisk` will be retained for backward compatibility and explicitly labeled as a heuristic Ritz-model prototype. It will not be described as the certified three-state policy.

The minimal behavioral repair is:

- count acquired pilot sketch columns separately from realized pilot rank;
- validate pilot, budget, threshold, probe distribution, and horizon inputs;
- support baseline-feasibility pilot capping using the same semantics as the sequential-pilot estimator;
- when a knee is resolved, stop pilot expansion and set the target to the already committed pilot width $B$, rather than retaining a distant power-law proposal;
- when the point-estimated marginal quantity at the current pilot floor is nonpositive, stop further heuristic pilot expansion instead of resetting the allocation at the next stage;
- retain the fitted power-law extrapolation only for the explicitly heuristic no-knee/positive-marginal branch;
- preserve pre-residual measurability, fresh residual probes, double projection, and exact rank-aware accounting.

This layer is intentionally not called baseline-safe. A point estimate of $M(q)$ is not a certificate.

## 3. Mathematical repair

### 3.1 Restore the near-oracle proof

`docs/proof_near_oracle_regret.md` must restore the removed definition of simultaneous tail confidence radius and Theorem 4, including its fixed-stage $2\varepsilon_R$ proof. The existing rank-aware corollary and Theorem 5 currently depend on these missing definitions.

### 3.2 Numbering

- Theorem 11 remains baseline-safe acceptance.
- Theorem 12 remains the marginal allocation identity.
- Lemma 12.1 will be the monotonicity/single-crossing lemma.
- The corrected exact step-spectrum result becomes Theorem 13.
- The existing Gaussian knee-detection specialization remains Theorem 15A.

### 3.3 Feasible-domain assumptions

For Theorem 12, require an integer $q$ such that both $q$ and $q+1$ are feasible:

$$
0\le q<d,
\qquad
m-2q-2>0.
$$

Let $\lambda_1\ge\cdots\ge\lambda_d\ge0$. Then

$$
\mathcal R(q+1)<\mathcal R(q)
\iff
M(q)>0.
$$

### 3.4 Single crossing

For every adjacent feasible pair,

$$
M(q+1)-M(q)
=(m-2q-2)(\lambda_{q+2}^2-\lambda_{q+1}^2)\le0.
$$

Thus $M(q)$ is nonincreasing, so the ideal full-rank risk decreases while $M(q)>0$ and increases after $M(q)<0$. Equality cases must be described as plateaus rather than unique minimizers.

### 3.5 Corrected step-spectrum theorem

Assume

$$
0<\eta<1,
\qquad
\lambda_i=1\ (i\le r_\star),
\qquad
\lambda_i=\eta\ (i>r_\star),
$$

with $0\le r_\star\le q_{\max}$ feasible. The ideal full-rank oracle has the unique minimizer $q^*=r_\star$ when

$$
2r_\star+2(d-r_\star)\eta^2<m<2d.
$$

The left inequality makes every pre-knee step beneficial; the right inequality makes every post-knee step harmful. Boundary equalities produce risk plateaus and must be treated separately.

The previous statement using only $m<2d$ will be marked false and replaced, not silently retained.

### 3.6 Certified marginal policy scope

For data-dependent stages and actions, confidence validity must be simultaneous over every stage/action pair that the policy may inspect, or be supplied by another valid sequential construction. Pointwise intervals are insufficient.

If the policy starts at $q_0$ and accepts only certified beneficial steps, it is baseline-safe for the enclosed risk by transitivity. If it starts below $q_0$ or stops early because of uncertainty, no baseline-safety claim follows without an additional comparison to $q_0$.

The full-rank criterion does not automatically extend to rank-aware actions $(q,r)$, where increasing $q$ need not increase $r$. That extension remains open.

## 4. Estimator API and accounting repair

Append optional parameters to preserve positional compatibility:

```python
preserve_baseline_feasibility=False
```

The implementation will record:

- `pilot_query_count` internally;
- `b_final=pilot_query_count`;
- `r_pilot_actual=Q_pilot.shape[1]`;
- `q_0`, requested/effective `b_max`, and pilot-cap status;
- the fitted marginal value at the pilot floor;
- the heuristic proposal before knee/marginal intervention;
- the final target and intervention reason;
- realized rank, residual count, stopping reason, and exact query count.

The exact identity remains

$$
q+r_{\mathrm{actual}}+\ell=m.
$$

No code path may derive range-finding query cost from realized rank.

## 5. Probe-identical equivalence repair

The existing test remains useful but will be corrected to use

$$
\ell=m-q_0-r_{\mathrm{actual}},
$$

not $m-2r_{\mathrm{actual}}$. It will add a rank-deficient case and will compare a shared implementation helper or actual estimator code path where practical, minimizing duplicated algorithm logic.

The empirical test will be described as a regression, not a general proof. The accompanying mathematical statement is that exact block orthogonalization and batch QR span the same sample range when neither procedure discards different directions through numerical rank thresholding.

## 6. Exact conditional-risk bridge

Create a new isolated script and preserve `results/risk_bridge_audit_summary.csv` as historical output.

The new diagnostic will import the frozen manifest and spectrum construction from `run_asymmetric_guard_heldout_benchmark.py`, thereby using exactly the same 24 spectra and matrix-orientation seeds.

The frozen production defaults are $d=500$, $m=160$, 200 randomized-basis trials per setup, allocation grid $q=0,1,\ldots,76$, basis seeds `60000 + trial_index`, and 20,000 paired-bootstrap resamples. CLI overrides exist only for reproducible smoke and sensitivity runs; production conclusions use these defaults.

For each setup, basis trial, and integer allocation on a predeclared feasible grid, draw one sketch matrix $S_{\max}$ and use nested prefixes $S_{\max}[:, :q]$. Let

$$
H_q=(I-Q_qQ_q^T)A(I-Q_qQ_q^T),
\qquad
\ell_q=m-q-r_q.
$$

Compute without residual-probe Monte Carlo:

$$
\mathcal R_G(Q_q)=\frac{2\|H_q\|_F^2}{\ell_q},
\qquad
\mathcal R_R(Q_q)=\frac{2}{\ell_q}\sum_{i\ne j}(H_q)_{ij}^2.
$$

For efficiency,

$$
\|H_q\|_F^2
=
\|A\|_F^2-2\|AQ_q\|_F^2+\|Q_q^TAQ_q\|_F^2,
$$

and the diagonal of $H_q$ will be computed from low-rank factors rather than forming dense projectors.

### Outputs

Write only after validation:

- `results/risk_bridge_exact_manifest.csv`;
- `results/risk_bridge_exact_trials.csv`;
- `results/risk_bridge_exact_curves.csv`;
- `results/risk_bridge_exact_minimizers.csv`.

Curves will include means, standard errors, and descriptive 95% normal intervals, computed as the sample mean plus or minus $1.96$ standard errors across randomized bases. Minimizer uncertainty will be assessed by paired percentile bootstrap over basis trials using the frozen bootstrap count. Results remain conditional on the fixed orientation for each setup, and multiple-comparison limitations will be stated.

The CLI will expose trial count, dimension, budget, bootstrap samples, allocation-grid endpoint, and output directory. Defaults will be frozen before the full run; smoke tests will use temporary output.

## 7. Tests and acceptance criteria

### Proof and helper tests

- verify Theorem 12 algebra numerically on random monotone spectra;
- verify $M(q)$ is nonincreasing;
- verify the corrected step theorem in its strict regime;
- verify boundary plateaus;
- retain the explicit counterexample showing $m<2d$ alone is insufficient;
- verify all three certificate decisions and invalid radii.

### Estimator tests

- rank-two and zero-matrix cases use exactly the requested budget;
- `b_final` and `r_pilot_actual` remain distinct;
- invalid pilot/horizon/probe configurations are rejected;
- a resolved $(r_\star,\eta)=(10,0.01)$ step case chooses the committed knee-resolution width rather than a distant extrapolation;
- every run satisfies $q+r_{\mathrm{actual}}+\ell=m$ and `oracle.query_count == m`;
- legacy estimators remain unchanged under identical seeds.

### Exact risk-bridge tests

- frozen manifest contains exactly the existing 24 unique cases;
- fast Frobenius and diagonal formulas match explicit dense $H_q$ on small matrices;
- Gaussian and Rademacher risks are nonnegative and finite;
- Rademacher risk equals zero for diagonal $H_q$ in the coordinate basis;
- Gaussian exact conditional risk matches high-sample residual Monte Carlo on a small fixture within a predeclared tolerance;
- output rows, keys, paired basis trials, seeds, and budgets are unique and aligned.

## 8. Verification sequence

1. Restore and audit proof dependencies and numbering.
2. Compile modified Python files.
3. Run targeted theorem/helper and marginal-estimator tests.
4. Run the full test suite with the existing `readline` workaround.
5. Run the exact risk bridge with two basis trials into `/private/tmp`.
6. Freeze defaults and execute the complete exact bridge.
7. Validate schemas, counts, finiteness, paired prefixes, and minimizer bootstrap output.
8. Run `git diff --check`.
9. Update `UROP_TRACKER.md`, workspace `memory.md`, and designated Obsidian notes with corrected theorem status and empirical conclusions.

## 9. Non-goals

- Do not invent or fit $C_b^M(q)$ during this implementation.
- Do not describe the heuristic estimator as certified or baseline-safe.
- Do not retune a fixed guard or knee padding from previous held-out results.
- Do not overwrite historical risk-bridge or guard artifacts.
- Do not claim that a minimizer observed on one fixed orientation is universal.
- Do not claim equivalence from failure to resolve a difference.

## 10. Expected decision after the bridge audit

The exact bridge will determine which target deserves confidence bounds:

- if ideal oracle and realized Gaussian risks align, spectral-tail confidence may be adequate for the Gaussian track;
- if they differ materially, confidence work should target realized residual energy rather than $T(q)$;
- if Gaussian and Rademacher risks differ materially, theory-aligned Gaussian and practical Rademacher policies must remain separate;
- only after this distinction is measured should the project choose the observable used by a future certified marginal allocator.
