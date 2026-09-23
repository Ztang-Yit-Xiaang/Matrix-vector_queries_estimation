# Rank-Aware Marginal Theory and Multi-Budget Rank-Deficient Risk-Bridge Design

**Date:** 2026-08-14

**Status:** Approved mathematical direction; revised written specification awaiting final user review

## 1. Objective and scientific scope

The completed exact conditional-risk bridge shows that, on the frozen 24 strictly positive spectra, every randomized range prefix retained full numerical rank:

$$
r_q=q.
$$

Consequently, its full-rank oracle and rank-aware oracle levels coincide identically. That experiment cannot answer what happens when a sketch query fails to create a new numerical basis direction. The next implementation cycle will therefore preserve the completed bridge and add an isolated rank-deficient supplement.

The scientific angle of this cycle is deliberately simple:

$$
\boxed{
\text{We are testing what changes when a matrix--vector query does not necessarily buy one useful low-rank direction.}
}
$$

The supplement addresses three questions:

1. When does a successful rank gain justify consuming both a sketch query and a basis-evaluation query?
2. What is the exact cost of a sketch query that fails to increase numerical rank?
3. How do these conclusions and the risk-minimizing allocation change with the total query budget $m$?

The active research question remains whether adaptive Hutch++ can make only justified allocation deviations. The rank-aware marginal theory and bridge are diagnostic foundations for a future certificate. They do not provide an online confidence radius and do not promote the current marginal estimator from `HEURISTIC` status.

Existing correct code, proof text, historical CSV files, and the frozen exact bridge artifacts must remain intact except for minimal cross-references required by the new theorem and corollary statements.

## 2. Classification ledger

The implementation and documentation must use the following labels.

- `PROVED`: Theorem 14, Corollary 14.1, Lemma 14.2, and the exact conditional-to-unconditional MSE corollary, under their explicitly stated domains.
- `EMPIRICALLY OBSERVED`: numerical-rank acceptance paths, risk curves, minimizers, regret curves, bootstrap minimizer frequencies, and budget sensitivity on the frozen matrices and seeds.
- `HEURISTIC`: the current Ritz-fit marginal allocator and any use of the offline marginal diagnostic as an online decision proxy.
- `OPEN`: a finite-sample computable simultaneous confidence bound for realized Gaussian or Rademacher risk differences.

The four bridge risks are diagnostic levels. They do not form a universal monotone inequality chain.

## 3. Assumption and notation ledger

Let

$$
A\in\mathbb R^{d\times d},
\qquad
A=A^T,
\qquad
A\succeq0,
$$

with ordered eigenvalues

$$
\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_d\ge0.
$$

Uppercase $S$ denotes a sketching matrix. For a $q$-column sketch prefix $S_q$, let $Q_q\in\mathbb R^{d\times r_q}$ be the nested accepted basis and let

$$
r_q=\operatorname{rank}_{\mathrm{numerical}}(AS_q),
\qquad
0\le r_q\le q.
$$

The query budget is split as

$$
q+r_q+\ell_q=m,
\qquad
\ell_q=m-q-r_q.
$$

Here $q$ products form $AS_q$, $r_q$ additional products form $AQ_q$, and $\ell_q$ products would be used by fresh residual probes in the corresponding estimator. The bridge evaluates exact conditional variance analytically and therefore does not execute those $\ell_q$ residual products.

Define the spectral tail energy

$$
T(r)=\sum_{i=r+1}^d\lambda_i^2
$$

and, for a state $(q,r)$, define the remaining residual-query denominator

$$
D=m-q-r.
$$

For an orthonormal basis $Q$, let

$$
R_Q=I-QQ^T,
\qquad
H_Q=R_QAR_Q.
$$

Define the exact Gaussian and Rademacher residual energies by

$$
E_G(Q)=\|H_Q\|_F^2,
\qquad
E_R(Q)=\sum_{i\ne j}(H_Q)_{ij}^2.
$$

All marginal comparisons below require both the old and new residual counts to be strictly positive.

## 4. Mathematical specification

### 4.1 Theorem 14: ideal rank-aware successful marginal

Assume $0\le r\le q$, $r<d$, and

$$
D=m-q-r>2.
$$

Consider an ideal successful action

$$
(q,r)\longrightarrow(q+1,r+1)
$$

whose new direction captures exactly the eigenvector associated with $\lambda_{r+1}$. The old and new ideal rank-aware risks are

$$
\mathcal R_{\mathrm{rank}}(q,r)=\frac{2T(r)}{D}
$$

and

$$
\mathcal R_{\mathrm{rank}}(q+1,r+1)
=
\frac{2\left(T(r)-\lambda_{r+1}^2\right)}{D-2}.
$$

Their exact difference is

$$
\mathcal R_{\mathrm{rank}}(q+1,r+1)
-
\mathcal R_{\mathrm{rank}}(q,r)
=
-\frac{2\left[D\lambda_{r+1}^2-2T(r)\right]}{D(D-2)}.
$$

Since $D(D-2)>0$, the new action improves ideal rank-aware risk if and only if

$$
\boxed{
D\lambda_{r+1}^2>2T(r).
}
$$

Equivalently,

$$
\boxed{
(m-q-r)\lambda_{r+1}^2>2T(r).
}
$$

This is an ideal eigenvector statement. Numerical rank gain alone does not imply that the new direction removes exactly $\lambda_{r+1}^2$ of spectral tail energy.

### 4.2 Corollary 14.1: failed-rank action

Assume $D=m-q-r>1$. Consider a sketch query that does not change the accepted basis:

$$
(q,r)\longrightarrow(q+1,r).
$$

The tail energy remains $T(r)$ while the residual count falls from $D$ to $D-1$. Therefore

$$
\begin{aligned}
\mathcal R_{\mathrm{rank}}(q+1,r)
-\mathcal R_{\mathrm{rank}}(q,r)
&=
\frac{2T(r)}{D-1}-\frac{2T(r)}{D}\\
&=
\boxed{\frac{2T(r)}{D(D-1)}}\ge0.
\end{aligned}
$$

The inequality is strict exactly when $T(r)>0$. If the current basis already captures the entire range of an exactly rank-deficient matrix, then $T(r)=0$ and both risks equal zero. The corollary must not claim strict harm in this boundary case.

### 4.3 Lemma 14.2: realized energy-drop marginal

Let $X\in\{G,R\}$ denote Gaussian or Rademacher conditional risk. Let $Q$ be the old nested basis, let $Q'$ be the new nested basis after one accepted direction, and assume

$$
D=m-q-r>2.
$$

Define

$$
\mathcal R_X(Q;q,r)=\frac{2E_X(Q)}{D}.
$$

For the successful action $(q,r,Q)\to(q+1,r+1,Q')$, the exact risk difference is

$$
\mathcal R_X(Q';q+1,r+1)-\mathcal R_X(Q;q,r)
=
-\frac{2M_X(Q,Q';q,r)}{D(D-2)},
$$

where

$$
\boxed{
M_X(Q,Q';q,r)
=
D\left[E_X(Q)-E_X(Q')\right]-2E_X(Q).
}
$$

Consequently,

$$
\boxed{
\mathcal R_X(Q';q+1,r+1)<\mathcal R_X(Q;q,r)
\iff
M_X(Q,Q';q,r)>0.
}
$$

If $E_X(Q)>0$, this criterion has the equivalent fractional-energy form

$$
\boxed{
\frac{E_X(Q)-E_X(Q')}{E_X(Q)}
>
\frac{2}{D}.
}
$$

Thus an accepted direction is beneficial exactly when the fraction of residual energy it removes exceeds the fractional residual-sample capacity lost when $D$ decreases to $D-2$. This division is valid only when $E_X(Q)>0$. If $E_X(Q)=0$, the fraction is undefined and strict improvement from zero risk is impossible.

For a failed-rank action, $Q'=Q$, the denominator changes from $D$ to $D-1$, and

$$
\boxed{
\mathcal R_X(Q;q+1,r)-\mathcal R_X(Q;q,r)
=
\frac{2E_X(Q)}{D(D-1)}\ge0.
}
$$

The ideal specialization $E_X(Q)=T(r)$ and $E_X(Q)-E_X(Q')=\lambda_{r+1}^2$ recovers Theorem 14. The experiment must use the realized $E_G$ and $E_R$ identities when assessing actual nested bases.

### 4.4 Exact conditional risk and unconditional MSE

Let $\mathcal G$ be the pre-residual sigma-algebra fixing $(Q,q,r,\ell)$, with $\ell=m-q-r>0$. Assume residual probes are fresh and conditionally independent of $\mathcal G$.

Conditional unbiasedness gives

$$
\mathbb E[\widehat t\mid\mathcal G]=\operatorname{tr}(A).
$$

For Gaussian probes,

$$
\operatorname{Var}(\widehat t\mid\mathcal G)
=
\frac{2E_G(Q)}{\ell}.
$$

For Rademacher probes,

$$
\operatorname{Var}(\widehat t\mid\mathcal G)
=
\frac{2E_R(Q)}{\ell}.
$$

The law of total variance then yields

$$
\begin{aligned}
\mathbb E\!\left[(\widehat t-\operatorname{tr}(A))^2\right]
&=
\mathbb E\!\left[\operatorname{Var}(\widehat t\mid\mathcal G)\right]
+
\mathbb E\!\left[
(\mathbb E[\widehat t\mid\mathcal G]-\operatorname{tr}(A))^2
\right]\\
&=
\mathbb E\!\left[\operatorname{Var}(\widehat t\mid\mathcal G)\right].
\end{aligned}
$$

Thus the mean exact conditional Gaussian or Rademacher risk over randomized bases equals the corresponding estimator MSE. This conclusion depends on conditional unbiasedness and fresh residual probes; it does not apply automatically to a biased or probe-reusing procedure.

### 4.5 Budget sensitivity implied by the formulas

For a fixed state $(q,r)$ and fixed spectrum, increasing $m$ increases

$$
D=m-q-r.
$$

The successful ideal condition

$$
D\lambda_{r+1}^2>2T(r)
$$

therefore becomes easier to satisfy. The failed-rank increment

$$
\frac{2T(r)}{D(D-1)}
$$

decreases as $m$ increases but remains nonnegative. These are pointwise statements for a fixed state. They do not, without additional single-crossing arguments, prove a universal monotonicity theorem for the global minimizer of every realized risk curve.

## 5. Existing-bridge postprocessing

The completed artifacts

- `results/risk_bridge_exact_manifest.csv`;
- `results/risk_bridge_exact_trials.csv`;
- `results/risk_bridge_exact_curves.csv`;
- `results/risk_bridge_exact_minimizers.csv`

must remain byte-for-byte unchanged.

Add a read-only postprocessor that consumes the completed trial CSV and writes:

- `results/risk_bridge_exact_regret_curves.csv`;
- `results/risk_bridge_exact_minimizer_frequencies.csv`.

The postprocessor must not reconstruct matrices or randomized bases. It must validate the input schema, complete paired trial/allocation grid, finite risks, and unique keys before producing any output.

For each setup and each risk type, it will report the mean curve, additive regret, baseline-normalized additive regret, multiplicative regret when defined, plateau information, and bootstrap minimizer frequency. The deterministic tie rule is the smallest minimizing $q$.

## 6. Rank-deficient multi-budget supplement

### 6.1 Frozen spectra

Use $d=500$ and the full cross-product

$$
r_\star\in\{5,15,30\},
\qquad
\eta\in\{0,10^{-14},10^{-10},10^{-6}\}.
$$

For $i=1,\ldots,d$, define

$$
\lambda_i=
\begin{cases}
1,&i\le r_\star,\\
\eta,&i>r_\star.
\end{cases}
$$

The roles of the four tail levels are:

- $\eta=0$: exact rank deficiency and possible exact zero risk after full signal-range capture;
- $\eta=10^{-14}$: positive mathematical tail below the frozen numerical-rank tolerance;
- $\eta=10^{-10}$: transition/control case above the default tolerance but still extremely small;
- $\eta=10^{-6}$: clearly full-rank low-noise control.

Every spectrum must be finite, nonnegative, and nonincreasing. Positivity is not required because $\eta=0$ is intentional.

### 6.2 Frozen budgets and feasible grids

The primary budget remains

$$
m=160.
$$

The budgets

$$
m\in\{80,240\}
$$

are pre-specified sensitivity analyses. No scientific headline may pool these roles without labeling the primary and sensitivity results separately.

Fix the minimum worst-case residual count at

$$
\ell_{\min}=8.
$$

For every budget, derive rather than hand-code the allocation endpoint:

$$
q_{\max}(m)
=
\min\!\left\{d,
\left\lfloor\frac{m-\ell_{\min}}{2}\right\rfloor
\right\}.
$$

This yields:

| Analysis role | $m$ | $q$ grid | Worst-case $\ell_q$ when $r_q=q$ |
|---|---:|---:|---:|
| Low-budget sensitivity | 80 | $0,\ldots,36$ | 8 |
| Primary | 160 | $0,\ldots,76$ | 8 |
| High-budget sensitivity | 240 | $0,\ldots,116$ | 8 |

The worst-case full-rank cap gives a common feasible grid independent of the random accepted rank. The experiment will not enlarge $q_{\max}$ post hoc when rank loss creates additional feasibility.

### 6.3 Fixed orientations and paired nested sketches

For each fixed $r_\star$, use one deterministic Gaussian-QR orientation of the $r_\star$-dimensional signal subspace and reuse that exact $U_\star$ for all four tail levels $\eta$. Because every tail eigenvalue equals $\eta$, the orientation of its orthogonal complement is irrelevant. Order the step ranks as supplied by the validated `--step-ranks` argument, let `rank_index` be the zero-based index in that order, freeze

```python
ORIENTATION_SEED_BASE = 52_000
```

and assign

```python
orientation_seed = ORIENTATION_SEED_BASE + rank_index
```

The four matrices at a fixed $r_\star$ are therefore

$$
A_\eta=\eta I+(1-\eta)U_\star U_\star^T
$$

with the same $U_\star$. Their only mathematical difference is $\eta$. Record both `orientation_group_id` and `orientation_seed`; if `matrix_seed` is retained for compatibility, it must equal `orientation_seed`. Different step ranks use separate orientation groups. The optional construction of one nested master basis across different $r_\star$ values is outside the frozen design.

Use 200 randomized-basis trials per spectrum. Freeze

```python
BASIS_SEED_BASE = 70_000
```

and assign `basis_seed = BASIS_SEED_BASE + basis_trial`.

For each spectrum and trial, draw one Rademacher sketch

$$
S_{\max}\in\{-1,+1\}^{d\times q_{\max,\mathrm{all}}},
\qquad
q_{\max,\mathrm{all}}=\max_m q_{\max}(m).
$$

For the production defaults, $q_{\max,\mathrm{all}}=116$. All budgets use prefixes of the same $S_{\max}$. For a fixed $r_\star$ and basis trial, the four $\eta$ levels also use the same $S_{\max}$. Therefore orientation group, basis-trial index, basis seed, and every common sketch prefix are paired across budgets and across $\eta$ within a fixed $r_\star$.

The results are conditional on the three fixed signal-subspace orientations. For a fixed realized residual operator $H_Q$, Gaussian conditional risk depends on $\|H_Q\|_F^2$, which is invariant under simultaneous orthogonal coordinate changes. Exact Rademacher conditional risk depends on the off-diagonal entries of $H_Q$ in the sampling coordinate basis and is not rotationally invariant. Accordingly, a difference between $\mathcal R_G$ and $\mathcal R_R$ may reflect Rademacher orientation sensitivity as well as the change of probe distribution.

### 6.4 Nested rank-aware basis construction

Starting from $Q_0$ empty, process the columns of $S_{\max}$ one at a time. For a new column $s_j$:

1. compute $y_j=As_j$;
2. apply two passes of projection against the accepted basis $Q_{j-1}$;
3. pass the residual column through the existing `_rank_aware_qr` tolerance contract, using the running unprojected sampled-range Frobenius scale as `reference_scale`;
4. if the column is rejected, set $Q_j=Q_{j-1}$ and $r_j=r_{j-1}$;
5. if the column is accepted, append the normalized direction, increment $r_j$ by one, and compute its corresponding $Aq_{r_j}$ product for exact low-rank/residual-energy updates.

This construction guarantees nested bases, nondecreasing accepted rank, and

$$
r_j-r_{j-1}\in\{0,1\}.
$$

The QR threshold is frozen at

```python
QR_RTOL = 1e-12
QR_ATOL = 0.0
```

for the production run. Every output row records both values. Changing either value defines a different numerical-rank experiment and must produce a separate manifest/output directory rather than overwrite the frozen results.

### 6.5 Four conditional-risk levels

For every spectrum, basis trial, budget, and feasible $q$, record:

**1. Full-rank ideal oracle risk**

$$
\mathcal R_{\mathrm{full}}(q;m)
=
\frac{2T(q)}{m-2q}.
$$

**2. Numerical-rank oracle risk**

$$
\mathcal R_{\mathrm{rank}}(q,r_q;m)
=
\frac{2T(r_q)}{m-q-r_q}.
$$

**3. Exact realized Gaussian conditional risk**

$$
\mathcal R_G(Q_q;m)
=
\frac{2E_G(Q_q)}{m-q-r_q}.
$$

**4. Exact realized Rademacher conditional risk**

$$
\mathcal R_R(Q_q;m)
=
\frac{2E_R(Q_q)}{m-q-r_q}.
$$

The script evaluates these conditional risks exactly from the nested basis. It does not run residual-probe Monte Carlo for every row.

### 6.6 Query-accounting interpretation

Every row must satisfy the hypothetical estimator accounting identity

$$
q+r_q+\ell_q=m.
$$

The bridge reuses one nested basis across many allocations and budgets and evaluates the residual variance analytically. Therefore its construction-time `MatVecOracle.query_count` is not expected to equal $m$ on every saved row. It tracks only the actually constructed $AS$ and $AQ$ products. The saved field `accounted_total_queries` must equal $m$, while `constructed_basis_queries` must equal $q+r_q$ for each prefix.

No dummy residual products may be issued merely to make an oracle counter equal $m$. Separate estimator regressions continue to require `oracle.query_count == m`.

## 7. Regret, zero minima, and minimizer definitions

For every risk type $X\in\{\mathrm{full},\mathrm{rank},G,R\}$, budget $m$, setup, and mean risk curve, define

$$
\mathcal R_{X,\min}(m)=\min_q\mathcal R_X(q;m).
$$

### 7.1 Additive regret

Always record

$$
\boxed{
C_X^{\mathrm{add}}(q;m)
=
\mathcal R_X(q;m)-\mathcal R_{X,\min}(m).
}
$$

This quantity is defined even when the minimum is zero.

### 7.2 Baseline-normalized additive regret

Record

$$
\boxed{
C_{X,0}(q;m)
=
\frac{\mathcal R_X(q;m)-\mathcal R_{X,\min}(m)}
{\mathcal R_X(0;m)}.
}
$$

The denominator is the same-budget, no-low-rank reference. The production manifest must validate $\mathcal R_X(0;m)>0$ for every configured curve. If a future override produces $\mathcal R_X(0;m)=0$, this normalized field is `NaN` with an explicit `zero_reference` flag. No small constant may be inserted into the denominator.

### 7.3 Multiplicative regret

When $\mathcal R_{X,\min}(m)>0$, record

$$
C_X^{\mathrm{mult}}(q;m)
=
\frac{\mathcal R_X(q;m)}{\mathcal R_{X,\min}(m)}-1.
$$

When the mathematical minimum is zero, multiplicative regret is undefined and must be stored as `NaN`. Replacing the zero denominator by an arbitrary $\varepsilon$ would define a different regularized score and is prohibited in the primary artifacts.

Raw risk levels must not be used to assess allocation quality across different budgets: increasing $m$ mechanically lowers estimator variance. Cross-budget conclusions must instead use additive regret, $\mathcal R_X(0;m)$-normalized additive regret, minimizer shifts, rank-acceptance behavior, or rejected-query penalties.

### 7.4 Numerical zero policy

The experiment must distinguish exact mathematical zero from a small positive tail.

- Only an explicitly exact-rank configuration with $\eta=0$, full signal-rank capture $r_q=r_\star$, and a passed backward-error residual check may canonicalize roundoff-sized realized energy to zero.
- The positive cases $\eta\in\{10^{-14},10^{-10},10^{-6}\}$ must not be classified as exact zero merely because their risk lies below a generic absolute tolerance.
- The raw pre-canonicalization energy and the backward-error ratio must remain in the trial output.

### 7.5 Plateaus and tie breaking

For every mean curve, record:

- smallest minimizing $q$;
- largest minimizing $q$;
- number of minimizing allocations;
- whether the minimum is zero;
- whether the minimum is a plateau.

Use the smallest minimizing $q$ as the deterministic representative in all summaries and bootstrap resamples. Plateau fields retain the information lost by deterministic tie breaking.

## 8. Marginal-transition diagnostics

For every allocation row with $q>0$, record the rank efficiency

$$
\boxed{
\operatorname{rank\_efficiency}(q)=\frac{r_q}{q}
}
$$

and record `NaN` at $q=0$. Do not insert a small denominator. Also record the canonical rejected-query count

$$
\boxed{
N_{\mathrm{fail}}(q)=q-r_q.
}
$$

Under the nested-prefix contract $r_0=0$ and $r_j-r_{j-1}\in\{0,1\}$,

$$
N_{\mathrm{fail}}(q)
=
\sum_{j=1}^q\mathbf 1\{r_j=r_{j-1}\}
=q-r_q.
$$

Therefore `rejected_query_count=q-r_q` is the single stored count; the transition sum is an asserted identity, not a redundant second data field.

For every transition from $q-1$ to $q$, record:

- old and new $q$;
- old and new accepted ranks;
- `rank_gain_accepted`;
- old and new $\ell$ for each budget;
- old and new values of all four risks;
- exact risk differences;
- the full-rank marginal from Theorem 12;
- the ideal rank-aware marginal from Theorem 14 on successful rank transitions;
- the realized $M_G$ and $M_R$ from Lemma 14.2 on successful rank transitions;
- the Corollary 14.1 failed-rank increment on rejected transitions;
- numerical identity residuals for every applicable formula.

For each risk level $X\in\{\mathrm{rank},G,R\}$, also accumulate the exact failed-transition increments through allocation $q$. If transition $j$ is rejected, define its old state by

$$
D_{j-1}=m-(j-1)-r_{j-1}.
$$

Its contributions are

$$
\Delta\mathcal R_{\mathrm{fail},\mathrm{rank},j}
=
\frac{2T(r_{j-1})}{D_{j-1}(D_{j-1}-1)}
$$

and

$$
\Delta\mathcal R_{\mathrm{fail},X,j}
=
\frac{2E_X(Q_{j-1})}{D_{j-1}(D_{j-1}-1)},
\qquad X\in\{G,R\}.
$$

Store their cumulative sums as `cumulative_failed_penalty_rank`, `cumulative_failed_penalty_gaussian`, and `cumulative_failed_penalty_rademacher`. These are descriptive path decompositions. They need not equal total regret because successful transitions and changing denominators also affect the risk path.

An ideal rank-aware marginal must not be described as realized spectral removal. The realized Gaussian and Rademacher energy drops are the correct transition diagnostics for the actual nested basis.

## 9. Output artifacts and schemas

Write the rank-deficient artifacts only after all trials and validations succeed:

- `results/risk_bridge_rank_deficient_manifest.csv`;
- `results/risk_bridge_rank_deficient_trials.csv`;
- `results/risk_bridge_rank_deficient_curves.csv`;
- `results/risk_bridge_rank_deficient_minimizers.csv`;
- `results/risk_bridge_rank_deficient_minimizer_frequencies.csv`.

### 9.1 Manifest

The manifest contains 36 rows: 12 spectra crossed with three budgets. It records spectrum parameters, analysis role, dimension, budget, derived $q_{\max}$, residual floor, orientation group and seed, basis-trial count, bootstrap count, QR tolerances, sketch distribution, and code/configuration version fields. For each $r_\star$, all four $\eta$ rows must have the same orientation group and seed.

### 9.2 Trial rows

The production trial table contains

$$
12\cdot200\cdot(37+77+117)=554{,}400
$$

rows. Its unique key is

```text
(spectrum_index, budget, basis_trial, q)
```

and it contains all allocation, rank, risk, energy, regret-source, query-accounting, seed, zero-policy, and marginal-transition fields, including `rank_efficiency`, `rejected_query_count`, and the three cumulative failed-transition penalties.

### 9.3 Mean curves

There are

$$
12(37+77+117)=2{,}772
$$

setup-budget-allocation rows. For every risk type, store mean, standard error, descriptive normal interval, additive regret, baseline-normalized additive regret, multiplicative regret or `NaN`, and zero/plateau flags.

The normal intervals describe uncertainty across randomized bases for each curve point. They are not simultaneous bands.

### 9.4 Minimizers

There are

$$
36\cdot4=144
$$

setup-budget-risk rows. Record representative and plateau minimizers, minimum mean risk, bootstrap percentile interval for the representative minimizer, trial-minimizer summaries, and bootstrap count.

### 9.5 Bootstrap minimizer frequencies

Record the complete feasible $q$ support, including zero-frequency allocations. There are

$$
4\cdot12(37+77+117)=11{,}088
$$

rows. Frequencies must sum to one within every spectrum-budget-risk group.

Freeze

```python
BOOTSTRAP_SEED_BASE = 20_260_815
```

and derive bootstrap randomness from the `rank_index`, risk type, and bootstrap replicate, not from the $\eta$-specific `spectrum_index`. Use the same bootstrap trial weights across budgets and across all four $\eta$ levels for a fixed $r_\star$. This preserves pairing in both budget and tail-level comparisons.

## 10. Adjustable arguments and frozen defaults

The new experiment must expose the following arguments. Defaults reproduce the approved production design; changing a default is allowed for smoke tests or explicitly labeled sensitivity runs, but must be recorded in the manifest and must not overwrite production artifacts.

### 10.1 Rank-deficient bridge CLI

```text
--budgets
--primary-budget
--trials
--dimension
--step-ranks
--tail-levels
--min-residual-probes
--bootstrap-samples
--qr-rtol
--qr-atol
--zero-backward-rtol
--orientation-seed-base
--basis-seed-base
--bootstrap-seed-base
--output-dir
```

| Argument | Frozen default | Validation and effect |
|---|---:|---|
| `--budgets` | `80 160 240` | Unique positive integers. Determines the three allocation grids. |
| `--primary-budget` | `160` | Must be present in `--budgets`; labels primary versus sensitivity rows. |
| `--trials` | `200` | Positive integer; controls randomized-basis replication. |
| `--dimension` | `500` | Positive integer at least as large as every step rank; $q_{\max}$ is automatically capped at the dimension. |
| `--step-ranks` | `5 15 30` | Unique positive integers not exceeding dimension. |
| `--tail-levels` | `0 1e-14 1e-10 1e-6` | Unique finite values in $[0,1)$. |
| `--min-residual-probes` | `8` | Positive integer; determines $q_{\max}(m)$ through the worst-case full-rank cap. |
| `--bootstrap-samples` | `20000` | Positive integer. |
| `--qr-rtol` | `1e-12` | Finite nonnegative float; changing it changes the numerical-rank experiment. |
| `--qr-atol` | `0.0` | Finite nonnegative float; changing it changes the numerical-rank experiment. |
| `--zero-backward-rtol` | `1e-12` | Finite nonnegative residual-norm ratio used only to recognize roundoff zero in $\eta=0$ cases after full signal-rank capture. |
| `--orientation-seed-base` | `52000` | Integer seed base; one seed is derived per step rank and reused across all tail levels for that rank. |
| `--basis-seed-base` | `70000` | Integer seed base shared across budgets by trial index. |
| `--bootstrap-seed-base` | `20260815` | Integer seed base shared across budgets and tail levels within a fixed step-rank group. |
| `--output-dir` | project `results/` | Alternate directories are required for smoke or overridden configurations. |

Booleans must be rejected where integers or real-valued arguments are required. Budgets, ranks, and seeds must be exactly integral. NaNs, infinities, duplicate grid entries, invalid primary-budget membership, negative tolerances, and configurations with no feasible residual allocation must raise `ValueError` before matrix construction.

If any scientific or numerical default differs from the production configuration, using the default project `results/` directory must raise `ValueError`. Overridden configurations require an explicit alternate output directory. This prevents a smoke or sensitivity run from silently replacing production artifacts.

The following values are derived and must not be independently edited:

```text
q_max(m) = min(dimension, (m - min_residual_probes) // 2)
ell(q)   = m - q - r_q
analysis_role = primary if m == primary_budget else sensitivity
```

The sketch distribution remains frozen as Rademacher in this implementation cycle. Adding a sketch-distribution switch would define a larger experiment and is outside scope.

### 10.2 Existing-bridge postprocessor CLI

```text
--trials-csv
--bootstrap-samples
--bootstrap-seed-base
--output-dir
```

The default input is `results/risk_bridge_exact_trials.csv`. The postprocessor is read-only with respect to every existing artifact. An alternate input must pass the same schema and pairing checks.

### 10.3 Programmatic configuration

The core runners must accept the same configuration as explicit function arguments rather than reading module globals internally. Module constants provide CLI defaults only. This makes small tests deterministic without monkeypatching global state.

## 11. Implementation architecture and files

### 11.1 Proof documents

Minimal edits:

- `docs/proof_rank_aware_risk.md`: add the exact Rademacher conditional variance equality and conditional-to-unconditional MSE corollary; preserve existing Theorem 3 material.
- `docs/proof_near_oracle_regret.md`: add Theorem 14, Corollary 14.1, and Lemma 14.2 after Theorem 13 or in a clearly cross-referenced rank-aware section; include the $E_X(Q)>0$ fractional-energy interpretation and its zero-energy boundary; update the classification header without renumbering existing results.
- `docs/proof_notation.md`: add $D$, $E_G$, $E_R$, $M_G$, $M_R$, and the successful/failed rank-aware transitions.

Theorem 15 and its knee-detection material remain supporting theory and are not changed by this cycle.

### 11.2 Analysis helper

Add a small pure module, `experiments/risk_bridge_regret.py`, responsible only for:

- deterministic plateau/minimizer detection;
- additive, baseline-normalized, and multiplicative regret;
- exact-zero policy inputs and flags;
- paired bootstrap minimizer frequencies;
- schema-independent validation of probability sums.

It must not construct matrices or bases.

### 11.3 Existing-bridge postprocessor

Add `experiments/postprocess_exact_risk_bridge.py`. It reads and validates the existing trial CSV, calls the pure regret helper, and writes only the two new postprocessed artifacts.

### 11.4 Rank-deficient runner

Add `experiments/run_rank_deficient_risk_bridge.py`. It owns:

- the 12-case spectrum manifest;
- deterministic matrix orientations;
- paired nested Rademacher sketches;
- incremental double-orthogonalized rank-aware basis construction;
- efficient exact residual-energy updates;
- multi-budget row expansion;
- transition diagnostics;
- calls to the shared regret/bootstrap helper;
- final schema and invariant validation;
- atomic final CSV writes.

It may import `_rank_aware_qr` from `src/trace_baseline.py`; it must not duplicate or silently alter that tolerance rule.

### 11.5 Tests

Add `tests/test_rank_deficient_risk_bridge.py` and minimally extend proof/helper tests where appropriate. Existing bridge tests remain unchanged except for imports of genuinely shared pure helpers if needed.

## 12. Numerical implementation details

### 12.1 Matrix construction

For each spectrum, generate $U_\star\in\mathbb R^{d\times r_\star}$ by an economic QR factorization of a deterministic Gaussian matrix. Represent the matrix by the exact structured operator

$$
A v
=
\eta v+(1-\eta)U_\star(U_\star^Tv).
$$

This operator is mathematically identical to a full orthogonal orientation of the step spectrum because

$$
A
=
U_\star U_\star^T
+\eta(I-U_\star U_\star^T)
=
\eta I+(1-\eta)U_\star U_\star^T.
$$

Wrap the callable in `MatVecOracle`; do not materialize a dense $d\times d$ matrix in the production loop. Record

$$
\operatorname{tr}(A)
=
r_\star+(d-r_\star)\eta
$$

and

$$
\|A\|_F^2
=
r_\star+(d-r_\star)\eta^2
$$

from the spectrum exactly.

### 12.2 Stable structure-aware residual energy

The generic subtraction identity

$$
\|R_QAR_Q\|_F^2
=
\|A\|_F^2
-2\|AQ\|_F^2
+\|Q^TAQ\|_F^2
$$

is mathematically correct but can catastrophically cancel when the true residual energy is of order $10^{-26}$ and the three terms being subtracted are of order one. It must not be the production formula for the $\eta=10^{-14}$ cases.

Every configured step matrix has the exact structure

$$
A=\eta I+(1-\eta)U_\star U_\star^T,
$$

where $U_\star\in\mathbb R^{d\times r_\star}$ contains the leading signal eigenvectors. Define

$$
Z_Q=R_QU_\star.
$$

Then

$$
H_Q=R_QAR_Q
=
\eta R_Q+(1-\eta)Z_QZ_Q^T.
$$

Because $R_Q$ is an orthogonal projector and $R_QZ_Q=Z_Q$, the exact Gaussian energy is

$$
\boxed{
E_G(Q)
=
\eta^2(d-r)
+2\eta(1-\eta)\|Z_Q\|_F^2
+(1-\eta)^2\|Z_Q^TZ_Q\|_F^2.
}
$$

This expression is a sum of nonnegative terms and retains the positive $\eta^2(d-r)$ tail rather than obtaining it by cancellation.

The residual diagonal is

$$
\operatorname{diag}(H_Q)
=
\eta\left(\mathbf 1-\operatorname{diag}(QQ^T)\right)
+(1-\eta)\operatorname{diag}(Z_QZ_Q^T),
$$

so the exact Rademacher energy is

$$
E_R(Q)
=
E_G(Q)-\|\operatorname{diag}(H_Q)\|_2^2.
$$

Compute $Z_Q$, the row norms of $Q$ and $Z_Q$, and the small $r_\star\times r_\star$ Gram matrix $Z_Q^TZ_Q$ without forming a dense projector. The generic subtraction identity may be retained only as a small-fixture cross-check in numerically well-scaled cases.

For the final Rademacher subtraction, define the local roundoff threshold

$$
\operatorname{tol}_E
=
128\,\varepsilon_{\mathrm{mach}}
\max\left\{E_G(Q),
\|\operatorname{diag}(H_Q)\|_2^2,
\operatorname{tiny}_{\mathrm{float64}}\right\}.
$$

A raw Rademacher energy in $[-\operatorname{tol}_E,0)$ may be clamped to zero with a recorded flag. A value below $-\operatorname{tol}_E$ must fail validation.

For an $\eta=0$ zero-minimum decision, use the recorded backward residual ratio

$$
\rho_H(Q)
=
\frac{\sqrt{E_G(Q)}}{\|A\|_F}.
$$

Canonicalization to mathematical zero requires $r_q=r_\star$ and

$$
\rho_H(Q)\le\texttt{zero\_backward\_rtol}.
$$

This test is never applied to a positive $\eta$ case.

### 12.3 Atomic output behavior

Compute and validate all data frames before writing final filenames. Write temporary files inside the requested output directory and replace final files only after all validations pass. A failed run must not leave a partially updated production artifact set.

Historical and frozen exact-bridge checksums must be captured before the run and verified afterward.

## 13. Tests and acceptance criteria

### 13.1 Mathematical identity tests

1. Theorem 14 finite differences match the stated marginal identity on random feasible spectra and states.
2. Corollary 14.1 matches the exact failed-rank increment, including $T(r)=0$ equality.
3. Lemma 14.2 matches explicit Gaussian and Rademacher risk differences on small nested bases.
4. The ideal specialization of Lemma 14.2 recovers Theorem 14.
5. When $E_X(Q)>0$, the sign criterion agrees exactly with $(E_X(Q)-E_X(Q'))/E_X(Q)>2/D$; the $E_X(Q)=0$ case remains undivided and cannot show strict improvement.
6. All denominator-domain violations are rejected rather than divided through.
7. Conditional Gaussian and Rademacher formulas match Monte Carlo on small fixtures within predeclared tolerances.

### 13.2 Spectrum and configuration tests

1. The default spectrum manifest has 12 unique spectra and 36 unique spectrum-budget rows.
2. Every spectrum is finite, nonnegative, and nonincreasing.
3. Exact-rank, below-tolerance, transition, and full-rank-control tail levels are represented exactly.
4. Default derived endpoints are 36, 76, and 116.
5. Every fixed step rank uses one identical orientation seed and identical $U_\star$ across all four tail levels; distinct step ranks have distinct orientation groups.
6. Malformed budgets, ranks, tails, tolerances, seeds, residual floors, and primary-budget choices are rejected.
7. Programmatic and CLI defaults agree.

### 13.3 Nested basis tests

1. $0\le r_q\le q$.
2. $r_q$ is nondecreasing and changes by at most one.
3. Accepted directions are orthonormal to numerical tolerance after double orthogonalization.
4. Exact-rank configurations never exceed $r_\star$.
5. A deterministic rank-two fixture exhibits both accepted and rejected transitions.
6. Changing `qr_rtol` on a threshold fixture changes rank only in the expected direction and records the changed configuration.
7. For every prefix, `rejected_query_count=q-r_q` equals the cumulative number of rejected transitions, and `rank_efficiency=r_q/q` for $q>0$ with `NaN` at $q=0$.

### 13.4 Risk and regret tests

1. Stable structure-aware Gaussian/Rademacher energies match explicit dense residual matrices on small fixtures, including $\eta=0$ and $\eta=10^{-14}$.
2. Every saved risk and additive regret is finite and nonnegative after permitted roundoff handling.
3. Baseline-normalized regret uses $\mathcal R_X(0;m)$ exactly and never inserts $\varepsilon$.
4. Multiplicative regret is finite when the minimum is positive and `NaN` only under the documented zero-minimum rule.
5. The $\eta=10^{-14}$ positive-tail case is not reclassified as exact zero.
6. Plateau endpoints and smallest-minimizer tie breaking are deterministic.
7. Bootstrap frequencies include the complete support and sum to one.
8. Cumulative failed-transition penalties equal direct sums of the applicable Corollary 14.1 and Lemma 14.2 failed-rank increments.

### 13.5 Budget and seed pairing tests

1. Every row satisfies $q+r_q+\ell_q=m$.
2. `constructed_basis_queries=q+r_q` and `accounted_total_queries=m`.
3. Common $(r_\star,\eta,\text{trial},q)$ prefixes use identical orientation and basis seeds across budgets.
4. At fixed $(r_\star,\text{trial},q)$, all four $\eta$ levels use the same $U_\star$, basis seed, and Rademacher sketch prefix.
5. Bootstrap trial weights are identical across budgets and $\eta$ levels within each fixed-$r_\star$ comparison group.
6. The three budgets contain exactly 200 observations per spectrum/allocation under production defaults.
7. The production row counts are 554,400 trials, 2,772 mean-curve rows, 144 minimizer rows, and 11,088 minimizer-frequency rows.

### 13.6 Preservation tests

1. The four original exact-bridge artifacts remain byte-identical.
2. Historical risk-bridge and asymmetric-guard CSV files remain unchanged.
3. No existing estimator changes under identical seeds.

## 14. Verification sequence

1. Record checksums of every preserved historical/frozen CSV.
2. Patch and line-by-line audit the proof notes.
3. Compile every modified or new Python file.
4. Run targeted theorem, configuration, nested-basis, and regret-helper tests.
5. Run the complete test suite using the established `readline` workaround.
6. Run a two-trial, reduced-dimension multi-budget smoke benchmark into `/private/tmp`.
7. Run the existing-bridge postprocessor and verify that it performs no basis reconstruction.
8. Execute the full 36-configuration, 554,400-row rank-deficient supplement.
9. Validate schemas, keys, row counts, risk domains, marginal identities, probability sums, seed pairing, and budget accounting.
10. Recheck preserved-file checksums.
11. Run `git diff --check`.
12. Update `UROP_TRACKER.md`, workspace `memory.md`, and the designated Obsidian research notes with theorem status, exact configurations, results, limitations, and remaining open questions.

## 15. Interpretation rules

The primary empirical conclusions must come from $m=160$. The $m=80$ and $m=240$ results answer a pre-specified budget-sensitivity question and must be labeled accordingly.

The analysis may report:

- how often sketch columns are rejected at each tail scale;
- how rank efficiency $r_q/q$, rejected-query count $q-r_q$, and cumulative rejected-query penalties evolve;
- whether rejected directions incur the nonnegative penalty predicted by Corollary 14.1;
- how the ideal rank-aware and realized Gaussian/Rademacher marginal signs differ;
- how minimizing allocations and their bootstrap frequencies shift with budget;
- how additive and $\mathcal R(0)$-normalized regret change with budget;
- whether the pointwise theory-guided budget direction is visible on the frozen instances.

Raw risk values at different budgets must not be presented as evidence of better or worse allocation. Increasing $m$ mechanically changes variance. Budget comparisons must use the regret, minimizer, rank-efficiency, or rejected-query diagnostics specified above.

Interpret differences between $\mathcal R_G$ and $\mathcal R_R$ with the coordinate system held fixed. They may reflect Rademacher sensitivity to basis orientation, not merely a generic difference between Gaussian and Rademacher noise.

The analysis must not claim:

- a universal QR threshold law from four tail levels;
- that numerical rank gain captures the next leading eigenvector;
- that bootstrap frequency proves a unique population minimizer;
- that a per-curve interval is a simultaneous or familywise confidence statement;
- that the offline exact marginal is a computable online certificate;
- that fixed-orientation results are rotationally universal;
- that undefined zero-denominator multiplicative regret can be repaired by an arbitrary small constant.

The post-experiment theoretical decision tree is pre-specified:

1. If $\mathcal R_{\mathrm{full}}$, $\mathcal R_{\mathrm{rank}}$, and $\mathcal R_G$ are close on the frozen instances, then $T(r)$ remains a plausible risk target and the next problem is estimating its level or marginal reliably.
2. If $\mathcal R_{\mathrm{rank}}$ and $\mathcal R_G$ differ materially, then randomized-basis leakage matters and the next target should be $E_G(Q)$ or $E_G(Q)-E_G(Q')$.
3. If $\mathcal R_G$ and $\mathcal R_R$ have materially different regret curves or minimizers, future adaptive policies may need to be probe-distribution specific.
4. If rejected directions create material regret, the future online theory should model the probability of accepting the next direction, not only the benefit conditional on acceptance.

The fourth branch motivates, but does not implement, the open quantity

$$
p_b
=
\Pr\!\left(r_{q+1}=r_q+1\mid\mathcal F_b\right)
$$

and the prospective decomposition

$$
p_b\,\Delta\mathcal R_{\mathrm{success}}
+
(1-p_b)\,\Delta\mathcal R_{\mathrm{fail}}.
$$

Constructing or estimating $p_b$ is explicitly outside this implementation cycle.

## 16. Documentation and research record

After implementation and analysis:

- update `UROP_TRACKER.md` with every touched file, theorem classification, configuration, row count, empirical conclusion, and open gap;
- update workspace `memory.md` and synchronize it to `/Users/chenyixin/Documents/Obsidian_Vault/Swati_Summer_Research/memory.md`;
- synchronize the modified proof notes and relevant research summary into the designated Obsidian directory;
- keep `Last updated: 2026-08-14` synchronized with the latest running-log entry;
- retain Theorem 15 as supporting knee-detection theory while describing baseline-safe, realized-risk allocation as the active research direction.

## 17. Fixed decisions

- The original exact bridge and its four artifacts are immutable inputs.
- The rank-deficient supplement is isolated and uses separate output names.
- The primary budget is $160$; $80$ and $240$ are sensitivity budgets.
- The production grid contains 12 spectra, 200 trials, and 554,400 trial-allocation rows.
- The production sketch distribution is Rademacher.
- For each fixed $r_\star$, one $U_\star$ and orientation seed are reused across all four $\eta$ levels; different step ranks retain separate orientation groups.
- The default QR tolerance is exactly the existing `rtol=10^{-12}`, `atol=0` contract.
- The $eta=0$ cases are exact rank deficient; positive tiny tails are never silently converted to exact zero.
- Any observed numerical-rank transition is conditional on the frozen `_rank_aware_qr` reference-scale contract and is not a universal eigenvalue threshold at $10^{-12}$.
- Additive and $\mathcal R(0)$-normalized regrets are primary descriptive metrics; multiplicative regret is omitted when its denominator is zero.
- No confidence radius is invented, estimated, or tuned in this cycle.
- No parameter is retuned after viewing the production supplement.
