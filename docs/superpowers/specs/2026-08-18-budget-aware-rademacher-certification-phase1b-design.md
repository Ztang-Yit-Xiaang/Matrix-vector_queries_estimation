# Phase 1B-A: Budget-Aware Realized Rademacher-Risk Certification Emulation

## 1. Purpose and research boundary

### Research question

Phase 1A established that a small batch of fresh Rademacher probes can contain useful information about the realized conditional risk of already-constructed Hutch++ bases. It treated those certification probes as an external diagnostic budget.

Phase 1B-A asks the next necessary question:

$$
\boxed{
\text{Does direct realized-risk certification remain beneficial after}\
\text{candidate construction and certification queries are charged?}
}
$$

This is a budget-aware offline emulation over the frozen Phase 1A outputs. It is not a new Adaptive Hutch++ allocator.

### Frozen boundary

This cycle will:

- reuse the frozen Phase 1A exact numerators and certification estimates;
- recompute candidate-versus-baseline decisions under the correct common sunk-cost denominator;
- charge all construction queries required to make both actions available;
- charge exactly the selected certification prefix size $s$;
- compare the paid selected risk with the original uncharged baseline risk;
- separately compare against a matched reserved-budget baseline;
- prove the exact accounting identities and the no-free-fallback lemma;
- quantify net benefit, net harm, abstention cost, oracle recoverability, and tail-insurance behavior;
- issue a preregistered budget-aware verdict.

This cycle will not:

- modify the frozen estimator;
- reconstruct or rerun any range sketch;
- generate new certification probes;
- spend actual final residual-probe queries;
- retune the Phase 1A empirical guard;
- introduce a baseline-only prescreen or another allocation policy;
- claim a theorem-level confidence certificate;
- overwrite Phase 1A or earlier historical artifacts;
- proceed to an online allocator.

The completed UROP estimator remains frozen. Phase 1B-A is continuation research.

---

## 2. Source artifacts and immutability

### Required Phase 1A inputs

Read, but never modify:

- `results/direct_rademacher_certification_phase1a_trials.parquet`;
- `results/direct_rademacher_certification_phase1a_truth.csv`;
- `results/direct_rademacher_certification_phase1a_manifest.csv`;
- `results/direct_rademacher_certification_phase1a_verdict.csv`.

The Phase 1A Parquet artifact contains exactly 960,000 rows. Its key is

$$
(r_\star,\texttt{basis\_trial},\eta,
\texttt{batch},\texttt{repetition},s).
$$

The Phase 1B-A manifest must record SHA-256 checksums for every Phase 1A input and verify that they are unchanged after analysis.

### No new random experiment

Phase 1B-A is deterministic conditional on:

- the frozen Phase 1A Parquet artifact;
- the declared bootstrap seed;
- the postprocessing code.

The only new randomness is cluster-bootstrap resampling used for descriptive uncertainty intervals. No new $S$, $Q$, $g$, or $Ag$ is generated.

---

## 3. Objects, notation, and assumption ledger

Let the baseline action be

$$
a_0=(q_0,r_0,Q_0),
$$

and let the candidate action be

$$
a=(q_a,r_a,Q_a).
$$

The actions are nested prefixes of the same frozen range-sketch path. Their exact realized Rademacher variance numerators are

$$
\sigma_0^2
=
\operatorname{Var}_g
\left(g^TR_{Q_0}AR_{Q_0}g\mid Q_0\right),
$$

and

$$
\sigma_a^2
=
\operatorname{Var}_g
\left(g^TR_{Q_a}AR_{Q_a}g\mid Q_a\right).
$$

The frozen Phase 1A artifact supplies exact values of these numerators and estimates

$$
\widehat\sigma_0^2,
\qquad
\widehat\sigma_a^2.
$$

The exact construction costs are

$$
c_0=q_0+r_0,
\qquad
c_a=q_a+r_a.
$$

Because both actions must exist before common-probe certification can compare them, the committed pre-certification construction cost is

$$
\boxed{
c_{\mathrm{pre}}=\max\{c_0,c_a\}.
}
$$

For the nested adjacent pairs in this experiment, this is the cost of the higher-prefix action. No cost may be refunded after the decision.

The exact residual capacity of the original baseline is

$$
\boxed{
\ell_0=m-c_0=m-q_0-r_0.
}
$$

If the budget-aware policy uses $s$ certification queries, its final residual capacity is

$$
\boxed{
\ell_{\mathrm{paid}}
=m-c_{\mathrm{pre}}-s.
}
$$

Every configuration on which paid risk is evaluated must satisfy

$$
\ell_0>0,
\qquad
\ell_{\mathrm{paid}}>0.
$$

If this fails, the paid architecture is physically infeasible. Because the
complete secondary grid was preregistered before this boundary was noticed,
retain that row with `accounting_feasible=False`, preserve its exact query
counts, and leave its paid-risk quantities undefined. Do not regularize the
denominator, delete the row, or renormalize an equal-rank summary over only the
remaining ranks. The primary configuration must be feasible in all three
ranks; otherwise the primary verdict is `INCONCLUSIVE`.

The analysis must never silently replace $r$ by $q$.

---

## 4. Exact budget-aware theory

### Lemma 15.1: Nested-construction reuse

Let $Q_{\mathrm{low}}\in\mathbb R^{d\times r_{\mathrm{low}}}$ and $Q_{\mathrm{high}}\in\mathbb R^{d\times r_{\mathrm{high}}}$ have orthonormal columns and satisfy

$$
\operatorname{range}(Q_{\mathrm{low}})
\subseteq
\operatorname{range}(Q_{\mathrm{high}}).
$$

Define

$$
T=Q_{\mathrm{high}}^TQ_{\mathrm{low}}.
$$

Because the orthogonal projector $Q_{\mathrm{high}}Q_{\mathrm{high}}^T$ acts as the identity on $\operatorname{range}(Q_{\mathrm{low}})$,

$$
\boxed{
Q_{\mathrm{low}}
=
Q_{\mathrm{high}}T.
}
$$

Applying the linear operator $A$ gives

$$
\boxed{
AQ_{\mathrm{low}}
=
(AQ_{\mathrm{high}})T.
}
$$

Thus, once the higher-prefix construction has produced and cached $Q_{\mathrm{high}}$ and $AQ_{\mathrm{high}}$, the lower basis and its matrix product are available without another oracle query.

For the frozen incremental range-sketch paths, every lower accepted basis is nested in every later accepted basis. Therefore constructing both actions in an adjacent pair costs

$$
\boxed{
c_{\mathrm{pre}}
=
q_{\mathrm{high}}+r_{\mathrm{high}}
=
\max\{c_0,c_a\},
}
$$

not $c_0+c_a$.

This lemma depends essentially on nested construction and cached $AQ_{\mathrm{high}}$. It does not justify maximum-cost accounting for independently generated or nonnested actions.

### Theorem 16: Common sunk-cost denominator

Assume both actions have been constructed and $s$ certification products have been spent before selection. Assume the final residual probes are fresh and independent of the construction and certification sigma-algebras.

Then choosing either available basis leaves exactly

$$
\ell_{\mathrm{paid}}=m-c_{\mathrm{pre}}-s
$$

final residual probes. Therefore the exact paid conditional risks are

$$
\boxed{
\mathcal R_{0,\mathrm{paid}}
=
\frac{\sigma_0^2}{\ell_{\mathrm{paid}}},
\qquad
\mathcal R_{a,\mathrm{paid}}
=
\frac{\sigma_a^2}{\ell_{\mathrm{paid}}}.
}
$$

The denominator is common because all construction and certification costs are sunk before the action is selected.

### Corollary 16.1: Paid candidate-versus-baseline ordering

Since $\ell_{\mathrm{paid}}>0$,

$$
\boxed{
\mathcal R_{a,\mathrm{paid}}
<
\mathcal R_{0,\mathrm{paid}}
\iff
\sigma_a^2<\sigma_0^2.
}
$$

Thus a budget-aware decision among already-constructed actions should compare numerator estimates, not the Phase 1A risks with their different original denominators.

### Theorem 17: No-free-fallback lemma

The original uncharged baseline risk is

$$
\mathcal R_0^{\mathrm{original}}
=
\frac{\sigma_0^2}{\ell_0}.
$$

Assume

$$
\sigma_0^2>0,
\qquad
c_{\mathrm{pre}}+s>c_0.
$$

Then

$$
\ell_{\mathrm{paid}}<\ell_0,
$$

and hence

$$
\boxed{
\frac{\mathcal R_{0,\mathrm{paid}}}
{\mathcal R_0^{\mathrm{original}}}
=
\frac{\ell_0}{\ell_{\mathrm{paid}}}
>1.
}
$$

Therefore rejecting or abstaining after candidate construction and certification cannot restore the original baseline risk.

If $\sigma_0^2=0$, both the original and paid fallback risks are zero. This boundary case must be stated separately rather than handled by dividing by $\sigma_0^2$.

### Corollary 17.1: Exact net-benefit threshold

Assume $\sigma_0^2>0$. The paid candidate improves upon the original baseline exactly when

$$
\frac{\sigma_a^2}{\ell_{\mathrm{paid}}}
<
\frac{\sigma_0^2}{\ell_0}.
$$

Equivalently,

$$
\boxed{
\frac{\sigma_a^2}{\sigma_0^2}
<
\frac{\ell_{\mathrm{paid}}}{\ell_0}
=
1-
\frac{(c_{\mathrm{pre}}-c_0)+s}{\ell_0}.
}
$$

This is the exact amount of numerator reduction required to pay for construction beyond the baseline and certification.

For the full-rank primary pair

$$
q_a=r_\star+1,
\qquad
q_0=r_\star,
\qquad
r_a=q_a,
\qquad
r_0=q_0,
$$

the additional construction cost is two queries, so

$$
\boxed{
\frac{\sigma_a^2}{\sigma_0^2}
<
1-\frac{s+2}{m-2r_\star}.
}
$$

### Corollary 17.2: Paid oracle lower bound

Among the two already-constructed actions, an exact oracle would attain

$$
\boxed{
\mathcal R_{\mathrm{oracle,paid}}
=
\frac{\min\{\sigma_0^2,\sigma_a^2\}}
{\ell_{\mathrm{paid}}}.
}
$$

Every measurable selection rule restricted to these two actions has conditional risk at least this value for the realized path.

If the paid oracle does not improve the mean original-baseline risk on the frozen population, then no certification rule over this fixed candidate pair can make the candidate-first architecture beneficial at that $s$ on that frozen population.

This last statement is finite-population and architecture-specific. It is not a universal impossibility theorem.

### Classification

- Lemma 15.1: `PROVED` under nested subspaces and cached higher-prefix products.
- Theorem 16 and Corollary 16.1: `PROVED` under the stated query-timing and freshness assumptions.
- Theorem 17 and Corollaries 17.1--17.2: `PROVED` on their explicit positive-denominator domains.
- Budget-aware performance of the empirical decision: `EMPIRICALLY ESTABLISHED` only after running the frozen postprocessor.
- The Phase 1A multiplicative rule remains an `EMPIRICAL DECISION RULE`, not a confidence certificate.
- Any orientation-universal or population-universal conclusion remains `OPEN`.

---

## 5. Frozen configurations

Use the complete Phase 1A grid:

$$
d=500,
\qquad
r_\star\in\{5,15,30\},
$$

$$
\eta\in\{10^{-10},10^{-6}\},
\qquad
m\in\{80,160,240\},
$$

$$
s\in\{4,8,16,32\},
\qquad
\varepsilon\in\left\{0,0.2,\frac13,0.5\right\}.
$$

Use all four Phase 1A numerator estimators:

- `sample_variance`;
- `paired_mean`;
- `mom_w1`;
- `mom_w2`.

The paired mean and MoM estimators remain secondary. The primary verdict uses sample variance because it was the only Phase 1A estimator that passed the preregistered $s=16$ gate.

### Action pairs

Retain the three frozen adjacent pairs:

1. `left`: candidate $q=r_\star-1$, baseline $q=r_\star$;
2. `primary`: candidate $q=r_\star+1$, baseline $q=r_\star$;
3. `right`: candidate $q=r_\star+2$, baseline $q=r_\star+1$.

For every pair compute

$$
c_{\mathrm{pre}}=\max\{q_a+r_a,q_0+r_0\}.
$$

### Primary configuration

The preregistered budget-aware verdict uses

$$
\boxed{
m=160,
\quad
\eta=10^{-6},
\quad
s=16,
\quad
\varepsilon=\frac13,
}
$$

with the `primary` pair and the `sample_variance` estimator.

No primary setting may be changed after inspecting Phase 1B-A outputs.

---

## 6. Budget-aware decision rule

For every stored Phase 1A repetition, retrieve

$$
\widehat\sigma_a^2,
\qquad
\widehat\sigma_0^2.
$$

Define

$$
\rho_\varepsilon
=
\frac{1-\varepsilon}{1+\varepsilon}.
$$

Use the same three-way rule as Phase 1A, now applied to the numerator estimates:

- if the estimates are equal, return `abstain`;
- if both estimates are zero, return `abstain`;
- accept the candidate if

  $$
  \widehat\sigma_a^2
  \le
  \rho_\varepsilon\widehat\sigma_0^2;
  $$

- reject the candidate if

  $$
  \widehat\sigma_0^2
  \le
  \rho_\varepsilon\widehat\sigma_a^2;
  $$

- otherwise return `abstain`.

Map decisions to final bases as follows:

$$
Q_{\mathrm{selected}}
=
\begin{cases}
Q_a,&\text{accept},\\
Q_0,&\text{reject or abstain}.
\end{cases}
$$

The exact paid selected risk is

$$
\boxed{
\mathcal R_{\mathrm{selected,paid}}
=
\frac{\sigma_{\mathrm{selected}}^2}
{\ell_{\mathrm{paid}}}.
}
$$

The certification probes are not reused as final residual probes. The final risk is evaluated analytically; no final probe is actually drawn.

### Why Phase 1A decisions cannot be copied

Phase 1A compared

$$
\frac{\widehat\sigma_a^2}{m-q_a-r_a}
\quad\text{and}\quad
\frac{\widehat\sigma_0^2}{m-q_0-r_0}.
$$

After both actions and certification have been paid for, the two final actions share $\ell_{\mathrm{paid}}$. Therefore copying the Phase 1A decision column would use the wrong denominator. Phase 1B-A must recompute every decision from the stored numerator estimates.

---

## 7. Four-risk cost decomposition

For every path and configuration report:

### Original baseline

$$
\boxed{
\mathcal R_0^{\mathrm{original}}
=
\frac{\sigma_0^2}{\ell_0}.
}
$$

### Paid baseline

$$
\boxed{
\mathcal R_{0,\mathrm{paid}}
=
\frac{\sigma_0^2}{\ell_{\mathrm{paid}}}.
}
$$

This isolates the unavoidable sunk-cost penalty when the final choice returns to $Q_0$.

### Paid exact oracle

$$
\boxed{
\mathcal R_{\mathrm{oracle,paid}}
=
\frac{\min\{\sigma_0^2,\sigma_a^2\}}
{\ell_{\mathrm{paid}}}.
}
$$

This isolates the maximum benefit recoverable from the fixed candidate pair after cost.

### Paid empirical selection

$$
\boxed{
\mathcal R_{\mathrm{selected,paid}}
=
\frac{\sigma_{\mathrm{selected}}^2}
{\ell_{\mathrm{paid}}}.
}
$$

This adds certification decision error.

### Additional controls

Also report:

$$
\mathcal R_{\mathrm{candidate,paid}}
=
\frac{\sigma_a^2}{\ell_{\mathrm{paid}}},
$$

and the matched reserved-budget comparator

$$
\mathcal R_0^{\mathrm{reserved}}
=
\mathcal R_{0,\mathrm{paid}}.
$$

The reserved comparator answers whether selection improves after all actions are forced to pay the same committed budget. It must never be described as preserving the original baseline.

---

## 8. Path-first estimands

A frozen range-sketch path is the scientific cluster. The 200 certification repetitions for one path approximate certification randomness conditional on that path.

For path $t$, let

$$
\overline{\mathcal R}_{\mathrm{selected,paid},t}
=
\frac1{200}
\sum_{u=1}^{200}
\mathcal R_{\mathrm{selected,paid},t,u}.
$$

For every rank, average these path-level quantities over its 200 frozen paths.

### Primary aggregate mean-risk ratio

Within rank $r_\star$, define

$$
\Gamma_{r_\star}
=
\frac{
\frac1{200}\sum_t
\overline{\mathcal R}_{\mathrm{selected,paid},t}
}{
\frac1{200}\sum_t
\mathcal R_{0,t}^{\mathrm{original}}
}.
$$

Then weight ranks equally:

$$
\boxed{
\Gamma
=
\frac13
\sum_{r_\star\in\{5,15,30\}}
\Gamma_{r_\star}.
}
$$

This ratio of mean risks, rather than the mean of pathwise ratios, is the primary conditional-MSE comparison.

### Secondary pathwise summaries

When $\mathcal R_{0,t}^{\mathrm{original}}>0$, define

$$
\gamma_t
=
\frac{
\overline{\mathcal R}_{\mathrm{selected,paid},t}
}{
\mathcal R_{0,t}^{\mathrm{original}}
}.
$$

Report:

- median $\gamma_t$;
- 90th, 95th, and 99th percentiles;
- maximum;
- fraction with $\gamma_t>1$;
- fraction with $\gamma_t<1$;
- pathwise additive regret;
- rank-specific versions of every summary.

Do not introduce an arbitrary floor if the baseline risk is zero. Exact-zero cases use additive quantities and are reported separately.

---

## 9. Cost, recoverability, and decision diagnostics

For every configuration report:

### Cost inflation

$$
\kappa_{\mathrm{cost}}
=
\frac{\ell_0}{\ell_{\mathrm{paid}}}.
$$

### Required numerator-retention threshold

$$
\theta_{\mathrm{net}}
=
\frac{\ell_{\mathrm{paid}}}{\ell_0}.
$$

The paid candidate beats the original baseline exactly when

$$
\sigma_a^2/\sigma_0^2<\theta_{\mathrm{net}}.
$$

### Oracle recoverability

Compute the paid-oracle aggregate ratio

$$
\Gamma_{\mathrm{oracle}}
=
\frac13
\sum_{r_\star}
\frac{
\operatorname{mean}_t
\mathcal R_{\mathrm{oracle,paid},t}
}{
\operatorname{mean}_t
\mathcal R_{0,t}^{\mathrm{original}}
}.
$$

### Paid-baseline cost ratio

Compute

$$
\Gamma_{\mathrm{paid\ baseline}}
=
\frac13
\sum_{r_\star}
\frac{
\operatorname{mean}_t
\mathcal R_{0,\mathrm{paid},t}
}{
\operatorname{mean}_t
\mathcal R_{0,t}^{\mathrm{original}}
}.
$$

### Decision efficiency

When the denominator is positive, define

$$
\mathrm{Efficiency}
=
\frac{
\Gamma_{\mathrm{paid\ baseline}}-\Gamma
}{
\Gamma_{\mathrm{paid\ baseline}}-\Gamma_{\mathrm{oracle}}
}.
$$

If the denominator is zero, report the efficiency as undefined rather than forcing a numerical value.

### Decision probabilities

Predeclare two different exact truth labels. The **paid-order** label is

$$
\boxed{
\text{paid candidate better}
\iff
\sigma_a^2<\sigma_0^2.
}
$$

It asks which already-paid basis should receive the common residual budget. The **net-benefit** label is

$$
\boxed{
\text{candidate repays the complete procedure}
\iff
\frac{\sigma_a^2}{\ell_{\mathrm{paid}}}
<
\frac{\sigma_0^2}{\ell_0}.
}
$$

These labels are not interchangeable. A candidate can be the better of the two already-paid bases while remaining worse than never starting the candidate-first procedure.

Report path-first, equal-rank:

- accept probability;
- reject probability;
- abstain probability;
- correct paid-order decision probability;
- false candidate selection probability: candidate accepted when $\sigma_a^2>\sigma_0^2$;
- missed paid-order candidate probability: candidate not accepted when $\sigma_a^2<\sigma_0^2$;
- net-beneficial candidate acceptance probability;
- net-harmful candidate acceptance probability.

These paid-order labels compare $\sigma_a^2$ with $\sigma_0^2$. They are distinct from the net-improvement label relative to the original baseline.

---

## 10. Catastrophic-path analysis

Retain the Phase 1A definition of catastrophicness using the exact original baseline risk within each $(r_\star,m,\eta)$ group.

Report top 1%, 5%, and 10% strata.

For each stratum report:

- aggregate selected-paid/original-baseline risk ratio;
- paid-oracle/original-baseline risk ratio;
- candidate acceptance probability;
- fraction of paths whose candidate overcomes the complete sunk cost;
- share of the original mean baseline risk;
- share of total recovered risk;
- median pathwise net ratio;
- fraction harmed after complete cost.

The purpose is to determine whether certification remains useful as tail-risk insurance after it charges typical paths for information.

---

## 11. Bootstrap uncertainty

Use 10,000 deterministic stratified cluster-bootstrap replicates with master seed

```text
93000
```

For each analyzed configuration:

1. resample the 200 frozen paths with replacement separately within each rank;
2. retain all 200 certification repetitions for every sampled path;
3. recompute each rank's ratio of mean risks;
4. average the three rank ratios equally;
5. store the ordinary percentile 95% interval.

Use one deterministic rank-stratified set of resampling weights across all
configurations. This preserves the frozen cross-configuration pairing and does
not change any configuration's marginal cluster-bootstrap distribution. Record
the shared stream contract in the manifest.

The primary interval is for $\Gamma$.

Also produce intervals for:

- $\Gamma_{\mathrm{oracle}}$;
- $\Gamma_{\mathrm{paid\ baseline}}$;
- accept, reject, and abstain probabilities;
- the fraction of paths harmed;
- catastrophic top-5% net ratio.

Intervals are conditional on the frozen orientations, paths, and Phase 1A certification experiment. They do not establish population-universal guarantees.

---

## 12. Preregistered verdict

The primary verdict uses only:

$$
m=160,
\quad
\eta=10^{-6},
\quad
s=16,
\quad
\varepsilon=\frac13,
$$

the `primary` pair, and `sample_variance`.

Let $[L_{95},U_{95}]$ be the bootstrap interval for $\Gamma$.

### Main verdict

- `NET BENEFIT` if

  $$
  U_{95}<1.
  $$

- `NET HARM` if

  $$
  L_{95}>1.
  $$

- `INCONCLUSIVE` if

  $$
  L_{95}\le1\le U_{95}.
  $$

### Tail-insurance modifier

Add

```text
TAIL-INSURANCE TRADEOFF
```

when the main verdict is `NET BENEFIT` but either:

- the median pathwise ratio exceeds one; or
- more than 50% of frozen paths have pathwise expected paid risk above their original baseline risk.

### Frozen-oracle feasibility flag

Report

```text
FROZEN-POPULATION ORACLE CANNOT PAY AT s
```

if the exact finite-population point value satisfies

$$
\Gamma_{\mathrm{oracle}}\ge1.
$$

This flag means no selection rule restricted to the fixed candidate pair can improve the frozen-population aggregate mean after cost at that $s$. It is not a universal impossibility claim.

The verdict must not be changed using secondary budgets, tail levels, pairs, estimators, sample sizes, or guard parameters.

---

## 13. Secondary analyses

Without changing the verdict, report:

- all $s\in\{4,8,16,32\}$;
- all $m\in\{80,160,240\}$;
- both $\eta$ levels;
- all three action pairs;
- all four estimators;
- all four $\varepsilon$ values;
- per-rank and equal-rank results;
- exact paid oracle and always-baseline/always-candidate controls;
- matched reserved-budget comparisons.

The sample-size curve is especially important because increasing $s$ can improve decision accuracy while simultaneously shrinking $\ell_{\mathrm{paid}}$.

Cross-$s$, cross-budget, cross-action, cross-estimator, and cross-$\eta$ analyses are paired because they inherit common Phase 1A randomness.

---

## 14. Implementation

### New postprocessor

Create:

`experiments/postprocess_direct_rademacher_certification_phase1b_budget.py`

The postprocessor must:

- read only the declared Phase 1A artifacts;
- validate their checksums, row counts, keys, and required columns;
- derive exact construction costs from stored $q$ and $r$;
- compute $c_{\mathrm{pre}}$, $\ell_0$, and $\ell_{\mathrm{paid}}$;
- retain infeasible secondary configurations with
  `accounting_feasible=False` and undefined paid-risk fields;
- refuse to aggregate an equal-rank paid-risk result if any constituent rank
  is infeasible;
- recompute numerator-based decisions;
- average certification repetitions within path;
- compute equal-rank summaries and bootstrap intervals;
- stage all outputs before atomic publication;
- refuse to overwrite existing Phase 1B-A outputs;
- never import or call the estimator runner.

### CLI

Expose read-only configuration flags:

```text
--phase1a-trials
--phase1a-truth
--phase1a-manifest
--output-dir
--bootstrap-samples
--bootstrap-seed
```

Defaults must reproduce the frozen analysis. Any nondefault output must use a separate output directory.

Scientific grid values come from the Phase 1A artifact and may not be silently overridden.

---

## 15. Output artifacts

Create isolated artifacts with prefix

```text
direct_rademacher_certification_phase1b_budget
```

### Manifest

Create:

`results/direct_rademacher_certification_phase1b_budget_manifest.csv`

Record:

- source checksums;
- source row counts and schemas;
- bootstrap contract;
- primary configuration;
- exact formulas;
- expected output rows;
- software versions;
- start and completion times;
- output checksums;
- final verdict.
- counts of feasible and infeasible path/configuration rows.

### Path-level artifact

Create:

`results/direct_rademacher_certification_phase1b_budget_paths.parquet`

Use one row per

$$
(r_\star,\texttt{basis\_trial},\eta,m,s,
\texttt{pair},\texttt{estimator},\varepsilon).
$$

Expected rows:

$$
3\times200\times2\times3\times4\times3\times4\times4
=
\boxed{691{,}200}.
$$

Store:

- all key fields and seeds;
- action labels, $q$, $r$, and construction costs;
- $c_{\mathrm{pre}}$, $\ell_0$, and $\ell_{\mathrm{paid}}$;
- `accounting_feasible`;
- exact numerators;
- original, paid-baseline, paid-candidate, and paid-oracle risks;
- path-averaged paid selected risk;
- accept, reject, and abstain probabilities;
- paid-order error probabilities;
- net-improvement indicators;
- pathwise net ratio when defined;
- cost multiplier and numerator threshold;
- catastrophic-stratum labels;
- finite-value and accounting flags.

### Summaries

Create CSVs for:

- aggregate mean-risk ratios;
- bootstrap intervals;
- cost decomposition;
- decision probabilities;
- catastrophic strata;
- sample-size sensitivity;
- primary verdict.

### Figures

Create publication-quality figures showing at least:

1. the four-risk cost decomposition;
2. selected/original mean-risk ratio versus $s$;
3. paid oracle versus empirical selection;
4. cost inflation and required numerator reduction versus rank;
5. accept/reject/abstain probabilities versus $s$;
6. pathwise net-ratio distributions;
7. catastrophic versus ordinary paths;
8. primary tail family versus benign control;
9. per-rank budget sensitivity.

### Report

Create:

`reports/direct_rademacher_risk_certification_phase1b_budget.md`

The report must contain:

- executive verdict;
- accounting diagram;
- Theorems 16--17 and corollaries;
- source immutability audit;
- primary cost decomposition;
- oracle recoverability;
- empirical decision performance;
- tail-insurance analysis;
- secondary sensitivity;
- limitations;
- decision on whether to proceed to a confidence theorem or redesign the architecture.

Use the evidence labels:

- `PROVED`;
- `EMPIRICALLY ESTABLISHED`;
- `EMPIRICAL DECISION RULE`;
- `OPEN`.

---

## 16. Tests

Create:

`tests/test_direct_rademacher_certification_phase1b_budget.py`

### Accounting tests

Verify:

- $c_0=q_0+r_0$;
- $c_a=q_a+r_a$;
- $c_{\mathrm{pre}}=\max\{c_0,c_a\}$;
- $\ell_0=m-c_0$;
- $\ell_{\mathrm{paid}}=m-c_{\mathrm{pre}}-s$;
- full-rank primary cost difference equals two;
- the scalar accounting helper rejects nonpositive paid residual capacity;
- the full-grid postprocessor retains preregistered infeasible secondary rows,
  marks them `accounting_feasible=False`, and leaves paid-risk fields
  undefined;
- infeasible ranks are never silently removed or used to renormalize an
  equal-rank summary;
- rank-deficient synthetic cases do not replace $r$ by $q$.

### Nested-reuse tests

For every frozen adjacent action pair verify

$$
\left\|
(I-Q_{\mathrm{high}}Q_{\mathrm{high}}^T)
Q_{\mathrm{low}}
\right\|_2
$$

is within the declared numerical tolerance. Also verify

$$
AQ_{\mathrm{low}}
\approx
AQ_{\mathrm{high}}
(Q_{\mathrm{high}}^TQ_{\mathrm{low}}).
$$

Synthetic nonnested actions must not be assigned maximum-cost reuse.

### Theorem identity tests

Numerically verify:

$$
\frac{\mathcal R_{0,\mathrm{paid}}}
{\mathcal R_0^{\mathrm{original}}}
=
\frac{\ell_0}{\ell_{\mathrm{paid}}}
$$

on the positive-risk domain.

Verify the exact candidate net-benefit equivalence and the zero-risk boundary.

Verify the paid-oracle lower bound against every available two-action choice.

### Decision tests

Verify:

- numerator-based decisions differ from old-denominator Phase 1A decisions in constructed examples;
- equality and double-zero produce abstention;
- accept/reject boundaries for every $\varepsilon$;
- reject and abstain both select the baseline basis but retain the paid denominator;
- no query cost is refunded after selection.

### Aggregation tests

Verify:

- 200 repetitions are averaged within path first;
- ratios of means are not replaced by means of pathwise ratios;
- ranks receive equal weight;
- exact-zero paths are excluded only from ratio summaries, not additive summaries;
- catastrophic strata are defined within rank and budget;
- bootstrap resamples paths as clusters and preserves all repetitions.

### Source and artifact tests

Verify:

- Phase 1A checksums are unchanged;
- required schemas and columns are present;
- exactly 691,200 path-level keys are unique;
- every required configuration is represented;
- required primary-gate fields are finite;
- paid-risk fields are finite exactly when `accounting_feasible=True` and are
  undefined when it is false;
- output checksums match the manifest;
- historical result artifacts remain unchanged.

---

## 17. Verification sequence

1. Compile the new postprocessor and tests.
2. Run targeted Phase 1B-A tests.
3. Run the complete maintained `tests/` suite with the repository's readline workaround.
4. Run a smoke postprocessing job into `/private/tmp` using a strict subset of paths but every decision branch and output schema.
5. Validate source checksums, path-first aggregation, and atomic publication in smoke output.
6. Run the complete 960,000-row source postprocessing analysis.
7. Validate the 691,200-row path artifact, all summary keys, bootstrap intervals, and verdict.
8. Verify that Phase 1A and historical artifacts are byte-for-byte unchanged.
9. Inspect every figure.
10. Run the complete maintained test suite again.
11. Run `git diff --check`.
12. Audit the proof and final report line by line for theorem-versus-empirical wording.

---

## 18. Documentation and research record

After implementation and analysis:

- update `UROP_TRACKER.md` with touched files, exact accounting, production sizes, primary result, verdict, and remaining gap;
- update workspace `memory.md`;
- update `CURRENT_STATE.md`;
- synchronize the tracker, memory, current state, Phase 1B-A report, and proof note into `/Users/chenyixin/Documents/Obsidian_Vault/Swati_Summer_Research`;
- update the tracker's `Last updated` date to the date of the newest running-log entry;
- preserve the completed UROP estimator freeze;
- describe Phase 1B-A as continuation research.

Create a focused proof note:

`docs/proof_budget_aware_certification.md`

The proof note must state Theorems 16--17 and their corollaries on explicit domains, including the exact-zero boundary and the finite-population limitation of the paid-oracle conclusion.

---

## 19. Fixed decisions and interpretation limits

- The candidate-first architecture is primary.
- The original uncharged action baseline is the scientific comparator.
- The reserved-budget baseline is secondary and may not be relabeled as original-baseline safety.
- Both actions must be constructed before direct realized-risk comparison.
- All committed construction queries are sunk.
- At a chosen sample size $s$, exactly $s$ certification queries are charged, even though Phase 1A generated a common 32-probe batch for offline reuse.
- Final residual probes are fresh conceptually and are not simulated.
- The final residual denominator is common across the two selected bases.
- Phase 1B-A decisions are recomputed from numerator estimates.
- The primary verdict uses sample variance only.
- The primary setting remains $m=160$, $\eta=10^{-6}$, $s=16$, $\varepsilon=1/3$, and the $r_\star+1$ versus $r_\star$ pair.
- No threshold, estimator, action pair, sample size, or verdict rule is retuned after outputs are inspected.
- A net-benefit result is conditional on the frozen orientations and basis paths.
- A net-harm result rejects only this candidate-first architecture at the tested costs; it does not disprove direct risk certification in general.
- A reserved-budget benefit does not imply preservation of the original baseline.
- The no-free-fallback theorem does not apply through division when the baseline variance is exactly zero; that boundary is handled separately.
- No confidence theorem or online allocator is implemented in this cycle.

---

## 20. Decision after Phase 1B-A

The next step is determined by the paid decomposition.

### Case A: empirical selection has net benefit

If $U_{95}<1$, proceed to a finite-sample theorem targeted to the actual numerator comparison and then design a physically feasible online schedule.

### Case B: paid oracle helps but empirical selection does not

If $\Gamma_{\mathrm{oracle}}<1$ but the empirical interval does not resolve below one, the architecture has recoverable value but the certification rule or sample size is inadequate. Study a sharper paired risk-difference statistic before changing the candidate family.

### Case C: paid oracle cannot help

If $\Gamma_{\mathrm{oracle}}\ge1$ on the frozen population, do not spend effort proving a theorem for this exact candidate-first timing. The next research problem becomes a lower-cost prescreen or a prospective value-of-information method that does not require constructing the full candidate first.

### Case D: mean benefit with typical-path harm

If the mean improves but the median path or majority of paths is harmed, classify the result as tail-risk insurance. Any future theorem or policy objective must explicitly decide whether it targets expected MSE, pathwise safety, or a tail-risk criterion.
