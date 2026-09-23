# Phase 1A: Offline Realized Rademacher-Risk Certification Feasibility

Date frozen: 2026-08-18

Status: approved, preregistered continuation-project design

## 1. Purpose and frozen research boundary

The research question is:

$$
\boxed{
\text{Can fresh probes reliably identify when one realized basis has lower Rademacher MSE than another?}
}
$$

For an already-constructed action

$$
a=(q_a,r_a,Q_a),\qquad
R_a=I-Q_aQ_a^\top,\qquad
B_a=R_aAR_a,
$$

define

$$
X^{(a)}(g)=g^\top B_ag,\qquad
g\sim\operatorname{Rad}(\pm1)^d.
$$

For symmetric $B_a$,

$$
\sigma_a^2
=
\operatorname{Var}_g(X^{(a)}\mid Q_a)
=
4\sum_{i<j}(B_a)_{ij}^2
=
2\sum_{i\ne j}(B_a)_{ij}^2.
$$

The frozen Hutch++ residual budget and exact conditional Rademacher risk are

$$
\ell_a(m)=m-q_a-r_a,
\qquad
\boxed{
\mathcal R_a(m)
=
\frac{\sigma_a^2}{\ell_a(m)}
=
\frac{\sigma_a^2}{m-q_a-r_a}.
}
$$

Phase 1A estimates only $\sigma_a^2$. It does not subtract the external certification sample size from the frozen denominator.

This cycle will reconstruct frozen bridge actions without changing the estimator, estimate realized Rademacher risk with fresh probes, evaluate selective action comparisons, compare four estimator families, analyze catastrophic paths, and issue a preregistered verdict.

This cycle will not modify the frozen estimator, invent another allocator, charge certification queries against the estimator budget, reuse certification probes as final residual probes, claim a finite-sample confidence theorem, retune thresholds after production, or overwrite historical bridge artifacts.

A positive result supports attempting the finite-sample concentration theorem. It is not a theorem-level safety certificate.

## 2. Freeze and implementation order

Implementation must follow:

$$
\boxed{\text{freeze spec}}
\longrightarrow
\boxed{\text{commit spec only}}
\longrightarrow
\boxed{\text{implement}}
\longrightarrow
\boxed{\text{smoke and audit}}
\longrightarrow
\boxed{\text{production run}}.
$$

The specification commit contains only this file and preserves every unrelated worktree change. No scientific threshold or hyperparameter may change after production outputs are inspected.

## 3. Frozen action reconstruction

Use

$$
A_\eta=\eta I+(1-\eta)U_\star U_\star^\top
$$

with

$$
d=500,\qquad
r_\star\in\{5,15,30\},\qquad
\eta\in\{10^{-10},10^{-6}\}.
$$

For fixed $r_\star$, both tail levels use the identical $U_\star$. The range sketch is the frozen coordinate-Rademacher matrix $S$. Final-probe risk labels do not change the range-sketch distribution.

Reconstruct 200 frozen basis paths for every $r_\star$ using the original orientation seeds, basis seeds, rank-aware QR procedure, nested $S$ prefixes, relative tolerance $10^{-12}$, and frozen absolute-tolerance behavior. There are 600 basis paths per tail level and 1,200 path-spectrum combinations.

For each path reconstruct

$$
q\in\{r_\star-1,r_\star,r_\star+1,r_\star+2\}.
$$

Cache for every action:

- $q$ and realized $r_q$;
- $Q_q$ and $AQ_q$;
- construction cost $q+r_q$;
- $\ell_q(m)=m-q-r_q$ for $m\in\{80,160,240\}$;
- exact $\sigma_q^2$ and exact $\mathcal R_q(m)$.

Never replace $r_q$ by $q$ without checking the reconstructed rank.

The action pairs, with candidate first, are:

1. left control: $(r_\star-1,r_\star)$;
2. primary: $(r_\star+1,r_\star)$;
3. right control: $(r_\star+2,r_\star+1)$.

The primary verdict uses the second pair.

### 3.1 Cached $AQ_q$ accounting

$AQ_q$ is computed and cached during reconstruction and charged only to a reconstruction oracle. Certification starts only after all four $Q_q$ and $AQ_q$ pairs exist. Certification must not recompute $AQ_q$.

Every 32-probe repetition therefore costs

$$
\underbrace{32}_{Ag_1,\ldots,Ag_{32}}
+
\underbrace{0}_{AQ_q\text{ calls during certification}}
=32
$$

certification queries regardless of action, estimator, budget, or comparison count.

### 3.2 Reconstruction validation

Reproduce frozen scalar quantities including $q$, $r_q$, rank diagnostics, denominators, exact residual energies, exact risks, and construction accounting.

The required subspace comparison is invariant to basis signs and rotations:

$$
\left\|
Q_{\rm new}Q_{\rm new}^\top
-
Q_{\rm reference}Q_{\rm reference}^\top
\right\|_2
\le \text{declared numerical tolerance}.
$$

Raw $Q$ equality is optional. Frozen code changes require a separately reported and audited genuine bug.

## 4. Certification probes and seed construction

For every frozen path and tail level, run four independent batches with 50 repetitions per batch. Every repetition produces

$$
G=[g_1,\ldots,g_{32}]\in\{-1,+1\}^{500\times32}.
$$

Use nested prefixes

$$
s\in\{4,8,16,32\},\qquad G_s=G[:,1:s].
$$

Set the certification master seed to 91000 and map ranks by

$$
5\mapsto0,\qquad15\mapsto1,\qquad30\mapsto2.
$$

For zero-based basis trial, batch, and repetition, the exact construction is:

    seed_sequence = np.random.SeedSequence(
        [91000, rank_index, basis_trial, batch, repetition]
    )
    cert_seed = int(
        seed_sequence.generate_state(1, dtype=np.uint64)[0]
    )
    rng = np.random.default_rng(cert_seed)
    bits = rng.integers(
        0, 2, size=(dimension, 32), dtype=np.int8
    )
    G_int8 = 2 * bits - 1
    G = G_int8.astype(np.float64)

The seed excludes $\eta$, $m$, $q$, $s$, pair, and estimator. Thus the identical probe matrix is reused across both tails, all budgets, actions, pairs, estimators, and nested sample sizes.

Store the scalar certification seed, its five components, and SHA-256 of contiguous int8 $G$. Do not store the full probe matrix. Assert seed invariance across all excluded fields.

Cross-tail, cross-budget, cross-action, cross-estimator, and cross-sample-size comparisons are paired and not independent.

## 5. Matrix-free certification calculation

Query $Ag$ once. For cached action $a$, calculate

$$
h_a=R_ag=g-Q_a(Q_a^\top g)
$$

and

$$
Ah_a=Ag-AQ_a(Q_a^\top g).
$$

Then

$$
X^{(a)}(g)=g^\top R_aAR_ag=h_a^\top Ah_a.
$$

The production runner must not explicitly form $R_a$, $B_a$, or a dense $d\times d$ residual matrix. The same $Ag$ serves all actions. Batched oracle calls are allowed only if the oracle counts all 32 columns.

Maintain distinct reconstruction and certification counters. Every repetition asserts that the certification count is exactly 32.

## 6. Certification estimators

For a fixed action and prefix, write the quadratic forms as $X_1,\ldots,X_s$.

### 6.1 Sample variance

$$
\widehat\sigma_{\rm SV}^2
=
\frac1{s-1}\sum_{j=1}^s(X_j-\bar X)^2.
$$

It is conditionally unbiased for $\sigma^2$.

Verify the all-pairs identity

$$
\widehat\sigma_{\rm SV}^2
=
\frac1{s(s-1)}
\sum_{i<j}(X_i-X_j)^2.
$$

The all-pairs form is an identity check, not a verdict estimator.

### 6.2 Paired observations and mean

Define

$$
W_j=\frac{(X_{2j-1}-X_{2j})^2}{2},
\qquad j=1,\ldots,s/2.
$$

Then $\mathbb E[W_j\mid Q]=\sigma^2$. The paired mean

$$
\widehat\sigma_{\rm paired}^2
=
\frac2s\sum_{j=1}^{s/2}W_j
$$

is unbiased and serves only as an ablation.

### 6.3 Median-of-means candidates

For mom_w1, use one $W_j$ per block:

$$
\widehat\sigma_{\rm mom\_w1}^2
=
\operatorname{median}\{W_1,\ldots,W_{s/2}\}.
$$

For mom_w2, use consecutive blocks of two:

$$
M_b=\frac{W_{2b-1}+W_{2b}}2,
\qquad
\widehat\sigma_{\rm mom\_w2}^2
=
\operatorname{median}\{M_1,\ldots,M_{s/4}\}.
$$

For an even number of values, the median is the arithmetic mean of the two central order statistics. The MoM families are not claimed unbiased.

At $s=16$, mom_w1 has eight blocks and mom_w2 has four. Predeclare the algebraic coincidences:

- at $s=4$, mom_w1 and mom_w2 equal the paired mean;
- at $s=8$, mom_w2 equals the paired mean.

These coincidences are not independent evidence.

For each estimator,

$$
\widehat{\mathcal R}_a(m)
=
\frac{\widehat\sigma_a^2}{m-q_a-r_a}.
$$

The external sample size $s$ is not deducted in Phase 1A.

## 7. Exact truth labels

For candidate $a$ and baseline $a_0$, define

$$
\Delta(a,a_0;m)
=
\mathcal R_a(m)-\mathcal R_{a_0}(m)
$$

and

$$
\tau
=
128\,\epsilon_{\rm machine}
\max\{\mathcal R_a(m),\mathcal R_{a_0}(m)\}.
$$

Classify the candidate as better when $\Delta<-\tau$, worse when $\Delta>\tau$, and tied when $|\Delta|\le\tau$.

Ties are reported separately and excluded from false-safe, false-rejection, true-better acceptance, and catastrophic-detection denominators.

Do not introduce a positive numerical floor. Relative error is defined only for positive exact risk, and log-risk ratios only when both exact risks are positive.

The truth-table key is

$$
(r_\star,\texttt{basis trial},\eta,m,q).
$$

## 8. Empirical selective decision rule

Use

$$
\varepsilon\in\{0,0.2,1/3,0.5\},
\qquad
\rho_\varepsilon=\frac{1-\varepsilon}{1+\varepsilon}.
$$

The primary setting has $\varepsilon=1/3$ and $\rho=1/2$.

For equal estimates, including two zeros, abstain. Otherwise:

- accept when $\widehat{\mathcal R}_a\le\rho_\varepsilon\widehat{\mathcal R}_{a_0}$;
- reject when $\widehat{\mathcal R}_{a_0}\le\rho_\varepsilon\widehat{\mathcal R}_a$;
- abstain otherwise.

This is an empirical guard, not a confidence certificate.

## 9. Operating metrics

For truly worse candidates use false-safe acceptance $FS$, correct rejection $CR$, and abstention $A_W$. Define

$$
\boxed{
\operatorname{FSR}_{\rm worse}
=
\frac{FS}{FS+CR+A_W}
=
\frac{FS}{N_{\rm worse}}.
}
$$

Candidate-better cases never enter this denominator.

For truly better candidates use correct acceptance $CA$, false rejection $FR$, and abstention $A_B$. Define

$$
\operatorname{FRR}_{\rm better}
=
\frac{FR}{CA+FR+A_B}
$$

and

$$
\operatorname{Acceptance}_{\rm better}
=
\frac{CA}{CA+FR+A_B}.
$$

For non-tied cases,

$$
\operatorname{Coverage}
=
\frac{CA+CR+FS+FR}
{N_{\rm better}+N_{\rm worse}}.
$$

Also report truth-specific abstention, error among decisions, eligible path counts, and tie counts. The formal gate uses truth-conditional rates, not error among decisions.

## 10. Path-first, equal-rank aggregation

The scientific cluster is the frozen path. For an eligible rank-specific set $E_r$,

$$
\widehat p_r
=
\frac1{|E_r|}
\sum_{t\in E_r}
\left[
\frac1{200}
\sum_{b=1}^4
\sum_{j=1}^{50}
\mathbf1\{\text{desired decision}_{t,b,j}\}
\right].
$$

Then

$$
\boxed{
\widehat p
=
\frac13
\sum_{r_\star\in\{5,15,30\}}\widehat p_{r_\star}.
}
$$

Batch stability uses the same definition with only that batch's 50 repetitions. Repetitions do not receive the scientific weight of independent paths.

## 11. Catastrophic-path strata

For each $(r_\star,m)$, rank the 200 paths by exact baseline risk at $q=r_\star$. Define top one, five, and ten percent as 2, 10, and 20 paths per rank.

The primary catastrophic gate uses

$$
m=160,\qquad\eta=10^{-6},\qquad\text{top 5 percent}.
$$

The eligible set is

$$
\boxed{
E_r^{\rm cat}
=
\{\text{top-5-percent paths}\}
\cap
\{\mathcal R_{r_\star+1}<\mathcal R_{r_\star}\}.
}
$$

Detection is the path-first probability of accepting $q=r_\star+1$ in this set.

If any observed rank has no eligible catastrophic path, do not drop or renormalize it. Report available ranks descriptively and make the required rank-balanced verdict INCONCLUSIVE.

## 12. Bootstrap uncertainty

Use 10,000 stratified cluster-bootstrap replicates.

Bootstrap within the population relevant to each conditional metric:

$$
E_r^{\rm worse}=\{\text{truly worse paths}\},
$$

$$
E_r^{\rm better}=\{\text{truly better paths}\},
$$

and $E_r^{\rm cat}$ as defined above.

For aggregate results, retain all 200 repetitions of every sampled path. For batch results, retain that batch's 50. This conditional resampling prevents bootstrap-created zero denominators when the observed eligible population is nonempty.

Within each replicate, resample paths within rank, compute path-first rank metrics, and average the three ranks equally.

Use ordinary percentile intervals

$$
\boxed{
\mathrm{CI}_{95\%}
=[q_{0.025},q_{0.975}].
}
$$

Safety gates use the upper endpoint; power and detection gates use the lower endpoint. These intervals are conditional on the frozen orientations and paths.

Set the bootstrap master seed to 92000 and derive streams from fixed integer indices for estimator, $s$, $\varepsilon$, $\eta$, $m$, pair, metric, and batch scope. Record the full mapping in the manifest.

## 13. Primary gate and verdict

The primary setting is

$$
m=160,\qquad
\eta=10^{-6},\qquad
s=16,\qquad
\varepsilon=1/3,
$$

for candidate $r_\star+1$ versus baseline $r_\star$.

The verdict estimators are sample variance, mom_w1, and mom_w2. Paired mean is excluded from the verdict.

An estimator passes only when all conditions hold:

1. Primary false-safe rate is at most 0.5 percent and its upper confidence endpoint is at most 1 percent.
2. False-rejection rate is at most 5 percent and its upper endpoint is at most 10 percent.
3. Top-five-percent true-better catastrophic detection is at least 75 percent and its lower endpoint is at least 60 percent.
4. Overall true-better acceptance is at least 35 percent and its lower endpoint is at least 25 percent.
5. At $\eta=10^{-10}$ for the primary pair and otherwise primary settings, false-safe rate is at most 0.5 percent and its upper endpoint at most 1 percent.
6. All point thresholds pass in at least three of four batches. Interval criteria apply to the aggregate analysis.

Verdicts are:

- STRONG GO: all three verdict estimators pass at $s=16$;
- QUALIFIED GO: at least one but not all pass at $s=16$;
- BORDERLINE: none passes at $s=16$ but at least one passes at $s=32$, still with $m=160$, $\varepsilon=1/3$, and the primary pair;
- NO-GO: none passes by $s=32$;
- INCONCLUSIVE: a required rank-balanced gate is unevaluable.

Secondary settings cannot change the verdict.

## 14. Secondary analyses

Report all sample sizes, epsilon values, budgets, tails, action pairs, ranks, equal-rank aggregates, and four estimators.

When defined, risk-estimation summaries include signed, absolute, relative, and log-ratio error, plus median, 90th, 95th, 99th percentiles, and maximum.

Decision summaries include false-safe rate, false rejection, true-better acceptance, correct rejection, truth-specific abstention, coverage, error among decisions, tie counts, and eligible counts.

Report top one, five, and ten percent catastrophic detection. Every paired comparison must be labeled as paired.

## 15. Outputs and schemas

Create isolated artifacts:

- results/direct_rademacher_certification_phase1a_manifest.csv;
- results/direct_rademacher_certification_phase1a_truth.csv;
- results/direct_rademacher_certification_phase1a_trials.parquet;
- summary CSV files for accuracy, operating rates, bootstrap intervals, catastrophic strata, batch stability, gate evaluation, and verdict;
- figures under results/figures/direct_rademacher_certification_phase1a/;
- reports/direct_rademacher_risk_certification_phase1a.md.

The manifest records all configuration, seed contracts, software versions, estimator and decision definitions, gates, row counts, input/output checksums, and timestamps.

The truth-table unique key is

$$
(r_\star,\texttt{basis trial},\eta,m,q).
$$

It stores seeds, $q$, $r_q$, $\ell_q$, construction count, exact numerator and risk, frozen comparisons, and reconstruction/subspace errors.

The wide Parquet unique key is

$$
(r_\star,\texttt{basis trial},\eta,\texttt{batch},\texttt{repetition},s).
$$

The expected production count is

$$
3\cdot200\cdot2\cdot4\cdot50\cdot4
=
\boxed{960,000}.
$$

It stores identifiers, seeds and hash, $s$, query counts, per-action estimates and exact numerators, per-budget risks, pair truth and decisions for every epsilon, degeneracy flags, and validation flags. Use deterministic wide-column templates.

The report includes the executive verdict, estimand, reconstruction/accounting audit, estimators, decision rule, gate results, catastrophic detection, sensitivities, bootstrap interpretation, limitations, and Phase 1B recommendation. Label statements as PROVED, EMPIRICALLY ESTABLISHED, EMPIRICAL DECISION RULE, or OPEN.

At minimum create figures for estimated versus exact risk, error distributions, safety/error rates versus $s$, useful acceptance and abstention, catastrophic detection, estimator comparison at $s=16$, batch stability, and primary versus benign control.

## 16. Tests and mathematical verification

Tests must cover:

1. exhaustive tiny symmetric-matrix verification of

   $$
   \operatorname{Var}(g^\top Bg)
   =
   4\sum_{i<j}B_{ij}^2;
   $$

2. an explicit numerical symmetry assertion for $B=R_QAR_Q$;
3. $\mathbb E[W]=\operatorname{Var}(X)$ on exhaustive small examples;
4. equality of sample variance and the all-pairs formula;
5. sample-variance and paired-mean unbiasedness where exhaustive verification is practical;
6. MoM blocking, even medians, small-$s$ degeneracies, and $s=16$ nondegeneracy;
7. dense versus cached-$AQ$ matrix-free quadratic forms;
8. exact seed reproduction, nested $S$, paired orientations, reconstruction, ranks, query accounting, projector equivalence, frozen risks, and denominators;
9. probe invariance across every excluded seed field and stream changes across included fields;
10. distinct reconstruction/certification counters, cached $AQ$, zero certification-time $AQ$ calls, and exactly 32 certification queries;
11. truth labels, tolerance ties, zero/equal abstention, guard boundaries, denominators, and tie exclusions;
12. path-first and equal-rank aggregation, batch scope, conditional bootstrap populations, percentile intervals, empty eligibility, INCONCLUSIVE, and no rank renormalization;
13. 960,000 unique Parquet keys, truth uniqueness, complete configuration support, finite required fields, stable hashes, unchanged frozen checksums, and historical immutability.

## 17. Verification sequence

1. Compile all modified and new Python files.
2. Run targeted mathematical, estimator, reconstruction, seed, accounting, schema, and bootstrap tests.
3. Run the maintained suite through the repository's readline workaround.
4. Run a reduced smoke experiment into a temporary directory with few paths and repetitions but all estimators and sample sizes.
5. Validate smoke schemas, counters, invariance, reconstruction, and uniqueness.
6. Run production.
7. Publish final artifacts only after computation and validation succeed.
8. Validate row counts, keys, grid coverage, eligibility, path-first aggregation, intervals, verdict, checksums, and historical immutability.
9. Run the maintained suite again.
10. Run git diff --check.
11. Audit the report line by line for theorem-versus-empirical wording.

## 18. Documentation and research record

After implementation and analysis:

- update the workspace UROP_TRACKER.md with touched files, frozen design, production counts, gate results, verdict, mathematical classification, and remaining Phase 1B question;
- update workspace memory.md and CURRENT_STATE.md;
- synchronize the tracker, memory, current-state note, and Phase 1A report to /Users/chenyixin/Documents/Obsidian_Vault/Swati_Summer_Research;
- synchronize the tracker's Last updated date with its newest log entry;
- retain the completed-UROP freeze and label Phase 1A as continuation work.

## 19. Fixed assumptions and interpretation limits

- There are 200 frozen paths for each rank.
- Results are conditional on the frozen orientations and path population.
- Range sketch notation is $S$ and certification vectors are $g$.
- Certification vectors are external diagnostics, not final residual probes.
- Pairing across tails, budgets, actions, estimators, and sample sizes is deliberate.
- Sample variance and paired mean are generally unbiased; the MoMs are not claimed generally unbiased.
- The multiplicative guard is empirical, not a theorem certificate.
- The primary gate is frozen at $m=160$, $s=16$, $\varepsilon=1/3$, $\eta=10^{-6}$, and $r_\star+1$ versus $r_\star$.
- The benign control uses $\eta=10^{-10}$ and the primary pair.
- BORDERLINE changes only $s$ to 32.
- No action, estimator, threshold, seed, or verdict rule is retuned after production.
- GO means that small fresh-probe batches are empirically informative under the frozen experiment.
- NO-GO means that $s\le32$ failed this architecture's preregistered safety and power requirements.
- Neither verdict proves or disproves other certification methods.
- Certification cost enters the total budget only in a later online architecture.
