# Adaptive Hutch++: query allocation, subspace capture, and the limits of risk certification

**Consolidated research report — September 16, 2026**  
**Project:** UROP / randomized numerical linear algebra  
**Faculty advisor:** Prof. Swati Padmanabhan  
**Status:** Source-backed current narrative; not a claim of a completed certified allocator.

This is the current consolidated report. The [earlier progress draft](urop_research_progress_report_aug2026.md) is preserved as history, including its original metadata and figures. Its PDF, LaTeX, and HTML exports have not been regenerated and are not corrected versions of this report. The [claim register](urop_claims_register_20260916.md) records what changed and why.

## Reading guide

Visual companions: [validated figures and full captions](urop_figure_guide_20260916.md), plus a separate [GPT-polished presentation edition](urop_gpt_polished_figures_20260916.md). Quantitative figures embedded in this report remain the original data-rendered versions.

For the main story, read Sections 1, 3, 9, and 12. Sections 2 and 4 explain the mathematics. Sections 6–8 explain what happened after introducing Rademacher risk estimation. Section 11 distinguishes reproducibility from a scientific guarantee.

## 1. Executive conclusion

The project began with a simple question: can Hutch++ improve by adapting how many queries it spends constructing a low-rank basis? The evidence does not support a uniformly better adaptive split. It supports a more specific conclusion:

> Spending more queries on low-rank capture is justified only when the actual reduction in probe-specific residual risk outweighs the loss of residual samples. A pilot's fitted spectrum, numerical rank, or sharp Ritz knee alone does not establish that reduction.

The work identifies several distinct obstacles: irreversible pilot commitment, incorrect allocation proposals, incomplete randomized capture, probe-distribution dependence, and numerical or signal-subspace rank loss. These are not interchangeable explanations.

The September 16 audit supplies an especially concrete example. A pilot produced eight independent sampled directions but captured only four of five dominant signal directions. Its Ritz gap looked excellent because an entire dominant direction was invisible. That one path dominated the coordinate-aligned experiment's error. This strengthens the research contribution as a diagnosis of what adaptive trace estimation must measure; it does not justify another untested allocator.

The current deliverable is therefore a coherent account of proved accounting/risk identities, controlled experiments, informative negative results, and an unresolved certification problem. Adaptive Hutch++ is the primary UROP story. TurboQuant and leverage-score work remain supporting RandNLA preparation in the [workspace tracker](../../../UROP_TRACKER.md), rather than competing central claims.

## 2. Authoritative notation and exact estimator risk

Let $A\in\mathbb R^{d\times d}$ be a fixed real symmetric matrix. PSD assumptions are required when interpreting the positive-eigenvalue spectral models below, but not for the conditional trace/variance identities in this section.

| Symbol | Meaning |
|---|---|
| $m$ | Total matrix–vector query budget |
| $S$ | Range-sketch matrix; do not substitute $\Omega$ |
| $q$ | Number of attempted sketch columns, costing $q$ products with $A$ |
| $Q\in\mathbb R^{d\times r}$ | Accepted orthonormal basis, with realized rank $r$ |
| $r_\star$ | Dominant signal dimension of a synthetic step spectrum; not necessarily $r$ |
| $R=I-QQ^\top$ | Residual projector |
| $H=RAR$ | Actual double-projected residual matrix |
| $\ell$ | Number of fresh final residual probes |

For the basic cached-basis architecture, constructing $AS$ and $AQ$ costs $q+r$. Thus

$$q+r+\ell=m,\qquad \ell=m-q-r>0.$$

Only when $r=q$ may this be shortened to $\ell=m-2q$. A rejected sketch column still costs a query even if it does not increase $r$.

Condition on all information $\mathcal G$ used to construct and select the action, before final residual probes are drawn. For conditionally iid fresh probes with $\mathbb E[gg^\top\mid\mathcal G]=I$,

$$\widehat t=\operatorname{tr}(Q^\top AQ)+\frac1\ell\sum_{j=1}^{\ell}g_j^\top H g_j.$$

Since $R^2=R$ and $Q^\top Q=I$,

$$\operatorname{tr}(Q^\top AQ)+\operatorname{tr}(RAR)=\operatorname{tr}(A).$$

Therefore $\mathbb E[\widehat t\mid\mathcal G]=\operatorname{tr}(A)$, including a randomly stopped pilot whose decision is measurable before fresh residual sampling.

For Rademacher probes, symmetry gives

$$g^\top Hg=\operatorname{tr}(H)+2\sum_{i<j}H_{ij}g_i g_j.$$

Distinct sign products in this sum are orthogonal in expectation. Consequently

$$\operatorname{Var}(g^\top Hg\mid\mathcal G)=4\sum_{i<j}H_{ij}^2=2E_R(Q),
\qquad E_R(Q)=\sum_{i\ne j}H_{ij}^2.$$

For Gaussian final probes, the corresponding variance is $2E_G(Q)$, with $E_G(Q)=\|H\|_F^2$. The exact conditional risks are therefore

$$\boxed{\mathcal R_R(Q;q,r)=\frac{2E_R(Q)}{m-q-r},\qquad
\mathcal R_G(Q;q,r)=\frac{2E_G(Q)}{m-q-r}.}$$

Because conditional bias is zero, averaging conditional risk over the basis/decision randomness gives unconditional MSE. A finite average of reconstructed conditional risks is still an empirical estimate of that expectation, not the population expectation itself.

Gaussian/Rademacher **risk labels describe final probes**. They do not automatically describe the range sketch. In particular, the legacy function `Gaussian_Hutch_pplus` draws a Gaussian range sketch and Rademacher residual probes. It also uses a different allocation. It is not an all-Gaussian rotation-invariant control.

Sources: [sequential unbiasedness](../docs/proof_sequential_unbiasedness.md), [rank-aware theory](../docs/proof_rank_aware_risk.md), and [the latest implementation audit](coordinate_gate_failure_audit_20260916.md).

## 3. How the research question changed

| Stage | Initial idea | What the evidence required us to distinguish |
|---|---|---|
| Spectral adaptation | Fit decay and choose $q$ | A fit to observed Ritz values need not predict the unobserved tail |
| Sequential pilot | Sample until a knee becomes visible | More information commits more queries; the final action must satisfy $q\ge B$ |
| Allocation guards | Clip movement around Standard Hutch++ | A fixed guard is a heuristic, not evidence that a particular move is beneficial |
| Exact risk bridge | Compare spectral and realized objectives | Correct dimension does not imply accurate capture |
| Small fresh-probe experiment | Estimate actual Rademacher residual variance | An empirically useful decision rule is not a finite-sample certificate |
| Budget-aware emulation | Pay for construction and certification | Reverting to a baseline basis cannot undo spent queries |
| Confidence studies | Prove explicit small-budget safety | The audited worst-case bounds are too conservative on the frozen grid |
| Exploratory Ritz gate | Use cheap pilot features instead | A sharp visible knee can conceal a completely missed direction |

Earlier approaches are retained as research history, not silently relabeled as successful final algorithms. For example, the boundary-anchored exponential model has an exact log-tail-ratio identity showing how slope error is amplified with extrapolation distance. That is a model-conditional sensitivity result—not a universal measured $10^{10}$ error factor. See [the exponential proof note](../docs/proof_exponential_sensitivity.md).

Likewise, a pilot smaller than the dominant dimension cannot reveal an adjacent gap beyond its available Ritz values. This observation does not prove a universal width threshold $b\ge1.33r_\star$. The older draft's numerical “law” is not adopted here.

## 4. The local mathematical decision: what buys another direction?

For a PSD matrix with decreasing eigenvalues, define the ideal tail $T(r)=\sum_{j>r}\lambda_j^2$ and $D=m-q-r$. The ideal rank-aware risk is

$$\mathcal R_{\rm rank}(q,r)=\frac{2T(r)}D.$$

If a successful next sketch query produces one accepted direction **and captures the next leading eigenvector**, then $D$ decreases by two. For $D>2$,

$$\mathcal R_{\rm new}-\mathcal R_{\rm old}
=-\frac{2[D\lambda_{r+1}^2-2T(r)]}{D(D-2)}.$$

That ideal action helps exactly when $D\lambda_{r+1}^2>2T(r)$. The ideal-capture assumption is essential.

For an actual accepted extension $Q\to Q'$, replace the ideal tail by $E_X$, where $X$ identifies the final-probe distribution. Direct subtraction gives

$$\mathcal R_X(Q';q+1,r+1)-\mathcal R_X(Q;q,r)
=-\frac{2\{D[E_X(Q)-E_X(Q')]-2E_X(Q)\}}{D(D-2)}.$$

If $E_X(Q)>0$, improvement is equivalent to

$$\boxed{\frac{E_X(Q)-E_X(Q')}{E_X(Q)}>\frac2D.}$$

In words: the fractional residual-energy reduction must exceed the fraction of residual-sample capacity surrendered. If the old energy is zero, use the undivided identity; a risk ratio is not defined. Rademacher off-diagonal energy need not decrease merely because the subspace grows.

A failed attempt that leaves the basis unchanged costs one query. For $D>1$, its risk change is $2E_X(Q)/[D(D-1)]\ge0$. It is zero when the residual risk is already zero. The ideal expression substitutes $T(r)$ for $E_X(Q)$.

These are exact accounting identities, not an implementable decision rule until the relevant energy or its change is known. Source: [rank-aware marginal theory and bridge](rank_deficient_bridge_analysis.md).

## 5. What the frozen bridge established

The four-level comparison keeps distinct

$$\frac{2T(q)}{m-2q},\quad \frac{2T(r_q)}{m-q-r_q},\quad
\frac{2E_G(Q_q)}{m-q-r_q},\quad \frac{2E_R(Q_q)}{m-q-r_q}.$$

They respectively represent full-rank ideal capture, rank-aware ideal capture, realized Gaussian risk, and realized Rademacher risk. The sequence is a comparison of assumptions, not a blanket chain of inequalities. In particular, a spectral tail alone does not encode coordinate-dependent Rademacher risk.

The rank-deficient bridge uses $d=500$, $r_\star\in\{5,15,30\}$, $\eta\in\{0,10^{-14},10^{-10},10^{-6}\}$, and $m\in\{80,160,240\}$, with 200 paths per setup. Orientations are paired across tail levels, and sketch prefixes are nested. The validated output contains 554,400 trial-allocation rows. Exact and numerical rank loss are distinguished from incomplete capture at full accepted rank.

At $\eta=10^{-6}$, the reported empirical-mean optima satisfy $q_{\rm rank}^\star=r_\star$ and $q_G^\star=q_R^\star=r_\star+1$. These are minimizers of finite empirical-mean curves, not universal optimality theorems.

At the square boundary, when $S_1=U_1^\top S$ is invertible, the step model has graph factor

$$F=\eta S_2S_1^{-1}.$$

A poorly conditioned $S_1$ amplifies the tail and worsens capture. Nested oversampling greatly reduced the observed extreme paths. But the median path preferred the original knee; the mean benefit came from tail protection. Gaussian mean-risk-difference bootstrap intervals at $m=160$ included zero, and a broader sensitivity grid produced optimal shifts of zero, one, two, and larger.

Thus the justified conclusion is **oversampling can protect against rare capture failures**, not “one extra column always suffices” or “catastrophes are eliminated.” Sources: [bridge analysis](rank_deficient_bridge_analysis.md) and [mechanism audit](q_rank_vs_realized_risk_mechanism.md).

## 6. Why estimating Rademacher risk directly was promising—and costly

### Phase 1A: does a small fresh sample carry useful information?

For a fixed basis, write $X_j=g_j^\top Hg_j$ and $\sigma^2=2E_R(Q)$. The unbiased sample variance $S_s^2$ estimates $\sigma^2$, so $S_s^2/\ell$ estimates the exact conditional risk. Candidate and baseline use common fresh probes. With cached $AQ$, the identity

$$A(g-Q Q^\top g)=Ag-AQ(Q^\top g)$$

lets one new $Ag$ serve several preconstructed actions. The cached products are not free: they were paid during construction.

Phase 1A contains 960,000 wide diagnostic rows using 200 paths per rank, two tail levels, four batches, 50 repetitions per batch, and $s\in\{4,8,16,32\}$. These rows are repeated measurements of frozen paths, not 960,000 independent randomized bases. Certification queries are external diagnostics in this phase; they are not yet deducted from $\ell$.

At the preregistered primary setting ($m=160$, $\eta=10^{-6}$, $s=16$), sample variance passed the empirical gate; the two small-block median-of-means candidates did not. The verdict was **QUALIFIED GO**.

| Sample-variance metric | Point estimate | Conditional 95% percentile interval |
|---|---:|---:|
| False-safe acceptance, conditional on truly worse paths | 0.0134% | [0.0028%, 0.0305%] |
| False rejection, conditional on truly better paths | 0.5157% | [0.2991%, 0.7722%] |
| Top-5% catastrophic true-better detection | 79.6667% | [69.0662%, 89.2833%] |
| Acceptance among truly better paths | 52.0991% | [40.3055%, 63.5069%] |

Rates average repetitions within paths, paths within rank, then ranks equally. Bootstrap intervals condition on the frozen orientations and path population. A low empirical false-safe rate is not a theorem about the probability of safety conditional on acceptance. Source: [Phase 1A report](direct_rademacher_risk_certification_phase1a.md).

![Figure 2 — empirical selective-risk metrics](../figures/urop_validated_20260916/figure2_empirical_signal.png)

**Figure 2.** Two selected empirical criteria at $s=16$. Bars are the original conditional 95% percentile intervals, not safety guarantees. There are 200 paths per rank, with 200 repeated certification trials per path. Only eligible truth-conditioned populations enter each metric. Sample variance alone passed the complete gate. See [full caption and source data](urop_figure_guide_20260916.md#figure-2-empirical-selective-risk-metrics).

### Phase 1B: what happens when those queries are charged?

For the **nested shared-prefix primary architecture**, cached construction has cost

$$c_{\rm pre}=\max\{q_0+r_0,q_a+r_a\}.$$

For other architectures use the actual committed query ledger, not this maximum. After $s$ certification queries, the available final denominator is $\ell_{\rm paid}=m-c_{\rm pre}-s$. Returning to the baseline basis does not restore the original $\ell_0=m-q_0-r_0$.

For positive baseline variance, a paid candidate beats the unstarted baseline exactly when

$$\frac{\sigma_a^2}{\sigma_0^2}<\frac{\ell_{\rm paid}}{\ell_0}.$$

The primary equal-rank mean of within-rank selected/original mean-risk ratios was **0.036329**, with conditional interval **[0.014628, 0.639830]**. The paid-oracle comparison was 0.035411; always paying and reverting gave 1.172197. About 95.83% of paths were harmed after cost. Large aggregate gains came from rare failures, not typical-path improvement.

This is **NET BENEFIT + TAIL-INSURANCE TRADEOFF**, not universal safety. The original comparator here is the frozen adjacent baseline action, not automatically Standard Hutch++'s $q_0=\lfloor m/3\rfloor$. Source: [budget-aware report](direct_rademacher_risk_certification_phase1b_budget.md) and the [tracker](../../../UROP_TRACKER.md).

![Figure 3 — certification cost and pathwise net effects](../figures/urop_validated_20260916/figure3_certification_cost.png)

**Figure 3.** Aggregate mean-risk improvement and typical-path harm coexist. Panel a averages within-rank ratios of mean risks; panel b shows all 600 individual pathwise ratios. They are different estimands. Exactly 575 paths are harmed after cost. The pooled pathwise median is 1.160714; the historical 1.172197 median summary averages rank-specific medians instead. The original shared-bootstrap intervals in panel a do not give a pathwise safety guarantee. See [full caption and source data](urop_figure_guide_20260916.md#figure-3-certification-cost-and-pathwise-net-effects).

## 7. What the negative confidence results actually say

The elementary conditional sample-variance identity is

$$\operatorname{Var}(S_s^2)=\frac1s\left[\mu_4-\frac{s-3}{s-1}\sigma^4\right].$$

Phase 1C retained the valid but conservative degree-two fourth-moment factor 81. The joint two-action Chebyshev radius was

$$\varepsilon_{\rm Ch}(s,\delta_{\rm joint})=
\sqrt{\frac{2[80+2/(s-1)]}{s\delta_{\rm joint}}}.$$

At $(s,\delta_{\rm joint})=(32,0.05)$ it is 10.004031, and it exceeds one throughout the declared smaller grid. That particular multiplicative construction is **PROVED BUT BUDGET-VACUOUS**.

Phase 1D then examined a truncation–Bernstein bound for the nondegenerate linear projection. On its frozen caps $T\ge64$, the retained variance envelope satisfies $80\le\nu_\kappa(T)\le81$. For an admissible candidate-underestimation radius $0<\varepsilon<1$,

$$D_{s,\kappa}(\varepsilon,T)>e^{-s/160}\ge e^{-0.2}\approx0.81873
\quad(s\le32).$$

Here $D_{s,\kappa}$ is the **proposed probability upper bound**. This inequality does not say that the actual failure probability exceeds 0.81873. It shows that this bound cannot certify the required small failure probability. The route-specific verdict is **STRONG LINEAR NO-GO**.

These results neither prove all separate-action confidence methods impossible nor refute the positive empirical signal. They identify limits of the explicitly audited bounds. Candidate-underestimation control needs a radius below one for division by $1-\varepsilon$; baseline-overestimation control requires a finite nonnegative radius, not necessarily one below one.

Sources: [Phase 1C](rademacher_sample_variance_confidence_phase1c.md) and [Phase 1D](rademacher_linear_projection_no_go_phase1d.md). The later degree-two refinement in Section 8 does not retroactively change the frozen 81-based calculations.

## 8. Paired differences and the remaining theorem gap

### Phase 2A: common probes can cancel fluctuations

For $Y_a=S_a^2/\ell_a$ and $Y_0=S_0^2/\ell_0$,

$$\operatorname{Var}(Y_a-Y_0)=\operatorname{Var}(Y_a)+\operatorname{Var}(Y_0)-2\operatorname{Cov}(Y_a,Y_0).$$

The independent benchmark is the sum of the two variances. Its formula is exact; its numerical values in the experiment are estimated. The primary equal-rank pairing ratio was **0.151032**, with interval **[0.137069, 0.166735]**, an observed reduction of approximately 84.9% relative to the plug-in benchmark.

The benefit was much weaker on catastrophic paths: their mean pairing ratio was **0.899565**, versus **0.111635** on ordinary paths. This limitation matters because the rare beneficial paths are the reason to seek certification. The cyclic shifted comparator is **shifted/decorrelated**, not an independent sequence. Source: [Phase 2A](paired_rademacher_risk_difference_phase2a.md).

### Phase 2B: a cleaner target still faces an unknown-scale problem

For disjoint pairs of fresh common probes, define

$$D_j=\frac{(X_{a,2j-1}-X_{a,2j})^2}{2\ell_a}
-\frac{(X_{0,2j-1}-X_{0,2j})^2}{2\ell_0}.$$

Conditionally, these are iid across pairs and have mean $\Delta_R=\sigma_a^2/\ell_a-\sigma_0^2/\ell_0$. The dependence between actions remains inside each observation. The centered variable is a degree-at-most-four polynomial in the pair's signs. The audited moment control in Phase 2B is

$$\|D_j-\Delta_R\|_4\le9\|D_j-\Delta_R\|_2,
\qquad \mathbb E(D_j-\Delta_R)^4\le6561\operatorname{Var}(D_j)^2.$$

The implemented data-only scale constructions remain vacuous for $s\le32$. Thus the result is **PROVED BUT BUDGET-VACUOUS**, not a useful online certificate. A bound for a canonical/degenerate U-statistic cannot be applied to the full sample variance without controlling its nondegenerate Hoeffding projection. Source: [Phase 2B](paired_rademacher_difference_confidence_phase2b.md).

### Phase 2C: structural information helps only under its stated assumptions

The recovered structural-prior theorem supplies a finite one-sided radius when a valid pre-supplied norm envelope and nested residual projectors are available. It does not supply that prior from the observations, demonstrate useful acceptance, or restore original-baseline safety after abstention.

The corrected degree-two fourth-moment calculation gives $(3+12\kappa^2)\sigma^4$ as an upper bound for a centered quadratic chaos. That is not a kurtosis bound for the degree-four signed-pair variable above. Its exact Chebyshev feasibility condition, for an assumed observation kurtosis bound $K$, is

$$\delta_{\rm scale}n(n-1)>(K-1)(n-1)+2.$$

At $K=3$ and $\delta_{\rm scale}=0.05$, $n=41$ is equality and $n=42$ is the first feasible integer. With two probes per signed observation, this means $s=84$, not 82. This is a limit of that route; it does not prove a norm prior necessary for every possible method. Source: [corrected Phase 2C report](structural_paired_difference_confidence_phase2c.md).

## 9. The exploratory gate and the newly verified missing-direction failure

The two-stage heuristic reuses an initial pilot and selects a small sketch if a sharp Ritz gap is detected; otherwise it aims for the Standard allocation. Reusing queries avoids a separate certification batch, but does not solve confidence certification. Returning to the same allocation is not a theorem of identical basis quality or MSE.

The recovered orientation experiment uses $d=100,m=60$, six spectra, 30 Haar orientations plus a coordinate orientation, and ten paths per orientation. In the rank-five step case ($\eta=0.001$), every gate triggers and selects $(q,r,\ell)=(8,8,44)$.

| Rank-five step group | Paths | Gated observed MSE | Standard observed MSE | Gated/Standard MSE ratio |
|---|---:|---:|---:|---:|
| Haar orientations | 300 | 2.839433e-7 | 1.672813e-6 | 0.169740 |
| Coordinate aligned | 10 | 3.303942e-3 | 7.209998e-7 | 4582.446 |

Both outcomes must be reported. A high gain on rotated examples does not invalidate the coordinate counterexample.

![Figure 1 — coordinate pilot capture and residual risk](../figures/urop_validated_20260916/figure1_capture_failure.png)

**Figure 1.** All ten coordinate paths, two explicitly selected Ritz spectra, and the exact failing signal sketch. Panel a displays conditional risk, not observed squared error. Every gated basis has rank eight, yet trial 7 misses a unit-eigenvalue direction because signal rows three and four are opposites. No population failure frequency is inferred from these ten paths. See [full caption and source data](urop_figure_guide_20260916.md#figure-1-coordinate-pilot-capture-and-residual-risk).

### Exact algebra behind the failure

On coordinate trial 7, rows three and four of $S_1=U_\star^\top S$ are exact negatives. Rational elimination gives signal rank four although the full sample and basis have rank eight. Let

$$w=(e_3+e_4)/\sqrt2.$$

Then $S^\top w=0$ and $Aw=w$. Therefore $(AS)^\top w=0$, and in exact arithmetic

$$Q^\top w=0,\qquad RARw=w.$$

The replay gives $\|Q^\top w\|_2\approx8.07\times10^{-14}$. The missing direction's largest subspace angle is 90 degrees. Eight independent sampled directions are not eight correctly targeted signal directions: the small tail fills out the accepted rank.

The Ritz matrix has four values near one and four near 0.001. The gap is approximately 6.907721 and the contrast approximately 69,077, so the later contrast threshold of 5 would not reject this recorded pilot.

For a particularly direct identifiability argument, set $A'=A-(1-\eta)ww^\top$. This matrix has only four unit eigenvalues but satisfies

$$A'S=AS,\qquad A'Q=AQ.$$

The entire pilot transcript is identical for $A$ and $A'$. Thus these pilot observations alone cannot always distinguish four from five dominant directions. This local construction does not rule out probabilistic guarantees or future fresh-query tests.

### Exact risk confirms that the failure precedes final-probe noise

The bad basis has $E_G\approx1.000091$ and $E_R\approx0.500004$. Its exact conditional Rademacher risk is **0.02272744518**, compared with ordinary gated coordinate-path risk near **1.31e-7**. Its realized squared error, 0.03303752, is only 1.45364 times its own conditional risk.

This one path contributes **99.9948% of total conditional risk** and **99.9942% of observed squared error** in the ten coordinate trials. The failure is therefore not explained away by an unlucky final draw. There is too little coordinate data to estimate a universal catastrophe frequency.

Even though the input is diagonal, the randomized residual need not be: the missed projector $ww^\top$ alone produces $g^\top ww^\top g=1+g_3g_4$, a variable taking values zero and two. Classical Rademacher Hutchinson is exact on the original diagonal matrix; randomized projection can create off-diagonal residual variance.

This event occurs with positive oversampling $q-r_\star=3$, not at the earlier zero-oversampling boundary. It is exact discrete signal-rank failure, rather than merely near-singular full-rank capture. The two mechanisms support the same warning but should not be conflated.

The focused audit replayed all 310 first-spectrum paths and recorded 620 Standard/gated basis diagnostics. Source: [coordinate-gate audit](coordinate_gate_failure_audit_20260916.md), including the exact rational witness, original RNG states, risk tables, and accounting.

## 10. Additional empirical work and its proper scope

### Synthetic-data neural Hessian benchmark

The matrix-free operator experiment uses a 4,254-parameter convolutional network trained on synthetic images. The model has ten outputs, but the current labels are only zero and one. Its Hessian need not be PSD. The floating-point reference trace is approximately 31.581270, obtained with 4,254 reference HVPs separate from estimator budgets.

| Budget | Standard Hutch++ MSE | Gated MSE |
|---:|---:|---:|
| 30 | 5.858498 | 5.334005 |
| 60 | 1.417330 | 0.841176 |
| 90 | 1.006494 | 0.570674 |
| 120 | 0.457086 | 0.370210 |

There are 30 trials per method/budget, 480 total across four methods. These point estimates demonstrate the operator implementation and exploratory gains on this model. They do not establish general ML performance, confidence-certified safety, or a wall-clock speed advantage. In particular, the increased number of residual samples alone does not explain the gain because the residual basis also changes. Sources: [Hessian report](pytorch_hessian_benchmark.md) and [recovery record](recovery_audit_20260914.md).

### Development grids and earlier real-data studies

The gate-feature study has 23 stored development configurations; its thresholds were evaluated on the same grid. Its empirical fixed-$q$ minima are noisy grid references, not exact conditional-risk oracles. The favorable trigger patterns do not supply a safety theorem. The later contrast filter is not the filter used in the recovered Haar/Hessian benchmarks. Source: [gating diagnostic](gating_diagnostics_predictability_map.md).

Earlier YearPredictionMSD and Wiki-Vote experiments remain part of the project history. The old draft quotes effective ranks, crossover budgets, and variance-reduction factors. Those particular numerical claims have not received a source-level reproduction check in this consolidation and are not used in its central conclusions. No universal law that effective rank alone determines performance is asserted. A verified final figure would need a clear operator definition, budget, probe law, replication unit, and uncertainty description.

## 11. Reproducibility, uncertainty, and figure policy

The September recovery corrected smoke-test writes that had overwritten untracked production CSVs. It reran the original protocols and validated the restored results. Original overwritten hashes were unavailable: this was protocol-based recovery, not byte-identical restoration. Output-directory isolation, checksum guards, and explicit query counters now protect the maintained runs. See [the recovery audit](recovery_audit_20260914.md).

The subsequent coordinate audit preserved 156 pre-existing source/result files and passed all **225 maintained tests**, including historical-result protection. Its 74,400 replay queries and 2,000 independent energy-check queries are diagnostic work, not additions silently charged to a 60-query estimator.

This report consolidation issues **no new matrix–vector queries**, changes no estimator or threshold, and does not regenerate historical results. The 225-test result belongs to the preceding implementation audit, not a claim that this documentation turn reran the entire suite.

Uncertainty must follow the unit of replication. Certification repetitions are clustered within frozen paths; paths can share orientations. Existing bootstrap intervals are conditional on the observed frozen population. They do not measure unobserved rare-event probabilities or orientation-universal performance. The exploratory Haar/Hessian MSE tables above have no newly claimed confidence intervals.

The old draft's eight illustrations and exported PDFs are preserved, but not automatically adopted as validated scientific figures. This current report now includes three source-backed figures, with editable SVGs, PNG previews, eight source-data CSVs, full captions, and provenance checks in the [figure guide](urop_figure_guide_20260916.md). The package is numbered by its conceptual reading order—capture, signal, cost—rather than first appearance in this phase-ordered report. Five focused figure tests passed; the full estimator suite was not rerun for this rendering-only change. No new queries or bootstrap samples were generated. Illustrations making “complete suppression” or universal-threshold claims require revision before use in a poster.

## 12. Final contribution, open question, and next deliverable

The strongest supported contribution is a sequence of increasingly precise distinctions:

1. Attempted sketch queries are not the same as realized basis rank.
2. Realized basis rank is not the same as dominant-subspace capture.
3. A spectral-tail surrogate is not the same as actual probe-specific risk.
4. A useful empirical comparison is not the same as a finite-sample certificate.
5. A safe accepted action is not the same as a baseline-safe complete policy after sunk costs.
6. A sharp pilot Ritz knee is not evidence that no important direction was missed.

The open problem is to obtain **observable, budget-feasible information about a realized risk difference or a capture failure**, with a valid uncertainty statement and a physically feasible fallback. A successful solution must pay for the information it uses and preserve freshness of final residual probes. No solution is claimed here.

For this UROP, the corrected narrative and three source-backed figures are ready for review and adaptation to the required report/poster format. The remaining presentation choice is the venue's format, length, and deadline—not another allocator or broad benchmark. A future capture-verification or paired-confidence project should have its own frozen design before implementation.

### Evidence-status summary

- **PROVED under stated assumptions:** accounting; conditional unbiasedness and risk identities; realized marginal algebra; the local missing-direction and identical-pilot-transcript construction; the route-specific confidence bounds recorded in their proof notes.
- **EMPIRICALLY ESTABLISHED on the stated datasets:** tail-dominated mean risks; Phase 1A/1B selective behavior and cost tradeoff; common-probe covariance gain; the reproduced coordinate failure and exploratory rotated/Hessian gains.
- **HEURISTIC:** pilot-driven allocation and the contrast threshold.
- **OPEN:** small-budget observable confidence with useful acceptance, complete-policy baseline safety, and external generalization of the exploratory gate.

The [claim register](urop_claims_register_20260916.md), [tracker](../../../UROP_TRACKER.md), and individual linked audit reports retain the detailed evidence trail. The original progress draft remains available; this report supersedes its scientific framing rather than deleting the project's history.
