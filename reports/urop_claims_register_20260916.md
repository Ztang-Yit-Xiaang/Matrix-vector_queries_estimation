# UROP claim register — September 16, 2026

This register supports the [current consolidated report](urop_research_report_20260916.md). It records corrections to the [historical progress draft](urop_research_progress_report_aug2026.md); no historical experiment is overwritten. “Unverified here” does not mean false, but excludes the claim from the current report's central evidence.

| Historical wording or possible misreading | Current disposition | Supporting source |
|---|---|---|
| Adaptive Hutch++ uniformly improves on Standard | Not established; retain both observed gains and failures | [Coordinate audit](coordinate_gate_failure_audit_20260916.md) |
| $q+r+\ell=m$ can always be replaced by $m-2q$ | Only when $r=q$; attempted queries and accepted rank differ | [Rank-aware proof](../docs/proof_rank_aware_risk.md) |
| A universal pilot “horizon law” is $b\ge1.33r_\star$ | No verified universal theorem found in the reviewed sources; do not present as proved | [Current mechanism audit](coordinate_gate_failure_audit_20260916.md); historical draft is the located source of the numerical claim |
| Pilot extrapolation errors universally explode by $10^{10}$ | Retain model-conditional sensitivity theory; that specific empirical magnitude is unverified here | [Exponential sensitivity proof](../docs/proof_exponential_sensitivity.md) |
| Oversampling always moves the optimum by exactly $+1$ | Finite empirical means show $+1$ in the frozen small-tail regime; other regimes differ | [Original mechanism audit](q_rank_vs_realized_risk_mechanism.md) |
| The worst 1% always carry 99.9% of risk | Contributions depend on rank, probe law, and regime; quote per-case evidence | [Tail table and uncertainty](q_rank_vs_realized_risk_mechanism.md) |
| One extra column completely eliminates catastrophic failure | Not a guarantee; current discrete signal-rank failure occurs at $q-r_\star=3$ | [Coordinate audit](coordinate_gate_failure_audit_20260916.md) |
| Correct numerical rank certifies signal capture | Refuted by signal rank four with accepted rank eight | [Exact rational witness](../results/coordinate_gate_failure_audit_20260916/exact_nullspace_witnesses.json) |
| 960,000 Phase 1A rows prove statistical safety | Empirical selective-rule evaluation, with repetitions clustered within frozen paths | [Phase 1A](direct_rademacher_risk_certification_phase1a.md) |
| $c_{\rm pre}=\max(q_a+r_a,q_0+r_0)$ for any architecture | Only the nested shared-prefix architecture; otherwise use committed query count | [Phase 2A accounting clarification](paired_rademacher_risk_difference_phase2a.md) |
| Abstention restores the unstarted baseline | Spent construction/certification queries remain spent | [Budget-aware emulation](direct_rademacher_risk_certification_phase1b_budget.md) |
| $D_{s,\kappa}>0.8187$ means actual failure probability exceeds 81.87% | It is a lower bound on this method's probability upper bound, showing bound vacuity | [Phase 1D](rademacher_linear_projection_no_go_phase1d.md) |
| Data-only certification is universally impossible at $s\le32$ | Only the audited explicit routes have been shown vacuous | [Phase 2B](paired_rademacher_difference_confidence_phase2b.md) |
| Pairing provably reduces variance by 84.9% in general | Covariance identity is exact; 84.9% is a conditional empirical aggregate | [Phase 2A](paired_rademacher_risk_difference_phase2a.md) |
| Pairing is equally effective on catastrophic paths | Not supported: recorded catastrophic mean ratio 0.899565 versus ordinary 0.111635 | [Phase 2A strata](paired_rademacher_risk_difference_phase2a.md) |
| A cyclic shift is an independent comparator | It is shifted/decorrelated; the sum-of-variances formula is the exact independence benchmark | [Phase 2A](paired_rademacher_risk_difference_phase2a.md) |
| The old displayed $3^{4/2\times2}$ proves the factor 6561 | That displayed arithmetic is wrong; the audited norm factor is 9, and $9^4=6561$ | [Phase 2B theorem summary](paired_rademacher_difference_confidence_phase2b.md) |
| A finite norm-prior radius proves practical usefulness | It needs a valid supplied prior and nesting; useful acceptance remains open | [Corrected Phase 2C](structural_paired_difference_confidence_phase2c.md) |
| The degree-two kurtosis bound sharpens degree-four signed pairs automatically | Different random variables and polynomial degrees; no such substitution justified | [Corrected Phase 2C](structural_paired_difference_confidence_phase2c.md) |
| $K=3,\delta=0.05$ gives $n=41,s=82$ | Strict feasibility begins at $n=42,s=84$ | [Exact boundary](structural_paired_difference_confidence_phase2c.md) |
| The two-stage gate solves the certification tax | It avoids a separate certification batch but is an uncertified heuristic | [Budget tradeoff](direct_rademacher_risk_certification_phase1b_budget.md), [gate failure](coordinate_gate_failure_audit_20260916.md) |
| Falling back to $q_0$ proves MSE equivalence | Same allocation does not establish identical basis quality or pathwise risk | [Recovery audit](recovery_audit_20260914.md) |
| Trigger consistency proves rotational invariance | Only finite-grid trigger rates were observed | [Orientation audit](haar_orientation_audit.md) |
| “Gaussian Hutch++” is an all-Gaussian rotation-invariant control | Gaussian range sketch, Rademacher residual probes, and a different allocation | [Implementation diagnosis](coordinate_gate_failure_audit_20260916.md) |
| Diagonal input implies diagonal randomized residual | False in general; $RAR$ can have substantial off-diagonal energy | [Coordinate proof and example](coordinate_gate_failure_audit_20260916.md) |
| Contrast 5 guarantees detection or guards against missing modes | Development-grid heuristic; the failing pilot has contrast approximately 69,077 | [Coordinate audit](coordinate_gate_failure_audit_20260916.md) |
| Neural Hessian results are a real-data ten-class PSD benchmark | Synthetic labels 0/1, ten model outputs, Hessian not assumed PSD | [Recovered Hessian report](pytorch_hessian_benchmark.md) |
| Effective rank strictly determines crossover, with universal $3.5$–$4.2\times$ gain | Not established; particular older real-data values require a separate provenance check | Historical draft and [current report Section 10](urop_research_report_20260916.md) |
| 49.1% headroom or 45.3% median-error reduction is a general MSE improvement | Metric/setup specific; not promoted to the central current conclusion | Historical draft; current report uses identified recovered MSE/conditional-risk tables |
| Eight illustrations mean the report/poster is publication-ready | Historical illustrations need source/claim validation; no new figure certification is claimed | [Current figure policy](urop_research_report_20260916.md) |

## Preservation and review scope

The historical Markdown body is retained unchanged beneath a superseded-status banner. Existing PDFs, LaTeX, HTML, source, tests, and result files are not regenerated in this documentation cycle. The new report is a consolidation of the reviewed local evidence, not a fresh audit of every historical notebook or external paper. Where a quantitative claim has not been traced sufficiently, it is labeled rather than promoted.

No scientific threshold is changed. No new empirical result, new theorem constant, or external-validation claim is created by this editorial work.
