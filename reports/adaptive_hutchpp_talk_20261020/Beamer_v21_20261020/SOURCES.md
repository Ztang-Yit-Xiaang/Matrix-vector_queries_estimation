# Sources and scientific scope

## Classical background

- M. F. Hutchinson, *A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines*, Communications in Statistics—Simulation and Computation 19(2), 433–450 (1990). [DOI](https://doi.org/10.1080/03610919008812866). Bibliography and technical formulation follow the primary Hutch++ paper.
- R. A. Meyer, C. Musco, C. Musco and D. P. Woodruff, *Hutch++: Optimal Stochastic Trace Estimation*, SOSA 2021. [Primary paper](https://arxiv.org/abs/2010.09649).
- D. Persson, A. Cortinovis and D. Kressner, *Improved Variants of the Hutch++ Algorithm for Trace Estimation*, SIAM Journal on Matrix Analysis and Applications 43(3), 1162–1185 (2022). [Primary paper](https://arxiv.org/abs/2109.10659).

## Earlier reliability analysis: slides 4–6

Source project: `Hutch++/Matrix-vector_queries_estimation/reports/latex/adaptive_hutchpp_20261019_integrated/`, sections `certification.tex` and `confidence.tex`.

- The all-pairs sample-variance risk difference is unbiased for the difference of the two fixed conditional final risks. Both actions and their positive denominators are fixed before independent checking data; final probes are fresh.
- Sharing checking probes reduces difference variance relative to equal-marginal independent estimates when the risk estimates have positive covariance. Sharing alone does not guarantee that sign.
- Slide 6 reconstructs Proposition T11's conservative two-action relative radius: `sqrt(2*(14+2/(s-1))/(s*0.05))`, for integer `s=2,...,32`. It uses the degree-two kurtosis envelope `K<=15`. The radius at 32 is `4.192927993811761`; every displayed radius exceeds one, making this particular variance-upper-bound inversion inconclusive.
- This is a limitation of that specified bound, not an impossibility result for all statistical methods. The separate finite paired bound requires frozen nested actions and valid pre-checking residual-norm information, among other assumptions. It controls acceptance and harm jointly, not harm conditional on acceptance or whole-policy safety.
- These analytical/statistical foundations are discussed as the project's framework and findings; classical identities and prior methods are not claimed as individually new inventions.

## Paid-policy experiment: slide 7

Source: integrated report `sections/experiments.tex`, E3; `figure_sources/figure3_path_source.csv`.

- All 600 recorded rows: budget 160, checking count 16, spectral floor `1e-6`, primary adjacent choices, ordinary sample-variance heuristic with epsilon `1/3` (factor-two acceptance guard).
- The ECDF uses each recorded `pathwise_net_ratio`, checked against `risk_paid_selected_mean / risk_original_baseline`. Exactly 575 ratios exceed one: 95.83% of recorded paths have higher average paid-policy risk.
- Each numerator averages the selected action's exact conditional risk over 200 checking repetitions, including all charged construction/checking costs. The denominator is the same path's original, unstarted baseline risk. These are not 600 realized final trace errors.
- The experiment does not evaluate the conservative confidence rule from slide 6. No safety guarantee is attached to the heuristic. Favorable aggregate mean-risk summaries can coexist with widespread pathwise harm because they weight the risk distribution differently.
- `figures/paid_risk_ecdf.csv` contains the 600 sorted ratios and their cumulative percentages.

## Allocation evidence: slide 8 (v21)

### Retained fixed-policy aggregate

The title, three bullets, takeaway and footnote remain exactly as in v20. The matched held-out summary is186 matrices,16 groups,744 budget cases; the selected24% rule reduces mean Frobenius-normalized MSE by19.79274653544545%, while the predefined15% rule reduces it by22.070413570394386%. These aggregate results come from V1 `policy_heldout_summary.csv`; they differ from the unweighted distribution of casewise gains reported below. The tested-floor near-optimal cap claim remains restricted to the V3 measured0.005/0.02 settings.

### Median selection for the new right panel

Sources: `q_finding/results_20261019_v1/main/risk_curves_by_q.csv`, `allocation_policy_comparison.csv`, `policy_heldout_summary.csv` and `manifest.json`.

1. Manifest test IDs and the selected-policy rows identify the same186 held-out matrices and744 matrix-budget cases. All saved per-q curves are present.
2. At m96, exclude both `power` and `signed_power` to match "outside power laws." This leaves144 cases:84step,28exponential,14signed-step,6hub,6ER,3path and3flat.
3. Compute gain=1−MSE(q23)/MSE(q32) from recorded means. Sort ascending by gain, then matrix ID for a deterministic tie-break.
4. The even sample's arithmetic median is17.95861937931968%, which is not any actual case. The user explicitly chose the lower-middle convention. Rank72/144 selects `11cfff958145affe`, gain17.757459438513912%. The upper middle is18.159779320125446% and is not selected.

Selected metadata: step spectrum, d512, head rank8, tail0.1, Haar orientation (rep1), matrix seed11825904100787601510, group `step:8:0.1`,64 saved range sketches. At m96, q23 gives MSE0.026855310105186145 and q32 gives0.0326537944010966. The full48-point curve spans q0–47. Its observed optimum is q23, coinciding with the blue rule dot, so no separate optimum marker is needed. The grey square denotes q32. The plotted reduction is rounded to17.8%.

### Retained left panel and chart interpretation

Source: Phase4 `results/spectral_floor_mechanism_20261019_v4/figures/source_data/figure2_actual_and_fixed_basis_risk_curves.csv`. The retained floor0.005 curve uses d512,a1.8,m96,double-Walsh orientation,256 sketches and48 recorded widths. Its observed optimum q22 lies inside the quarter cap q24 with zero estimated excess. All48 values are unchanged. The new right panel is a fixed-width comparison, not an offline cap test; the shared subtitle was updated accordingly. Independent logarithmic vertical ranges show the full curves without clipping. Heights across panels should not be compared as normalized risks.

### Descriptive distribution, reported off-slide only

All744 held-out matrix-budget cases include all families at budgets60/96/160/256. At each budget compare q=floor(0.24m+0.5) with q=floor(m/3); these are14vs20,23vs32,38vs53,61vs85. Each curve-derived MSE is checked against its per-case policy row. All standard denominators are positive.

- Positive gain:539/744=72.44623655913979%.
- Standard better:205/744=27.553763440860213%. No ties.
- Median gain26.73711084733321%;10th percentile−77.94605091015714%;90th percentile50.14557958742955%. Percentiles use linear interpolation; cases are unweighted and repeated budgets of the same matrix are not independent new matrices.
- Standard-better cases by family: exponential94/112, step78/336, signed-step30/56, hub3/24. No standard-better cases in the other held-out family labels.

The data directory includes all744 gains, all144 eligible cases in selection order, the selected configuration and source hashes. The script only reads existing CSVs and replots recorded means. No experiment, report or primary data file was modified or rerun. These finite-sample descriptive results are not population guarantees.
