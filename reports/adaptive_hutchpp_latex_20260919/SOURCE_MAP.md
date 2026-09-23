# Source and result map

## Writing reference

[Hutch++: Optimal Stochastic Trace Estimation](https://arxiv.org/abs/2010.09649), version 5, supplies the organizational model, not our claims or language. Its introduction/contribution framing, preliminaries, theory sections, experiments, and appendices motivate the structure. We omit an optimal-complexity/lower-bound claim because our project has not established one.

## Preserved theorem–proof content

The original is archived at `provenance/source_report.md`; its original assembly hash appears in `provenance/previous_edition/assembly_manifest.json`. This revision preserves all 17 proof bodies byte-for-byte while clarifying statement hypotheses, cross-references, and surrounding exposition.

| Original label | Manuscript source | Preserved mathematical content |
| --- | --- | --- |
| T1 | `sections/preliminaries.tex` | Rank-aware query ledger and committed cost |
| T2 | `sections/preliminaries.tex` | Conditional unbiasedness with fresh probes |
| T3 | `sections/preliminaries.tex` | Gaussian/Rademacher conditional risk |
| T4 | `sections/allocation.tex` | Successful and failed marginal transitions; zero-risk case |
| T4.1 | `sections/allocation.tex` | Ideal spectral criterion and sharp-step boundary |
| T5 | `sections/capture.tex` | Structured residual energy and square-sketch graph, split into two lemmas |
| T6 | `sections/capture.tex` | Missed direction and identical pilot transcript |
| T7 | `sections/certification.tex` | Sample-variance moments and full Hoeffding decomposition |
| T8 | `sections/certification.tex` | Unbiased paired difference and covariance cancellation |
| T9 | `sections/certification.tex` | Simultaneous safe acceptance; distinction from conditional-on-acceptance safety |
| T10 | `sections/certification.tex` | Paid fallback and original-baseline cost |
| T11 | `sections/confidence.tex` | Joint 81-plus-Chebyshev budget vacuity |
| T12 | `sections/confidence.tex` | Route-specific truncation–Bernstein lower-tail no-go |
| T13 | `sections/confidence.tex` | Conditional structural-prior paired radius |
| T14 | `sections/technical.tex` | Corrected fourth-moment identity and bound |
| T15 | `sections/technical.tex` | Strict scale-estimation boundary: 42 pairs / 84 probes |

The original T labels are retained as stable internal LaTeX keys, while the visible result numbers follow section numbering. Nothing renumbers the older independent proof notes.

## Empirical evidence

All six findings remain in `sections/experiments.tex`.

| Finding | Content and evidence to review |
| --- | --- |
| E1 | Frozen rank-deficient bridge and zero-oversampling mechanism; finite empirical mean minimizers, not universal optima |
| E2 | Phase 1A's 960,000-row external-certification experiment, empirical gate, eligible-path bootstrap |
| E3 | Phase 1B paid-policy mean/pathwise tradeoff; correct pooled-median distinction |
| E4 | Phase 2A covariance cancellation; the associated Phase 2B theory is moved to the confidence section |
| E5 | Recovered orientation comparison and exact coordinate missed-direction witness |
| E6 | Exploratory synthetic-data Hessian study; historical real-data claims remain qualified |

Exact repository paths for each copied report and proof note appear under `provenance_mapping` in `provenance/previous_edition/assembly_manifest.json`. The archived Markdown source retains its claim-local links, making the generic local research-record bibliography entry auditable rather than a substitute for source evidence. Internal historical study codes occur in these companion records, not the compiled paper.

## Figures

| File in `figures/` | Role | Authoritative source |
| --- | --- | --- |
| `workflow.png` and `.svg` | Illustrated construction, projection, and accounting | `figures/urop_journal_20260919/` in repository |
| `models.png` and `.svg` | Branched risk models and variance masks | Same original schematic package |
| `figure1_capture_failure.png` | Quantitative coordinate failure | `figures/urop_validated_20260916/` |
| `figure2_empirical_signal.png` | Quantitative Phase 1A rates | Same validated package |
| `figure3_certification_cost.png` | Quantitative paid-policy tradeoff | Same validated package |

The capture plot is byte-identical to the validated original. The other two numerical plots remove internal study labels only; all eight exported source CSVs are byte-identical to their validated originals. The old proof-map artwork remains under `provenance/previous_edition/`, outside the compiled paper. The earlier GPT-polished numerical companions are not used as data figures.

## Bibliographic audit

- Meyer et al.: arXiv:2010.09649v5, the writing reference and Hutch++ background.
- Halko, Martinsson, and Tropp: SIAM Review 53(2), 217–288 (2011), DOI 10.1137/090771806; randomized range-finding context.
- Cortinovis and Kressner: Foundations of Computational Mathematics 22, 875–903 (2022), DOI 10.1007/s10208-021-09525-9; Theorem 2, equation (8), with its symmetry/nonzero/zero-diagonal assumptions.
- O'Donnell: CMU Boolean analysis lecture 16, Corollary 1.3; degree-dependent hypercontractivity. No unverified publication date is supplied.
- Local project records: unpublished reports and proof notes bundled for provenance, not externally peer-reviewed sources.

## Changes intentionally not made

No estimator change, new experiment, threshold tuning, new mathematical claim, new bootstrap interval, or new PDF. The expanded proof information is retained. Only manuscript structure, notation typesetting, theorem environments, cross-references, bibliographic presentation, and accompanying explanatory prose were revised.
