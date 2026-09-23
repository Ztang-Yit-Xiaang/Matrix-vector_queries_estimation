# Journal-style manuscript revision

User-approved direction: revise the September 17 LaTeX manuscript, remove internal phase names, replace the slide-like risk-model graphic, and use an independent mathematical/readability audit. This is editorial and proof-clarification work, not an estimator experiment.

## Revision contract

- Preserve the September 17 source package and ZIP. Publish a separate September 18 revision.
- Keep every substantive result, proof, empirical quantity, and material limitation. Remove repeated framing, slogans, and internal implementation commentary.
- Use descriptive experiment names rather than internal phase codes throughout the compiled paper, including captions and figure text.
- Move development chronology, file counts, test logs, production-recovery details, and assembly history into companion provenance notes. Retain a concise reproducibility statement and any material data limitation in the paper.
- Fix ambiguous assumptions and cross-references identified by the independent audit. In particular, the ideal spectral-tail marginal concerns the Gaussian/Frobenius surrogate, not general realized Rademacher risk.
- Retain the three quantitative figures' numerical geometry and source data. Remove internal phase labels from their displayed text if present without changing measurements.

## Replacement Figure 2

Use a compact journal workflow with one shared input (matrix, allocation, and accepted basis). Branch into an ideal spectral model and a realized-basis model. The ideal branch distinguishes full-rank and accepted-rank denominators. The realized branch forms the residual operator and splits into Gaussian and Rademacher risk as parallel outputs. Arrows denote construction/definition, not inequalities or a chronological sequence of estimators.

Use white background, thin rules, restrained color, uniform typography, aligned boxes, short labels, and exact equations. No icons, gradients, oversized heading, or slogan box. Prefer editable vector/LaTeX geometry for mathematical text. A compact formula table is the rejected alternative because the user approved a workflow chart; merely recoloring the old slide would not address its structure.

## September 19 scope extension: illustrated Figure 1

The user also requests an illustrated Figure 1 informed by established papers. The inspected sources, three-panel construction/trace-split/query-ledger design, mathematical safeguards, and alternatives are recorded in `docs/figure_reference_review_20260919.md`. Add spectrum and residual-matrix illustrations to the approved Figure 2 branches. The user approved this layout on September 19 and added a beginner-reader test: the picture should be interesting and understandable to a student outside the field. Publish a separate September 19 revision and preserve the September 17 package.

## Implementation checklist

1. Copy the original source into a separate September 19 package and checksum the original files.
2. Use the existing Python/matplotlib figure workflow for original schematic-led composites; export editable SVG and high-resolution PNG. No image-generated equations or invented numerical measurements.
3. Integrate the two figures with concise captions. Keep the three data figures unchanged. Move the old proof-map artwork and project chronology into companion provenance, retaining scientific assumptions in the paper.
4. Apply the independent audit's targeted mathematical qualifications and readability fixes, preserve every formal result/proof and all six empirical findings, and remove internal phase labels from compiled text.
5. Independently review the diagrams, run focused mathematical/figure tests, compile the source in draft mode, inspect the rendered figure previews, check references and overflow, and package the source ZIP. Do not deliver a new manuscript PDF.
6. Record exact changes and limitations, verify original hashes, and update/synchronize the research notes.

## Verification and delivery

Independent subagent performs read-only logic and readability review. Main agent reconciles findings before edits. Compile with draft-mode LaTeX/BibTeX, verify references and overflow, run focused mathematical checks, inspect the replacement chart, and check for internal phase names in compiled sources and figure labels. Preserve original artifacts by hashes. Deliver revised LaTeX source/ZIP; no new standalone paper PDF, experiments, thresholds, allocator changes, or matrix-vector queries. Update tracker, memory, current state, and recoverably synchronize to Obsidian.

Self-review: no scientific thresholds or unresolved design choices; no equivalence between full rank and capture; no implication that Gaussian and Rademacher models form a monotone risk chain. Detailed proof preservation takes precedence over an arbitrary length target.
