# Source-backed UROP figure contract — September 16, 2026

## Scope and export contract

Continue the current Markdown report with three static scientific figures. Use
the existing Python/matplotlib workflow. No new experiment, bootstrap, fitted
threshold, estimator modification, or synthetic observations. Export editable
SVG and 300-dpi PNG, approximately 183 mm wide, with readable 8-point body text.
Do not generate another PDF. Historical figures and experimental results remain
unchanged. Output goes into a new dedicated figure directory; preview in a
temporary directory first. Source-data CSVs and a provenance manifest accompany
the images. User-research figures have no vendor branding.

Palette: blue and orange plus neutral grey/black/white. Markers, fill, and line
styles carry distinctions in addition to color. Neutral descriptive titles;
supported conclusions belong in captions. Logarithmic axes are explicitly
marked, never silently truncated; point/interval plots replace logarithmic bars.

## Figure 1 — Coordinate pilot capture and residual risk

Conclusion: full accepted rank and a sharp Ritz knee can coexist with a missed
dominant direction and large conditional Rademacher risk.

Archetype: quantitative/mechanism composite, with the all-path risk plot as the
main panel. Panels:

1. Exact conditional Rademacher risk for all ten coordinate paths, Standard and
   gated methods. Logarithmic risk axis. The same trial index does not imply
   common random sketches/probes between methods. No confidence bars: each point
   is a computed conditional risk, not a sample-variance estimate.
2. Actual pilot Ritz values on the worst gated path and on the median-risk path
   among the other nine gated paths. This illustrative selection is explicit;
   panel 1 retains every path. Show the tail level and rank ordering.
3. The exact integer signal sketch on the worst path. Display all 40 signs and
   highlight the opposite rows; distinguish signal rank four from accepted rank
   eight. The witness is exact rational arithmetic; plotted QR diagnostics are
   floating-point measurements.

Sources: `results/coordinate_gate_failure_audit_20260916/paths.csv` and
`exact_nullspace_witnesses.json`. Record selected trial identifiers and data
grains. The finite ten-path population cannot support a universal failure rate.

## Figure 2 — Empirical selective-risk metrics

Conclusion: at the frozen primary setting, sample variance has a substantially
lower false-safe rate than the two small-block MoM plug-in rules, while the
observed catastrophic-detection rates are broadly similar.

Archetype: two-panel quantitative comparison, horizontal point/interval plots.
Use all three predeclared primary estimator families at s=16, not an invented
fourth category. Panel 1: false-safe rate among truly worse paths (log percentage
axis). Panel 2: acceptance among top-5% catastrophic truly better paths (linear
percentage axis). Retain the original 95% percentile intervals from 10,000
eligible-path, rank-stratified cluster-bootstrap replicates. Show the original
point thresholds 0.5% and 75%; label these as selected gate criteria, not a complete
gate or a theorem-level confidence certificate.

Primary setting: d=500, m=160, eta=1e-6, epsilon=1/3, q=r_star+1 versus r_star,
600 frozen paths (200 per rank) and 200 certification repetitions per path.
Truth-eligible path counts must accompany the source data/caption. Repetitions
are not independent research paths. Cross-estimator metrics reuse common probes.

Sources: Phase 1A bootstrap, operating-rate, and catastrophic CSVs. Do not compute
new uncertainty or imply that a small false-safe rate proves conditional safety.

## Figure 3 — Certification cost and pathwise net effects

Conclusion: the frozen paid selector improves the equal-rank mean of within-rank
mean-risk ratios while harming most individual paths. Those are different
estimands, not contradictory measurements.

Archetype: two-panel comparison/distribution. Panel 1: paid fallback, paid oracle,
and empirical selection risk ratios relative to the unstarted adjacent baseline,
with the original shared-bootstrap intervals and a no-change reference at one.
Use a log point/interval plot, not bars. Panel 2: ECDF of all 600 pathwise expected
selected/original risk ratios at the same primary setting. Paths each average
200 certification repetitions. The path population has 200 paths per rank, so
its unweighted ECDF also weights ranks equally. Mark ratio one and report the
fraction harmed; do not confuse this distribution with the ratio-of-means
summary in panel 1.

Sources: Phase 1B bootstrap CSV and filtered paths Parquet; s=16, sample variance,
eta=1e-6, m=160, epsilon=1/3, primary adjacent pair. The baseline is not generally
Standard Hutch++ at floor(m/3). The oracle is offline diagnostic truth, not an
implementable certified policy. Common-probe and frozen-orientation limitations
remain. No confidence band for the ECDF is invented.

## Validation and integration

Before exporting: check source checksums, grid selection, uniqueness, positive
log-axis inputs, exact witness signs/rank metadata, CI ordering, eligible counts,
and reproduction of reported summary ratios directly from the 600 path rows.
Export the actual plotted data with contextual columns. Inspect each PNG at
report width; check SVG editable text, labels, boundaries, and panel readability.
Caption every panel with its metric, denominator, n, and uncertainty scope.
Embed the verified PNGs into the current report at their corresponding sections.
Keep complete captions and source-data links in a figure guide suitable for
report/poster reuse. Record verification results without claiming a fresh full
algorithm-suite run for a plotting-only change.
