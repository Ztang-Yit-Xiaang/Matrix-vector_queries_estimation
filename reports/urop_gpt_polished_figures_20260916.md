# GPT-polished figure presentation edition

Updated: 2026-09-16. Created at the user's request with the built-in GPT image tool.

All three current figures now have a coordinated presentation edition: clearer titles, stronger visual hierarchy, more balanced spacing, and an explicit takeaway. Original data, estimators, thresholds, results, PNGs, and editable SVGs are unchanged. No PDF was produced.

**Scientific-use boundary:** these are generatively redrawn raster graphics. Visual review checks labels and interpretation but cannot certify every plotted coordinate or error-bar endpoint. Use the [validated source figures](urop_figure_guide_20260916.md) for quantitative submission, measurement, and exact reproduction. These polished versions are presentation companions, not new scientific evidence.

## 1. Full numerical rank can miss a signal direction

![GPT-polished capture-failure figure](../figures/urop_gpt_polished_20260916/figure1_capture_failure_gpt.png)

The key distinction is between eight accepted basis directions and only four independent signal directions. The visible sign matrix retains the opposite third/fourth rows. The subtitle correctly says five leading eigenvalues rather than calling the positive-tail matrix rank five.

Visual checks: ten indexed coordinate paths; high-risk trial 7; Standard versus gated marker distinction; reference/failure Ritz knees at five/four; all 40 matrix signs; opposite-row outline; budgets (8,8,44) and (20,20,20); conditional-risk wording. No universal failure rate is claimed.

## 2. Fresh probes reveal useful risk information

![GPT-polished empirical-risk figure](../figures/urop_gpt_polished_20260916/figure2_empirical_signal_gpt.png)

The two panels retain false-safe acceptance and catastrophic-path detection as different conditional metrics. The 0.5% and 75% lines are point-estimate gate thresholds, not confidence guarantees. The comparison remains sample variance versus two median-of-means constructions.

Visual checks: logarithmic percentage axis in the safety panel; linear percentage axis in detection; three estimator rows; threshold labels; conditional 95% bootstrap wording; 600 frozen paths and 200 certification repetitions per path; explicit “not a confidence certificate” warning. The complete gate and eligibility definitions remain in the validated figure guide. The precise numerical intervals remain in its source CSVs; the generated line lengths are not independently certified.

## 3. Lower mean risk does not mean every path benefits

![GPT-polished certification-cost figure](../figures/urop_gpt_polished_20260916/figure3_certification_cost_gpt.png)

The design highlights why a favorable aggregate mean-risk comparison can coexist with harm on 575/600 paths (95.83%). The left panel averages within-rank ratios of mean risks; the right shows individual pathwise ratios. These are different estimands.

Visual checks: displayed ratios 1.1722, 0.0354, and 0.0363; original unstarted adjacent baseline; charged costs; log axes; cumulative distribution rather than a density; 95.83% and 575/600 annotation; separate bootstrap-versus-no-confidence-band descriptions. The generated ECDF follows the qualitative shape of the original but is not an exact rendering of its 600 steps.

## Files and provenance

- [Final prompt set](../figures/urop_gpt_polished_20260916/prompts.md)
- [Checksums and generation manifest](../figures/urop_gpt_polished_20260916/manifest.json)
- [Original figure guide and full captions](urop_figure_guide_20260916.md)
- [Current research report](urop_research_report_20260916.md)

The built-in image-generation skill provided the editing workflow; the scientific-figure skill supplied the integrity and review checks. The user's explicit GPT-image request governs the rendering method. All generated images were saved into the project without replacing originals. Research logs and Obsidian records retain this distinction between presentation polish and validated quantitative output.
