# Illustrated journal revision audit

Completed 2026-09-20; edition directory retains its September 19 start date.

## Outcome

Created a separate illustrated manuscript in `latex/adaptive_hutchpp_20260919/`. The September 17 source edition remains unchanged. Figure 1 now uses matrix silhouettes, projection geometry, and query-budget strips; Figure 2 branches from shared inputs into spectral surrogates and actual-basis risks. The original schematics are editable SVGs and high-resolution PNGs, not generated bitmap text.

The scientific-figure workflow supplied the panel contract and legibility checks; the proof-audit workflow protected trace/risk/accounting distinctions; the academic-polishing workflow separated the scientific argument from project chronology. The prose is an English research manuscript with a general mathematical-journal audience. Its notation ledger keeps S, q, r, k, and fresh g distinct. No submission-specific journal formatting claim is made.

## Preserved and clarified

- All 17 result/proof pairs remain; every proof body matches the previous edition byte-for-byte. All six empirical findings remain.
- Qualified the ideal spectral marginal as Gaussian/Frobenius-only; a diagonal counterexample explains why it is not generally a Rademacher identity.
- Added common feasible-grid domains and explicit structural-prior conditioning, repaired graph/energy lemma references, and removed an unnecessary even-s restriction.
- Stated the exploratory pilot rule and its distinct orientation/Hessian thresholds from the actual implementation. No estimator code was edited.
- Explained that factor 15 also leaves the specified joint Chebyshev radius above one at s=32, without broadening the 81-based truncation no-go.
- Removed internal phase codes from the compiled manuscript and two numerical-plot subtitles. Moved chronology and the older proof-map illustration into companion provenance. Historical metadata is preserved there, not erased.
- Replaced repetitive captions with self-contained definitions of plotted quantities and uncertainty. Internal gate-verdict labels were replaced with measured operating rates; complete historical gate details remain in provenance.

## Verification

- 30 focused tests passed: 20 existing mathematical regressions, five existing quantitative-figure tests, and five new manuscript/schematic regressions.
- All 48 files listed in the previous edition's validation manifest retain their original hashes.
- All eight regenerated numerical source CSVs exactly match the validated original CSVs. The numerical builder reads the same seven protected inputs and checks their hashes before and after. No new queries, resampling, or parameter choices occurred.
- Draft-mode pdfLaTeX, BibTeX, and resolving passes completed with no undefined references/citations or overfull boxes. No manuscript PDF was produced. This is not a rendered full-page manuscript review.
- Both actual figure PNGs were inspected by the main agent and an independent reviewer. The reviewer requested input/operator labels, explicit ledger equations, variance-mask labels, and a spectral cutoff; all were implemented.
- Independent targeted source/figure review found no blocking issue. This informal review is not formal proof verification or empirical reproduction.

## Remaining review

The user should assess whether the illustrations are engaging and understandable. Author/affiliation metadata and a venue-specific rendered layout still need confirmation before submission. Original figures and mathematical illustrations are clearly distinguished from measured plots. No online allocator work is authorized or needed for this editorial cycle.
