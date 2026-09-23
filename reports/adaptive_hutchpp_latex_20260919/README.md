# Adaptive Hutch++ — illustrated journal revision

Started September 19; completed September 20, 2026. Open **main.tex**, or upload the accompanying ZIP to Overleaf and select main.tex. No manuscript PDF is delivered.

The revision replaces the two slide-like conceptual figures with original mathematical illustrations: sketch/basis construction, projected residual probes, query budgets, and branching ideal/realized risk models. Editable SVGs accompany the embedded high-resolution PNGs. Three measured plots retain their numerical content; two have phase codes removed from their subtitles. The old proof-map illustration and development chronology are preserved outside the compiled paper under provenance/previous_edition/.

All 17 formal statements, 17 proofs, and six empirical findings remain. Internal phase names and repeated implementation commentary are removed from the compiled paper. Necessary boundary cases and empirical limitations remain. Targeted corrections restrict the spectral-tail corollary to the Gaussian/Frobenius model, repair lemma references and feasible domains, permit odd sample sizes for ordinary sample variance, and specify conditioning for the structural-prior theorem. The exploratory pilot rule is now defined explicitly.

## Reader and notation contract

The first figure should explain the method to a student outside the field before its equations are read. Use S for the sketch, q for attempted width, r for accepted rank, k for the dominant spectral rank, and g for a fresh probe. A trace decomposition is not a two-term matrix decomposition. Risk-mask pictures describe variance contributions, not removal of diagonal trace information. All drawn geometry and budget widths are schematic.

## Build and verification

Compile with pdfLaTeX, BibTeX, and two resolving pdfLaTeX passes. This delivery uses draft-mode compilation, which checks typesetting and references without creating a manuscript PDF. Figure PNGs are visually reviewed separately; full rendered manuscript pages are not reviewed. See validation_manifest.json for executed checks and limitations.

SOURCE_MAP.md retains the source evidence mapping. Detailed development records, including historical phase names, are deliberately kept in provenance/ rather than the paper. The September 17 source package and ZIP are unchanged. No new estimator, empirical run, oracle query, bootstrap, or threshold selection is part of this revision.

Confirm author/affiliation metadata and submission requirements before external circulation. The byline remains neutral. Scientific and mathematical review by the author remains necessary; automated and independent informal audits are not formal verification.
