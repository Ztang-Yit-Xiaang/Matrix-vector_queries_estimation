# Theorem–proof report: editorial and mathematical audit

Date: 2026-09-17. Scope: exposition, structure diagrams, and small deterministic identity checks. No estimator, allocation rule, scientific threshold, or historical numerical artifact is changed.

## Deliverable

[The theorem–proof edition](urop_theorem_proof_report_20260917.md) expands the September 16 narrative into 15 report-local numbered results, a spectral corollary, six empirical findings, an assumption ledger, historical progression, and a source-migration table. It is approximately 7,700 whitespace-delimited words, versus approximately 3,800 in the prior consolidated narrative. The earlier report is preserved byte-for-byte. No new PDF is generated.

Three conceptual diagrams explain the budget workflow, proof dependencies, and four risk models. They were generated with the built-in GPT image tool. The original quantitative PNG/SVG/CSV figures remain authoritative and are embedded unchanged. The [prompt and structure contract](../figures/urop_structure_20260917/prompts_and_contract.md) records generation and corrective edits. Visual inspection rejected a first draft that portrayed confidence validity too automatically and fixed “every query” to “every sketch query” in the ideal model.

## Proof review

- Scoped every theorem to its assumptions, distinguishing real symmetry from PSD, and accepted rank from attempted width and dominant rank.
- Distinguished the pre-certification and pre-final information sets; retained conditional freshness.
- Restricted max-construction accounting to nested shared-prefix caching; otherwise required the actual ledger.
- Required positive denominators for success/failure comparisons, and handled zero residual risk without an artificial floor.
- Added an explicit example in which nested projection increases Rademacher off-diagonal energy.
- Derived the square-sketch principal-angle formula and qualified its rectangular extension; made no iid Gaussian assumption for the rotated Rademacher block.
- Kept the missing-direction transcript argument local, without claiming a population impossibility theorem.
- Expanded the exact sample-variance moment proof and Hoeffding decomposition; canonical-kernel concentration alone is insufficient.
- Distinguished joint error control from error conditional on acceptance, and accepted-action safety from complete-policy safety.
- Reverified the imported hypercontractivity statement and Cortinovis–Kressner Theorem 2, equation (8), against their primary sources.
- Explicitly identified the Phase 1D floor as a floor on a probability upper bound, not actual failure probability.
- Kept the structural norm prior as an assumption; handled its zero branch and possible failure allocation.
- Checked the corrected fourth-moment coefficients through even-edge counting. During drafting, corrected the auxiliary expression for sigma to the fourth power to 16a+32b before final validation.
- Retained the strict Chebyshev inequality: 41 is equality and 42 is first feasible at K=3, delta=0.05.
- Preserved the instance-specific +1 observation, conditional bootstrap interpretation, coordinate failure, exploratory Hessian limits, and recovery qualifications.

## Verification performed in this turn

The focused mathematical module contains 20 passing cases, including exhaustive tiny sign/sample enumerations, graph/energy identities, the exact transcript witness, risk algebra, probability-bound floors, and paid-fallback distinctions. The five existing validated-figure tests also pass: **25 tests total, 6.65 seconds** in the local recovery environment, using the existing readline workaround. These are regression checks, not substitutes for universal proofs.

The full maintained estimator suite was not rerun for this editorial-only change. The previously recorded 225-test result belongs to the coordinate audit and is not relabelled as a new run.

The preservation snapshot covers 275 pre-existing files in source, tests, results, proof documents, reports, and figures. The finalizer verifies every checksum, local report link, display-math delimiter pair, all 15 result headings, all six images, and image dimensions. Its generated manifest records output hashes. Research records are updated separately, so no frozen file needs modification to point to the new edition.

## Synchronization and next step

The tracker, workspace memory, current-state note, new report, audit, and conceptual figures are synchronized to the designated Obsidian directory. Existing replaced notes are archived recoverably; report-relative links are adjusted for the vault's flat note layout. A synchronization manifest records source and destination hashes.

Next: review the theorem–proof edition and adapt it to the eventual report/poster format. The revision does not authorize another allocator, another benchmark, or an unbudgeted confidence claim.
