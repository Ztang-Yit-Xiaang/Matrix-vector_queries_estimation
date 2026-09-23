# Manuscript logic and readability audit — 18 September 2026

Read-only review of the September 17 manuscript by the main agent and an independent mathematical subagent. This is not independent reproduction of the empirical datasets.

## Verdict

The core mathematics is coherent. One spectral statement needs an explicit scope qualification; several domains and references need repair. The principal readability problem is mixing a scientific argument with an internal research log.

## Mathematical and logical fixes

1. `sections/allocation.tex:73`: explicitly restrict the ideal-tail corollary to the Gaussian/Frobenius spectral surrogate. Exact leading-eigenvector capture does not imply Rademacher residual energy equals T(r). For A=diag(2,1), Q=e1, the tail is 1 while off-diagonal residual energy is 0.
2. `allocation.tex:3–15`: require positive denominators, and a common feasible grid for model comparisons.
3. `capture.tex:59`: replace the nonexistent “Statement A” with the structured-energy lemma. At line 91, reference the graph lemma (`thm:T5b`), not the energy lemma.
4. `certification.tex:29`: sample variance requires integer s>=2, not even s. Evenness applies only to disjoint-pair constructions.
5. `confidence.tex:143`: state conditioning on pre-certification information, integer n>=1, and a finite nonnegative norm bound explicitly.
6. Explain why the specified 81-based confidence constructions are considered despite the improved quadratic-chaos constant 15. The latter still gives a relative radius approximately 4.192928 in the same joint Chebyshev construction at s=32 and joint failure 0.05. This is a derived comparison, not a new empirical result or universal impossibility theorem.

## Editorial fixes

- Replace phase codes with scientific study names in text, captions, tables, and figure labels.
- Move development chronology, test/file counts, assembly history, and detailed recovery records out of the compiled paper. Preserve these in companion records; keep material data limitations and a concise reproducibility statement.
- Remove sentences announcing that a qualification is important, an analysis is analytic, or an old figure is retained. State the assumption or result directly.
- Preserve joint-error versus acceptance-conditional safety, paid fallback versus original-baseline safety, and other claim-specific qualifications. Consolidate repeated general disclaimers.
- State the empirical rule and relevant thresholds, or report operating rates without internal GO/NO-GO labels.
- Define the exploratory gated estimator before comparing it. Read its actual implementation rather than infer its rule from figures.
- Replace Figure 2 with common inputs branching into spectral and realized-basis models. Gaussian and Rademacher risks are parallel outputs. Use short labels, exact equations, aligned boxes, thin rules, and no icons, gradients, oversized title, or slogan panel.

## Checks and status

The independent auditor checked conditional risk, successful/failed marginal identities, structured energy, square-sketch capture, pilot transcript construction, sample-variance moments, Hoeffding decomposition, paired covariance, paid fallback, truncation floor, structural-prior radius, and the strict scale boundary. Exact-sign enumeration on 100 small symmetric zero-diagonal matrices agreed with the corrected fourth-moment identity to maximum relative discrepancy 1.67e-15; this supplements rather than proves the identity.

The conceptual layout is approved. Its exact written scope is `docs/superpowers/specs/2026-09-18-journal-manuscript-revision-design.md`, committed alone as `7bbde0b`. The design workflow requires written-scope review before editing. Manuscript sources, figures, ZIP, estimator code, and empirical artifacts are unchanged. Preserve the September 17 edition and publish a separate revision.
