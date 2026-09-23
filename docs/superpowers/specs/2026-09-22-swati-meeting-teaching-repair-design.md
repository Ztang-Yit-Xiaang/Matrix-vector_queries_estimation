# Swati meeting presentation: understanding-first repair

Date: 2026-09-22

Status: written design approved by the user's implementation request. Implemented on 2026-09-22 as a separate teaching revision; the original deck is preserved.

## 1. Purpose

The user loses the narrative after the current fourth slide and needs to understand the material well enough to explain it to Swati. Success is the presenter's ability to explain the research, not the number of results displayed. Rebuild the presentation as a guided mathematical explanation, using attractive explanatory visuals and substantial teaching notes.

Keep returning to one question: **Should the next matrix-vector query improve the captured subspace or estimate the remaining trace?**

The selected approach is example-led teaching. A results-first talk is shorter but caused the current comprehension gap. An exhaustive theorem-first talk preserves detail but would overload the main narrative. The selected approach introduces a result when its motivating question has become clear, with complete proof coverage in the technical appendix and notes.

## 2. Scope and preservation

- Produce a separate revised PPTX. Preserve `reports/swati_meeting_20260922/output/Swati_Research_Meeting.pptx` and its sources.
- Proposed revised output: `reports/swati_meeting_20260922/output/Swati_Research_Meeting_Teaching_Revision.pptx`.
- Plan 18 main slides, approximately 25–30 minutes, plus 12 technical backup slides. This replaces the original 12-main-slide teaching scope, not the original file.
- Use the current September 20 manuscript edition in `reports/latex/adaptive_hutchpp_20260919/` and validated historical data.
- Preserve every formal result through an explicit main-slide/appendix/notes coverage map. Do not imply all results are original contributions.
- No experiments, resampling, estimator edits, new scientific claims, manuscript modifications, or historical-result overwrites.
- No internal phase names in audience-facing content.
- The deck is a saved-file deliverable. Do not claim live PowerPoint editing or native PowerPoint inspection unless those actions actually succeed.

## 3. Learning structure

The initial four slides establish the problem, method, and budget before any research finding. The next slides explain the old allocation objective and its assumptions. Only then introduce capture failure and motivate Rademacher-specific risk. Finish with direct risk estimation, decision uncertainty, information cost, and empirical usefulness.

The historical distinction must be explicit: Rademacher randomness already appeared in the implementation. The later contribution is a shift toward exact realized Rademacher-risk analysis, not the invention or first adoption of a Rademacher estimator.

Do not group all mathematics before Rademacher or describe everything afterward as numerical verification. Later results include sample-variance identities, paired differences, conditional safety, paid-fallback accounting, and route-specific confidence limitations.

## 4. Main-slide sequence

| No. | Subject | Audience learning goal | Main visual |
|---|---|---|---|
| 1 | The allocation question | Understand the decision sought from the research | Two competing uses of the next query |
| 2 | Estimating a trace through matrix-vector products | Know the target and what information a query supplies | Matrix diagonal and input/output operator illustration |
| 3 | How Hutch++ splits the work | Understand captured trace plus estimated residual trace | Captured subspace and remaining component, with a short unbiasedness explanation |
| 4 | The query budget | Distinguish attempted sketch count q, accepted rank r, and fresh residual count ell | One budget bar with q+r+ell=m; no advanced result yet |
| 5 | Our original spectral-tail model | Understand T(q) and ideal leading-eigenspace capture | Ordered eigenvalue bars with the remaining squared-energy tail marked |
| 6 | Choosing the original allocation | Understand why minimizing the tail alone is wrong | Tail, residual-probe count, and ideal risk schematic aligned on q |
| 7 | The value of another ideal direction | Understand the local gain-versus-cost comparison | One eigenvalue removed from the tail versus two residual queries lost |
| 8 | What a randomized basis can miss | Separate dimension from capture quality | Ideal and tilted subspaces; square signal sketch with poor conditioning |
| 9 | What one extra column changed in the experiment | Interpret the finite extra-column result and rare-path contribution | Source-backed knee comparison, plus a simple nested-column illustration |
| 10 | Why the probe distribution matters | See why Gaussian energy is not the exact Rademacher risk | Diagonal residual example: random signs give the same quadratic form every time |
| 11 | The exact Rademacher risk | Understand why off-diagonal residual entries produce variance | Residual matrix with diagonal and off-diagonal contributions distinguished |
| 12 | Measuring the risk of the actual basis | Understand sample variance as an unbiased variance estimate | Fresh quadratic-form observations and their spread |
| 13 | Comparing candidate and baseline | Understand shared probes and the paired difference | One probe stream branching to two cached bases, then a risk comparison |
| 14 | Estimation versus justified acceptance | Distinguish empirical decisions from a proved certificate | Overlapping versus separated risk intervals; label the illustration schematic |
| 15 | The cost of learning | Understand original-baseline versus paid-policy accounting | Before/after budget bars with construction and certification costs |
| 16 | What the numerical studies establish | Separate useful discrimination from its query-cost tradeoff | Source-backed discrimination chart and full pathwise paid-risk distribution, on one legible comparison slide |
| 17 | What the mathematics establishes and leaves open | Understand route-specific vacuity and conditional structural-prior results | A restrained dependency diagram with proved statements and open assumptions |
| 18 | Discussion with Swati | Summarize the claim the presenter can defend and ask one focused research question | Simple research-choice diagram: a useful, cost-aware confidence bound or a capture diagnostic |

If slide 16 cannot remain legible at presentation size, place the complete paid-risk distribution in its dedicated backup and show a directly labelled summary on the main slide. Do not omit the many-path harm qualification from the main narrative.

## 5. One running budget example

Use m=160 and q=r=15. The current residual count is 130. A successful extra direction changes q and r to 16 and leaves 128 residual probes.

For either exact risk law, write its relevant energy as E. For E>0 and a successful rank gain, the added direction improves risk exactly when its fractional energy reduction exceeds 2/130, approximately 1.54 percent. First explain the analogous ideal-tail tradeoff; revisit it after defining actual residual energy.

This is an algebraic teaching example, not a newly measured experimental result. Never state that an attempted query necessarily buys a direction. The failed-rank case belongs in the notes and appendix.

## 6. Mathematical guardrails

- Define risk in words as expected squared trace-estimation error, conditional on the already constructed basis when appropriate.
- Use S for the range sketch and g for final or certification probes. Distinguish the two sources of randomness.
- State q+r+ell=m before using m-2q. The latter requires r=q.
- The trace split is an identity of traces; do not illustrate it as a false two-term matrix decomposition that omits cross blocks.
- The original objective 2T(q)/(m-2q) is the ideal Gaussian/Frobenius model, exact under perfect leading-eigenspace capture. It is not generally exact Rademacher risk.
- Minimize over feasible integer q with positive denominator. A committed reusable pilot imposes q>=B.
- Conditional isotropy supplies unbiasedness. Conditional independence is also needed for the stated variance of an average.
- For H=RAR, Gaussian risk is 2||H||_F^2/ell and Rademacher risk is 2 sum_{i!=j} H_ij^2/ell. The same-basis ordering does not order risks across different bases.
- The diagonal example H=diag(2,1) has Rademacher quadratic form identically 3 and Gaussian variance 10. It illustrates probe-law dependence and is not a claim that every residual is diagonal.
- A full-rank square signal block can have a large inverse. For rectangular blocks, distinguish a contained graph and bounds from a whole-range graph equality. Do not treat rotated coordinate-Rademacher blocks as independent Gaussian entries.
- The observed +1 minimizer concerns the tested finite mean risks. It is not a universal oversampling theorem or a claim that every path improves.
- The exact missing-direction example is a separate positive-oversampling discrete failure and appears in the appendix after the simpler geometric mechanism.
- General realized Rademacher energy need not decrease under nested projection.
- Sample variance is unbiased for conditional variance, but an unbiased estimate is not a confidence certificate. Sample variance and the all-pairs expression are the same statistic.
- Pairing exploits covariance; it does not guarantee variance reduction for every pair. A shifted comparator is not independent.
- c_pre=max(q_x+r_x) applies only when nested shared-prefix construction actually reuses the required products. Otherwise count actual committed products.
- A valid accepted-action guarantee does not refund sunk costs or make abstention equivalent to the unstarted baseline.
- Confidence-bound vacuity is a limitation of the specified proof route. A lower bound on a probability upper bound is not a lower bound on actual failure probability.
- Structural-prior certificates require their stated valid prior. Finite radius does not imply useful acceptance.
- Separate theorem statements, finite empirical observations, and proposed future work throughout.

## 7. Visual system

Use a light background, generous whitespace, large labels, and the existing restrained blue/teal palette. Reserve amber for query cost. Use the same visual meaning for captured and residual components throughout.

At least 14 of the 18 main slides should contain a substantive explanatory visual. Each visual must explain a relationship, quantity, mechanism, or decision. Do not fill slides with icons, decorative cards, gradients, or dense bullet lists.

Required visual families:

1. Editable query-budget bars reused consistently at construction, marginal gain, and paid-certification stages.
2. Eigenvalue bars with the remaining tail explicitly identified.
3. Clearly labelled schematic tradeoff curves; do not present illustrative curves as measured evidence.
4. Geometric capture illustration showing that equal dimension need not mean equal alignment.
5. Diagonal/off-diagonal residual illustration for the Rademacher variance proof.
6. Editable shared-probe and decision workflows.
7. Quantitative charts derived directly from preserved data, with populations and uncertainty defined.

Use image generation only for conceptual artwork that benefits from illustration, with exact mathematics and labels checked and preferably overlaid as editable text. Never generate quantitative evidence as an image. Keep requested diagrams, data charts, and tables editable. Avoid large on-slide equations without prior visual explanation.

## 8. Speaker notes and teaching checkpoints

Every main slide contains these note sections, sized to the idea rather than a word quota:

- **Understand:** plain-language explanation, including why this slide follows.
- **Say:** natural first-person speaking script.
- **Reason:** assumptions and a short proof or derivation where relevant.
- **If asked:** a likely question from Swati and a scoped answer.
- **Next:** one transition to the next question.
- **Sources:** manuscript or frozen evidence pointers.

Explain each new symbol before using it. Keep notes sufficiently detailed for rehearsal without requiring memorization of formulas. At the end of slides 4, 7, 11, and 15 include a private presenter checkpoint: explain the current idea without reading the equation. Put answers in notes, not an audience quiz.

Each main slide must pass three questions: What problem does it address? Why is its claim true or supported? What does it change about the research?

## 9. Technical appendix and complete coverage

Plan 12 backup slides with fuller proof sketches in notes:

1. Exact accounting and conditional unbiasedness.
2. Exact Gaussian/Rademacher variance derivations.
3. Successful/failed marginal identities and ideal spectral corollary.
4. Structured step-matrix residual-energy decomposition.
5. Square graph lemma, principal angles, and rectangular qualifications.
6. Exact missing-direction witness and pilot indistinguishability.
7. Sample-variance moments and Hoeffding decomposition.
8. Common-probe risk differences and covariance.
9. Simultaneous safe acceptance and paid-fallback distinction.
10. Chebyshev and truncation–Bernstein route-specific no-go proofs.
11. Conditional norm-prior certificate, corrected fourth moment, and scale-estimation boundary.
12. Experimental populations, pairing evidence, complete paid-path distribution, and interpretation limits.

Before export, map all 17 manuscript formal results and six empirical findings to their retained locations. Appendix slides may summarize statements visually while notes carry the derivations. Do not remove hypotheses to reduce slide density.

## 10. Verification and handoff

1. Preserve checksums of the existing deck, manuscript sources, and quantitative source files used.
2. Audit the teaching order and notation before producing artwork.
3. Check every derivation against the authoritative manuscript and every numerical claim against its preserved source.
4. Render every slide at presentation size. Inspect chart labels, equations, image crops, and readability.
5. Check speaker notes, transitions, proof coverage, and absence of internal phase labels.
6. Check diagram logic independently of visual appeal. Attractive artwork must not change the mathematics.
7. Export the separate revised PPTX and verify the preservation checksums.
8. Update the tracker, project memory, current-state note, and Obsidian research records with actual completion status.

## 11. Planning checklist and review gate

- Context inspected: existing plan, project tracker, memory, current-state note, and prior manuscript audit.
- User need clarified: comprehension deteriorates after slide 4; prioritize explainability and good visuals.
- Approach selected: example-led teaching rather than results-first or an exhaustive theorem sequence.
- Conceptual design approved in chat.
- This written specification is the next review checkpoint.
- Deck implementation and image generation have not started for this revision.

Self-review: no unresolved placeholders or new scientific parameters. The expanded slide count and duration are explicit. Mathematical caveats, evidence provenance, output preservation, and notes requirements are specified. Written-spec approval is required before implementation.
