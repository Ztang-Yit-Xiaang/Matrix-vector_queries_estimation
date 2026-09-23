# Conceptual figure specification and prompts — 2026-09-17

Built-in GPT image generation; conceptual diagrams only. They do not replace data-rendered plots or proofs. Final versions were visually checked for labels and arrow meaning. The proof diagram is a reading map, not a complete list of assumptions: orthonormality, conditional isotropy, iid probes, positive denominators and distribution-specific fourth moments are stated in the report.

## workflow

Create a polished academic vector-style infographic, wide landscape 16:9, white background, navy text, teal for computation, gold for information, muted red for sunk cost. No plots or invented numbers. Title 'Two workflows, two budget ledgers'. Exactly two horizontal lanes. Upper lane heading 'Frozen trace estimator'. Boxes left to right: 'Range sketch / q queries' arrow 'Accepted basis / rank r' arrow 'Cache AQ / r queries' arrow 'Fresh residual probes / ell queries' arrow 'Trace estimate'. Below upper lane show 'q + r + ell = m'. Tiny note: 'Pilot queries are reused inside q'. Lower lane heading 'Candidate-first certification'. Boxes left to right: 'Construct candidates / actual cost c_pre' arrow 'Fresh certification / s queries' arrow 'Choose or abstain' arrow 'Fresh final probes / m - c_pre - s queries'. Under it, a red label 'Abstention does not refund spent queries'. Footer 'Shared-prefix construction can reuse work; otherwise count every committed query.' Use very legible large typography and generous margins. This is a conceptual workflow, not empirical evidence. No extraneous decorative images.


## dependencies

Create a polished academic conceptual diagram, wide landscape16:9, whitebackground, navytext, tealproof boxes, amberassumption boxes, redlimitationfooter. Title 'What proves what?'. Clear top-to-bottom DAG in two separated lanes, with no crossing arrows. Left lane label 'Estimator correctness': box 'Action fixed before final probes' arrow box 'Fresh independent residual probes' arrow box 'Conditional unbiasedness' arrow box 'Exact conditional risk' arrow box 'Realized marginal decision'. Right lane label 'Certification': box 'Fresh certification probes' arrow box 'Risk estimates' arrow box 'Valid simultaneous confidence bounds' arrow box 'Accepted-action safety'. Connect the 'Exact conditional risk' box to 'Risk estimates' with a single horizontal dotted arrow labelled 'target'. Beneath right lane, a clearly separated red outlined warning 'Accepted-action safety is not whole-policy safety'. Bottom footnote 'Construction and certification costs remain after abstention.' Distinguish 'Risk estimates' from 'Valid simultaneous confidence bounds': estimates alone are not enough. All text verbatim, precise, readable. No formulas, no invented empirical numbers, no icons necessary.

### Corrective edit

Edit this diagram retaining its white/navy/teal visual style and wide layout. Remove ALL amber assumption boxes entirely: they are scientifically misleading. Make exactly two rows of teal boxes, no other boxes except red warning. Top row: 'Symmetry + fresh final probes' -> 'Conditional unbiasedness' -> 'Exact conditional risk' -> 'Marginal-risk identity'. Bottom row: 'Fresh certification probes' -> 'Risk estimates' -> 'Accepted-action safety'. Above the arrow FROM Risk estimates TO Accepted-action safety, put a prominent amber label 'REQUIRES valid simultaneous bounds'. A dotted arrow from Exact conditional risk to Risk estimates must end at Risk estimates, label 'target'. Do not depict confidence validity as an automatic consequence of an estimate. Title 'Proof structure and the missing assumption'. Red bottom warning 'Accepted-action safety is not whole-policy safety'. Footer 'Construction and certification costs remain after abstention.' No invented assumptions or extra labels.

## models

Create an elegant academic conceptual infographic wide landscape16:9 whitebackground navytext, minimal teal/gold palette. Title 'Four risk models: what changes?'. Four equally sized cards aligned left to right linked by arrows explicitly labelled 'change assumptions'. Card1 heading 'Full ideal' body 'Every query gains rank' and 'Perfect leading-eigenspace capture'. Card2 heading 'Rank-aware ideal' body 'Use accepted rank' and 'Still assume perfect capture'. Card3 heading 'Realized Gaussian' body 'Use the actual basis' and 'Count all residual energy'. Card4 heading 'Realized Rademacher' body 'Use the actual basis' and 'Count off-diagonal residual energy'. Under cards1and2 a bracket labelled 'Spectral surrogate'; undercards3and4 bracket 'Exact conditional risk'. Prominent bottom note 'The arrows are NOT inequalities'. Second bottom line 'Rank, capture quality, and coordinate orientation are distinct.' No quantitative plots or formulas; large readable labels. Precise conceptual structure for mathematical report.

### Corrective edit

Make just one scientific wording correction in the first card: change 'Every query gains rank' to 'Every sketch query gains rank'. Keep every other word, arrow, layout, and color unchanged.

## Exact structure contract

- Workflow: retained range sketch q; accepted basis rank r; cached AQ costs r; fresh final ell; total q+r+ell=m. In the candidate-first workflow the actual committed construction c_pre and certification s leave m-c_pre-s; abstention refunds neither.
- Proof map: fresh isotropic final probes and a fixed orthonormal basis yield unbiasedness; final-probe laws supply exact variance; algebra gives marginal risk. Fresh certification gives estimates, but accepted-action safety additionally requires valid simultaneous bounds. No implication of whole-policy safety.
- Risk models: full ideal, rank-aware ideal, realized Gaussian, realized Rademacher. Arrows change assumptions, not an ordered chain of risks. Gaussian and Rademacher refer to final probes.

The first dependency draft added misleading automatic-confidence/assumption wording and was rejected. The selected revision explicitly labels simultaneous bounds as a requirement. The first risk-model draft said every query rather than every sketch query; the selected revision fixes that distinction.

