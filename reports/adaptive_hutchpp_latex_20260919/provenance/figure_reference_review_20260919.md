# Manuscript illustrations: reference review and proposed revision

Date: 2026-09-19. Status: reference inspection complete; both layouts approved for implementation. The user's final criterion is accessibility and interest to a student outside the field.

## Reference evidence

The following figure pages were rendered and visually inspected. These are established papers in the relevant field, not a claim that a journal's reputation measures graphical quality. Borrow visual principles, not artwork or empirical content.

- Martinsson and Tropp, *Randomized Numerical Linear Algebra: Foundations & Algorithms*, Acta Numerica (2020), [Figures 11–12, PDF pages 98 and 100](https://arxiv.org/pdf/2002.01387). Figure 11 pairs a sparse tree with its matrix partition. Figure 12 explains successive computations with matrix shapes and a small, consistent set of highlighted blocks. The mathematical objects do the explanatory work; captions supply detail.
- Halko, Martinsson and Tropp, *Finding Structure with Randomness*, SIAM Review (2011), [Figure 7.1, PDF page 39](https://arxiv.org/pdf/0909.4061). Two linked geometric views use direct labels and consistent colors to relate boundaries and sampled points. This is a geometry illustration, not a randomized-range-finder workflow.
- Meyer et al., *Hutch++: Optimal Stochastic Trace Estimation*, SOSA (2021), [Figure 1, PDF page 12](https://arxiv.org/pdf/2010.09649). Four aligned quantitative panels reuse line styles and colors; the caption defines the median and interquartile bands over 200 trials. Apply this consistency to our measured plots, without treating it as a workflow template.

## Recommendation: mathematical illustration, not decorated boxes

Use an original three-panel Figure 1. It should explain one claim: a query can construct a basis, evaluate its captured trace, or estimate the remaining trace; certification is a separate, irreversible expenditure.

**(a) Construct the basis.** Draw dimensionally consistent matrix silhouettes for S, Y=AS, Q, and AQ. Label attempted sketch width q separately from accepted rank r. Show the two product costs q and r. Do not imply that orthogonalization guarantees leading-eigenspace capture or that discarded columns refund queries. No extra oracle cost is assigned to QR itself.

**(b) Split the trace.** Draw the Q subspace and a projected fresh probe h=R_Q g, alongside the small captured matrix Q^T A Q and residual operator H_Q=R_Q A R_Q. Display the trace identity, not the generally false matrix identity A=QQ^T A QQ^T+R_Q A R_Q. If a block-matrix view is used, retain cross blocks and explain that their traces vanish. Geometry is schematic, not an empirical angle measurement.

**(c) Account for every query.** Use two aligned symbolic ledgers: q+r+ell=m for the frozen estimator, and c_pre+s+ell_paid=m for candidate-first certification. Mark certification as an optional branch; its probes and the final estimator probes are distinct and fresh. c_pre is the actual committed construction count. Only a genuinely reusable nested shared-prefix construction permits c_pre=max_x(q_x+r_x). Segment widths must either be nonquantitative and labeled schematic or use an explicitly declared numerical example. Abstention does not refund cost.

Figure 2 retains the approved shared-input branching structure. Add small spectrum and residual-matrix illustrations: an ideal-tail branch separates the full-rank and rank-aware models; the actual-basis branch forms H_Q and produces parallel Gaussian and Rademacher risks. Highlight all entries for Gaussian variance and off-diagonal entries for Rademacher variance. This masking describes a variance functional, not a modification of A; diagonal entries still contribute to the trace. No arrow asserts a risk inequality.

## Alternatives considered

1. **Recommended:** matrix/geometry/budget composite. More explanatory without more prose.
2. Pure process flow. Compact, but still largely text boxes and therefore does not answer the request for illustration.
3. Formula table. Precise and economical, but duplicates the equations already in the paper and does not visualize the mechanism.

## Shared style and scope

White background; thin outlines; direct labels; consistent basis, residual, and certification colors; small panel letters; manuscript-sized typography. No trophies, decorative icons, gradients, banners, internal phase codes, or oversized figure titles. Captions define symbols and necessary qualifications once. Use editable vector/LaTeX geometry for exact mathematical labels. Preserve the existing figures and measured data. Do not add unmeasured curves, angles, or performance claims.

The independent logic audit from September 18 remains applicable. Integrate its proof qualifications and concise-prose edits with the visual revision. Deliver a separate source package, not an overwrite of the September 17 manuscript; no new standalone manuscript PDF.
