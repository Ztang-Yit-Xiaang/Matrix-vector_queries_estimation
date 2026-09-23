# Phase 2B report: direct paired risk-difference confidence

## Executive verdict

\[
\boxed{\texttt{PROVED BUT BUDGET-VACUOUS}}
\]

The direct common-probe target is mathematically cleaner than two separate
risk intervals: disjoint probe pairs produce conditionally iid signed
observations whose expectation is exactly the paid-candidate minus
original-baseline Rademacher risk difference.  However, the only legal
data-only scale controls closed with explicit constants are unusable for
\(s\le32\).

No frozen probes were replayed, no matrix--vector queries were issued, and no
estimator or allocator was changed.

## What was proved

For \(n=s/2\), define

\[
D_j=
\frac{(X_{a,2j-1}-X_{a,2j})^2}{2\ell_a}
-
\frac{(X_{0,2j-1}-X_{0,2j})^2}{2\ell_0}.
\]

Then, conditionally on the fixed matrix and bases,

\[
\mathbb E D_j=\Delta_R
=\frac{\sigma_a^2}{\ell_a}-\frac{\sigma_0^2}{\ell_0},
\]

and the \(D_j\)'s are iid because they use disjoint probe pairs.  Common-probe
dependence is retained inside each signed observation.

The centered variable \(P_j=D_j-\Delta_R\) is a degree-at-most-four
Rademacher polynomial.  The verified Boolean hypercontractive theorem gives

\[
\|P_j\|_4\le9\|P_j\|_2,
\qquad
\mathbb EP_j^4\le6561(\mathbb EP_j^2)^2.
\]

This is a theorem-level universal bound, not a fitted empirical constant.

## Why the elementary route stops

With \(v_D=\operatorname{Var}(D_j)\), the exact sample-variance identity and
the factor \(6561\) imply

\[
\frac{\operatorname{Var}(S_D^2)}{v_D^2}
\le
\frac{6561-(n-3)/(n-1)}{n}.
\]

Even at the most favorable frozen point, \(s=32\), \(n=16\), and
\(\delta_{\rm joint}=0.10\), the relative scale radius exceeds \(90\).  A
finite observable variance upper bound requires that radius to be below one.
It never is.

## Why the robust MoM baseline also stops

For independent centered copies \(P,P'\), let

\[
W=\frac{(P-P')^2}{2}.
\]

Then \(\mathbb EW=v_D\) and

\[
\operatorname{Var}(W)\le3281v_D^2.
\]

A Chebyshev-valid scale block with relative error at most one and failure
probability strictly below \(1/4\) needs at least \(13125\) such observations.
The entire frozen \(s=32\) experiment provides only \(16\) signed observations,
or at most eight independent scale pairs.  Median amplification cannot begin.

This is a no-go for the declared self-contained robust construction, not for
all conceivable robust estimators.

## Why the complete U-statistic was not pursued

The complete paired U-statistic has a nondegenerate linear Hoeffding
projection plus a canonical remainder.  Canonical U-statistic inequalities
apply only to the remainder.  Because the necessary linear projection still
lacks an observable small-sample scale bound, the approved stopping rule says
to stop before implementing the canonical term.

## Imported-theorem audit

- `PROVED`: O'Donnell's degree-\(d\) Boolean hypercontractive corollary gives
  the factor \(9\) in \(L^4/L^2\) and hence \(6561\) in fourth moments.
- `REJECTED AS INAPPLICABLE`: Maurer--Pontil empirical Bernstein assumes
  bounded observations; the scale-free signed \(D_j\)'s have no legal known
  range.
- `SCOPE ONLY`: Giné--Latała--Zinn concerns canonical U-statistics and cannot
  certify the full nondegenerate statistic.
- `NOT IMPORTED`: no Catoni constant was used without an exact observable
  theorem-to-API mapping.

Primary sources:

- <https://www.cs.cmu.edu/~odonnell/boolean-analysis/lecture16.pdf>
- <https://arxiv.org/abs/0907.3740>
- <https://arxiv.org/abs/0909.5366>
- <https://arxiv.org/abs/math/0003228>

## Classification

### PROVED

- Exact unbiasedness and iid structure of the signed-pair reduction.
- Degree-four and fourth-moment factor audits.
- Uniform budget vacuity of the elementary scale route.
- Uniform budget vacuity of the specified robust scale/MoM baseline.

### EMPIRICALLY ESTABLISHED

- Phase 2A's common-probe pairing signal remains valid historical evidence; it
  is not used as a theorem constant here.

### INCOMPLETE

- A sharper observable unknown-scale one-sided theorem.
- A full-U theorem that also controls its linear projection.

### OPEN

- Whether structural matrix information or a genuine small-ball argument can
  replace the worst-case factor \(6561\) without fitting to held-out paths.

## Recommendation

Do not implement a certified allocator and do not replay the frozen probes.
For the current UROP, freeze the theory here and write the final report.  A
future continuation may study a sharper direct-difference theorem, but it must
change the mathematical information available—not merely rerun the same
worst-case scale argument.
