# Phase 2B proof audit: direct paired Rademacher risk difference

## Status and scope

This note studies one candidate--baseline pair fixed before fresh common
Rademacher probes are drawn.  It does not select an action and it does not
modify Hutch++.  All probability statements are conditional on the
pre-certification sigma-algebra \(\mathcal G\), which contains the matrix,
bases, ranks, denominators, and fixed pair.

The final classification is:

\[
\boxed{\texttt{PROVED BUT BUDGET-VACUOUS}.}
\]

This classification applies to the two explicit independent-pair theorem
routes proved below on \(s\in\{4,8,16,32\}\).  It is not an impossibility
theorem for every direct difference certificate.

## 1. Exact signed-pair reduction — PROVED

For action \(x\in\{a,0\}\), let

\[
X_{x,i}=g_i^T R_x A R_x g_i,
\qquad
\sigma_x^2=\operatorname{Var}(X_{x,i}\mid\mathcal G),
\]

where the \(g_i\)'s are conditionally iid coordinate-Rademacher probes.  With
fixed positive denominators \(\ell_a,\ell_0\), define

\[
D_j=
\frac{(X_{a,2j-1}-X_{a,2j})^2}{2\ell_a}
-
\frac{(X_{0,2j-1}-X_{0,2j})^2}{2\ell_0}.
\]

For iid copies \(X,X'\),

\[
\mathbb E\frac{(X-X')^2}{2}
=\frac12\{\operatorname{Var}(X)+\operatorname{Var}(X')\}
=\operatorname{Var}(X).
\]

Therefore

\[
\boxed{
\mathbb E[D_j\mid\mathcal G]
=\frac{\sigma_a^2}{\ell_a}-\frac{\sigma_0^2}{\ell_0}
=\Delta_R.
}
\]

Different \(D_j\)'s use disjoint probe pairs, so they are conditionally iid.
The candidate and baseline terms inside a single \(D_j\) use the same probes;
their dependence is intentionally preserved.

## 2. Degree and fourth-moment audit — PROVED

For a fixed symmetric residual matrix \(H_x=R_xAR_x\), the diagonal of
\(g^TH_xg\) is deterministic because \(g_i^2=1\).  Its centered part is a
multilinear polynomial of degree two in the Rademacher coordinates.  Squaring
a difference of two such forms gives degree at most four.  Hence

\[
P_j=D_j-\Delta_R
\]

is a centered multilinear polynomial of degree at most four in the \(2d\)
independent signs belonging to one probe pair.

O'Donnell's hypercontractive corollary states that a degree-at-most \(d\)
polynomial on the uniform Boolean cube satisfies

\[
\|f\|_q\le(q-1)^{d/2}\|f\|_2,
\qquad q\ge2.
\]

For \(q=4,d=4\),

\[
\|P_j\|_4\le9\|P_j\|_2.
\]

Writing \(v_D=\mathbb E[P_j^2\mid\mathcal G]\), this yields

\[
\boxed{
\mathbb E[P_j^4\mid\mathcal G]
\le9^4v_D^2
=6561v_D^2.
}
\]

Source: Ryan O'Donnell, *Analysis of Boolean Functions*, Lecture 16,
Corollary 1.3:
<https://www.cs.cmu.edu/~odonnell/boolean-analysis/lecture16.pdf>.

The constant is deliberately worst-case.  No Phase 2A empirical kurtosis or
matrix-specific norm is substituted for it.

## 3. Elementary observable-scale route — PROVED BUT BUDGET-VACUOUS

Let \(n=s/2\) and let \(S_D^2\) be the ordinary sample variance of
\(D_1,\ldots,D_n\).  For iid data with fourth central moment \(\mu_{4,D}\),

\[
\operatorname{Var}(S_D^2\mid\mathcal G)
=\frac1n\left[
\mu_{4,D}-\frac{n-3}{n-1}v_D^2
\right].
\]

Using \(\mu_{4,D}\le Kv_D^2\), with \(K=6561\), gives

\[
\frac{\operatorname{Var}(S_D^2\mid\mathcal G)}{v_D^2}
\le
A_n
:=
\frac{K-(n-3)/(n-1)}{n}.
\]

Chebyshev then gives

\[
\Pr\{|S_D^2-v_D|\ge\varepsilon_vv_D\mid\mathcal G\}
\le\frac{A_n}{\varepsilon_v^2}.
\]

An observable upper confidence bound for \(v_D\) based on \(S_D^2\) requires
\(\varepsilon_v<1\).  Under the frozen equal split
\(\delta_{\rm scale}=\delta_{\rm joint}/2\), the proved radius is

\[
\varepsilon_v=\sqrt{A_n/\delta_{\rm scale}}.
\]

The most favorable frozen point is \(s=32\), \(n=16\), and
\(\delta_{\rm joint}=0.10\).  Even there,

\[
A_{16}
=\frac{6561-13/15}{16}
>410,
\qquad
\varepsilon_v>\sqrt{410/0.05}>90.
\]

Thus \(\varepsilon_v>1\) everywhere on the frozen grid.  Cantelli can bound
the mean if \(v_D\) is known, but the required observable variance upper bound
does not close.  The one-sided radius is therefore infinite rather than fitted
or silently regularized.

## 4. Self-contained robust scale/MoM route — PROVED BUT BUDGET-VACUOUS

Let \(P=D-\Delta_R\) and let \(P'\) be an independent copy.  Define the
nonnegative scale observation

\[
W=\frac{(P-P')^2}{2}.
\]

Then \(\mathbb EW=v_D\).  Since \(P,P'\) are centered and iid,

\[
\mathbb E(P-P')^4
=2\mathbb EP^4+6v_D^2.
\]

Consequently,

\[
\mathbb EW^2
=\frac12\mathbb EP^4+\frac32v_D^2
\le\left(\frac K2+\frac32\right)v_D^2,
\]

and hence

\[
\boxed{
\operatorname{Var}(W)
\le\frac{K+1}{2}v_D^2
=3281v_D^2.
}
\]

For a block mean of \(b\) independent \(W\)'s, Chebyshev gives

\[
\Pr\{\bar W_b<(1-\varepsilon_v)v_D\}
\le\frac{3281}{b\varepsilon_v^2}.
\]

To make one block fail with probability strictly below \(1/4\) while retaining
\(\varepsilon_v\le1\), one needs

\[
b>4(3281)=13124,
\]

so the first admissible integer is \(13125\).  With \(s\le32\), the signed-pair
sample has \(n\le16\), and even if every \(D_j\) were sacrificed to scale
estimation it would provide at most eight independent \(W\)'s.  Therefore this
explicit robust scale/MoM construction is analytically budget-vacuous before
any mean-estimation block is allocated.

Catoni's paper confirms that bounded-kurtosis adaptation requires careful
variance estimation and distinguishes observable intervals from adaptations
that are not observable.  We do not import an unverified Catoni constant or
claim that the self-contained MoM baseline is optimal:
<https://arxiv.org/abs/0909.5366>.

## 5. Why bounded empirical Bernstein is rejected — PROVED SCOPE CHECK

Maurer and Pontil's empirical Bernstein Theorem 4 assumes iid observations in
\([0,1]\).  The signed \(D_j\)'s are generally unbounded under the scale-free
problem class, and Phase 2B forbids importing a matrix-dependent range bound.
The theorem is therefore not applicable:
<https://arxiv.org/abs/0907.3740>.

## 6. Complete U-statistic route — INCOMPLETE AND STOPPED

The complete common-probe statistic equals the difference of the two sample
variances divided by their denominators.  Its Hoeffding decomposition is

\[
\widehat\Delta_U-\Delta_R
=\frac2s\sum_iK_1(Y_i)
+\binom{s}{2}^{-1}\sum_{i<j}K_2(Y_i,Y_j),
\]

where

\[
K_1(Y)=
\frac{Z_a^2-\sigma_a^2}{2\ell_a}
-\frac{Z_0^2-\sigma_0^2}{2\ell_0}
\]

is nondegenerate and

\[
K_2(Y,Y')=-\frac{Z_aZ_a'}{\ell_a}+\frac{Z_0Z_0'}{\ell_0}
\]

is canonical.  Giné--Latała--Zinn supply canonical U-statistic inequalities,
but those results cannot be applied to the complete nondegenerate statistic.
Because the necessary linear projection has no closed observable small-sample
scale bound in the audited routes, the preregistered stopping rule prevents a
separate \(K_2\) implementation:
<https://arxiv.org/abs/math/0003228>.

This route is `INCOMPLETE`, not refuted.

## 7. Exact-zero branch

If \(v_D=0\) is known from the conditional law, then
\(D_j=\Delta_R\) almost surely and the exact radius is zero.  Under the strict
data-only API, observing equal \(D_j\)'s does not prove \(v_D=0\).  Therefore
the implementation records a zero empirical variance but still abstains.

## 8. Final safe statement

### PROVED

- The independent signed-pair observations are conditionally iid and have
  mean \(\Delta_R\).
- Their centered law is a degree-at-most-four Rademacher polynomial and obeys
  the uniform fourth-moment factor \(6561\).
- The elementary sample-variance scale route is budget-vacuous for every
  frozen \((s,\delta)\).
- The declared pairwise-scale MoM route needs a scale block of at least
  \(13125\), far beyond the frozen budget.

### EMPIRICALLY ESTABLISHED ELSEWHERE

- Phase 2A found aggregate common-probe variance reduction, with much weaker
  cancellation on the catastrophic paths that matter most.

### INCOMPLETE

- A sharper observable unknown-scale mean theorem with explicit constants for
  this exact signed-pair class.
- A full-U theorem that separately closes its nondegenerate projection.

### OPEN

- Whether matrix-structural side information, a genuine small-ball theorem, or
  another direct one-sided construction can make \(s\le32\) useful.

The correct Phase 2B conclusion is narrow:

> Under the strict data-only contract, the two explicit independent-pair
> confidence routes proved here do not yield a finite observable one-sided
> radius on the frozen certification grid.  No replay or allocator is
> authorized.
