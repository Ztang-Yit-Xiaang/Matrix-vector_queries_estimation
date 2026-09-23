# Phase 1C: Sample-Variance Confidence Theorem

**Audit classification:** the exact moment identities, Hoeffding decomposition, joint Chebyshev certificate, and net-safe acceptance implication are `PROVED` on their stated domains. The $81+$Chebyshev certificate is `PROVED BUT BUDGET-VACUOUS` on the frozen $s\le32$ grid. The sharper Hoeffding-component route is `INCOMPLETE` because its constants do not close numerically.

## 1. Why this theorem is the next question

Phase 1A found that ordinary sample variance can empirically distinguish the realized Rademacher risks of two already-constructed bases. Phase 1B then showed that the comparison may still pay for itself after construction and certification costs, but only as insurance against rare catastrophic paths. What remains is a theorem-level question: can the sample variances enclose both unknown variance numerators simultaneously with a useful finite-sample radius?

The word *simultaneously* matters. The candidate is accepted only after comparing it with the baseline. Two unrelated pointwise statements, each advertised at the public failure probability, do not establish a joint event at that same probability.

## 2. Assumption ledger

Condition on the sigma-algebra $\mathcal G$ containing both preconstructed actions and every decision made before certification. For each action $x\in\{a,0\}$, let $Q_x\in\mathbb R^{d\times r_x}$ have orthonormal columns and define

\[
R_x=I-Q_xQ_x^T.
\]

The matrix $A\in\mathbb R^{d\times d}$ is symmetric. Conditional on $\mathcal G$, the vectors $g_1,\ldots,g_s$ are independent and identically distributed coordinate-Rademacher vectors and are independent of $\mathcal G$. Put

\[
X_{x,j}=g_j^TR_xAR_xg_j,
\qquad
\sigma_x^2=\operatorname{Var}(X_{x,j}\mid\mathcal G).
\]

The unbiased sample variance is

\[
S_{x,s}^2
=
\frac1{s-1}\sum_{j=1}^s(X_{x,j}-\overline X_x)^2,
\qquad s\ge2.
\]

All probability and expectation statements below are conditional on $\mathcal G$. We suppress the conditioning only when it improves readability.

## 3. Exact first and second moments of sample variance

### Theorem 18: Exact sample-variance moments

Let $X_1,\ldots,X_s$ be conditionally iid, with conditional mean $\mu$, variance $\sigma^2$, and finite fourth central moment

\[
\mu_4=\mathbb E[(X_1-\mu)^4\mid\mathcal G].
\]

Then

\[
\mathbb E[S_s^2\mid\mathcal G]=\sigma^2
\]

and

\[
\boxed{
\operatorname{Var}(S_s^2\mid\mathcal G)
=
\frac1s
\left[
\mu_4-
\frac{s-3}{s-1}\sigma^4
\right].
}
\]

#### Proof strategy

Center first, rewrite sample variance using the raw sum and sample mean, and expand the square. Independence removes every term containing an unpaired centered factor.

#### Proof

Write $Y_i=X_i-\mu$, and set

\[
A_s=\sum_{i=1}^sY_i^2,
\qquad
B_s=\sum_{i=1}^sY_i.
\]

Since $\sum_i(Y_i-\overline Y)^2=A_s-B_s^2/s$, we have

\[
S_s^2=\frac{A_s-B_s^2/s}{s-1}.
\]

The conditional expectations satisfy

\[
\mathbb E[A_s]=s\sigma^2,
\qquad
\mathbb E[B_s^2]=s\sigma^2.
\]

Therefore

\[
\mathbb E[S_s^2]
=
\frac{s\sigma^2-\sigma^2}{s-1}
=
\sigma^2.
\]

For the second moment, independence and centering give

\[
\mathbb E[A_s^2]
=
s\mu_4+s(s-1)\sigma^4.
\]

They also give

\[
\mathbb E[B_s^4]
=
s\mu_4+3s(s-1)\sigma^4.
\]

Finally, expanding $A_sB_s^2$ shows that the terms with distinct unpaired indices vanish, leaving

\[
\mathbb E[A_sB_s^2]
=
s\mu_4+s(s-1)\sigma^4.
\]

Substitution into

\[
\mathbb E[(S_s^2)^2]
=
\frac1{(s-1)^2}
\left[
\mathbb E[A_s^2]
-\frac2s\mathbb E[A_sB_s^2]
+\frac1{s^2}\mathbb E[B_s^4]
\right]
\]

and simplification yield

\[
\mathbb E[(S_s^2)^2]
=
\frac{\mu_4}{s}
+
\frac{s^2-2s+3}{s(s-1)}\sigma^4.
\]

Subtracting $(\mathbb E S_s^2)^2=\sigma^4$ gives

\[
\operatorname{Var}(S_s^2)
=
\frac{\mu_4}{s}
-
\frac{s-3}{s(s-1)}\sigma^4,
\]

which is the claimed identity. The same formula is also recorded in the sample-variance literature; see [Benhamou, *A few properties of sample variance*](https://arxiv.org/abs/1809.03774). $\square$

## 4. The Rademacher-chaos fourth moment

For a symmetric residual matrix $H=R_xAR_x$, the centered quadratic form is

\[
Z
=
g^THg-\operatorname{tr}(H)
=
2\sum_{i<j}H_{ij}g_ig_j.
\]

Thus $Z$ is a homogeneous degree-two Rademacher polynomial. The Bonami hypercontractive inequality gives, for a degree-at-most-two polynomial at $p=4$,

\[
\|Z\|_4\le(4-1)^{2/2}\|Z\|_2=3\|Z\|_2.
\]

Raising both sides to the fourth power yields

\[
\boxed{
\mathbb E[Z^4]
\le81(\mathbb E[Z^2])^2
=81\sigma^4.
}
\]

The constant $81$ is conservative. Phase 1C does not replace it by a smaller number merely because a smaller constant looks plausible empirically. The original hypercontractive line of results goes back to [Bonami's 1970 paper](https://www.numdam.org/article/AIF_1970__20_2_335_0.pdf).

Combining this bound with Theorem 18 gives, when $\sigma^2>0$,

\[
\frac{\operatorname{Var}(S_s^2)}{\sigma^4}
\le
\frac1s
\left[
81-\frac{s-3}{s-1}
\right]
=
\frac1s\left(80+\frac2{s-1}\right).
\]

If $\sigma^2=0$, the variable $X$ is almost surely constant and $S_s^2=0$ almost surely. That boundary is handled directly; the proof never divides by $\sigma^2$ there.

## 5. Simultaneous Chebyshev certificate

### Theorem 19: Two-action scale-free certificate

Let $0<\delta_{\mathrm{joint}}<1$. Allocate

\[
\delta_{\mathrm{point}}=\frac{\delta_{\mathrm{joint}}}{2}
\]

to each of the two fixed actions. Define

\[
\varepsilon_{\mathrm{Ch}}(s,\delta_{\mathrm{joint}})
=
\sqrt{
\frac{2[80+2/(s-1)]}{s\delta_{\mathrm{joint}}}
}.
\]

Then, with conditional probability at least $1-\delta_{\mathrm{joint}}$, both actions satisfy

\[
|S_{x,s}^2-\sigma_x^2|
\le
\varepsilon_{\mathrm{Ch}}\sigma_x^2.
\]

#### Proof

For each action with positive variance, Chebyshev's inequality and the preceding relative-variance bound give

\[
\Pr\left(
|S_{x,s}^2-\sigma_x^2|
\ge\varepsilon\sigma_x^2
\right)
\le
\frac{80+2/(s-1)}{s\varepsilon^2}.
\]

Choosing the displayed radius makes the right side equal to $\delta_{\mathrm{joint}}/2$. A union bound over the two fixed actions gives total failure probability at most $\delta_{\mathrm{joint}}$. Zero-variance actions satisfy the event with probability one. $\square$

## 6. Why the proved certificate is budget-vacuous

At the primary joint failure probability,

\[
\varepsilon_{\mathrm{Ch}}(32,0.05)
=
\sqrt{\frac{2(80+2/31)}{32(0.05)}}
\approx10.004.
\]

The squared radius is

\[
\frac{2}{\delta_{\mathrm{joint}}}
\left(
\frac{80}{s}+\frac2{s(s-1)}
\right).
\]

Both summands strictly decrease for real $s>1$. Hence the radius is strictly decreasing, and its smallest value on the integer grid $2\le s\le32$ occurs at $s=32$. Since that smallest value is already greater than one,

\[
\boxed{
\varepsilon_{\mathrm{Ch}}(s,0.05)>1
\quad\text{for every }2\le s\le32.
}
\]

This is an analytic impossibility for this *particular certificate*, not evidence that sample variance contains no information. Phase 1A already demonstrated empirical information. The theorem is too conservative to turn that information into a multiplicative lower confidence bound at the available sample sizes.

## 7. Hoeffding decomposition and the first unsupported sharper step

Sample variance has the U-statistic representation

\[
S_s^2
=
\binom{s}{2}^{-1}\sum_{i<j}\frac{(X_i-X_j)^2}{2}.
\]

Let $h(x,y)=(x-y)^2/2$. Direct conditional expectation gives

\[
h_1(x)
=
\mathbb E[h(x,X')]-\sigma^2
=
\frac{(x-\mu)^2-\sigma^2}{2}.
\]

Subtracting the mean and the two first projections gives

\[
h_2(x,y)
=
h(x,y)-\sigma^2-h_1(x)-h_1(y)
=
-(x-\mu)(y-\mu).
\]

Therefore

\[
\boxed{
S_s^2-\sigma^2
=
\frac2s\sum_{i=1}^s h_1(X_i)
+
\binom{s}{2}^{-1}\sum_{i<j}h_2(X_i,X_j).
}
\]

For fixed $x$,

\[
\mathbb E[h_2(x,X')]=-(x-\mu)\mathbb E[X'-\mu]=0.
\]

Thus $h_2$ is canonical. In contrast, $h_1(X)$ is generally a nonconstant random variable, so the complete sample-variance kernel is not completely degenerate.

This distinction is not cosmetic. Adamczak's result is explicitly a moment inequality for **completely degenerate** U-statistics; see the [original paper](https://arxiv.org/abs/math/0506026). Applying it to the full sample variance would skip the first projection and would be invalid.

The sharper route allocates

\[
\delta_{x,\mathrm{linear}}
=
\delta_{x,\mathrm{degenerate}}
=
\frac{\delta_{\mathrm{joint}}}{4}
\]

for each action. The four component probabilities therefore sum exactly to the public joint failure probability.

### Audit verdict for the sharper route

The decomposition itself is `PROVED`. The first unresolved implementation-ready step is a numerical, scale-free tail bound for

\[
h_1(X)=\frac{Z^2-\sigma^2}{2},
\]

where $Z$ is degree-two chaos. This is a degree-at-most-four Rademacher polynomial, but a useful certificate requires explicit constants, not only asymptotic moment growth. The canonical term also requires every norm and universal constant in the selected U-statistic theorem to be reduced to an explicit function of $s$ and $\delta$. Phase 1C does not close either component below one on the frozen grid.

Accordingly, the sharper theorem is `INCOMPLETE`. No constant is fitted from the frozen data.

## 8. Net-safe acceptance relative to the original baseline

Assume a simultaneous relative radius $0\le\varepsilon<1$ has been proved. Then

\[
\frac{S_x^2}{1+\varepsilon}
\le
\sigma_x^2
\le
\frac{S_x^2}{1-\varepsilon}.
\]

The candidate is guaranteed no worse than the original baseline whenever

\[
\frac{S_a^2}{(1-\varepsilon)\ell_{\mathrm{paid}}}
\le
\frac{S_0^2}{(1+\varepsilon)\ell_0}.
\]

Rearranging positive factors gives the implemented rule

\[
\boxed{
S_a^2
\le
\frac{1-\varepsilon}{1+\varepsilon}
\frac{\ell_{\mathrm{paid}}}{\ell_0}
S_0^2.
}
\]

The factor $\ell_{\mathrm{paid}}/\ell_0$ cannot be dropped. Without it, the rule compares the two variance numerators or paid-order risks; it does not certify improvement over the original unstarted baseline.

If the baseline estimate is zero, if either denominator is nonpositive, if the estimates are invalid, or if $\varepsilon\ge1$, the implementation abstains. In particular, the proved Chebyshev method abstains everywhere on the Phase 1C grid.

## 9. Final theorem ledger

### PROVED

- Conditional unbiasedness and the exact variance formula for sample variance.
- The $81$ fourth-moment implication for centered quadratic Rademacher chaos under the stated hypercontractive theorem.
- The simultaneous two-action Chebyshev certificate with an explicit equal failure split.
- The analytic monotonicity and vacuity result for every $s\le32$ at joint failure probability $0.05$.
- The Hoeffding decomposition and canonical status of $h_2$.
- The net-safe accepted-candidate implication with the paid/original denominator factor.

### PROVED BUT BUDGET-VACUOUS

- The $81+$Chebyshev certificate: valid, but its relative radius exceeds one throughout the frozen budget grid.

### INCOMPLETE

- A sharper explicit scale-free radius combining the nondegenerate linear projection and canonical U-statistic remainder.

### OPEN

- Whether another estimator or a sharper theorem yields a nonvacuous simultaneous certificate at $s\le32$.
- A candidate timing architecture that remains useful on ordinary paths.
- An online allocator; none is implemented in Phase 1C.

## 10. Phase 1D extension: analytic no-go for the frozen linear-tail route

Phase 1D asks whether a sharper large-deviation inequality can at least control
the necessary nondegenerate Hoeffding projection.  It does not change the
sample-variance estimator or analyze the canonical remainder.

Let

\[
H=RAR,
\qquad
C=H-\operatorname{diag}(H),
\qquad
Z=g^TCg,
\]

where $C$ is symmetric and zero diagonal.  On the positive-variance branch,

\[
\sigma^2=\operatorname{Var}(g^THg\mid\mathcal G)=2\lVert C\rVert_F^2>0,
\qquad
\kappa=\frac{\lVert C\rVert_2}{\lVert C\rVert_F}\in(0,1].
\]

Theorem 2, equation (8), of Cortinovis and Kressner gives

\[
\Pr(|g^TCg|\ge t\mid\mathcal G)
\le
2\exp\!\left(
-\frac{t^2}{8\lVert C\rVert_F^2+8t\lVert C\rVert_2}
\right).
\]

Substituting $t=u\sigma$ and defining $V=Z^2/\sigma^2$ yields

\[
\Pr(V\ge v\mid\mathcal G)
\le
p_\kappa(v)
:=
\min\!\left\{
1,
2\exp\!\left[-\frac{v}{4+4\sqrt{2v}\kappa}\right]
\right\}.
\]

This is an upper large-deviation bound.  It is not a lower-tail or small-ball
bound for $Z^2$.

The first Hoeffding projection satisfies

\[
\frac{(2/s)\sum_i h_1(X_i)}{\sigma^2}
=
\frac1s\sum_{i=1}^s
\left(\frac{Z_i^2}{\sigma^2}-1\right)
=:L_s.
\]

For $V^{(T)}=\min\{V,T\}$, the frozen truncation analysis uses

\[
\beta_\kappa(T)
=
4e^{-a_{\kappa,T}\sqrt T}
\left(
\frac{\sqrt T}{a_{\kappa,T}}
+
\frac1{a_{\kappa,T}^2}
\right),
\qquad
a_{\kappa,T}
=
\frac1{4\sqrt2\kappa+4/\sqrt T},
\]

and the capped-variance envelope

\[
\nu_\kappa(T)
=
\min\!\left\{
81,
\frac{T^2}{4},
81-\max\{0,1-\beta_\kappa(T)\}^2
\right\}.
\]

### Proposition 10.1: frozen variance-envelope floor

For every $\kappa\in(0,1]$ and every frozen cap $T\ge64$,

\[
\boxed{80\le\nu_\kappa(T)\le81.}
\]

Indeed, $T^2/4\ge1024$, while

\[
0\le\max\{0,1-\beta_\kappa(T)\}\le1
\]

implies that the third entry in the minimum belongs to $[80,81]$.

### Theorem 10.2: lower-tail no-go for the frozen truncation--Bernstein family

Let $s\le32$, $0<\varepsilon<1$, $\kappa\in(0,1]$, and let $T\ge64$ be one of
the frozen caps.  If $\beta_\kappa(T)<\varepsilon$, put

\[
x=\varepsilon-\beta_\kappa(T)\in(0,1).
\]

The frozen one-sided Bernstein expression is

\[
D_{s,\kappa}(\varepsilon,T)
=
\exp\!\left[
-\frac{s x^2}{2(\nu_\kappa(T)+x/3)}
\right].
\]

By Proposition 10.1 and $x<1$,

\[
0<
\frac{s x^2}{2(\nu_\kappa(T)+x/3)}
<
\frac{s}{160}.
\]

Consequently,

\[
\boxed{
D_{s,\kappa}(\varepsilon,T)
>
e^{-s/160}
\ge
e^{-0.2}
\approx0.8187307531.
}
\]

If $\beta_\kappa(T)\ge\varepsilon$, the cap supplies no certificate and its
reported probability bound is one, so the same obstruction remains.  The
largest declared directed component probability is $0.10/2=0.05$.  Hence no
frozen sample size, cap, structural ratio, or failure allocation closes this
candidate-underestimation bound.

The theorem is `PROVED` for this explicit bound family, and its deterministic
verdict is `STRONG LINEAR NO-GO`.

This is route-specific.  It does not prove that ordinary sample variance, a
different confidence construction, or realized-risk certification is
universally impossible.  Since this necessary linear component already fails,
Phase 1D does not analyze the canonical $h_2$ term.

For a directed decision, only the candidate-underestimation radius has the
algebraic requirement $\varepsilon_{a,-}<1$.  A baseline-overestimation radius
may be any finite nonnegative number, although a large value is practically
weak.  A future direct target is the common-probe risk difference

\[
\widehat\Delta_R
=
\binom{s}{2}^{-1}
\sum_{i<j}
\left[
\frac{(X_{a,i}-X_{a,j})^2}{2\ell_a}
-
\frac{(X_{0,i}-X_{0,j})^2}{2\ell_0}
\right],
\]

whose one-sided analysis may retain covariance cancellation discarded by two
separate confidence intervals.  That theorem remains `OPEN`.
