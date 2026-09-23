# Phase 1C: Sample-Variance Confidence Theorem and Usefulness Audit

## Executive verdict

**THEOREM ONLY / BUDGET-VACUOUS**

Phase 1C proves a valid, scale-free, simultaneous confidence theorem for the ordinary sample variances of one fixed candidate--baseline pair. The theorem is not useful at the available certification budgets: at the primary joint failure probability $0.05$, its relative radius decreases from $28.40$ at $s=4$ to only $10.00$ at $s=32$. A multiplicative lower confidence bound requires a radius below one.

The sharper Hoeffding-decomposition route is classified `INCOMPLETE`. The decomposition is exact, but no explicit numerical tail radius was closed for both the nondegenerate linear projection and the canonical remainder. No universal constant was fitted from frozen data.

No Hutch++ estimator was changed, no allocator was added, and no matrix--vector query was issued.

## What was proved

For one conditionally iid sample $X_1,\ldots,X_s$, the unbiased sample variance satisfies

\[
\mathbb E[S_s^2\mid\mathcal G]=\sigma^2
\]

and

\[
\operatorname{Var}(S_s^2\mid\mathcal G)
=
\frac1s
\left[
\mu_4-
\frac{s-3}{s-1}\sigma^4
\right].
\]

The centered Rademacher quadratic form is a degree-two chaos. Retaining the audited hypercontractive bound

\[
\mu_4\le81\sigma^4
\]

gives

\[
\frac{\operatorname{Var}(S_s^2\mid\mathcal G)}{\sigma^4}
\le
\frac1s\left(80+\frac2{s-1}\right).
\]

Splitting the public joint failure probability equally across the two fixed actions and applying a union bound yields

\[
\boxed{
\varepsilon_{\mathrm{Ch}}(s,\delta_{\mathrm{joint}})
=
\sqrt{
\frac{2[80+2/(s-1)]}{s\delta_{\mathrm{joint}}}
}.
}
\]

This statement is conditional on the pre-certification sigma-algebra and does not cover searching over additional data-dependent actions.

## Analytic vacuity result

At the primary failure probability, the proved radii are:

| $s$ | $\varepsilon_{\mathrm{Ch}}(s,0.05)$ | Multiplicative lower bound available? |
|---:|---:|:---|
| 4 | 28.401878 | No |
| 8 | 20.035682 | No |
| 16 | 14.153916 | No |
| 32 | 10.004031 | No |

The squared radius is a positive constant times

\[
\frac{80}{s}+\frac2{s(s-1)},
\]

which strictly decreases for $s>1$. Therefore $s=32$ is the best point in the frozen grid. Since even that radius is greater than ten, the certificate is analytically vacuous for every $s\le32$. This conclusion was reached before looking at empirical performance and therefore does not select $s$ post hoc.

## Why a radius above one stops the decision rule

If $0\le\varepsilon<1$, the simultaneous event implies

\[
\frac{S_x^2}{1+\varepsilon}
\le
\sigma_x^2
\le
\frac{S_x^2}{1-\varepsilon}.
\]

The lower endpoint depends on $1-\varepsilon$. Once $\varepsilon\ge1$, that multiplicative lower bound is no longer positive and cannot support an upper-versus-lower comparison.

The implemented accepted-candidate rule would be

\[
S_a^2
\le
\frac{1-\varepsilon}{1+\varepsilon}
\frac{\ell_{\mathrm{paid}}}{\ell_0}
S_0^2.
\]

The factor $\ell_{\mathrm{paid}}/\ell_0$ is essential: it charges the candidate's sunk construction and certification cost when comparing it with the original baseline. Because the proved radius is at least one throughout the grid, the implementation abstains everywhere rather than pretending that a one-sided lower bound exists.

## Hoeffding audit

The sample variance is the U-statistic with kernel

\[
h(x,y)=\frac{(x-y)^2}{2}.
\]

Its exact Hoeffding projections are

\[
h_1(x)=\frac{(x-\mu)^2-\sigma^2}{2}
\]

and

\[
h_2(x,y)=-(x-\mu)(y-\mu).
\]

Thus

\[
S_s^2-\sigma^2
=
\frac2s\sum_i h_1(X_i)
+
\binom{s}{2}^{-1}\sum_{i<j}h_2(X_i,X_j).
\]

The second kernel is canonical, but the first projection is generally nondegenerate. The Adamczak theorem located during the audit explicitly treats completely degenerate U-statistics. It may therefore be considered for (h_2), not for the full sample variance. See [Adamczak's original paper](https://arxiv.org/abs/math/0506026).

The first unsupported sharper step is an explicit, useful scale-free tail constant for the linear variable

\[
h_1(X)=\frac{Z^2-\sigma^2}{2},
\]

where (Z) is degree-two Rademacher chaos. The degenerate component also needs every theorem constant and kernel norm reduced numerically. Until both components close, combining them would be incomplete.

## Relationship to the frozen empirical evidence

The frozen Phase 1B primary empirical selector had a selected/original mean-risk ratio of $0.036329$, with conditional bootstrap interval $[0.014628,0.639830]$, and accepted the candidate on $80.03\%$ of net-beneficial top-5% catastrophic paths.

Those results show that the observations contain useful information. They do not make this conservative theorem nonvacuous. Phase 1C deliberately does not choose the empirically best $s$ after seeing those outcomes. Because no proved radius falls below one, the practical bootstrap gate is not run; its output artifact is an explicit zero-row table with a stable schema.

The tension is scientifically informative:

\[
\boxed{
\text{empirically strong signal}
\quad\not\Rightarrow\quad
\text{useful distribution-free finite-sample certificate}.
}
\]

## Artifact audit

- Certificate-grid rows: 24, covering two theorem routes, four sample sizes, and three joint failure probabilities.
- Gate rows: 1.
- Practical bootstrap rows: 0, by the preregistered analytic stopping rule.
- New matrix--vector queries: 0.
- Frozen Phase 1A and Phase 1B checksums: verified before and after the audit.
- Historical result checksums: verified.
- Online allocator implemented: no.

## Evidence ledger

### PROVED

- Exact expectation and variance of unbiased sample variance.
- The retained $81$ fourth-moment implication for centered quadratic Rademacher chaos.
- The two-action joint Chebyshev certificate.
- Strict decrease and $s\le32$ vacuity at joint failure probability $0.05$.
- The Hoeffding decomposition and degeneracy of (h_2).
- The accepted-candidate net-safety implication with the paid/original denominator factor.

### PROVED BUT BUDGET-VACUOUS

- The $81+$Chebyshev certificate on the frozen Phase 1C grid.

### EMPIRICALLY ESTABLISHED

- Phase 1A/1B sample variance contains strong selective information under the frozen path population, especially on catastrophic paths.

### INCOMPLETE

- A sharper explicit scale-free radius combining the linear and canonical Hoeffding terms.

### OPEN

- Whether a sharper theorem or a different variance estimator can certify the same signal with $s\le32$.
- Whether an online timing architecture can avoid paying on ordinary paths.
- Orientation-external validation and any online certified allocator.

## Recommendation

Do not implement an allocator from the present theorem. The next mathematical step, if the continuation project proceeds, is to attack the nondegenerate linear projection first and determine whether an explicit scale-free constant can realistically fall below one at $s\le32$. If it cannot, the research should consider either stronger assumptions with an observable scale, a different robust variance estimator, or a larger certification budget—each as a separately frozen design.
