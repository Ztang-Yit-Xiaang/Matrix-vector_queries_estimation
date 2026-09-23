# Phase 1D Revision: Analytic Linear Lower-Tail No-Go

**Status:** revised frozen design for implementation after written-spec review on 2026-08-19.

## 1. Purpose and boundary

Phase 1C proved a valid sample-variance theorem whose fourth-moment-plus-Chebyshev radius is budget-vacuous. Phase 1D isolates the necessary nondegenerate Hoeffding component and asks

\[
\boxed{
\text{Can a large-deviation bound for quadratic Rademacher chaos control the
candidate-underestimation tail of the linear projection at }s\le32?
}
\]

For the frozen truncation--Bernstein family below, the answer is already analytic: `STRONG LINEAR NO-GO`.

This phase will preserve the correct imported theorem, normalization, zero-variance branch, truncation-bias argument, and factor audit. It will prove the lower-tail obstruction, verify it on frozen grids, and issue isolated no-query artifacts.

This phase will not implement upper-tail radius search, cap optimization, bootstrap analysis, structural sensitivity optimization, canonical-kernel concentration, a direct-difference theorem, or an allocator. It will not claim that ordinary sample-variance certification is universally impossible.

## 2. Assumption ledger and exact variance

Condition on the pre-certification sigma-algebra \(\mathcal G\). For one fixed action, let

\[
H=RAR,
\qquad
R=I-QQ^T,
\]

where \(A\in\mathbb R^{d\times d}\) is symmetric and \(Q\) has orthonormal columns. Define

\[
C=H-\operatorname{diag}(H).
\]

Then \(C=C^T\) and \(C_{ii}=0\). For a coordinate-Rademacher vector \(g\), independent of \(\mathcal G\), set

\[
X=g^THg,
\qquad
Z=X-\mathbb E[X\mid\mathcal G]=g^TCg.
\]

The exact conditional variance is

\[
\boxed{
\sigma^2=\operatorname{Var}(X\mid\mathcal G)=2\lVert C\rVert_F^2.
}
\]

The observations \(X_1,\ldots,X_s\) are conditionally iid given \(\mathcal G\), with \(s\ge2\).

If \(\sigma^2=0\), then \(C=0\), \(Z=0\), and both Hoeffding components vanish almost surely. No expression containing \(Z/\sigma\) is evaluated. If \(\sigma^2>0\), then \(C\ne0\), so the imported theorem applies after symmetry and zero diagonal are verified.

## 3. Imported theorem and normalization

Use Theorem 2, equation (8), of Cortinovis and Kressner, *On Randomized Trace Estimates for Indefinite Matrices with an Application to Determinants*, Foundations of Computational Mathematics 22 (2022), 875--903, https://doi.org/10.1007/s10208-021-09525-9.

For a nonzero symmetric zero-diagonal matrix \(C\), their theorem gives

\[
\Pr(|g^TCg|\ge t\mid\mathcal G)
\le
2\exp\!\left(
-\frac{t^2}{8\lVert C\rVert_F^2+8t\lVert C\rVert_2}
\right),
\qquad t>0.
\]

The proof and code call this Theorem 2, not “Theorem 8”; 8 is the displayed equation number.

On the positive-variance branch define

\[
\kappa=\frac{\lVert C\rVert_2}{\lVert C\rVert_F}\in(0,1].
\]

Since \(\sigma=\sqrt2\lVert C\rVert_F\), substitution of \(t=u\sigma\) yields

\[
\boxed{
\Pr(|Z|/\sigma\ge u\mid\mathcal G)
\le
\min\!\left\{
1,
2\exp\!\left[-\frac{u^2}{4+4\sqrt2\kappa u}\right]
\right\}.
}
\]

The outer minimum with one is mandatory before tail integration.

Let

\[
V=Z^2/\sigma^2.
\]

Then \(V\ge0\), \(\mathbb EV=1\), and

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

This is a large-deviation bound. It is not a small-ball theorem for \(Z\).

## 4. Hoeffding factor audit

For

\[
h(x,y)=\frac{(x-y)^2}{2},
\]

the nondegenerate projection is

\[
h_1(X)=\frac{Z^2-\sigma^2}{2}.
\]

With

\[
W=V-1=\frac{Z^2-\sigma^2}{\sigma^2},
\]

the normalized linear component is

\[
\boxed{
L_s
=
\frac{(2/s)\sum_i h_1(X_i)}{\sigma^2}
=
\frac1s\sum_iW_i.
}
\]

If \(Y=(Z^2-\sigma^2)/(2\sigma^2)\) is used, then \(L_s=2\bar Y\). Tests must protect this factor.

## 5. Frozen truncation--Bernstein family

For \(T>1\), define

\[
V^{(T)}=\min\{V,T\}.
\]

The retained degree-two hypercontractive result gives

\[
\mathbb E[V^2\mid\mathcal G]
=
\frac{\mathbb E[Z^4\mid\mathcal G]}{\sigma^4}
\le81.
\]

Put

\[
a_{\kappa,T}
=
\frac{1}{4\sqrt2\kappa+4/\sqrt T}.
\]

For \(u\ge\sqrt T\),

\[
\frac{u^2}{4+4\sqrt2\kappa u}
\ge
a_{\kappa,T}u.
\]

Therefore the truncation bias obeys

\[
\begin{aligned}
b_T
&:=\mathbb E[(V-T)_+\mid\mathcal G]\\
&\le
\beta_\kappa(T)\\
&:=4e^{-a_{\kappa,T}\sqrt T}
\left(
\frac{\sqrt T}{a_{\kappa,T}}
+
\frac1{a_{\kappa,T}^2}
\right).
\end{aligned}
\]

The frozen capped-variance envelope is

\[
\boxed{
\nu_\kappa(T)
=
\min\!\left\{
81,
\frac{T^2}{4},
81-\max\{0,1-\beta_\kappa(T)\}^2
\right\}.
}
\]

For \(0<\varepsilon<1\) and \(\beta_\kappa(T)<\varepsilon\), let

\[
x=\varepsilon-\beta_\kappa(T)\in(0,1).
\]

One-sided Bernstein gives

\[
\boxed{
\Pr(L_s\le-\varepsilon\mid\mathcal G)
\le
D_{s,\kappa}(\varepsilon,T)
:=
\exp\!\left[
-\frac{s x^2}{2(\nu_\kappa(T)+x/3)}
\right].
}
\]

If \(\beta_\kappa(T)\ge\varepsilon\), that cap supplies no lower-tail certificate and its reported bound is one.

The frozen cap grid is

\[
\mathcal T
=
\{64,128,256,512,1024,2048,4096,8192,16384\}.
\]

## 6. Proposition: variance-envelope floor

For every \(\kappa\in(0,1]\) and \(T\in\mathcal T\),

\[
\boxed{80\le\nu_\kappa(T)\le81.}
\]

Indeed, \(T\ge64\) implies

\[
T^2/4\ge1024.
\]

Also,

\[
0\le\max\{0,1-\beta_\kappa(T)\}\le1,
\]

so

\[
80
\le
81-\max\{0,1-\beta_\kappa(T)\}^2
\le81.
\]

The minimum defining \(\nu_\kappa(T)\) is thus the minimum of 81, a quantity at least 1024, and a quantity in \([80,81]\). The conclusion is uniform in \(\kappa\).

## 7. Theorem: analytic linear lower-tail no-go

Let

\[
s\in\{4,8,16,32\},
\quad
\kappa\in(0,1],
\quad
T\in\mathcal T,
\quad
0<\varepsilon<1.
\]

If the cap is admissible, then \(0<x<1\) and \(\nu_\kappa(T)\ge80\). Hence

\[
0
<
\frac{s x^2}{2(\nu_\kappa(T)+x/3)}
<
\frac{s}{160}.
\]

Negating and exponentiating yields

\[
D_{s,\kappa}(\varepsilon,T)
>
e^{-s/160}.
\]

Since \(s\le32\),

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

If the cap is inadmissible, its reported bound is one, so the same conclusion holds. Since the cap grid is finite, its minimum is also strictly larger than \(e^{-0.2}\).

For

\[
\delta_{\mathrm{joint}}\in\{0.01,0.05,0.10\},
\]

the largest declared directed component probability is the optimistic value

\[
0.10/2=0.05.
\]

Because \(e^{-0.2}>0.05\), no final-compatible or optimistic allocation can close the candidate lower-tail bound.

The theorem status is `PROVED` for this explicit bound family. The deterministic Phase 1D verdict is

\[
\boxed{\texttt{STRONG LINEAR NO-GO}.}
\]

## 8. Directed gate correction

An eventual decision would need

\[
\sigma_a^2
\le
\frac{S_a^2}{1-\varepsilon_{a,-}},
\qquad
\sigma_0^2
\ge
\frac{S_0^2}{1+\varepsilon_{0,+}}.
\]

The candidate radius has the algebraic requirement

\[
\boxed{\varepsilon_{a,-}<1.}
\]

The baseline upper radius only needs to be finite and nonnegative. Requiring \(\varepsilon_{0,+}<1\) would be a practical-strength criterion, not an algebraic necessity.

The corresponding sufficient comparison would be

\[
S_a^2
\le
\frac{1-\varepsilon_{a,-}}{1+\varepsilon_{0,+}}
\frac{\ell_{\mathrm{paid}}}{\ell_0}S_0^2.
\]

Phase 1D does not implement this decision rule because its necessary candidate component already fails.

## 9. Confidence allocations

The public \(\delta_{\mathrm{joint}}\) refers to a future joint candidate--baseline decision.

The final-compatible directed linear probability is

\[
\delta_{\mathrm{linear}}^{\mathrm{final}}
=
\delta_{\mathrm{joint}}/4,
\]

reserving two quarters for a hypothetical canonical analysis. The optimistic necessary-condition allocation is

\[
\delta_{\mathrm{linear}}^{\mathrm{optimistic}}
=
\delta_{\mathrm{joint}}/2.
\]

These probabilities range from 0.0025 to 0.05. All are below \(e^{-0.2}\). No empirical result or grid point changes the verdict.

## 10. Minimal implementation

Add `src/rademacher_linear_projection_no_go.py` with:

```python
@dataclass(frozen=True)
class LinearLowerTailNoGoAudit:
    sample_size: int
    joint_delta: float
    allocation: str
    directed_component_delta: float
    cap_minimum: float
    variance_envelope_floor: float
    analytic_probability_floor: float
    maximum_declared_component_delta: float
    theorem_status: str
    verdict: str
```

Expose:

```python
linear_lower_tail_no_go_audit(
    sample_size: int,
    joint_delta: float,
    allocation: str = "final_compatible",
) -> LinearLowerTailNoGoAudit
```

`joint_delta` always means the joint probability. No public argument named only `delta` is allowed.

The module also provides validated pure functions for:

- \(p_\kappa(v)\);
- \(\beta_\kappa(T)\);
- \(\nu_\kappa(T)\);
- \(D_{s,\kappa}(\varepsilon,T)\);
- \(e^{-s/160}\);
- deterministic verdict construction.

Do not implement radius inversion, bisection, cap selection, bootstrap analysis, structural optimization, canonical concentration, a decision API, or an allocator.

Add `experiments/postprocess_rademacher_linear_projection_no_go_phase1d.py` and `tests/test_rademacher_linear_projection_no_go_phase1d.py`.

## 11. Frozen regression grids and artifacts

The theorem grid uses

\[
s\in\{4,8,16,32\},
\quad
\delta_{\mathrm{joint}}\in\{0.01,0.05,0.10\},
\quad
\text{allocation}\in\{\texttt{final\_compatible},\texttt{optimistic}\},
\]

for exactly 24 rows.

Its unique key is

\[
(s,\delta_{\mathrm{joint}},\text{allocation}).
\]

The cap-regression grid uses the four sample sizes, nine caps,

\[
\kappa\in\{1,0.5,0.25,0.1,1/\sqrt{500}\},
\]

and

\[
\varepsilon\in\{0.1,0.25,0.5,0.75,0.99\},
\]

for exactly 900 rows.

Its unique key is

\[
(s,T,\kappa,\varepsilon).
\]

Every row must satisfy \(80\le\nu_\kappa(T)\le81\). Each admissible row must satisfy \(D_{s,\kappa}>e^{-s/160}\); each inadmissible row reports one. Numerical calculations verify the proof but never select parameters.

Create:

- `results/rademacher_linear_projection_no_go_phase1d_manifest.csv`;
- `results/rademacher_linear_projection_no_go_phase1d_theorem_grid.csv`;
- `results/rademacher_linear_projection_no_go_phase1d_cap_regression.csv`;
- `results/rademacher_linear_projection_no_go_phase1d_verdict.csv`.

The manifest records the source DOI, frozen grids, allocation definitions, constants 81, 80, 64, and \(e^{-0.2}\), frozen-input and output checksums, `new_matvec_queries = 0`, and the route-specific interpretation. No bootstrap artifact is created.

## 12. Proof, report, and tests

Minimally extend `docs/proof_rademacher_sample_variance_confidence.md` with the imported theorem, normalization, factor identity, truncation envelope, variance-floor proposition, no-go theorem, directed gate correction, and limitation.

Create `reports/rademacher_linear_projection_no_go_phase1d.md` using:

- `PROVED` for the exact identities and no-go theorem;
- `PROVED BUT ROUTE-VACUOUS` for the frozen family at \(s\le32\);
- `OPEN` for squared-chaos small-ball or direct paired-difference theory;
- `NOT ATTEMPTED` for canonical concentration and online allocation.

Tests cover:

1. exhaustive small-dimensional verification of \(\sigma^2=2\lVert C\rVert_F^2\);
2. zero variance without normalization;
3. input validation;
4. dimensional-to-normalized tail substitution;
5. bounds in \([0,1]\) and tail monotonicity;
6. the factor identities;
7. the exponential envelope and bias integral;
8. the variance floor across all caps and \(\kappa\)-values;
9. strict probability-floor dominance for every admissible row;
10. the inadmissible-cap branch;
11. monotonic extension from \(s=32\) to smaller sizes;
12. both confidence allocations;
13. deterministic `STRONG LINEAR NO-GO` for all 24 theorem rows;
14. the candidate-versus-baseline gate distinction;
15. exact 24-row and 900-row schemas;
16. zero new matrix--vector queries;
17. unchanged historical checksums and Phase 1C compatibility;
18. the complete maintained suite and `git diff --check`.

## 13. Verification sequence

1. Compile new and modified Python files.
2. Run targeted proof, grid, schema, and checksum tests.
3. Run the complete maintained suite with the existing `readline` workaround.
4. Run a no-query smoke audit into `/private/tmp`.
5. Validate row counts, unique keys, strict inequalities, and verdict.
6. Run the production no-query audit.
7. Validate historical immutability.
8. Run the complete maintained suite again.
9. Run `git diff --check`.
10. Audit proof and report wording line by line.

## 14. Research stopping decision

Because the necessary candidate lower-tail component fails even under the optimistic allocation, stop before the canonical Hoeffding kernel. There is no Phase 1E canonical-kernel implementation in this branch.

The preferred future target is the direct common-probe risk difference

\[
\boxed{
\widehat\Delta_R
=
\binom{s}{2}^{-1}
\sum_{i<j}
\left[
\frac{(X_{a,i}-X_{a,j})^2}{2\ell_a}
-
\frac{(X_{0,i}-X_{0,j})^2}{2\ell_0}
\right].
}
\]

A future theorem would seek

\[
\Delta_R\le\widehat\Delta_R+C,
\]

so that \(\widehat\Delta_R+C\le0\) directly certifies improvement and can exploit common-probe covariance. A genuine squared-chaos small-ball theorem is the alternative. Neither is implemented in Phase 1D.

## 15. Documentation and fixed decisions

After implementation, update `CURRENT_STATE.md`, workspace `UROP_TRACKER.md`, and workspace `memory.md`. Read and reconcile the existing chat-derived and implementation history rather than replacing prior correct entries. Synchronize the tracker, memory, current state, proof note, and Phase 1D report into `/Users/chenyixin/Documents/Obsidian_Vault/Swati_Summer_Research`.

Keep `UROP_TRACKER.md`'s `Last updated` date synchronized with its newest running-log entry. Record touched files, theorem status, artifact row counts, test counts, checksums, and the next open question. Label Phase 1D as continuation research and preserve the completed UROP estimator.

Fixed decisions:

- imported result: Cortinovis--Kressner Theorem 2, equation (8);
- fourth-moment factor: 81;
- minimum frozen cap: 64;
- joint probabilities: \(0.01,0.05,0.10\);
- final-compatible probability: \(\delta_{\mathrm{joint}}/4\);
- optimistic probability: \(\delta_{\mathrm{joint}}/2\);
- analytic verdict cannot be changed by empirical outcomes;
- baseline upper radius below one is not algebraically required;
- `STRONG LINEAR NO-GO` is route-specific;
- no new matrix--vector query, canonical theorem, direct-difference theorem, or allocator is implemented.
