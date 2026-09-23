# Adaptive Hutch++: from query accounting to realized-risk certification

**Theorem–proof research report · September 17, 2026**  
UROP / randomized numerical linear algebra · Faculty advisor: Prof. Swati Padmanabhan

## Abstract

Adaptive Hutch++ asks how to divide a fixed matrix–vector query budget between constructing a low-rank approximation and estimating its residual trace. This report develops the mathematical distinctions that emerged while investigating that question. Attempted sketch width, accepted numerical rank, dominant-subspace capture, and residual estimator risk are different quantities. Their differences explain why accurate knee detection, a full-rank basis, or a favorable fitted spectrum need not justify an allocation decision.

We first establish exact rank-aware accounting, conditional unbiasedness, probe-specific conditional risk, and a realized marginal-risk criterion. We then connect subspace capture to residual energy and give an exact pilot-transcript construction showing how an important direction can remain invisible despite a sharp Ritz gap. The certification analysis proves unbiased variance estimation, its Hoeffding decomposition, conditional safe-acceptance implications, and the cost of an already-committed fallback. Two explicit small-budget confidence routes are valid but vacuous; this is a limitation of those routes, not a general impossibility theorem. Controlled empirical studies demonstrate useful fresh-probe information, rare-path sensitivity, and a substantial certification-cost tradeoff. They do not establish a uniformly superior or fully certified adaptive allocator.

The resulting research question is precise: **what observable, budget-feasible information can justify another low-rank query, or certify a realized risk difference, without sacrificing correctness or hiding the cost of fallback?**

## Reading map and evidence convention

Read Parts I–III for estimator correctness and the capture mechanism; Part IV for what happened after introducing direct Rademacher-risk estimation; Part V for confidence limits and pairing; Part VI for the recovered experiments and final research position. Every mathematical result below has its assumptions, statement, proof, and interpretation separated. Empirical findings have their own headings and are never presented as theorems.

Numbering **T1–T15 belongs only to this report**. It does not renumber legacy proof files, whose historical theorem numbers sometimes overlap. The [September 16 narrative](urop_research_report_20260916.md), [claim register](urop_claims_register_20260916.md), and original proof notes remain intact. The historical [August progress draft](urop_research_progress_report_aug2026.md), including its exported PDF/LaTeX/HTML, is preserved but is not the current correction record. No new PDF accompanies this revision.

![Structure A — proof dependencies and the missing confidence assumption](../figures/urop_structure_20260917/dependencies.png)

**Structure A.** Reading map, not a replacement for the assumptions in the theorems. The upper chain additionally requires orthonormal projectors, positive residual denominators, conditional isotropy, and the specified final-probe law. The lower chain deliberately does **not** infer valid confidence bounds merely from unbiased risk estimates. GPT-generated conceptual artwork was reviewed for these distinctions; it contains no experimental measurements.

Throughout, `PROVED` means a mathematical implication under its stated assumptions. `EMPIRICALLY ESTABLISHED` means supported on the specified finite experiment. `EMPIRICAL DECISION RULE` means a heuristic selection rule, not a confidence theorem. `OPEN` identifies a missing guarantee rather than silently filling it with a plausible mechanism.

---

## Part I. Definitions, accounting, and estimator correctness

### 1. Assumption and notation ledger

Let $A\in\mathbb R^{d\times d}$ be fixed and real symmetric. Positive semidefiniteness is needed for the ordered positive-eigenvalue models, but not for the trace and variance identities. Let $Q\in\mathbb R^{d\times r}$ have orthonormal columns. Empty $Q$ is permitted. Define

$$R_Q=I-QQ^\top,\qquad H_Q=R_QAR_Q.$$

These matrices are mathematical objects; production implementations apply the projectors implicitly rather than materializing dense residual matrices.

| Symbol | Meaning and distinction |
|---|---|
| $m$ | Total estimator query budget |
| $S$ | Range-sketch matrix; uppercase $S$ is reserved for this object |
| $q$ | Attempted sketch columns / products forming $AS$ |
| $r=r_q$ | Accepted basis rank; $r\le q$, not automatically $q$ |
| $k=r_\star$ | Dominant rank in a synthetic step model, not accepted rank |
| $B$ | Stopped reusable pilot width; $B\le q$ |
| $\ell$ | Fresh final residual-probe count |
| $s$ | Fresh certification-probe count, separate from $\ell$ |
| $c_{\rm pre}$ | Actual committed candidate-construction cost |
| $\mathcal F$ | Information fixing candidate actions before certification |
| $\mathcal G$ | Information fixing the selected action before final estimation, including certification if used |
| $\widehat\sigma_{x,s}^{,2}$ | Unbiased sample variance for action $x$; written $S_{x,s}^2$ in some legacy notes |

Conditioning on $\mathcal F$ or $\mathcal G$ freezes the corresponding actions and their residual denominators. Fresh probes are independent of that information and conditionally iid. A random pilot stage or allocation is allowed; freshness, rather than a deterministic stopping time, is the essential condition.

### T1. Exact construction accounting (`PROVED`, architectural statement)

**Assumptions.** Pilot products are retained inside the final $q$-column range sketch. An accepted $r$-column basis is used, and $AQ$ is queried once and cached. No additional oracle calls are omitted from the ledger.

**Statement.** The basic architecture has

$$\boxed{q+r+\ell=m,\qquad \ell=m-q-r>0.}$$

The special case $\ell=m-2q$ holds only when $r=q$. A rejected sketch direction costs one query even when it gains no rank.

**Proof.** A reusable pilot of width $B$ followed by $q-B$ new columns costs $B+(q-B)=q$, not $B+q$. Computing $AQ$ costs $r$ additional products, and final residual estimation costs $\ell$. Adding these disjoint costs gives the identity. There is no cancellation of an attempted query merely because QR rejects its direction. $\square$

**Interpretation.** A pilot is both information and an irreversible commitment: it restricts the eventual choice to $q\ge B$. For several candidates, $c_{\rm pre}=\max_x(q_x+r_x)$ is justified only when a nested, shared-prefix construction actually reuses all the required products. Otherwise use the actual committed query count. This qualification is essential, not an implementation detail.

![Structure B — estimator and certification workflows](../figures/urop_structure_20260917/workflow.png)

**Structure B.** Two different ledgers. The upper workflow is the cached-basis estimator in T1. The lower workflow spends on candidate construction and certification before final probes. Returning to a baseline basis does not undo either expenditure. Phase 1A deliberately treated certification as external evidence; Phase 1B charged it.

### T2. Conditional unbiasedness, including sequential decisions (`PROVED`)

**Assumptions.** $Q$ and the positive integer $\ell$ are $\mathcal G$-measurable. Each final probe satisfies $\mathbb E[g_jg_j^\top\mid\mathcal G]=I$. The requisite expectations exist.

**Statement.** For

$$\widehat t=\operatorname{tr}(Q^\top AQ)+\frac1\ell\sum_{j=1}^{\ell}g_j^\top H_Qg_j,$$

we have

$$\mathbb E[\widehat t\mid\mathcal G]=\operatorname{tr}(A),\qquad
\mathbb E\widehat t=\operatorname{tr}(A).$$

**Proof.** Conditional isotropy gives $\mathbb E[g_j^\top H_Qg_j\mid\mathcal G]=\operatorname{tr}(H_Q)$. Because $R_Q^2=R_Q$, cyclicity of trace yields

$$\operatorname{tr}(H_Q)=\operatorname{tr}(AR_Q)
=\operatorname{tr}(A)-\operatorname{tr}(Q^\top AQ).$$

Thus the two pieces of the estimator add in conditional expectation to $\operatorname{tr}(A)$. Taking another expectation proves the unconditional assertion. $\square$

**Interpretation.** The decision may use a stopped pilot or a certification batch. It must be completed before drawing the final probes. Reusing the selection probes as final probes invalidates this proof unless an additional argument restores conditional isotropy. Independence between final probes is needed for the variance formula below, but not for the expectation calculation itself. No optional-stopping theorem is invoked.

### T3. Exact conditional Gaussian and Rademacher risk (`PROVED`)

**Assumptions.** Those of T2, with conditionally independent final probes. They are either standard Gaussian or independent coordinate Rademacher signs. Define

$$E_G(Q)=\|H_Q\|_F^2,\qquad E_R(Q)=\sum_{i\ne j}(H_Q)_{ij}^2.$$

**Statement.** The conditional mean-squared errors are exactly

$$\boxed{\mathcal R_G(Q)=\frac{2E_G(Q)}\ell,
\qquad \mathcal R_R(Q)=\frac{2E_R(Q)}\ell.}$$

Their expectations over basis and decision randomness equal unconditional MSE.

**Proof.** For a Rademacher vector,

$$g^\top H_Qg-\operatorname{tr}(H_Q)
=2\sum_{i<j}(H_Q)_{ij}g_ig_j.$$

For distinct unordered pairs, the corresponding products have zero cross-expectation: some independent sign occurs an odd number of times. Each individual product has second moment one. Hence the single-probe variance is $4\sum_{i<j}(H_Q)_{ij}^2=2E_R(Q)$. For Gaussian probes, orthogonally diagonalize $H_Q$. Rotational invariance reduces the centered quadratic form to $\sum_i\lambda_i(z_i^2-1)$, whose independent terms have variances $2\lambda_i^2$. Its variance is therefore $2E_G(Q)$. Averaging $\ell$ independent terms divides either variance by $\ell$. T2 makes conditional bias zero. The law of total expectation applied to squared error then gives unconditional MSE. $\square$

**Interpretation and boundary cases.** Gaussian/Rademacher here describes the **final probes**, not the range sketch. For the same $Q$ and $\ell$, $E_R\le E_G$; different allocations or different bases cannot be compared by that inequality alone. A diagonal $H_Q$ has zero Rademacher variance even if it has positive Gaussian variance. A diagonal $A$ need not leave a diagonal $H_Q$ after randomized projection. These facts will matter in Part VI.

Sources: [sequential proof](../docs/proof_sequential_unbiasedness.md), [rank-aware risk proof](../docs/proof_rank_aware_risk.md), and [notation ledger](../docs/proof_notation.md).

---

## Part II. The marginal value of another query

### 2. Four objectives, not one interchangeable surrogate

For PSD $A$, order its eigenvalues as $\lambda_1\ge\cdots\ge\lambda_d\ge0$ and write $T(j)=\sum_{i>j}\lambda_i^2$. The bridge compares

$$\mathcal R_{\rm full}(q)=\frac{2T(q)}{m-2q},\qquad
\mathcal R_{\rm rank}(q)=\frac{2T(r_q)}{m-q-r_q},$$

$$\mathcal R_G(q)=\frac{2E_G(Q_q)}{m-q-r_q},\qquad
\mathcal R_R(q)=\frac{2E_R(Q_q)}{m-q-r_q}.$$

The first imagines one perfectly targeted direction per sketch query. The second corrects accepted rank but still assumes ideal targeting. The last two use the actual basis and are exact conditional risks for their respective final-probe laws.

![Structure C — four risk models](../figures/urop_structure_20260917/models.png)

**Structure C.** The arrows change assumptions; they are not a blanket chain of inequalities. The final two cards are exact conditional targets, while the first two remain ideal spectral models. In particular, replacing $q$ by $r_q$ fixes accounting but does not fix leakage.

### T4. Realized successful and failed marginal identities (`PROVED`)

**Assumptions.** Put $D=m-q-r$. A successful extension adds one attempted sketch and one accepted direction, yielding $(q+1,r+1,Q')$, and $D>2$. Use either $E=E_G$ or $E=E_R$ consistently.

**Statement.** Writing $E=E(Q)$ and $E'=E(Q')$,

$$\mathcal R'-\mathcal R
=-\frac{2\{D(E-E')-2E\}}{D(D-2)}.$$

If $E>0$, the extension is beneficial exactly when

$$\boxed{\frac{E-E'}E>\frac2D}
\quad\Longleftrightarrow\quad
\boxed{\frac{E'}E<\frac{D-2}D.}$$

For a failed-rank extension with **unchanged basis projector** and $D>1$,

$$\mathcal R'-\mathcal R=\frac{2E}{D(D-1)}\ge0.$$

**Proof.** For success, subtract $2E/D$ from $2E'/(D-2)$ and collect the numerator. Its denominator is positive, so the sign is the opposite of $D(E-E')-2E$. Division by $DE$ is permissible when $E>0$ and gives the equivalent inequalities. For failure, the unchanged projector keeps the numerator fixed; subtracting $2E/D$ from $2E/(D-1)$ gives the stated expression. $\square$

**Interpretation.** A successful direction costs two final samples: one sketch query and one $AQ$ query. Its fractional energy reduction must exceed $2/D$. A rejected direction costs one sample without changing the residual. If $E=0$, no nonnegative risk can strictly improve it; a failed direction remains at zero rather than being strictly harmful. These cases must not be hidden by adding a numerical floor.

For $m=160,q=r=k=15$, $D=130$. A successful extra direction pays for itself if $E'/E<128/130\approx0.984615$, or the energy drops by more than $1.53846\%$. The identity applies to each coupled path. A condition using ratios of empirical mean energies addresses a mean-risk comparison, not the fraction of individually improved paths.

**A useful caution.** Nested subspaces need not make Rademacher residual energy decrease. Let $A=I_2$, initially $Q$ empty. Then $E_R=0$. Adding $u=(1,1)^\top/\sqrt2$ leaves $H=I-uu^\top$ with two off-diagonal entries $-1/2$, so $E_R'=1/2$. Projection reduced total energy but created off-diagonal energy. Thus T4 does not assume an energy decrease; it tests whether one occurred and was large enough.

### Corollary T4.1. Ideal spectral marginal and step knee (`PROVED`)

**Assumptions.** Ideal accepted directions are leading eigenvectors. A successful extension captures $\lambda_{r+1}$. Then $E=T(r)$ and $E'=T(r)-\lambda_{r+1}^2$.

**Statement and proof.** Substitution into T4 gives

$$\boxed{(m-q-r)\lambda_{r+1}^2>2T(r).}$$

Only when $r=q$ does this reduce to $(m-2q)\lambda_{q+1}^2>2T(q)$. Define $M(q)=(m-2q)\lambda_{q+1}^2-2T(q)$. Direct subtraction gives

$$M(q+1)-M(q)=(m-2q-2)(\lambda_{q+2}^2-\lambda_{q+1}^2)\le0$$

on the feasible adjacent full-rank grid. Hence this ideal objective has a single-crossing marginal sign, with ties possible.

For the step spectrum $\lambda_{1:k}=1$, $\lambda_{k+1:d}=\eta$, with $0<\eta<1$, the pre-knee marginal numerator is $m-2k-2(d-k)\eta^2$, and the post-knee numerator is $\eta^2(m-2d)$. If

$$2k+2(d-k)\eta^2<m<2d$$

and the relevant allocations are feasible, risk decreases up to $k$ and increases after it. The unique ideal full-rank minimizer is $k$. At $\eta=0$, the post-knee risk can instead be identically zero; uniqueness must not be asserted. $\square$

The derivation proves a property of the ideal tail model. It does not prove that a randomized basis at $q=k$ captures the first $k$ eigenvectors. Sources: [marginal/regret proof](../docs/proof_near_oracle_regret.md) and [rank-aware extension](../docs/proof_rank_aware_risk.md).

---

## Part III. How rank and capture separate

### T5. Structured residual energy and the zero-oversampling graph (`PROVED`)

**Assumptions.** Let $A=\eta I+(1-\eta)U_1U_1^\top$, where $U_1$ has $k$ orthonormal columns and $0\le\eta<1$. Complete it to an orthogonal basis $[U_1,U_2]$. Define $Z_Q=R_QU_1$.

**Statement A: residual energy.** For any orthonormal $Q$,

$$H_Q=\eta R_Q+(1-\eta)Z_QZ_Q^\top,$$

$$\boxed{E_G(Q)=\eta^2(d-r)
+2\eta(1-\eta)\|Z_Q\|_F^2
+(1-\eta)^2\|Z_Q^\top Z_Q\|_F^2.}$$

**Proof.** Insert the structured $A$ into $R_QAR_Q$. In the squared Frobenius norm, $\|R_Q\|_F^2=d-r$, $R_QZ_Q=Z_Q$, and $\|Z_QZ_Q^\top\|_F^2=\|Z_Q^\top Z_Q\|_F^2$. Expanding yields the formula. Each term is nonnegative. $\square$

**Why this matters.** The formula exhibits leakage directly and avoids subtracting several order-one numbers to recover a tiny residual energy. Tiny positive tails are not canonicalized to exact zero. The $\eta=0$ zero-risk treatment in the frozen implementation also requires its stated capture/backward-residual checks.

**Statement B: square signal block.** Let $S_1=U_1^\top S$, $S_2=U_2^\top S$. If $q=k$ and $S_1$ is invertible, set $F=\eta S_2S_1^{-1}$. For $Q$ spanning $AS$,

$$\boxed{\|R_QU_1\|_2=\frac{\|F\|_2}{\sqrt{1+\|F\|_2^2}}.}$$

**Proof.** In the eigenbasis, $AS$ is represented by $[S_1^\top,\eta S_2^\top]^\top$. Multiplication on the right by $S_1^{-1}$ does not change its range, which is therefore the graph of $F$, spanned by $[I,F^\top]^\top$. An orthonormal basis is

$$\begin{bmatrix}I\\F\end{bmatrix}(I+F^\top F)^{-1/2}.$$

Its overlap with the signal coordinate space has singular values $(1+\sigma_i(F)^2)^{-1/2}$. These are the cosines of the principal angles, so the largest sine is exactly the displayed expression. $\square$

**Interpretation.** Full rank of $S_1$ does not imply a small inverse. Poor conditioning can amplify a nonzero tail and create leakage, which Statement A converts to residual energy. A large inverse alone is not sufficient evidence: the product $S_2S_1^{-1}$, not $S_1^{-1}$ in isolation, determines this graph error.

For $q>k$ and full row rank $S_1$, the range contains the graph with factor $\eta S_2S_1^\dagger$. It may contain additional useful directions, so the square-case equality becomes a capture bound, not generally an equality for the whole range. On a nested path, appending a signal column $v$ changes the Gram matrix to $S_1S_1^\top+vv^\top$. The minimum eigenvalue cannot decrease; thus the full-row-rank pseudoinverse norm cannot increase. Neither strict improvement nor good conditioning is guaranteed. Coordinate-Rademacher $S$ also does not make the rotated block $U_1^\top S$ iid Gaussian or iid Rademacher.

### Empirical finding E1. Why the bridge preferred one extra column

The frozen rank-deficient bridge used $d=500$, $k\in\{5,15,30\}$, $\eta\in\{0,10^{-14},10^{-10},10^{-6}\}$, budgets $m\in\{80,160,240\}$, and 200 paths per setup. Matrix-orientation seeds were 52000–52002, one per rank and reused across tail levels; basis seeds were 70000–70199. Range sketches were nested coordinate-Rademacher prefixes. The frozen rank-aware QR used relative tolerance $10^{-12}$ and absolute tolerance zero, relative to its sampled-column reference scale—not a universal eigenvalue threshold.

The allocation grids ended at 36, 76, and 116 for the respective budgets, retaining a residual floor of eight in the full-rank planning grid. There were 554,400 trial-allocation rows. At $\eta=10^{-6}$, **all 138,600 individual rows** satisfied $r_q=q$, not merely an average-rank statement.

Here $q_X^\star$ denotes the chosen minimizer of the finite empirical mean risk curve on the tested allocation grid, using the frozen plateau/tie convention; it is not an individual path's minimizer or the unknown population optimum. In all nine rank–budget combinations at that tail level, the recorded minima were

$$q_{\rm rank}^\star=k,\qquad q_G^\star=q_R^\star=k+1.$$

At $m=160$, the reproduced means were:

| $k$ | Gaussian risk at $k$ | Gaussian risk at $k+1$ | Rademacher risk at $k$ | Rademacher risk at $k+1$ |
|---:|---:|---:|---:|---:|
| 5 | $6.5616\times10^{-11}$ | $6.6759\times10^{-12}$ | $5.8695\times10^{-11}$ | $7.9857\times10^{-14}$ |
| 15 | $1.2123\times10^{-8}$ | $7.5641\times10^{-12}$ | $1.2059\times10^{-8}$ | $2.4140\times10^{-13}$ |
| 30 | $1.5521\times10^{-11}$ | $9.5731\times10^{-12}$ | $6.5597\times10^{-12}$ | $5.9134\times10^{-13}$ |

The denominator thresholds $(D-2)/D$ are respectively $148/150$, $128/130$, and $98/100$. They require energy reductions of only $1.3333\%$, $1.5385\%$, and $2\%$. To compare means correctly, multiply each mean-risk ratio by $(D-2)/D$ to obtain the corresponding ratio of mean energies. The large reductions in the table are not explained by the denominator; the denominator penalizes the extra direction.

The [mechanism audit](q_rank_vs_realized_risk_mechanism.md) connected poorly conditioned near-square signal blocks, principal-angle errors, and tail-dominated conditional risk along nested paths. However, median behavior often preferred $k$, Gaussian bootstrap intervals for the mean difference at $m=160$ included zero, and sensitivity experiments produced shifts of zero, one, two, or more. Rare one- or two-path failures can dominate a finite mean. The strongest conclusion is therefore **modest oversampling can protect against rare capture failures; exactly one extra column is not a universal optimum**.

### T6. A missing signal direction can be invisible to the pilot (`PROVED`, local construction)

**Assumptions.** $A$ is symmetric, $w$ is a unit vector with $Aw=w$, and $S^\top w=0$. Let $Q$ span $AS$. For the PSD step model, $w$ is one of its unit-eigenvalue directions and $0\le\eta<1$.

**Statement.** In exact arithmetic,

$$Q^\top w=0,\quad R_Qw=w,\quad H_Qw=w.$$

Moreover, $A'=A-(1-\eta)ww^\top$ replaces this eigenvalue by $\eta$ and has the same observed products:

$$A'S=AS,\qquad A'Q=AQ.$$

**Proof.** Symmetry gives $(AS)^\top w=S^\top Aw=S^\top w=0$. Thus $w$ is orthogonal to the sampled range and to $Q$. The first three identities follow by projecting $Aw=w$. In the two product differences, $ww^\top S=0$ and $ww^\top Q=0$, proving transcript equality. In the step model, modifying one eigenvalue in its own orthogonal eigendirection leaves the other eigenvalues unchanged and preserves PSD. $\square$

**Interpretation.** A deterministic pilot decision based only on those products cannot distinguish these two matrices on this transcript. This is a local identifiability limitation, not a distribution-level impossibility theorem: the constructed alternative can depend on the realized sketch. It leaves room for probabilistic assumptions or additional fresh queries. The observed positive-oversampling example in Part VI is different from the full-row-rank conditioning mechanism of T5.

---

## Part IV. Direct risk estimation and what safety would require

### 3. The new target and the common-probe workflow

Condition on $\mathcal F$, fixing candidate $a$ and baseline $0$. For fresh Rademacher certification probes, put

$$X_{x,j}=g_j^\top H_xg_j,\qquad
\sigma_x^2=\operatorname{Var}(X_{x,j}\mid\mathcal F)=2E_R(Q_x),\quad x\in\{a,0\}.$$

The target is $\mathcal R_x=\sigma_x^2/\ell_x$. Estimating a fitted decay exponent is no longer necessary: one can estimate the variance of the actual residual quadratic form.

With cached $AQ_x$, one product $Ag$ serves every action:

$$h_x=g-Q_x(Q_x^\top g),\qquad
Ah_x=Ag-AQ_x(Q_x^\top g),\qquad X_x=h_x^\top Ah_x.$$

The cache must exist before certification and be charged to construction. A batch of 32 products $Ag_j$ costs 32 certification queries, not 32 times the action count. These observations remain external evidence in Phase 1A; they are not reused as final estimator probes.

### T7. Exact sample-variance moments and Hoeffding decomposition (`PROVED`)

**Assumptions.** Conditional on the frozen action, $X_1,\ldots,X_s$ are iid, $s\ge2$, with mean $\mu$, variance $\sigma^2$, and finite fourth central moment $\mu_4$. Define

$$\widehat\sigma_s^{,2}=\frac1{s-1}\sum_{i=1}^s(X_i-\bar X)^2.$$

**Statement.** Its expectation and variance are

$$\mathbb E\widehat\sigma_s^{,2}=\sigma^2,\qquad
\operatorname{Var}(\widehat\sigma_s^{,2})
=\frac1s\left[\mu_4-\frac{s-3}{s-1}\sigma^4\right].$$

Here and below moments are conditional when the action is random. Its U-statistic representation and decomposition are

$$\widehat\sigma_s^{,2}=\binom{s}{2}^{-1}\sum_{i<j}\frac{(X_i-X_j)^2}{2}
=\frac1{s(s-1)}\sum_{i<j}(X_i-X_j)^2,$$

$$h_1(x)=\frac{(x-\mu)^2-\sigma^2}{2},\qquad
h_2(x,y)=-(x-\mu)(y-\mu),$$

$$\widehat\sigma_s^{,2}-\sigma^2
=\frac2s\sum_i h_1(X_i)+\binom{s}{2}^{-1}\sum_{i<j}h_2(X_i,X_j).$$

**Proof of the moments.** Set $Z_i=X_i-\mu$, $M=\sum_iZ_i^2$, and $L=\sum_iZ_i$. Then $\widehat\sigma_s^{,2}=(M-L^2/s)/(s-1)$. Independence and centering give

$$\mathbb EM=s\sigma^2,\quad\mathbb EL^2=s\sigma^2,$$

$$\mathbb EM^2=s\mu_4+s(s-1)\sigma^4,$$

$$\mathbb EL^4=s\mu_4+3s(s-1)\sigma^4,\qquad
\mathbb E(ML^2)=\mathbb EM^2.$$

For the last identity, every cross term from $L^2-M$ contains at least one independent centered factor to its first power. Substitution proves unbiasedness and

$$\mathbb E(\widehat\sigma_s^{,2})^2
=\frac{\mu_4}s+\frac{s^2-2s+3}{s(s-1)}\sigma^4.$$

Subtract $\sigma^4$ to obtain the variance formula.

**Proof of the decomposition.** The deterministic identity $\sum_{i<j}(X_i-X_j)^2=s\sum_i(X_i-\bar X)^2$ gives the U-statistic formula. For $h(x,y)=(x-y)^2/2$, computing $\mathbb E[h(x,X')]-\sigma^2$ gives $h_1$. Subtracting $\sigma^2+h_1(x)+h_1(y)$ from $h(x,y)$ gives $h_2$. Summing over pairs yields the displayed coefficients. Finally $\mathbb E[h_2(X,X')\mid X]=0$ by centering, so only the second component is canonical/degenerate in general. $\square$

**Interpretation.** The linear projection is generally nonzero, although special distributions can make it vanish. A theorem for completely degenerate U-statistics cannot bound the full sample variance without also controlling this projection. If $\sigma^2=0$, then $X$ is constant almost surely and the sample variance is exactly zero; no division by $\sigma^2$ is needed.

### T8. Direct common-probe risk differences (`PROVED`)

**Assumptions.** Candidate and baseline are frozen before the common iid certification probes, and their denominators are fixed and positive.

**Statement.** The statistic

$$\widehat\Delta=\frac{\widehat\sigma_{a,s}^{,2}}{\ell_a}
-\frac{\widehat\sigma_{0,s}^{,2}}{\ell_0}$$

is unbiased for $\Delta=\sigma_a^2/\ell_a-\sigma_0^2/\ell_0$, and is itself a paired order-two U-statistic. For $Y_x=\widehat\sigma_{x,s}^{,2}/\ell_x$,

$$\operatorname{Var}(Y_a-Y_0)=\operatorname{Var}(Y_a)+\operatorname{Var}(Y_0)-2\operatorname{Cov}(Y_a,Y_0).$$

**Proof.** Apply T7 separately to each action and use linearity of expectation; independence between actions is unnecessary. Subtract their U-statistic kernels to obtain the paired representation. Expansion of the centered square gives the variance identity. $\square$

**Interpretation.** Pairing reduces variance relative to independent observations with the same marginals exactly when the covariance is positive. It is not guaranteed merely by sharing probes. The mathematical independent benchmark is the sum of the two marginal variances; empirical estimates of that sum remain estimates. A cyclic shift of the same repetitions is a shifted/decorrelated comparator, not an independent sample.

### T9. Simultaneous bounds imply safe acceptance (`PROVED`, conditional certificate)

**Assumptions.** On an event $E$ of probability at least $1-\delta_{\rm joint}$, every action in a fixed finite pre-certification candidate set satisfies $L_x\le\mathcal R_x\le U_x$. The baseline belongs to the set. Selection is measurable.

**Statement.** Any accepted action satisfying $U_a\le L_0$ has $\mathcal R_a\le\mathcal R_0$ on $E$. If no candidate passes and the baseline is still feasible with the same risk used in the bounds, choosing it preserves this comparison.

**Proof.** On the single simultaneous event, the selected action obeys

$$\mathcal R_a\le U_a\le L_0\le\mathcal R_0.$$

If the action is the unchanged feasible baseline, the inequality holds with equality. $\square$

**Why simultaneity matters.** The action is chosen using the estimates. Separate pointwise coverage statements without a joint event do not justify substituting a selected index. A union bound over candidates or preregistered stages supplies a valid joint event if the allocated failure probabilities sum to $\delta_{\rm joint}$. That also handles a random selected stage; it does not require an optional-stopping assertion.

**Directed variance form.** Suppose the available event states

$$\sigma_a^2\le\frac{\widehat\sigma_a^{,2}}{1-\varepsilon_{a,-}},\qquad
\sigma_0^2\ge\frac{\widehat\sigma_0^{,2}}{1+\varepsilon_{0,+}},$$

where $0\le\varepsilon_{a,-}<1$ and $0\le\varepsilon_{0,+}<\infty$. An accepted candidate is safe relative to the original baseline if

$$\boxed{\widehat\sigma_a^{,2}\le
\frac{1-\varepsilon_{a,-}}{1+\varepsilon_{0,+}}
\frac{\ell_{\rm paid}}{\ell_0}\widehat\sigma_0^{,2}.}$$

This follows by dividing the two valid bounds by their positive denominators and chaining inequalities. Only the candidate-underestimation radius must be below one algebraically. A very large baseline-overestimation radius is weak but not undefined.

The confidence statement bounds $\Pr(\text{accept and harmful})$ by $\delta_{\rm joint}$. It does **not** automatically bound $\Pr(\text{harmful}\mid\text{accept})$ by the same number. Conditioning on a possibly rare acceptance event changes the probability statement. Nor does accepted-action safety prove whole-policy safety when abstention has a cost.

### T10. Paid fallback is not the original baseline (`PROVED`)

**Assumptions.** Original baseline construction costs $c_0$, whereas candidate-first construction costs $c_{\rm pre}$ and certification costs $s$. Both residual budgets are positive:

$$\ell_0=m-c_0,\qquad \ell_{\rm paid}=m-c_{\rm pre}-s.$$

**Statement.** Returning to exactly the baseline basis after paying extra costs changes its risk from $\sigma_0^2/\ell_0$ to $\sigma_0^2/\ell_{\rm paid}$. For $\sigma_0^2>0$, the ratio is

$$\boxed{\frac{\mathcal R_{\rm fallback}}{\mathcal R_{\rm original}}
=\frac{\ell_0}{\ell_{\rm paid}}.}$$

A candidate with positive baseline variance improves net risk exactly when

$$\frac{\sigma_a^2}{\sigma_0^2}<\frac{\ell_{\rm paid}}{\ell_0}.$$

**Proof.** T3 applies after all sunk decisions and costs are fixed. The returned basis has the same numerator, but fewer final probes whenever $c_{\rm pre}+s>c_0$. Taking the ratio proves the first identity; cross-multiplying positive denominators proves the second. $\square$

If the baseline risk is zero, ratios are undefined and no nonnegative-risk candidate strictly improves it. The equality case must be handled directly. Even an oracle that selects the smaller available numerator cannot refund construction or certification. This is why Phase 1A's useful signal and Phase 1B's useful policy are different questions. Source: [budget-aware proof](../docs/proof_budget_aware_certification.md).

### Empirical finding E2. Phase 1A found a useful signal, not a theorem

Phase 1A reconstructed 200 frozen paths for each of three ranks and two tail levels, with four batches of 50 certification repetitions and nested $s\in\{4,8,16,32\}$. The wide artifact has 960,000 rows. The same fresh certification probes served all actions, budgets, estimators, and both tail levels. These comparisons are paired, not independent experiments.

The estimators were ordinary sample variance, the independent-pair mean, and two medians of block means. For $W_j=(X_{2j-1}-X_{2j})^2/2$, $\mathbb EW_j=\sigma^2$; the sample variance and pair mean are unbiased families. `mom_w1` takes the median of individual $W_j$ values; `mom_w2` takes the median of two-$W$ block means. At $s=16$ they have eight and four blocks. Neither is generally unbiased. At $s=4$ both coincide with the pair mean, and at $s=8$ `mom_w2` does too; these degeneracies are not independent evidence.

The primary comparison was $q=k+1$ against $q=k$ at $m=160$, $\eta=10^{-6}$, $s=16$. The empirical guard used $\varepsilon=1/3$, so acceptance required an estimated candidate risk no more than half the estimated baseline risk. Rejection used the reverse ratio; equality or insufficient separation meant abstention. No finite-sample confidence event was attached to this rule.

| Sample-variance primary metric | Point estimate | Conditional 95% bootstrap interval |
|---|---:|---:|
| False-safe rate among truly worse paths | 0.0134% | [0.0028%, 0.0305%] |
| False-rejection rate among truly better paths | 0.5157% | [0.2991%, 0.7722%] |
| Top-5% true-better catastrophic detection | 79.6667% | [69.0662%, 89.2833%] |
| Acceptance among all truly better paths | 52.0991% | [40.3055%, 63.5069%] |

Rates average repetitions within each frozen path, eligible paths within rank, then ranks equally. False-safe denominators contain only truly worse paths; false-rejection denominators only truly better paths. Catastrophic sets are defined within rank from the exact baseline risk and restricted to truly better primary candidates. The 10,000-replicate bootstrap resamples eligible path clusters within rank, not individual repetitions. Empty required rank populations are unevaluable rather than silently discarded. Truth ties and undefined zero-risk relative errors are treated separately.

The benign $\eta=10^{-10}$ control false-safe estimate was 0.0008% with interval [0%, 0.0025%]. Sample variance passed the complete primary gate. The MoM false-safe point estimates were 4.6971% and 1.5478%; they did not pass. The preregistered verdict was **QUALIFIED GO**, not strong go. The pair mean remained an ablation rather than a verdict-selecting method.

![Data figure 2 — empirical direct-risk signal](../figures/urop_validated_20260916/figure2_empirical_signal.png)

**Data figure 2.** The original data-rendered comparison is retained. Its small false-safe rate is an empirical operating rate on a frozen population, not theorem-level conditional safety. [Full caption, source tables, and interval definitions](urop_figure_guide_20260916.md); [Phase 1A report](direct_rademacher_risk_certification_phase1a.md).

### Empirical finding E3. Paying for certification changed the conclusion

In the primary Phase 1B emulation, the same selection information was charged against the total budget. Across 600 frozen paths, the equal-rank average of within-rank **ratios of mean risks** was 0.036329, with conditional bootstrap interval [0.014628, 0.639830]. The paid oracle was 0.035411 and the paid fallback 1.172197.

Nevertheless, **575 of 600 individual paths—95.83%—were harmed** relative to their original baseline. The pooled median pathwise ratio was 1.160714. The historical value 1.172197 described an equal-rank average of within-rank medians in the relevant summary, not that pooled median. Means of ratios, ratios of means, and averages of medians are distinct statistics.

![Data figure 3 — certification cost and rare-path insurance](../figures/urop_validated_20260916/figure3_certification_cost.png)

**Data figure 3.** A favorable aggregate mean can coexist with harm on most paths because a small set of catastrophic baselines dominates total risk. This is a form of rare-path protection, not uniform improvement. The baseline here is the preregistered adjacent action, not automatically Standard Hutch++ at $q=\lfloor m/3\rfloor$. The figure preserves every path in its empirical distribution. [Full caption and sources](urop_figure_guide_20260916.md); [Phase 1B report](direct_rademacher_risk_certification_phase1b_budget.md).

---

## Part V. Why the confidence attempts did not close the practical gap

### 4. Imported moment control, with its scope made explicit

For a multilinear polynomial of degree at most $h$ in independent uniform signs, hypercontractivity gives $\|P\|_4\le3^{h/2}\|P\|_2$. Taking $h=2$ yields the retained quadratic-chaos fourth-moment bound $\mu_4\le81\sigma^4$. Taking $h=4$ instead gives the fourth-moment factor $9^4=6561$. These degree-dependent constants cannot be interchanged. The imported inequality is stated in [O'Donnell's hypercontractivity lecture, Corollary 1.3](https://www.cs.cmu.edu/~odonnell/boolean-analysis/lecture16.pdf); the substitutions and applications below are the report's calculations.

### T11. The frozen 81-plus-Chebyshev certificate is budget-vacuous (`PROVED`)

**Assumptions.** Use T7 with $\mu_4\le81\sigma^4$, and require simultaneous two-action control with total failure probability $\delta_{\rm joint}$. Allocate $\delta_{\rm joint}/2$ to each action. Handle zero variance directly as in T7.

**Statement.** A valid common relative-deviation radius is

$$\varepsilon_{\rm Ch}(s,\delta_{\rm joint})
=\sqrt{\frac{2[80+2/(s-1)]}{s\delta_{\rm joint}}}.$$

At $\delta_{\rm joint}=0.05$, it exceeds one for every $2\le s\le32$.

**Proof.** T7 gives

$$\frac{\operatorname{Var}(\widehat\sigma_s^{,2})}{\sigma^4}
\le\frac{80+2/(s-1)}s.$$

Chebyshev at relative deviation $\varepsilon$ bounds the one-action failure probability by this quantity divided by $\varepsilon^2$. Equating it to $\delta_{\rm joint}/2$ yields the radius; the union bound supplies simultaneity without requiring action independence. Both $80/s$ and $2/[s(s-1)]$ decrease for $s\ge2$. Thus the smallest radius on the grid is at 32, where

$$\varepsilon_{\rm Ch}(32,0.05)
=\sqrt{\frac{2(80+2/31)}{1.6}}\approx10.00403>1.$$

All smaller sample sizes are also vacuous for a positive denominator $1-\varepsilon$. $\square$

**Interpretation.** The event is valid, but it cannot provide a finite multiplicative variance **upper** bound by dividing the observed variance by $1-\varepsilon$. This is an analytic conclusion, not a failed numerical search. It neither disproves Phase 1A's observed signal nor establishes impossibility for other certificates. The joint confidence API must expose `joint_delta`, with explicit per-action/component splits; an unlabelled `delta` is ambiguous.

### T12. The frozen truncation–Bernstein linear route is also vacuous (`PROVED`)

**Assumptions and imported theorem.** Let $C=H-\operatorname{diag}(H)$ be nonzero, symmetric, and zero diagonal. Then $Z=g^\top Cg$, $\sigma^2=2\|C\|_F^2$, and $\kappa=\|C\|_2/\|C\|_F\le1$. [Cortinovis–Kressner, Theorem 2, equation (8)](https://link.springer.com/article/10.1007/s10208-021-09525-9) bounds its tail by

$$\Pr(|Z|\ge t)\le2\exp\!\left[-\frac{t^2}{8\|C\|_F^2+8t\|C\|_2}\right].$$

Substitute $t=u\sigma$ and define $V=Z^2/\sigma^2$. Then $\mathbb EV=1$ and

$$\Pr(V\ge v)\le p_\kappa(v)
=\min\left\{1,2\exp\!\left[-\frac{v}{4+4\sqrt{2v}\,\kappa}\right]\right\}.$$

The zero-variance branch is deterministic and needs no normalization. The normalized nondegenerate Hoeffding component is exactly $s^{-1}\sum_i(V_i-1)$, since $2h_1/\sigma^2=V-1$.

**Truncation calculation.** For $T>0$, define $V^{(T)}=\min(V,T)$ and

$$a_T=(4\sqrt2\kappa+4/\sqrt T)^{-1},\qquad
\beta_\kappa(T)=4e^{-a_T\sqrt T}
\left(\frac{\sqrt T}{a_T}+\frac1{a_T^2}\right).$$

For $v\ge T$, the tail exponent is at least $a_T\sqrt v$. Therefore

$$\mathbb E(V-V^{(T)})=\int_T^\infty\Pr(V>v)\,dv
\le\int_T^\infty2e^{-a_T\sqrt v}\,dv=\beta_\kappa(T).$$

The last equality follows by substituting $z=\sqrt v$ and integrating $4ze^{-a_Tz}$. Hence $\mathbb EV^{(T)}\ge\max(0,1-\beta_\kappa(T))$. The retained fourth-moment and bounded-range controls give the capped variance envelope

$$\nu_\kappa(T)=\min\left\{81,\frac{T^2}{4},
81-\max(0,1-\beta_\kappa(T))^2\right\}.$$

The third term subtracts a lower bound on the squared mean from an upper bound on the second moment. For the lower-tail Bernstein bound, $\mathbb EV^{(T)}-V^{(T)}\le\mathbb EV^{(T)}\le1$, so its one-sided boundedness constant is one, not $T$.

**Statement.** In the frozen family $T\ge64$, whenever $0<\varepsilon<1$ and $x=\varepsilon-\beta_\kappa(T)>0$, the proposed lower-tail probability upper bound

$$D_{s,\kappa}(\varepsilon,T)
=\exp\!\left[-\frac{sx^2}{2(\nu_\kappa(T)+x/3)}\right]$$

satisfies

$$\boxed{D_{s,\kappa}(\varepsilon,T)>e^{-s/160}
\ge e^{-0.2}\approx0.81873\quad(s\le32).}$$

**Proof.** Since $T^2/4\ge1024$ and $0\le\max(0,1-\beta)\le1$, the minimum defining $\nu$ lies in $[80,81]$. Also $0<x<1$. The exponent's positive magnitude is strictly below $s/160$, because its numerator is below $s$ and its denominator exceeds 160. Exponentiating gives the strict floor. For $s\le32$, $e^{-s/160}\ge e^{-0.2}$. If $\beta\ge\varepsilon$, the cap provides no admissible positive deviation and cannot rescue the bound. Even the most permissive declared allocation is $0.10/2=0.05$, far below the floor. $\square$

**Interpretation.** This is a lower bound on a proposed **upper bound**, not a lower bound on the actual failure probability. The verdict is **STRONG LINEAR NO-GO for this frozen truncation family**. It is uniform over its caps, $\kappa$ values, sample sizes, and confidence allocations. It does not say that underestimation actually occurs with probability 0.81873, that a better small-ball argument is impossible, or that the canonical $h_2$ term alone resolves the full sample variance. The unnecessary canonical-term continuation was stopped rather than expanding an already-vacuous route.

Sources: [Phase 1C report](rademacher_sample_variance_confidence_phase1c.md), [Phase 1D report](rademacher_linear_projection_no_go_phase1d.md), and [sample-variance proof note](../docs/proof_rademacher_sample_variance_confidence.md). Later improvements to degree-two moments do not retroactively alter these frozen 81-based results.

### Empirical finding E4. Common-probe cancellation was strong on average, weaker on catastrophes

Phase 2A's equal-rank empirical ratio of paired variance to the plug-in independent benchmark was **0.151032**, with interval **[0.137069, 0.166735]**: approximately 84.9% lower. Yet catastrophic paths had mean pairing ratio **0.899565**, compared with **0.111635** for ordinary paths. The strongest cancellation therefore did not occur on the rare paths most relevant to the risk problem. This is evidence for analyzing the paired difference, not evidence that its confidence problem has been solved. Source: [Phase 2A](paired_rademacher_risk_difference_phase2a.md).

For disjoint pairs of common probes, write

$$D_j=\frac{(X_{a,2j-1}-X_{a,2j})^2}{2\ell_a}
-\frac{(X_{0,2j-1}-X_{0,2j})^2}{2\ell_0}.$$

These variables are conditionally iid across pairs, with $\mathbb ED_j=\Delta$. Their within-pair action dependence is retained. The centered observation is a degree-at-most-four polynomial in the pair's signs, so the general hypercontractive fourth-moment bound is 6561, not 81 or 15. The audited data-only scale constructions in [Phase 2B](paired_rademacher_difference_confidence_phase2b.md) remain **PROVED BUT BUDGET-VACUOUS** for $s\le32$.

### T13. A valid norm prior supplies a finite paired radius (`PROVED`, structural assumptions required)

**Assumptions.** Before certification, a genuine bound $\|H_0\|_F\le M_0$ is available, and candidate and baseline are nested so that $R_aR_0=R_a$. Denominators are fixed and positive. There are $n$ disjoint common-probe pairs. Use the retained quadratic-chaos fourth-moment bound 81.

**Statement.** For $\bar D=n^{-1}\sum_jD_j$, define

$$\bar v=164M_0^4\left(\frac1{\ell_a}+\frac1{\ell_0}\right)^2.$$

Then for $0<\delta<1$,

$$\Pr\left(\Delta\le\bar D+
\sqrt{\frac{\bar v}{n}\frac{1-\delta}{\delta}}\right)\ge1-\delta.$$

**Proof.** Nesting gives $H_a=R_aH_0R_a$, hence $\|H_a\|_F\le M_0$ and $\sigma_x^2\le2M_0^2$ for both actions. For $W=(X_1-X_2)^2/2$, centered iid expansion yields

$$\operatorname{Var}(W)=\frac{\mu_4+\sigma^4}{2}\le41\sigma^4.$$

Indeed $\mathbb E(Z_1-Z_2)^4=2\mu_4+6\sigma^4$; divide by four and subtract $\sigma^4$. The triangle inequality for centered $L^2$ norms now gives

$$\operatorname{Var}(D_j)\le41
\left(\frac{\sigma_a^2}{\ell_a}+\frac{\sigma_0^2}{\ell_0}\right)^2\le\bar v.$$

Independence across pairs gives $\operatorname{Var}(\bar D)\le\bar v/n$. Cantelli's one-sided inequality yields the displayed radius. To see its constant directly, for a centered variable $Y$ of variance $v$, Markov's inequality on $(Y+b)^2$ gives $\Pr(Y\ge t)\le(v+b^2)/(t+b)^2$. Minimizing over $b\ge0$ at $b=v/t$ gives $v/(v+t^2)$. Apply this to $Y=\Delta-\bar D$ and solve for failure probability $\delta$. $\square$

**Interpretation.** If $M_0=0$, both residuals vanish and the radius-zero conclusion is deterministic, avoiding the $t=0$ division in the auxiliary Cantelli calculation. The prior is an assumption, not a statistic magically supplied at no query cost. If its validity is itself probabilistic, that failure probability must enter the total confidence budget. A finite radius does not establish useful acceptance, solve prior estimation, or make abstention safe relative to an unstarted baseline. This is the qualified structural result recovered in [Phase 2C](structural_paired_difference_confidence_phase2c.md).

### T14. A corrected degree-two fourth moment (`PROVED`; not a degree-four replacement)

**Assumptions.** $C$ is real symmetric, nonzero, zero diagonal; $Z=g^\top Cg$ for coordinate Rademacher $g$. Put $\sigma^2=2\|C\|_F^2$ and $\kappa=\|C\|_2/\|C\|_F$.

**Statement.** The corrected identity and its consequence are

$$\mathbb EZ^4=3\sigma^4+48\operatorname{tr}(C^4)
-96\sum_i[(C^2)_{ii}]^2+32\sum_{i,j}C_{ij}^4,$$

$$\boxed{\mathbb EZ^4\le(3+12\kappa^2)\sigma^4\le15\sigma^4.}$$

**Proof.** Write $Z=2\sum_{i<j}C_{ij}g_ig_j$. In its fourth-power expansion only edge multisets with even degree at every vertex survive. They consist of one edge repeated four times, two distinct edges each repeated twice, or four distinct edges forming a four-cycle. Their multiplicities in the edge expansion are respectively 1, 6, and 24. Collecting these terms into matrix traces gives the displayed identity; the corrected coefficients 48, $-96$, and 32 retain the repeated-edge adjustments.

For more explicit bookkeeping, let $a=\sum_{i<j}C_{ij}^4$, let $b$ sum $C_e^2C_f^2$ over unordered distinct edge pairs, let $b_{\rm adj}$ restrict to pairs sharing a vertex, and let $c$ sum products over undirected four-cycles. The edge expansion is $16a+96b+384c$. Meanwhile $\sigma^4=16a+32b$, $\operatorname{tr}(C^4)=2a+4b_{\rm adj}+8c$, $\sum_i[(C^2)_{ii}]^2=2a+2b_{\rm adj}$, and $\sum_{ij}C_{ij}^4=2a$. Substitution verifies every coefficient.

Now $\sum_{ij}C_{ij}^4\le\sum_i(\sum_jC_{ij}^2)^2=\sum_i[(C^2)_{ii}]^2$, so the combined correction after the trace term is nonpositive. Also $\operatorname{tr}(C^4)\le\|C\|_2^2\|C\|_F^2$. Divide by $\sigma^4=4\|C\|_F^4$ to obtain $3+12\kappa^2$, and use $\kappa\le1$. $\square$

The same identity gives a lower bound $3-24\kappa^2$ on the normalized fourth moment, since $\sum_i[(C^2)_{ii}]^2\le\|C\|_2^2\|C\|_F^2$. Thus small $\kappa$ squeezes this quadratic-chaos kurtosis toward three. It does not make three a universal lower bound, and it says nothing directly about the kurtosis of the signed degree-four $D_j-\Delta$.

### T15. The exact scale-estimation boundary (`PROVED`, route-specific)

**Assumptions.** A scalar observation has variance $v>0$ and fourth central moment at most $Kv^2$, with $K\ge1$. Estimate its variance with $n\ge2$ iid observations and use the T7–Chebyshev relative radius at failure probability $0<\delta_{\rm scale}<1$.

**Statement.** That radius is below one exactly when

$$\boxed{\delta_{\rm scale}n(n-1)>(K-1)(n-1)+2.}$$

At $K=3$ and $\delta_{\rm scale}=0.05$, the first feasible integer is $n=42$. If each observation uses two probes, this requires $s=84$, not 82.

**Proof.** T7 bounds the relative sample-variance variance by $[(K-1)+2/(n-1)]/n$. Chebyshev's relative radius squared divides this by $\delta_{\rm scale}$. Requiring it below one and multiplying by the positive $n(n-1)\delta_{\rm scale}$ gives the inequality. At $n=41$, both sides equal 82; strict feasibility fails. At 42, the left side is 86.1 and the right side 84. The expression decreases with $n$, proving first feasibility at 42. $\square$

This boundary neither proves that a norm prior is necessary for every certificate nor licenses assuming $K=3$ for an arbitrary signed-pair observation. It diagnoses one explicit scale-estimation route. Sources for T13–T15: [structural proof note](../docs/proof_structural_paired_difference_confidence.md) and [the recovery correction](recovery_audit_20260914.md).

---

## Part VI. Empirical counterexamples, history, and the final claim

### Empirical finding E5. A sharp pilot knee missed an entire dominant direction

The recovered orientation experiment used $d=100,m=60$, six spectra, 30 Haar orientations plus a coordinate orientation, and ten paths per orientation. For the rank-five step spectrum with $\eta=0.001$, every gate triggered and selected $(q,r,\ell)=(8,8,44)$. Standard Hutch++ used $(20,20,20)$. The finite observed comparison was:

| Orientation group | Paths | Gated MSE | Standard MSE | Gated / Standard |
|---|---:|---:|---:|---:|
| Haar orientations | 300 | $2.839433\times10^{-7}$ | $1.672813\times10^{-6}$ | 0.169740 |
| Coordinate aligned | 10 | $3.303942\times10^{-3}$ | $7.209998\times10^{-7}$ | 4582.446 |

The coordinate result is not erased by the rotated gain. Coordinate-Rademacher sketches are not rotation invariant, and the exact Rademacher residual risk depends on coordinates too.

![Data figure 1 — a full-rank pilot with incomplete signal capture](../figures/urop_validated_20260916/figure1_capture_failure.png)

**Data figure 1.** All ten coordinate paths, selected Ritz spectra, and the exact failing signal sketch. The plotted risk is conditional risk, not one final-probe squared error. Trial 7 has signal-sketch rank four but accepted rank eight. [Full caption and data provenance](urop_figure_guide_20260916.md).

In the seed-2026 replay, rows three and four of the failing signal block are exact negatives. Rational elimination verifies rank four. Therefore $w=(e_3+e_4)/\sqrt2$ satisfies $S^\top w=0$ and $Aw=w$. T6 applies. Numerically, $\|Q^\top w\|_2\approx8.07\times10^{-14}$ and the largest signal-subspace angle is 90 degrees. The small tail supplies additional accepted directions, so full numerical rank eight does not imply five captured signal directions.

The observed Ritz matrix has four values near one and four near 0.001. Its gap is approximately 6.907721 and contrast approximately 69,077. A contrast threshold of five would not exclude it. The pilot observes an excellent but incomplete knee. The alternative $A'=A-(1-\eta)ww^\top$ from T6 has only four dominant eigenvalues and exactly the same $AS$ and $AQ$ observations.

The failing basis has $E_G\approx1.000091$, $E_R\approx0.500004$, and exact conditional Rademacher risk **0.02272744518**. Ordinary gated coordinate paths are near **$1.31\times10^{-7}$**. The failing path's observed squared error is 0.03303752, just 1.45364 times its own conditional risk. It contributes **99.9948% of total conditional risk** and **99.9942% of observed squared error** among the ten coordinate trials. Thus the large error is already encoded in the basis; it is not explained solely by an unlucky final probe.

The missed projector illustrates coordinate dependence: $g^\top ww^\top g=1+g_3g_4$, taking values zero and two. This illustration is not used to lower-bound the variance of a sum without checking covariance; the total energies above were computed separately. Plain Rademacher Hutchinson is exact on the original diagonal matrix, but a randomized residual projector can create off-diagonal variance.

This event has **positive oversampling $q-k=3$** and exact discrete signal-rank failure. It must not be conflated with the invertible-but-poorly-conditioned zero-oversampling mechanism. Both invalidate “correct numerical dimension implies adequate capture,” for different reasons.

The focused [coordinate audit](coordinate_gate_failure_audit_20260916.md) replayed 310 first-spectrum paths and recorded 620 Standard/gated basis diagnostics. Diagnostic work used 74,400 replay queries plus 2,000 independent energy-check queries, separately accounted from each 60-query estimator. The legacy `Gaussian_Hutch_pplus` comparator has a Gaussian range sketch but Rademacher final probes, and here uses $q=15$ rather than Standard's 20. It is not an all-Gaussian rotation-invariant control.

### Empirical finding E6. Additional gains remain exploratory

The matrix-free Hessian benchmark used a 4,254-parameter convolutional network on synthetic images. The model has ten outputs, but its labels are only zero and one. The Hessian need not be PSD; T2–T3 still apply to a symmetric operator, while PSD spectral interpretations require additional justification. The floating-point reference trace was approximately 31.581270, obtained using 4,254 reference HVPs outside the estimator budgets.

| Query budget | Standard Hutch++ MSE | Gated MSE |
|---:|---:|---:|
| 30 | 5.858498 | 5.334005 |
| 60 | 1.417330 | 0.841176 |
| 90 | 1.006494 | 0.570674 |
| 120 | 0.457086 | 0.370210 |

There were 30 trials per method/budget and 480 total across four methods. These point estimates show operator feasibility and gains on this model, not broad ML generalization, a certified safety guarantee, or a wall-clock advantage. The gain cannot be attributed solely to more residual samples because the basis changes too. Sources: [Hessian benchmark](pytorch_hessian_benchmark.md) and [recovery audit](recovery_audit_20260914.md).

The [gate-feature development study](gating_diagnostics_predictability_map.md) contains 23 configurations. Its thresholds were assessed on that same grid; fixed-$q$ empirical minima are noisy grid references, not exact conditional-risk oracles. The later contrast filter was not active in the recovered Haar/Hessian protocols, so those results do not validate that filter.

Earlier YearPredictionMSD and Wiki-Vote studies, including quoted effective ranks, crossover budgets, and reduction factors, remain historical. Those numerical claims have not received source-level reproduction in this consolidation and are not central evidence here. A final external-data claim needs its operator definition, probe law, accounting, replication unit, and uncertainty verified. TurboQuant and leverage-score work remain supporting RandNLA preparation rather than competing main UROP narratives.

### 5. Historical progression without deleting superseded ideas

| Stage | Question attempted | What survives after the audits |
|---|---|---|
| Spectral-decay fitting | Can an exponent determine the split? | A visible fit need not predict the unseen tail |
| Multimodel / soft allocation | Can model choice or shrinkage repair fitting? | Model accuracy and allocation risk are different objectives |
| Sequential pilot | Can a detected knee guide the split? | Fresh-probe unbiasedness holds, but pilot commitment costs remain |
| Symmetric / asymmetric guards | Can clipping preserve baseline robustness? | Fixed guards are heuristics; no uniform improvement was established |
| Ideal and realized risk bridges | What objective does allocation actually minimize? | Rank correction alone does not account for subspace leakage |
| Zero-oversampling audit | Why did the finite mean prefer $k+1$? | Rare conditioning failures support oversampling, not a universal +1 rule |
| Phase 1A | Can fresh probes estimate the right risk? | Ordinary sample variance passed the empirical gate |
| Phase 1B | Is the information worth its queries? | Rare-path protection can coexist with harm on most paths |
| Phase 1C / 1D | Can explicit small-budget bounds certify it? | The audited routes are mathematically valid but vacuous |
| Phase 2A / 2B | Does common-probe cancellation help? | It often does empirically; unknown scale still obstructs the audited theorem |
| Phase 2C | Can structural prior information help? | A valid prior gives a conditional finite radius, not free prior knowledge |
| Exploratory two-stage gate | Can cheap pilot features avoid certification cost? | An exact missed direction can survive an extremely sharp visible knee |
| Recovery and consolidation | Which results can be trusted now? | Protocol reruns, corrected moments, checksums, and explicit claim status |

This progression is a sequence of tested distinctions, not a claim that each new method superseded the previous one successfully. Failed hypotheses narrowed the mathematical question.

### 6. Reproducibility and figure policy

September smoke tests overwrote some untracked production CSVs. Recovery reran the original protocols and validated replacement artifacts. Because original overwritten hashes were unavailable, this was **protocol-based recovery**, not byte-identical restoration. Output-directory isolation, checksum guards, and explicit query counters now protect the maintained experiments. The recovery report remains the authoritative explanation of that incident.

The preceding coordinate audit preserved 156 pre-existing source/result files and passed 225 maintained tests. The quantitative figure builder subsequently passed five focused tests and preserved its seven historical inputs. Those are prior-run records; this editorial revision does not claim to have rerun the full algorithm suite.

This edition makes **no estimator changes, no threshold changes, no experimental reruns, no new oracle queries, and no new bootstrap samples**. Its local mathematical regression checks test identities and boundary cases; they supplement rather than replace the proofs. The accompanying [editorial audit](theorem_report_audit_20260917.md) records the actual checks and preserved-file count for this revision.

Uncertainty follows the replication unit. Certification repetitions are clustered within frozen paths, and paths can share orientations. Existing intervals are conditional on that frozen population. They do not estimate unobserved rare-event probabilities or establish universal performance across orientations. No new confidence intervals are claimed for the exploratory Haar/Hessian tables.

The report contains three new GPT-generated **conceptual structure diagrams** and the three original **data-rendered quantitative figures**. Their roles are different. Exact assumptions and equations live in the text. Quantitative PNG/SVG plots retain source CSVs and checksums in the [validated figure guide](urop_figure_guide_20260916.md). The earlier [GPT-polished quantitative companions](urop_gpt_polished_figures_20260916.md) remain separate presentation assets, not replacements for numerical geometry. New conceptual prompts, edits, and diagram contracts are stored [with the diagrams](../figures/urop_structure_20260917/prompts_and_contract.md).

### 7. Final contribution and next research question

The strongest supported claim is not “Adaptive Hutch++ beats Standard Hutch++.” It is:

> Adaptive trace-estimation decisions require information about realized, probe-specific residual risk. Numerical rank, an ideal spectral tail, and a sharp visible Ritz knee each omit mechanisms that can matter decisively. The information used to repair those omissions must itself be paid for.

The established distinctions are concrete: attempted queries versus accepted rank; rank versus capture; spectral surrogate versus conditional risk; empirical discrimination versus a confidence theorem; and accepted-action safety versus a budget-safe complete policy. T1–T15 explain these distinctions algebraically, while E1–E6 locate their practical consequences without promoting finite experiments to universal laws.

The preferred continuation is an **observable, budget-feasible one-sided confidence statement for a common-probe realized risk difference**, or a fresh-query capture diagnostic with an explicit cost and uncertainty model. A usable architecture must preserve final-probe freshness, identify what information is known before selection, and keep its fallback physically feasible. A genuine small-ball analysis is an alternative mathematical route. Neither is implemented as a new allocator here.

For the UROP, the task is now report/poster assembly and review. Further algorithm development should have a separate frozen design rather than silently extending the completed experiment.

> **PROVED.** Under the explicit assumptions: accounting, unbiasedness, probe-specific risk, marginal identities, structured capture formulas, the identical-pilot-transcript construction, sample-variance moments and decomposition, simultaneous safe acceptance, paid-fallback cost, and the stated route-specific confidence results.

> **EMPIRICALLY ESTABLISHED.** On the recorded finite populations: rare-path sensitivity, instance-specific oversampling shifts, Phase 1A's sample-variance signal, Phase 1B's mean/pathwise tradeoff, common-probe variance reduction, the coordinate failure, and the exploratory rotated/Hessian point estimates.

> **STRONGLY SUPPORTED MECHANISM.** Near-square conditioning can produce leakage that dominates finite mean risk; exact discrete signal-rank loss can conceal an entire leading direction from a pilot. The latter has an exact transcript witness, while its population frequency and the broader conditioning–risk distribution remain outside that local proof.

> **OPEN / CONJECTURAL.** Uniformly useful small-budget observable certificates, a fully baseline-safe candidate-first policy with charged information costs, fixed-oversampling near-optimality, and generalization of the exploratory gate beyond its tested settings.

### 8. Source map and preservation of the original information

| September 16 narrative material | Location in this edition |
|---|---|
| Executive conclusion, notation, exact risk | Abstract; Part I; final contribution |
| Research history and local marginal decision | Part II; historical progression |
| Frozen bridge and the +1 mechanism | Part III, T5 and E1 |
| Phase 1A and 1B | Part IV, T7–T10 and E2–E3 |
| Negative confidence results | Part V, T11–T12 |
| Pairing and structural theory | Part V, E4 and T13–T15 |
| Exact coordinate gate failure | T6; Part VI, E5 |
| Hessian, development grid, older real data | Part VI, E6 |
| Recovery, uncertainty, figure status | Reproducibility and figure policy |
| Final claims and open direction | Final contribution and evidence boxes |

The [full earlier narrative](urop_research_report_20260916.md) is retained without edits; the revision expands its mathematics rather than deleting original content. Detailed historical entries remain in the [UROP tracker](../../../UROP_TRACKER.md) and [workspace memory](../../../memory.md). The [claims register](urop_claims_register_20260916.md) continues to identify which earlier claims were corrected, qualified, or left unverified.
