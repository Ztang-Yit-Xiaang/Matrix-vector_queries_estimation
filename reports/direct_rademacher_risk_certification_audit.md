# Direct Rademacher Risk Certification: Mathematical Feasibility Audit

**Status:** promising future-project direction; offline estimator identities are proved, while an end-to-end baseline-safe online policy remains open.

## 1. Narrow research question

The final rank-deficient bridge supports the following continuation question:

$$
\boxed{
\text{Can realized Rademacher residual-risk differences be estimated from a small number of fresh matrix--vector queries with simultaneous finite-sample confidence?}
}
$$

This question targets the realized, probe-specific quantity selected by the final bridge rather than another spectral-tail model.

Let $A\in\mathbb R^{d\times d}$ be symmetric, let $Q\in\mathbb R^{d\times r}$ have orthonormal columns, and define

$$
R_Q=I-QQ^T,
\qquad
H_Q=R_QAR_Q.
$$

For a fresh Rademacher vector $g\in\{-1,+1\}^d$, define

$$
X_Q(g)=g^TH_Qg.
$$

Conditioned on a basis $Q$ fixed before $g$ is drawn,

$$
\mathbb E[X_Q(g)\mid Q]=\operatorname{tr}(H_Q)
$$

and

$$
\operatorname{Var}(X_Q(g)\mid Q)
=2\sum_{i\ne j}(H_Q)_{ij}^2
=2E_R(Q).
$$

If the eventual residual estimator averages $\ell$ fresh Rademacher quadratic forms, its exact conditional risk is

$$
\mathcal R_R(Q;\ell)=\frac{\operatorname{Var}(X_Q(g)\mid Q)}{\ell}
=\frac{2E_R(Q)}{\ell}.
$$

## 2. Sample-variance identity

Let $g_1,\ldots,g_s$ be fresh independent certification probes and set $X_j=X_Q(g_j)$. The ordinary unbiased sample variance

$$
S_Q^2=\frac{1}{s-1}\sum_{j=1}^s(X_j-\bar X)^2
$$

satisfies

$$
\boxed{
\mathbb E[S_Q^2\mid Q]=\operatorname{Var}(X_Q(g)\mid Q)=2E_R(Q).
}
$$

Consequently, for a denominator $\ell$ fixed before the certification probes,

$$
\widehat{\mathcal R}_R(Q;\ell)=\frac{S_Q^2}{\ell}
$$

is conditionally unbiased for $\mathcal R_R(Q;\ell)$.

The equivalent U-statistic identity is

$$
S_Q^2
=\frac{1}{s(s-1)}\sum_{1\le i<j\le s}(X_i-X_j)^2.
$$

For two bases $Q_a,Q_0$ and fixed positive denominators $\ell_a,\ell_0$, common certification probes give

$$
\widehat\Delta_R(a,0)
=\frac{S_a^2}{\ell_a}-\frac{S_0^2}{\ell_0},
$$

with

$$
\boxed{
\mathbb E[\widehat\Delta_R(a,0)\mid Q_a,Q_0]
=\mathcal R_R(Q_a;\ell_a)-\mathcal R_R(Q_0;\ell_0).
}
$$

The common probes can reduce comparison noise, but variance reduction is not automatic: it depends on the covariance induced by the two residual matrices.

## 3. Matrix-free common-probe evaluation

If $AQ$ is stored and one query supplies $Ag$, then

$$
R_Qg=g-Q(Q^Tg),
$$

and

$$
A(R_Qg)=Ag-AQ(Q^Tg).
$$

Thus the same queried vector $Ag$ can evaluate $g^TR_QAR_Qg$ for several bases that were all constructed before $g$ was drawn and whose $AQ$ products are already available. This is exact linear algebra and avoids dense residual matrices.

## 4. Major budget and timing correction

The denominator $m-q-r$ is correct for the original estimator only before certification queries are charged. If $s$ certification products are part of the same total budget and the action-specific construction cost is exactly $q+r$, then

$$
\ell_{\mathrm{final}}=m-q-r-s.
$$

More generally, if constructing several candidates has already committed $c_{\mathrm{pre}}$ oracle products, then spent queries cannot be refunded after selection:

$$
\boxed{
\ell_{\mathrm{final}}=m-c_{\mathrm{pre}}-s.
}
$$

Therefore:

1. Dividing by $m-q-r$ is valid for an **offline statistical-feasibility study with an external certification budget**.
2. A budget-faithful online emulation must deduct certification and all irrevocably committed construction queries.
3. A candidate basis must already exist before its realized risk can be certified. If extra directions are built first, their query cost is already sunk even if the candidate is rejected.
4. Directly certifying an already-realized basis does not by itself solve the prospective question of whether an unconstructed next direction is worth buying.

This timing issue prevents the sample-variance identity alone from closing the full baseline-safety gap. A policy must preserve a genuinely feasible fallback action at the time of certification.

## 5. Filtration requirement

Let $\mathcal G$ contain every random choice and oracle result used to construct the finite candidate set, all bases $Q_a$, all stored products $AQ_a$, and all final-risk denominators $\ell_a$. The certification probes must be independent of $\mathcal G$.

A simultaneous conditional statement of the form

$$
\Pr\!\left(
\forall a\in\mathcal A:
|\widehat\Delta_R(a,0)-\Delta_R(a,0)|\le C_s(a,0,\delta)
\,\middle|\,\mathcal G
\right)\ge1-\delta
$$

then yields unconditional validity by the tower property. If the candidate set or bases are changed using the certification probes, this argument no longer applies without a new adaptive-data-analysis proof.

Final residual probes must also be fresh and independent after the certification-based action is selected. Reusing certification probes in the final trace estimate is not covered by the existing conditional-unbiasedness theorem.

## 6. A proved fourth-moment feasibility lemma

Write

$$
Z_Q=X_Q-\mathbb E X_Q
=2\sum_{i<j}(H_Q)_{ij}g_ig_j.
$$

This is a degree-two multilinear Rademacher chaos. The Bonami--Beckner hypercontractive inequality gives

$$
\|Z_Q\|_4\le3\|Z_Q\|_2.
$$

If $\sigma_Q^2=\mathbb E Z_Q^2=2E_R(Q)$, then

$$
\mathbb E Z_Q^4\le81\sigma_Q^4.
$$

Take an independent copy $Z_Q'$ and define the disjoint-pair statistic

$$
W_Q=\frac{(X_Q-X_Q')^2}{2}
=\frac{(Z_Q-Z_Q')^2}{2}.
$$

Then

$$
\mathbb E W_Q=\sigma_Q^2,
$$

while independence and centering give

$$
\mathbb E(Z_Q-Z_Q')^4
=2\mathbb E Z_Q^4+6\sigma_Q^4
\le168\sigma_Q^4.
$$

Therefore

$$
\boxed{
\mathbb E W_Q^2\le42\sigma_Q^4,
\qquad
\operatorname{Var}(W_Q)\le41\sigma_Q^4.
}
$$

This scale-free relative second-moment bound is the clearest initial route to a finite-sample theorem. Applying a median-of-means construction to independent $W_Q$ pairs yields a relative-error estimate using order

$$
O\!\left(\varepsilon^{-2}\log\frac{|\mathcal A|}{\delta}\right)
$$

probe pairs, with conservative constants inherited from the bound above. A simultaneous multiplicative event can be converted into computable upper and lower risk bounds. For example, on

$$
(1-\varepsilon)\sigma_a^2
\le\widehat\sigma_a^2
\le(1+\varepsilon)\sigma_a^2,
$$

one has

$$
\frac{\widehat\sigma_a^2}{(1+\varepsilon)\ell_a}
\le\mathcal R_R(a)
\le
\frac{\widehat\sigma_a^2}{(1-\varepsilon)\ell_a}.
$$

Thus a sufficient baseline-safe acceptance rule is

$$
\boxed{
\frac{\widehat\sigma_a^2}{(1-\varepsilon)\ell_a}
\le
\frac{\widehat\sigma_0^2}{(1+\varepsilon)\ell_0}.
}
$$

This is a preliminary feasibility result, not yet an implementation-ready theorem: the exact median-of-means constants, zero-risk boundary, simultaneous allocation of failure probabilities, and budget-feasible online action set still require a complete proof and design.

## 7. Why Hanson--Wright is not yet a complete answer

Hanson--Wright controls the centered quadratic form using matrix scales such as $\|H_Q\|_F$ and $\|H_Q\|_2$. Those quantities are themselves unknown in the matrix-free setting, so the inequality does not automatically produce a computable data-dependent radius. It remains useful as a tail tool or as a route when valid deterministic norm upper bounds are available.

Similarly, $(X_i-X_j)^2$ is a fourth-degree object. Squaring a sub-exponential variable does not generally leave it sub-exponential, so a generic sample-variance Bernstein bound cannot be invoked without checking stronger tail assumptions. The disjoint-pair finite-variance route avoids that unsupported step. Concentration results for order-two U-statistics may later sharpen the all-pairs estimator, but they are not required for the first feasibility theorem.

Primary references for these routes include:

- Mark Rudelson and Roman Vershynin, [*Hanson--Wright inequality and sub-gaussian concentration*](https://doi.org/10.1214/ECP.v18-2865) (2013);
- Aline Bonami, [*Étude des coefficients de Fourier des fonctions de $L^p(G)$*](https://www.numdam.org/item/AIF_1970__20_2_335_0/) (1970);
- Christian Houdré and Patricia Reynaud-Bouret, [*Exponential Inequalities, with Constants, for U-statistics of Order Two*](https://hdl.handle.net/1853/31305) (2003).

## 8. Recommended two-stage program

### Phase 1A: offline statistical feasibility

Reconstruct the frozen bridge bases from their saved seeds without changing the estimator. Treat certification probes as an external diagnostic budget. For $s\in\{4,8,16,32\}$:

- compare ordinary sample variance, the all-pairs U-statistic, and an independent-pair median-of-means estimator;
- report absolute error universally and relative error only when the exact risk is positive;
- evaluate sign classification for pre-specified candidate-versus-baseline pairs;
- report false-safe acceptance, false rejection, and abstention separately;
- stratify by ordinary and rare catastrophic range-capture paths.

The central feasibility metric is false-safe acceptance, not average relative error.

### Phase 1B: budget-aware offline emulation

Freeze a precise online timing diagram. Deduct the certification probes and the maximum already-committed construction cost. Compare the certified improvement with the risk increase caused by spending $s$ queries on certification. This determines whether the statistical certificate is useful after its own cost is paid.

### Phase 2: confidence theorem

Prove a simultaneous conditional confidence result for a finite, pre-certification action set. The first recommended route is the disjoint-pair median-of-means construction above. Only after this theorem and a feasible fallback architecture are complete should it be combined with the existing baseline-safe decision theorem.

## 9. Final audit verdict

The direct realized-risk idea is **conditionally valid and scientifically well motivated**. The sample-variance and paired-difference identities are exact, and hypercontractivity supplies a promising finite-moment route to confidence bounds. However, it is not yet correct to say that this alone closes the UROP's baseline-safety gap. The remaining obstacle is jointly statistical and algorithmic:

$$
\boxed{
\text{construct simultaneous confidence bounds while charging certification and preserving a feasible fallback action.}
}
$$

This direction belongs to a future continuation. The frozen UROP estimator and completed bridge should remain unchanged.
