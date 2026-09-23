# Rank-Deficient Multi-Budget Risk Bridge: Results and Decision Gate

**Date:** 2026-08-15

**Classification:** `EMPIRICALLY OBSERVED` on the frozen matrices, orientations, seeds, budgets, and numerical-rank contract. Theorem 14, Corollary 14.1, and Lemma 14.2 remain `PROVED` under their stated assumptions.

## 1. Frozen experiment

The supplement uses

$$
d=500,
\qquad
r_\star\in\{5,15,30\},
\qquad
\eta\in\{0,10^{-14},10^{-10},10^{-6}\},
$$

with budgets

$$
m\in\{80,160,240\}.
$$

There are 200 randomized nested-basis trials per spectrum, 20,000 paired bootstrap resamples, and 554,400 trial-allocation rows. For each fixed $r_\star$, all four tail levels use exactly the same $U_\star$, trial-indexed basis seed, and Rademacher sketch prefix.

The five production artifacts have validated sizes:

| Artifact | Data rows |
|---|---:|
| Manifest | 36 |
| Trial paths | 554,400 |
| Mean curves | 2,772 |
| Minimizers | 144 |
| Bootstrap minimizer frequencies | 11,088 |

Every trial row is unique and finite and satisfies

$$
q+r_q+\ell_q=m.
$$

The actual diagnostic construction count is recorded separately as $q+r_q$. All 32 CSV files that existed before this cycle remained byte-identical.

## 2. Numerical-rank ladder

At the primary budget $m=160$:

- For $\eta=0$, every trial reaches $r_q=r_\star$ at $q=r_\star$ and then rejects every subsequent sketch column.
- For $\eta=10^{-14}$, every trial also has $r_{r_\star}=r_\star$. By $q=76$, a single extra tail direction is accepted in 22/200, 26/200, and 15/200 trials for $r_\star=5,15,30$, respectively. This confirms that `rtol=10^{-12}` is a projected-column/reference-scale rule, not a universal eigenvalue cutoff.
- For $\eta\in\{10^{-10},10^{-6}\}$, every production trial remains numerically full rank through $q=76$.

For the exact-rank cases at $q=76$, the rank efficiencies are

$$
\frac{5}{76}=0.0658,
\qquad
\frac{15}{76}=0.1974,
\qquad
\frac{30}{76}=0.3947.
$$

Thus the bridge directly exposes attempted sketch queries that do not purchase new accepted basis directions.

## 3. Rejected-query cost

When $\eta=0$, the full signal range is captured at $q=r_\star$ and every exact conditional risk is zero. Later rejected queries therefore have zero penalty, exactly matching the equality case in Corollary 14.1.

When $\eta=10^{-14}$, the tail is mathematically positive. Continuing from $q=r_\star$ to the budget-specific endpoint raises ideal rank-aware risk. At $m=160$, the relative increases are approximately

| $r_\star$ | Relative rank-risk increase at $q=76$ |
|---:|---:|
| 5 | 90.1% |
| 15 | 88.7% |
| 30 | 85.4% |

These increases are pathwise consequences of spending residual capacity after useful rank has essentially saturated. They do not make $q>r_\star$ optimal.

The multi-budget pointwise prediction is also visible. Holding the same state endpoint $q=36$ fixed, the mean cumulative ideal rejected-query penalties for $r_\star=5$ decrease from

$$
1.126\times10^{-27}
\quad(m=80)
$$

to

$$
1.717\times10^{-28}
\quad(m=160)
$$

and

$$
6.689\times10^{-29}
\quad(m=240).
$$

The same decreasing direction holds for $r_\star=15$ and $30$. This is a fixed-state comparison. At each budget's different maximum $q$, larger budgets permit more rejected attempts, so cumulative endpoint comparisons answer a different question.

## 4. Four-risk minimizers

Across all three budgets:

- Full-oracle and rank-aware-oracle risks are minimized at $q=r_\star$ in all 12 spectra.
- Realized Gaussian and Rademacher risks are minimized at $q=r_\star$ for $\eta\in\{0,10^{-14},10^{-10}\}$.
- For $\eta=10^{-6}$, both realized risks are minimized at $q=r_\star+1$ for every rank and budget.

At the primary budget:

| $r_\star$ | $q^*_{\rm full}$ | $q^*_{\rm rank}$ | $q^*_G$ at $\eta=10^{-6}$ | $q^*_R$ at $\eta=10^{-6}$ |
|---:|---:|---:|---:|---:|
| 5 | 5 | 5 | 6 | 6 |
| 15 | 15 | 15 | 16 | 16 |
| 30 | 30 | 30 | 31 | 31 |

The paired-bootstrap Gaussian minimizer intervals at $m=160$ are $[r_\star,r_\star+1]$ in these three cases. The Rademacher intervals are the point values $r_\star+1$.

## 5. Range-capture leakage remains decisive

The $\eta=10^{-6}$ paths satisfy $r_q=q$, so rank loss cannot explain their allocation discrepancy. Nevertheless, choosing the rank-oracle action $q=r_\star$ rather than the realized Gaussian minimizer $q=r_\star+1$ produces the following mean-risk penalties:

| $r_\star$ | $\mathcal R_G(r_\star)/\min_q\mathcal R_G(q)$ | Excess Gaussian risk |
|---:|---:|---:|
| 5 | 9.83 | 882.9% |
| 15 | 1,602.68 | 160,168% |
| 30 | 1.62 | 62.1% |

The mechanism is heavy-tailed randomized range capture. The extra direction improves Gaussian risk in only 5.5%, 7.0%, and 7.5% of individual trials for $r_\star=5,15,30$, but the worst trial-level risk ratios at $q=r_\star$ versus $r_\star+1$ are approximately $1.77\times10^3$, $3.20\times10^5$, and $1.11\times10^2$. Rare near-square range-finder failures dominate the mean conditional risk, which equals estimator MSE after averaging over bases.

This is the key result of the supplement:

$$
\boxed{
\text{Correct numerical rank does not imply adequate randomized subspace capture.}
}
$$

Consequently, replacing $T(q)$ by $T(r_q)$ is not sufficient to repair the allocation target.

### 5.1 Why the empirical optimum moves by one direction

The observed identity

$$
q_G^\star=q_{\mathrm{rank}}^\star+1=r_\star+1
$$

is an empirical result on the frozen instances, not a theorem that Gaussian Hutch++ universally needs one extra direction.

The ideal rank oracle acts as though an $r_\star$-dimensional basis captures the leading eigenspace exactly. If $U_1\in\mathbb R^{d\times r_\star}$ contains the leading eigenvectors, that oracle effectively compares against $Q=U_1$. Because the spectral tail after $r_\star$ is tiny, it stops at

$$
q_{\mathrm{rank}}^\star=r_\star.
$$

The randomized range finder does not receive $U_1$ directly. Write

$$
A
=
U_1\Lambda_1U_1^T
+
U_2\Lambda_2U_2^T
$$

and partition the Rademacher sketch $S$ in the eigenbasis as

$$
S_1=U_1^TS,
\qquad
S_2=U_2^TS.
$$

Then

$$
AS
=
U_1\Lambda_1S_1
+
U_2\Lambda_2S_2.
$$

At $q=r_\star$, the dominant block $S_1=U_1^TS$ is square. Having $r_q=q=r_\star$ proves that the complete sampled range $AS$ has the requested dimension; it does not prove that $S_1$ is well conditioned or even nonsingular. Although $S$ is coordinate Rademacher, the rotated block $U_1^TS$ is generally **not** an iid Rademacher matrix, so iid square-Rademacher singularity results do not apply directly. When $\eta>0$, the tail block can supply linear independence to $AS$ even when dominant-space recovery is poor.

When $S_1$ has full row rank, deterministic subspace-error expressions contain the amplification factor

$$
\Lambda_2S_2S_1^\dagger\Lambda_1^{-1}.
$$

Thus a tiny $\Lambda_2$ does not by itself force tiny subspace error. A small singular value of $S_1$ makes $\|S_1^\dagger\|_2$ large and can amplify the tail contamination. This is the precise sense in which

$$
\boxed{
\text{full rank does not imply good conditioning or good subspace capture.}
}
$$

At $q=r_\star+1$, the block $S_1\in\mathbb R^{r_\star\times(r_\star+1)}$ has one redundant column. That single oversampling direction does not guarantee good conditioning, but on the frozen trials it provides enough redundancy to suppress the rare catastrophic capture failures that dominate mean risk.

The exact cost--benefit condition is also small. For one realized nested path with $E_G(Q_{r_\star})>0$, the extra direction is beneficial exactly when

$$
\frac{E_G(Q_{r_\star+1})}{E_G(Q_{r_\star})}
<
\frac{m-2r_\star-2}{m-2r_\star}.
$$

For $(m,r_\star)=(160,15)$, the right-hand side is

$$
\frac{128}{130}\approx0.9846.
$$

Only a 1.54% residual-energy reduction is needed to compensate for losing two residual probes.

Finally, this inequality is a per-trial statement, whereas the reported $q_G^\star$ minimizes the empirical mean conditional risk over 200 randomized bases. In the $\eta=10^{-6}$ cases, moving to $r_\star+1$ improves only 5.5%--7.5% of individual Gaussian paths. The empirical mean nevertheless prefers the extra direction because it acts as insurance against a small number of extremely large losses at the square-sketch boundary. Therefore the correct interpretation is

$$
\boxed{
\text{one extra direction was sufficient on these instances to control heavy-tailed range-capture risk.}
}
$$

It is not a universal $+1$ allocation theorem.

## 6. Gaussian versus Rademacher risk

The Gaussian and Rademacher minimizers coincide in all 36 configurations in this supplement. However, their risk levels do not coincide. At their shared primary-budget minimizers, $\mathcal R_R/\mathcal R_G$ is approximately 1.0%--1.2%, 3.0%--3.2%, and 6.0%--6.2% for $r_\star=5,15,30$ across the positive tail levels.

That scale difference is conditional on the three fixed orientations and reflects the off-diagonal coordinate energy seen by Rademacher probes. This supplement alone does not require different minimizing actions, but the earlier 24-spectrum bridge already found materially different Gaussian and Rademacher optima on larger-tail step spectra. The combined evidence therefore does not justify a universal probe-agnostic certificate.

## 7. Multi-budget marginal direction

For the successful move $q=r_\star\to r_\star+1$ at $\eta=10^{-6}$, the mean realized marginal $M_G$ increases with $m$ for every rank. The fraction of trials with $M_G>0$ also rises modestly as the budget grows. This matches the pointwise theory: increasing $D=m-q-r$ makes a given energy drop easier to justify.

The global minimizing integers do not change across $m\in\{80,160,240\}$ in the frozen configurations. The pointwise budget theorem therefore survives, but it does not force a visible global minimizer shift on this grid.

## 8. Decision gate

The bridge answers the target-selection question:

$$
\boxed{
\text{Future certification should target realized, probe-specific conditional risk differences.}
}
$$

For the implemented Rademacher estimator, the primary target is

$$
\mathcal R_R(Q;q,r)
=
\frac{2E_R(Q)}{m-q-r},
\qquad
E_R(Q)=\sum_{i\ne j}(R_QAR_Q)_{ij}^2.
$$

Equivalently, a future policy should seek a simultaneous finite-sample bound for differences such as

$$
\Delta_R(a,a_0)
=
\mathcal R_R(a)-\mathcal R_R(a_0),
$$

or for the realized Rademacher marginal. $T(r)$ remains a useful ideal diagnostic, and $E_G(Q)$ remains the correct Gaussian target and a useful orientation-invariant control, but neither is the final certification target for the practical Rademacher estimator.

## 9. Scope stop

No new adaptive estimator is introduced. The acceptance-probability model

$$
p_b\Delta\mathcal R_{\rm success}
+
(1-p_b)\Delta\mathcal R_{\rm fail}
$$

remains future work. Rejected-query costs are real when positive tail risk remains, but the larger allocation failure in this supplement occurs even with $r_q=q$ and is caused by rare range-capture leakage. The UROP experimental scope should now freeze and the final report should be organized around this result.
