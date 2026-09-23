# Why the realized-risk optimum moves past the rank oracle

## A. Executive verdict

### Verdict: PARTIALLY VERIFIED

For the frozen step-spectrum bridge at \(\eta=10^{-6}\), the proposed mechanism is strongly supported and contains one exact mathematical link:

\[
\boxed{
\text{zero oversampling at }q=r_\star
\longrightarrow
\text{rare amplification by }S_1^{-1}
\longrightarrow
\text{poor dominant-subspace capture}
\longrightarrow
\text{large residual energy and risk}.
}
\]

The frozen calculation contains no numerical-rank bug. Every one of the 200 paths at every relevant prefix has \(r_q=q\), and the bridge uses

\[
\ell(q)=m-q-r_q=m-2q.
\]

At \(q=r_\star\), however, \(S_1=U_1^TS_q\) is square. It can be invertible while having a large pseudoinverse. For the step matrix, the exact graph factor is

\[
F=\eta S_2S_1^{-1}.
\]

It determines the dominant-subspace error exactly at this boundary. In the frozen paths, \(\|F\|_2\), principal-angle error, and realized risk are almost perfectly rank-correlated. The worst one or two paths dominate the empirical mean. Adding the nested \((r_\star+1)\)-st column makes \(S_1\) rectangular, raises its smallest singular value pathwise, and sharply suppresses those outliers.

The qualification is essential: the empirical *mean* prefers \(r_\star+1\), but the median path prefers \(r_\star\). Gaussian paired-bootstrap intervals at \(m=160\) include both minimizers. A broader sensitivity grid also produces shifts of 0, 1, 2, and larger. Thus the evidence supports **modest positive oversampling as tail-risk protection**, not a universal theorem that one extra column is always optimal.

## B. Definition audit

The definitions below come from the frozen implementation, not from notation alone.

- \(q\): the number of attempted range-sketch columns, hence \(q\) products \(As_j\). The range sketch \(S_q\in\{-1,+1\}^{d\times q}\) is coordinate Rademacher.
- \(r_q\): the number of columns accepted by the incremental rank-aware QR. Its cutoff is
  \[
  \max\{\texttt{atol},\texttt{rtol}\times\texttt{reference\_scale}\},
  \qquad \texttt{rtol}=10^{-12},\quad\texttt{atol}=0,
  \]
  where the reference scale is based on accumulated sampled-column norms. It is not an eigenvalue threshold.
- \(r_\star\): the dimension of the dominant eigenspace \(U_1\) in
  \[
  A=\eta I+(1-\eta)U_1U_1^T.
  \]
  Thus \(A\) has \(r_\star\) eigenvalues equal to 1 and \(d-r_\star\) eigenvalues equal to \(\eta\).
- \(q_{\rm rank}^\star\): the smallest member of the numerical minimum plateau of the 200-trial empirical-mean curve
  \[
  \mathcal R_{\rm rank}(q,r_q)=\frac{2T(r_q)}{m-q-r_q},
  \qquad T(r)=\sum_{j>r}\lambda_j^2.
  \]
- \(q_G^\star\): the corresponding smallest minimizer of the empirical mean of
  \[
  \mathcal R_G(Q_q;q,r_q)=\frac{2E_G(Q_q)}{m-q-r_q},
  \quad E_G(Q)=\|(I-QQ^T)A(I-QQ^T)\|_F^2.
  \]
- \(q_R^\star\): the corresponding smallest minimizer of the empirical mean of
  \[
  \mathcal R_R(Q_q;q,r_q)=\frac{2E_R(Q_q)}{m-q-r_q},
  \quad E_R(Q)=\sum_{i\ne j}[H_Q]_{ij}^2,
  \]
  with \(H_Q=(I-QQ^T)A(I-QQ^T)\).

The words Gaussian and Rademacher in the last two risks describe the *final residual-probe distribution*. Both use the same bases produced by a Rademacher **range** sketch.

The phenomenon occurs only for the frozen small-tail family \(\eta=10^{-6}\):

| quantity | frozen value |
|---|---:|
| dimension \(d\) | 500 |
| dominant ranks \(r_\star\) | 5, 15, 30 |
| budgets \(m\) | 80, 160, 240; primary 160 |
| range-sketch law | coordinate Rademacher |
| basis trials | 200 per setup |
| orientation seeds | 52000, 52001, 52002 |
| basis seeds | 70000 through 70199 |
| QR tolerance | relative \(10^{-12}\), absolute 0 |
| maximum \(q\) | 36, 76, 116 for \(m=80,160,240\) |
| residual floor | 8 |

For fixed \(r_\star\), the orientation is paired across \(\eta\), all prefixes use a common nested sketch, and budgets truncate the same path.

### Query accounting

The bridge constructs \(q\) columns of \(AS_q\), accepts \(r_q\) basis columns, and constructs \(AQ_q\) using another \(r_q\) matrix-vector products. Hence

\[
\text{construction cost}=q+r_q,
\qquad
\ell=m-q-r_q.
\]

All 138,600 saved rows at \(\eta=10^{-6}\), across all tested ranks, budgets, and prefixes, satisfy \(r_q=q\) individually. Therefore \(\ell=m-2q\) is exact for every relevant realization, not only in aggregate. The bridge is an offline diagnostic: it evaluates hypothetical residual risks analytically and does not issue dummy residual queries.

## C. Reproduction

The frozen minimizers reproduce exactly:

| \(m\) | \(r_\star\) | \(q_{\rm rank}^\star\) | \(q_G^\star\) | \(q_R^\star\) |
|---:|---:|---:|---:|---:|
| 80 | 5 | 5 | 6 | 6 |
| 80 | 15 | 15 | 16 | 16 |
| 80 | 30 | 30 | 31 | 31 |
| 160 | 5 | 5 | 6 | 6 |
| 160 | 15 | 15 | 16 | 16 |
| 160 | 30 | 30 | 31 | 31 |
| 240 | 5 | 5 | 6 | 6 |
| 240 | 15 | 15 | 16 | 16 |
| 240 | 30 | 30 | 31 | 31 |

At \(m=160\), the empirical mean and median risks around the knee are:

| \(r_\star\) | \(q\) | mean \(\mathcal R_G\) | median \(\mathcal R_G\) | mean \(\mathcal R_R\) | median \(\mathcal R_R\) |
|---:|---:|---:|---:|---:|---:|
| 5 | 3 | \(2.5974\times10^{-2}\) | \(2.5974\times10^{-2}\) | \(2.5776\times10^{-2}\) | \(2.5775\times10^{-2}\) |
| 5 | 4 | \(1.3158\times10^{-2}\) | \(1.3158\times10^{-2}\) | \(1.3083\times10^{-2}\) | \(1.3083\times10^{-2}\) |
| 5 | 5 | \(6.5616\times10^{-11}\) | \(6.6003\times10^{-12}\) | \(5.8695\times10^{-11}\) | \(6.5777\times10^{-14}\) |
| 5 | 6 | \(6.6759\times10^{-12}\) | \(6.6757\times10^{-12}\) | \(7.9857\times10^{-14}\) | \(7.9846\times10^{-14}\) |
| 5 | 7 | \(6.7535\times10^{-12}\) | \(6.7535\times10^{-12}\) | \(9.4256\times10^{-14}\) | \(9.4257\times10^{-14}\) |
| 15 | 13 | \(2.9851\times10^{-2}\) | \(2.9851\times10^{-2}\) | \(2.9613\times10^{-2}\) | \(2.9613\times10^{-2}\) |
| 15 | 14 | \(1.5152\times10^{-2}\) | \(1.5152\times10^{-2}\) | \(1.5060\times10^{-2}\) | \(1.5062\times10^{-2}\) |
| 15 | 15 | \(1.2123\times10^{-8}\) | \(7.4627\times10^{-12}\) | \(1.2059\times10^{-8}\) | \(2.2301\times10^{-13}\) |
| 15 | 16 | \(7.5641\times10^{-12}\) | \(7.5628\times10^{-12}\) | \(2.4140\times10^{-13}\) | \(2.4107\times10^{-13}\) |
| 15 | 17 | \(7.6669\times10^{-12}\) | \(7.6668\times10^{-12}\) | \(2.5967\times10^{-13}\) | \(2.5967\times10^{-13}\) |
| 30 | 28 | \(3.8462\times10^{-2}\) | \(3.8462\times10^{-2}\) | \(3.8156\times10^{-2}\) | \(3.8158\times10^{-2}\) |
| 30 | 29 | \(1.9608\times10^{-2}\) | \(1.9608\times10^{-2}\) | \(1.9490\times10^{-2}\) | \(1.9491\times10^{-2}\) |
| 30 | 30 | \(1.5521\times10^{-11}\) | \(9.4023\times10^{-12}\) | \(6.5597\times10^{-12}\) | \(5.6201\times10^{-13}\) |
| 30 | 31 | \(9.5731\times10^{-12}\) | \(9.5721\times10^{-12}\) | \(5.9134\times10^{-13}\) | \(5.9120\times10^{-13}\) |
| 30 | 32 | \(9.7507\times10^{-12}\) | \(9.7504\times10^{-12}\) | \(6.2166\times10^{-13}\) | \(6.2163\times10^{-13}\) |

The mean/median separation already signals that the \(+1\) result is driven by the upper tail.

![Risk distributions around the knee](../results/figures/q_rank_vs_realized_risk_mechanism/figure_1_risk_around_knee.png)

## D. Query-cost decomposition

Let \(k=r_\star\) and suppose \(r_q=q\). Moving from \(k\) to \(k+1\) changes the residual budget from

\[
D=m-2k
\quad\text{to}\quad
D-2=m-2k-2.
\]

For either exact conditional-risk energy \(E_X\), \(X\in\{G,R\}\),

\[
\mathcal R_X(k+1)<\mathcal R_X(k)
\iff
\frac{E_X(Q_{k+1})}{E_X(Q_k)}<\frac{D-2}{D}.
\]

Thus the fractional energy removed must exceed \(2/D\), the fractional residual-sample capacity lost.

| \(m\) | \(k=5\) threshold | \(k=15\) threshold | \(k=30\) threshold |
|---:|---:|---:|---:|
| 80 | 0.971429 | 0.960000 | 0.900000 |
| 160 | 0.986667 | 0.984615 | 0.980000 |
| 240 | 0.991304 | 0.990476 | 0.988889 |

At \(m=160\), only a 1.33%, 1.54%, or 2.00% energy reduction is required. But the *typical* path does not achieve it:

| \(k\) | energy | ratio of empirical means | median paired ratio | required ratio |
|---:|---|---:|---:|---:|
| 5 | Gaussian | 0.100385 | 0.997957 | 0.986667 |
| 15 | Gaussian | 0.000614 | 0.997855 | 0.984615 |
| 30 | Gaussian | 0.604446 | 0.997756 | 0.980000 |
| 5 | Rademacher | 0.001342 | 1.197620 | 0.986667 |
| 15 | Rademacher | 0.000020 | 1.064306 | 0.984615 |
| 30 | Rademacher | 0.088344 | 1.030965 | 0.980000 |

Only 5.5%, 7.0%, and 7.5% of Gaussian paths improve at \(k+1\); only 6%, 9%, and 10% of Rademacher paths improve. Yet those paths contain 90.49%, 99.94%, and 43.93% of the Gaussian risk at \(k\), and 99.89%, 99.998%, and 92.28% of the Rademacher risk. The ratio of empirical means is therefore enormous because rare paths are repaired, not because the usual path gets slightly better.

## E. Conditioning analysis

Write an orthogonal eigenspace decomposition

\[
A=U_1U_1^T+\eta U_2U_2^T,
\qquad
S_1=U_1^TS_q,
\qquad
S_2=U_2^TS_q.
\]

Then

\[
AS_q=U_1S_1+\eta U_2S_2.
\]

At \(q=k\), \(S_1\in\mathbb R^{k\times k}\). Full numerical rank of \(AS_q\) does **not** imply good conditioning of \(S_1\). Because \(\eta>0\), \(A\) is invertible, so \(AS_q\) can be full column rank even if the dominant block is nearly singular.

The relevant block is a projected sign matrix \(U_1^TS_q\), not an iid Rademacher matrix. Therefore iid-entry rectangular random-matrix theorems cannot be imported without additional work.

At \(m=160\), the conditioning diagnostics change as follows:

| \(k\) | quantity | \(q=k\) median | \(q=k\) 99th pct. | \(q=k\) max | \(q=k+1\) median | \(q=k+1\) 99th pct. | \(q=k+1\) max |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | \(\|S_1^\dagger\|_2\) | 4.19 | 169.79 | 1370.95 | 1.92 | 14.18 | 23.99 |
| 15 | \(\|S_1^\dagger\|_2\) | 8.09 | 275.02 | 5084.63 | 3.62 | 35.53 | 54.38 |
| 30 | \(\|S_1^\dagger\|_2\) | 9.99 | 337.37 | 698.98 | 5.06 | 25.37 | 30.97 |
| 5 | \(\|\eta S_2S_1^\dagger\|_2\) | \(9.30\!\times\!10^{-5}\) | \(3.83\!\times\!10^{-3}\) | \(3.07\!\times\!10^{-2}\) | \(4.25\!\times\!10^{-5}\) | \(3.15\!\times\!10^{-4}\) | \(5.15\!\times\!10^{-4}\) |
| 15 | \(\|\eta S_2S_1^\dagger\|_2\) | \(1.81\!\times\!10^{-4}\) | \(5.96\!\times\!10^{-3}\) | \(1.13\!\times\!10^{-1}\) | \(7.69\!\times\!10^{-5}\) | \(8.13\!\times\!10^{-4}\) | \(1.18\!\times\!10^{-3}\) |
| 30 | \(\|\eta S_2S_1^\dagger\|_2\) | \(2.24\!\times\!10^{-4}\) | \(7.61\!\times\!10^{-3}\) | \(1.51\!\times\!10^{-2}\) | \(1.10\!\times\!10^{-4}\) | \(5.69\!\times\!10^{-4}\) | \(6.29\!\times\!10^{-4}\) |

The nested extra column gives

\[
S_{1,k+1}S_{1,k+1}^T=S_{1,k}S_{1,k}^T+ss^T,
\]

so \(\sigma_{\min}(S_{1,k+1})\ge\sigma_{\min}(S_{1,k})\) pathwise and \(\|S_{1,k+1}^\dagger\|_2\le\|S_{1,k}^\dagger\|_2\) when both have full row rank. This monotonicity does not promise a strict or sufficient improvement, but it explains why the extra column can reduce fragility.

The full pre-QR condition number of \(AS_q\) is not the principal diagnostic at \(q=k+1\): the extra tail-scale direction can make its smallest singular value \(O(\eta)\) even while dominant-subspace capture improves.

![Conditioning factor versus risk](../results/figures/q_rank_vs_realized_risk_mechanism/figure_3_conditioning_vs_risk.png)

## F. Subspace-quality analysis

At the square boundary, if \(S_1\) is invertible,

\[
\operatorname{range}(AS_k)
=
\operatorname{range}\!\begin{bmatrix}I\\F\end{bmatrix},
\qquad
F=\eta S_2S_1^{-1}.
\]

This is an exact graph representation in the \([U_1,U_2]\) coordinates. Consequently,

\[
\tan\theta_i=\sigma_i(F),
\]

and in particular

\[
\|(I-Q_kQ_k^T)U_1\|_2
=
\frac{\|F\|_2}{\sqrt{1+\|F\|_2^2}},
\qquad
\sigma_{\min}(U_1^TQ_k)
=
\frac{1}{\sqrt{1+\|F\|_2^2}}.
\]

This proves the conditioning-to-angle arrow for the square setup, with one refinement: a large \(\|S_1^{-1}\|\) creates an amplification opportunity, but \(S_2\) must excite the bad direction. The complete factor \(\eta S_2S_1^{-1}\), not condition number alone, is decisive.

For \(q\ge k\), if \(S_1\) has full row rank,

\[
AS_qS_1^\dagger=U_1+\eta U_2S_2S_1^\dagger,
\]

so

\[
\|(I-Q_qQ_q^T)U_1\|_2
\le
\eta\|S_2S_1^\dagger\|_2.
\]

At \(q=k+1\) this is a bound rather than an exact graph identity because the range has one extra dimension.

The largest observed principal angles at \(q=k\) versus \(q=k+1\) are:

| \(k\) | maximum angle at \(k\) | maximum angle at \(k+1\) |
|---:|---:|---:|
| 5 | 1.756° | 0.0295° |
| 15 | 6.432° | 0.0675° |
| 30 | 0.867° | 0.0360° |

Both prefixes have \(r_q=q\) in every path. Thus Figure 2 directly separates numerical rank from geometric capture.

![Rank versus subspace quality](../results/figures/q_rank_vs_realized_risk_mechanism/figure_2_rank_vs_subspace_quality.png)

![Subspace error versus risk](../results/figures/q_rank_vs_realized_risk_mechanism/figure_4_subspace_error_vs_risk.png)

## G. Risk-mechanism analysis

Let

\[
Z=(I-Q_qQ_q^T)U_1.
\]

For the structured step matrix, the exact nonnegative Gaussian energy decomposition is

\[
\boxed{
E_G(Q_q)
=
\eta^2(d-q)
+2\eta(1-\eta)\|Z\|_F^2
+(1-\eta)^2\|Z^TZ\|_F^2.
}
\]

If \(\theta_i\) are the principal angles from \(U_1\) to the range of \(Q_q\), then

\[
E_G(Q_q)
=
T(q)
+2\eta(1-\eta)\sum_i\sin^2\theta_i
+(1-\eta)^2\sum_i\sin^4\theta_i.
\]

This is the exact subspace-error-to-Gaussian-energy arrow. It also explains how small angles can matter when the ideal tail is only \(O(\eta^2)\): the leakage terms can dominate that tiny baseline.

For Rademacher probes,

\[
E_R(Q_q)=E_G(Q_q)-\sum_i[H_Q]_{ii}^2.
\]

Therefore Gaussian energy controls the total Frobenius residual, while Rademacher energy removes its coordinate-diagonal part. Principal angles alone do not determine \(E_R\); coordinate orientation also matters. The Rademacher arrow is therefore established by exact evaluation plus empirical association, not by an angle-only theorem.

At \(q=k\), Spearman rank correlations across the 200 paths are:

| \(k\) | \(\|S_1^\dagger\|_2\) vs. \(\mathcal R_G\) | graph factor vs. \(\mathcal R_G\) | projection-Frobenius error vs. \(\mathcal R_G\) | graph factor vs. \(\mathcal R_R\) |
|---:|---:|---:|---:|---:|
| 5 | 0.9971 | 0.9969 | 1.0000 | 0.9977 |
| 15 | 0.9978 | 0.9977 | 1.0000 | 0.9985 |
| 30 | 0.9954 | 0.9965 | 1.0000 | 0.9975 |

All corresponding rank-correlation p-values are numerically tiny. These are descriptive associations, not independent causal experiments, but they align with the exact graph and energy identities.

## H. Catastrophic-path analysis

At \(m=160\), the upper tail at \(q=k\) contributes the following fractions of total empirical risk:

| \(k\) | risk | worst 1% | worst 5% | worst 10% | single worst path |
|---:|---|---:|---:|---:|---:|
| 5 | Gaussian | 89.98% | 90.44% | 90.95% | 89.91% |
| 15 | Gaussian | 99.94% | 99.94% | 99.94% | 99.94% |
| 30 | Gaussian | 37.05% | 42.38% | 45.47% | 34.33% |
| 5 | Rademacher | 99.83% | 99.89% | 99.90% | 99.81% |
| 15 | Rademacher | 99.997% | 99.998% | 99.998% | 99.995% |
| 30 | Rademacher | 84.97% | 91.77% | 92.28% | 79.41% |

For \(k=5,15,30\), the worst 1% have median \(\|S_1^\dagger\|_2\) of approximately 770, 2711, and 538, compared with 4.14, 8.01, and 9.85 among the remaining paths. Their median graph factors are about \(1.73\times10^{-2}\), \(6.00\times10^{-2}\), and \(1.15\times10^{-2}\), versus \(9.23\times10^{-5}\), \(1.79\times10^{-4}\), and \(2.22\times10^{-4}\).

Removing the worst 2% of \(q=k\) paths changes all three Gaussian minimizers back to \(k\). For Rademacher risk, removing the worst 2% suffices for \(k=5\), while 5% suffices for all three ranks. This is direct evidence that the empirical-mean \(+1\) optimum is tail insurance.

The original paired bootstrap agrees only partly. At \(m=160\), the 95% intervals for the Gaussian mean-risk difference all cross zero, while the Rademacher intervals are positive. These intervals are conditional on the one orientation and one observed 200-path empirical distribution; they do not measure the probability of unseen catastrophic paths under new orientations.

![Catastrophic-tail comparison](../results/figures/q_rank_vs_realized_risk_mechanism/figure_5_catastrophic_tail.png)

## I. Comparison with the ideal oracle and whether +1 is special

The rank oracle assumes the best \(r_q\)-dimensional eigenspace has been captured. At \(q=k\), its residual energy is

\[
T(k)=(d-k)\eta^2.
\]

It does not see the leakage terms in the exact Gaussian identity. Across the frozen paths, the ratio \(E_G(Q_k)/T(k)\) has medians near 1 but maxima of

| \(k\) | median \(E_G/T\) at \(k\) | maximum \(E_G/T\) at \(k\) | maximum \(E_G/T\) at \(k+1\) |
|---:|---:|---:|---:|
| 5 | 1.00004 | 1,787.66 | 1.00122 |
| 15 | 1.00016 | 324,734.52 | 1.00974 |
| 30 | 1.00025 | 113.37 | 1.00205 |

Thus the ideal tail is accurate for the typical path but misses rare enormous leakage at exactly the square boundary. The nested extra column almost completely closes this ideal/realized gap.

![Ideal tail versus realized energy](../results/figures/q_rank_vs_realized_risk_mechanism/figure_6_ideal_tail_vs_realized_energy.png)

### Sensitivity audit

An additional diagnostic, without changing the frozen estimator, used:

\[
d\in\{250,500\},\quad
k\in\{5,15,30\},\quad
\eta\in\{10^{-8},10^{-6},10^{-4},10^{-2}\},
\]

\[
m\in\{80,160,240\},
\]

with three new orientations/batches and 50 nested-sketch paths per batch. All tested paths in this diagnostic retained full numerical rank. The Gaussian shift distribution over 54 batch/budget/rank/dimension cases per \(\eta\) was:

| \(\eta\) | shift 0 | shift 1 | shift 2 | larger |
|---:|---:|---:|---:|---:|
| \(10^{-8}\) | 48 | 6 | 0 | 0 |
| \(10^{-6}\) | 31 | 23 | 0 | 0 |
| \(10^{-4}\) | 0 | 21 | 32 | 1 |
| \(10^{-2}\) | 0 | 0 | 0 | 54 |

For Rademacher risk the counts were:

| \(\eta\) | shift 0 | shift 1 | shift 2 | larger |
|---:|---:|---:|---:|---:|
| \(10^{-8}\) | 48 | 6 | 0 | 0 |
| \(10^{-6}\) | 16 | 38 | 0 | 0 |
| \(10^{-4}\) | 0 | 10 | 40 | 4 |
| \(10^{-2}\) | 0 | 0 | 0 | 54 |

The safe conclusion is option B: modest oversampling often helps near a sharp low-rank boundary, but the exact optimal amount depends on tail scale, budget, dimension, orientation, and probe-specific risk. One extra column is especially visible in the frozen \(\eta=10^{-6}\) regime; it is not structurally universal.

## J. Mathematical statement we can safely use

> **Mixed theoretical and empirical statement.** For the tested step matrices, exact numerical rank at \(q=r_\star\) does not guarantee accurate dominant-subspace recovery. At this zero-oversampling boundary, the square projected sketch block can be invertible yet poorly conditioned. The exact factor \(\eta S_2S_1^{-1}\) then controls principal-angle error, and the exact Gaussian-energy identity converts that error into residual risk. In the frozen \(\eta=10^{-6}\) experiment, rare amplified paths dominate the empirical mean. The nested \((r_\star+1)\)-st sketch column sharply suppresses these paths, so the empirical-mean realized-risk optimum moves from \(r_\star\) to \(r_\star+1\). The amount of useful oversampling is not universally one.

The standard randomized range-finder decomposition is consistent with the general framework in Halko, Martinsson, and Tropp, *Finding Structure with Randomness* ([SIAM Review DOI](https://doi.org/10.1137/090771806)). The identities used above were derived directly for this PSD step model and verified numerically against the reconstructed frozen paths; no iid theorem was assumed for \(U_1^TS\).

## Final theorem/conjecture separation

### PROVED

\[
\boxed{
\mathcal R_X(q+1)<\mathcal R_X(q)
\iff
\frac{E_X(Q_{q+1})}{E_X(Q_q)}<\frac{m-2q-2}{m-2q},
\quad X\in\{G,R\},
}
\]

provided \(r_q=q\), \(r_{q+1}=q+1\), the denominators are positive, and \(E_X(Q_q)>0\).

At \(q=k\) with invertible \(S_1\), the exact graph representation, principal-angle formulas, and Gaussian residual-energy decomposition above hold. Under nested columns, the smallest singular value of the full-row-rank dominant block cannot decrease.

### EMPIRICALLY ESTABLISHED

The frozen implementation is correctly accounted, has \(r_q=q\) pathwise, and reproduces \(q_{\rm rank}^\star=k\), \(q_G^\star=q_R^\star=k+1\) for all nine tested \((k,m)\) combinations at \(\eta=10^{-6}\). The mean is dominated by rare high-risk square-boundary paths; the median path prefers \(k\). The nested extra column sharply reduces conditioning, angle, and risk extremes.

### STRONGLY SUPPORTED MECHANISM

\[
\boxed{
\text{near-singular }S_1
\to
\text{large }\eta S_2S_1^{-1}
\to
\text{dominant-subspace leakage}
\to
\text{large }E_G\text{ and }E_R
\to
\text{empirical-mean preference for oversampling}.
}
\]

The first two arrows are exact for Gaussian geometry at \(q=k\); the Rademacher-risk arrow also depends on coordinate orientation and is supported empirically.

### OPEN / CONJECTURAL

No fixed \(+1\) rule is proved. It remains open to characterize the oversampling needed for near-optimal realized Gaussian or Rademacher risk under projected coordinate-Rademacher sketches, to establish orientation-uniform tail bounds, and to determine whether direct realized-risk certification can safely detect these rare failures with few additional queries.
