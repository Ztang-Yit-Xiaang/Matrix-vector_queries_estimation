# Common Notation for the Adaptive Hutch++ Proofs

This file is the notation ledger for all formal proof notes in this directory. A symbol should not be reused for a different mathematical object without an explicit local warning.

| Symbol | Meaning |
|---|---|
| $A\in\mathbb R^{d\times d}$ | target matrix; symmetry or positive semidefiniteness is stated when required |
| $m$ | total matrix-vector query budget |
| $b$ | deterministic pilot size at one candidate stage |
| $B$ | random stopped pilot size |
| $N_{\mathrm{stg}}$ | number of candidate pilot stages |
| $j$ and $J$ | deterministic and random stage indices, respectively |
| $q$ | final target sketch width, with $B\le q$ |
| $q_0$ | feasible Standard-Hutch++ baseline sketch width |
| $S\in\mathbb R^{d\times q}$ | sketching matrix; $S=[S_{\mathrm{pilot}},S_{\mathrm{extra}}]$ when pilot columns are reused |
| $Y=A S$ | range-finding sample matrix |
| $Q\in\mathbb R^{d\times r}$ | final orthonormal basis |
| $r=r_{\mathrm{actual}}$ | realized basis rank in the allocation and risk proofs |
| $r_\star$ | true signal/knee rank in the ideal step-spectrum proofs |
| $p=b-r_\star$ | number of post-knee oversampling columns in the step-spectrum proofs |
| $r_Y=\operatorname{rank}(Y)$ | sample-matrix rank in the step-spectrum proof |
| $p_Y=r_Y-r_\star$ | realized number of post-knee Ritz positions; $p_Y=p$ when $S$ has full column rank and $A\succ0$ |
| $s_i$ | singular values of $U^T Q$ in the principal-angle proof; lowercase $s_i$ is not the sketch matrix |
| $\ell=m-q-r$ | residual probe count |
| $P=QQ^T$ and $R=I-P$ | captured-subspace and residual orthogonal projectors |
| $\mathcal G$ | full pre-residual sigma-algebra |
| $g_1,\ldots,g_\ell$ | fresh residual probes |
| $\delta$ | probability-of-failure level |
| $\mathcal R(q)$ | target allocation risk; its exact meaning must be stated locally |
| $\widehat{\mathcal R}_b(q)$ | pilot-stage estimate of $\mathcal R(q)$ |
| $T(q)=\sum_{i=q+1}^d\lambda_i^2$ | exact squared spectral-tail energy in the full-rank oracle model |
| $M(q)=(m-2q)\lambda_{q+1}^2-2T(q)$ | full-rank marginal risk quantity; $M(q)>0$ means the move $q\to q+1$ lowers oracle risk |
| $\widehat M_b(q),C_b^M(q)$ | pilot estimate of $M(q)$ and a stated confidence radius; construction of a computable valid radius remains open |
| $e_b(q)$ | simultaneous confidence radius for $\widehat{\mathcal R}_b(q)$ |
| $U_b^R(q),L_b^R(q)$ | upper and lower confidence bounds for $\mathcal R(q)$ |
| $a=(q,r)$ and $a_0=(q_0,r_0)$ | rank-aware candidate and baseline actions |
| $D=m-q-r$ | residual-query denominator at a rank-aware state |
| $E_G(Q)=\|RAR\|_F^2$ | exact Gaussian residual energy for the realized basis |
| $E_R(Q)=\sum_{i\ne j}(RAR)_{ij}^2$ | exact Rademacher off-diagonal residual energy in the sampling coordinates |
| $M_X=D[E_X(Q)-E_X(Q')]-2E_X(Q)$ | realized successful-direction marginal for $X\in\{G,R\}$ |
| $(q,r,Q)\to(q+1,r+1,Q')$ | successful rank-aware transition; it consumes two residual-query slots |
| $(q,r,Q)\to(q+1,r,Q)$ | failed-rank transition; it consumes one residual-query slot |
| $\Delta_\alpha$ | exponential-slope estimation error; this is not a failure probability |
| $\gamma_{\mathrm{gap}}=\log\tau_{\mathrm R}$ | log Ritz-gap threshold |
| $\gamma_{\mathrm{shrink}}$ | soft-allocation shrinkage weight |
| $h$ | power/subspace-iteration depth in the optional power-iteration comparison |

## Reserved-symbol rule

Uppercase $S$ always denotes a sketching matrix. In particular, it is never used for a stopping-stage index. The step-spectrum proof uses the blocks
$$S_1=U^T S,\qquad S_2=U_\perp^T S,$$
while the random stopping-stage index is $J$ and the stopped pilot size is $B=b_J$.
