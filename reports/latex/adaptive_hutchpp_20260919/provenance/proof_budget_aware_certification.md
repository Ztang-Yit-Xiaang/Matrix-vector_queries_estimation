# Budget-Aware Direct Certification: Lemma 15.1 and Theorems 16--17

**Classification:** `PROVED` on the explicit nested-construction, positive-denominator, and freshness domains below.

## Assumption ledger

Let $A\in\mathbb R^{d\times d}$ be symmetric. Let $Q_0$ and $Q_a$ be orthonormal bases constructed before certification. Let $q_x$ count sampled products, let $r_x$ be the accepted rank, and let $c_x=q_x+r_x$. Certification uses $s\ge1$ new products. Final residual probes are fresh after selection.

## Lemma 15.1: Nested-construction reuse

If $\operatorname{range}(Q_{\rm low})\subseteq\operatorname{range}(Q_{\rm high})$ and both bases have orthonormal columns, put $T=Q_{\rm high}^TQ_{\rm low}$. The higher-space projector fixes every column of the lower basis, so

$$
Q_{\rm low}=Q_{\rm high}Q_{\rm high}^TQ_{\rm low}=Q_{\rm high}T.
$$

Linearity then gives

$$
AQ_{\rm low}=(AQ_{\rm high})T.
$$

Thus a cached higher-prefix construction supplies the lower basis and its image without another oracle product. For the frozen incremental paths, constructing both actions costs $\max\{c_0,c_a\}$ rather than $c_0+c_a$. The conclusion does not extend to nonnested actions. $\blacksquare$

## Theorem 16: Common sunk-cost denominator

Once both actions have been constructed, the committed cost is $c_{\rm pre}=\max\{c_0,c_a\}$. Certification spends another $s$ products. Since spent queries cannot be refunded, either final basis has

$$
\ell_{\rm paid}=m-c_{\rm pre}-s
$$

fresh residual products. If $\ell_{\rm paid}>0$, the exact conditional risks are

$$
\mathcal R_{0,\rm paid}=\frac{\sigma_0^2}{\ell_{\rm paid}},
\qquad
\mathcal R_{a,\rm paid}=\frac{\sigma_a^2}{\ell_{\rm paid}}.
$$

The common positive denominator proves that the paid candidate is better exactly when $\sigma_a^2<\sigma_0^2$. $\blacksquare$

## Theorem 17: No-free-fallback lemma

The original baseline has $\ell_0=m-c_0$ residual products and risk $\sigma_0^2/\ell_0$. If $\sigma_0^2>0$ and $c_{\rm pre}+s>c_0$, then $\ell_{\rm paid}<\ell_0$, and

$$
\frac{\mathcal R_{0,\rm paid}}{\mathcal R_0^{\rm original}}
=\frac{\ell_0}{\ell_{\rm paid}}>1.
$$

Therefore rejection or abstention after construction and certification cannot restore the original baseline. If $\sigma_0^2=0$, both fallback risks are zero and the displayed ratio is undefined; this boundary must not be handled by division. $\blacksquare$

## Corollary 17.1: Net-benefit threshold

On the domain $\sigma_0^2>0$, the paid candidate beats the original baseline exactly when

$$
\frac{\sigma_a^2}{\sigma_0^2}
<\frac{\ell_{\rm paid}}{\ell_0}
=1-\frac{(c_{\rm pre}-c_0)+s}{\ell_0}.
$$

For the full-rank transition $r_\star\to r_\star+1$, the added construction cost is two, so the right side is $1-(s+2)/(m-2r_\star)$. $\blacksquare$

## Corollary 17.2: Paid two-action oracle

The smallest conditional risk available after both actions and certification have been paid is

$$
\mathcal R_{\rm oracle,paid}
=\frac{\min\{\sigma_0^2,\sigma_a^2\}}{\ell_{\rm paid}}.
$$

Every selector restricted to these bases is pointwise no better. If the finite frozen-population mean of this oracle is no smaller than the original-baseline mean, no selector over this pair can pay for this timing on that frozen population. This is not an orientation-universal impossibility theorem. $\blacksquare$

## Audit result

- The maximum-cost rule requires nested subspaces and cached higher-prefix products.
- Every division requires the displayed positive denominator.
- Paid-order truth $\sigma_a^2<\sigma_0^2$ is distinct from net-benefit truth $\sigma_a^2/\ell_{\rm paid}<\sigma_0^2/\ell_0$.
- Certification-based selection is followed by fresh residual probes; reuse of certification probes is not covered.
