# Lemma 1 & Theorem 3: Exact Query Accounting, Realized Variance & Rank-Aware Risk

**Classification**: `PROVED`

**Notation**: See [`proof_notation.md`](proof_notation.md). Uppercase $S$ denotes the sketching matrix and $r=r_{\mathrm{actual}}$ denotes the realized basis rank.

---

## 1. Lemma 1: Exact Query Accounting with Reusable Pilot

Assume:
1. The stopped pilot size satisfies $B \le q$, and all pilot columns are retained as part of the final $q$-column sketch $S = [S_{\text{pilot}}, S_{\text{extra}}]$;
2. The basis is extended incrementally so previously computed products $A Q_{\text{old}}$ are preserved;
3. The final basis contains $r = r_{\text{actual}} \le q$ orthonormal columns;
4. The residual stage uses an integer number $\ell \ge 1$ of matrix-vector products;
5. The algorithm exhausts an exact total budget of $m$ matrix-vector products.

Then the total query count is:
$$\boxed{q + r + \ell = m} \implies \boxed{\ell = m - q - r}$$

### Proof
- Pilot size $B$ uses $B$ matvecs. Extra sketch columns $q - B$ use $q - B$ matvecs. Total range-finding cost: $B + (q - B) = q$.
- Range-finding basis QR evaluation $A Q$ costs $r = r_{\text{actual}}$ matvecs.
- Residual estimation uses $\ell$ projected probe matvecs.
- Total cost: $q + r + \ell$. Setting $q + r + \ell = m$ yields $\ell = m - q - r$.
- When $r = q$ (full rank), $\ell = m - 2q$. Thus $m - 2q$ is strictly the full-rank special case. $\blacksquare$

---

## 2. Theorem 3: Gaussian Conditional Variance

Assume $A = A^T$. Let $\mathcal{G}$ be the pre-residual sigma-algebra that fixes the orthonormal basis $Q \in \mathbb{R}^{d \times r}$ and the allocation variables $q,r,\ell$, where $\ell=m-q-r>0$.
Let $A_{\mathrm{res}} = R A R = (I - Q Q^T) A (I - Q Q^T)$.
If, conditional on $\mathcal{G}$, the probes $g_1, \dots, g_\ell$ are independent with distribution $\mathcal{N}(0, I_d)$, then:
$$\boxed{\operatorname{Var}(\widehat{t} \mid \mathcal{G}) = \frac{2}{\ell} \|R A R\|_F^2 = \frac{2}{m - q - r} \|R A R\|_F^2}$$

### Proof
Conditional on $\mathcal{G}$, $Q,q,r,\ell$ and $\operatorname{tr}(Q^T A Q)$ are fixed. Since $A = A^T$ and $R = R^T$, $A_{\mathrm{res}} = R A R$ is real symmetric.
Diagonalize $A_{\mathrm{res}} = W \Lambda W^T$ with eigenvalues $\mu_1, \dots, \mu_d$. By rotational invariance of Gaussian distributions, $z = W^T g \sim \mathcal{N}(0, I_d)$.
Thus $g^T A_{\mathrm{res}} g = z^T \Lambda z = \sum_{i=1}^d \mu_i z_i^2$.

Since $z_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$, $\mathbb{E}[z_i^2] = 1$ and $\mathbb{E}[z_i^4] = 3 \implies \operatorname{Var}(z_i^2) = 2$.
Therefore:
$$\operatorname{Var}(g^T A_{\mathrm{res}} g) = \sum_{i=1}^d \mu_i^2 \operatorname{Var}(z_i^2) = 2 \sum_{i=1}^d \mu_i^2 = 2 \|A_{\mathrm{res}}\|_F^2 = 2 \|R A R\|_F^2$$
Averaging $\ell$ independent copies divides variance by $\ell$. Substituting $\ell = m - q - r$ gives:
$$\boxed{\operatorname{Var}(\widehat{t} \mid \mathcal{G}) = \frac{2}{m - q - r} \|R A R\|_F^2}$$
$\blacksquare$

---

## 3. Corollary 3.1: Exact Rademacher Conditional Variance

If, conditional on $\mathcal{G}$, the $g_j$ are independent Rademacher probes ($\pm 1$ with probability $1/2$):
$$
\boxed{
\operatorname{Var}(\widehat{t}\mid\mathcal G)
=
\frac{2}{\ell}\sum_{u\ne v}(A_{\mathrm{res}})_{uv}^2
=
\frac{2}{m-q-r}\sum_{u\ne v}(RAR)_{uv}^2.
}
$$

Indeed, the diagonal part of $g^TA_{\mathrm{res}}g$ is deterministic because $g_u^2=1$. For symmetric $A_{\mathrm{res}}$, the off-diagonal part is

$$
2\sum_{u<v}(A_{\mathrm{res}})_{uv}g_ug_v.
$$

Distinct unordered-pair products are uncorrelated, so the one-probe variance is

$$
4\sum_{u<v}(A_{\mathrm{res}})_{uv}^2
=
2\sum_{u\ne v}(A_{\mathrm{res}})_{uv}^2.
$$

Averaging $\ell$ independent probes divides by $\ell$. Consequently,

$$
\operatorname{Var}(\widehat t\mid\mathcal G)
\le
\frac{2}{m-q-r}\|RAR\|_F^2,
$$

but the off-diagonal formula above is the exact Rademacher conditional variance.

### Corollary 3.2: Exact conditional risk averages to unconditional MSE

Assume the estimator is conditionally unbiased,

$$
\mathbb E[\widehat t\mid\mathcal G]=\operatorname{tr}(A),
$$

and the residual probes are fresh and conditionally distributed as specified in Theorem 3 or Corollary 3.1. The law of total variance gives

$$
\begin{aligned}
\mathbb E[(\widehat t-\operatorname{tr}(A))^2]
&=
\mathbb E[\operatorname{Var}(\widehat t\mid\mathcal G)]
+
\operatorname{Var}(\mathbb E[\widehat t\mid\mathcal G])\\
&=
\mathbb E[\operatorname{Var}(\widehat t\mid\mathcal G)].
\end{aligned}
$$

Thus averaging the exact Gaussian or Rademacher conditional risk over randomized pre-residual bases equals the estimator's unconditional MSE. This conclusion does not automatically apply when residual probes are reused in basis selection or conditional unbiasedness fails. $\blacksquare$

---

## 4. Definitions of Risk Functions

1. **Realized Rank-Aware Risk**:
   $$\boxed{\mathcal{R}_{\text{real}}(Q; q, r) = \frac{2 \|R A R\|_F^2}{m - q - r}}$$
2. **Rank-Aware Oracle Risk**: when $A \succeq 0$, its eigenvalues are ordered $\lambda_1 \ge \dots \ge \lambda_d \ge 0$, $Q = V_r$ spans the leading $r$ eigenvectors, and $T(r) = \sum_{i=r+1}^d \lambda_i^2$,
   $$\boxed{\mathcal{R}_{\text{rank}}(q, r) = \frac{2 T(r)}{m - q - r}}$$
3. **Full-Rank Oracle Planning Surrogate** (when $r = q$):
   $$\boxed{\mathcal{R}_{\text{full}}(q) = \frac{2 T(q)}{m - 2q}}$$
