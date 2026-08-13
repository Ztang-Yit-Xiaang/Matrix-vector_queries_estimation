# Lemma 1 & Theorem 3: Exact Query Accounting, Realized Variance & Rank-Aware Risk

**Classification**: `PROVED`

---

## 1. Lemma 1: Exact Query Accounting with Reusable Pilot

Assume:
1. Sequential pilot columns are retained as part of the final $q$-column sketch $S = [S_{\text{pilot}}, S_{\text{extra}}]$;
2. The basis is extended incrementally so previously computed products $A Q_{\text{old}}$ are preserved;
3. The final basis contains $r = r_{\text{actual}} \le q$ orthonormal columns;
4. The residual stage uses $\ell$ matrix-vector products.

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

Assume $A = A^T$, and condition on a fixed orthonormal basis $Q \in \mathbb{R}^{d \times r}$.
Let $B_Q = R A R = (I - Q Q^T) A (I - Q Q^T)$.
If $g_1, \dots, g_\ell \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, I_d)$ are independent Gaussian probes, then:
$$\boxed{\operatorname{Var}(\widehat{t} \mid Q) = \frac{2}{\ell} \|R A R\|_F^2 = \frac{2}{m - q - r} \|R A R\|_F^2}$$

### Proof
Conditional on $Q$, $\operatorname{tr}(Q^T A Q)$ is deterministic. Since $A = A^T$ and $R = R^T$, $B_Q = R A R$ is real symmetric.
Diagonalize $B_Q = U \Lambda U^T$ with eigenvalues $\mu_1, \dots, \mu_d$. By rotational invariance of Gaussian distributions, $z = U^T g \sim \mathcal{N}(0, I_d)$.
Thus $g^T B_Q g = z^T \Lambda z = \sum_{i=1}^d \mu_i z_i^2$.

Since $z_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$, $\mathbb{E}[z_i^2] = 1$ and $\mathbb{E}[z_i^4] = 3 \implies \operatorname{Var}(z_i^2) = 2$.
Therefore:
$$\operatorname{Var}(g^T B_Q g) = \sum_{i=1}^d \mu_i^2 \operatorname{Var}(z_i^2) = 2 \sum_{i=1}^d \mu_i^2 = 2 \|B_Q\|_F^2 = 2 \|R A R\|_F^2$$
Averaging $\ell$ independent copies divides variance by $\ell$. Substituting $\ell = m - q - r$ gives:
$$\boxed{\operatorname{Var}(\widehat{t} \mid Q) = \frac{2}{m - q - r} \|R A R\|_F^2}$$
$\blacksquare$

---

## 3. Corollary 3.1: Rademacher Probe Variance Upper Bound

If $g_j$ are independent Rademacher probes ($\pm 1$ with probability $1/2$):
$$\operatorname{Var}(g^T B_Q g) = 2 \sum_{a \ne b} (B_Q)_{ab}^2 \le 2 \|B_Q\|_F^2 \implies \boxed{\operatorname{Var}(\widehat{t} \mid Q) \le \frac{2}{m - q - r} \|R A R\|_F^2}$$
Thus, the Frobenius risk formula is exact for Gaussian probes and an upper bound for Rademacher probes.

---

## 4. Definitions of Risk Functions

1. **Realized Rank-Aware Risk**:
   $$\boxed{\mathcal{R}_{\text{real}}(Q; q, r) = \frac{2 \|R A R\|_F^2}{m - q - r}}$$
2. **Rank-Aware Oracle Risk** (when $Q = V_r$):
   $$\boxed{\mathcal{R}_{\text{rank}}(q, r) = \frac{2 T(r)}{m - q - r}}$$
3. **Full-Rank Oracle Planning Surrogate** (when $r = q$):
   $$\boxed{\mathcal{R}_{\text{full}}(q) = \frac{2 T(q)}{m - 2q}}$$
