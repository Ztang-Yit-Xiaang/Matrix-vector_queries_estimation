# Theorem: Rank-Aware Oracle Risk & Exact Gaussian Variance

**Classification**: `PROVED`

---

## 1. Theorem Statement

Let $A \in \mathbb{R}^{d \times d}$ be a real symmetric positive semidefinite (PSD) matrix with spectral decomposition $A = V \Lambda V^T$, where $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$.
Define the exact squared tail energy:
$$T(r) = \sum_{i=r+1}^d \lambda_i^2 = \|A - A_r\|_F^2$$

Let $q = q_{\text{target}}$ be the target sketch width, and $r = r_{\text{actual}} \le q$ be the realized rank of the orthonormal basis $Q \in \mathbb{R}^{d \times r}$.
Under exact query accounting:
$$q + r + \ell = m \implies \ell = m - q - r$$

1. **Exact Conditional Variance**:
   $$\boxed{\operatorname{Var}(\widehat{t} \mid \mathcal{G}) = \frac{2 \|R A R\|_F^2}{\ell} = \frac{2 \|R A R\|_F^2}{m - q - r}}$$
2. **Rank-Aware Oracle Risk Surrogate**:
   If the realized basis $Q$ spans the leading $r$-dimensional eigenspace ($Q = V_r$), then $\|R A R\|_F^2 = T(r)$, and:
   $$\boxed{R_{\text{rank}}(q, r) = \frac{2 T(r)}{m - q - r}}$$
3. **Full-Rank Oracle Planning Surrogate**:
   In the special full-rank case where $r = q$:
   $$\boxed{R_{\text{full}}(q) = \frac{2 T(q)}{m - 2q}}$$

---

## 2. Complete Mathematical Proof

### Step 1: Gaussian Quadratic Form Variance Identity
Let $B = R A R$. Since $A$ is real symmetric and $R = I - Q Q^T$ is symmetric, $B$ is real symmetric.
For $g \sim \mathcal{N}(0, I_d)$, by the standard Gaussian quadratic form variance identity:
$$\operatorname{Var}(g^T B g) = 2 \|B\|_F^2 = 2 \|R A R\|_F^2$$

### Step 2: Averaging $\ell$ Independent Residual Probes
Conditional on $\mathcal{G}$, $g_1, \dots, g_\ell$ are independent. Therefore:
$$\operatorname{Var}\left( \frac{1}{\ell} \sum_{j=1}^\ell g_j^T R A R g_j \;\middle|\; \mathcal{G} \right) = \frac{1}{\ell^2} \sum_{j=1}^\ell \operatorname{Var}(g_j^T R A R g_j \mid \mathcal{G}) = \frac{2 \|R A R\|_F^2}{\ell}$$

Substituting $\ell = m - q - r$:
$$\boxed{\operatorname{Var}(\widehat{t} \mid \mathcal{G}) = \frac{2 \|R A R\|_F^2}{m - q - r}}$$

### Step 3: Derivation of Rank-Aware Oracle Risk Surrogate
If $Q = V_r = [v_1, \dots, v_r]$ (the exact top-$r$ eigenvectors of $A$), then $P = \sum_{i=1}^r v_i v_i^T$ and $R = \sum_{i=r+1}^d v_i v_i^T$.
Then $R A R = \sum_{i=r+1}^d \lambda_i v_i v_i^T$.
The squared Frobenius norm is:
$$\|R A R\|_F^2 = \operatorname{tr}((R A R)^2) = \sum_{i=r+1}^d \lambda_i^2 = T(r)$$

Plugging $\|R A R\|_F^2 = T(r)$ into the conditional variance formula gives:
$$\boxed{R_{\text{rank}}(q, r) = \frac{2 T(r)}{m - q - r}}$$
$\blacksquare$

---

## 3. Implementation & Documentation Distinction
- $R_{\text{full}}(q) = \frac{2 \widehat{T}(q)}{m - 2q}$ is a **full-rank planning surrogate** used during pilot optimization before $r_{\text{actual}}$ is known.
- $R_{\text{rank}}(q, r) = \frac{2 \widehat{T}(r)}{m - q - r}$ is the **rank-aware oracle surrogate** matching exact post-QR query accounting.
- Realized variance $V_{\text{realized}} = \frac{2 \|R A R\|_F^2}{m - q - r}$ is the actual sample variance.
