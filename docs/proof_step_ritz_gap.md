# Theorems 9 & 10: Deterministic Step Ritz Structure & Principal-Angle Reduction

**Classification**: `PROVED` (Theorems 9, 10 & Corollary 9.1) & `PROVED UNDER EXPLICIT ASSUMPTIONS` (Corollary 10.1 generic high-probability knee condition).

---

## 1. Theorem 9: Ritz Values for Ideal Step Spectrum

Let $A = \eta I_d + (1 - \eta) U U^T$ where $U \in \mathbb{R}^{d \times r}$, $U^T U = I_r$, and $0 < \eta < 1$.
$A$ has eigenvalues $1$ (multiplicity $r$) and $\eta$ (multiplicity $d - r$).

Let $Q \in \mathbb{R}^{d \times b}$, $Q^T Q = I_b$, where pilot size $b = r + p \ge r + 1$ ($p \ge 1$).
Let $s_1 \ge \dots \ge s_r \ge 0$ be the singular values of $U^T Q \in \mathbb{R}^{r \times b}$.
Let $\theta_1 \ge \dots \ge \theta_b$ be the ordered Ritz values of $Q^T A Q$.

Then:
1. **Top-$r$ Ritz Values**:
   $$\boxed{\theta_i = \eta + (1 - \eta) s_i^2, \qquad 1 \le i \le r}$$
2. **Exact Post-Knee Ritz Noise Floor**:
   $$\boxed{\theta_{r+1} = \theta_{r+2} = \dots = \theta_b = \eta \qquad (\text{exact!})}$$
3. **Exact Ritz Ratio**:
   $$\boxed{\frac{\theta_r}{\theta_{r+1}} = \frac{\eta + (1 - \eta) s_r^2}{\eta}}$$

### Proof
$Q^T A Q = \eta I_b + (1 - \eta) (U^T Q)^T (U^T Q)$.
The non-zero eigenvalues of $(U^T Q)^T (U^T Q) \in \mathbb{R}^{b \times b}$ are $s_1^2, \dots, s_r^2$.
Because $\operatorname{rank}(U^T Q) \le r$, it has at least $b - r = p$ exact zero eigenvalues.
Adding $\eta I_b$ shifts all eigenvalues by $\eta$. Thus $\theta_{r+1} = \dots = \theta_b = \eta$. $\blacksquare$

---

## 2. Corollary 9.1: Sufficient Deterministic Gap Condition & Necessary Horizon

1. **Sufficient Gap Condition**: $\frac{\theta_r}{\theta_{r+1}} > \tau \iff \boxed{s_r^2 > \frac{\eta(\tau - 1)}{1 - \eta}}$.
2. **Necessary Horizon Condition**: Forming the ordered Ritz ratio $\frac{\theta_r}{\theta_{r+1}}$ requires the existence of $\theta_{r+1}$, which strictly requires:
   $$\boxed{b \ge r + 1 \iff p = b - r \ge 1}$$
   This proves that at least one post-knee Ritz position ($p \ge 1$) is **mathematically necessary** for the ordered ratio detector to be defined.

---

## 3. Theorem 10: Deterministic Principal-Angle Reduction

Let $\Omega \in \mathbb{R}^{d \times (r+p)}$ be a sketch matrix. Express $\Omega$ in the eigenbasis of $A$:
$$\begin{bmatrix} U^T \\ U_\perp^T \end{bmatrix} \Omega = \begin{bmatrix} \Omega_1 \\ \Omega_2 \end{bmatrix}, \qquad \Omega_1 \in \mathbb{R}^{r \times (r+p)}, \quad \Omega_2 \in \mathbb{R}^{(d-r) \times (r+p)}$$
Let $Y = A \Omega$ and $Q = \operatorname{orth}(Y)$. Assume $\Omega_1$ has full row rank.

1. **Principal-Angle Tangent Bound**:
   $$\boxed{\tan \Theta_{\max} \le \eta \|\Omega_2 \Omega_1^\dagger\|_2}$$
2. **Singular Value Lower Bound**:
   $$\boxed{s_r^2 = \cos^2 \Theta_{\max} \ge \frac{1}{1 + \eta^2 \|\Omega_2 \Omega_1^\dagger\|_2^2}}$$

### Proof
In eigenbasis $[U, U_\perp]$, $Y = A \Omega = \begin{bmatrix} U \Omega_1 \\ \eta U_\perp \Omega_2 \end{bmatrix}$.
Multiplying by $\Omega_1^\dagger$: $Y \Omega_1^\dagger = \begin{bmatrix} U \\ U_\perp F \end{bmatrix}$ where $F = \eta \Omega_2 \Omega_1^\dagger$.
Range of $Y$ contains graph subspace $\mathcal{G} = \operatorname{range}\begin{bmatrix} I_r \\ F \end{bmatrix}$, with $\tan \Theta_{\max}(\operatorname{range}(U), \mathcal{G}) = \|F\|_2 = \eta \|\Omega_2 \Omega_1^\dagger\|_2$.
Since $\mathcal{G} \subseteq \operatorname{range}(Q)$, principal angle to $Q$ cannot be larger. Thus $\tan \Theta_{\max} \le \eta \|\Omega_2 \Omega_1^\dagger\|_2$, and $\cos^2 \Theta_{\max} = \frac{1}{1 + \tan^2 \Theta_{\max}} \ge \frac{1}{1 + \eta^2 \|\Omega_2 \Omega_1^\dagger\|_2^2}$. $\blacksquare$

---

## 4. Corollary 10.1: Generic High-Probability Knee-Detection Condition

Suppose a random matrix bound establishes $\mathbb{P}\left( \|\Omega_2 \Omega_1^\dagger\|_2 \le K_{r, d, p, \delta} \right) \ge 1 - \delta$.
Then a sufficient condition for $\mathbb{P}\left( \frac{\theta_r}{\theta_{r+1}} > \tau \right) \ge 1 - \delta$ is:
$$\boxed{\frac{\eta + \frac{1-\eta}{1 + \eta^2 K_{r, d, p, \delta}^2}}{\eta} > \tau}$$

> **Important Limitation & Unsupported Formula Status**:
> The explicit guessed formula $p \ge \left\lceil \frac{\log(1/\delta) + \log(d-r) + \log(\tau r)}{\log(1/\eta)} \right\rceil$ is **WITHDRAWN as unproved**.
> Closed-form evaluation of $K_{r, d, p, \delta}$ remains open research theory.
