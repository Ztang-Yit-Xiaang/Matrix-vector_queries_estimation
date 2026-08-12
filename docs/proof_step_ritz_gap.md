# Theorem: Deterministic Step-Spectrum Ritz Structure & Principal-Angle Gap Control

**Classification**: `PROVED` (Deterministic Ritz eigenvalue identity & Principal angle bound) & `PROVED UNDER EXPLICIT ASSUMPTIONS` (High-probability knee detection conditional on Gaussian matrix norm bounds).

---

## 1. Theorem Statements

### Theorem 7 (Deterministic Step-Spectrum Ritz Value Structure)
Let $A \in \mathbb{R}^{d \times d}$ have stylized step spectrum:
$$A = U \begin{bmatrix} I_r & 0 \\ 0 & \eta I_{d-r} \end{bmatrix} U^T = \eta I_d + (1 - \eta) P_r, \qquad 0 < \eta < 1$$
where $P_r = U_r U_r^T$ is the orthogonal projector onto the top-$r$ eigenspace.

Let $S \in \mathbb{R}^{d \times b}$ be a random pilot matrix with pilot size $b = r + p$ ($p \ge 1$).
Let $Y = A S$ and $Q = \operatorname{orth}(Y) \in \mathbb{R}^{d \times b}$.
Let $\theta_1 \ge \theta_2 \ge \dots \ge \theta_b$ be the ordered Ritz values of $Q^T A Q$.

Then:
1. **Top-$r$ Ritz Values**:
   $$\boxed{\theta_j = \eta + (1 - \eta) \sigma_j^2(U_r^T Q), \qquad j = 1, \dots, r}$$
2. **Exact Post-Knee Ritz Noise Floor**:
   $$\boxed{\theta_{r+1} = \theta_{r+2} = \dots = \theta_{r+p} = \eta \qquad (\text{exact!})}$$

---

### Theorem 8 (Deterministic Principal-Angle Control & Ritz Gap Lower Bound)
In the eigenbasis of $A$, let $U^T S = \begin{bmatrix} \Omega_1 \\ \Omega_2 \end{bmatrix}$ with $\Omega_1 \in \mathbb{R}^{r \times (r+p)}$ and $\Omega_2 \in \mathbb{R}^{(d-r) \times (r+p)}$.
Let $\kappa \equiv \|\Omega_2 \Omega_1^\dagger\|_2$.

1. **Top-$r$ Singular Value Lower Bound**:
   $$\sigma_r^2(U_r^T Q) \ge \frac{1}{1 + \eta^2 \kappa^2}$$
2. **Exact Ritz Ratio Lower Bound**:
   $$\boxed{\frac{\theta_r}{\theta_{r+1}} \ge 1 + \frac{1 - \eta}{\eta [1 + \eta^2 \kappa^2]}}$$
3. **Sufficient Deterministic Condition for Knee Detection Ratio $\tau > 1$**:
   A threshold ratio $\theta_r / \theta_{r+1} > \tau$ is guaranteed whenever $\kappa^2 < \frac{1}{\eta^2} \left[ \frac{1 - \eta}{\eta (\tau - 1)} - 1 \right]$.

---

## 2. Complete Mathematical Proofs

### Proof of Theorem 7 (Deterministic Step-Spectrum Ritz Structure)
Using $A = \eta I_d + (1 - \eta) P_r$:
$$Q^T A Q = Q^T \left( \eta I_d + (1 - \eta) U_r U_r^T \right) Q = \eta I_b + (1 - \eta) Q^T U_r U_r^T Q$$

Let $C = U_r^T Q \in \mathbb{R}^{r \times b}$. Then $Q^T U_r U_r^T Q = C^T C$.
The matrix $C^T C$ is $b \times b$ with rank at most $r$.
Its non-zero eigenvalues are $\sigma_1^2(C), \dots, \sigma_r^2(C)$.
Because $b = r + p$, $C^T C$ has at least $p$ exact zero eigenvalues.

Therefore, the eigenvalues of $Q^T A Q = \eta I_b + (1 - \eta) C^T C$ are:
$$\theta_j = \eta + (1 - \eta) \sigma_j^2(C), \qquad j = 1, \dots, r$$
$$\theta_{r+1} = \dots = \theta_{r+p} = \eta + (1 - \eta) \cdot 0 = \eta$$
$\blacksquare$

---

### Proof of Theorem 8 (Deterministic Principal-Angle Control & Ritz Gap)
1. In the eigenbasis of $A$:
   $$U^T Y = U^T A S = \begin{bmatrix} I_r & 0 \\ 0 & \eta I_{d-r} \end{bmatrix} \begin{bmatrix} \Omega_1 \\ \Omega_2 \end{bmatrix} = \begin{bmatrix} \Omega_1 \\ \eta \Omega_2 \end{bmatrix}$$
2. Post-multiplying by $\Omega_1^\dagger$ (which exists and satisfies $\Omega_1 \Omega_1^\dagger = I_r$ with probability 1 for $p \ge 0$):
   $$\begin{bmatrix} \Omega_1 \\ \eta \Omega_2 \end{bmatrix} \Omega_1^\dagger = \begin{bmatrix} I_r \\ \eta \Omega_2 \Omega_1^\dagger \end{bmatrix}$$
   Let $F = \eta \Omega_2 \Omega_1^\dagger$.
3. The graph subspace $\mathcal{Z} = \operatorname{range}\begin{bmatrix} I_r \\ F \end{bmatrix} \subseteq \operatorname{range}(Y) = \operatorname{range}(Q)$.
   For graph subspace $\mathcal{Z}$, $\tan \Theta_{\max} = \|F\|_2 = \eta \|\Omega_2 \Omega_1^\dagger\|_2 = \eta \kappa$.
   Therefore, $\cos^2 \Theta_{\max} = \frac{1}{1 + \eta^2 \kappa^2}$.
4. Since $\mathcal{Z} \subseteq \operatorname{range}(Q)$, projection onto $Q$ gives a lower bound on singular values:
   $$\sigma_r^2(U_r^T Q) \ge \cos^2 \Theta_{\max} = \frac{1}{1 + \eta^2 \kappa^2}$$
5. Substituting into $\theta_r = \eta + (1 - \eta) \sigma_r^2(U_r^T Q)$:
   $$\theta_r \ge \eta + \frac{1 - \eta}{1 + \eta^2 \kappa^2}$$
6. Forming the ratio $\theta_r / \theta_{r+1}$ with $\theta_{r+1} = \eta$:
   $$\frac{\theta_r}{\theta_{r+1}} \ge \frac{\eta + \frac{1 - \eta}{1 + \eta^2 \kappa^2}}{\eta} = 1 + \frac{1 - \eta}{\eta [1 + \eta^2 \kappa^2]}$$
$\blacksquare$

---

## 3. High-Probability Gaussian Matrix Bound Extension
Using operator norm bounds for Gaussian matrices $\Omega_1 \in \mathbb{R}^{r \times (r+p)}$ and $\Omega_2 \in \mathbb{R}^{(d-r) \times (r+p)}$:
- $\|\Omega_2\|_2 \le \sqrt{d - r} + \sqrt{r + p} + t_2$ with probability $1 - e^{-t_2^2 / 2}$.
- $\sigma_{\min}(\Omega_1) \ge \sqrt{r + p} - \sqrt{r} - t_1$ with probability $1 - e^{-t_1^2 / 2}$.

Conditional on these high-probability events, $\kappa \le \frac{\|\Omega_2\|_2}{\sigma_{\min}(\Omega_1)}$ is bounded, guaranteeing $\theta_r / \theta_{r+1} > \tau$ with probability at least $1 - \delta_1 - \delta_2$.
