# Theorems 7 & 8: Boundary-Anchored Exponential Tail Ratio & Safe Extrapolation Bound

**Classification**: `PROVED` (Exact log ratio & Taylor expansion) & `PROVED UNDER EXPLICIT ASSUMPTIONS` ($D_{\max}$ safe extrapolation bound).

**Notation**: See [`proof_notation.md`](proof_notation.md). The exponential-slope error is $\Delta_\alpha$; $\delta$ remains reserved for a probability-of-failure level elsewhere in the proof system.

---

## 1. Theorem 7: Exact Boundary-Anchored Exponential Tail Ratio

Fix an integer $q\ge b$ and assume the boundary value satisfies $\theta_b>0$. Assume that for all $i > b$, true singular values follow exponential decay anchored at $\theta_b$:
$$\lambda_i = \theta_b e^{-\alpha(i - b)}, \qquad \alpha > 0$$
Suppose the pilot estimates $\widehat{\alpha} = \alpha + \Delta_\alpha>0$ and predicts the tail anchored at the **same boundary value** $\theta_b$:
$$\widehat{\lambda}_i = \theta_b e^{-(\alpha + \Delta_\alpha)(i - b)}.$$

For an infinite tail ($d \to \infty$), $T_\alpha(q) = \sum_{i=q+1}^\infty \lambda_i^2 = \theta_b^2 \frac{e^{-2\alpha(q+1-b)}}{1 - e^{-2\alpha}}$.
The exact logarithmic tail ratio is:
$$\boxed{\log \frac{T_{\alpha+\Delta_\alpha}(q)}{T_\alpha(q)} = -2 \Delta_\alpha (q + 1 - b) + \log \frac{1 - e^{-2\alpha}}{1 - e^{-2(\alpha + \Delta_\alpha)}}.}$$

---

## 2. Corollary 7.1: Correct First-Order Taylor Expansion

Let $h(\alpha) = \log(1 - e^{-2\alpha})$. Then $h'(\alpha) = \frac{2 e^{-2\alpha}}{1 - e^{-2\alpha}}$.
Expanding $h(\alpha + \Delta_\alpha) = h(\alpha) + h'(\alpha) \Delta_\alpha + O(\Delta_\alpha^2)$:
$$h(\alpha) - h(\alpha + \Delta_\alpha) = -\frac{2 e^{-2\alpha}}{1 - e^{-2\alpha}} \Delta_\alpha + O(\Delta_\alpha^2).$$

Substituting into the exact log ratio formula:
$$\log \frac{T_{\alpha+\Delta_\alpha}(q)}{T_\alpha(q)} = -2 \Delta_\alpha (q + 1 - b) - \frac{2 e^{-2\alpha}}{1 - e^{-2\alpha}} \Delta_\alpha + O(\Delta_\alpha^2).$$

Extracting only the extrapolation-distance growing term $-2 \Delta_\alpha (q - b)$:
$$\boxed{\log \frac{T_{\alpha+\Delta_\alpha}(q)}{T_\alpha(q)} = -2 \Delta_\alpha (q - b) + O(\Delta_\alpha).}$$

> **Mathematical Clarification**: The remainder is $O(\Delta_\alpha)$ when only $-2\Delta_\alpha(q-b)$ is extracted because constant first-order terms in $\Delta_\alpha$ remain. If all first-order terms are explicitly written, the remaining error is $O(\Delta_\alpha^2)$.
> The critical physical interpretation is that $-2\Delta_\alpha(q-b)$ is the **only first-order term that grows without bound** as extrapolation distance $(q - b)$ increases.

---

## 3. Theorem 8: Rigorous Relative-Error Sensitivity Bound

Retain the assumptions $q\ge b$ and $\theta_b>0$. Fix $0 < \Delta_0 < \alpha$ and assume $|\Delta_\alpha| \le \Delta_0$, which guarantees $\alpha+\Delta_\alpha>0$.
Define $C_{\alpha, \Delta_0} \equiv \frac{2 e^{-2(\alpha - \Delta_0)}}{1 - e^{-2(\alpha - \Delta_0)}}$.

1. Logarithmic tail error bound:
   $$\boxed{\left| \log \frac{T_{\alpha+\Delta_\alpha}(q)}{T_\alpha(q)} \right| \le \left[ 2(q + 1 - b) + C_{\alpha, \Delta_0} \right] |\Delta_\alpha|.}$$
2. Relative tail energy error bound:
   $$\boxed{\left| \frac{T_{\alpha+\Delta_\alpha}(q)}{T_\alpha(q)} - 1 \right| \le \exp\left( [2(q+1-b) + C_{\alpha, \Delta_0}] |\Delta_\alpha| \right) - 1.}$$
3. A sufficient slope-precision condition for relative error $\le \varepsilon$:
   $$\boxed{|\Delta_\alpha| \le \frac{\log(1 + \varepsilon)}{2(q + 1 - b) + C_{\alpha, \Delta_0}}.}$$
   In the regime of small $\varepsilon$ and growing extrapolation distance $q-b\ge1$, with $\alpha$ and $\Delta_0$ fixed, the right-hand side is $O\!\left(\varepsilon/(q-b)\right)$.

### Proof
Let $h(x)=\log(1-e^{-2x})$. On the interval $[\alpha-\Delta_0,\alpha+\Delta_0]$,
$$h'(x)=\frac{2e^{-2x}}{1-e^{-2x}}$$
is positive and decreasing. By the Mean Value Theorem,
$$|h(\alpha+\Delta_\alpha)-h(\alpha)|\le C_{\alpha,\Delta_0}|\Delta_\alpha|.$$
Combining this with the exact ratio in Theorem 7 and using $q+1-b\ge1$ proves the logarithmic bound. If $x$ denotes the logarithmic-error bound and $\rho=T_{\alpha+\Delta_\alpha}(q)/T_\alpha(q)>0$, then $e^{-x}\le\rho\le e^x$, so $|\rho-1|\le e^x-1$. Finally, requiring $x\le\log(1+\varepsilon)$ gives the displayed sufficient condition. $\blacksquare$

---

## 4. Corollary 8.1: Deterministic Safeguard from $D_{\max}$

Suppose the sequential pilot stopping rule enforces $q - b \le D_{\max}$. Then:
$$\boxed{\left| \frac{T_{\alpha+\Delta_\alpha}(q)}{T_\alpha(q)} - 1 \right| \le \exp\left( [2(D_{\max}+1) + C_{\alpha, \Delta_0}] |\Delta_\alpha| \right) - 1.}$$
This imposes a **deterministic upper cap** on the amplification of slope-estimation error. The cap can still be numerically large when $D_{\max}$ or $|\Delta_\alpha|$ is large, so it is a safeguard rather than a guarantee of small error.
