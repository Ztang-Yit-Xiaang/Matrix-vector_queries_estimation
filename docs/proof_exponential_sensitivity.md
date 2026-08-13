# Theorems 7 & 8: Boundary-Anchored Exponential Tail Ratio & Safe Extrapolation Bound

**Classification**: `PROVED` (Exact log ratio & Taylor expansion) & `PROVED UNDER EXPLICIT ASSUMPTIONS` ($D_{\max}$ safe extrapolation bound).

---

## 1. Theorem 7: Exact Boundary-Anchored Exponential Tail Ratio

Assume that for all $i > b$, true singular values follow exponential decay anchored at boundary Ritz value $\theta_b$:
$$\lambda_i = \theta_b e^{-\alpha(i - b)}, \qquad \alpha > 0$$
Suppose the pilot estimates $\widehat{\alpha} = \alpha + \delta$ and predicts the tail anchored at the **same boundary value** $\theta_b$:
$$\widehat{\lambda}_i = \theta_b e^{-(\alpha + \delta)(i - b)}$$

For an infinite tail ($d \to \infty$), $T_\alpha(q) = \sum_{i=q+1}^\infty \lambda_i^2 = \theta_b^2 \frac{e^{-2\alpha(q+1-b)}}{1 - e^{-2\alpha}}$.
The exact logarithmic tail ratio is:
$$\boxed{\log \frac{T_{\alpha+\delta}(q)}{T_\alpha(q)} = -2 \delta (q + 1 - b) + \log \frac{1 - e^{-2\alpha}}{1 - e^{-2(\alpha + \delta)}}}$$

---

## 2. Corollary 7.1: Correct First-Order Taylor Expansion

Let $h(\alpha) = \log(1 - e^{-2\alpha})$. Then $h'(\alpha) = \frac{2 e^{-2\alpha}}{1 - e^{-2\alpha}}$.
Expanding $h(\alpha + \delta) = h(\alpha) + h'(\alpha) \delta + O(\delta^2)$:
$$h(\alpha) - h(\alpha + \delta) = -\frac{2 e^{-2\alpha}}{1 - e^{-2\alpha}} \delta + O(\delta^2)$$

Substituting into the exact log ratio formula:
$$\log \frac{T_{\alpha+\delta}(q)}{T_\alpha(q)} = -2 \delta (q + 1 - b) - \frac{2 e^{-2\alpha}}{1 - e^{-2\alpha}} \delta + O(\delta^2)$$

Extracting only the extrapolation-distance growing term $-2 \delta (q - b)$:
$$\boxed{\log \frac{T_{\alpha+\delta}(q)}{T_\alpha(q)} = -2 \delta (q - b) + O(\delta)}$$

> **Mathematical Clarification**: The remainder is $O(\delta)$ when only $-2\delta(q-b)$ is extracted because constant first-order terms in $\delta$ remain. If all first-order terms are explicitly written, the remaining error is $O(\delta^2)$.
> The critical physical interpretation is that $-2\delta(q-b)$ is the **only first-order term that grows without bound** as extrapolation distance $(q - b)$ increases.

---

## 3. Theorem 8: Rigorous Relative-Error Sensitivity Bound

Fix $0 < \delta_0 < \alpha$. Assume $|\delta| \le \delta_0$.
Define $C_{\alpha, \delta_0} \equiv \frac{2 e^{-2(\alpha - \delta_0)}}{1 - e^{-2(\alpha - \delta_0)}}$.

1. Logarithmic tail error bound:
   $$\boxed{\left| \log \frac{T_{\alpha+\delta}(q)}{T_\alpha(q)} \right| \le \left[ 2(q + 1 - b) + C_{\alpha, \delta_0} \right] |\delta|}$$
2. Relative tail energy error bound:
   $$\boxed{\left| \frac{T_{\alpha+\delta}(q)}{T_\alpha(q)} - 1 \right| \le \exp\left( [2(q+1-b) + C_{\alpha, \delta_0}] |\delta| \right) - 1}$$
3. Slope precision requirement for relative error $\le \varepsilon$:
   $$\boxed{|\delta| \le \frac{\log(1 + \varepsilon)}{2(q + 1 - b) + C_{\alpha, \delta_0}} = O\left( \frac{\varepsilon}{q - b} \right)}$$

---

## 4. Corollary 8.1: Deterministic Safeguard from $D_{\max}$

Suppose the sequential pilot stopping rule enforces $q - b \le D_{\max}$. Then:
$$\boxed{\left| \frac{T_{\alpha+\delta}(q)}{T_\alpha(q)} - 1 \right| \le \exp\left( [2(D_{\max}+1) + C_{\alpha, \delta_0}] |\delta| \right) - 1}$$
This imposes a **deterministic upper cap** on the amplification of slope-estimation error, preventing deep extrapolation failures.
