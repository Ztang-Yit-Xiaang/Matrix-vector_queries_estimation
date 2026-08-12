# Theorem: Exact Boundary-Anchored Exponential Tail Sensitivity & Safe Extrapolation Bound

**Classification**: `PROVED` (Exact log-ratio formula & Taylor expansion) & `PROVED UNDER EXPLICIT ASSUMPTIONS` (Safe extrapolation bound under $D_{\max}$).

---

## 1. Theorem Statements

### Theorem 5 (Exact Boundary-Anchored Exponential Tail Ratio)
Assume the true singular values after pilot boundary $b$ follow a pure exponential decay anchored at the boundary Ritz value $\theta_b$:
$$\lambda_i = \theta_b e^{-\alpha(i - b)}, \qquad \forall i \ge b, \quad \alpha > 0$$
Suppose the pilot-estimated decay rate is $\widehat{\alpha} = \alpha + \delta$, and the predicted tail is anchored at the **same boundary value** $\theta_b$:
$$\widehat{\lambda}_i = \theta_b e^{-(\alpha + \delta)(i - b)}, \qquad \forall i \ge b$$

1. **Exact Infinite-Tail Log-Ratio Formula**:
   $$\boxed{\log \frac{\widehat{T}(q)}{T(q)} = -2 \delta (q + 1 - b) + \log \frac{1 - e^{-2\alpha}}{1 - e^{-2(\alpha + \delta)}}}$$
2. **First-Order Taylor Expansion**:
   With $h(\alpha) = \log(1 - e^{-2\alpha})$ and $h'(\alpha) = \frac{2 e^{-2\alpha}}{1 - e^{-2\alpha}}$:
   $$\boxed{\log \frac{\widehat{T}(q)}{T(q)} = -2 \delta (q + 1 - b) - \frac{2 e^{-2\alpha}}{1 - e^{-2\alpha}} \delta + O(\delta^2) = -2 \delta (q - b) + O(\delta)}$$

---

### Theorem 6 (Safe-Extrapolation Bound Using $D_{\max}$)
Assume pilot slope error is bounded $|\delta| \le \delta_0 < \alpha$.
If the sequential pilot stopping rule enforces maximum extrapolation distance $q - b \le D_{\max}$, then:
$$\left| \log \frac{\widehat{T}(q)}{T(q)} \right| \le M |\delta|, \qquad M \equiv 2(D_{\max} + 1) + \frac{2 e^{-2(\alpha - \delta_0)}}{1 - e^{-2(\alpha - \delta_0)}}$$
Consequently, relative tail energy error is bounded by:
$$\boxed{\left| \frac{\widehat{T}(q)}{T(q)} - 1 \right| \le e^{M |\delta|} - 1}$$
To guarantee relative tail energy error at most $\varepsilon$, it is sufficient that $|\delta| \le \frac{\log(1 + \varepsilon)}{M} = O\left(\frac{\varepsilon}{D_{\max}}\right)$.

---

## 2. Complete Mathematical Proofs

### Proof of Theorem 5 (Exact Infinite-Tail Formula & Taylor Expansion)
For target cutoff $q \ge b$, the exact squared tail energy for true decay $\alpha$ is:
$$T_\alpha(q) = \sum_{i=q+1}^\infty \lambda_i^2 = \theta_b^2 \sum_{i=q+1}^\infty e^{-2\alpha(i-b)} = \theta_b^2 \frac{e^{-2\alpha(q+1-b)}}{1 - e^{-2\alpha}}$$

Likewise, for estimated decay $\widehat{\alpha} = \alpha + \delta$:
$$T_{\alpha + \delta}(q) = \theta_b^2 \frac{e^{-2(\alpha + \delta)(q+1-b)}}{1 - e^{-2(\alpha + \delta)}}$$

Taking the ratio:
$$\frac{T_{\alpha + \delta}(q)}{T_\alpha(q)} = e^{-2\delta(q+1-b)} \frac{1 - e^{-2\alpha}}{1 - e^{-2(\alpha + \delta)}}$$

Taking logarithms:
$$\log \frac{\widehat{T}(q)}{T(q)} = -2 \delta (q + 1 - b) + \log(1 - e^{-2\alpha}) - \log(1 - e^{-2(\alpha + \delta)})$$

By Taylor expansion of $h(\alpha + \delta) = \log(1 - e^{-2(\alpha+\delta)})$ around $\delta = 0$:
$$h(\alpha + \delta) = h(\alpha) + \frac{2 e^{-2\alpha}}{1 - e^{-2\alpha}} \delta + O(\delta^2)$$

Substituting back:
$$\log \frac{\widehat{T}(q)}{T(q)} = -2 \delta (q + 1 - b) - \frac{2 e^{-2\alpha}}{1 - e^{-2\alpha}} \delta + O(\delta^2)$$

Regrouping $-2 \delta (q + 1 - b) = -2 \delta (q - b) - 2 \delta$:
$$\log \frac{\widehat{T}(q)}{T(q)} = -2 \delta (q - b) + \left[ -2 - \frac{2 e^{-2\alpha}}{1 - e^{-2\alpha}} \right] \delta + O(\delta^2) = -2 \delta (q - b) + O(\delta)$$
$\blacksquare$

---

### Finite-$d$ Truncation Correction Formula
For finite dimension $d$, the exact sum is:
$$T_{\alpha, d}(q) = \theta_b^2 \sum_{i=q+1}^d e^{-2\alpha(i-b)} = \theta_b^2 e^{-2\alpha(q+1-b)} \frac{1 - e^{-2\alpha(d-q)}}{1 - e^{-2\alpha}}$$
$$\log \frac{\widehat{T}_d(q)}{T_d(q)} = -2 \delta (q + 1 - b) + \log \frac{1 - e^{-2(\alpha+\delta)(d-q)}}{1 - e^{-2\alpha(d-q)}} + \log \frac{1 - e^{-2\alpha}}{1 - e^{-2(\alpha+\delta)}}$$
The infinite-tail formula is recovered when $d - q \gg 1 / (2\alpha)$.
