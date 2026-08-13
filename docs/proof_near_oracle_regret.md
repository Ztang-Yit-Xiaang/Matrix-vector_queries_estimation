# Theorems 4, 5 & 6: Tail Uncertainty Radius, Near-Oracle Regret & Soft Safety Shrinkage

**Classification**: `PROVED` (Theorems 4, 5) & `PROVED UNDER EXPLICIT ASSUMPTIONS` (Theorem 6 under Lipschitz risk).

---

## 1. Definition of Simultaneous Confidence Radius $U_b(q, \delta)$

For a fixed pilot size $b$, let $\widehat{T}_b(q)$ be the pilot-based estimate of $T(q) = \sum_{i=q+1}^d \lambda_i^2$.
Let the feasible full-rank planning set be $\mathcal{Q}_b = \{ q \in \mathbb{Z} : b \le q \le q_{\max} \}$.

A simultaneous confidence radius is any function $U_b(q, \delta) \ge 0$ satisfying:
$$\boxed{\mathbb{P}\left( \forall q \in \mathcal{Q}_b : |\widehat{T}_b(q) - T(q)| \le U_b(q, \delta) \right) \ge 1 - \delta}$$

Define the maximum risk approximation error:
$$\boxed{\varepsilon_R(b, \delta) \equiv \max_{q \in \mathcal{Q}_b} \frac{2 U_b(q, \delta)}{m - 2q}}$$

---

## 2. Theorem 4: Near-Oracle Regret at a Fixed Pilot Stage

Let $\mathcal{R}(q) = \frac{2 T(q)}{m - 2q}$ be the full-rank oracle risk, and $\widehat{\mathcal{R}}_b(q) = \frac{2 \widehat{T}_b(q)}{m - 2q}$ be the estimated risk.
Let $q_b^* = \arg\min_{q \in \mathcal{Q}_b} \mathcal{R}(q)$ be the pilot-constrained oracle action, and $\widehat{q}_b = \arg\min_{q \in \mathcal{Q}_b} \widehat{\mathcal{R}}_b(q)$ be the action selected from the estimated risk curve.

On the confidence event $|\widehat{T}_b(q) - T(q)| \le U_b(q, \delta)$:
$$\boxed{\mathcal{R}(\widehat{q}_b) - \mathcal{R}(q_b^*) \le 2 \varepsilon_R(b, \delta) = 4 \max_{q \in \mathcal{Q}_b} \frac{U_b(q, \delta)}{m - 2q}}$$

### Proof
1. For every $q \in \mathcal{Q}_b$:
   $$|\widehat{\mathcal{R}}_b(q) - \mathcal{R}(q)| = \frac{2 |\widehat{T}_b(q) - T(q)|}{m - 2q} \le \frac{2 U_b(q, \delta)}{m - 2q} \le \varepsilon_R(b, \delta)$$
2. In particular, $\mathcal{R}(\widehat{q}_b) \le \widehat{\mathcal{R}}_b(\widehat{q}_b) + \varepsilon_R$.
3. Since $\widehat{q}_b = \arg\min_q \widehat{\mathcal{R}}_b(q)$, $\widehat{\mathcal{R}}_b(\widehat{q}_b) \le \widehat{\mathcal{R}}_b(q_b^*)$.
4. Applying uniform risk approximation at $q_b^*$: $\widehat{\mathcal{R}}_b(q_b^*) \le \mathcal{R}(q_b^*) + \varepsilon_R$.
5. Combining: $\mathcal{R}(\widehat{q}_b) \le \mathcal{R}(q_b^*) + 2 \varepsilon_R \implies \boxed{\mathcal{R}(\widehat{q}_b) - \mathcal{R}(q_b^*) \le 2 \varepsilon_R}$.
The factor $2$ accounts for converting estimated risk to true risk once at $\widehat{q}_b$ and once at $q_b^*$. $\blacksquare$

---

## 3. Theorem 5: Sequential Union Bound Across Pilot Stages

Let candidate pilot sizes be $b_1, \dots, b_K$. Choose per-stage failure probabilities $\delta_1, \dots, \delta_K$ such that $\sum_{s=1}^K \delta_s \le \delta$.
If stage $s$ satisfies $\mathbb{P}(E_s) \ge 1 - \delta_s$ for $E_s = \{ \forall q \in \mathcal{Q}_{b_s} : |\widehat{T}_{b_s}(q) - T(q)| \le U_{b_s}(q, \delta_s) \}$, then by the union bound:
$$\mathbb{P}\left( \bigcap_{s=1}^K E_s \right) \ge 1 - \sum_{s=1}^K \delta_s \ge 1 - \delta$$

Consequently, at the random stopping stage $B \in \{b_1, \dots, b_K\}$:
$$\boxed{\mathcal{R}(\widehat{q}_B) - \mathcal{R}(q_B^*) \le 2 \varepsilon_R(B, \delta_B)}$$
holds with probability at least $1 - \delta$. No optional stopping theorem is required for confidence validity. $\blacksquare$

---

## 4. Theorem 6: Regret of Soft Allocation Safety Shrinkage

Assume oracle risk is discrete-Lipschitz: $|\mathcal{R}(q+1) - \mathcal{R}(q)| \le L$.
Let $q_{\text{final}} = \operatorname{round}\left( (1-\gamma) q_0 + \gamma q_{\text{adapt}} \right)$ for baseline $q_0$ and $0 \le \gamma \le 1$.
If $\mathcal{R}(q_{\text{adapt}}) - \mathcal{R}(q_b^*) \le 2 \varepsilon_R$, then:
$$\boxed{\mathcal{R}(q_{\text{final}}) - \mathcal{R}(q_b^*) \le 2 \varepsilon_R + L \left[ (1 - \gamma) |q_0 - q_{\text{adapt}}| + \frac{1}{2} \right]}$$

### Proof
1. $\mathcal{R}(q_{\text{final}}) - \mathcal{R}(q_b^*) = \left( \mathcal{R}(q_{\text{final}}) - \mathcal{R}(q_{\text{adapt}}) \right) + \left( \mathcal{R}(q_{\text{adapt}}) - \mathcal{R}(q_b^*) \right)$.
2. By Lipschitz continuity: $\mathcal{R}(q_{\text{final}}) - \mathcal{R}(q_{\text{adapt}}) \le L |q_{\text{final}} - q_{\text{adapt}}|$.
3. Before rounding, $|(1-\gamma) q_0 + \gamma q_{\text{adapt}} - q_{\text{adapt}}| = (1-\gamma) |q_0 - q_{\text{adapt}}|$.
4. Rounding changes a real number by at most $1/2$. Thus $|q_{\text{final}} - q_{\text{adapt}}| \le (1-\gamma) |q_0 - q_{\text{adapt}}| + 1/2$.
5. Substituting yields the result. $\blacksquare$

---

## 5. Commitment Cost vs. Decision Cost Decomposition
- Commitment cost ratio: $C_{\text{commit}} = \frac{\mathcal{R}(q_b^*)}{\mathcal{R}(q^*)} - 1$.
- Decision cost ratio: $C_{\text{decision}} = \frac{\mathcal{R}(\widehat{q}_b)}{\mathcal{R}(q_b^*)} - 1$.
- Exact factorization identity: $\boxed{1 + C_{\text{total}} = (1 + C_{\text{decision}})(1 + C_{\text{commit}})}$.
