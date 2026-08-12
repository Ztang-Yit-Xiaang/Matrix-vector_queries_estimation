# Theorem: Near-Oracle Allocation Regret & Soft Safety Layer Cost

**Classification**: `PROVED` (Tail-to-Risk implication & $2\varepsilon_R$ Argmin Lemma) & `PROVED UNDER EXPLICIT ASSUMPTIONS` (Soft Safety Layer bound under Lipschitz risk).

---

## 1. Theorem Statements

### Theorem 2 (Tail Error Implies Oracle Risk Error)
Let $\mathcal{A}_b = \{ (q, r) : b \le r \le q, m - q - r \ge 1 \}$ be the feasible action set at pilot stage $b$.
Suppose the pilot tail estimate $\widehat{T}_b(r)$ satisfies $|\widehat{T}_b(r) - T(r)| \le U_b(r, \delta)$ for all $(q, r) \in \mathcal{A}_b$.
Define estimated risk $\widehat{R}_b(q, r) = \frac{2 \widehat{T}_b(r)}{m - q - r}$ and true oracle risk $R(q, r) = \frac{2 T(r)}{m - q - r}$.

Then the uniform risk approximation error is bounded by:
$$\boxed{|\widehat{R}_b(q, r) - R(q, r)| \le \frac{2 U_b(r, \delta)}{m - q - r} \le \varepsilon_R(b, \delta)}$$
where $\varepsilon_R(b, \delta) \equiv \max_{(q, r) \in \mathcal{A}_b} \frac{2 U_b(r, \delta)}{m - q - r}$.

---

### Theorem 3 (The $2\varepsilon_R$ Near-Oracle Regret Lemma)
Let $a = (q, r) \in \mathcal{A}_b$ represent an action.
Let $a^* = \arg\min_{a \in \mathcal{A}_b} R(a)$ be the true oracle action, and $\widehat{a} = \arg\min_{a \in \mathcal{A}_b} \widehat{R}_b(a)$ be the action selected from the estimated risk curve.

If $|\widehat{R}_b(a) - R(a)| \le \varepsilon_R$ for all $a \in \mathcal{A}_b$, then:
$$\boxed{R(\widehat{a}) - R(a^*) \le 2 \varepsilon_R}$$

Combining Theorems 2 and 3 yields the rank-aware tail-error-to-regret bound:
$$\boxed{R(\widehat{a}) - R(a^*) \le 4 \max_{(q, r) \in \mathcal{A}_b} \frac{U_b(r, \delta)}{m - q - r}}$$

---

### Theorem 4 (Cost of the Soft Allocation Safety Layer)
Assume the full-rank oracle planning risk $R(q)$ satisfies a Lipschitz condition $|R(q_1) - R(q_2)| \le L |q_1 - q_2|$ over the feasible integer allocation range.
Let $q_0$ be the standard Hutch++ default allocation, $\widehat{q} = \arg\min_q \widehat{R}(q)$, and $x = (1-\gamma) q_0 + \gamma \widehat{q}$ ($0 \le \gamma \le 1$).
Let $q_{\text{final}} = \operatorname{round}(x)$.

Then:
$$\boxed{R(q_{\text{final}}) - R(q^*) \le 2 \varepsilon_R + L \left[ (1 - \gamma) |q_0 - \widehat{q}| + \frac{1}{2} \right]}$$

---

## 2. Complete Mathematical Proofs

### Proof of Theorem 3 (The $2\varepsilon_R$ Argmin Regret Lemma)
1. At the selected action $\widehat{a}$:
   $$R(\widehat{a}) \le \widehat{R}_b(\widehat{a}) + \varepsilon_R$$
2. Because $\widehat{a} = \arg\min_a \widehat{R}_b(a)$, we have $\widehat{R}_b(\widehat{a}) \le \widehat{R}_b(a^*)$:
   $$R(\widehat{a}) \le \widehat{R}_b(a^*) + \varepsilon_R$$
3. Applying uniform approximation at the true oracle action $a^*$:
   $$\widehat{R}_b(a^*) \le R(a^*) + \varepsilon_R$$
4. Substituting into step 2:
   $$R(\widehat{a}) \le R(a^*) + 2 \varepsilon_R \implies \boxed{R(\widehat{a}) - R(a^*) \le 2 \varepsilon_R}$$
$\blacksquare$

---

### Proof of Theorem 4 (Soft Safety Layer Regret Bound)
1. By nearest-integer rounding, $|q_{\text{final}} - x| \le \frac{1}{2}$.
2. The distance between real-valued soft allocation $x$ and adaptive minimizer $\widehat{q}$ is:
   $$|x - \widehat{q}| = |(1-\gamma) q_0 + \gamma \widehat{q} - \widehat{q}| = (1-\gamma) |q_0 - \widehat{q}|$$
3. By triangle inequality:
   $$|q_{\text{final}} - \widehat{q}| \le |q_{\text{final}} - x| + |x - \widehat{q}| \le (1-\gamma) |q_0 - \widehat{q}| + \frac{1}{2}$$
4. Decomposition of risk regret:
   $$\begin{aligned}
   R(q_{\text{final}}) - R(q^*) &= \left( R(q_{\text{final}}) - R(\widehat{q}) \right) + \left( R(\widehat{q}) - R(q^*) \right) \\
   &\le L |q_{\text{final}} - \widehat{q}| + 2 \varepsilon_R \\
   &\le 2 \varepsilon_R + L \left[ (1-\gamma) |q_0 - \widehat{q}| + \frac{1}{2} \right]
   \end{aligned}$$
$\blacksquare$

---

## 3. Commitment Cost vs. Decision Cost Decomposition
Define:
- Pilot-constrained oracle: $q_b^* = \arg\min_{q \ge b} R(q)$.
- Commitment cost ratio: $C_{\text{commit}} = \frac{R(q_b^*)}{R(q^*)} - 1$.
- Decision cost ratio: $C_{\text{decision}} = \frac{R(\widehat{q}_b)}{R(q_b^*)} - 1$.

Multiplicative regret factorization identity:
$$\boxed{1 + C_{\text{total}} = (1 + C_{\text{decision}}) (1 + C_{\text{commit}})}$$
