# Theorem 1: Sequential-Pilot Adaptive Unbiasedness

**Classification**: `PROVED`

---

## 1. Theorem Statement

Let $A \in \mathbb{R}^{d \times d}$ be any square real matrix (not necessarily symmetric or PSD for unbiasedness).
Let $m$ be the total matrix-vector query budget.

Suppose a sequential pilot algorithm evaluates candidate pilot sizes $b_1 < b_2 < \dots < b_K$ and stops at a random stage $B \in \{b_1, \dots, b_K\}$ according strictly to information observed before fresh residual probes are sampled.

After stopping at stage $B$, the algorithm may:
- Select a target sketch width $q = q_{\text{target}} \ge B$;
- Acquire additional sketch columns;
- Construct an orthonormal basis $Q \in \mathbb{R}^{d \times r}$ with actual rank $r = r_{\text{actual}} \le q$;
- Compute $A Q$ (costing $r$ queries);
- Determine the number of residual probes $\ell = m - q_{\text{target}} - r_{\text{actual}} \ge 1$.

Let $\mathcal{G}$ denote the $\sigma$-algebra containing **all randomness and decisions made before the residual probes are drawn**, including:
$$B, \quad q_{\text{target}}, \quad Q, \quad r_{\text{actual}}, \quad A Q, \quad \ell$$
all pilot observations, Ritz values, model weights, stopping decisions, and extension sketch randomness.

Assume the residual probes $g_1, g_2, \dots, g_\ell \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, I_d)$ are independent of $\mathcal{G}$.

The estimator:
$$\widehat{t} = \operatorname{tr}(Q^T A Q) + \frac{1}{\ell} \sum_{j=1}^\ell g_j^T (I - Q Q^T) A (I - Q Q^T) g_j$$
satisfies:
$$\boxed{\mathbb{E}[\widehat{t}] = \operatorname{tr}(A)}$$
unconditionally, even though $B, q, r, Q, \ell$ are random variables.

---

## 2. Complete Mathematical Proof

### Step 1: Conditioning on Pre-Residual $\sigma$-Algebra $\mathcal{G}$
Conditioning on $\mathcal{G}$ fixes the orthonormal basis $Q \in \mathbb{R}^{d \times r}$, the orthogonal residual projector $R = I - Q Q^T$, and the residual probe count $\ell \ge 1$.

For each probe $g_j \sim \mathcal{N}(0, I_d)$, its conditional expectation and covariance are:
$$\mathbb{E}[g_j \mid \mathcal{G}] = 0, \qquad \mathbb{E}[g_j g_j^T \mid \mathcal{G}] = I_d$$

### Step 2: Conditional Expectation of Each Residual Probe
Using trace cyclicity:
$$\begin{aligned}
\mathbb{E}[g_j^T R A R g_j \mid \mathcal{G}] &= \mathbb{E}[\operatorname{tr}(g_j^T R A R g_j) \mid \mathcal{G}] \\
&= \mathbb{E}[\operatorname{tr}(R A R g_j g_j^T) \mid \mathcal{G}] \\
&= \operatorname{tr}\left( R A R \, \mathbb{E}[g_j g_j^T \mid \mathcal{G}] \right) \\
&= \operatorname{tr}(R A R)
\end{aligned}$$

### Step 3: Conditional Expectation of the Averaged Residual Term
Because $\ell$ is fixed conditional on $\mathcal{G}$:
$$\mathbb{E}\left[ \frac{1}{\ell} \sum_{j=1}^\ell g_j^T R A R g_j \;\middle|\; \mathcal{G} \right] = \frac{1}{\ell} \sum_{j=1}^\ell \operatorname{tr}(R A R) = \operatorname{tr}(R A R)$$

Hence, the full conditional expectation of the estimator is:
$$\mathbb{E}[\widehat{t} \mid \mathcal{G}] = \operatorname{tr}(Q^T A Q) + \operatorname{tr}(R A R)$$

### Step 4: Simplification of Trace Identities
Let $P = Q Q^T$. By trace cyclicity:
$$\operatorname{tr}(Q^T A Q) = \operatorname{tr}(A Q Q^T) = \operatorname{tr}(A P)$$

Because $R = I - P$ is an orthogonal projector ($R^2 = R$):
$$\begin{aligned}
\operatorname{tr}(R A R) &= \operatorname{tr}(A R^2) = \operatorname{tr}(A R) \\
&= \operatorname{tr}(A (I - P)) = \operatorname{tr}(A) - \operatorname{tr}(A P)
\end{aligned}$$

Summing both terms:
$$\mathbb{E}[\widehat{t} \mid \mathcal{G}] = \operatorname{tr}(A P) + \operatorname{tr}(A) - \operatorname{tr}(A P) = \operatorname{tr}(A)$$

### Step 5: Unconditional Expectation by Tower Property
By the law of total expectation (Tower Property):
$$\mathbb{E}[\widehat{t}] = \mathbb{E}\left[ \mathbb{E}[\widehat{t} \mid \mathcal{G}] \right] = \mathbb{E}[\operatorname{tr}(A)] = \operatorname{tr}(A)$$
$$\boxed{\mathbb{E}[\widehat{t}] = \operatorname{tr}(A)}$$
$\blacksquare$

---

## 3. Important Theoretical Takeaway
No optional stopping theorem is required for unbiasedness. The only essential requirement is that **all adaptive decisions (stopping time $B$, target $q_{\text{target}}$, basis $Q$, and probe count $\ell$) are determined before fresh residual probes $g_j$ are generated**.
