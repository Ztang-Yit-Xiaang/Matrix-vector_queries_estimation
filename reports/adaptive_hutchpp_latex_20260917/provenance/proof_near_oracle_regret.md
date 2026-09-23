# Theorems 4--6 and 11--13: Regret, Baseline Safety, and Marginal Allocation

**Classification**: `PROVED` (Theorems 4, 5, 12, Lemma 12.1, Theorem 13, Theorem 14, Corollary 14.1, and Lemma 14.2) and `PROVED UNDER EXPLICIT ASSUMPTIONS` (Theorem 6 under Lipschitz risk; Theorem 11 and its corollaries on stated simultaneous-confidence events).

---

## 1. Commitment Regret vs. Decision Regret Factorization

Let $q^* = \arg\min_q \mathcal{R}(q)$ be the unrestricted full-rank oracle allocation.
Once the sequential pilot expands to random size $B$, allocations below $B$ are impossible.
Define the pilot-constrained oracle allocation:
$$\boxed{q_B^* \equiv \arg\min_{q \ge B} \mathcal{R}(q)}$$

Let $q_{\text{sel}}$ be the final allocation selected by the algorithm.
Decompose allocation error into two sequential stages:
$$q^* \quad \longrightarrow \quad q_B^* \quad \longrightarrow \quad q_{\text{sel}}$$

1. **Pilot Commitment Regret Ratio**:
   $$\boxed{C_{\text{commit}} \equiv \frac{\mathcal{R}(q_B^*)}{\mathcal{R}(q^*)} - 1}$$
   (Measures the cost of acquiring too much pilot information).

2. **Allocation Decision Regret Ratio**:
   $$\boxed{C_{\text{decision}} \equiv \frac{\mathcal{R}(q_{\text{sel}})}{\mathcal{R}(q_B^*)} - 1}$$
   (Measures the cost of making the wrong allocation decision after pilot information is acquired).

3. **Multiplicative Regret Factorization Identity**:
   $$\frac{\mathcal{R}(q_{\text{sel}})}{\mathcal{R}(q^*)} = \left(1 + C_{\text{commit}}\right) \left(1 + C_{\text{decision}}\right) \implies \boxed{1 + C_{\text{total}} = (1 + C_{\text{commit}})(1 + C_{\text{decision}})}$$

---

## 2. Theorem 12: Marginal Allocation Theorem and Marginal Risk Quantity $M(q)$

Let $\lambda_1\ge\cdots\ge\lambda_d\ge0$. Consider full-rank oracle risk $\mathcal{R}(q) = \frac{2 T(q)}{m - 2q}$ where $T(q) = \sum_{i=q+1}^d \lambda_i^2$. Assume $q$ is an integer for which both $q$ and $q+1$ are feasible:
$$
0\le q<d,
\qquad
m-2q-2>0.
$$
Using $T(q+1) = T(q) - \lambda_{q+1}^2$:

$$\begin{aligned}
\mathcal{R}(q+1) - \mathcal{R}(q) &= \frac{2(T(q) - \lambda_{q+1}^2)}{m - 2(q+1)} - \frac{2 T(q)}{m - 2q} \\
&= \frac{2(T(q) - \lambda_{q+1}^2)(m - 2q) - 2 T(q)(m - 2q - 2)}{(m - 2q)(m - 2q - 2)} \\
&= \frac{-2(m - 2q)\lambda_{q+1}^2 + 4 T(q)}{(m - 2q)(m - 2q - 2)}
\end{aligned}$$

Thus $\mathcal{R}(q+1) < \mathcal{R}(q)$ if and only if $-2(m - 2q)\lambda_{q+1}^2 + 4 T(q) < 0$.
Dividing by $-2$:

$$\boxed{\mathcal{R}(q+1) < \mathcal{R}(q) \iff (m - 2q)\lambda_{q+1}^2 > 2 T(q)}$$

> **Physical Meaning**: Adding one more low-rank direction improves oracle risk if and only if the **energy removed by one more direction** ($(m - 2q)\lambda_{q+1}^2$) exceeds the **cost of losing two residual queries** ($2 T(q)$).

Define the **Marginal Risk Quantity**:
$$\boxed{M(q) \equiv (m - 2q)\lambda_{q+1}^2 - 2 T(q)}$$

The positive denominator in the displayed finite difference proves
$$
\operatorname{sign}\!\left(\mathcal R(q+1)-\mathcal R(q)\right)
=-\operatorname{sign}(M(q)).
$$

### Lemma 12.1: Monotonicity and single crossing

Whenever the adjacent marginal quantities are defined and their risk comparisons are feasible,
$$
\begin{aligned}
M(q+1)-M(q)
&=(m-2q-2)\lambda_{q+2}^2-2T(q+1)
 -(m-2q)\lambda_{q+1}^2+2T(q)\\
&=(m-2q-2)\left(\lambda_{q+2}^2-\lambda_{q+1}^2\right)\le0.
\end{aligned}
$$
Thus $M(q)$ is nonincreasing on the feasible integer grid. Consequently, $\mathcal R(q)$ decreases while $M(q)>0$ and increases after $M(q)<0$. If $M(q)=0$, the adjacent risks are equal; equality cases are plateaus and do not support a unique-minimizer claim. $\blacksquare$

### Theorem 13: Corrected exact step-spectrum minimizer

Let
$$
0<\eta<1,
\qquad
\lambda_i=1\quad(i\le r_\star),
\qquad
\lambda_i=\eta\quad(i>r_\star),
$$
and assume $r_\star$ is a feasible full-rank allocation. If
$$
\boxed{
2r_\star+2(d-r_\star)\eta^2<m<2d,
}
$$
then the unique minimizer of $\mathcal R(q)$ on the feasible integer allocation grid is $q^*=r_\star$.

For every feasible $q<r_\star$,
$$
T(q)=r_\star-q+(d-r_\star)\eta^2
$$
and hence
$$
M(q)=m-2r_\star-2(d-r_\star)\eta^2>0.
$$
The risk therefore decreases strictly up to the knee. For every feasible $q\ge r_\star$ with $q+1$ feasible,
$$
T(q)=(d-q)\eta^2,
\qquad
M(q)=(m-2d)\eta^2<0,
$$
so the risk increases strictly after the knee. These two facts prove uniqueness. If either budget inequality is an equality, the corresponding adjacent comparisons form a plateau rather than a strict improvement. $\blacksquare$

> **Correction of the earlier claim.** The condition $m<2d$ controls only the post-knee sign. It does not guarantee that moving toward the knee is beneficial. For example, $(d,m,r_\star,\eta)=(500,160,30,0.5)$ gives $\mathcal R(0)=1.84375<\mathcal R(30)=2.35$, so the statement based on $m<2d$ alone is false.

---

## 3. Theorem 14: Ideal Rank-Aware Marginal Allocation

Let $A\succeq0$ have ordered eigenvalues

$$
\lambda_1\ge\cdots\ge\lambda_d\ge0,
$$

and define

$$
T(r)=\sum_{i=r+1}^d\lambda_i^2.
$$

Consider a feasible old state $(q,r)$ satisfying

$$
0\le r\le q,
\qquad
r<d,
\qquad
D=m-q-r>2.
$$

Assume the successful action $(q,r)\to(q+1,r+1)$ captures exactly a leading eigendirection associated with $\lambda_{r+1}$. The old and new ideal rank-aware risks are

$$
\mathcal R_{\mathrm{rank}}(q,r)=\frac{2T(r)}{D}
$$

and

$$
\mathcal R_{\mathrm{rank}}(q+1,r+1)
=
\frac{2[T(r)-\lambda_{r+1}^2]}{D-2}.
$$

Their exact difference is

$$
\boxed{
\mathcal R_{\mathrm{rank}}(q+1,r+1)
-
\mathcal R_{\mathrm{rank}}(q,r)
=
-\frac{2[D\lambda_{r+1}^2-2T(r)]}{D(D-2)}.
}
$$

Because $D(D-2)>0$,

$$
\boxed{
\mathcal R_{\mathrm{rank}}(q+1,r+1)
<
\mathcal R_{\mathrm{rank}}(q,r)
\iff
D\lambda_{r+1}^2>2T(r).
}
$$

This is an ideal next-eigenvector theorem. A numerical rank increase by itself does not establish that the realized randomized direction removes $\lambda_{r+1}^2$ of tail energy. $\blacksquare$

### Corollary 14.1: Failed-rank action

Assume $D=m-q-r>1$ and consider a sketch query that leaves the accepted basis and rank unchanged:

$$
(q,r)\longrightarrow(q+1,r).
$$

Then

$$
\boxed{
\mathcal R_{\mathrm{rank}}(q+1,r)
-
\mathcal R_{\mathrm{rank}}(q,r)
=
\frac{2T(r)}{D(D-1)}\ge0.
}
$$

The inequality is strict exactly when $T(r)>0$. If an exactly rank-deficient matrix has already had its complete range captured, then $T(r)=0$ and the risk remains zero. Thus a failed query is nonbeneficial but not strictly harmful in this boundary case. $\blacksquare$

### Lemma 14.2: Realized energy-drop marginal

Let $X\in\{G,R\}$ identify Gaussian or Rademacher residual probes. Let $Q$ be the old accepted basis and $Q'$ the nested basis after one accepted direction. Write

$$
\mathcal R_X(Q;q,r)=\frac{2E_X(Q)}{D},
\qquad
D=m-q-r>2,
$$

where $E_G(Q)=\|R_QAR_Q\|_F^2$ and $E_R(Q)=\sum_{i\ne j}(R_QAR_Q)_{ij}^2$. Define

$$
\boxed{
M_X(Q,Q';q,r)
=
D[E_X(Q)-E_X(Q')]-2E_X(Q).
}
$$

For the successful transition $(q,r,Q)\to(q+1,r+1,Q')$,

$$
\boxed{
\mathcal R_X(Q';q+1,r+1)-\mathcal R_X(Q;q,r)
=
-\frac{2M_X(Q,Q';q,r)}{D(D-2)}.
}
$$

Therefore

$$
\boxed{
\mathcal R_X(Q';q+1,r+1)<\mathcal R_X(Q;q,r)
\iff
M_X(Q,Q';q,r)>0.
}
$$

When $E_X(Q)>0$, division by the positive quantity $D E_X(Q)$ gives the equivalent interpretation

$$
\boxed{
\frac{E_X(Q)-E_X(Q')}{E_X(Q)}>\frac{2}{D}.
}
$$

Thus the fractional residual energy removed must exceed the fractional residual-sample capacity lost when $D$ decreases to $D-2$. If $E_X(Q)=0$, this ratio is undefined and strict improvement from zero risk is impossible.

For a failed-rank transition $Q'=Q$ and $(q,r)\to(q+1,r)$, the denominator instead becomes $D-1$, so

$$
\boxed{
\mathcal R_X(Q;q+1,r)-\mathcal R_X(Q;q,r)
=
\frac{2E_X(Q)}{D(D-1)}\ge0.
}
$$

The ideal specialization $Q=V_r$, $Q'=V_{r+1}$, and $E_X=T$ recovers Theorem 14. $\blacksquare$

---

## 4. Three-State Certified Marginal Decision

Suppose $\widehat{M}_b(q)$ estimates $M(q)$ and a nonnegative radius $C_b^M(q)$ is valid on an explicitly stated confidence event:

$$\boxed{
\begin{cases}
\text{Increase } q \to q+1, & \text{if } \widehat{M}_b(q) - C_b^M(q) > 0 \quad (\text{Certified Positive Marginal Gain}) \\
\text{Stop } q, & \text{if } \widehat{M}_b(q) + C_b^M(q) < 0 \quad (\text{Certified Negative Marginal Gain}) \\
\text{Learn More } (B \to B + \Delta b), & \text{if } 0 \in [\widehat{M}_b(q) - C_b^M(q), \widehat{M}_b(q) + C_b^M(q)] \quad (\text{Uncertain})
\end{cases}
}$$

The first two decisions are deterministic implications on the confidence event because the entire interval lies on one side of zero. If stages or actions are selected from the data, validity must be simultaneous over every inspected stage/action pair, or follow from another valid sequential construction. This proof does not construct $C_b^M(q)$; that finite-sample problem remains `OPEN`.

If a policy starts at $q_0$ and accepts only certified beneficial adjacent moves, Theorem 12 and transitivity give baseline safety for the enclosed full-rank oracle risk. Starting below $q_0$ or stopping early because of uncertainty does not by itself imply baseline safety. The rank-aware action $(q,r)$ also requires separate analysis because increasing $q$ need not increase $r$.

---

## 5. Definition and Theorem 4: Fixed-Stage Near-Oracle Regret

Fix a pilot size $b$ and a finite, nonempty feasible set
$$
\mathcal Q_b\subseteq\{q\in\mathbb Z:b\le q,\ m-2q>0\}.
$$
Define
$$
\mathcal R(q)=\frac{2T(q)}{m-2q},
\qquad
\widehat{\mathcal R}_b(q)=\frac{2\widehat T_b(q)}{m-2q}.
$$
Suppose the simultaneous tail-confidence event is
$$
E_b=
\left\{
\forall q\in\mathcal Q_b:
|\widehat T_b(q)-T(q)|\le U_b(q,\delta)
\right\},
$$
where $U_b(q,\delta)\ge0$, and define the induced uniform risk radius
$$
\varepsilon_R(b,\delta)
=
\max_{q\in\mathcal Q_b}
\frac{2U_b(q,\delta)}{m-2q}.
$$
Let
$$
q_b^*\in\arg\min_{q\in\mathcal Q_b}\mathcal R(q),
\qquad
\widehat q_b\in\arg\min_{q\in\mathcal Q_b}\widehat{\mathcal R}_b(q),
$$
with deterministic tie breaking. On $E_b$,
$$
\boxed{
\mathcal R(\widehat q_b)-\mathcal R(q_b^*)
\le2\varepsilon_R(b,\delta)
=4\max_{q\in\mathcal Q_b}\frac{U_b(q,\delta)}{m-2q}.
}
$$

Indeed, $E_b$ implies
$$
|\widehat{\mathcal R}_b(q)-\mathcal R(q)|
\le\varepsilon_R(b,\delta)
\qquad(q\in\mathcal Q_b).
$$
Therefore
$$
\mathcal R(\widehat q_b)
\le\widehat{\mathcal R}_b(\widehat q_b)+\varepsilon_R
\le\widehat{\mathcal R}_b(q_b^*)+\varepsilon_R
\le\mathcal R(q_b^*)+2\varepsilon_R.
$$
The two appearances of $\varepsilon_R$ are the conversions from estimated to true risk at the data-dependent minimizer and at the oracle minimizer. $\blacksquare$

### Corollary 4.1: Rank-Aware Two-Dimensional Action Set

Let $\mathcal{A}_b$ be a finite, nonempty set of feasible actions $a=(q,r)$ satisfying $b\le q$, $0\le r\le q$, and $m-q-r>0$. Define
$$\mathcal{R}_{\text{rank}}(q,r)=\frac{2T(r)}{m-q-r}, \qquad \widehat{\mathcal{R}}_b(q,r)=\frac{2\widehat{T}_b(r)}{m-q-r}.$$
Suppose a simultaneous confidence event satisfies
$$\left|\widehat{T}_b(r)-T(r)\right|\le U_b(r,\delta) \qquad \text{for every }(q,r)\in\mathcal{A}_b.$$
Define
$$\varepsilon_R^{\text{rank}}(b,\delta)=\max_{(q,r)\in\mathcal{A}_b}\frac{2U_b(r,\delta)}{m-q-r}.$$
If $a^*$ minimizes $\mathcal{R}_{\text{rank}}$ and $\widehat a$ minimizes $\widehat{\mathcal{R}}_b$ over $\mathcal{A}_b$, then the same two-conversion argument as Theorem 4 gives
$$\boxed{\mathcal{R}_{\text{rank}}(\widehat a)-\mathcal{R}_{\text{rank}}(a^*)\le 2\varepsilon_R^{\text{rank}}(b,\delta)=4\max_{(q,r)\in\mathcal{A}_b}\frac{U_b(r,\delta)}{m-q-r}.}$$

> **Scope clarification:** This corollary is a deterministic implication on the stated simultaneous-confidence event. Constructing a finite-sample, computable radius $U_b(r,\delta)$ for heterogeneous unknown spectra remains an open part of the project.

---

## 6. Theorem 5: Sequential Union Bound Across Pilot Stages

Let candidate pilot sizes be $b_1, \dots, b_{N_{\mathrm{stg}}}$. Choose per-stage failure probabilities $\delta_1, \dots, \delta_{N_{\mathrm{stg}}}$ such that $\sum_{j=1}^{N_{\mathrm{stg}}} \delta_j \le \delta$.
If stage $j$ satisfies $\mathbb{P}(E_j) \ge 1 - \delta_j$ for $E_j = \{ \forall q \in \mathcal{Q}_{b_j} : |\widehat{T}_{b_j}(q) - T(q)| \le U_{b_j}(q, \delta_j) \}$, then by the union bound:
$$\mathbb{P}\left( \bigcap_{j=1}^{N_{\mathrm{stg}}} E_j \right) \ge 1 - \sum_{j=1}^{N_{\mathrm{stg}}} \delta_j \ge 1 - \delta.$$

Let $J\in\{1,\dots,N_{\mathrm{stg}}\}$ denote the random stopping-stage index and let $B=b_J$. Consequently, at the random stopping stage:
$$\boxed{\mathcal{R}(\widehat{q}_{b_J}) - \mathcal{R}(q_{b_J}^*) \le 2 \varepsilon_R(b_J, \delta_J)}$$
holds with probability at least $1 - \delta$. No optional stopping theorem is required for confidence validity. $\blacksquare$

---

## 7. Theorem 6: Regret of Soft Allocation Safety Shrinkage

Assume oracle risk is defined on an integer interval containing the feasible actions and is discrete-Lipschitz there: $|\mathcal{R}(q+1) - \mathcal{R}(q)| \le L$.
Let the baseline $q_0$ and adaptive action $q_{\text{adapt}}$ lie in this interval, let $0 \le \gamma_{\mathrm{shrink}} \le 1$, and define $q_{\text{final}} = \operatorname{round}\left( (1-\gamma_{\mathrm{shrink}}) q_0 + \gamma_{\mathrm{shrink}} q_{\text{adapt}} \right)$.
If $\mathcal{R}(q_{\text{adapt}}) - \mathcal{R}(q_b^*) \le 2 \varepsilon_R$, then:
$$\boxed{\mathcal{R}(q_{\text{final}}) - \mathcal{R}(q_b^*) \le 2 \varepsilon_R + L \left[ (1 - \gamma_{\mathrm{shrink}}) |q_0 - q_{\text{adapt}}| + \frac{1}{2} \right]}$$

### Proof
1. $\mathcal{R}(q_{\text{final}}) - \mathcal{R}(q_b^*) = \left( \mathcal{R}(q_{\text{final}}) - \mathcal{R}(q_{\text{adapt}}) \right) + \left( \mathcal{R}(q_{\text{adapt}}) - \mathcal{R}(q_b^*) \right)$.
2. By Lipschitz continuity: $\mathcal{R}(q_{\text{final}}) - \mathcal{R}(q_{\text{adapt}}) \le L |q_{\text{final}} - q_{\text{adapt}}|$.
3. Before rounding, $|(1-\gamma_{\mathrm{shrink}}) q_0 + \gamma_{\mathrm{shrink}} q_{\text{adapt}} - q_{\text{adapt}}| = (1-\gamma_{\mathrm{shrink}}) |q_0 - q_{\text{adapt}}|$.
4. Rounding changes a real number by at most $1/2$. Thus $|q_{\text{final}} - q_{\text{adapt}}| \le (1-\gamma_{\mathrm{shrink}}) |q_0 - q_{\text{adapt}}| + 1/2$.
5. Substituting yields the result. $\blacksquare$

---

## 8. Commitment Cost vs. Decision Cost Decomposition
- Assume $\mathcal{R}(q^*)>0$ and $\mathcal{R}(q_b^*)>0$ so the ratios are defined.
- Total cost ratio: $C_{\text{total}} = \frac{\mathcal{R}(\widehat{q}_b)}{\mathcal{R}(q^*)} - 1$.
- Commitment cost ratio: $C_{\text{commit}} = \frac{\mathcal{R}(q_b^*)}{\mathcal{R}(q^*)} - 1$.
- Decision cost ratio: $C_{\text{decision}} = \frac{\mathcal{R}(\widehat{q}_b)}{\mathcal{R}(q_b^*)} - 1$.
- Exact factorization identity: $\boxed{1 + C_{\text{total}} = (1 + C_{\text{decision}})(1 + C_{\text{commit}})}$.

---

## 9. Theorem 11: Baseline-Safe Acceptance at a Fixed Pilot Stage

Fix a pilot stage $b$. Let $\mathcal Q_b$ be a finite, nonempty feasible action set containing the Standard-Hutch++ baseline action $q_0$. Let $\mathcal R(q)$ be the target risk and let $\widehat{\mathcal R}_b(q)$ be its pilot-based estimate. Suppose nonnegative radii $e_b(q)$ define the simultaneous event
$$
E_b=
\left\{
\forall q\in\mathcal Q_b:
\left|\widehat{\mathcal R}_b(q)-\mathcal R(q)\right|\le e_b(q)
\right\}.
$$
Define
$$
U_b^R(q)=\widehat{\mathcal R}_b(q)+e_b(q),
\qquad
L_b^R(q)=\widehat{\mathcal R}_b(q)-e_b(q),
$$
and the certifiably safe set
$$
\mathcal C_b=
\left\{
q\in\mathcal Q_b:U_b^R(q)\le L_b^R(q_0)
\right\}.
$$
The policy selects $q_0$ when $\mathcal C_b=\varnothing$. Otherwise it selects any measurable $q_{\mathrm{safe}}\in\mathcal C_b$; minimizing $U_b^R(q)$ over $\mathcal C_b$ with a deterministic tie-breaking rule is one canonical choice.

On $E_b$, every selected nonbaseline candidate satisfies
$$
\mathcal R(q_{\mathrm{safe}})
\le U_b^R(q_{\mathrm{safe}})
\le L_b^R(q_0)
\le \mathcal R(q_0).
$$
If the certified set is empty, $q_{\mathrm{safe}}=q_0$ and the same conclusion holds by equality. Therefore,
$$
\boxed{
\Pr(E_b)\ge1-\delta
\quad\Longrightarrow\quad
\Pr\!\left(\mathcal R(q_{\mathrm{safe}})\le\mathcal R(q_0)\right)
\ge1-\delta.
}
$$

### Proof audit

The first and last inequalities in the displayed chain are precisely the two sides of the confidence event. The middle inequality is the definition of membership in $\mathcal C_b$. The fallback case is deterministic. These arguments exhaust the cases. $\blacksquare$

> **Why simultaneous validity is necessary.** The candidate is selected after inspecting the entire estimated risk curve. Separate pointwise statements for each fixed $q$ do not in general control the error at this data-dependent choice. The event must cover every action that the selection rule may choose, or a separate valid post-selection argument must be supplied.

### Corollary 11.1: Risk-Difference Certificate

For $q\in\mathcal Q_b$, define
$$
\Delta(q)=\mathcal R(q)-\mathcal R(q_0).
$$
On any event for which
$$
\left|\widehat\Delta_b(q)-\Delta(q)\right|\le C_b(q),
$$
the acceptance condition
$$
\widehat\Delta_b(q)+C_b(q)\le0
$$
implies
$$
\Delta(q)
\le \widehat\Delta_b(q)+C_b(q)
\le0.
$$
Hence $\mathcal R(q)\le\mathcal R(q_0)$. Replacing the final acceptance inequality by a strict inequality gives strict risk improvement on the same confidence event. This is a deterministic implication; constructing a computable, simultaneously valid $C_b(q)$ is open.

### Corollary 11.2: Rank-Aware Baseline Safety

For a rank-deficient problem, let the action be $a=(q,r)$ and define
$$
\mathcal R_{\mathrm{rank}}(a)
=
\frac{2T(r)}{m-q-r}.
$$
Let $\mathcal A_b$ be a finite, nonempty feasible action set containing the baseline action
$$
a_0=(q_0,r_0).
$$
Suppose simultaneous action-level confidence bounds satisfy
$$
L_b^R(a)\le\mathcal R_{\mathrm{rank}}(a)\le U_b^R(a)
\qquad\text{for every }a\in\mathcal A_b.
$$
Accept only actions in
$$
\mathcal C_b^{\mathrm{rank}}
=
\left\{a\in\mathcal A_b:U_b^R(a)\le L_b^R(a_0)\right\},
$$
with fallback to $a_0$. Repeating the three-inequality argument from Theorem 11 yields
$$
\boxed{
\mathcal R_{\mathrm{rank}}(a_{\mathrm{safe}})
\le
\mathcal R_{\mathrm{rank}}(a_0)
}
$$
on the simultaneous event. The baseline is the pair $(q_0,r_0)$, not merely the scalar $q_0$; this prevents the argument from silently imposing $r=q$.

### Corollary 11.3: Sequential Stopping Stages

Let the candidate pilot stages be $b_1,\ldots,b_{N_{\mathrm{stg}}}$, and assign stage-wise failure probabilities $\delta_j$ with
$$
\sum_{j=1}^{N_{\mathrm{stg}}}\delta_j\le\delta.
$$
Suppose the simultaneous action-level event $E_{b_j}$ required by Theorem 11 satisfies
$$
\Pr(E_{b_j})\ge1-\delta_j
$$
at every stage. The union bound gives
$$
\Pr\!\left(\bigcap_{j=1}^{N_{\mathrm{stg}}}E_{b_j}\right)
\ge1-\sum_{j=1}^{N_{\mathrm{stg}}}\delta_j
\ge1-\delta.
$$
On this intersection the Theorem 11 comparison is valid at every stage, hence also at the random stopping stage $B=b_J$. No optional-stopping theorem is needed because validity was established simultaneously across the finite stage grid.

### Scope, Heuristic Status, and Open Work

- **`PROVED UNDER AN EXPLICIT SIMULTANEOUS-CONFIDENCE EVENT`:** Theorem 11 and Corollaries 11.1--11.3 are deterministic implications on their stated events.
- **`OPEN`:** No finite-sample, computable construction of $e_b(q)$ or $C_b(q)$ is supplied for heterogeneous unknown spectra.
- **`HEURISTIC`:** The fixed asymmetric guard $q\in[q_0,q_0+4]$ is not the confidence-certified policy of Theorem 11.
- **`EMPIRICALLY OBSERVED`:** Held-out MSE comparisons assess the heuristic but do not prove baseline safety.
- **Risk scope:** A certificate applies only to the risk quantity enclosed by its confidence bounds. Certifying an oracle surrogate such as $2T(q)/(m-2q)$ does not automatically certify the actual estimator MSE, especially when realized rank, pilot construction, or probe distribution changes the target.
