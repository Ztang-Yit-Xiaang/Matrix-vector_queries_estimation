# Theorems 9, 10 & 15A: Step Ritz Structure and Gaussian Knee Detection

**Classification**: `PROVED` (Theorems 9, 10, and 15A under their stated assumptions) and `PROVED UNDER AN EXPLICIT RANDOM-MATRIX EVENT` (Corollary 10.1). The corrected allocation result is Theorem 13 in [`proof_near_oracle_regret.md`](proof_near_oracle_regret.md).

**Notation**: See [`proof_notation.md`](proof_notation.md). Uppercase $S$ is the sketching matrix, $r_\star$ is the true step/knee rank, and lowercase $s_i$ denotes a principal-angle singular value.

---

## 1. Theorem 9: Ritz Values for Ideal Step Spectrum

Let $A = \eta I_d + (1 - \eta) U U^T$ where $U \in \mathbb{R}^{d \times r_\star}$, $U^T U = I_{r_\star}$, and $0 < \eta < 1$.
$A$ has eigenvalues $1$ (multiplicity $r_\star$) and $\eta$ (multiplicity $d - r_\star$).

Let $Q \in \mathbb{R}^{d \times b}$, $Q^T Q = I_b$, where pilot size $b = r_\star + p \ge r_\star + 1$ ($p \ge 1$).
Let $s_1 \ge \dots \ge s_{r_\star} \ge 0$ be the singular values of $U^T Q \in \mathbb{R}^{r_\star \times b}$.
Let $\theta_1 \ge \dots \ge \theta_b$ be the ordered Ritz values of $Q^T A Q$.

Then:
1. **Top-$r_\star$ Ritz Values**:
   $$\boxed{\theta_i = \eta + (1 - \eta) s_i^2, \qquad 1 \le i \le r_\star}$$
2. **Exact Post-Knee Ritz Noise Floor**:
   $$\boxed{\theta_{r_\star+1} = \theta_{r_\star+2} = \dots = \theta_b = \eta \qquad (\text{exact!})}$$
3. **Exact Ritz Ratio**:
   $$\boxed{\frac{\theta_{r_\star}}{\theta_{r_\star+1}} = \frac{\eta + (1 - \eta) s_{r_\star}^2}{\eta}}$$

### Proof
$Q^T A Q = \eta I_b + (1 - \eta) (U^T Q)^T (U^T Q)$.
The eigenvalues of $(U^T Q)^T (U^T Q) \in \mathbb{R}^{b \times b}$ consist of $s_1^2, \dots, s_{r_\star}^2$ (some of which may be zero) together with exactly $b-r_\star$ additional zeros.
Because $\operatorname{rank}(U^T Q) \le r_\star$, it has at least $b - r_\star = p$ exact zero eigenvalues.
Adding $\eta I_b$ shifts all eigenvalues by $\eta$. Thus $\theta_{r_\star+1} = \dots = \theta_b = \eta$. $\blacksquare$

---

## 2. Theorem 10: Deterministic Principal-Angle Reduction (Using Sketch Matrix $S$)

Let $S \in \mathbb{R}^{d \times (r_\star+p)}$ be a sketching matrix (using standard $S$ notation). Express $S$ in the eigenbasis of $A$:
$$\begin{bmatrix} U^T \\ U_\perp^T \end{bmatrix} S = \begin{bmatrix} S_1 \\ S_2 \end{bmatrix}, \qquad S_1 \in \mathbb{R}^{r_\star \times (r_\star+p)}, \quad S_2 \in \mathbb{R}^{(d-r_\star) \times (r_\star+p)}$$
Let $Y = A S$ and $Q = \operatorname{orth}(Y)$. Assume $S_1$ has full row rank.

1. **Principal-Angle Tangent Bound**:
   $$\boxed{\tan \Theta_{\max} \le \eta \|S_2 S_1^\dagger\|_2}$$
2. **Singular Value Lower Bound**:
   $$\boxed{s_{r_\star}^2 = \cos^2 \Theta_{\max} \ge \frac{1}{1 + \eta^2 \|S_2 S_1^\dagger\|_2^2}}$$
3. **Exact Ritz Ratio Lower Bound**:
   $$\boxed{\frac{\theta_{r_\star}}{\theta_{r_\star+1}} \ge 1 + \frac{1 - \eta}{\eta [1 + \eta^2 \|S_2 S_1^\dagger\|_2^2]}}$$

### Proof
Using the orthogonal decomposition $[U,U_\perp]$, we have $Y=U S_1+\eta U_\perp S_2$. Since $S_1$ has full row rank, $S_1S_1^\dagger=I_{r_\star}$, and therefore
$$Y S_1^\dagger=U+\eta U_\perp S_2 S_1^\dagger=[U,U_\perp]\begin{bmatrix}I_{r_\star}\\F\end{bmatrix}, \qquad F=\eta S_2 S_1^\dagger.$$
Thus, $\operatorname{range}(Y)$ contains the graph subspace $\mathcal{G}=[U,U_\perp]\operatorname{range}\begin{bmatrix} I_{r_\star} \\ F \end{bmatrix}$, with $\tan \Theta_{\max}(\operatorname{range}(U), \mathcal{G}) = \|F\|_2 = \eta \|S_2 S_1^\dagger\|_2$.
Since $\mathcal{G} \subseteq \operatorname{range}(Q)$, principal angle to $Q$ cannot be larger. Thus $\tan \Theta_{\max} \le \eta \|S_2 S_1^\dagger\|_2$, and $\cos^2 \Theta_{\max} = \frac{1}{1 + \tan^2 \Theta_{\max}} \ge \frac{1}{1 + \eta^2 \|S_2 S_1^\dagger\|_2^2}$. $\blacksquare$

---

## 3. Corrected Step-Allocation Cross-Reference (Theorem 13)

Let $A$ have step spectrum $\lambda_i=1$ for $i\le r_\star$ and $\lambda_i=\eta$ for $i>r_\star$, with $0<\eta<1$ and feasible $r_\star$. For full-rank oracle risk $\mathcal R(q)=2T(q)/(m-2q)$, Theorem 13 proves
$$
\boxed{
q^*=r_\star
\quad\text{uniquely when}\quad
2r_\star+2(d-r_\star)\eta^2<m<2d.
}
$$
The two budget inequalities control different sides of the knee:

1. **Post-Knee Regime ($q \ge r_\star$)**:
   $$
   M(q)=(m-2d)\eta^2<0
   $$
   follows from $m<2d$. Therefore
   $$\boxed{q \ge r_\star \implies \mathcal{R}(q+1) > \mathcal{R}(q)}$$
   for every feasible adjacent comparison after the knee.

2. **Pre-Knee Regime ($q < r_\star$)**:
   $$
   M(q)=m-2r_\star-2(d-r_\star)\eta^2>0
   $$
   requires the separate lower bound on $m$. It implies strict decrease toward the knee.

Together the signs yield the unique minimizer. If either inequality is an equality, the corresponding adjacent risks form a plateau. The earlier statement based on $m<2d$ alone was false; it controlled only the post-knee sign. The full proof and the counterexample $(d,m,r_\star,\eta)=(500,160,30,0.5)$ are in Theorem 13. This allocation theorem is distinct from Theorems 9, 10, and 15A, which concern Ritz structure and knee detection.

---

## 4. Corollary 10.1: Generic High-Probability Knee-Detection Condition

In addition to the assumptions of Theorem 10, assume $r_Y=\operatorname{rank}(Y)\ge r_\star+1$ and define $p_Y=r_Y-r_\star\ge1$. Then $\theta_{r_\star+1}$ exists. Applying Theorem 9 with its basis dimension set to $r_Y$ gives exactly $p_Y$ post-knee Ritz values at $\eta$. Here $p=b-r_\star$ still denotes sketch oversampling, whereas $p_Y$ counts realized post-knee Ritz positions. A sufficient condition for the two counts to agree is that $b\le d$ and $S$ has full column rank; since $A\succ0$, this gives $r_Y=b$ and $p_Y=p$.

Suppose a random matrix bound establishes $\mathbb{P}\left( \|S_2 S_1^\dagger\|_2 \le K_{r_\star, d, p, \delta} \right) \ge 1 - \delta$.
Then a sufficient condition for $\mathbb{P}\left( \frac{\theta_{r_\star}}{\theta_{r_\star+1}} > \tau_{\mathrm R} \right) \ge 1 - \delta$ is:
$$\boxed{\frac{\eta + \frac{1-\eta}{1 + \eta^2 K_{r_\star, d, p, \delta}^2}}{\eta} > \tau_{\mathrm R}}$$

> **Important Limitation & Unsupported Formula Status**:
> The explicit guessed formula $p \ge \left\lceil \frac{\log(1/\delta) + \log(d-r_\star) + \log(\tau_{\mathrm R} r_\star)}{\log(1/\eta)} \right\rceil$ is **WITHDRAWN as unproved**.
> A generic or sharp evaluation of $K_{r_\star,d,p,\delta}$ depends on the sketch distribution. Section 5 supplies a valid conservative specialization for an i.i.d. standard Gaussian sketch; sharp and distribution-specific numerical versions remain open.
> Full row rank of $S_1$ alone is not enough to define the Ritz ratio: repeated sketch columns can yield $\operatorname{rank}(Y)=r_\star$, leaving $\theta_{r_\star+1}$ undefined.

---

## 5. Theorem 15A: An Explicit Standard-Gaussian Constant

This theorem supplies the previously unnamed random-matrix result in Theorem 15 of the dependency-chain note (Corollary 10.1 above). It is a valid, deliberately conservative specialization to an i.i.d. standard Gaussian sketch.

Assume $r_\star\ge1$, $p\ge4$, $b=r_\star+p\le d$, $0<\delta<1$, and $S\in\mathbb{R}^{d\times b}$ has independent $\mathcal N(0,1)$ entries. By rotational invariance, the blocks
$$S_1=U^T S\in\mathbb{R}^{r_\star\times b},\qquad S_2=U_\perp^T S\in\mathbb{R}^{(d-r_\star)\times b}$$
are independent standard Gaussian matrices. Define
$$u_\delta=\sqrt{2\log(2/\delta)},\qquad z_{p,\delta}=(2/\delta)^{1/(p+1)},$$
and
$$\boxed{K^{\mathrm G}_{r_\star,d,p,\delta}=\left(\sqrt{d-r_\star}+\sqrt{r_\star+p}+u_\delta\right)\frac{e\sqrt{r_\star+p}}{p+1}\,z_{p,\delta}.}$$
Then
$$\boxed{\mathbb P\!\left(\|S_2 S_1^\dagger\|_2\le K^{\mathrm G}_{r_\star,d,p,\delta}\right)\ge1-\delta.}$$

### Proof
Proposition 10.4 of Halko, Martinsson, and Tropp gives, for $p\ge4$ and every scalar $z\ge1$,
$$\mathbb P\!\left(\|S_1^\dagger\|_2\ge \frac{e\sqrt{r_\star+p}}{p+1}z\right)\le z^{-(p+1)}.$$
Choosing $z=z_{p,\delta}$ makes this failure probability $\delta/2$. Their Gaussian expectation and concentration bounds also give
$$\mathbb P\!\left(\|S_2\|_2\ge\sqrt{d-r_\star}+\sqrt{r_\star+p}+u_\delta\right)\le e^{-u_\delta^2/2}=\delta/2.$$
On the intersection of the two events, submultiplicativity yields
$$\|S_2 S_1^\dagger\|_2\le\|S_2\|_2\|S_1^\dagger\|_2\le K^{\mathrm G}_{r_\star,d,p,\delta}.$$
The union bound proves the claim. Moreover, $S$ has full column rank almost surely and $A\succ0$, so $\operatorname{rank}(Y)=b\ge r_\star+1$ almost surely. Thus the post-knee Ritz value required by Corollary 10.1 exists. $\blacksquare$

> **Source:** N. Halko, P.-G. Martinsson, and J. A. Tropp, “Finding Structure with Randomness,” *SIAM Review* 53(2), 2011, Proposition 10.4 and the Gaussian norm bounds in Propositions 10.1 and 10.3, https://doi.org/10.1137/090771806.

> **Sharpness limitation:** This result closes the logical gap for Gaussian sketches, but the separate-norm product bound can be loose. A direct analysis of the matrix-variate ratio $S_2S_1^\dagger$, or a bound that exploits all $b$ dimensions of $\operatorname{range}(Y)$, could give a materially smaller sufficient pilot size.

### Rademacher implementation route

The current sequential implementation uses independent Rademacher sketch columns rather than Gaussian columns. The Gaussian constant above therefore does not directly certify that code path. Nevertheless, a new random-matrix theory is not required at the qualitative level.

Indeed, if $\xi_j\in\{-1,+1\}^d$ is a Rademacher column, then $U^T\xi_j$ and $U_\perp^T\xi_j$ are isotropic subgaussian vectors with a universal subgaussian-norm bound. The rows of $S_1^T$ and $S_2^T$ are independent across $j$. Applying the two-sided singular-value theorem for matrices with independent isotropic subgaussian rows gives a universal constant $C_{\mathrm R}>0$ such that, with $v_\delta=\sqrt{\log(4/\delta)}$ and positive denominator,
$$\boxed{K^{\mathrm{Rad}}_{r_\star,d,p,\delta}=\frac{\sqrt{r_\star+p}+C_{\mathrm R}(\sqrt{d-r_\star}+v_\delta)}{\sqrt{r_\star+p}-C_{\mathrm R}(\sqrt{r_\star}+v_\delta)}}$$
satisfies
$$\mathbb P\!\left(\|S_2 S_1^\dagger\|_2\le K^{\mathrm{Rad}}_{r_\star,d,p,\delta}\right)\ge1-\delta.$$
Dependence between $S_1$ and $S_2$ is harmless here because the union bound does not require independence.

> **Source and remaining limitation:** This route uses Roman Vershynin, *High-Dimensional Probability*, Theorem 4.6.1 (two-sided bounds for matrices with independent isotropic subgaussian rows), https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-1.pdf. The published theorem uses an unspecified universal constant. Therefore it proves existence and scaling of a Rademacher certificate, but an implementation-ready numerical pilot threshold still requires explicit constant tracking or a sharper Rademacher-specific result.

---

## 6. The Role and Admissible Range of the Gap Threshold

To avoid confusing a ratio threshold with a log-gap threshold, write
$$\tau_{\mathrm R}>1\qquad\text{for the Ritz-ratio threshold},$$
and
$$\gamma_{\mathrm{gap}}=\log\tau_{\mathrm R}>0\qquad\text{for the log-gap threshold}.$$
The exact step-spectrum ratio is
$$\frac{\theta_{r_\star}}{\theta_{r_\star+1}}=1+\frac{1-\eta}{\eta}s_{r_\star}^2.$$
Therefore the detector condition is equivalent to
$$\boxed{\frac{\theta_{r_\star}}{\theta_{r_\star+1}}>\tau_{\mathrm R}\iff s_{r_\star}^2>\rho_\tau,\qquad \rho_\tau:=\frac{\eta(\tau_{\mathrm R}-1)}{1-\eta}.}$$

Because $0\le s_{r_\star}^2\le1$, a nontrivial strict threshold must satisfy
$$\boxed{1<\tau_{\mathrm R}<1/\eta,\qquad\text{equivalently}\qquad 0<\gamma_{\mathrm{gap}}<\log(1/\eta).}$$
The lower endpoint accepts arbitrarily weak alignment, while the upper endpoint asks for the unattainable strict inequality $s_{r_\star}^2>1$.

Substituting a high-probability product bound $K$ into Theorem 10 gives the sufficient condition
$$1+\frac{1-\eta}{\eta(1+\eta^2K^2)}>\tau_{\mathrm R}.$$
For $1<\tau_{\mathrm R}<1/\eta$, this is exactly equivalent to
$$\boxed{K<\kappa(\eta,\tau_{\mathrm R}),\qquad \kappa(\eta,\tau_{\mathrm R}):=\sqrt{\frac{1-\eta\tau_{\mathrm R}}{\eta^3(\tau_{\mathrm R}-1)}}.}$$
In log-gap notation,
$$\boxed{K<\sqrt{\frac{1-\eta e^{\gamma_{\mathrm{gap}}}}{\eta^3(e^{\gamma_{\mathrm{gap}}}-1)}}.}$$
Consequently, Theorem 15A proves knee detection with probability at least $1-\delta$ whenever
$$\boxed{K^{\mathrm G}_{r_\star,d,p,\delta}<\kappa(\eta,\tau_{\mathrm R}).}$$

An interpretable way to select the threshold is to first choose a required squared alignment $\rho\in(0,1)$ and then set
$$\boxed{\tau_{\mathrm R}=1+\frac{1-\eta}{\eta}\rho.}$$
This exposes the tradeoff: increasing $\tau_{\mathrm R}$ reduces false gap declarations, but demands stronger subspace capture and eventually makes the guarantee impossible as $\tau_{\mathrm R}\uparrow1/\eta$.

For example, a fixed code threshold $\gamma_{\mathrm{gap}}=1.5$ means $\tau_{\mathrm R}=e^{1.5}\approx4.482$, but its alignment meaning changes strongly with the step-floor parameter:

| $\eta$ | Required $\rho_\tau=\eta(e^{1.5}-1)/(1-\eta)$ | Interpretation |
|---:|---:|---|
| $0.01$ | $0.0352$ | permissive alignment requirement |
| $0.10$ | $0.3869$ | moderate alignment requirement |
| $0.20$ | $0.8704$ | very stringent alignment requirement |

Thus a universal fixed `tau_gap` does not represent a universal detection standard across different spectral floors. When $\eta$ can be estimated reliably, calibrating through a chosen $\rho$ is mathematically more stable than fixing $\gamma_{\mathrm{gap}}$ across all step spectra.

> **Implementation notation audit:** `Adaptive_Hutch_pplus_SequentialPilot` computes $g_j=\log(\theta_j/\theta_{j+1})$ and compares it with `tau_gap`. Therefore the code parameter is $\gamma_{\mathrm{gap}}$, not $\tau_{\mathrm R}$. For example, `tau_gap=1.5` means a ratio threshold $\tau_{\mathrm R}=e^{1.5}\approx4.48$, not $1.5$.
> The historical script `experiments/test_sequential_pilot.py` still passes the obsolete keyword `tau_plateau`; against the current function signature this raises `TypeError` rather than setting the threshold. Its reported threshold-dependent results require a corrected rerun.

---

## 7. Relation to the Withdrawn Logarithmic Formula

The earlier expression
$$p\ge\left\lceil\frac{\log(1/\delta)+\log(d-r_\star)+\log(\tau_{\mathrm R}r_\star)}{\log(1/\eta)}\right\rceil$$
does **not** follow from the current oversampling argument when $p=b-r_\star$ counts extra sketch columns. In Theorem 15A, oversampling changes the Gaussian conditioning factor $K^{\mathrm G}_{r_\star,d,p,\delta}$ algebraically through $p+1$, $\sqrt{r_\star+p}$, and $(2/\delta)^{1/(p+1)}$; it does not create a factor $\eta^p$.

A denominator $\log(1/\eta)$ arises naturally in a different algorithmic parameter: the number of power/subspace-iteration steps. If the range sketch were $Y=A^hS$ for an integer power depth $h\ge1$, the graph argument would contain
$$\tan\Theta_{\max}\le\eta^h\|S_2 S_1^\dagger\|_2,$$
and a product bound $K$ would give
$$s_{r_\star}^2\ge\frac{1}{1+\eta^{2h}K^2}.$$
For $1<\tau_{\mathrm R}<1/\eta$, a sufficient condition would then be
$$\boxed{h>\frac{\log K^2+\log\!\left(\frac{\eta(\tau_{\mathrm R}-1)}{1-\eta\tau_{\mathrm R}}\right)}{2\log(1/\eta)}.}$$
Thus the old expression has the qualitative shape of a **power-depth** bound combined with a crude estimate of $K^2$, but it is not a proved oversampling bound. Reusing the symbol $p$ for both roles would conflate two different mechanisms: extra columns improve Gaussian conditioning, whereas power iterations suppress the spectral tail geometrically.
