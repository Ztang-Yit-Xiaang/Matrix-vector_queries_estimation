# Matrix-Vector Queries Trace Estimation

This project implements **three variants of the Hutch++ algorithm** to estimate the trace of a matrix using only **matrix-vector multiplication queries**. The algorithms are applied to a real-world dataset (**Wiki-Vote network**) to estimate the number of **triangles in a graph**.

The project demonstrates how **randomized numerical linear algebra** can be used to compute expensive quantities efficiently when the matrix is too large to form explicitly.

---

# 1. Problem Statement

Many problems in numerical linear algebra require computing the **trace of a matrix**
$\text{tr}(A)$.

However, in large-scale applications the matrix \(A\) is often **not explicitly available**.  
Instead, we can only compute **matrix-vector products**
$Av$

through a function or oracle.

The challenge is therefore:

> Estimate $\text{tr}(A)$ using as few matrix-vector multiplications as possible.

This problem appears in many applications such as:

- log-determinant estimation
- spectral density estimation
- graph analytics
- triangle counting in networks

---

# 2. Triangle Counting Example

In this project we apply trace estimation to the **Wiki-Vote network dataset**.

The dataset represents a voting network where:

- **nodes** represent Wikipedia users
- **edges** represent votes between users

We analyze the **triangle structure** of the network.

Triangles represent **mutual support groups**, which are an important indicator of community structure in social networks.

---

### Graph formulation

Let $B$

be the adjacency matrix of the graph.

A classical identity from graph theory states:

$$
\text{Number of triangles} = \frac{1}{6}\text{tr}(B^3)
$$

where:

- $B^3$ counts the number of **3-step paths**
- the trace extracts **closed walks**

Each triangle contributes **6 closed walks**, hence the division by 6.

Therefore the problem reduces to:

> Estimate $\text{tr}(B^3)$

without explicitly computing $B^3$.

---

# 3. Trace Estimation Algorithms

## 3.1 Hutchinson's Estimator

The classical estimator uses random vectors $g$:

$$
\text{tr}(A) \approx \frac{1}{m}\sum_{i=1}^{m} g_i^T A g_i
$$

This estimator requires

$$
O(1/\epsilon^2)
$$

matrix-vector queries to achieve a \((1\pm \epsilon)\) approximation.

---

## 3.2 Hutch++

Hutch++ improves Hutchinson’s estimator by combining:

- a **low-rank approximation**
- a **stochastic trace estimator**

The estimator is

$$
\text{Hutch++}(A) =
\text{tr}(Q^T A Q) +
\frac{3}{m}
\text{tr}(G^T(I - QQ^T)A(I - QQ^T)G)
$$

This reduces the query complexity to

$O(1/\epsilon)$

which is a **quadratic improvement** over Hutchinson's estimator.

---

# 4. Implemented Algorithms

This project implements three variants:

### 1️⃣ Hutch++

Adaptive trace estimator with variance reduction.

### 2️⃣ NA-Hutch++

Non-adaptive variant where all matrix-vector queries are generated beforehand.

### 3️⃣ Gaussian-Hutch++

Variant using Gaussian random vectors that allows tighter variance analysis.

---

# 5. Matrix-Vector Oracle

Instead of forming \(B^3\), we define a **matrix-vector oracle**

\[
A = B^3
\]

so that
$$Av = B(B(Bv))$$


This avoids constructing the dense matrix \(B^3\) and allows efficient computation using sparse matrix operations.

---

# 6. Pipeline

The full workflow of the project is:

```
Wiki-Vote dataset
↓
Construct adjacency matrix $B$
↓
Define matrix-vector oracle $A = B^3$
↓
Run Hutch++ estimators
↓
Estimate $\text{tr}(A)$
↓
Triangle count = $\text{tr}(A) / 6$
```

---

# 7. Running the Code

Example usage:

```python
vote_matrix = load_wiki_Vote_as_graph()

d = vote_matrix.shape[0]

vote_A = count_triangles(vote_matrix)

m = 5000

hutch_pp_estimate = Hutch_pplus(vote_A, m, d)
na_hutch_pp_estimate = NA_Hutch_pplus(vote_A, m, d)
gaussian_hutch_pp_estimate = Gaussian_Hutch_pplus(vote_A, m, d)

print("Hutch++ estimate:", hutch_pp_estimate / 6)
```

# 8. Expected Output

The Wiki-Vote dataset contains roughly

≈ 608,000 triangles

The randomized estimators should approximate this value with small relative error.

# 10. References

Raphael A. Meyer, Cameron Musco, Christopher Musco, and David P. Woodruff.
**Hutch++: Optimal Stochastic Trace Estimation.**
NeurIPS / arXiv:2010.09649
