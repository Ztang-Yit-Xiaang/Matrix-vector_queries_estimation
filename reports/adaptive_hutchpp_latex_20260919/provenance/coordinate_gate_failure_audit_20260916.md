# Coordinate-aligned gated Hutch++ failure: mechanism audit

Date: 2026-09-16. Status: **VERIFIED for the reconstructed failure path**.

## 1. Executive result

The recovered coordinate-aligned rank-five step failure is not just an unlucky
final trace estimate. One Rademacher pilot loses an entire dominant eigendirection
because two signal rows are exactly opposite. The full sampled range still has
eight independent columns: tail directions fill out its numerical rank. The pilot
therefore displays four large Ritz values, and the gate confidently mistakes the
visible rank for the matrix's dominant rank.

The missing direction remains in the residual. Its conditional Rademacher risk is
approximately **0.0227274**, versus about **1.31e-7** on the other coordinate-aligned
gated paths. This single path contributes **99.9942% of the observed squared
error** and **99.9948% of the total exact conditional risk** in the ten-path
coordinate case. The failure exists before final-probe randomness is drawn.

No estimator, threshold, original source, or historical result was changed.
This is diagnosis, not an improved allocator or new held-out benchmark.

## 2. Reproduction and definitions

Replayed the first spectrum of `experiments/run_haar_orientation_audit.py`:

\[
A=U\operatorname{diag}(1,1,1,1,1,0.001,\ldots,0.001)U^\top,
\quad d=100,\quad m=60,\quad r_\star=5.
\]

The original seed is 2026. One shared RNG generates matrices and all estimator
draws. The order is Hutchinson, Standard Hutch++, Gaussian-sketch Hutch++, then
the two-stage gated method, for each trial. An individual row's `seed=2026` is
**not** sufficient to restart that row: earlier draws must be replayed or the
saved per-method RNG state restored.

The audit reproduces all four outputs on 10 coordinate-aligned paths and 300
paths across 30 Haar orientations. There are 620 detailed basis records for
Standard and gated methods. Maximum absolute discrepancy in any reproduced
squared error is **1.12e-16** (rounded upward). These are the old realizations,
not new independent evidence. Standard and gated methods use different sketches
and final probes, not a common-random-number coupling.

Parameters remain `b_0=8`, `tau_gap=1.5`, `p_oversample=2`, no contrast filter,
and pivoted rank-aware QR with `rtol=1e-12`, `atol=0`. The reference scale is
the Frobenius norm of the sampled matrix.

Here \(q\) counts attempted sketch columns, \(r\) is accepted basis rank,
and \(\ell=m-q-r\). Every inspected gated path has \((q,r,\ell)=(8,8,44)\);
every Standard path has \((20,20,20)\). Every replayed estimator costs exactly
60 oracle queries. There is **no numerical rank loss in the full basis**.

The diagnostic replays cost 74,400 queries (310 paths × four methods × 60).
Independent blockwise exact-energy checks on the 20 coordinate bases cost a
further 2,000 queries. Total audit cost: **76,400 queries**, recorded separately
from the frozen per-estimator budget. Synthetic structured energy calculations
require no further oracle products; no dense residual matrix is constructed.

## 3. The exact missing direction

On zero-based trial **7** (the eighth coordinate path), let
\(U_\star=[e_1,\ldots,e_5]\) and \(S_1=U_\star^\top S\in\{-1,1\}^{5\times8}\).
Its third and fourth rows are

\[
(1,1,-1,-1,1,1,-1,-1),\qquad
(-1,-1,1,1,-1,-1,1,1).
\]

Exact rational elimination verifies \(\operatorname{rank}(S_1)=4\), not five.
An exact null vector of \(S_1^\top\) is \(v=(0,0,1,1,0)^\top\).
Thus the unit dominant direction

\[
w=\frac{e_3+e_4}{\sqrt2}
\]

satisfies \(S^\top w=0\) and \(Aw=w\).

**PROVED in exact arithmetic.** Writing \(Y=AS\), symmetry gives
\(Y^\top w=S^\top Aw=S^\top w=0\). Hence \(Q^\top w=0\) for any
orthonormal basis of \(\operatorname{range}(Y)\). With \(R=I-QQ^\top\),

\[
Rw=w,\qquad RARw=w,\qquad \|RAR\|_F^2\ge1.
\]

This last bound concerns Gaussian residual energy; it is **not**, by itself,
a lower bound on Rademacher off-diagonal energy.

The floating-point replay gives \(\|Q^\top w\|_2=8.07\times10^{-14}\),
consistent with the exact witness. The smallest signal singular value is
\(7.72\times10^{-17}\), but the smallest singular value of the full \(AS\)
is approximately 0.00848 and QR accepts all eight columns.

| Diagnostic | Other nine coordinate gated paths | Bad path |
|---|---:|---:|
| Exact signal rank | 5 | 4 |
| Full accepted rank | 8 | 8 |
| Largest subspace sine error | median 0.008424 | 1 |
| Largest principal angle | below 0.70 degrees | 90 degrees |
| Detected knee | 5 | 4 |
| Maximum adjacent log-Ritz gap | approximately 6.9077 | 6.907721 |
| Gap contrast | approximately 69,077 | 69,077.213 |

The conditional signal-rank test uses exact rational arithmetic only for
coordinate-aligned integer \(S_1\). Rotated signal blocks are checked numerically;
we do not call their floating-point ranks exact algebraic ranks.

## 4. Why a sharp Ritz gap did not protect the gate

For this step family,

\[
Q^\top AQ=\eta I_8+(1-\eta)(U_\star^\top Q)^\top(U_\star^\top Q).
\]

The signal block has rank four on the bad path, so the Ritz matrix has only four
eigenvalues above the tail level. The observed values are approximately

\[
0.999995,\ 0.999993,\ 0.999988,\ 0.999966,
\ 0.001,\ 0.001,\ 0.001,\ 0.001.
\]

The gate consequently selects
\(q=\max\{8,4+2\}=8\). It also selects eight on the ordinary paths because
\(\max\{8,5+2\}=8\). **The catastrophic case does not involve a different final
allocation within this ten-path group; it involves a different quality of the
same-sized basis.**

There is a stronger, local identifiability observation. Define

\[
A'=A-(1-\eta)ww^\top.
\]

This alternative PSD matrix has four unit eigenvalues and all remaining
eigenvalues equal to \(\eta\). Yet

\[
A'S=AS,\qquad A'Q=AQ,
\]

because \(w^\top S=w^\top Q=0\). Thus the entire pilot transcript is identical
for \(A\) and \(A'\). A deterministic function of that transcript cannot always
distinguish their dominant ranks. This is a statement about these fixed
observations, **not** an impossibility theorem for randomized guarantees or
future fresh-query diagnostics.

The later contrast threshold of 5 would not screen out this particular recorded
pilot: its contrast exceeds 69,000. This is a direct feature comparison, not a
rerun or validation of that later allocator variant.

## 5. Residual geometry versus final-probe noise

Condition on the complete pre-residual action. Fresh independent Rademacher
probes imply unbiasedness and exact risk

\[
\mathcal R_R(Q)=\frac{2E_R(Q)}{\ell},\qquad
E_R(Q)=\sum_{i\ne j}(RAR)_{ij}^2.
\]

For Gaussian final probes on the **same fixed basis**, the diagnostic risk is
\(\mathcal R_G(Q)=2E_G(Q)/\ell\), where \(E_G=\|RAR\|_F^2\).
No Gaussian final probes were drawn to estimate this analytic quantity.

For the structured step matrix set \(Z=(I-QQ^\top)U_\star\). Then

\[
H=RAR=\eta R+(1-\eta)ZZ^\top,
\]

\[
E_G=\eta^2(d-r)+2\eta(1-\eta)\|Z\|_F^2
 +(1-\eta)^2\|Z^\top Z\|_F^2.
\]

We sum squares of small column blocks of \(H\), omitting diagonal entries for
\(E_R\), and independently check all coordinate cases by blockwise oracle
applications. This avoids subtractive cancellation and does not form dense
\(R\) or \(H\).

On the bad path:

\[
E_G\approx1.000091,\quad E_R\approx0.500004,\quad
\mathcal R_G\approx0.045459,\quad
\mathcal R_R=0.02272744518.
\]

The leading missing projector \(ww^\top\) alone gives
\(g^\top ww^\top g=1+g_3g_4\), which is 0 or 2 with equal probabilities.
Its variance is one, giving risk \(1/44\approx0.0227273\). The full calculated
risk agrees closely. This is an explanation of the dominant term, not an
assumption that all residual contributions are independent.

The actual squared error is 0.03303752, only **1.45364 times its own conditional
risk**. Final-probe noise changes the observed value, but cannot explain away
the approximately 173,000-fold increase over ordinary gated conditional risk.
For comparison, the independently drawn Standard basis at the same trial index
has about 18,106 times smaller conditional risk. This is a diagnostic contrast,
not a coupled one-column intervention.

Although \(A\) is diagonal, **\(RAR\) generally is not**. Classical Rademacher
Hutchinson is exact on this diagonal input; randomized low-rank projection can
introduce off-diagonal residual entries. The previous broad phrase “diagonal
residual implies zero error on the coordinate case” applies to the ideal
eigenbasis residual, not the actual randomized basis here.

## 6. Distributional evidence and limits

| Orientation / method | Paths | Observed MSE | Mean exact Rademacher risk | Median exact risk |
|---|---:|---:|---:|---:|
| Coordinate / gated | 10 | 3.303942e-3 | 2.272863e-3 | 1.309110e-7 |
| Coordinate / Standard | 10 | 7.209998e-7 | 1.254788e-6 | 1.255168e-6 |
| Haar / gated | 300 | 2.839433e-7 | 3.337032e-7 | 3.307227e-7 |
| Haar / Standard | 300 | 1.672813e-6 | 1.584249e-6 | 1.584367e-6 |

Coordinate gated/Standard ratios are approximately **4582.45** for observed MSE
and **1811.35** for mean conditional risk. The analogous Haar ratios are **0.16974**
and **0.21064**. Removing final-probe noise changes the numerical ratios, but not
the central contrast. No signal-rank failure was found in the 300 rotated gated
paths; this does not establish zero failure probability.

Only ten paths were available at the coordinate orientation. One dominates both
the observed error and conditional risk. We therefore do **not** estimate a
universal catastrophe rate, attach a reliable population tail interval, or
claim that a finite-grid orientation contrast is rotational invariance.
The 300 Haar paths share 30 orientations, so they are not 300 independently
sampled orientations either.

This result is distinct from the earlier frozen zero-oversampling bridge. Here
\(q-r_\star=3\), not zero, and the coordinate-Rademacher signal block is exactly
singular. Positive oversampling improves opportunities for capture but does not
deterministically guarantee full signal rank for a discrete sketch.

## 7. Comparator-label correction

The unchanged `Gaussian_Hutch_pplus` function draws a **Gaussian range sketch**
and **Rademacher final residual probes**, with \(q=15\) at \(m=60\), rather than
Standard's \(q=20\). Its name does not identify an all-Gaussian, rotationally
invariant estimator. Comparisons mix a range-distribution change and an allocation
change; they do not isolate Gaussian versus Rademacher residual noise. The
original Haar runner's “rotation-invariant control” comments should not be used
as a scientific claim. The audit leaves that source untouched and records the
correction here and in current research memory.

## 8. Evidence labels and next decision

**PROVED:** an exactly missed dominant direction remains an eigenvector of
\(RAR\); the pilot-transcript indistinguishability construction; conditional
Rademacher/Gaussian variance identities; and the distinction between full sample
rank and dominant-signal rank.

**EMPIRICALLY VERIFIED:** the exact integer nullspace witness, recorded feature
values, full-rank QR, risks, error concentration, and reconstruction of the
original 310 paths.

**NOT ESTABLISHED:** a universal failure probability, a safe contrast threshold,
the number of extra columns that would repair every failure, or a uniformly
better allocator. No threshold is tuned from this failure.

The next deliverable should be the corrected UROP narrative: **a visible Ritz
knee certifies separation among captured directions, not completeness of
capture**. Any continuation should separately preregister a fresh-evidence
capture check and charge its cost. That remains future work, not implemented
here. It is unnecessary to invent another allocator to explain this result.

## 9. Reproducibility and validation

Implementation: `experiments/audit_coordinate_gate_failure.py`.
Tests: `tests/test_coordinate_gate_failure.py`.
Artifacts: `results/coordinate_gate_failure_audit_20260916/` contains `paths.csv`,
`summary.csv`, exact rational witnesses, per-method RNG states, and a manifest.
The manifest verifies **156 pre-existing result/source files unchanged** and
records output hashes, versions, and query costs. A reduced coordinate-only
smoke run used `/private/tmp`; output directories must be new.

Eight focused tests cover exact nullspaces, query recording, structured and
oracle energy agreement, exact small-matrix Rademacher variance, replay, and
overwrite refusal. **All 225 maintained tests pass**, including historical
artifact immutability. Compilation and `git diff --check` also pass.
Repository-root test discovery also encounters a pre-existing import failure
in `experiments/test_adaptive_hutch.py`; the maintained suite is `tests/`.
