# Design: Zero-Oversampling Mechanism Audit

**Date:** 2026-08-16  
**Status:** approved by the user's detailed task specification  
**Scope:** isolated diagnostics only; no Adaptive Hutch++ estimator or frozen bridge modification

## Research question

The frozen rank-deficient bridge found that, at the positive small-tail control $\eta=10^{-6}$, every relevant path has $r_q=q$, while the ideal rank-aware risk is minimized at $q=r_\star$ and the empirical mean Gaussian and Rademacher risks are minimized at $q=r_\star+1$. This audit will test whether the shift is explained by the square, zero-oversampling dominant sketch block at $q=r_\star$, rare poor conditioning, degraded dominant-subspace capture, and the resulting upper tail of realized residual risk.

The audit will not assume that $+1$ is universal. It will classify mathematical identities, frozen empirical results, new diagnostic evidence, mechanism claims, and open statements separately.

## Repository reconstruction

The audit will read the frozen implementation and outputs to recover the exact meanings of $q$, $r_q$, $r_\star$, and the four risk minimizers. It will validate the production configuration, seeds, Rademacher sketch distribution, numerical-rank rule, nested coupling, construction count $q+r_q$, and residual count $\ell=m-q-r_q$ before interpreting any risk.

## Diagnostic reconstruction

A new script will deterministically reconstruct the frozen $d=500$, $\eta=10^{-6}$ bases from the recorded orientation and basis seeds. It will duplicate the bridge's incremental, reorthogonalized rank-aware QR path only inside the diagnostic and assert equality with the frozen CSV for ranks, Gaussian/Rademacher energies, and risks. Any mismatch is a blocking bug and will be reported before interpretation.

For $q\in\{r_\star-2,\ldots,r_\star+2\}$ and all 200 frozen trials, it will record:

- singular values, condition number, pseudoinverse norm, stable rank, and row rank of $S_1=U_1^TS_q$;
- singular values and conditioning of the full pre-QR sketch $Y_q=AS_q$;
- numerical rank returned by the frozen QR rule;
- $\|(I-Q_qQ_q^T)U_1\|_2$, its Frobenius counterpart, the largest principal angle, and $\sigma_{\min}(U_1^TQ_q)$ when defined;
- exact stable $E_G(Q_q)$ and $E_R(Q_q)$ and their risks under every frozen budget;
- the ideal tail $T(q)$ and ideal-to-realized gaps;
- the graph-amplification diagnostic $\|\eta S_2S_1^\dagger\|_2$ when $S_1$ has full row rank.

The same sketch prefixes will be used at $q$ and $q+1$, preserving the original nested coupling.

## Statistical analysis

The primary analysis will report means, medians, 90th/95th/99th percentiles, maxima, bootstrap descriptive intervals, Spearman correlations, quantile stratification, and worst 1%/5%/10% shares of the total risk. It will calculate the exact pathwise criterion

$$
\frac{E_X(Q_{q+1})}{E_X(Q_q)}
<
\frac{m-2q-2}{m-2q},
\qquad X\in\{G,R\},
$$

and distinguish pathwise improvement frequency from the minimizer of the empirical mean curve.

## Sensitivity analysis

A separate, explicitly developmental sensitivity grid will vary tail level, dimension, dominant rank, total budget, and seed batch while retaining the same structured step matrix and Rademacher sketch family. It will use fewer trials than the frozen production run and will never overwrite production artifacts. Its purpose is to classify the exact $+1$ as systematic, instance-specific, or one member of a broader modest-oversampling pattern.

The default sensitivity grid is:

$$
d\in\{250,500\},\quad
r_\star\in\{5,15,30\},\quad
\eta\in\{10^{-8},10^{-6},10^{-4},10^{-2}\},
$$

$$
m\in\{80,160,240\},
$$

with three 50-trial seed batches. Only feasible ranks and allocations are retained. Mean curves are scanned over the full budget-feasible range, not merely $r_\star\pm2$.

## Output artifacts

The isolated script will create:

- `results/q_rank_vs_realized_risk_mechanism_trials.csv`;
- `results/q_rank_vs_realized_risk_mechanism_summary.csv`;
- `results/q_rank_vs_realized_risk_mechanism_sensitivity.csv`;
- `results/q_rank_vs_realized_risk_mechanism_tail.csv`;
- six or more PNG figures under `results/figures/q_rank_vs_realized_risk_mechanism/`;
- `reports/q_rank_vs_realized_risk_mechanism.md`.

The six required figures will use distribution/relationship forms rather than mean-only lines: knee-risk distributions, rank versus subspace quality, conditioning versus risk, subspace error versus risk, upper-tail survival/comparison, and ideal tail versus realized energy. Static Matplotlib output is selected because the deliverables are publication-ready repository files.

## Validation

Focused tests will verify:

1. exact reconstruction of frozen ranks and energies on representative seeds;
2. $q+r_q+\ell=m$ and, when $r_q=q$, $\ell=m-2q$;
3. principal-angle and projection identities;
4. the local risk inequality in both directions;
5. nested sketch prefixes;
6. finite/nonnegative outputs and expected schemas;
7. sensitivity outputs never overwrite frozen bridge artifacts.

After targeted tests, the full existing test suite will run. Figures will be opened and visually inspected. Historical result checksums will be compared before and after the diagnostic.

## Interpretation boundary

The strongest acceptable conclusion is mixed theoretical/empirical: the local risk inequality and deterministic graph-subspace relation may be proved under explicit rank assumptions; the claim that rare square-boundary conditioning failures drive the frozen mean-risk shift must be established empirically; and no fixed oversampling guarantee or universal $+1$ theorem will be claimed.
