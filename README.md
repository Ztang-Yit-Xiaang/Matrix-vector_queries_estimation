# Adaptive Hutch++ trace estimation research

This repository contains the Hutch++ research track from the UROP Summer 2026 project with Swati Padmanabhan. It studies adaptive matrix-vector query allocation, realized Gaussian/Rademacher residual risk, and the limits of finite-sample risk certification.

The main scientific conclusion is deliberately conservative: the exploratory adaptive gates are useful research prototypes, but this repository does not claim a universally safe or confidence-certified online allocator.

## Research status

- The fixed Hutch++ baselines, query accounting, rank-aware risk identities, and several certification limits are documented and tested.
- The zero-oversampling audit explains why a sharp pilot Ritz gap does not certify complete dominant-subspace capture.
- Phase 1A--1D and Phase 2A--2C records distinguish empirical evidence, proved identities, route-specific no-go results, and unresolved confidence questions.
- `Adaptive_Hutch_pplus_TwoStageGated` is exploratory. Its development benchmarks are not held-out validation and should not be read as a safety guarantee.
- The current teaching deck, manuscript packages, reports, figures, and provenance records are under `reports/`, `figures/`, and `results/`.

For the current project narrative and open questions, see [`CURRENT_STATE.md`](CURRENT_STATE.md). The broader workspace tracker is maintained alongside this repository.

## Repository layout

```text
src/          Estimators, matrix-vector oracles, and mathematical helper modules
experiments/  Reproducible experiment runners and report/figure builders
tests/        Unit, query-accounting, numerical, and artifact regression tests
docs/         Specifications, notation ledgers, and proof notes
reports/      Research reports, manuscript source/packages, audits, and decks
figures/      Validated figures, editable SVGs, source CSVs, and manifests
results/      Frozen experiment outputs, checksums, and recovery records
```

The primary implementation is [`src/trace_baseline.py`](src/trace_baseline.py). All estimator matrix-vector products should pass through `MatVecOracle` so query budgets remain auditable.

## Core estimators

| Estimator | Function | Role |
| --- | --- | --- |
| Hutchinson | `Hutchinson` | Uniform stochastic trace baseline |
| Standard Hutch++ | `Hutch_pplus` | Fixed-budget Hutch++ baseline |
| Non-adaptive Hutch++ | `NA_Hutch_pplus` | Single-batch non-adaptive comparison |
| Gaussian Hutch++ | `Gaussian_Hutch_pplus` | Gaussian-sketch comparison |
| Sequential pilot | `Adaptive_Hutch_pplus_SequentialPilot` | Guarded adaptive prototype |
| Marginal-risk pilot | `Adaptive_Hutch_pplus_MarginalRisk` | Marginal energy-drop prototype |
| Two-stage gated | `Adaptive_Hutch_pplus_TwoStageGated` | Exploratory reused-pilot gate |

The standard accounting identity is

```text
q + r_actual + ell = m
```

where `q` is construction work, `r_actual` is the accepted basis rank, and `ell` is the residual-probe count. Implementations must preserve this identity and must not allocate an explicit dense residual projector when a matrix-free application is available.

## Setup

Use Python 3.10--3.12 with NumPy, SciPy, Matplotlib, pandas, and pytest. A virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib pandas pytest
```

If the project environment already provides these packages, no installation is needed.

## Verification

Run the maintained tests from the repository root:

```bash
python -m pytest -q
```

Focused checks can be run individually, for example:

```bash
python -m pytest -q tests/test_twostage_gated_estimator.py
python -m pytest -q tests/test_coordinate_gate_failure.py
python -m pytest -q tests/test_structural_paired_difference_confidence_phase2c.py
```

Some benchmark and artifact tests depend on compiled numerical libraries or optional scientific packages. If a local environment terminates during collection, record that as an environment failure rather than interpreting it as a passing or failing scientific result.

## Reproducibility and provenance

Experiment runners are in `experiments/`; validated outputs and manifests are in `results/`. Research figures retain source CSVs or editable SVGs when applicable. Reports state whether a result is proved, empirically established on frozen paths, exploratory, or still open.

The current deliverables include:

- the source-backed UROP report and claims register;
- theorem--proof and LaTeX manuscript packages;
- coordinate-gate, orientation, certification, and paired-confidence audits;
- validated figure packages with source data;
- the September 2026 supervisor and teaching presentations under `reports/swati_meeting_20260922/output/`.

Generated build directories and temporary PowerPoint lock files are intentionally ignored. Final artifacts should be committed only when they are useful to reproduce, inspect, or cite the project.

The GitHub synchronization snapshot retains source code, summaries, manifests, reports, figures, and final presentations. A few multi-hundred-megabyte raw trial dumps remain local rather than being uploaded to ordinary Git storage; their producing scripts and compact summaries are retained so the experiments can be regenerated.

## Important interpretation rules

1. A numerical rank or visible Ritz knee is not a proof of complete signal capture.
2. Mean-risk improvement does not imply pathwise improvement; rare catastrophic paths can dominate the mean.
3. Conditional residual-risk estimates and paid-policy safety use different denominators and must not be conflated.
4. Bootstrap intervals over frozen orientations and paths are empirical summaries, not universal confidence theorems.
5. The exploratory gates remain separate from the frozen baseline and should not be presented as certified replacements.

## License

This research code is released under the MIT License. See the repository history and the report provenance records for authorship, source attribution, and validation details.
