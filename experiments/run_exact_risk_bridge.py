"""Exact conditional-risk bridge on the frozen 24-spectrum manifest."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as la


ROOT_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from run_asymmetric_guard_heldout_benchmark import (  # noqa: E402
    build_eigenvalues,
    build_oriented_matrix,
    build_setup_manifest,
)


DEFAULT_DIMENSION = 500
DEFAULT_BUDGET = 160
DEFAULT_TRIALS = 200
DEFAULT_BOOTSTRAP_SAMPLES = 20_000
DEFAULT_Q_MAX = 76
BASIS_SEED_BASE = 60_000
BOOTSTRAP_SEED_BASE = 20_260_814
RISK_COLUMNS = ("oracle_risk", "gaussian_risk", "rademacher_risk")


def compute_nested_conditional_risks(matrix, eigenvalues, budget, sketch_matrix):
    """Compute exact risks for every nested prefix of one fixed sketch matrix S."""
    matrix = np.asarray(matrix, dtype=np.float64)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    sketch_matrix = np.asarray(sketch_matrix, dtype=np.float64)
    dimension = matrix.shape[0]
    if matrix.shape != (dimension, dimension):
        raise ValueError("matrix must be square.")
    if eigenvalues.shape != (dimension,):
        raise ValueError("eigenvalues must match the matrix dimension.")
    if sketch_matrix.ndim != 2 or sketch_matrix.shape[0] != dimension:
        raise ValueError("sketch_matrix must have shape (dimension, q_max).")
    q_max = sketch_matrix.shape[1]
    if q_max > min(dimension, (budget - 2) // 2):
        raise ValueError("The requested sketch endpoint is infeasible for the budget.")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("matrix and eigenvalues must be finite.")

    sampled_range = matrix @ sketch_matrix
    if q_max > 0:
        basis, triangular = la.qr(sampled_range, mode="economic")
        diagonal = np.abs(np.diag(triangular))
        cutoff = 1e-12 * float(la.norm(sampled_range, ord="fro"))
        if np.any(diagonal <= cutoff):
            raise RuntimeError(
                "A nested sketch prefix is numerically rank deficient; "
                "the frozen strictly-positive spectra are expected to have r_q=q."
            )
        matrix_basis = matrix @ basis
        compressed = basis.T @ matrix_basis
    else:
        basis = np.empty((dimension, 0), dtype=np.float64)
        matrix_basis = np.empty((dimension, 0), dtype=np.float64)
        compressed = np.empty((0, 0), dtype=np.float64)

    spectral_squares = eigenvalues**2
    tail_energy = np.cumsum(spectral_squares[::-1])[::-1]
    matrix_frobenius_sq = float(np.sum(spectral_squares))
    matrix_diagonal = np.diag(matrix).copy()
    diagonal_pa = np.zeros(dimension, dtype=np.float64)
    diagonal_pap = np.zeros(dimension, dtype=np.float64)
    aq_frobenius_sq = 0.0
    compressed_frobenius_sq = 0.0

    rows = []
    for q in range(q_max + 1):
        if q > 0:
            index = q - 1
            direction = basis[:, index]
            matrix_direction = matrix_basis[:, index]
            cross_coefficients = compressed[:index, index]
            projected_matrix_direction = (
                basis[:, :index] @ cross_coefficients
                if index > 0
                else np.zeros(dimension, dtype=np.float64)
            )
            diagonal_pa += direction * matrix_direction
            diagonal_pap += (
                2.0 * direction * projected_matrix_direction
                + compressed[index, index] * direction**2
            )
            aq_frobenius_sq += float(np.dot(matrix_direction, matrix_direction))
            compressed_frobenius_sq += float(
                2.0 * np.dot(cross_coefficients, cross_coefficients)
                + compressed[index, index] ** 2
            )

        residual_frobenius_sq = max(
            0.0,
            matrix_frobenius_sq
            - 2.0 * aq_frobenius_sq
            + compressed_frobenius_sq,
        )
        residual_diagonal = matrix_diagonal - 2.0 * diagonal_pa + diagonal_pap
        residual_diagonal_sq = float(np.dot(residual_diagonal, residual_diagonal))
        residual_off_diagonal_sq = max(
            0.0, residual_frobenius_sq - residual_diagonal_sq
        )
        realized_rank = q
        residual_count = budget - q - realized_rank
        oracle_tail = float(tail_energy[q]) if q < dimension else 0.0
        rows.append(
            {
                "q": q,
                "r_actual": realized_rank,
                "ell": residual_count,
                "oracle_risk": 2.0 * oracle_tail / residual_count,
                "gaussian_risk": 2.0 * residual_frobenius_sq / residual_count,
                "rademacher_risk": 2.0 * residual_off_diagonal_sq / residual_count,
                "residual_frobenius_sq": residual_frobenius_sq,
                "residual_diagonal_sq": residual_diagonal_sq,
            }
        )
    return pd.DataFrame(rows)


def summarize_curves(trials):
    """Average paired basis trials and attach descriptive normal intervals."""
    rows = []
    keys = ["setup_index", "spectrum_family", "setup_name", "matrix_seed", "q"]
    for key_values, selected in trials.groupby(keys, sort=True):
        row = dict(zip(keys, key_values))
        row["observations"] = len(selected)
        row["r_actual"] = int(selected["r_actual"].iloc[0])
        row["ell"] = int(selected["ell"].iloc[0])
        for risk_name in RISK_COLUMNS:
            values = selected[risk_name].to_numpy(dtype=np.float64)
            mean = float(np.mean(values))
            standard_error = (
                float(np.std(values, ddof=1) / np.sqrt(values.size))
                if values.size > 1
                else 0.0
            )
            row[f"mean_{risk_name}"] = mean
            row[f"se_{risk_name}"] = standard_error
            row[f"ci_low_{risk_name}"] = max(0.0, mean - 1.96 * standard_error)
            row[f"ci_high_{risk_name}"] = mean + 1.96 * standard_error
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_minimizers(trials, bootstrap_samples):
    """Report curve minimizers and paired percentile-bootstrap uncertainty."""
    rows = []
    for setup_index, selected in trials.groupby("setup_index", sort=True):
        selected = selected.sort_values(["basis_trial", "q"])
        setup_name = selected["setup_name"].iloc[0]
        q_values = np.sort(selected["q"].unique())
        trial_indices = np.sort(selected["basis_trial"].unique())
        matrices = {}
        for risk_name in RISK_COLUMNS:
            pivot = selected.pivot(index="basis_trial", columns="q", values=risk_name)
            pivot = pivot.reindex(index=trial_indices, columns=q_values)
            if pivot.isna().any().any():
                raise RuntimeError(f"Incomplete paired curve for {setup_name}, {risk_name}.")
            matrices[risk_name] = pivot.to_numpy(dtype=np.float64)

        rng = np.random.default_rng(BOOTSTRAP_SEED_BASE + int(setup_index))
        weights = rng.multinomial(
            len(trial_indices),
            np.full(len(trial_indices), 1.0 / len(trial_indices)),
            size=bootstrap_samples,
        )
        for risk_name in RISK_COLUMNS:
            risk_matrix = matrices[risk_name]
            mean_curve = risk_matrix.mean(axis=0)
            curve_minimizer = int(q_values[int(np.argmin(mean_curve))])
            trial_minimizers = q_values[np.argmin(risk_matrix, axis=1)]
            if risk_name == "oracle_risk":
                bootstrap_minimizers = np.full(bootstrap_samples, curve_minimizer)
            else:
                bootstrap_curves = weights @ risk_matrix / len(trial_indices)
                bootstrap_minimizers = q_values[np.argmin(bootstrap_curves, axis=1)]
            interval = np.quantile(bootstrap_minimizers, (0.025, 0.975))
            rows.append(
                {
                    "setup_index": int(setup_index),
                    "spectrum_family": selected["spectrum_family"].iloc[0],
                    "setup_name": setup_name,
                    "matrix_seed": int(selected["matrix_seed"].iloc[0]),
                    "risk_name": risk_name,
                    "curve_minimizer_q": curve_minimizer,
                    "minimum_mean_risk": float(np.min(mean_curve)),
                    "mean_trial_minimizer_q": float(np.mean(trial_minimizers)),
                    "median_trial_minimizer_q": float(np.median(trial_minimizers)),
                    "bootstrap_q_ci_low": float(interval[0]),
                    "bootstrap_q_ci_high": float(interval[1]),
                    "bootstrap_samples": int(bootstrap_samples),
                }
            )
    return pd.DataFrame(rows)


def validate_outputs(manifest, trials, curves, minimizers, requested_trials, q_max, budget):
    expected_trial_rows = 24 * requested_trials * (q_max + 1)
    if len(manifest) != 24 or manifest["setup_name"].nunique() != 24:
        raise RuntimeError("The exact bridge must use the frozen 24-case manifest.")
    if len(trials) != expected_trial_rows:
        raise RuntimeError(f"Expected {expected_trial_rows} trial rows, found {len(trials)}.")
    if trials.duplicated(["setup_name", "basis_trial", "q"]).any():
        raise RuntimeError("Duplicate exact-risk trial key detected.")
    if len(curves) != 24 * (q_max + 1):
        raise RuntimeError("Curve summary must contain every setup/allocation pair.")
    if len(minimizers) != 24 * len(RISK_COLUMNS):
        raise RuntimeError("Minimizer summary must contain three risks per setup.")
    numeric = trials[list(RISK_COLUMNS) + ["residual_frobenius_sq", "residual_diagonal_sq"]]
    if not np.all(np.isfinite(numeric.to_numpy())) or np.any(numeric.to_numpy() < -1e-12):
        raise RuntimeError("Exact risks must be finite and nonnegative.")
    if not np.all(trials["r_actual"] == trials["q"]):
        raise RuntimeError("Frozen positive spectra unexpectedly lost nested sketch rank.")
    if not np.all(trials["q"] + trials["r_actual"] + trials["ell"] == budget):
        raise RuntimeError("Exact-risk allocation accounting failed.")
    expected_seeds = BASIS_SEED_BASE + trials["basis_trial"].to_numpy()
    if not np.array_equal(trials["basis_seed"].to_numpy(), expected_seeds):
        raise RuntimeError("Basis seeds are not aligned by trial index.")


def run_exact_bridge(trials_per_setup, dimension, budget, bootstrap_samples, q_max, output_dir):
    """Run the frozen exact-risk bridge and write four validated CSV artifacts."""
    for name, value in (
        ("trials", trials_per_setup),
        ("dimension", dimension),
        ("budget", budget),
        ("bootstrap_samples", bootstrap_samples),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if q_max < 0 or q_max > min(dimension, (budget - 2) // 2):
        raise ValueError("q_max is outside the feasible allocation grid.")

    manifest = build_setup_manifest().copy()
    manifest["dimension"] = dimension
    manifest["budget"] = budget
    manifest["basis_trials"] = trials_per_setup
    manifest["q_min"] = 0
    manifest["q_max"] = q_max
    records = []
    for setup in manifest.to_dict("records"):
        eigenvalues = build_eigenvalues(setup, dimension)
        matrix = build_oriented_matrix(eigenvalues, setup["matrix_seed"])
        print(f"Running exact bridge {setup['setup_name']} ({setup['setup_index'] + 1}/24)", flush=True)
        for basis_trial in range(trials_per_setup):
            basis_seed = BASIS_SEED_BASE + basis_trial
            rng = np.random.default_rng(basis_seed)
            sketch_matrix = rng.choice([-1.0, 1.0], size=(dimension, q_max))
            risks = compute_nested_conditional_risks(
                matrix, eigenvalues, budget, sketch_matrix
            )
            risks.insert(0, "basis_seed", basis_seed)
            risks.insert(0, "basis_trial", basis_trial)
            risks.insert(0, "matrix_seed", int(setup["matrix_seed"]))
            risks.insert(0, "setup_name", setup["setup_name"])
            risks.insert(0, "spectrum_family", setup["spectrum_family"])
            risks.insert(0, "setup_index", int(setup["setup_index"]))
            records.append(risks)

    trials = pd.concat(records, ignore_index=True)
    curves = summarize_curves(trials)
    minimizers = summarize_minimizers(trials, bootstrap_samples)
    validate_outputs(
        manifest, trials, curves, minimizers, trials_per_setup, q_max, budget
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "manifest": output_dir / "risk_bridge_exact_manifest.csv",
        "trials": output_dir / "risk_bridge_exact_trials.csv",
        "curves": output_dir / "risk_bridge_exact_curves.csv",
        "minimizers": output_dir / "risk_bridge_exact_minimizers.csv",
    }
    manifest.to_csv(artifacts["manifest"], index=False)
    trials.to_csv(artifacts["trials"], index=False)
    curves.to_csv(artifacts["curves"], index=False)
    minimizers.to_csv(artifacts["minimizers"], index=False)
    for path in artifacts.values():
        print(f"Saved {path}")
    return manifest, trials, curves, minimizers


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--q-max", type=int, default=DEFAULT_Q_MAX)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "results")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_exact_bridge(
        trials_per_setup=arguments.trials,
        dimension=arguments.dimension,
        budget=arguments.budget,
        bootstrap_samples=arguments.bootstrap_samples,
        q_max=arguments.q_max,
        output_dir=arguments.output_dir,
    )
