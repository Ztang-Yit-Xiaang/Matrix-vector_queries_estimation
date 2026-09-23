"""Diagnose why realized-risk optima move beyond the ideal step rank.

This is an offline diagnostic.  It reconstructs the frozen nested Rademacher
range sketches from their saved seeds and never modifies the estimator or the
frozen bridge artifacts.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.stats import spearmanr


ROOT_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
SRC_DIR = ROOT_DIR / "src"
for directory in (EXPERIMENTS_DIR, SRC_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from run_rank_deficient_risk_bridge import (  # noqa: E402
    DEFAULT_BASIS_SEED_BASE,
    DEFAULT_BUDGETS,
    DEFAULT_DIMENSION,
    DEFAULT_MIN_RESIDUAL_PROBES,
    DEFAULT_ORIENTATION_SEED_BASE,
    DEFAULT_QR_ATOL,
    DEFAULT_QR_RTOL,
    DEFAULT_STEP_RANKS,
    _energy_state,
    apply_structured_step,
    compute_nested_rank_path,
    make_signal_basis,
    q_max_for_budget,
    spectral_tail_energy,
)
from trace_baseline import _rank_aware_qr  # noqa: E402


DEFAULT_ETA = 1e-6
DEFAULT_TRIALS = 200
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results"
DEFAULT_FROZEN_TRIALS = DEFAULT_OUTPUT_DIR / "risk_bridge_rank_deficient_trials.csv"
DEFAULT_SENSITIVITY_DIMENSIONS = (250, 500)
DEFAULT_SENSITIVITY_ETAS = (1e-8, 1e-6, 1e-4, 1e-2)
DEFAULT_SENSITIVITY_BATCHES = 3
DEFAULT_SENSITIVITY_TRIALS = 50
DEFAULT_SENSITIVITY_ORIENTATION_SEED = 91_000
DEFAULT_SENSITIVITY_BASIS_SEED = 93_000

OUTPUT_FILENAMES = {
    "trials": "q_rank_vs_realized_risk_mechanism_trials.csv",
    "summary": "q_rank_vs_realized_risk_mechanism_summary.csv",
    "tail": "q_rank_vs_realized_risk_mechanism_tail.csv",
    "sensitivity": "q_rank_vs_realized_risk_mechanism_sensitivity.csv",
}

INK = "#243447"
BLUE = "#3568A8"
BLUE_LIGHT = "#9DB9DA"
GOLD = "#D39B2A"
GOLD_LIGHT = "#E9CF91"
GREY = "#8C96A3"
GRID = "#DCE2E8"


def _safe_ratio(numerator, denominator):
    if denominator == 0.0:
        return np.nan
    return float(numerator / denominator)


def _svd_diagnostics(matrix, full_row_target=None):
    matrix = np.asarray(matrix, dtype=np.float64)
    singular_values = la.svdvals(matrix)
    if singular_values.size == 0:
        return {
            "sigma_max": 0.0,
            "sigma_min": np.nan,
            "condition": np.nan,
            "pinv_norm": np.nan,
            "stable_rank": 0.0,
            "numerical_rank": 0,
            "full_row_rank": False if full_row_target else True,
        }
    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * sigma_max
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    positive = singular_values[singular_values > tolerance]
    pinv_norm = float(1.0 / positive[-1]) if positive.size else np.nan
    condition = (
        float(sigma_max / sigma_min) if sigma_min > tolerance else np.nan
    )
    stable_rank = float(np.sum(singular_values**2) / sigma_max**2)
    return {
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "condition": condition,
        "pinv_norm": pinv_norm,
        "stable_rank": stable_rank,
        "numerical_rank": numerical_rank,
        "full_row_rank": (
            numerical_rank == int(full_row_target)
            if full_row_target is not None
            else numerical_rank == min(matrix.shape)
        ),
    }


def _subspace_diagnostics(signal_basis, basis):
    dimension, step_rank = signal_basis.shape
    rank = basis.shape[1]
    if rank:
        residual_signal = signal_basis - basis @ (basis.T @ signal_basis)
    else:
        residual_signal = signal_basis.copy()
    singular_values = la.svdvals(residual_signal)
    projection_op = float(singular_values[0]) if singular_values.size else 0.0
    projection_fro = float(la.norm(residual_signal, ord="fro"))
    overlap = signal_basis.T @ basis
    overlap_singular = la.svdvals(overlap)
    if rank >= step_rank and overlap_singular.size >= step_rank:
        sigma_min_overlap = float(overlap_singular[-1])
    else:
        sigma_min_overlap = 0.0
    largest_angle = float(math.asin(np.clip(projection_op, 0.0, 1.0)))
    return {
        "subspace_projection_op": projection_op,
        "subspace_projection_fro": projection_fro,
        "largest_principal_angle_rad": largest_angle,
        "largest_principal_angle_deg": float(np.degrees(largest_angle)),
        "sigma_min_u1tq": sigma_min_overlap,
        "residual_signal": residual_signal,
    }


def _diagnostic_row(signal_basis, eta, sketch_prefix, basis, frozen_row):
    dimension, step_rank = signal_basis.shape
    q = sketch_prefix.shape[1]
    rank = basis.shape[1]
    s1 = signal_basis.T @ sketch_prefix
    s1_diag = _svd_diagnostics(s1, full_row_target=step_rank)
    sample = apply_structured_step(sketch_prefix, signal_basis, eta)
    y_diag = _svd_diagnostics(sample)
    subspace = _subspace_diagnostics(signal_basis, basis)
    projected_signal = subspace.pop("residual_signal")
    basis_row_norm_sq = (
        np.sum(basis**2, axis=1) if rank else np.zeros(dimension, dtype=float)
    )
    stable_energy = _energy_state(
        signal_basis, projected_signal, basis_row_norm_sq, eta, rank
    )
    for column in ("gaussian_energy", "rademacher_energy"):
        if not np.isclose(
            stable_energy[column],
            float(frozen_row[column]),
            rtol=2e-9,
            atol=1e-28,
        ):
            raise RuntimeError(f"Diagnostic reconstruction disagrees for {column}.")
    graph_op = np.nan
    graph_fro = np.nan
    graph_projection_bound = np.nan
    graph_identity_error = np.nan
    if q >= step_rank and s1_diag["full_row_rank"]:
        s1_pinv = la.pinv(s1)
        tail_sample = sketch_prefix - signal_basis @ s1
        graph_ambient = eta * tail_sample @ s1_pinv
        graph_op = float(la.norm(graph_ambient, ord=2))
        graph_fro = float(la.norm(graph_ambient, ord="fro"))
        graph_projection_bound = float(graph_op / np.sqrt(1.0 + graph_op**2))
        if q == step_rank:
            graph_identity_error = float(
                abs(graph_projection_bound - subspace["subspace_projection_op"])
            )
    ideal_tail = spectral_tail_energy(dimension, step_rank, eta, q)
    return {
        "q": q,
        "r_actual": rank,
        "rank_gain_accepted": bool(frozen_row["rank_gain_accepted"]),
        "reference_scale": float(frozen_row["reference_scale"]),
        "projected_sample_norm": float(frozen_row["projected_sample_norm"]),
        "rank_acceptance_cutoff": float(frozen_row["rank_acceptance_cutoff"]),
        "gaussian_energy": float(stable_energy["gaussian_energy"]),
        "rademacher_energy": float(stable_energy["rademacher_energy"]),
        "ideal_tail_q": ideal_tail,
        "gaussian_to_ideal_tail": _safe_ratio(
            stable_energy["gaussian_energy"], ideal_tail
        ),
        "rademacher_to_ideal_tail": _safe_ratio(
            stable_energy["rademacher_energy"], ideal_tail
        ),
        "s1_sigma_max": s1_diag["sigma_max"],
        "s1_sigma_min": s1_diag["sigma_min"],
        "s1_condition": s1_diag["condition"],
        "s1_pinv_norm": s1_diag["pinv_norm"],
        "s1_stable_rank": s1_diag["stable_rank"],
        "s1_numerical_rank": s1_diag["numerical_rank"],
        "s1_full_row_rank": s1_diag["full_row_rank"],
        "y_sigma_min": y_diag["sigma_min"],
        "y_condition": y_diag["condition"],
        "y_numerical_rank": y_diag["numerical_rank"],
        "graph_factor_op": graph_op,
        "graph_factor_fro": graph_fro,
        "graph_projection_bound": graph_projection_bound,
        "graph_identity_error": graph_identity_error,
        **subspace,
    }


def reconstruct_mechanism_path(
    signal_basis,
    eta,
    sketch_matrix,
    q_values,
    qr_rtol=DEFAULT_QR_RTOL,
    qr_atol=DEFAULT_QR_ATOL,
):
    """Reconstruct selected frozen prefixes while retaining the accepted bases."""
    q_values = tuple(sorted(set(int(value) for value in q_values)))
    if not q_values or q_values[0] < 1:
        raise ValueError("q_values must contain positive prefix sizes.")
    q_max = q_values[-1]
    sketch_matrix = np.asarray(sketch_matrix, dtype=np.float64)
    if sketch_matrix.shape[1] < q_max:
        raise ValueError("Sketch does not contain the requested prefix.")
    frozen_path = compute_nested_rank_path(
        signal_basis,
        eta,
        sketch_matrix[:, :q_max],
        qr_rtol=qr_rtol,
        qr_atol=qr_atol,
    ).set_index("q")
    dimension = signal_basis.shape[0]
    accepted_basis = np.empty((dimension, q_max), dtype=np.float64)
    accepted_rank = 0
    reference_scale_sq = 0.0
    rows = []
    for column_index in range(q_max):
        sample = apply_structured_step(
            sketch_matrix[:, column_index], signal_basis, eta
        )
        reference_scale_sq += float(np.dot(sample, sample))
        if accepted_rank:
            current = accepted_basis[:, :accepted_rank]
            residual = sample - current @ (current.T @ sample)
            residual = residual - current @ (current.T @ residual)
        else:
            residual = sample.copy()
        candidate, candidate_rank = _rank_aware_qr(
            residual[:, None],
            reference_scale=float(np.sqrt(reference_scale_sq)),
            rtol=qr_rtol,
            atol=qr_atol,
        )
        if candidate_rank == 1:
            accepted_basis[:, accepted_rank] = candidate[:, 0]
            accepted_rank += 1
        q = column_index + 1
        frozen_row = frozen_path.loc[q]
        if accepted_rank != int(frozen_row["r_actual"]):
            raise RuntimeError("Diagnostic QR rank disagrees with frozen path.")
        if q in q_values:
            rows.append(
                _diagnostic_row(
                    signal_basis,
                    eta,
                    sketch_matrix[:, :q],
                    accepted_basis[:, :accepted_rank].copy(),
                    frozen_row,
                )
            )
    return pd.DataFrame(rows)


def add_budget_and_local_fields(frame, budgets):
    expanded = []
    for budget in budgets:
        selected = frame.loc[frame["q"] * 2 < budget].copy()
        selected["budget"] = int(budget)
        selected["ell"] = budget - selected["q"] - selected["r_actual"]
        selected["constructed_basis_queries"] = selected["q"] + selected["r_actual"]
        selected["accounted_total_queries"] = budget
        selected["rank_risk"] = [
            2.0
            * spectral_tail_energy(
                int(row.dimension),
                int(row.step_rank),
                float(row.eta),
                int(row.r_actual),
            )
            / int(row.ell)
            for row in selected.itertuples(index=False)
        ]
        selected["gaussian_risk"] = (
            2.0 * selected["gaussian_energy"] / selected["ell"]
        )
        selected["rademacher_risk"] = (
            2.0 * selected["rademacher_energy"] / selected["ell"]
        )
        expanded.append(selected)
    result = pd.concat(expanded, ignore_index=True)
    keys = ["step_rank", "basis_trial", "budget"]
    result = result.sort_values(keys + ["q"]).reset_index(drop=True)
    grouped = result.groupby(keys, sort=False)
    result["next_q"] = grouped["q"].shift(-1)
    for energy in ("gaussian", "rademacher"):
        next_energy = grouped[f"{energy}_energy"].shift(-1)
        next_risk = grouped[f"{energy}_risk"].shift(-1)
        result[f"{energy}_energy_ratio_next"] = next_energy / result[
            f"{energy}_energy"
        ]
        result[f"{energy}_risk_difference_next"] = (
            next_risk - result[f"{energy}_risk"]
        )
        result[f"{energy}_local_improves"] = (
            result[f"{energy}_risk_difference_next"] < 0.0
        ).astype(float)
    next_ell = grouped["ell"].shift(-1)
    result["local_energy_ratio_threshold"] = next_ell / result["ell"]
    invalid_next = result["next_q"] != result["q"] + 1
    local_columns = [
        column
        for column in result.columns
        if column.endswith("_next")
        or column.endswith("_improves")
        or column == "local_energy_ratio_threshold"
    ]
    result.loc[invalid_next, local_columns] = np.nan
    return result


def reconstruct_frozen_trials(
    trials=DEFAULT_TRIALS,
    dimension=DEFAULT_DIMENSION,
    step_ranks=DEFAULT_STEP_RANKS,
    eta=DEFAULT_ETA,
    budgets=DEFAULT_BUDGETS,
):
    q_max_all = max(
        q_max_for_budget(dimension, budget, DEFAULT_MIN_RESIDUAL_PROBES)
        for budget in budgets
    )
    records = []
    for rank_index, step_rank in enumerate(step_ranks):
        signal_basis = make_signal_basis(
            dimension, step_rank, DEFAULT_ORIENTATION_SEED_BASE + rank_index
        )
        q_values = range(max(1, step_rank - 2), step_rank + 3)
        for basis_trial in range(trials):
            basis_seed = DEFAULT_BASIS_SEED_BASE + basis_trial
            rng = np.random.default_rng(basis_seed)
            sketch = rng.choice([-1.0, 1.0], size=(dimension, q_max_all))
            local = reconstruct_mechanism_path(
                signal_basis, eta, sketch, q_values
            )
            local["dimension"] = dimension
            local["step_rank"] = step_rank
            local["eta"] = eta
            local["orientation_seed"] = DEFAULT_ORIENTATION_SEED_BASE + rank_index
            local["basis_trial"] = basis_trial
            local["basis_seed"] = basis_seed
            local["q_offset"] = local["q"] - step_rank
            records.append(local)
    return add_budget_and_local_fields(pd.concat(records, ignore_index=True), budgets)


def validate_against_frozen(reconstructed, frozen_path=DEFAULT_FROZEN_TRIALS):
    frozen = pd.read_csv(frozen_path)
    frozen = frozen.loc[
        np.isclose(frozen["eta"], DEFAULT_ETA)
        & frozen["q"].between(frozen["step_rank"] - 2, frozen["step_rank"] + 2)
    ]
    keys = ["step_rank", "basis_trial", "budget", "q"]
    columns = ["r_actual", "ell", "gaussian_energy", "rademacher_energy", "gaussian_risk", "rademacher_risk"]
    merged = reconstructed.merge(
        frozen[keys + columns], on=keys, suffixes=("", "_frozen"), validate="one_to_one"
    )
    if len(merged) != len(reconstructed):
        raise RuntimeError("Frozen reconstruction key support is incomplete.")
    for column in columns:
        left = merged[column].to_numpy(dtype=float)
        right = merged[f"{column}_frozen"].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=2e-9, atol=1e-28):
            raise RuntimeError(f"Frozen reconstruction failed for {column}.")


def _bootstrap_interval(values, statistic, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        value = float(statistic(values))
        return value, value
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = values[rng.integers(0, values.size, size=values.size)]
        estimates[index] = statistic(draw)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def summarize_primary(trials, bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES):
    rows = []
    group_keys = ["dimension", "step_rank", "eta", "budget", "q", "q_offset"]
    metrics = [
        "gaussian_energy",
        "rademacher_energy",
        "gaussian_risk",
        "rademacher_risk",
        "s1_sigma_min",
        "s1_pinv_norm",
        "graph_factor_op",
        "subspace_projection_op",
        "subspace_projection_fro",
        "largest_principal_angle_deg",
        "gaussian_to_ideal_tail",
        "rademacher_to_ideal_tail",
    ]
    for group_index, (key, group) in enumerate(trials.groupby(group_keys, sort=True)):
        row = dict(zip(group_keys, key))
        row["observations"] = len(group)
        row["full_rank_rate"] = float(np.mean(group["r_actual"] == group["q"]))
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=np.float64)
            if values.size == 0:
                for statistic in (
                    "mean",
                    "median",
                    "p90",
                    "p95",
                    "p99",
                    "max",
                    "mean_ci_low",
                    "mean_ci_high",
                ):
                    row[f"{statistic}_{metric}"] = np.nan
                continue
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"median_{metric}"] = float(np.median(values))
            row[f"p90_{metric}"] = float(np.quantile(values, 0.90))
            row[f"p95_{metric}"] = float(np.quantile(values, 0.95))
            row[f"p99_{metric}"] = float(np.quantile(values, 0.99))
            row[f"max_{metric}"] = float(np.max(values))
            low, high = _bootstrap_interval(
                values,
                np.mean,
                bootstrap_samples,
                120_000 + group_index * len(metrics) + metrics.index(metric),
            )
            row[f"mean_ci_low_{metric}"] = low
            row[f"mean_ci_high_{metric}"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _tail_share(values, fraction):
    values = np.asarray(values, dtype=np.float64)
    count = max(1, int(np.ceil(fraction * values.size)))
    order = np.sort(values)[::-1]
    return float(np.sum(order[:count]) / np.sum(order)), count


def tail_and_relationship_summary(trials):
    rows = []
    for (step_rank, budget), group in trials.groupby(["step_rank", "budget"], sort=True):
        knee = group.loc[group["q"] == step_rank].copy()
        next_rows = group.loc[group["q"] == step_rank + 1].copy()
        paired = knee.merge(
            next_rows,
            on=["step_rank", "budget", "basis_trial"],
            suffixes=("_k", "_kp1"),
            validate="one_to_one",
        )
        for risk in ("gaussian", "rademacher"):
            values = paired[f"{risk}_risk_k"].to_numpy(dtype=float)
            next_values = paired[f"{risk}_risk_kp1"].to_numpy(dtype=float)
            improves = next_values < values
            row = {
                "step_rank": step_rank,
                "budget": budget,
                "risk_name": risk,
                "observations": len(paired),
                "mean_risk_k": float(np.mean(values)),
                "median_risk_k": float(np.median(values)),
                "mean_risk_kp1": float(np.mean(next_values)),
                "median_risk_kp1": float(np.median(next_values)),
                "improvement_count": int(np.count_nonzero(improves)),
                "improvement_rate": float(np.mean(improves)),
                "improving_path_risk_share": float(np.sum(values[improves]) / np.sum(values)),
                "mean_energy_ratio": float(
                    paired[f"{risk}_energy_kp1"].mean()
                    / paired[f"{risk}_energy_k"].mean()
                ),
                "median_path_energy_ratio": float(
                    np.median(
                        paired[f"{risk}_energy_kp1"]
                        / paired[f"{risk}_energy_k"]
                    )
                ),
                "required_energy_ratio": float(
                    (budget - 2 * step_rank - 2) / (budget - 2 * step_rank)
                ),
            }
            for fraction in (0.01, 0.05, 0.10):
                share, count = _tail_share(values, fraction)
                label = int(round(100 * fraction))
                row[f"top_{label}pct_risk_share"] = share
                row[f"top_{label}pct_count"] = count
            median_next = float(np.median(next_values))
            row["count_above_10x_next_median"] = int(
                np.count_nonzero(values > 10.0 * median_next)
            )
            row["count_above_100x_next_median"] = int(
                np.count_nonzero(values > 100.0 * median_next)
            )
            for diagnostic in (
                "s1_pinv_norm",
                "graph_factor_op",
                "subspace_projection_op",
                "subspace_projection_fro",
            ):
                result = spearmanr(
                    paired[f"{diagnostic}_k"], values, nan_policy="omit"
                )
                row[f"spearman_{diagnostic}_vs_risk"] = float(result.statistic)
                row[f"spearman_p_{diagnostic}_vs_risk"] = float(result.pvalue)
            rows.append(row)
    return pd.DataFrame(rows)


def _sensitivity_energy_curves(signal_basis, eta, sketch_matrix):
    dimension, step_rank = signal_basis.shape
    sample = apply_structured_step(sketch_matrix, signal_basis, eta)
    basis, upper = la.qr(sample, mode="economic")
    reference = np.sqrt(np.cumsum(np.sum(sample**2, axis=0)))
    diagonal = np.abs(np.diag(upper))
    accepted = diagonal > DEFAULT_QR_RTOL * reference
    if not np.all(accepted):
        frozen = compute_nested_rank_path(signal_basis, eta, sketch_matrix)
        full_rank = frozen.loc[frozen["q"] > 0, "r_actual"] == frozen.loc[
            frozen["q"] > 0, "q"
        ]
        if not np.all(full_rank):
            raise RuntimeError("Sensitivity path encountered numerical rank loss.")
        raise RuntimeError(
            "Sensitivity QR disagrees with the frozen incremental QR acceptance path."
        )
    projected_signal = signal_basis.copy()
    row_norm_sq = np.zeros(dimension, dtype=np.float64)
    gaussian = np.empty(sketch_matrix.shape[1] + 1, dtype=np.float64)
    rademacher = np.empty_like(gaussian)
    initial = _energy_state(
        signal_basis, projected_signal, row_norm_sq, eta, rank=0
    )
    gaussian[0] = initial["gaussian_energy"]
    rademacher[0] = initial["rademacher_energy"]
    for index in range(sketch_matrix.shape[1]):
        direction = basis[:, index]
        coefficients = direction @ signal_basis
        projected_signal -= np.outer(direction, coefficients)
        row_norm_sq += direction**2
        state = _energy_state(
            signal_basis,
            projected_signal,
            row_norm_sq,
            eta,
            rank=index + 1,
        )
        gaussian[index + 1] = state["gaussian_energy"]
        rademacher[index + 1] = state["rademacher_energy"]
    return gaussian, rademacher


def run_sensitivity(
    dimensions=DEFAULT_SENSITIVITY_DIMENSIONS,
    step_ranks=DEFAULT_STEP_RANKS,
    etas=DEFAULT_SENSITIVITY_ETAS,
    budgets=DEFAULT_BUDGETS,
    batches=DEFAULT_SENSITIVITY_BATCHES,
    trials=DEFAULT_SENSITIVITY_TRIALS,
):
    rows = []
    for dimension_index, dimension in enumerate(dimensions):
        q_max_all = max(
            q_max_for_budget(dimension, budget, DEFAULT_MIN_RESIDUAL_PROBES)
            for budget in budgets
        )
        for rank_index, step_rank in enumerate(step_ranks):
            if step_rank >= dimension or 2 * step_rank + 2 >= max(budgets):
                continue
            for batch in range(batches):
                orientation_seed = (
                    DEFAULT_SENSITIVITY_ORIENTATION_SEED
                    + 100 * dimension_index
                    + 10 * rank_index
                    + batch
                )
                signal_basis = make_signal_basis(
                    dimension, step_rank, orientation_seed
                )
                energy_by_eta = {
                    eta: {"gaussian": [], "rademacher": []} for eta in etas
                }
                for trial in range(trials):
                    seed = (
                        DEFAULT_SENSITIVITY_BASIS_SEED
                        + 10_000 * batch
                        + trial
                    )
                    rng = np.random.default_rng(seed)
                    sketch = rng.choice(
                        [-1.0, 1.0], size=(dimension, q_max_all)
                    )
                    for eta in etas:
                        gaussian, rademacher = _sensitivity_energy_curves(
                            signal_basis, eta, sketch
                        )
                        energy_by_eta[eta]["gaussian"].append(gaussian)
                        energy_by_eta[eta]["rademacher"].append(rademacher)
                for eta in etas:
                    energy_arrays = {
                        key: np.asarray(values, dtype=np.float64)
                        for key, values in energy_by_eta[eta].items()
                    }
                    for budget in budgets:
                        q_max = q_max_for_budget(
                            dimension, budget, DEFAULT_MIN_RESIDUAL_PROBES
                        )
                        q_values = np.arange(q_max + 1)
                        denominator = budget - 2 * q_values
                        rank_curve = np.array(
                            [
                                2.0
                                * spectral_tail_energy(
                                    dimension, step_rank, eta, int(q)
                                )
                                / int(denominator[q])
                                for q in q_values
                            ]
                        )
                        row = {
                            "dimension": dimension,
                            "step_rank": step_rank,
                            "eta": eta,
                            "budget": budget,
                            "batch": batch,
                            "orientation_seed": orientation_seed,
                            "trials": trials,
                            "q_rank_star": int(q_values[np.argmin(rank_curve)]),
                        }
                        for risk in ("gaussian", "rademacher"):
                            energies = energy_arrays[risk][:, : q_max + 1]
                            risks = 2.0 * energies / denominator[None, :]
                            mean_curve = np.mean(risks, axis=0)
                            q_star = int(q_values[np.argmin(mean_curve)])
                            row[f"q_{risk}_star"] = q_star
                            row[f"{risk}_shift_from_rank"] = (
                                q_star - row["q_rank_star"]
                            )
                            if step_rank + 1 <= q_max:
                                row[f"{risk}_mean_risk_ratio_k_to_kp1"] = float(
                                    np.mean(risks[:, step_rank])
                                    / np.mean(risks[:, step_rank + 1])
                                )
                                row[f"{risk}_path_improvement_rate"] = float(
                                    np.mean(
                                        risks[:, step_rank + 1]
                                        < risks[:, step_rank]
                                    )
                                )
                        rows.append(row)
    return pd.DataFrame(rows)


def _style_axis(axis, title, subtitle=None):
    axis.set_title(
        title,
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
        y=1.085,
        pad=0,
    )
    if subtitle:
        axis.text(
            0.0,
            1.025,
            subtitle,
            transform=axis.transAxes,
            fontsize=8,
            color=GREY,
            va="bottom",
        )
    axis.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def make_figures(trials, figure_dir):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    primary = trials.loc[trials["budget"] == 160].copy()
    paths = []

    # Figure 1: risk distributions around the knee.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    for axis, (step_rank, group) in zip(axes, primary.groupby("step_rank", sort=True)):
        data = [
            group.loc[group["q_offset"] == offset, "gaussian_risk"]
            for offset in range(-2, 3)
        ]
        axis.boxplot(
            data,
            positions=range(-2, 3),
            widths=0.55,
            showfliers=True,
            flierprops={"markersize": 2, "markerfacecolor": GOLD, "markeredgecolor": GOLD},
            boxprops={"color": BLUE},
            medianprops={"color": INK, "linewidth": 1.4},
            whiskerprops={"color": BLUE},
            capprops={"color": BLUE},
        )
        means = [float(np.mean(values)) for values in data]
        axis.scatter(range(-2, 3), means, marker="D", s=24, color=GOLD, label="mean")
        axis.set_yscale("log")
        axis.set_xlabel(r"$q-r_\star$")
        _style_axis(axis, rf"$r_\star={step_rank}$", "Gaussian conditional risk; 200 nested paths, m=160")
    axes[0].set_ylabel("Conditional risk (log scale)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = figure_dir / "figure_1_risk_around_knee.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # Figure 2: full rank versus subspace quality.
    selected = primary.loc[primary["q_offset"].isin([0, 1])]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    offsets = np.array([0, 1])
    rank_colors = (BLUE_LIGHT, BLUE, INK)
    rank_markers = ("o", "s", "^")
    for rank_index, (step_rank, group) in enumerate(selected.groupby("step_rank", sort=True)):
        positions = offsets + (rank_index - 1) * 0.18
        rank_rates = [
            np.mean(group.loc[group["q_offset"] == offset, "r_actual"] == group.loc[group["q_offset"] == offset, "q"])
            for offset in offsets
        ]
        axes[0].plot(
            positions,
            rank_rates,
            marker=rank_markers[rank_index],
            color=rank_colors[rank_index],
            label=rf"$r_\star={step_rank}$",
        )
        medians = [
            np.median(group.loc[group["q_offset"] == offset, "subspace_projection_op"])
            for offset in offsets
        ]
        p99 = [
            np.quantile(group.loc[group["q_offset"] == offset, "subspace_projection_op"], 0.99)
            for offset in offsets
        ]
        axes[1].plot(
            positions,
            medians,
            marker=rank_markers[rank_index],
            color=rank_colors[rank_index],
            label=rf"$r_\star={step_rank}$ median",
        )
        axes[1].scatter(positions, p99, marker="D", color=GOLD, s=24)
    axes[0].set_xticks(offsets, [r"$r_\star$", r"$r_\star+1$"])
    axes[0].set_ylim(0.95, 1.005)
    axes[0].set_ylabel("Fraction with $r_q=q$")
    axes[0].legend(frameon=False, fontsize=8)
    _style_axis(axes[0], "Numerical rank", "Every frozen path remains full rank")
    axes[1].set_xticks(offsets, [r"$r_\star$", r"$r_\star+1$"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$\|(I-Q_qQ_q^T)U_1\|_2$")
    _style_axis(axes[1], "Dominant-subspace error", "Circles: median; diamonds: 99th percentile")
    axes[1].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    path = figure_dir / "figure_2_rank_vs_subspace_quality.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # Figure 3: graph amplification and risk at the square boundary.
    square = primary.loc[primary["q_offset"] == 0]
    fig, axis = plt.subplots(figsize=(6.8, 5.0))
    for rank_index, (step_rank, group) in enumerate(square.groupby("step_rank", sort=True)):
        axis.scatter(
            group["graph_factor_op"],
            group["gaussian_risk"],
            s=18,
            alpha=0.62,
            color=rank_colors[rank_index],
            marker=rank_markers[rank_index],
            label=rf"$r_\star={step_rank}$",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$\|\eta(I-U_1U_1^T)S_qS_1^{\dagger}\|_2$")
    axis.set_ylabel("Gaussian conditional risk")
    axis.legend(frameon=False)
    _style_axis(axis, "Graph amplification versus realized risk", "Square dominant block at q=r★; m=160")
    fig.tight_layout()
    path = figure_dir / "figure_3_conditioning_vs_risk.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # Figure 4: principal-angle error and risk.
    fig, axis = plt.subplots(figsize=(6.8, 5.0))
    for rank_index, (step_rank, group) in enumerate(square.groupby("step_rank", sort=True)):
        axis.scatter(
            group["subspace_projection_op"],
            group["gaussian_risk"],
            s=18,
            alpha=0.62,
            color=rank_colors[rank_index],
            marker=rank_markers[rank_index],
            label=rf"$r_\star={step_rank}$",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$\|(I-Q_qQ_q^T)U_1\|_2=\sin\theta_{\max}$")
    axis.set_ylabel("Gaussian conditional risk")
    axis.legend(frameon=False)
    _style_axis(axis, "Subspace error versus realized risk", "Exact Gaussian mechanism at q=r★; m=160")
    fig.tight_layout()
    path = figure_dir / "figure_4_subspace_error_vs_risk.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # Figure 5: upper-tail survival comparison.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, (step_rank, group) in zip(axes, selected.groupby("step_rank", sort=True)):
        normalizer = np.median(group.loc[group["q_offset"] == 1, "gaussian_risk"])
        for offset, color, label in ((0, GOLD, r"$q=r_\star$"), (1, BLUE, r"$q=r_\star+1$")):
            values = np.sort(group.loc[group["q_offset"] == offset, "gaussian_risk"] / normalizer)
            survival = (values.size - np.arange(values.size)) / values.size
            axis.step(values, survival, where="post", color=color, label=label)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"Risk / median risk at $r_\star+1$")
        _style_axis(axis, rf"$r_\star={step_rank}$", "Empirical survival function; Gaussian risk")
    axes[0].set_ylabel("Fraction at or above")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = figure_dir / "figure_5_catastrophic_tail.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # Figure 6: ideal tail versus realized residual energy.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for axis, (step_rank, group) in zip(axes, primary.groupby("step_rank", sort=True)):
        positions = np.arange(-2, 3)
        gaussian_median = [
            np.median(group.loc[group["q_offset"] == offset, "gaussian_to_ideal_tail"])
            for offset in positions
        ]
        gaussian_p99 = [
            np.quantile(group.loc[group["q_offset"] == offset, "gaussian_to_ideal_tail"], 0.99)
            for offset in positions
        ]
        rademacher_p99 = [
            np.quantile(group.loc[group["q_offset"] == offset, "rademacher_to_ideal_tail"], 0.99)
            for offset in positions
        ]
        axis.plot(positions, gaussian_median, marker="o", color=BLUE, label="Gaussian median")
        axis.plot(positions, gaussian_p99, marker="D", color=GOLD, label="Gaussian p99")
        axis.plot(positions, rademacher_p99, marker="s", color=INK, linestyle="--", label="Rademacher p99")
        axis.axhline(1.0, color=GREY, linewidth=1.0, linestyle=":")
        axis.set_yscale("log")
        axis.set_xlabel(r"$q-r_\star$")
        _style_axis(axis, rf"$r_\star={step_rank}$", "Realized energy divided by ideal spectral tail")
    axes[0].set_ylabel("Energy / $T(q)$")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = figure_dir / "figure_6_ideal_tail_vs_realized_energy.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def validate_primary_outputs(trials, summary, tail):
    if trials.empty or summary.empty or tail.empty:
        raise RuntimeError("Mechanism outputs must be nonempty.")
    if trials.duplicated(["step_rank", "basis_trial", "budget", "q"]).any():
        raise RuntimeError("Duplicate mechanism trial key.")
    if np.any(trials["q"] + trials["r_actual"] + trials["ell"] != trials["budget"]):
        raise RuntimeError("Mechanism query accounting failed.")
    relevant = trials["eta"] == DEFAULT_ETA
    if not np.all(trials.loc[relevant, "r_actual"] == trials.loc[relevant, "q"]):
        raise RuntimeError("Frozen positive-tail mechanism path lost numerical rank.")
    if not np.all(trials.loc[relevant, "ell"] == trials.loc[relevant, "budget"] - 2 * trials.loc[relevant, "q"]):
        raise RuntimeError("Full-rank residual denominator is incorrect.")
    numeric = trials.select_dtypes(include=[np.number])
    required_finite = [
        column for column in numeric.columns if not column.endswith("_next")
    ]
    if np.any(~np.isfinite(numeric[required_finite].dropna().to_numpy())):
        raise RuntimeError("Mechanism output contains nonfinite required diagnostics.")
    square = trials["q"] == trials["step_rank"]
    if np.any(trials.loc[square, "graph_identity_error"] > 2e-8):
        raise RuntimeError("Square graph/principal-angle identity failed.")


def _atomic_write(frames, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    temporary = []
    for name, frame in frames.items():
        final_path = output_dir / OUTPUT_FILENAMES[name]
        temporary_path = output_dir / f".{OUTPUT_FILENAMES[name]}.tmp"
        frame.to_csv(temporary_path, index=False)
        temporary.append((temporary_path, final_path))
        paths[name] = final_path
    for temporary_path, final_path in temporary:
        temporary_path.replace(final_path)
    return paths


def run_mechanism_audit(
    output_dir=DEFAULT_OUTPUT_DIR,
    primary_trials=DEFAULT_TRIALS,
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
    sensitivity=True,
    sensitivity_dimensions=DEFAULT_SENSITIVITY_DIMENSIONS,
    sensitivity_etas=DEFAULT_SENSITIVITY_ETAS,
    sensitivity_batches=DEFAULT_SENSITIVITY_BATCHES,
    sensitivity_trials=DEFAULT_SENSITIVITY_TRIALS,
):
    output_dir = Path(output_dir)
    trials = reconstruct_frozen_trials(trials=primary_trials)
    if primary_trials == DEFAULT_TRIALS:
        validate_against_frozen(trials)
    summary = summarize_primary(trials, bootstrap_samples=bootstrap_samples)
    tail = tail_and_relationship_summary(trials)
    if sensitivity:
        sensitivity_frame = run_sensitivity(
            dimensions=tuple(sensitivity_dimensions),
            etas=tuple(sensitivity_etas),
            batches=int(sensitivity_batches),
            trials=int(sensitivity_trials),
        )
    else:
        sensitivity_frame = pd.DataFrame(
            columns=[
                "dimension",
                "step_rank",
                "eta",
                "budget",
                "batch",
                "q_rank_star",
                "q_gaussian_star",
                "q_rademacher_star",
            ]
        )
    validate_primary_outputs(trials, summary, tail)
    paths = _atomic_write(
        {
            "trials": trials,
            "summary": summary,
            "tail": tail,
            "sensitivity": sensitivity_frame,
        },
        output_dir,
    )
    figures = make_figures(
        trials, output_dir / "figures" / "q_rank_vs_realized_risk_mechanism"
    )
    return trials, summary, tail, sensitivity_frame, paths, figures


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary-trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument(
        "--sensitivity-dimensions",
        type=int,
        nargs="+",
        default=DEFAULT_SENSITIVITY_DIMENSIONS,
    )
    parser.add_argument(
        "--sensitivity-etas",
        type=float,
        nargs="+",
        default=DEFAULT_SENSITIVITY_ETAS,
    )
    parser.add_argument(
        "--sensitivity-batches", type=int, default=DEFAULT_SENSITIVITY_BATCHES
    )
    parser.add_argument(
        "--sensitivity-trials", type=int, default=DEFAULT_SENSITIVITY_TRIALS
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = vars(parse_args())
    arguments["sensitivity"] = not arguments.pop("skip_sensitivity")
    _, _, _, _, output_paths, figure_paths = run_mechanism_audit(**arguments)
    for output_path in output_paths.values():
        print(f"Saved {output_path}")
    for figure_path in figure_paths:
        print(f"Saved {figure_path}")
