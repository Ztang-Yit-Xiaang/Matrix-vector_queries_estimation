"""Frozen rank-deficient, multi-budget exact conditional-risk bridge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as la


ROOT_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
SRC_DIR = ROOT_DIR / "src"
for directory in (EXPERIMENTS_DIR, SRC_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from risk_bridge_regret import (  # noqa: E402
    minimum_plateau,
    paired_bootstrap_minimizer_frequencies,
    regret_columns,
)
from trace_baseline import MatVecOracle, _rank_aware_qr  # noqa: E402


DEFAULT_BUDGETS = (80, 160, 240)
DEFAULT_PRIMARY_BUDGET = 160
DEFAULT_TRIALS = 200
DEFAULT_DIMENSION = 500
DEFAULT_STEP_RANKS = (5, 15, 30)
DEFAULT_TAIL_LEVELS = (0.0, 1e-14, 1e-10, 1e-6)
DEFAULT_MIN_RESIDUAL_PROBES = 8
DEFAULT_BOOTSTRAP_SAMPLES = 20_000
DEFAULT_QR_RTOL = 1e-12
DEFAULT_QR_ATOL = 0.0
DEFAULT_ZERO_BACKWARD_RTOL = 1e-12
DEFAULT_ORIENTATION_SEED_BASE = 52_000
DEFAULT_BASIS_SEED_BASE = 70_000
DEFAULT_BOOTSTRAP_SEED_BASE = 20_260_815
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results"

RISK_COLUMNS = (
    "full_risk",
    "rank_risk",
    "gaussian_risk",
    "rademacher_risk",
)
RISK_LABELS = {
    "full_risk": "full",
    "rank_risk": "rank",
    "gaussian_risk": "gaussian",
    "rademacher_risk": "rademacher",
}
OUTPUT_FILENAMES = {
    "manifest": "risk_bridge_rank_deficient_manifest.csv",
    "trials": "risk_bridge_rank_deficient_trials.csv",
    "curves": "risk_bridge_rank_deficient_curves.csv",
    "minimizers": "risk_bridge_rank_deficient_minimizers.csv",
    "frequencies": "risk_bridge_rank_deficient_minimizer_frequencies.csv",
}


def _validate_integer(name, value, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _validate_integer_tuple(name, values, minimum=1):
    values = tuple(_validate_integer(name, value, minimum) for value in values)
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must be nonempty and contain unique values.")
    return values


def _validate_float(name, value, minimum=0.0, strict_upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number.")
    value = float(value)
    if not np.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}.")
    if strict_upper is not None and value >= strict_upper:
        raise ValueError(f"{name} must be less than {strict_upper}.")
    return value


def q_max_for_budget(dimension, budget, min_residual_probes):
    dimension = _validate_integer("dimension", dimension, 1)
    budget = _validate_integer("budget", budget, 1)
    min_residual_probes = _validate_integer(
        "min_residual_probes", min_residual_probes, 1
    )
    q_max = min(dimension, (budget - min_residual_probes) // 2)
    if q_max < 0:
        raise ValueError("Budget leaves no feasible allocation with the residual floor.")
    return q_max


def validate_configuration(
    budgets=DEFAULT_BUDGETS,
    primary_budget=DEFAULT_PRIMARY_BUDGET,
    trials=DEFAULT_TRIALS,
    dimension=DEFAULT_DIMENSION,
    step_ranks=DEFAULT_STEP_RANKS,
    tail_levels=DEFAULT_TAIL_LEVELS,
    min_residual_probes=DEFAULT_MIN_RESIDUAL_PROBES,
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
    qr_rtol=DEFAULT_QR_RTOL,
    qr_atol=DEFAULT_QR_ATOL,
    zero_backward_rtol=DEFAULT_ZERO_BACKWARD_RTOL,
    orientation_seed_base=DEFAULT_ORIENTATION_SEED_BASE,
    basis_seed_base=DEFAULT_BASIS_SEED_BASE,
    bootstrap_seed_base=DEFAULT_BOOTSTRAP_SEED_BASE,
):
    budgets = _validate_integer_tuple("budgets", budgets)
    primary_budget = _validate_integer("primary_budget", primary_budget, 1)
    if primary_budget not in budgets:
        raise ValueError("primary_budget must be present in budgets.")
    trials = _validate_integer("trials", trials, 1)
    dimension = _validate_integer("dimension", dimension, 1)
    step_ranks = _validate_integer_tuple("step_ranks", step_ranks)
    if max(step_ranks) > dimension:
        raise ValueError("Every step rank must not exceed dimension.")
    tail_levels = tuple(
        _validate_float("tail_levels", eta, minimum=0.0, strict_upper=1.0)
        for eta in tail_levels
    )
    if not tail_levels or len(tail_levels) != len(set(tail_levels)):
        raise ValueError("tail_levels must be nonempty and unique.")
    min_residual_probes = _validate_integer(
        "min_residual_probes", min_residual_probes, 1
    )
    bootstrap_samples = _validate_integer("bootstrap_samples", bootstrap_samples, 1)
    qr_rtol = _validate_float("qr_rtol", qr_rtol)
    qr_atol = _validate_float("qr_atol", qr_atol)
    zero_backward_rtol = _validate_float(
        "zero_backward_rtol", zero_backward_rtol
    )
    orientation_seed_base = _validate_integer(
        "orientation_seed_base", orientation_seed_base, 0
    )
    basis_seed_base = _validate_integer("basis_seed_base", basis_seed_base, 0)
    bootstrap_seed_base = _validate_integer(
        "bootstrap_seed_base", bootstrap_seed_base, 0
    )
    q_maxima = {
        budget: q_max_for_budget(dimension, budget, min_residual_probes)
        for budget in budgets
    }
    return {
        "budgets": budgets,
        "primary_budget": primary_budget,
        "trials": trials,
        "dimension": dimension,
        "step_ranks": step_ranks,
        "tail_levels": tail_levels,
        "min_residual_probes": min_residual_probes,
        "bootstrap_samples": bootstrap_samples,
        "qr_rtol": qr_rtol,
        "qr_atol": qr_atol,
        "zero_backward_rtol": zero_backward_rtol,
        "orientation_seed_base": orientation_seed_base,
        "basis_seed_base": basis_seed_base,
        "bootstrap_seed_base": bootstrap_seed_base,
        "q_maxima": q_maxima,
    }


def _is_default_configuration(config):
    expected = validate_configuration()
    return all(config[key] == expected[key] for key in expected)


def build_spectrum_manifest(config):
    """Build the 12 step spectra and their three-budget manifest."""
    spectra = []
    spectrum_index = 0
    for rank_index, step_rank in enumerate(config["step_ranks"]):
        orientation_seed = config["orientation_seed_base"] + rank_index
        for eta in config["tail_levels"]:
            eta_label = "0" if eta == 0.0 else f"{eta:.0e}"
            spectra.append(
                {
                    "spectrum_index": spectrum_index,
                    "rank_index": rank_index,
                    "orientation_group_id": f"rank_{step_rank}",
                    "orientation_seed": orientation_seed,
                    "matrix_seed": orientation_seed,
                    "spectrum_family": "rank_deficient_step",
                    "setup_name": f"step_r{step_rank}_eta_{eta_label}",
                    "step_rank": step_rank,
                    "eta": eta,
                }
            )
            spectrum_index += 1
    spectra = pd.DataFrame(spectra)
    rows = []
    for spectrum in spectra.to_dict("records"):
        for budget in config["budgets"]:
            rows.append(
                {
                    **spectrum,
                    "dimension": config["dimension"],
                    "budget": budget,
                    "analysis_role": (
                        "primary"
                        if budget == config["primary_budget"]
                        else "sensitivity"
                    ),
                    "q_min": 0,
                    "q_max": config["q_maxima"][budget],
                    "min_residual_probes": config["min_residual_probes"],
                    "basis_trials": config["trials"],
                    "bootstrap_samples": config["bootstrap_samples"],
                    "qr_rtol": config["qr_rtol"],
                    "qr_atol": config["qr_atol"],
                    "zero_backward_rtol": config["zero_backward_rtol"],
                    "sketch_distribution": "rademacher",
                    "basis_seed_base": config["basis_seed_base"],
                    "bootstrap_seed_base": config["bootstrap_seed_base"],
                    "configuration_version": "rank_bridge_v1",
                }
            )
    manifest = pd.DataFrame(rows)
    expected_rows = len(config["step_ranks"]) * len(config["tail_levels"]) * len(
        config["budgets"]
    )
    if len(manifest) != expected_rows:
        raise RuntimeError("Manifest cross-product is incomplete.")
    paired = manifest.groupby("step_rank")[["orientation_seed", "matrix_seed"]].nunique()
    if np.any(paired.to_numpy() != 1):
        raise RuntimeError("Tail levels do not share one orientation per step rank.")
    return manifest


def make_signal_basis(dimension, step_rank, orientation_seed):
    rng = np.random.default_rng(orientation_seed)
    gaussian = rng.normal(size=(dimension, step_rank))
    basis, _ = la.qr(gaussian, mode="economic")
    return basis


def apply_structured_step(vector, signal_basis, eta):
    vector = np.asarray(vector, dtype=np.float64)
    return eta * vector + (1.0 - eta) * signal_basis @ (signal_basis.T @ vector)


def spectral_tail_energy(dimension, step_rank, eta, rank):
    if rank < 0 or rank > dimension:
        raise ValueError("rank is outside the spectrum.")
    if rank < step_rank:
        return float(step_rank - rank + (dimension - step_rank) * eta**2)
    return float((dimension - rank) * eta**2)


def spectrum_value(step_rank, eta, one_based_index):
    return 1.0 if one_based_index <= step_rank else eta


def _energy_state(signal_basis, projected_signal, basis_row_norm_sq, eta, rank):
    dimension, step_rank = signal_basis.shape
    projected_signal_row_norm_sq = np.sum(projected_signal**2, axis=1)
    gram = projected_signal.T @ projected_signal
    raw_gaussian = float(
        eta**2 * (dimension - rank)
        + 2.0 * eta * (1.0 - eta) * np.sum(projected_signal**2)
        + (1.0 - eta) ** 2 * np.sum(gram**2)
    )
    residual_diagonal = (
        eta * (1.0 - basis_row_norm_sq)
        + (1.0 - eta) * projected_signal_row_norm_sq
    )
    diagonal_energy = float(np.dot(residual_diagonal, residual_diagonal))
    raw_rademacher = raw_gaussian - diagonal_energy
    roundoff_tolerance = float(
        128.0
        * np.finfo(np.float64).eps
        * max(raw_gaussian, diagonal_energy, np.finfo(np.float64).tiny)
    )
    if raw_rademacher < -roundoff_tolerance:
        raise RuntimeError("Rademacher energy is negative beyond roundoff tolerance.")
    rademacher_clamped = bool(raw_rademacher < 0.0)
    return {
        "raw_gaussian_energy": raw_gaussian,
        "raw_rademacher_energy": raw_rademacher,
        "residual_diagonal_energy": diagonal_energy,
        "rademacher_roundoff_tolerance": roundoff_tolerance,
        "rademacher_roundoff_clamped": rademacher_clamped,
        "gaussian_energy": raw_gaussian,
        "rademacher_energy": max(0.0, raw_rademacher),
    }


def compute_nested_rank_path(
    signal_basis,
    eta,
    sketch_matrix,
    qr_rtol=DEFAULT_QR_RTOL,
    qr_atol=DEFAULT_QR_ATOL,
    zero_backward_rtol=DEFAULT_ZERO_BACKWARD_RTOL,
):
    """Construct one nested rank-aware path and exact stable residual energies."""
    signal_basis = np.asarray(signal_basis, dtype=np.float64)
    sketch_matrix = np.asarray(sketch_matrix, dtype=np.float64)
    if signal_basis.ndim != 2 or sketch_matrix.ndim != 2:
        raise ValueError("signal_basis and sketch_matrix must be matrices.")
    dimension, step_rank = signal_basis.shape
    if sketch_matrix.shape[0] != dimension:
        raise ValueError("Sketch and signal basis dimensions do not match.")
    eta = _validate_float("eta", eta, 0.0, 1.0)
    qr_rtol = _validate_float("qr_rtol", qr_rtol)
    qr_atol = _validate_float("qr_atol", qr_atol)
    zero_backward_rtol = _validate_float(
        "zero_backward_rtol", zero_backward_rtol
    )
    if not np.all(np.isfinite(signal_basis)) or not np.all(np.isfinite(sketch_matrix)):
        raise ValueError("Basis and sketch entries must be finite.")
    if not np.allclose(
        signal_basis.T @ signal_basis,
        np.eye(step_rank),
        rtol=1e-11,
        atol=1e-12,
    ):
        raise ValueError("signal_basis must have orthonormal columns.")

    q_max = sketch_matrix.shape[1]
    accepted_basis = np.empty((dimension, q_max), dtype=np.float64)
    accepted_rank = 0
    projected_signal = signal_basis.copy()
    basis_row_norm_sq = np.zeros(dimension, dtype=np.float64)
    reference_scale_sq = 0.0
    matrix_frobenius = float(
        np.sqrt(step_rank + (dimension - step_rank) * eta**2)
    )
    oracle = MatVecOracle(
        lambda vector: apply_structured_step(vector, signal_basis, eta), d=dimension
    )
    rows = []

    def append_row(q, accepted, projected_norm, cutoff):
        energy = _energy_state(
            signal_basis, projected_signal, basis_row_norm_sq, eta, accepted_rank
        )
        backward_ratio = float(
            np.sqrt(max(0.0, energy["raw_gaussian_energy"])) / matrix_frobenius
        )
        exact_zero = bool(
            eta == 0.0
            and accepted_rank == step_rank
            and backward_ratio <= zero_backward_rtol
        )
        if exact_zero:
            energy["gaussian_energy"] = 0.0
            energy["rademacher_energy"] = 0.0
        rows.append(
            {
                "q": q,
                "r_actual": accepted_rank,
                "rank_gain_accepted": accepted,
                "rank_efficiency": (
                    float(accepted_rank / q) if q > 0 else np.nan
                ),
                "rejected_query_count": q - accepted_rank,
                "constructed_basis_queries": oracle.query_count,
                "reference_scale": float(np.sqrt(reference_scale_sq)),
                "projected_sample_norm": projected_norm,
                "rank_acceptance_cutoff": cutoff,
                "backward_residual_ratio": backward_ratio,
                "exact_zero_canonicalized": exact_zero,
                **energy,
            }
        )

    append_row(0, False, np.nan, np.nan)
    for column_index in range(q_max):
        sample = oracle(sketch_matrix[:, column_index])
        reference_scale_sq += float(np.dot(sample, sample))
        if accepted_rank > 0:
            current_basis = accepted_basis[:, :accepted_rank]
            residual = sample - current_basis @ (current_basis.T @ sample)
            residual = residual - current_basis @ (current_basis.T @ residual)
        else:
            residual = sample.copy()
        projected_norm = float(la.norm(residual))
        reference_scale = float(np.sqrt(reference_scale_sq))
        candidate, candidate_rank = _rank_aware_qr(
            residual[:, None],
            reference_scale=reference_scale,
            rtol=qr_rtol,
            atol=qr_atol,
        )
        accepted = bool(candidate_rank == 1)
        if accepted:
            direction = candidate[:, 0]
            accepted_basis[:, accepted_rank] = direction
            accepted_rank += 1
            oracle(direction)
            coefficients = direction @ signal_basis
            projected_signal -= np.outer(direction, coefficients)
            basis_row_norm_sq += direction**2
        cutoff = max(qr_atol, qr_rtol * reference_scale)
        append_row(column_index + 1, accepted, projected_norm, cutoff)

    path = pd.DataFrame(rows)
    if not np.all(path["constructed_basis_queries"] == path["q"] + path["r_actual"]):
        raise RuntimeError("Constructed basis query accounting failed.")
    ranks = path["r_actual"].to_numpy(dtype=int)
    if np.any(np.diff(ranks) < 0) or np.any(np.diff(ranks) > 1):
        raise RuntimeError("Nested rank path violates one-step monotonicity.")
    rejected = (~path["rank_gain_accepted"].to_numpy(dtype=bool)[1:]).cumsum()
    if not np.array_equal(rejected, path["rejected_query_count"].to_numpy(dtype=int)[1:]):
        raise RuntimeError("Rejected-transition count does not equal q-r_q.")
    if accepted_rank > 0:
        gram_error = float(
            la.norm(
                accepted_basis[:, :accepted_rank].T
                @ accepted_basis[:, :accepted_rank]
                - np.eye(accepted_rank),
                ord=2,
            )
        )
    else:
        gram_error = 0.0
    path["final_basis_orthogonality_error"] = gram_error
    return path


def _expand_path_for_budget(path, dimension, step_rank, eta, budget, q_max):
    selected = path.loc[path["q"] <= q_max].copy().reset_index(drop=True)
    selected["budget"] = budget
    selected["ell"] = budget - selected["q"] - selected["r_actual"]
    selected["accounted_total_queries"] = budget
    if np.any(selected["ell"] <= 0):
        raise RuntimeError("A saved allocation has no residual probes.")

    full_risks = []
    rank_risks = []
    for row in selected.itertuples(index=False):
        full_tail = spectral_tail_energy(dimension, step_rank, eta, int(row.q))
        rank_tail = spectral_tail_energy(
            dimension, step_rank, eta, int(row.r_actual)
        )
        full_risks.append(2.0 * full_tail / (budget - 2 * int(row.q)))
        rank_risks.append(2.0 * rank_tail / int(row.ell))
    selected["full_risk"] = full_risks
    selected["rank_risk"] = rank_risks
    selected["gaussian_risk"] = 2.0 * selected["gaussian_energy"] / selected["ell"]
    selected["rademacher_risk"] = (
        2.0 * selected["rademacher_energy"] / selected["ell"]
    )

    transition_fields = {
        "delta_full_risk": [np.nan],
        "delta_rank_risk": [np.nan],
        "delta_gaussian_risk": [np.nan],
        "delta_rademacher_risk": [np.nan],
        "full_marginal": [np.nan],
        "ideal_rank_marginal": [np.nan],
        "realized_gaussian_marginal": [np.nan],
        "realized_rademacher_marginal": [np.nan],
        "failed_penalty_rank_transition": [0.0],
        "failed_penalty_gaussian_transition": [0.0],
        "failed_penalty_rademacher_transition": [0.0],
        "rank_marginal_identity_residual": [np.nan],
        "gaussian_marginal_identity_residual": [np.nan],
        "rademacher_marginal_identity_residual": [np.nan],
    }
    for q in range(1, len(selected)):
        old = selected.iloc[q - 1]
        new = selected.iloc[q]
        delta_full = float(new["full_risk"] - old["full_risk"])
        delta_rank = float(new["rank_risk"] - old["rank_risk"])
        delta_gaussian = float(new["gaussian_risk"] - old["gaussian_risk"])
        delta_rademacher = float(
            new["rademacher_risk"] - old["rademacher_risk"]
        )
        for key, value in (
            ("delta_full_risk", delta_full),
            ("delta_rank_risk", delta_rank),
            ("delta_gaussian_risk", delta_gaussian),
            ("delta_rademacher_risk", delta_rademacher),
        ):
            transition_fields[key].append(value)
        old_q = q - 1
        old_rank = int(old["r_actual"])
        old_denominator = budget - old_q - old_rank
        full_tail = spectral_tail_energy(dimension, step_rank, eta, old_q)
        full_marginal = (
            (budget - 2 * old_q) * spectrum_value(step_rank, eta, q) ** 2
            - 2.0 * full_tail
        )
        transition_fields["full_marginal"].append(full_marginal)
        accepted = bool(new["rank_gain_accepted"])
        if accepted:
            rank_tail = spectral_tail_energy(
                dimension, step_rank, eta, old_rank
            )
            rank_marginal = (
                old_denominator
                * spectrum_value(step_rank, eta, old_rank + 1) ** 2
                - 2.0 * rank_tail
            )
            gaussian_marginal = (
                old_denominator
                * (old["gaussian_energy"] - new["gaussian_energy"])
                - 2.0 * old["gaussian_energy"]
            )
            rademacher_marginal = (
                old_denominator
                * (old["rademacher_energy"] - new["rademacher_energy"])
                - 2.0 * old["rademacher_energy"]
            )
            scale = old_denominator * (old_denominator - 2)
            transition_fields["ideal_rank_marginal"].append(rank_marginal)
            transition_fields["realized_gaussian_marginal"].append(
                gaussian_marginal
            )
            transition_fields["realized_rademacher_marginal"].append(
                rademacher_marginal
            )
            transition_fields["failed_penalty_rank_transition"].append(0.0)
            transition_fields["failed_penalty_gaussian_transition"].append(0.0)
            transition_fields["failed_penalty_rademacher_transition"].append(0.0)
            transition_fields["rank_marginal_identity_residual"].append(
                delta_rank + 2.0 * rank_marginal / scale
            )
            transition_fields["gaussian_marginal_identity_residual"].append(
                delta_gaussian + 2.0 * gaussian_marginal / scale
            )
            transition_fields["rademacher_marginal_identity_residual"].append(
                delta_rademacher + 2.0 * rademacher_marginal / scale
            )
        else:
            denominator_product = old_denominator * (old_denominator - 1)
            rank_penalty = 2.0 * spectral_tail_energy(
                dimension, step_rank, eta, old_rank
            ) / denominator_product
            gaussian_penalty = (
                2.0 * old["gaussian_energy"] / denominator_product
            )
            rademacher_penalty = (
                2.0 * old["rademacher_energy"] / denominator_product
            )
            transition_fields["ideal_rank_marginal"].append(np.nan)
            transition_fields["realized_gaussian_marginal"].append(np.nan)
            transition_fields["realized_rademacher_marginal"].append(np.nan)
            transition_fields["failed_penalty_rank_transition"].append(rank_penalty)
            transition_fields["failed_penalty_gaussian_transition"].append(
                gaussian_penalty
            )
            transition_fields["failed_penalty_rademacher_transition"].append(
                rademacher_penalty
            )
            transition_fields["rank_marginal_identity_residual"].append(
                delta_rank - rank_penalty
            )
            transition_fields["gaussian_marginal_identity_residual"].append(
                delta_gaussian - gaussian_penalty
            )
            transition_fields["rademacher_marginal_identity_residual"].append(
                delta_rademacher - rademacher_penalty
            )
    for key, values in transition_fields.items():
        selected[key] = values
    selected["cumulative_failed_penalty_rank"] = selected[
        "failed_penalty_rank_transition"
    ].cumsum()
    selected["cumulative_failed_penalty_gaussian"] = selected[
        "failed_penalty_gaussian_transition"
    ].cumsum()
    selected["cumulative_failed_penalty_rademacher"] = selected[
        "failed_penalty_rademacher_transition"
    ].cumsum()
    return selected


def _curve_and_minimizer_summaries(trials, config):
    curve_rows = []
    minimizer_rows = []
    frequency_rows = []
    group_keys = [
        "spectrum_index",
        "rank_index",
        "setup_name",
        "step_rank",
        "eta",
        "orientation_seed",
        "budget",
        "analysis_role",
    ]
    for key_values, group in trials.groupby(group_keys, sort=True, dropna=False):
        group_info = dict(zip(group_keys, key_values))
        q_values = np.sort(group["q"].unique())
        trial_indices = np.sort(group["basis_trial"].unique())
        curve_by_q = group.groupby("q", sort=True)
        local_rows = {int(q): {**group_info, "q": int(q)} for q in q_values}
        for q, selected in curve_by_q:
            row = local_rows[int(q)]
            row["observations"] = len(selected)
            for diagnostic in (
                "r_actual",
                "ell",
                "rank_efficiency",
                "rejected_query_count",
                "cumulative_failed_penalty_rank",
                "cumulative_failed_penalty_gaussian",
                "cumulative_failed_penalty_rademacher",
            ):
                values = selected[diagnostic].to_numpy(dtype=np.float64)
                row[f"mean_{diagnostic}"] = (
                    np.nan if np.all(np.isnan(values)) else float(np.nanmean(values))
                )
            for risk_name in RISK_COLUMNS:
                values = selected[risk_name].to_numpy(dtype=np.float64)
                mean = float(np.mean(values))
                se = (
                    float(np.std(values, ddof=1) / np.sqrt(values.size))
                    if values.size > 1
                    else 0.0
                )
                row[f"mean_{risk_name}"] = mean
                row[f"se_{risk_name}"] = se
                row[f"ci_low_{risk_name}"] = max(0.0, mean - 1.96 * se)
                row[f"ci_high_{risk_name}"] = mean + 1.96 * se

        for risk_index, risk_name in enumerate(RISK_COLUMNS):
            mean_curve = np.array(
                [local_rows[int(q)][f"mean_{risk_name}"] for q in q_values]
            )
            regrets = regret_columns(q_values, mean_curve)
            for index, q in enumerate(q_values):
                row = local_rows[int(q)]
                label = RISK_LABELS[risk_name]
                row[f"additive_regret_{label}"] = regrets["additive_regret"][index]
                row[f"baseline_normalized_additive_regret_{label}"] = regrets[
                    "baseline_normalized_additive_regret"
                ][index]
                row[f"multiplicative_regret_{label}"] = regrets[
                    "multiplicative_regret"
                ][index]
                row[f"minimum_is_zero_{label}"] = regrets["minimum_is_zero"]
                row[f"minimum_is_plateau_{label}"] = regrets[
                    "minimum_is_plateau"
                ]

            pivot = group.pivot(index="basis_trial", columns="q", values=risk_name)
            pivot = pivot.reindex(index=trial_indices, columns=q_values)
            if pivot.isna().any().any():
                raise RuntimeError("Incomplete paired risk curve.")
            risk_matrix = pivot.to_numpy(dtype=np.float64)
            bootstrap_seed = (
                config["bootstrap_seed_base"]
                + int(group_info["rank_index"]) * len(RISK_COLUMNS)
                + risk_index
            )
            bootstrap = paired_bootstrap_minimizer_frequencies(
                risk_matrix,
                q_values,
                config["bootstrap_samples"],
                bootstrap_seed,
            )
            plateau = minimum_plateau(q_values, mean_curve)
            trial_minimizers = q_values[np.argmin(risk_matrix, axis=1)]
            minimizer_rows.append(
                {
                    **group_info,
                    "risk_name": risk_name,
                    "representative_minimizer_q": plateau["representative_q"],
                    "minimum_q_smallest": plateau["minimum_q_smallest"],
                    "minimum_q_largest": plateau["minimum_q_largest"],
                    "minimum_q_count": plateau["minimum_q_count"],
                    "minimum_is_zero": plateau["minimum_is_zero"],
                    "minimum_is_plateau": plateau["minimum_is_plateau"],
                    "minimum_mean_risk": plateau["minimum_risk"],
                    "mean_trial_minimizer_q": float(np.mean(trial_minimizers)),
                    "median_trial_minimizer_q": float(np.median(trial_minimizers)),
                    "bootstrap_q_ci_low": bootstrap["ci_low"],
                    "bootstrap_q_ci_high": bootstrap["ci_high"],
                    "bootstrap_samples": config["bootstrap_samples"],
                    "bootstrap_seed": bootstrap_seed,
                }
            )
            for q, count, frequency in zip(
                bootstrap["q_values"],
                bootstrap["counts"],
                bootstrap["frequencies"],
            ):
                frequency_rows.append(
                    {
                        **group_info,
                        "risk_name": risk_name,
                        "q": int(q),
                        "bootstrap_count": int(count),
                        "bootstrap_frequency": float(frequency),
                        "bootstrap_samples": config["bootstrap_samples"],
                        "bootstrap_seed": bootstrap_seed,
                    }
                )
        curve_rows.extend(local_rows[int(q)] for q in q_values)
    return (
        pd.DataFrame(curve_rows),
        pd.DataFrame(minimizer_rows),
        pd.DataFrame(frequency_rows),
    )


def validate_outputs(manifest, trials, curves, minimizers, frequencies, config):
    spectrum_count = len(config["step_ranks"]) * len(config["tail_levels"])
    expected_trial_rows = spectrum_count * config["trials"] * sum(
        q_max + 1 for q_max in config["q_maxima"].values()
    )
    expected_curve_rows = spectrum_count * sum(
        q_max + 1 for q_max in config["q_maxima"].values()
    )
    expected_minimizers = spectrum_count * len(config["budgets"]) * len(
        RISK_COLUMNS
    )
    expected_frequencies = len(RISK_COLUMNS) * expected_curve_rows
    if len(manifest) != spectrum_count * len(config["budgets"]):
        raise RuntimeError("Manifest row count is incorrect.")
    if len(trials) != expected_trial_rows or len(curves) != expected_curve_rows:
        raise RuntimeError("Trial or curve row count is incorrect.")
    if len(minimizers) != expected_minimizers or len(frequencies) != expected_frequencies:
        raise RuntimeError("Minimizer output row count is incorrect.")
    if trials.duplicated(["spectrum_index", "budget", "basis_trial", "q"]).any():
        raise RuntimeError("Duplicate trial key detected.")
    if np.any(trials["q"] + trials["r_actual"] + trials["ell"] != trials["budget"]):
        raise RuntimeError("Accounted budget identity failed.")
    if np.any(
        trials["constructed_basis_queries"]
        != trials["q"] + trials["r_actual"]
    ):
        raise RuntimeError("Constructed basis query identity failed.")
    numeric = trials[list(RISK_COLUMNS) + ["gaussian_energy", "rademacher_energy"]]
    if np.any(~np.isfinite(numeric.to_numpy())) or np.any(numeric.to_numpy() < 0.0):
        raise RuntimeError("Risk outputs must be finite and nonnegative.")
    q_positive = trials["q"] > 0
    if not np.allclose(
        trials.loc[q_positive, "rank_efficiency"],
        trials.loc[q_positive, "r_actual"] / trials.loc[q_positive, "q"],
    ):
        raise RuntimeError("Rank efficiency is inconsistent.")
    if not trials.loc[~q_positive, "rank_efficiency"].isna().all():
        raise RuntimeError("Rank efficiency at q=0 must be NaN.")
    if np.any(trials["rejected_query_count"] != trials["q"] - trials["r_actual"]):
        raise RuntimeError("Rejected-query count is inconsistent.")
    identity_columns = [
        "rank_marginal_identity_residual",
        "gaussian_marginal_identity_residual",
        "rademacher_marginal_identity_residual",
    ]
    for column in identity_columns:
        values = trials[column].dropna().to_numpy(dtype=np.float64)
        scales = np.maximum(
            1.0,
            trials.loc[trials[column].notna(), list(RISK_COLUMNS)].max(axis=1).to_numpy(),
        )
        if np.any(np.abs(values) > 2e-10 * scales):
            raise RuntimeError(f"Marginal identity failed for {column}.")
    frequency_sums = frequencies.groupby(
        ["spectrum_index", "budget", "risk_name"]
    )["bootstrap_frequency"].sum()
    if not np.allclose(frequency_sums.to_numpy(), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("Bootstrap frequency support is incomplete.")
    paired = trials.groupby("step_rank")[["orientation_seed"]].nunique()
    if np.any(paired.to_numpy() != 1):
        raise RuntimeError("Orientation pairing across eta failed.")
    return {
        "trial_rows": expected_trial_rows,
        "curve_rows": expected_curve_rows,
        "minimizer_rows": expected_minimizers,
        "frequency_rows": expected_frequencies,
    }


def _atomic_write_frames(frames, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = {}
    for name, frame in frames.items():
        final_path = output_dir / OUTPUT_FILENAMES[name]
        temporary_path = output_dir / f".{OUTPUT_FILENAMES[name]}.tmp"
        frame.to_csv(temporary_path, index=False)
        temporary_paths[name] = (temporary_path, final_path)
    for temporary_path, final_path in temporary_paths.values():
        temporary_path.replace(final_path)
    return {name: final for name, (_, final) in temporary_paths.items()}


def run_rank_deficient_bridge(output_dir=DEFAULT_OUTPUT_DIR, **kwargs):
    config = validate_configuration(**kwargs)
    output_dir = Path(output_dir).resolve()
    if output_dir == DEFAULT_OUTPUT_DIR.resolve() and not _is_default_configuration(config):
        raise ValueError(
            "Nondefault configurations must use an alternate output directory."
        )
    manifest = build_spectrum_manifest(config)
    spectrum_manifest = manifest.drop_duplicates("spectrum_index").sort_values(
        "spectrum_index"
    )
    q_max_all = max(config["q_maxima"].values())
    records = []
    signal_bases = {
        int(step_rank): make_signal_basis(
            config["dimension"],
            int(step_rank),
            config["orientation_seed_base"] + rank_index,
        )
        for rank_index, step_rank in enumerate(config["step_ranks"])
    }
    for spectrum in spectrum_manifest.to_dict("records"):
        step_rank = int(spectrum["step_rank"])
        signal_basis = signal_bases[step_rank]
        print(
            f"Running {spectrum['setup_name']} "
            f"({int(spectrum['spectrum_index']) + 1}/{len(spectrum_manifest)})",
            flush=True,
        )
        for basis_trial in range(config["trials"]):
            basis_seed = config["basis_seed_base"] + basis_trial
            rng = np.random.default_rng(basis_seed)
            sketch_matrix = rng.choice(
                [-1.0, 1.0], size=(config["dimension"], q_max_all)
            )
            path = compute_nested_rank_path(
                signal_basis,
                float(spectrum["eta"]),
                sketch_matrix,
                qr_rtol=config["qr_rtol"],
                qr_atol=config["qr_atol"],
                zero_backward_rtol=config["zero_backward_rtol"],
            )
            for budget in config["budgets"]:
                expanded = _expand_path_for_budget(
                    path,
                    config["dimension"],
                    step_rank,
                    float(spectrum["eta"]),
                    budget,
                    config["q_maxima"][budget],
                )
                for key in (
                    "spectrum_index",
                    "rank_index",
                    "orientation_group_id",
                    "orientation_seed",
                    "matrix_seed",
                    "spectrum_family",
                    "setup_name",
                    "step_rank",
                    "eta",
                ):
                    expanded[key] = spectrum[key]
                expanded["analysis_role"] = (
                    "primary"
                    if budget == config["primary_budget"]
                    else "sensitivity"
                )
                expanded["basis_trial"] = basis_trial
                expanded["basis_seed"] = basis_seed
                expanded["exact_trace"] = step_rank + (
                    config["dimension"] - step_rank
                ) * float(spectrum["eta"])
                records.append(expanded)
    trials = pd.concat(records, ignore_index=True)
    curves, minimizers, frequencies = _curve_and_minimizer_summaries(
        trials, config
    )
    validate_outputs(manifest, trials, curves, minimizers, frequencies, config)
    artifacts = _atomic_write_frames(
        {
            "manifest": manifest,
            "trials": trials,
            "curves": curves,
            "minimizers": minimizers,
            "frequencies": frequencies,
        },
        output_dir,
    )
    for path in artifacts.values():
        print(f"Saved {path}")
    return manifest, trials, curves, minimizers, frequencies


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--primary-budget", type=int, default=DEFAULT_PRIMARY_BUDGET)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--step-ranks", type=int, nargs="+", default=DEFAULT_STEP_RANKS)
    parser.add_argument("--tail-levels", type=float, nargs="+", default=DEFAULT_TAIL_LEVELS)
    parser.add_argument(
        "--min-residual-probes", type=int, default=DEFAULT_MIN_RESIDUAL_PROBES
    )
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--qr-rtol", type=float, default=DEFAULT_QR_RTOL)
    parser.add_argument("--qr-atol", type=float, default=DEFAULT_QR_ATOL)
    parser.add_argument(
        "--zero-backward-rtol", type=float, default=DEFAULT_ZERO_BACKWARD_RTOL
    )
    parser.add_argument(
        "--orientation-seed-base", type=int, default=DEFAULT_ORIENTATION_SEED_BASE
    )
    parser.add_argument("--basis-seed-base", type=int, default=DEFAULT_BASIS_SEED_BASE)
    parser.add_argument(
        "--bootstrap-seed-base", type=int, default=DEFAULT_BOOTSTRAP_SEED_BASE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = vars(parse_args())
    output_directory = arguments.pop("output_dir")
    run_rank_deficient_bridge(output_dir=output_directory, **arguments)
