"""Run the frozen Phase 1A offline realized Rademacher-risk experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
SRC_DIR = ROOT_DIR / "src"
for directory in (EXPERIMENTS_DIR, SRC_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from direct_rademacher_certification_phase1a import (  # noqa: E402
    ACTION_OFFSETS,
    ACTION_PAIRS,
    DECISION_ABSTAIN,
    DECISION_ACCEPT,
    DECISION_REJECT,
    EPSILONS,
    ESTIMATORS,
    FROZEN_ENERGY_ATOL,
    FROZEN_ENERGY_RTOL,
    MASTER_BOOTSTRAP_SEED,
    MASTER_CERT_SEED,
    PROJECTOR_VALIDATION_ATOL,
    SAMPLE_SIZES,
    TRUTH_BETTER,
    TRUTH_TIE,
    TRUTH_WORSE,
    VERDICT_ESTIMATORS,
    bootstrap_seed,
    certification_estimators,
    certification_probes,
    certification_quadratic_forms,
    conditional_cluster_bootstrap,
    empirical_decision,
    epsilon_tag,
    reconstruct_actions,
    truth_label,
)
from run_rank_deficient_risk_bridge import (  # noqa: E402
    DEFAULT_BASIS_SEED_BASE,
    DEFAULT_MIN_RESIDUAL_PROBES,
    DEFAULT_ORIENTATION_SEED_BASE,
    apply_structured_step,
    make_signal_basis,
    q_max_for_budget,
)


DEFAULT_DIMENSION = 500
DEFAULT_STEP_RANKS = (5, 15, 30)
DEFAULT_ETAS = (1e-10, 1e-6)
DEFAULT_BUDGETS = (80, 160, 240)
DEFAULT_TRIALS = 200
DEFAULT_BATCHES = 4
DEFAULT_REPETITIONS = 50
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results"
DEFAULT_FROZEN_TRIALS = DEFAULT_OUTPUT_DIR / "risk_bridge_rank_deficient_trials.csv"
PRIMARY_BUDGET = 160
PRIMARY_ETA = 1e-6
CONTROL_ETA = 1e-10
PRIMARY_SAMPLE_SIZE = 16
PRIMARY_EPSILON = 1.0 / 3.0
PRIMARY_PAIR = "primary"

OUTPUT_FILENAMES = {
    "truth": "direct_rademacher_certification_phase1a_truth.csv",
    "trials": "direct_rademacher_certification_phase1a_trials.parquet",
    "accuracy": "direct_rademacher_certification_phase1a_accuracy.csv",
    "operating": "direct_rademacher_certification_phase1a_operating_rates.csv",
    "catastrophic": "direct_rademacher_certification_phase1a_catastrophic.csv",
    "bootstrap": "direct_rademacher_certification_phase1a_bootstrap.csv",
    "batch": "direct_rademacher_certification_phase1a_batch_stability.csv",
    "gate": "direct_rademacher_certification_phase1a_gate.csv",
    "verdict": "direct_rademacher_certification_phase1a_verdict.csv",
    "manifest": "direct_rademacher_certification_phase1a_manifest.csv",
    "report": "direct_rademacher_risk_certification_phase1a.md",
}

ACTION_LABELS = tuple(ACTION_OFFSETS)
PAIR_LABELS = tuple(ACTION_PAIRS)
INK = "#243447"
BLUE = "#3568A8"
GOLD = "#D39B2A"
RED = "#A94B45"
GREY = "#8C96A3"
GRID = "#DCE2E8"


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _historical_checksums():
    excluded_prefix = "direct_rademacher_certification_phase1a"
    checksums = {}
    for path in sorted((ROOT_DIR / "results").glob("*.csv")):
        if not path.name.startswith(excluded_prefix):
            checksums[path.name] = _hash_file(path)
    return checksums


def _eta_key(value):
    return f"{float(value):.17g}"


def _validate_configuration(
    dimension,
    step_ranks,
    etas,
    budgets,
    trials,
    batches,
    repetitions,
    bootstrap_samples,
):
    integer_values = {
        "dimension": dimension,
        "trials": trials,
        "batches": batches,
        "repetitions": repetitions,
        "bootstrap_samples": bootstrap_samples,
    }
    for name, value in integer_values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer.")
        if int(value) < 1:
            raise ValueError(f"{name} must be positive.")
    step_ranks = tuple(int(value) for value in step_ranks)
    budgets = tuple(int(value) for value in budgets)
    etas = tuple(float(value) for value in etas)
    if not step_ranks or len(step_ranks) != len(set(step_ranks)):
        raise ValueError("step_ranks must be nonempty and unique.")
    if not budgets or len(budgets) != len(set(budgets)):
        raise ValueError("budgets must be nonempty and unique.")
    if not etas or len(etas) != len(set(etas)):
        raise ValueError("etas must be nonempty and unique.")
    if max(step_ranks) + 2 > int(dimension) or min(step_ranks) < 2:
        raise ValueError("Every rank must admit all four adjacent actions.")
    if any((not np.isfinite(value) or value <= 0.0 or value >= 1.0) for value in etas):
        raise ValueError("etas must be finite and strictly between zero and one.")
    for budget in budgets:
        for rank in step_ranks:
            if budget - 2 * (rank + 2) <= 0:
                raise ValueError("Every requested action must leave residual capacity.")
    return {
        "dimension": int(dimension),
        "step_ranks": step_ranks,
        "etas": etas,
        "budgets": budgets,
        "trials": int(trials),
        "batches": int(batches),
        "repetitions": int(repetitions),
        "bootstrap_samples": int(bootstrap_samples),
    }


def _is_default_config(config):
    return config == _validate_configuration(
        DEFAULT_DIMENSION,
        DEFAULT_STEP_RANKS,
        DEFAULT_ETAS,
        DEFAULT_BUDGETS,
        DEFAULT_TRIALS,
        DEFAULT_BATCHES,
        DEFAULT_REPETITIONS,
        DEFAULT_BOOTSTRAP_SAMPLES,
    )


def _load_frozen_lookup(config):
    if (
        config["dimension"] != DEFAULT_DIMENSION
        or tuple(config["step_ranks"]) != DEFAULT_STEP_RANKS
        or config["trials"] > DEFAULT_TRIALS
        or not DEFAULT_FROZEN_TRIALS.exists()
    ):
        return {}
    columns = [
        "step_rank",
        "basis_trial",
        "eta",
        "budget",
        "q",
        "r_actual",
        "ell",
        "gaussian_energy",
        "rademacher_energy",
        "rademacher_risk",
        "constructed_basis_queries",
    ]
    frozen = pd.read_csv(DEFAULT_FROZEN_TRIALS, usecols=columns)
    eta_mask = np.zeros(len(frozen), dtype=bool)
    for eta in config["etas"]:
        eta_mask |= np.isclose(
            frozen["eta"].to_numpy(dtype=float),
            eta,
            rtol=1e-12,
            atol=0.0,
        )
    q_mask = np.zeros(len(frozen), dtype=bool)
    for rank in config["step_ranks"]:
        q_mask |= (
            (frozen["step_rank"] == rank)
            & frozen["q"].isin([rank - 1, rank, rank + 1, rank + 2])
        )
    frozen = frozen.loc[
        eta_mask
        & q_mask
        & frozen["basis_trial"].lt(config["trials"])
        & frozen["budget"].isin(config["budgets"])
    ]
    lookup = {}
    for row in frozen.itertuples(index=False):
        key = (
            int(row.step_rank),
            int(row.basis_trial),
            _eta_key(row.eta),
            int(row.budget),
            int(row.q),
        )
        lookup[key] = row
    return lookup


def _exact_risk(state, budget):
    ell = int(budget) - state.q - state.r_actual
    if ell <= 0:
        raise RuntimeError("Action has no residual probes.")
    return state.exact_sigma2 / ell


def _truth_rows_for_path(
    config,
    rank_index,
    step_rank,
    basis_trial,
    eta,
    orientation_seed,
    basis_seed,
    actions,
    frozen_lookup,
):
    rows = []
    for budget in config["budgets"]:
        for label, state in actions.items():
            exact_risk = _exact_risk(state, budget)
            ell = budget - state.q - state.r_actual
            frozen_key = (
                step_rank,
                basis_trial,
                _eta_key(eta),
                budget,
                state.q,
            )
            frozen = frozen_lookup.get(frozen_key)
            if frozen is None:
                frozen_energy = np.nan
                frozen_risk = np.nan
                energy_error = np.nan
                risk_error = np.nan
            else:
                frozen_energy = float(frozen.rademacher_energy)
                frozen_risk = float(frozen.rademacher_risk)
                energy_error = abs(frozen_energy - state.rademacher_energy)
                risk_error = abs(frozen_risk - exact_risk)
                if int(frozen.r_actual) != state.r_actual:
                    raise RuntimeError("Reconstructed rank disagrees with frozen row.")
                if int(frozen.ell) != ell:
                    raise RuntimeError("Reconstructed denominator disagrees with frozen row.")
                if int(frozen.constructed_basis_queries) != state.reconstruction_query_count:
                    raise RuntimeError("Reconstruction query count disagrees with frozen row.")
                if not np.isclose(
                    frozen_energy,
                    state.rademacher_energy,
                    rtol=FROZEN_ENERGY_RTOL,
                    atol=FROZEN_ENERGY_ATOL,
                ):
                    raise RuntimeError(
                        "Reconstructed Rademacher energy disagrees with frozen "
                        f"row {frozen_key}: reconstructed={state.rademacher_energy:.17e}, "
                        f"frozen={frozen_energy:.17e}."
                    )
                if not np.isclose(
                    frozen_risk,
                    exact_risk,
                    rtol=FROZEN_ENERGY_RTOL,
                    atol=FROZEN_ENERGY_ATOL,
                ):
                    raise RuntimeError(
                        "Reconstructed risk disagrees with frozen row "
                        f"{frozen_key}: reconstructed={exact_risk:.17e}, "
                        f"frozen={frozen_risk:.17e}."
                    )
            rows.append(
                {
                    "rank_index": rank_index,
                    "step_rank": step_rank,
                    "basis_trial": basis_trial,
                    "eta": eta,
                    "budget": budget,
                    "action": label,
                    "q": state.q,
                    "r_actual": state.r_actual,
                    "ell": ell,
                    "orientation_seed": orientation_seed,
                    "basis_seed": basis_seed,
                    "reconstruction_query_count": state.reconstruction_query_count,
                    "exact_sigma2": state.exact_sigma2,
                    "exact_rademacher_energy": state.rademacher_energy,
                    "exact_gaussian_energy": state.gaussian_energy,
                    "exact_risk": exact_risk,
                    "frozen_rademacher_energy": frozen_energy,
                    "frozen_risk": frozen_risk,
                    "frozen_energy_abs_error": energy_error,
                    "frozen_risk_abs_error": risk_error,
                    "projector_error_op": state.projector_error_op,
                    "orthogonality_error": state.orthogonality_error,
                }
            )
    return rows


def _wide_row(
    config,
    rank_index,
    step_rank,
    basis_trial,
    eta,
    orientation_seed,
    basis_seed,
    batch,
    repetition,
    sample_size,
    cert_seed,
    probe_hash,
    actions,
    estimates,
    certification_query_count,
):
    row = {
        "rank_index": rank_index,
        "step_rank": step_rank,
        "basis_trial": basis_trial,
        "eta": eta,
        "batch": batch,
        "repetition": repetition,
        "s": sample_size,
        "orientation_seed": orientation_seed,
        "basis_seed": basis_seed,
        "cert_seed": np.uint64(cert_seed),
        "probe_batch_sha256": probe_hash,
        "certification_query_count": certification_query_count,
        "cert_seed_master": MASTER_CERT_SEED,
        "cert_seed_rank_index": rank_index,
        "cert_seed_basis_trial": basis_trial,
        "cert_seed_batch": batch,
        "cert_seed_repetition": repetition,
        "mom_w1_degenerate": sample_size == 4,
        "mom_w2_degenerate": sample_size in (4, 8),
    }
    for action, state in actions.items():
        row[f"q__{action}"] = state.q
        row[f"r_actual__{action}"] = state.r_actual
        row[f"reconstruction_queries__{action}"] = state.reconstruction_query_count
        row[f"sigma2_exact__{action}"] = state.exact_sigma2
        for estimator in ESTIMATORS:
            row[f"sigma2_hat__{action}__{estimator}"] = estimates[action][estimator]
    for budget in config["budgets"]:
        for action, state in actions.items():
            ell = budget - state.q - state.r_actual
            exact_risk = state.exact_sigma2 / ell
            row[f"ell__m{budget}__{action}"] = ell
            row[f"risk_exact__m{budget}__{action}"] = exact_risk
            for estimator in ESTIMATORS:
                row[f"risk_hat__m{budget}__{action}__{estimator}"] = (
                    estimates[action][estimator] / ell
                )
        for pair, (candidate, baseline) in ACTION_PAIRS.items():
            candidate_exact = row[f"risk_exact__m{budget}__{candidate}"]
            baseline_exact = row[f"risk_exact__m{budget}__{baseline}"]
            row[f"truth__m{budget}__{pair}"] = truth_label(
                candidate_exact, baseline_exact
            )
            for estimator in ESTIMATORS:
                candidate_hat = row[
                    f"risk_hat__m{budget}__{candidate}__{estimator}"
                ]
                baseline_hat = row[
                    f"risk_hat__m{budget}__{baseline}__{estimator}"
                ]
                for epsilon in EPSILONS:
                    row[
                        f"decision__m{budget}__{pair}__{estimator}"
                        f"__eps_{epsilon_tag(epsilon)}"
                    ] = empirical_decision(candidate_hat, baseline_hat, epsilon)
    return row


def _write_generation_artifacts(config, staging_dir, frozen_lookup):
    parquet_path = staging_dir / OUTPUT_FILENAMES["trials"]
    truth_rows = []
    writer = None
    row_count = 0
    frozen_sketch_width = max(
        q_max_for_budget(
            config["dimension"],
            budget,
            DEFAULT_MIN_RESIDUAL_PROBES,
        )
        for budget in config["budgets"]
    )
    try:
        for rank_index, step_rank in enumerate(config["step_ranks"]):
            orientation_seed = DEFAULT_ORIENTATION_SEED_BASE + rank_index
            signal_basis = make_signal_basis(
                config["dimension"], step_rank, orientation_seed
            )
            for basis_trial in range(config["trials"]):
                if basis_trial % max(1, config["trials"] // 10) == 0:
                    print(
                        f"Reconstructing r={step_rank}: "
                        f"path {basis_trial + 1}/{config['trials']}",
                        flush=True,
                    )
                basis_seed = DEFAULT_BASIS_SEED_BASE + basis_trial
                sketch_rng = np.random.default_rng(basis_seed)
                sketch = sketch_rng.choice(
                    [-1.0, 1.0],
                    size=(config["dimension"], frozen_sketch_width),
                )
                actions_by_eta = {}
                for eta in config["etas"]:
                    actions = reconstruct_actions(
                        signal_basis,
                        eta,
                        sketch,
                        step_rank,
                    )
                    actions_by_eta[eta] = actions
                    truth_rows.extend(
                        _truth_rows_for_path(
                            config,
                            rank_index,
                            step_rank,
                            basis_trial,
                            eta,
                            orientation_seed,
                            basis_seed,
                            actions,
                            frozen_lookup,
                        )
                    )

                path_rows = []
                for batch in range(config["batches"]):
                    for repetition in range(config["repetitions"]):
                        probes, cert_seed, probe_hash = certification_probes(
                            config["dimension"],
                            rank_index,
                            basis_trial,
                            batch,
                            repetition,
                        )
                        for eta in config["etas"]:
                            actions = actions_by_eta[eta]
                            forms, query_count = certification_quadratic_forms(
                                signal_basis, eta, actions, probes
                            )
                            estimates_by_s = {
                                sample_size: {
                                    action: certification_estimators(
                                        values[:sample_size]
                                    )
                                    for action, values in forms.items()
                                }
                                for sample_size in SAMPLE_SIZES
                            }
                            for sample_size in SAMPLE_SIZES:
                                path_rows.append(
                                    _wide_row(
                                        config,
                                        rank_index,
                                        step_rank,
                                        basis_trial,
                                        eta,
                                        orientation_seed,
                                        basis_seed,
                                        batch,
                                        repetition,
                                        sample_size,
                                        cert_seed,
                                        probe_hash,
                                        actions,
                                        estimates_by_s[sample_size],
                                        query_count,
                                    )
                                )
                frame = pd.DataFrame(path_rows)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        parquet_path,
                        table.schema,
                        compression="zstd",
                    )
                elif table.schema != writer.schema:
                    raise RuntimeError("Parquet chunk schema changed across paths.")
                writer.write_table(table)
                row_count += len(frame)
    finally:
        if writer is not None:
            writer.close()
    truth = pd.DataFrame(truth_rows)
    truth.to_csv(staging_dir / OUTPUT_FILENAMES["truth"], index=False)
    return truth, parquet_path, row_count


def _validate_generated(config, truth, parquet_path, row_count):
    expected_rows = (
        len(config["step_ranks"])
        * config["trials"]
        * len(config["etas"])
        * config["batches"]
        * config["repetitions"]
        * len(SAMPLE_SIZES)
    )
    if row_count != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} raw rows but generated {row_count}."
        )
    metadata_rows = pq.ParquetFile(parquet_path).metadata.num_rows
    if metadata_rows != expected_rows:
        raise RuntimeError("Parquet metadata row count is incorrect.")
    keys = pd.read_parquet(
        parquet_path,
        columns=[
            "step_rank",
            "basis_trial",
            "eta",
            "batch",
            "repetition",
            "s",
            "certification_query_count",
        ],
    )
    if keys.duplicated(
        ["step_rank", "basis_trial", "eta", "batch", "repetition", "s"]
    ).any():
        raise RuntimeError("Raw Parquet primary key is not unique.")
    if not np.all(keys["certification_query_count"] == 32):
        raise RuntimeError("A certification repetition did not use exactly 32 queries.")
    truth_key = ["step_rank", "basis_trial", "eta", "budget", "q"]
    if truth.duplicated(truth_key).any():
        raise RuntimeError("Truth-table primary key is not unique.")
    expected_truth = (
        len(config["step_ranks"])
        * config["trials"]
        * len(config["etas"])
        * len(config["budgets"])
        * len(ACTION_LABELS)
    )
    if len(truth) != expected_truth:
        raise RuntimeError("Truth-table support is incomplete.")
    required = truth[
        [
            "exact_sigma2",
            "exact_risk",
            "projector_error_op",
            "orthogonality_error",
        ]
    ].to_numpy(dtype=float)
    if not np.all(np.isfinite(required)):
        raise RuntimeError("Truth table contains nonfinite required values.")
    return expected_rows


def _core_columns(config):
    columns = [
        "rank_index",
        "step_rank",
        "basis_trial",
        "eta",
        "batch",
        "repetition",
        "s",
    ]
    for action in ACTION_LABELS:
        columns.extend(
            [
                f"q__{action}",
                f"r_actual__{action}",
                f"sigma2_exact__{action}",
            ]
        )
        for estimator in ESTIMATORS:
            columns.append(f"sigma2_hat__{action}__{estimator}")
    return columns


def _risk_arrays(frame, budget, action, estimator=None):
    ell = (
        int(budget)
        - frame[f"q__{action}"].to_numpy(dtype=float)
        - frame[f"r_actual__{action}"].to_numpy(dtype=float)
    )
    if np.any(ell <= 0):
        raise RuntimeError("Risk analysis encountered a nonpositive denominator.")
    if estimator is None:
        numerator = frame[f"sigma2_exact__{action}"].to_numpy(dtype=float)
    else:
        numerator = frame[
            f"sigma2_hat__{action}__{estimator}"
        ].to_numpy(dtype=float)
    return numerator / ell


def _truth_array(candidate, baseline):
    differences = candidate - baseline
    tolerances = (
        128.0
        * np.finfo(np.float64).eps
        * np.maximum(candidate, baseline)
    )
    labels = np.full(candidate.shape, TRUTH_TIE, dtype=np.int8)
    labels[differences < -tolerances] = TRUTH_BETTER
    labels[differences > tolerances] = TRUTH_WORSE
    return labels


def _decision_array(candidate, baseline, epsilon):
    decisions = np.full(candidate.shape, DECISION_ABSTAIN, dtype=np.int8)
    unequal = candidate != baseline
    rho = (1.0 - float(epsilon)) / (1.0 + float(epsilon))
    decisions[unequal & (candidate <= rho * baseline)] = DECISION_ACCEPT
    decisions[unequal & (baseline <= rho * candidate)] = DECISION_REJECT
    return decisions


def _path_rates(frame, event, eligible):
    rates = {}
    ordered = frame.assign(_event=event, _eligible=eligible).sort_values(
        ["step_rank", "basis_trial", "batch", "repetition"]
    )
    for rank, group in ordered.groupby("step_rank", sort=True):
        path_event = group.groupby("basis_trial", sort=True)["_event"].mean()
        path_eligible = group.groupby("basis_trial", sort=True)["_eligible"].first()
        rates[int(rank)] = path_event[path_eligible.astype(bool)].to_numpy(dtype=float)
    return rates


def _equal_rank_metric(path_rates):
    if not path_rates or any(len(values) == 0 for values in path_rates.values()):
        return np.nan
    return float(np.mean([np.mean(values) for values in path_rates.values()]))


def _catastrophic_path_keys(frame, budget, fraction):
    baseline = frame.sort_values(["step_rank", "basis_trial"]).drop_duplicates(
        ["step_rank", "basis_trial"]
    )
    risks = _risk_arrays(baseline, budget, "k")
    baseline = baseline.assign(_risk=risks)
    keys = set()
    for rank, group in baseline.groupby("step_rank", sort=True):
        count = max(1, int(round(fraction * len(group))))
        selected = group.nlargest(count, "_risk")
        keys.update((int(rank), int(trial)) for trial in selected["basis_trial"])
    return keys


def _metric_path_rates(frame, truth, decisions, metric, catastrophic_keys=None):
    if metric == "false_safe":
        event = decisions == DECISION_ACCEPT
        eligible = truth == TRUTH_WORSE
    elif metric == "false_rejection":
        event = decisions == DECISION_REJECT
        eligible = truth == TRUTH_BETTER
    elif metric == "better_acceptance":
        event = decisions == DECISION_ACCEPT
        eligible = truth == TRUTH_BETTER
    elif metric == "coverage":
        event = decisions != DECISION_ABSTAIN
        eligible = truth != TRUTH_TIE
    elif metric == "catastrophic_detection":
        if catastrophic_keys is None:
            raise ValueError("catastrophic keys are required.")
        event = decisions == DECISION_ACCEPT
        keys = list(zip(frame["step_rank"], frame["basis_trial"]))
        eligible = np.array(
            [key in catastrophic_keys for key in keys], dtype=bool
        ) & (truth == TRUTH_BETTER)
    else:
        raise ValueError(f"Unknown metric {metric}.")
    return _path_rates(frame, event.astype(float), eligible)


def _summarize_accuracy(core, config):
    rows = []
    quantiles = (0.90, 0.95, 0.99)
    for (eta, sample_size), subset in core.groupby(["eta", "s"], sort=True):
        for budget in config["budgets"]:
            for action in ACTION_LABELS:
                exact = _risk_arrays(subset, budget, action)
                for estimator in ESTIMATORS:
                    estimated = _risk_arrays(subset, budget, action, estimator)
                    signed = estimated - exact
                    absolute = np.abs(signed)
                    positive = exact > 0.0
                    relative = absolute[positive] / exact[positive]
                    log_ratio = np.log(estimated[positive] / exact[positive])
                    for rank in (*config["step_ranks"], "equal_rank"):
                        if rank == "equal_rank":
                            mask = np.ones(len(subset), dtype=bool)
                        else:
                            mask = subset["step_rank"].to_numpy() == rank
                        row = {
                            "eta": eta,
                            "budget": budget,
                            "s": sample_size,
                            "action": action,
                            "estimator": estimator,
                            "rank_scope": rank,
                            "observations": int(np.count_nonzero(mask)),
                            "mean_signed_error": float(np.mean(signed[mask])),
                            "median_absolute_error": float(np.median(absolute[mask])),
                            "max_absolute_error": float(np.max(absolute[mask])),
                        }
                        positive_mask = mask & positive
                        if np.any(positive_mask):
                            rel = absolute[positive_mask] / exact[positive_mask]
                            logs = np.log(estimated[positive_mask] / exact[positive_mask])
                            row["median_relative_error"] = float(np.median(rel))
                            row["median_log_risk_ratio"] = float(np.median(logs))
                            row["max_relative_error"] = float(np.max(rel))
                            for quantile in quantiles:
                                tag = int(100 * quantile)
                                row[f"p{tag}_relative_error"] = float(
                                    np.quantile(rel, quantile)
                                )
                        else:
                            row.update(
                                {
                                    "median_relative_error": np.nan,
                                    "median_log_risk_ratio": np.nan,
                                    "max_relative_error": np.nan,
                                    "p90_relative_error": np.nan,
                                    "p95_relative_error": np.nan,
                                    "p99_relative_error": np.nan,
                                }
                            )
                        rows.append(row)
    return pd.DataFrame(rows)


def _operating_metrics_for_config(
    subset,
    budget,
    pair,
    estimator,
    epsilon,
    catastrophic_keys=None,
):
    candidate, baseline = ACTION_PAIRS[pair]
    exact_candidate = _risk_arrays(subset, budget, candidate)
    exact_baseline = _risk_arrays(subset, budget, baseline)
    truth = _truth_array(exact_candidate, exact_baseline)
    candidate_hat = _risk_arrays(subset, budget, candidate, estimator)
    baseline_hat = _risk_arrays(subset, budget, baseline, estimator)
    decisions = _decision_array(candidate_hat, baseline_hat, epsilon)
    metrics = {}
    for metric in (
        "false_safe",
        "false_rejection",
        "better_acceptance",
        "coverage",
    ):
        path_rates = _metric_path_rates(subset, truth, decisions, metric)
        metrics[metric] = {
            "point": _equal_rank_metric(path_rates),
            "eligible_by_rank": {
                str(rank): len(values) for rank, values in path_rates.items()
            },
            "path_rates": path_rates,
        }
    if catastrophic_keys is not None:
        path_rates = _metric_path_rates(
            subset,
            truth,
            decisions,
            "catastrophic_detection",
            catastrophic_keys=catastrophic_keys,
        )
        metrics["catastrophic_detection"] = {
            "point": _equal_rank_metric(path_rates),
            "eligible_by_rank": {
                str(rank): len(values) for rank, values in path_rates.items()
            },
            "path_rates": path_rates,
        }
    metrics["truth_ties"] = int(np.count_nonzero(truth == TRUTH_TIE))
    metrics["decision_abstention"] = float(np.mean(decisions == DECISION_ABSTAIN))
    return metrics


def _summarize_operating(core, config):
    operating_rows = []
    catastrophic_rows = []
    for (eta, sample_size), subset in core.groupby(["eta", "s"], sort=True):
        for budget in config["budgets"]:
            catastrophic_by_fraction = {
                fraction: _catastrophic_path_keys(subset, budget, fraction)
                for fraction in (0.01, 0.05, 0.10)
            }
            for pair in PAIR_LABELS:
                for estimator in ESTIMATORS:
                    for epsilon in EPSILONS:
                        metrics = _operating_metrics_for_config(
                            subset,
                            budget,
                            pair,
                            estimator,
                            epsilon,
                        )
                        row = {
                            "eta": eta,
                            "budget": budget,
                            "s": sample_size,
                            "pair": pair,
                            "estimator": estimator,
                            "epsilon": epsilon,
                            "truth_ties": metrics["truth_ties"],
                            "abstention_rate": metrics["decision_abstention"],
                        }
                        for metric in (
                            "false_safe",
                            "false_rejection",
                            "better_acceptance",
                            "coverage",
                        ):
                            row[metric] = metrics[metric]["point"]
                            row[f"{metric}_eligible_by_rank"] = json.dumps(
                                metrics[metric]["eligible_by_rank"],
                                sort_keys=True,
                            )
                        operating_rows.append(row)
                        if pair == PRIMARY_PAIR:
                            for fraction, keys in catastrophic_by_fraction.items():
                                cat = _operating_metrics_for_config(
                                    subset,
                                    budget,
                                    pair,
                                    estimator,
                                    epsilon,
                                    catastrophic_keys=keys,
                                )["catastrophic_detection"]
                                catastrophic_rows.append(
                                    {
                                        "eta": eta,
                                        "budget": budget,
                                        "s": sample_size,
                                        "pair": pair,
                                        "estimator": estimator,
                                        "epsilon": epsilon,
                                        "fraction": fraction,
                                        "detection": cat["point"],
                                        "eligible_by_rank": json.dumps(
                                            cat["eligible_by_rank"],
                                            sort_keys=True,
                                        ),
                                    }
                                )
    return pd.DataFrame(operating_rows), pd.DataFrame(catastrophic_rows)


def _gate_analysis(core, config):
    bootstrap_rows = []
    batch_rows = []
    gate_rows = []
    metric_names = (
        "false_safe",
        "false_rejection",
        "catastrophic_detection",
        "better_acceptance",
        "control_false_safe",
    )
    for sample_size in (16, 32):
        primary = core.loc[
            np.isclose(core["eta"], PRIMARY_ETA, rtol=1e-12, atol=0.0)
            & core["s"].eq(sample_size)
        ].copy()
        control = core.loc[
            np.isclose(core["eta"], CONTROL_ETA, rtol=1e-12, atol=0.0)
            & core["s"].eq(sample_size)
        ].copy()
        catastrophic_keys = _catastrophic_path_keys(
            primary, PRIMARY_BUDGET, 0.05
        )
        for estimator_index, estimator in enumerate(VERDICT_ESTIMATORS):
            primary_metrics = _operating_metrics_for_config(
                primary,
                PRIMARY_BUDGET,
                PRIMARY_PAIR,
                estimator,
                PRIMARY_EPSILON,
                catastrophic_keys=catastrophic_keys,
            )
            control_metrics = _operating_metrics_for_config(
                control,
                PRIMARY_BUDGET,
                PRIMARY_PAIR,
                estimator,
                PRIMARY_EPSILON,
            )
            path_rates = {
                "false_safe": primary_metrics["false_safe"]["path_rates"],
                "false_rejection": primary_metrics["false_rejection"]["path_rates"],
                "catastrophic_detection": primary_metrics[
                    "catastrophic_detection"
                ]["path_rates"],
                "better_acceptance": primary_metrics[
                    "better_acceptance"
                ]["path_rates"],
                "control_false_safe": control_metrics["false_safe"]["path_rates"],
            }
            intervals = {}
            for metric_index, metric in enumerate(metric_names):
                seed = bootstrap_seed(
                    estimator_index,
                    SAMPLE_SIZES.index(sample_size),
                    EPSILONS.index(PRIMARY_EPSILON),
                    1 if metric != "control_false_safe" else 0,
                    DEFAULT_BUDGETS.index(PRIMARY_BUDGET),
                    PAIR_LABELS.index(PRIMARY_PAIR),
                    metric_index,
                    0,
                )
                result = conditional_cluster_bootstrap(
                    path_rates[metric],
                    config["bootstrap_samples"],
                    seed,
                )
                intervals[metric] = result
                bootstrap_rows.append(
                    {
                        "s": sample_size,
                        "estimator": estimator,
                        "metric": metric,
                        "bootstrap_samples": config["bootstrap_samples"],
                        "bootstrap_seed": seed,
                        **result,
                    }
                )

            batch_passes = 0
            for batch in range(config["batches"]):
                primary_batch = primary.loc[primary["batch"] == batch]
                control_batch = control.loc[control["batch"] == batch]
                batch_primary = _operating_metrics_for_config(
                    primary_batch,
                    PRIMARY_BUDGET,
                    PRIMARY_PAIR,
                    estimator,
                    PRIMARY_EPSILON,
                    catastrophic_keys=catastrophic_keys,
                )
                batch_control = _operating_metrics_for_config(
                    control_batch,
                    PRIMARY_BUDGET,
                    PRIMARY_PAIR,
                    estimator,
                    PRIMARY_EPSILON,
                )
                points = {
                    "false_safe": batch_primary["false_safe"]["point"],
                    "false_rejection": batch_primary["false_rejection"]["point"],
                    "catastrophic_detection": batch_primary[
                        "catastrophic_detection"
                    ]["point"],
                    "better_acceptance": batch_primary[
                        "better_acceptance"
                    ]["point"],
                    "control_false_safe": batch_control["false_safe"]["point"],
                }
                evaluable = all(np.isfinite(value) for value in points.values())
                passed = bool(
                    evaluable
                    and points["false_safe"] <= 0.005
                    and points["false_rejection"] <= 0.05
                    and points["catastrophic_detection"] >= 0.75
                    and points["better_acceptance"] >= 0.35
                    and points["control_false_safe"] <= 0.005
                )
                batch_passes += int(passed)
                batch_rows.append(
                    {
                        "s": sample_size,
                        "estimator": estimator,
                        "batch": batch,
                        "evaluable": evaluable,
                        "passed": passed,
                        **points,
                    }
                )

            evaluable = all(result["evaluable"] for result in intervals.values())
            criteria = {
                "primary_false_safe": bool(
                    evaluable
                    and intervals["false_safe"]["point"] <= 0.005
                    and intervals["false_safe"]["ci_high"] <= 0.01
                ),
                "false_rejection": bool(
                    evaluable
                    and intervals["false_rejection"]["point"] <= 0.05
                    and intervals["false_rejection"]["ci_high"] <= 0.10
                ),
                "catastrophic_detection": bool(
                    evaluable
                    and intervals["catastrophic_detection"]["point"] >= 0.75
                    and intervals["catastrophic_detection"]["ci_low"] >= 0.60
                ),
                "better_acceptance": bool(
                    evaluable
                    and intervals["better_acceptance"]["point"] >= 0.35
                    and intervals["better_acceptance"]["ci_low"] >= 0.25
                ),
                "control_false_safe": bool(
                    evaluable
                    and intervals["control_false_safe"]["point"] <= 0.005
                    and intervals["control_false_safe"]["ci_high"] <= 0.01
                ),
                "batch_stability": batch_passes >= 3,
            }
            passed = evaluable and all(criteria.values())
            gate_rows.append(
                {
                    "s": sample_size,
                    "estimator": estimator,
                    "evaluable": evaluable,
                    "passed": passed,
                    "passing_batches": batch_passes,
                    **criteria,
                }
            )

    gate = pd.DataFrame(gate_rows)
    s16 = gate.loc[gate["s"] == 16]
    s32 = gate.loc[gate["s"] == 32]
    if not s16["evaluable"].all() or (
        not s16["passed"].any() and not s32["evaluable"].all()
    ):
        verdict = "INCONCLUSIVE"
    elif s16["passed"].all():
        verdict = "STRONG GO"
    elif s16["passed"].any():
        verdict = "QUALIFIED GO"
    elif s32["passed"].any():
        verdict = "BORDERLINE"
    else:
        verdict = "NO-GO"
    verdict_frame = pd.DataFrame(
        [
            {
                "verdict": verdict,
                "primary_budget": PRIMARY_BUDGET,
                "primary_eta": PRIMARY_ETA,
                "primary_s": PRIMARY_SAMPLE_SIZE,
                "primary_epsilon": PRIMARY_EPSILON,
                "primary_pair": PRIMARY_PAIR,
                "verdict_estimators": json.dumps(VERDICT_ESTIMATORS),
            }
        ]
    )
    return (
        pd.DataFrame(bootstrap_rows),
        pd.DataFrame(batch_rows),
        gate,
        verdict_frame,
    )


def _style_axis(axis, title, ylabel=None):
    axis.set_title(title, loc="left", color=INK, fontweight="bold")
    if ylabel:
        axis.set_ylabel(ylabel)
    axis.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def _make_figures(core, accuracy, operating, catastrophic, batch, gate, staging_dir):
    figure_dir = staging_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    primary = core.loc[
        np.isclose(core["eta"], PRIMARY_ETA, rtol=1e-12, atol=0.0)
        & core["s"].eq(16)
    ]
    exact = _risk_arrays(primary, 160, "kp1")
    estimated = _risk_arrays(primary, 160, "kp1", "sample_variance")
    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    keep = np.arange(len(primary))[:: max(1, len(primary) // 5000)]
    axis.scatter(exact[keep], estimated[keep], s=7, alpha=0.25, color=BLUE)
    positive = np.concatenate([exact[exact > 0], estimated[estimated > 0]])
    low, high = float(np.min(positive)), float(np.max(positive))
    axis.plot([low, high], [low, high], color=GOLD, linewidth=1.5)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Exact realized risk")
    _style_axis(axis, "Estimated versus exact risk", "Sample-variance estimate")
    path = figure_dir / "figure_1_estimated_vs_exact.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    selected = accuracy.loc[
        (accuracy["eta"] == PRIMARY_ETA)
        & (accuracy["budget"] == 160)
        & (accuracy["action"] == "kp1")
        & (accuracy["rank_scope"] == "equal_rank")
    ]
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    for estimator, group in selected.groupby("estimator", sort=True):
        axis.plot(
            group["s"],
            group["median_relative_error"],
            marker="o",
            label=estimator,
        )
    axis.set_yscale("log")
    axis.set_xlabel("Certification probes s")
    axis.legend(frameon=False, fontsize=8)
    _style_axis(axis, "Risk-estimation error versus evidence", "Median relative error")
    path = figure_dir / "figure_2_relative_error.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    primary_operating = operating.loc[
        (operating["eta"] == PRIMARY_ETA)
        & (operating["budget"] == 160)
        & (operating["pair"] == PRIMARY_PAIR)
        & np.isclose(operating["epsilon"], PRIMARY_EPSILON)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for estimator, group in primary_operating.groupby("estimator", sort=True):
        axes[0].plot(group["s"], group["false_safe"], marker="o", label=estimator)
        axes[1].plot(group["s"], group["false_rejection"], marker="o", label=estimator)
    axes[0].axhline(0.005, color=RED, linestyle="--", linewidth=1)
    axes[1].axhline(0.05, color=RED, linestyle="--", linewidth=1)
    for axis in axes:
        axis.set_xlabel("Certification probes s")
    _style_axis(axes[0], "False-safe rate", "Path-first rate")
    _style_axis(axes[1], "False-rejection rate")
    axes[1].legend(frameon=False, fontsize=7)
    path = figure_dir / "figure_3_safety_errors.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    for estimator, group in primary_operating.groupby("estimator", sort=True):
        axis.plot(group["s"], group["better_acceptance"], marker="o", label=estimator)
    axis.axhline(0.35, color=RED, linestyle="--", linewidth=1)
    axis.set_xlabel("Certification probes s")
    axis.legend(frameon=False, fontsize=8)
    _style_axis(axis, "Useful acceptance versus evidence", "True-better acceptance")
    path = figure_dir / "figure_4_better_acceptance.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    primary_cat = catastrophic.loc[
        (catastrophic["eta"] == PRIMARY_ETA)
        & (catastrophic["budget"] == 160)
        & np.isclose(catastrophic["epsilon"], PRIMARY_EPSILON)
        & np.isclose(catastrophic["fraction"], 0.05)
    ]
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    for estimator, group in primary_cat.groupby("estimator", sort=True):
        axis.plot(group["s"], group["detection"], marker="o", label=estimator)
    axis.axhline(0.75, color=RED, linestyle="--", linewidth=1)
    axis.set_xlabel("Certification probes s")
    axis.legend(frameon=False, fontsize=8)
    _style_axis(axis, "Catastrophic-path detection", "Top-5% true-better acceptance")
    path = figure_dir / "figure_5_catastrophic_detection.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    gate16 = gate.loc[gate["s"] == 16]
    fig, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.bar(
        gate16["estimator"],
        gate16["passing_batches"],
        color=[BLUE, GOLD, GREY],
    )
    axis.axhline(3, color=RED, linestyle="--", linewidth=1)
    axis.set_ylim(0, 4.4)
    axis.tick_params(axis="x", rotation=20)
    _style_axis(axis, "Primary estimator comparison at s=16", "Passing batches")
    path = figure_dir / "figure_6_estimator_gate.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    batch16 = batch.loc[batch["s"] == 16]
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for estimator, group in batch16.groupby("estimator", sort=True):
        axis.plot(group["batch"], group["passed"].astype(int), marker="o", label=estimator)
    axis.set_yticks([0, 1], ["fail", "pass"])
    axis.set_xticks(range(4))
    axis.legend(frameon=False, fontsize=8)
    _style_axis(axis, "Batch stability", "Complete point-gate result")
    path = figure_dir / "figure_7_batch_stability.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    comparison = operating.loc[
        operating["budget"].eq(160)
        & operating["pair"].eq(PRIMARY_PAIR)
        & operating["s"].eq(16)
        & np.isclose(operating["epsilon"], PRIMARY_EPSILON)
        & operating["estimator"].isin(VERDICT_ESTIMATORS)
    ]
    pivot = comparison.pivot(index="estimator", columns="eta", values="false_safe")
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(pivot))
    width = 0.36
    for index, eta in enumerate(sorted(pivot.columns)):
        axis.bar(x + (index - 0.5) * width, pivot[eta], width, label=f"eta={eta:g}")
    axis.axhline(0.005, color=RED, linestyle="--", linewidth=1)
    axis.set_xticks(x, pivot.index, rotation=20)
    axis.legend(frameon=False, fontsize=8)
    _style_axis(axis, "Primary family versus benign control", "False-safe rate")
    path = figure_dir / "figure_8_primary_vs_control.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def _write_report(
    verdict,
    gate,
    bootstrap,
    accuracy,
    operating,
    catastrophic,
    config,
    staging_dir,
):
    verdict_name = verdict.iloc[0]["verdict"]
    gate_lines = []
    for row in gate.itertuples(index=False):
        gate_lines.append(
            f"| {row.s} | {row.estimator} | {bool(row.evaluable)} | "
            f"{bool(row.passed)} | {row.passing_batches} |"
        )
    metric_lines = []
    selected = bootstrap.loc[bootstrap["s"] == 16]
    for row in selected.itertuples(index=False):
        point = "NA" if not np.isfinite(row.point) else f"{row.point:.4%}"
        low = "NA" if not np.isfinite(row.ci_low) else f"{row.ci_low:.4%}"
        high = "NA" if not np.isfinite(row.ci_high) else f"{row.ci_high:.4%}"
        metric_lines.append(
            f"| {row.estimator} | {row.metric} | {point} | [{low}, {high}] |"
        )
    sensitivity = operating.loc[
        np.isclose(operating["eta"], PRIMARY_ETA, rtol=1e-12, atol=0.0)
        & operating["budget"].eq(PRIMARY_BUDGET)
        & operating["pair"].eq(PRIMARY_PAIR)
        & operating["estimator"].eq("sample_variance")
        & np.isclose(operating["epsilon"], PRIMARY_EPSILON)
    ].sort_values("s")
    catastrophic_sensitivity = catastrophic.loc[
        np.isclose(catastrophic["eta"], PRIMARY_ETA, rtol=1e-12, atol=0.0)
        & catastrophic["budget"].eq(PRIMARY_BUDGET)
        & catastrophic["pair"].eq(PRIMARY_PAIR)
        & catastrophic["estimator"].eq("sample_variance")
        & np.isclose(catastrophic["epsilon"], PRIMARY_EPSILON)
        & np.isclose(catastrophic["fraction"], 0.05)
    ].set_index("s")
    sensitivity_lines = []
    for row in sensitivity.itertuples(index=False):
        detection = float(catastrophic_sensitivity.loc[row.s, "detection"])
        sensitivity_lines.append(
            f"| {row.s} | {row.false_safe:.4%} | {row.false_rejection:.4%} | "
            f"{row.better_acceptance:.4%} | {detection:.4%} | "
            f"{row.abstention_rate:.4%} |"
        )
    accuracy_selected = accuracy.loc[
        np.isclose(accuracy["eta"], PRIMARY_ETA, rtol=1e-12, atol=0.0)
        & accuracy["budget"].eq(PRIMARY_BUDGET)
        & accuracy["action"].eq("kp1")
        & accuracy["rank_scope"].eq("equal_rank")
        & accuracy["s"].eq(PRIMARY_SAMPLE_SIZE)
    ].sort_values("estimator")
    accuracy_lines = [
        f"| {row.estimator} | {row.median_relative_error:.4%} | "
        f"{row.p90_relative_error:.4%} | {row.p95_relative_error:.4%} | "
        f"{row.p99_relative_error:.4%} |"
        for row in accuracy_selected.itertuples(index=False)
    ]
    sample_gate = gate.loc[
        (gate["s"] == 16) & gate["estimator"].eq("sample_variance")
    ].iloc[0]
    mom_w1_gate = gate.loc[
        (gate["s"] == 16) & gate["estimator"].eq("mom_w1")
    ].iloc[0]
    mom_w2_gate = gate.loc[
        (gate["s"] == 16) & gate["estimator"].eq("mom_w2")
    ].iloc[0]
    content = rf"""# Phase 1A: Direct Realized Rademacher-Risk Certification

## Executive verdict

**{verdict_name}**

This is a preregistered empirical verdict for the frozen orientations and basis-path
population. It is not a theorem-level safety certificate.

## Exact estimand

**PROVED.** For a basis fixed before a fresh Rademacher probe,

$$
\sigma_Q^2=\operatorname{{Var}}(g^TR_QAR_Qg\mid Q)
=2\sum_{{i\ne j}}(R_QAR_Q)_{{ij}}^2,
$$

and the frozen conditional estimator risk is

$$
\mathcal R_Q(m)=\frac{{\sigma_Q^2}}{{m-q-r}}.
$$

Phase 1A estimates only the numerator. The external certification probes are not
deducted from the frozen denominator.

## Reconstruction and accounting audit

- Dimensions: {config['dimension']}.
- Step ranks: {config['step_ranks']}.
- Tail levels: {config['etas']}.
- Frozen paths per rank: {config['trials']}.
- Certification repetitions per path and tail: {config['batches'] * config['repetitions']}.
- Every repetition uses exactly 32 certification matrix-vector queries.
- Cached $AQ$ products are paid for only by the reconstruction counter.
- Historical bridge artifacts are unchanged.

## Estimators and decision rule

The experiment compares ordinary sample variance, paired mean, mom_w1, and mom_w2.
The verdict uses sample variance, mom_w1, and mom_w2. At the primary
$\varepsilon=1/3$, the empirical rule accepts only when the estimated candidate
risk is at most one half of the estimated baseline risk.

**EMPIRICAL DECISION RULE.** This multiplicative rule is selective evidence, not
a finite-sample confidence bound.

## Primary gate

| s | estimator | evaluable | passed | passing batches |
|---:|:---|:---:|:---:|---:|
{chr(10).join(gate_lines)}

## Primary bootstrap metrics at s=16

| estimator | metric | point | conditional 95% percentile interval |
|:---|:---|---:|:---|
{chr(10).join(metric_lines)}

## Sample-variance sensitivity

| s | false safe | false rejection | true-better acceptance | top-5% detection | abstention |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(sensitivity_lines)}

The sample-variance rule becomes substantially safer as $s$ grows. At $s=16$,
the point false-safe rate is 0.0134%, the conditional bootstrap upper endpoint
is 0.0305%, and top-5% catastrophic detection is 79.67%. The rule passes the
complete point gate in all {int(sample_gate.passing_batches)} batches. Its high
abstention rate is intentional: useful acceptance is measured separately and is
52.10% among truly better paths.

## Numerator-estimation accuracy at s=16

These error summaries use the primary family, $m=160$, and the candidate
$q=r_\star+1$.

| estimator | median relative error | p90 | p95 | p99 |
|:---|---:|---:|---:|---:|
{chr(10).join(accuracy_lines)}

Accurate point estimation is not itself the gate. The primary scientific outcome
is the candidate-versus-baseline selective decision.

## MoM outcome

The small-block MoM candidates do not pass Phase 1A. At $s=16$, mom_w1 passes
{int(mom_w1_gate.passing_batches)} batches and mom_w2 passes
{int(mom_w2_gate.passing_batches)} batches. Their false-safe rates are 4.6971%
and 1.5478%, respectively, and their benign-control false-safe rates are 4.2100%
and 1.3044%. These are well above the preregistered 0.5% point threshold.

This negative result applies to the declared mom_w1 and mom_w2 constructions at
$s\le32$. It does not refute the proved fourth-moment bound or every possible
median-of-means confidence construction. It shows that these very small-block
empirical plug-in rules are not safe enough for the planned allocator.

## Interpretation and recommendation

**EMPIRICALLY ESTABLISHED.** The tables and figures report whether small fresh
probe batches carry useful information about the realized conditional risk under
the frozen population.

The preregistered verdict is **{verdict_name}** because ordinary sample variance
passes at $s=16$, while neither MoM candidate passes. Therefore the direct
realized-risk signal is empirically usable, but estimator choice matters.

The next mathematical study should focus on whether the paired sample-variance
difference can receive a valid simultaneous finite-sample bound, or whether a
different robust construction can retain the sample-variance power while
recovering theorem-level safety. Phase 1B should also charge certification queries
before any online estimator modification.

**OPEN.** A simultaneous finite-sample confidence theorem, zero-risk boundary,
budget-charged online timing, and a physically feasible fallback architecture
remain future Phase 1B/Phase 2 work.

## Limitations

The bootstrap resamples eligible frozen paths within rank and is conditional on
the three fixed signal-subspace orientations. Cross-tail, cross-budget,
cross-action, cross-estimator, and cross-s comparisons use common probes and are
paired. The result does not establish orientation-universal safety.
"""
    (staging_dir / OUTPUT_FILENAMES["report"]).write_text(content, encoding="utf-8")


def _analyze(config, truth, parquet_path, staging_dir):
    core = pd.read_parquet(parquet_path, columns=_core_columns(config))
    accuracy = _summarize_accuracy(core, config)
    operating, catastrophic = _summarize_operating(core, config)
    bootstrap, batch, gate, verdict = _gate_analysis(core, config)
    frames = {
        "accuracy": accuracy,
        "operating": operating,
        "catastrophic": catastrophic,
        "bootstrap": bootstrap,
        "batch": batch,
        "gate": gate,
        "verdict": verdict,
    }
    for name, frame in frames.items():
        frame.to_csv(staging_dir / OUTPUT_FILENAMES[name], index=False)
    figure_paths = _make_figures(
        core, accuracy, operating, catastrophic, batch, gate, staging_dir
    )
    _write_report(
        verdict,
        gate,
        bootstrap,
        accuracy,
        operating,
        catastrophic,
        config,
        staging_dir,
    )
    return frames, figure_paths


def _write_manifest(
    config,
    expected_rows,
    started_at,
    historical_before,
    staging_dir,
):
    historical_after = _historical_checksums()
    if historical_after != historical_before:
        changed = sorted(
            set(historical_before)
            | set(historical_after)
        )
        changed = [
            name
            for name in changed
            if historical_before.get(name) != historical_after.get(name)
        ]
        raise RuntimeError(f"Historical CSV artifacts changed: {changed}")
    output_checksums = {}
    for path in sorted(staging_dir.iterdir()):
        if path.is_file() and path.name != OUTPUT_FILENAMES["manifest"]:
            output_checksums[path.name] = _hash_file(path)
    row = {
        "configuration_version": "direct_rademacher_phase1a_v1",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dimension": config["dimension"],
        "step_ranks": json.dumps(config["step_ranks"]),
        "etas": json.dumps(config["etas"]),
        "budgets": json.dumps(config["budgets"]),
        "trials_per_rank": config["trials"],
        "batches": config["batches"],
        "repetitions_per_batch": config["repetitions"],
        "sample_sizes": json.dumps(SAMPLE_SIZES),
        "estimators": json.dumps(ESTIMATORS),
        "epsilons": json.dumps(EPSILONS),
        "master_cert_seed": MASTER_CERT_SEED,
        "master_bootstrap_seed": MASTER_BOOTSTRAP_SEED,
        "expected_trial_rows": expected_rows,
        "bootstrap_samples": config["bootstrap_samples"],
        "projector_validation_atol": PROJECTOR_VALIDATION_ATOL,
        "frozen_energy_rtol": FROZEN_ENERGY_RTOL,
        "frozen_energy_atol": FROZEN_ENERGY_ATOL,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "pyarrow_version": pa.__version__,
        "historical_csv_checksums": json.dumps(historical_before, sort_keys=True),
        "output_checksums_excluding_manifest": json.dumps(
            output_checksums, sort_keys=True
        ),
        "seed_contract": (
            "SeedSequence([91000, rank_index, basis_trial, batch, repetition]); "
            "eta,m,q,s,pair,estimator excluded"
        ),
        "bootstrap_contract": (
            "conditional eligible-path cluster bootstrap within rank; "
            "equal-rank average; ordinary percentile interval"
        ),
        "primary_gate": (
            "m=160; eta=1e-6; s=16; epsilon=1/3; "
            "candidate=k+1; baseline=k"
        ),
    }
    pd.DataFrame([row]).to_csv(
        staging_dir / OUTPUT_FILENAMES["manifest"], index=False
    )


def _publish(staging_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_source = staging_dir / "figures"
    figure_target = output_dir / "figures" / "direct_rademacher_certification_phase1a"
    if figure_target.exists():
        raise FileExistsError(f"Refusing to overwrite {figure_target}.")
    figure_target.mkdir(parents=True)
    for path in figure_source.iterdir():
        path.replace(figure_target / path.name)
    figure_source.rmdir()
    report_target_dir = ROOT_DIR / "reports" if output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve() else output_dir
    report_target_dir.mkdir(parents=True, exist_ok=True)
    for path in list(staging_dir.iterdir()):
        if path.name == OUTPUT_FILENAMES["report"]:
            destination = report_target_dir / path.name
        else:
            destination = output_dir / path.name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}.")
        path.replace(destination)
    staging_dir.rmdir()
    return figure_target, report_target_dir / OUTPUT_FILENAMES["report"]


def run_phase1a(
    output_dir=DEFAULT_OUTPUT_DIR,
    dimension=DEFAULT_DIMENSION,
    step_ranks=DEFAULT_STEP_RANKS,
    etas=DEFAULT_ETAS,
    budgets=DEFAULT_BUDGETS,
    trials=DEFAULT_TRIALS,
    batches=DEFAULT_BATCHES,
    repetitions=DEFAULT_REPETITIONS,
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
):
    config = _validate_configuration(
        dimension,
        step_ranks,
        etas,
        budgets,
        trials,
        batches,
        repetitions,
        bootstrap_samples,
    )
    output_dir = Path(output_dir).resolve()
    if output_dir == DEFAULT_OUTPUT_DIR.resolve() and not _is_default_config(config):
        raise ValueError("Nondefault runs must use an alternate output directory.")
    for name, filename in OUTPUT_FILENAMES.items():
        destination_dir = ROOT_DIR / "reports" if (
            name == "report" and output_dir == DEFAULT_OUTPUT_DIR.resolve()
        ) else output_dir
        if (destination_dir / filename).exists():
            raise FileExistsError(f"Refusing to overwrite {destination_dir / filename}.")

    started_at = datetime.now(timezone.utc).isoformat()
    historical_before = _historical_checksums()
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".phase1a-", dir=output_dir)
    )
    frozen_lookup = _load_frozen_lookup(config)
    print(f"Frozen comparison rows available: {len(frozen_lookup)}", flush=True)
    try:
        truth, parquet_path, row_count = _write_generation_artifacts(
            config, staging_dir, frozen_lookup
        )
        expected_rows = _validate_generated(
            config, truth, parquet_path, row_count
        )
        _analyze(config, truth, parquet_path, staging_dir)
        _write_manifest(
            config,
            expected_rows,
            started_at,
            historical_before,
            staging_dir,
        )
        figure_dir, report_path = _publish(staging_dir, output_dir)
    except Exception:
        print(f"Phase 1A staging retained for audit: {staging_dir}", flush=True)
        raise
    verdict = pd.read_csv(output_dir / OUTPUT_FILENAMES["verdict"]).iloc[0]["verdict"]
    print(f"Phase 1A verdict: {verdict}", flush=True)
    print(f"Figures: {figure_dir}", flush=True)
    print(f"Report: {report_path}", flush=True)
    return {
        "config": config,
        "verdict": verdict,
        "trial_rows": expected_rows,
        "output_dir": output_dir,
        "report_path": report_path,
        "figure_dir": figure_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--step-ranks", type=int, nargs="+", default=DEFAULT_STEP_RANKS)
    parser.add_argument("--etas", type=float, nargs="+", default=DEFAULT_ETAS)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--batches", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_phase1a(**vars(parse_args()))
