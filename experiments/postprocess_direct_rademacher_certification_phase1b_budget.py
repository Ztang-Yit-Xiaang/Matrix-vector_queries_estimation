"""Budget-aware Phase 1B-A postprocessing of frozen Phase 1A certification data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
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
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from direct_rademacher_certification_phase1a import (  # noqa: E402
    ACTION_PAIRS,
    DECISION_ABSTAIN,
    DECISION_ACCEPT,
    DECISION_REJECT,
    EPSILONS,
    ESTIMATORS,
    SAMPLE_SIZES,
    epsilon_tag,
)


DEFAULT_RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_PHASE1A_TRIALS = (
    DEFAULT_RESULTS_DIR / "direct_rademacher_certification_phase1a_trials.parquet"
)
DEFAULT_PHASE1A_TRUTH = (
    DEFAULT_RESULTS_DIR / "direct_rademacher_certification_phase1a_truth.csv"
)
DEFAULT_PHASE1A_MANIFEST = (
    DEFAULT_RESULTS_DIR / "direct_rademacher_certification_phase1a_manifest.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 93_000
DEFAULT_STEP_RANKS = (5, 15, 30)
DEFAULT_ETAS = (1e-10, 1e-6)
DEFAULT_BUDGETS = (80, 160, 240)
DEFAULT_PATHS_PER_RANK = 200
DEFAULT_REPETITIONS_PER_PATH = 200
PRIMARY_BUDGET = 160
PRIMARY_ETA = 1e-6
PRIMARY_S = 16
PRIMARY_EPSILON = 1.0 / 3.0
PRIMARY_PAIR = "primary"
PRIMARY_ESTIMATOR = "sample_variance"
EXPECTED_SOURCE_ROWS = 960_000
EXPECTED_PATH_ROWS = 691_200
TOLERANCE_FACTOR = 128.0 * np.finfo(np.float64).eps

OUTPUT_PREFIX = "direct_rademacher_certification_phase1b_budget"
OUTPUTS = {
    "manifest": f"{OUTPUT_PREFIX}_manifest.csv",
    "paths": f"{OUTPUT_PREFIX}_paths.parquet",
    "summary": f"{OUTPUT_PREFIX}_summary.csv",
    "bootstrap": f"{OUTPUT_PREFIX}_bootstrap.csv",
    "cost": f"{OUTPUT_PREFIX}_cost_decomposition.csv",
    "decisions": f"{OUTPUT_PREFIX}_decision_probabilities.csv",
    "catastrophic": f"{OUTPUT_PREFIX}_catastrophic.csv",
    "sensitivity": f"{OUTPUT_PREFIX}_sample_size_sensitivity.csv",
    "verdict": f"{OUTPUT_PREFIX}_verdict.csv",
    "report": "direct_rademacher_risk_certification_phase1b_budget.md",
}

ACTION_LABELS = ("km1", "k", "kp1", "kp2")
PAIR_LABELS = tuple(ACTION_PAIRS)
INK = "#25364A"
BLUE = "#3568A8"
GOLD = "#D39B2A"
RED = "#A94B45"
GREEN = "#4E7D62"
GREY = "#8C96A3"


@dataclass(frozen=True)
class PaidAccounting:
    """Exact query accounting once both nested actions and certification are paid."""

    candidate_cost: int
    baseline_cost: int
    committed_cost: int
    original_ell: int
    paid_ell: int

    @property
    def extra_cost(self):
        return self.committed_cost - self.baseline_cost

    @property
    def cost_multiplier(self):
        return self.original_ell / self.paid_ell

    @property
    def net_numerator_threshold(self):
        return self.paid_ell / self.original_ell


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nested_reuse_errors(q_low, aq_low, q_high, aq_high):
    """Return projector and cached-image errors for Lemma 15.1."""
    q_low = np.asarray(q_low, dtype=float)
    aq_low = np.asarray(aq_low, dtype=float)
    q_high = np.asarray(q_high, dtype=float)
    aq_high = np.asarray(aq_high, dtype=float)
    if q_low.ndim != 2 or q_high.ndim != 2:
        raise ValueError("Bases must be matrices.")
    if q_low.shape[0] != q_high.shape[0]:
        raise ValueError("Bases must share the ambient dimension.")
    if aq_low.shape != q_low.shape or aq_high.shape != q_high.shape:
        raise ValueError("Each cached image must match its basis shape.")
    projector_residual = q_low - q_high @ (q_high.T @ q_low)
    reconstructed_aq = aq_high @ (q_high.T @ q_low)
    return {
        "projector_error": float(np.linalg.norm(projector_residual, ord=2)),
        "cached_aq_error": float(np.linalg.norm(aq_low - reconstructed_aq, ord=2)),
    }


def paid_accounting(
    budget,
    candidate_q,
    candidate_r,
    baseline_q,
    baseline_r,
    sample_size,
    *,
    nested=True,
):
    """Apply Theorem 16 accounting to two available actions."""
    values = (
        budget,
        candidate_q,
        candidate_r,
        baseline_q,
        baseline_r,
        sample_size,
    )
    if any(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError("Query counts must be integers, not booleans.")
    if any(int(value) != value for value in values):
        raise ValueError("Query counts must be integers.")
    budget, candidate_q, candidate_r, baseline_q, baseline_r, sample_size = (
        int(value) for value in values
    )
    if not nested:
        raise ValueError("Maximum-cost reuse requires nested actions.")
    if min(candidate_q, candidate_r, baseline_q, baseline_r) < 0:
        raise ValueError("Construction counts must be nonnegative.")
    if sample_size <= 0:
        raise ValueError("Certification sample size must be positive.")
    if candidate_r > candidate_q or baseline_r > baseline_q:
        raise ValueError("Accepted rank cannot exceed attempted sketch queries.")
    candidate_cost = candidate_q + candidate_r
    baseline_cost = baseline_q + baseline_r
    committed_cost = max(candidate_cost, baseline_cost)
    original_ell = budget - baseline_cost
    paid_ell = budget - committed_cost - sample_size
    if original_ell <= 0:
        raise ValueError("The original baseline has no residual capacity.")
    if paid_ell <= 0:
        raise ValueError("The paid procedure has no residual capacity.")
    return PaidAccounting(
        candidate_cost=candidate_cost,
        baseline_cost=baseline_cost,
        committed_cost=committed_cost,
        original_ell=original_ell,
        paid_ell=paid_ell,
    )


def paid_risk_decomposition(candidate_sigma2, baseline_sigma2, accounting):
    """Return Theorems 16--17 risk quantities for scalar or array numerators."""
    candidate = np.asarray(candidate_sigma2, dtype=float)
    baseline = np.asarray(baseline_sigma2, dtype=float)
    if candidate.shape != baseline.shape:
        raise ValueError("Candidate and baseline numerators must have matching shapes.")
    if np.any(~np.isfinite(candidate)) or np.any(~np.isfinite(baseline)):
        raise ValueError("Numerators must be finite.")
    if np.any(candidate < 0.0) or np.any(baseline < 0.0):
        raise ValueError("Numerators must be nonnegative.")
    original = baseline / accounting.original_ell
    paid_baseline = baseline / accounting.paid_ell
    paid_candidate = candidate / accounting.paid_ell
    paid_oracle = np.minimum(candidate, baseline) / accounting.paid_ell
    return {
        "original_baseline": original,
        "paid_baseline": paid_baseline,
        "paid_candidate": paid_candidate,
        "paid_oracle": paid_oracle,
    }


def budget_decision_array(candidate_hat, baseline_hat, epsilon):
    """Apply the frozen guard to variance numerators under a common denominator."""
    candidate = np.asarray(candidate_hat, dtype=float)
    baseline = np.asarray(baseline_hat, dtype=float)
    if candidate.shape != baseline.shape:
        raise ValueError("Estimated numerator arrays must have matching shapes.")
    if np.any(~np.isfinite(candidate)) or np.any(~np.isfinite(baseline)):
        raise ValueError("Estimated numerators must be finite.")
    if np.any(candidate < 0.0) or np.any(baseline < 0.0):
        raise ValueError("Estimated numerators must be nonnegative.")
    epsilon = float(epsilon)
    if epsilon not in EPSILONS:
        raise ValueError("epsilon is outside the frozen grid.")
    rho = (1.0 - epsilon) / (1.0 + epsilon)
    decisions = np.full(candidate.shape, DECISION_ABSTAIN, dtype=np.int8)
    unequal = candidate != baseline
    decisions[unequal & (candidate <= rho * baseline)] = DECISION_ACCEPT
    decisions[unequal & (baseline <= rho * candidate)] = DECISION_REJECT
    return decisions


def _truth_array(candidate, baseline):
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    tolerance = TOLERANCE_FACTOR * np.maximum(candidate, baseline)
    difference = candidate - baseline
    labels = np.zeros(candidate.shape, dtype=np.int8)
    labels[difference < -tolerance] = -1
    labels[difference > tolerance] = 1
    return labels


def _required_source_columns():
    columns = [
        "rank_index",
        "step_rank",
        "basis_trial",
        "eta",
        "batch",
        "repetition",
        "s",
        "orientation_seed",
        "basis_seed",
    ]
    for action in ACTION_LABELS:
        columns.extend(
            [
                f"q__{action}",
                f"r_actual__{action}",
                f"reconstruction_queries__{action}",
                f"sigma2_exact__{action}",
            ]
        )
        for estimator in ESTIMATORS:
            columns.append(f"sigma2_hat__{action}__{estimator}")
    return columns


def _validate_source_files(trials_path, truth_path, manifest_path):
    for path in (trials_path, truth_path, manifest_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 1:
        raise RuntimeError("Phase 1A manifest must have exactly one row.")
    checksums = json.loads(manifest.loc[0, "output_checksums_excluding_manifest"])
    expected_trials = checksums.get(Path(trials_path).name)
    expected_truth = checksums.get(Path(truth_path).name)
    if expected_trials is None or expected_truth is None:
        raise RuntimeError("Phase 1A manifest is missing frozen source checksums.")
    if _hash_file(trials_path) != expected_trials:
        raise RuntimeError("Phase 1A trial checksum mismatch.")
    if _hash_file(truth_path) != expected_truth:
        raise RuntimeError("Phase 1A truth checksum mismatch.")
    parquet_file = pq.ParquetFile(trials_path)
    if parquet_file.metadata.num_rows != EXPECTED_SOURCE_ROWS:
        raise RuntimeError("Phase 1A trial row count is not the frozen 960,000 rows.")
    missing = sorted(set(_required_source_columns()) - set(parquet_file.schema.names))
    if missing:
        raise RuntimeError(f"Phase 1A trial artifact is missing columns: {missing}")
    truth = pd.read_csv(truth_path)
    if len(truth) != 14_400:
        raise RuntimeError("Phase 1A truth table must contain 14,400 rows.")
    return manifest.iloc[0].to_dict(), {
        Path(trials_path).name: expected_trials,
        Path(truth_path).name: expected_truth,
        Path(manifest_path).name: _hash_file(manifest_path),
    }


def _load_core(trials_path, max_paths_per_rank=None):
    core = pd.read_parquet(trials_path, columns=_required_source_columns())
    if max_paths_per_rank is not None:
        max_paths_per_rank = int(max_paths_per_rank)
        if max_paths_per_rank <= 0:
            raise ValueError("max_paths_per_rank must be positive.")
        core = core.loc[core["basis_trial"] < max_paths_per_rank].copy()
    expected_repetitions = DEFAULT_REPETITIONS_PER_PATH
    counts = core.groupby(
        ["step_rank", "basis_trial", "eta", "s"], sort=False
    ).size()
    if not np.all(counts.to_numpy() == expected_repetitions):
        raise RuntimeError("Every retained path/eta/s group must contain 200 repetitions.")
    return core


def _group_mean(codes, group_count, values):
    sums = np.bincount(codes, weights=np.asarray(values, dtype=float), minlength=group_count)
    counts = np.bincount(codes, minlength=group_count)
    if np.any(counts == 0):
        raise RuntimeError("Path aggregation produced an empty group.")
    return sums / counts


def _catastrophic_sets(base_paths, budget):
    risks = base_paths["sigma2_exact__k"].to_numpy(dtype=float) / (
        int(budget)
        - base_paths["q__k"].to_numpy(dtype=int)
        - base_paths["r_actual__k"].to_numpy(dtype=int)
    )
    marked = base_paths[["step_rank", "basis_trial"]].copy()
    marked["risk"] = risks
    result = {0.01: set(), 0.05: set(), 0.10: set()}
    for rank, group in marked.groupby("step_rank", sort=True):
        for fraction in result:
            count = max(1, int(round(fraction * len(group))))
            selected = group.nlargest(count, "risk")
            result[fraction].update(
                (int(rank), int(trial)) for trial in selected["basis_trial"]
            )
    return result


def build_path_artifact(core, budgets=DEFAULT_BUDGETS):
    """Collapse 200 certification repetitions into 691,200 path/config rows."""
    frames = []
    for (eta, sample_size), subset in core.groupby(["eta", "s"], sort=True):
        subset = subset.sort_values(
            ["step_rank", "basis_trial", "batch", "repetition"]
        ).reset_index(drop=True)
        path_keys = pd.MultiIndex.from_frame(subset[["step_rank", "basis_trial"]])
        codes, uniques = pd.factorize(path_keys, sort=True)
        group_count = len(uniques)
        first_indices = np.flatnonzero(
            np.r_[True, codes[1:] != codes[:-1]]
        )
        if len(first_indices) != group_count:
            raise RuntimeError("Path factorization is not contiguous.")
        base_paths = subset.iloc[first_indices].reset_index(drop=True)
        for budget in budgets:
            catastrophic = _catastrophic_sets(base_paths, budget)
            for pair, (candidate_action, baseline_action) in ACTION_PAIRS.items():
                q_candidate = base_paths[f"q__{candidate_action}"].to_numpy(dtype=int)
                r_candidate = base_paths[
                    f"r_actual__{candidate_action}"
                ].to_numpy(dtype=int)
                q_baseline = base_paths[f"q__{baseline_action}"].to_numpy(dtype=int)
                r_baseline = base_paths[
                    f"r_actual__{baseline_action}"
                ].to_numpy(dtype=int)
                candidate_cost = q_candidate + r_candidate
                baseline_cost = q_baseline + r_baseline
                committed_cost = np.maximum(candidate_cost, baseline_cost)
                original_ell = int(budget) - baseline_cost
                paid_ell = int(budget) - committed_cost - int(sample_size)
                if np.any(original_ell <= 0):
                    raise RuntimeError("A frozen baseline has no residual capacity.")
                feasible = paid_ell > 0
                sigma_candidate_path = base_paths[
                    f"sigma2_exact__{candidate_action}"
                ].to_numpy(dtype=float)
                sigma_baseline_path = base_paths[
                    f"sigma2_exact__{baseline_action}"
                ].to_numpy(dtype=float)
                paid_baseline = np.full(group_count, np.nan)
                paid_candidate = np.full(group_count, np.nan)
                paid_oracle = np.full(group_count, np.nan)
                paid_baseline[feasible] = (
                    sigma_baseline_path[feasible] / paid_ell[feasible]
                )
                paid_candidate[feasible] = (
                    sigma_candidate_path[feasible] / paid_ell[feasible]
                )
                paid_oracle[feasible] = (
                    np.minimum(
                        sigma_candidate_path[feasible],
                        sigma_baseline_path[feasible],
                    )
                    / paid_ell[feasible]
                )
                exact = {
                    "original": sigma_baseline_path / original_ell,
                    "paid_baseline": paid_baseline,
                    "paid_candidate": paid_candidate,
                    "paid_oracle": paid_oracle,
                }
                paid_order = _truth_array(sigma_candidate_path, sigma_baseline_path)
                net_order = np.zeros(group_count, dtype=np.int8)
                net_order[feasible] = _truth_array(
                    exact["paid_candidate"][feasible],
                    exact["original"][feasible],
                )
                path_pairs = list(
                    zip(base_paths["step_rank"], base_paths["basis_trial"])
                )
                catastrophic_flags = {
                    fraction: np.array(
                        [key in keys for key in path_pairs], dtype=bool
                    )
                    for fraction, keys in catastrophic.items()
                }
                sigma_candidate_reps = subset[
                    f"sigma2_exact__{candidate_action}"
                ].to_numpy(dtype=float)
                sigma_baseline_reps = subset[
                    f"sigma2_exact__{baseline_action}"
                ].to_numpy(dtype=float)
                for estimator in ESTIMATORS:
                    candidate_hat = subset[
                        f"sigma2_hat__{candidate_action}__{estimator}"
                    ].to_numpy(dtype=float)
                    baseline_hat = subset[
                        f"sigma2_hat__{baseline_action}__{estimator}"
                    ].to_numpy(dtype=float)
                    for epsilon in EPSILONS:
                        decisions = budget_decision_array(
                            candidate_hat, baseline_hat, epsilon
                        )
                        accepted = decisions == DECISION_ACCEPT
                        rejected = decisions == DECISION_REJECT
                        abstained = decisions == DECISION_ABSTAIN
                        selected_sigma = np.where(
                            accepted,
                            sigma_candidate_reps,
                            sigma_baseline_reps,
                        )
                        paid_ell_reps = paid_ell[codes]
                        feasible_reps = feasible[codes]
                        selected_risk = np.full(len(selected_sigma), np.nan)
                        selected_risk[feasible_reps] = (
                            selected_sigma[feasible_reps]
                            / paid_ell_reps[feasible_reps]
                        )
                        paid_order_reps = paid_order[codes]
                        net_order_reps = net_order[codes]
                        cost_multiplier = np.full(group_count, np.nan)
                        net_numerator_threshold = np.full(group_count, np.nan)
                        cost_multiplier[feasible] = (
                            original_ell[feasible] / paid_ell[feasible]
                        )
                        net_numerator_threshold[feasible] = (
                            paid_ell[feasible] / original_ell[feasible]
                        )
                        frame = pd.DataFrame(
                            {
                                "rank_index": base_paths["rank_index"].to_numpy(),
                                "step_rank": base_paths["step_rank"].to_numpy(),
                                "basis_trial": base_paths["basis_trial"].to_numpy(),
                                "eta": float(eta),
                                "budget": int(budget),
                                "s": int(sample_size),
                                "pair": pair,
                                "estimator": estimator,
                                "epsilon": float(epsilon),
                                "orientation_seed": base_paths[
                                    "orientation_seed"
                                ].to_numpy(),
                                "basis_seed": base_paths["basis_seed"].to_numpy(),
                                "candidate_action": candidate_action,
                                "baseline_action": baseline_action,
                                "q_candidate": q_candidate,
                                "r_candidate": r_candidate,
                                "q_baseline": q_baseline,
                                "r_baseline": r_baseline,
                                "candidate_cost": candidate_cost,
                                "baseline_cost": baseline_cost,
                                "committed_cost": committed_cost,
                                "extra_construction_cost": (
                                    committed_cost - baseline_cost
                                ),
                                "ell_original": original_ell,
                                "ell_paid": paid_ell,
                                "accounting_feasible": feasible,
                                "cost_multiplier": cost_multiplier,
                                "net_numerator_threshold": net_numerator_threshold,
                                "sigma2_candidate": sigma_candidate_path,
                                "sigma2_baseline": sigma_baseline_path,
                                "risk_original_baseline": exact["original"],
                                "risk_paid_baseline": exact["paid_baseline"],
                                "risk_paid_candidate": exact["paid_candidate"],
                                "risk_paid_oracle": exact["paid_oracle"],
                                "risk_paid_selected_mean": np.where(
                                    feasible,
                                    _group_mean(
                                        codes,
                                        group_count,
                                        np.nan_to_num(selected_risk, nan=0.0),
                                    ),
                                    np.nan,
                                ),
                                "accept_probability": _group_mean(
                                    codes, group_count, accepted
                                ),
                                "reject_probability": _group_mean(
                                    codes, group_count, rejected
                                ),
                                "abstain_probability": _group_mean(
                                    codes, group_count, abstained
                                ),
                                "paid_order_truth": paid_order,
                                "net_benefit_truth": net_order,
                                "correct_paid_order_probability": _group_mean(
                                    codes,
                                    group_count,
                                    (
                                        ((paid_order_reps < 0) & accepted)
                                        | ((paid_order_reps > 0) & rejected)
                                    ),
                                ),
                                "false_candidate_probability": _group_mean(
                                    codes,
                                    group_count,
                                    (paid_order_reps > 0) & accepted,
                                ),
                                "missed_paid_candidate_probability": _group_mean(
                                    codes,
                                    group_count,
                                    (paid_order_reps < 0) & ~accepted,
                                ),
                                "net_beneficial_accept_probability": _group_mean(
                                    codes,
                                    group_count,
                                    (net_order_reps < 0) & accepted,
                                ),
                                "net_harmful_accept_probability": _group_mean(
                                    codes,
                                    group_count,
                                    (net_order_reps > 0) & accepted,
                                ),
                                "catastrophic_top1": catastrophic_flags[0.01],
                                "catastrophic_top5": catastrophic_flags[0.05],
                                "catastrophic_top10": catastrophic_flags[0.10],
                            }
                        )
                        positive = (
                            frame["risk_original_baseline"].to_numpy() > 0.0
                        ) & feasible
                        ratio = np.full(group_count, np.nan)
                        ratio[positive] = (
                            frame.loc[positive, "risk_paid_selected_mean"].to_numpy()
                            / frame.loc[positive, "risk_original_baseline"].to_numpy()
                        )
                        frame["pathwise_net_ratio"] = ratio
                        frame["path_harmed"] = feasible & (
                            frame["risk_paid_selected_mean"]
                            > frame["risk_original_baseline"]
                        )
                        required = [
                            "risk_original_baseline",
                            "risk_paid_baseline",
                            "risk_paid_candidate",
                            "risk_paid_oracle",
                            "risk_paid_selected_mean",
                            "cost_multiplier",
                            "net_numerator_threshold",
                        ]
                        numeric_finite = np.isfinite(frame[required]).all(axis=1)
                        frame["finite_when_feasible"] = (~feasible) | numeric_finite
                        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    expected = (
        core["step_rank"].nunique()
        * core.groupby("step_rank")["basis_trial"].nunique().min()
        * core["eta"].nunique()
        * len(tuple(budgets))
        * core["s"].nunique()
        * len(ACTION_PAIRS)
        * len(ESTIMATORS)
        * len(EPSILONS)
    )
    if len(result) != expected:
        raise RuntimeError(f"Path artifact has {len(result)} rows, expected {expected}.")
    key = [
        "step_rank",
        "basis_trial",
        "eta",
        "budget",
        "s",
        "pair",
        "estimator",
        "epsilon",
    ]
    if result.duplicated(key).any():
        raise RuntimeError("Path artifact key is not unique.")
    return result


def _scope_summary(group):
    feasibility = group["accounting_feasible"].to_numpy(dtype=bool)
    if np.any(feasibility) and not np.all(feasibility):
        raise RuntimeError("A rank/configuration has mixed paid feasibility.")
    if not np.all(feasibility):
        return {
            "paths": len(group),
            "evaluable": False,
            **{
                key: np.nan
                for key in (
                    "original_baseline_mean",
                    "paid_baseline_mean",
                    "paid_candidate_mean",
                    "paid_oracle_mean",
                    "paid_selected_mean",
                    "selected_mean_ratio",
                    "paid_baseline_mean_ratio",
                    "paid_candidate_mean_ratio",
                    "paid_oracle_mean_ratio",
                    "median_pathwise_ratio",
                    "p90_pathwise_ratio",
                    "p95_pathwise_ratio",
                    "p99_pathwise_ratio",
                    "max_pathwise_ratio",
                    "fraction_paths_harmed",
                    "accept_probability",
                    "reject_probability",
                    "abstain_probability",
                    "correct_paid_order_probability",
                    "false_candidate_probability",
                    "missed_paid_candidate_probability",
                    "net_beneficial_accept_probability",
                    "net_harmful_accept_probability",
                    "mean_cost_multiplier",
                    "mean_net_numerator_threshold",
                )
            },
        }
    original_mean = float(group["risk_original_baseline"].mean())
    selected_mean = float(group["risk_paid_selected_mean"].mean())
    paid_baseline_mean = float(group["risk_paid_baseline"].mean())
    candidate_mean = float(group["risk_paid_candidate"].mean())
    oracle_mean = float(group["risk_paid_oracle"].mean())
    ratios = group["pathwise_net_ratio"].dropna().to_numpy(dtype=float)
    return {
        "paths": len(group),
        "evaluable": True,
        "original_baseline_mean": original_mean,
        "paid_baseline_mean": paid_baseline_mean,
        "paid_candidate_mean": candidate_mean,
        "paid_oracle_mean": oracle_mean,
        "paid_selected_mean": selected_mean,
        "selected_mean_ratio": selected_mean / original_mean,
        "paid_baseline_mean_ratio": paid_baseline_mean / original_mean,
        "paid_candidate_mean_ratio": candidate_mean / original_mean,
        "paid_oracle_mean_ratio": oracle_mean / original_mean,
        "median_pathwise_ratio": float(np.median(ratios)) if len(ratios) else np.nan,
        "p90_pathwise_ratio": float(np.quantile(ratios, 0.90)) if len(ratios) else np.nan,
        "p95_pathwise_ratio": float(np.quantile(ratios, 0.95)) if len(ratios) else np.nan,
        "p99_pathwise_ratio": float(np.quantile(ratios, 0.99)) if len(ratios) else np.nan,
        "max_pathwise_ratio": float(np.max(ratios)) if len(ratios) else np.nan,
        "fraction_paths_harmed": float(group["path_harmed"].mean()),
        "accept_probability": float(group["accept_probability"].mean()),
        "reject_probability": float(group["reject_probability"].mean()),
        "abstain_probability": float(group["abstain_probability"].mean()),
        "correct_paid_order_probability": float(
            group["correct_paid_order_probability"].mean()
        ),
        "false_candidate_probability": float(
            group["false_candidate_probability"].mean()
        ),
        "missed_paid_candidate_probability": float(
            group["missed_paid_candidate_probability"].mean()
        ),
        "net_beneficial_accept_probability": float(
            group["net_beneficial_accept_probability"].mean()
        ),
        "net_harmful_accept_probability": float(
            group["net_harmful_accept_probability"].mean()
        ),
        "mean_cost_multiplier": float(group["cost_multiplier"].mean()),
        "mean_net_numerator_threshold": float(
            group["net_numerator_threshold"].mean()
        ),
    }


CONFIG_COLUMNS = ["eta", "budget", "s", "pair", "estimator", "epsilon"]


def summarize_paths(paths):
    """Return per-rank and equal-rank point summaries."""
    rows = []
    for config, group in paths.groupby(CONFIG_COLUMNS, sort=True):
        rank_rows = []
        for rank, rank_group in group.groupby("step_rank", sort=True):
            metrics = _scope_summary(rank_group)
            row = dict(zip(CONFIG_COLUMNS, config))
            row.update({"rank_scope": str(int(rank)), **metrics})
            rows.append(row)
            rank_rows.append(metrics)
        equal = dict(zip(CONFIG_COLUMNS, config))
        equal["rank_scope"] = "equal_rank"
        equal["evaluable"] = bool(all(entry["evaluable"] for entry in rank_rows))
        for key in rank_rows[0]:
            if key == "evaluable":
                continue
            values = np.array([entry[key] for entry in rank_rows], dtype=float)
            equal[key] = (
                float(np.mean(values)) if equal["evaluable"] else np.nan
            )
        rows.append(equal)
    result = pd.DataFrame(rows)
    denominator = (
        result["paid_baseline_mean_ratio"] - result["paid_oracle_mean_ratio"]
    )
    result["decision_efficiency"] = np.where(
        denominator > 0.0,
        (result["paid_baseline_mean_ratio"] - result["selected_mean_ratio"])
        / denominator,
        np.nan,
    )
    return result


def _bootstrap_config(group, samples, seed):
    metric_names = (
        "selected_mean_ratio",
        "paid_oracle_mean_ratio",
        "paid_baseline_mean_ratio",
        "fraction_paths_harmed",
        "accept_probability",
        "abstain_probability",
    )
    if not bool(group["accounting_feasible"].all()):
        return {metric: (np.nan, np.nan) for metric in metric_names}
    rng = np.random.default_rng(seed)
    selected_ratios = np.empty(samples)
    oracle_ratios = np.empty(samples)
    paid_baseline_ratios = np.empty(samples)
    harmed = np.empty(samples)
    accept = np.empty(samples)
    abstain = np.empty(samples)
    rank_groups = {
        int(rank): rank_group.sort_values("basis_trial")
        for rank, rank_group in group.groupby("step_rank", sort=True)
    }
    for index in range(samples):
        selected_rank = []
        oracle_rank = []
        paid_baseline_rank = []
        harmed_rank = []
        accept_rank = []
        abstain_rank = []
        for rank_group in rank_groups.values():
            draw = rng.integers(0, len(rank_group), size=len(rank_group))
            sampled = rank_group.iloc[draw]
            original_mean = sampled["risk_original_baseline"].mean()
            selected_rank.append(sampled["risk_paid_selected_mean"].mean() / original_mean)
            oracle_rank.append(sampled["risk_paid_oracle"].mean() / original_mean)
            paid_baseline_rank.append(
                sampled["risk_paid_baseline"].mean() / original_mean
            )
            harmed_rank.append(sampled["path_harmed"].mean())
            accept_rank.append(sampled["accept_probability"].mean())
            abstain_rank.append(sampled["abstain_probability"].mean())
        selected_ratios[index] = np.mean(selected_rank)
        oracle_ratios[index] = np.mean(oracle_rank)
        paid_baseline_ratios[index] = np.mean(paid_baseline_rank)
        harmed[index] = np.mean(harmed_rank)
        accept[index] = np.mean(accept_rank)
        abstain[index] = np.mean(abstain_rank)
    arrays = {
        "selected_mean_ratio": selected_ratios,
        "paid_oracle_mean_ratio": oracle_ratios,
        "paid_baseline_mean_ratio": paid_baseline_ratios,
        "fraction_paths_harmed": harmed,
        "accept_probability": accept,
        "abstain_probability": abstain,
    }
    return {
        metric: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for metric, values in arrays.items()
    }


def bootstrap_summaries(paths, summary, samples, master_seed):
    """Cluster-bootstrap every configuration with shared rank-stratified draws.

    Reusing the same path-resampling weights across configurations preserves the
    pairing already present in Phase 1A.  Matrix multiplication makes the full
    10,000-replicate grid practical without changing the bootstrap estimand.
    """
    samples = int(samples)
    master_seed = int(master_seed)
    metric_names = (
        "selected_mean_ratio",
        "paid_oracle_mean_ratio",
        "paid_baseline_mean_ratio",
        "fraction_paths_harmed",
        "accept_probability",
        "abstain_probability",
    )
    value_columns = (
        "risk_original_baseline",
        "risk_paid_selected_mean",
        "risk_paid_oracle",
        "risk_paid_baseline",
        "path_harmed",
        "accept_probability",
        "abstain_probability",
    )
    configs = (
        paths[CONFIG_COLUMNS]
        .drop_duplicates()
        .sort_values(CONFIG_COLUMNS)
        .reset_index(drop=True)
    )
    configs["config_id"] = np.arange(len(configs), dtype=int)
    equal_summary = (
        summary.loc[summary["rank_scope"].eq("equal_rank")]
        .merge(configs, on=CONFIG_COLUMNS, how="inner", validate="one_to_one")
        .sort_values("config_id")
    )
    if len(equal_summary) != len(configs):
        raise RuntimeError("Equal-rank summary does not cover every configuration.")
    evaluable = equal_summary["evaluable"].to_numpy(dtype=bool)

    rank_data = []
    for rank_index, (rank, rank_group) in enumerate(
        paths.groupby("step_rank", sort=True)
    ):
        merged = rank_group.merge(
            configs,
            on=CONFIG_COLUMNS,
            how="inner",
            validate="many_to_one",
        ).sort_values(["config_id", "basis_trial"])
        counts = merged.groupby("config_id", sort=True).size().to_numpy()
        if len(counts) != len(configs) or len(np.unique(counts)) != 1:
            raise RuntimeError("Bootstrap configurations do not share one path grid.")
        path_count = int(counts[0])
        values = merged[list(value_columns)].to_numpy(dtype=float)
        values = values.reshape(len(configs), path_count, len(value_columns))
        values = np.transpose(values, (1, 0, 2))
        rng = np.random.default_rng(
            np.random.SeedSequence([master_seed, rank_index, int(rank)])
        )
        weights = rng.multinomial(
            path_count,
            np.full(path_count, 1.0 / path_count),
            size=samples,
        ).astype(np.float64)
        weights /= path_count
        rank_data.append((weights, values))

    quantiles = {
        metric: (np.full(len(configs), np.nan), np.full(len(configs), np.nan))
        for metric in metric_names
    }
    chunk_size = 128
    for start in range(0, len(configs), chunk_size):
        stop = min(start + chunk_size, len(configs))
        width = stop - start
        boot = {
            metric: np.zeros((samples, width), dtype=np.float64)
            for metric in metric_names
        }
        for weights, values in rank_data:
            original = weights @ values[:, start:stop, 0]
            selected = weights @ values[:, start:stop, 1]
            oracle = weights @ values[:, start:stop, 2]
            paid_baseline = weights @ values[:, start:stop, 3]
            boot["selected_mean_ratio"] += (selected / original) / len(rank_data)
            boot["paid_oracle_mean_ratio"] += (oracle / original) / len(rank_data)
            boot["paid_baseline_mean_ratio"] += (
                paid_baseline / original
            ) / len(rank_data)
            boot["fraction_paths_harmed"] += (
                weights @ values[:, start:stop, 4]
            ) / len(rank_data)
            boot["accept_probability"] += (
                weights @ values[:, start:stop, 5]
            ) / len(rank_data)
            boot["abstain_probability"] += (
                weights @ values[:, start:stop, 6]
            ) / len(rank_data)
        local_evaluable = evaluable[start:stop]
        target_indices = start + np.flatnonzero(local_evaluable)
        for metric, matrix in boot.items():
            if np.any(local_evaluable):
                low, high = np.quantile(
                    matrix[:, local_evaluable], (0.025, 0.975), axis=0
                )
                quantiles[metric][0][target_indices] = low
                quantiles[metric][1][target_indices] = high

    rows = []
    for config_id, config_row in configs.iterrows():
        config_values = {column: config_row[column] for column in CONFIG_COLUMNS}
        for metric in metric_names:
            rows.append(
                {
                    **config_values,
                    "metric": metric,
                    "point": float(equal_summary.iloc[config_id][metric]),
                    "ci_low": float(quantiles[metric][0][config_id]),
                    "ci_high": float(quantiles[metric][1][config_id]),
                    "bootstrap_samples": samples,
                    "bootstrap_seed": master_seed,
                    "bootstrap_stream": "shared_rank_stratified_multinomial",
                }
            )
    return pd.DataFrame(rows)


def catastrophic_summary(paths):
    rows = []
    for config, group in paths.groupby(CONFIG_COLUMNS, sort=True):
        for fraction, column in (
            (0.01, "catastrophic_top1"),
            (0.05, "catastrophic_top5"),
            (0.10, "catastrophic_top10"),
        ):
            selected = group.loc[group[column]].copy()
            rank_metrics = []
            for _, rank_group in selected.groupby("step_rank", sort=True):
                metric = _scope_summary(rank_group)
                original_total = group.loc[
                    group["step_rank"].eq(rank_group["step_rank"].iloc[0]),
                    "risk_original_baseline",
                ].sum()
                metric["baseline_risk_share"] = float(
                    rank_group["risk_original_baseline"].sum() / original_total
                )
                recoverable = (
                    rank_group["risk_paid_baseline"]
                    - rank_group["risk_paid_oracle"]
                ).sum()
                recovered = (
                    rank_group["risk_paid_baseline"]
                    - rank_group["risk_paid_selected_mean"]
                ).sum()
                metric["recovered_risk_share"] = (
                    float(recovered / recoverable) if recoverable > 0.0 else np.nan
                )
                metric["candidate_repays_fraction"] = float(
                    (rank_group["net_benefit_truth"] < 0).mean()
                )
                rank_metrics.append(metric)
            row = {**dict(zip(CONFIG_COLUMNS, config)), "fraction": fraction}
            row["evaluable"] = bool(
                all(metric["evaluable"] for metric in rank_metrics)
            )
            for key in rank_metrics[0]:
                if key == "evaluable":
                    continue
                values = [metric[key] for metric in rank_metrics]
                row[key] = (
                    float(np.mean(values)) if row["evaluable"] else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _primary_mask(frame):
    return (
        np.isclose(frame["eta"], PRIMARY_ETA)
        & frame["budget"].eq(PRIMARY_BUDGET)
        & frame["s"].eq(PRIMARY_S)
        & frame["pair"].eq(PRIMARY_PAIR)
        & frame["estimator"].eq(PRIMARY_ESTIMATOR)
        & np.isclose(frame["epsilon"], PRIMARY_EPSILON)
    )


def evaluate_verdict(summary, bootstrap):
    primary = summary.loc[_primary_mask(summary) & summary["rank_scope"].eq("equal_rank")]
    if len(primary) != 1:
        raise RuntimeError("Primary summary row is missing or duplicated.")
    primary = primary.iloc[0]
    interval = bootstrap.loc[
        _primary_mask(bootstrap)
        & bootstrap["metric"].eq("selected_mean_ratio")
    ]
    if len(interval) != 1:
        raise RuntimeError("Primary bootstrap interval is missing or duplicated.")
    interval = interval.iloc[0]
    if interval.ci_high < 1.0:
        verdict = "NET BENEFIT"
    elif interval.ci_low > 1.0:
        verdict = "NET HARM"
    else:
        verdict = "INCONCLUSIVE"
    tail_tradeoff = bool(
        verdict == "NET BENEFIT"
        and (
            primary.median_pathwise_ratio > 1.0
            or primary.fraction_paths_harmed > 0.5
        )
    )
    oracle_cannot_pay = bool(primary.paid_oracle_mean_ratio >= 1.0)
    label = verdict
    if tail_tradeoff:
        label += " + TAIL-INSURANCE TRADEOFF"
    return pd.DataFrame(
        [
            {
                "verdict": verdict,
                "verdict_label": label,
                "tail_insurance_tradeoff": tail_tradeoff,
                "frozen_oracle_cannot_pay": oracle_cannot_pay,
                "selected_mean_ratio": primary.selected_mean_ratio,
                "ci_low": interval.ci_low,
                "ci_high": interval.ci_high,
                "paid_oracle_mean_ratio": primary.paid_oracle_mean_ratio,
                "paid_baseline_mean_ratio": primary.paid_baseline_mean_ratio,
                "median_pathwise_ratio": primary.median_pathwise_ratio,
                "fraction_paths_harmed": primary.fraction_paths_harmed,
                "accept_probability": primary.accept_probability,
                "abstain_probability": primary.abstain_probability,
            }
        ]
    )


def _make_figures(summary, catastrophic, verdict, staging_dir):
    figure_dir = staging_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    primary_family = summary.loc[
        np.isclose(summary["eta"], PRIMARY_ETA)
        & summary["budget"].eq(PRIMARY_BUDGET)
        & summary["pair"].eq(PRIMARY_PAIR)
        & summary["estimator"].eq(PRIMARY_ESTIMATOR)
        & np.isclose(summary["epsilon"], PRIMARY_EPSILON)
        & summary["rank_scope"].eq("equal_rank")
    ].sort_values("s")
    paths = []

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    row = primary_family.loc[primary_family["s"].eq(PRIMARY_S)].iloc[0]
    labels = ["Original", "Paid fallback", "Paid oracle", "Empirical"]
    values = [1.0, row.paid_baseline_mean_ratio, row.paid_oracle_mean_ratio, row.selected_mean_ratio]
    axis.bar(labels, values, color=[GREY, RED, GREEN, BLUE])
    axis.axhline(1.0, color=INK, linestyle="--", linewidth=1)
    axis.set_ylabel("Mean-risk ratio to original baseline")
    axis.set_title("Primary candidate-first cost decomposition")
    fig.tight_layout()
    path = figure_dir / "figure_1_cost_decomposition.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(primary_family["s"], primary_family["selected_mean_ratio"], marker="o", label="Empirical")
    axis.plot(primary_family["s"], primary_family["paid_oracle_mean_ratio"], marker="o", label="Paid oracle")
    axis.plot(primary_family["s"], primary_family["paid_baseline_mean_ratio"], marker="o", label="Paid fallback")
    axis.axhline(1.0, color=INK, linestyle="--", linewidth=1)
    axis.set_xlabel("Certification probes s")
    axis.set_ylabel("Mean-risk ratio")
    axis.set_title("Net value versus certification cost")
    axis.legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "figure_2_net_ratio_vs_s.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(primary_family["s"], primary_family["accept_probability"], marker="o", label="Accept")
    axis.plot(primary_family["s"], primary_family["reject_probability"], marker="o", label="Reject")
    axis.plot(primary_family["s"], primary_family["abstain_probability"], marker="o", label="Abstain")
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Certification probes s")
    axis.set_ylabel("Path-first probability")
    axis.set_title("Budget-aware decision probabilities")
    axis.legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "figure_3_decisions_vs_s.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    rank_primary = summary.loc[
        np.isclose(summary["eta"], PRIMARY_ETA)
        & summary["budget"].eq(PRIMARY_BUDGET)
        & summary["s"].eq(PRIMARY_S)
        & summary["pair"].eq(PRIMARY_PAIR)
        & summary["estimator"].eq(PRIMARY_ESTIMATOR)
        & np.isclose(summary["epsilon"], PRIMARY_EPSILON)
        & ~summary["rank_scope"].eq("equal_rank")
    ].copy()
    rank_primary["required_reduction"] = 1.0 - rank_primary["mean_net_numerator_threshold"]
    axis.bar(rank_primary["rank_scope"], rank_primary["required_reduction"], color=GOLD)
    axis.set_xlabel("Step rank")
    axis.set_ylabel("Required numerator reduction")
    axis.set_title("Cost that the candidate must repay")
    fig.tight_layout()
    path = figure_dir / "figure_4_required_reduction.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(primary_family["s"], primary_family["median_pathwise_ratio"], marker="o", label="Median path")
    axis.plot(primary_family["s"], primary_family["p90_pathwise_ratio"], marker="o", label="90th percentile")
    axis.axhline(1.0, color=INK, linestyle="--", linewidth=1)
    axis.set_xlabel("Certification probes s")
    axis.set_ylabel("Pathwise paid/original ratio")
    axis.set_title("Typical-path and upper-tail cost")
    axis.legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "figure_5_pathwise_ratios.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    cat = catastrophic.loc[
        np.isclose(catastrophic["eta"], PRIMARY_ETA)
        & catastrophic["budget"].eq(PRIMARY_BUDGET)
        & catastrophic["s"].eq(PRIMARY_S)
        & catastrophic["pair"].eq(PRIMARY_PAIR)
        & catastrophic["estimator"].eq(PRIMARY_ESTIMATOR)
        & np.isclose(catastrophic["epsilon"], PRIMARY_EPSILON)
    ].sort_values("fraction")
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(100 * cat["fraction"], cat["selected_mean_ratio"], marker="o", label="Empirical")
    axis.plot(100 * cat["fraction"], cat["paid_oracle_mean_ratio"], marker="o", label="Paid oracle")
    axis.axhline(1.0, color=INK, linestyle="--", linewidth=1)
    axis.set_yscale("log")
    axis.set_xlabel("Catastrophic stratum (%)")
    axis.set_ylabel("Mean-risk ratio (log scale)")
    axis.set_title("Net value on catastrophic paths")
    axis.legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "figure_6_catastrophic_paths.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    eta_control = summary.loc[
        summary["budget"].eq(PRIMARY_BUDGET)
        & summary["s"].eq(PRIMARY_S)
        & summary["pair"].eq(PRIMARY_PAIR)
        & summary["estimator"].eq(PRIMARY_ESTIMATOR)
        & np.isclose(summary["epsilon"], PRIMARY_EPSILON)
        & summary["rank_scope"].eq("equal_rank")
    ].sort_values("eta")
    axis.bar([f"eta={value:g}" for value in eta_control["eta"]], eta_control["selected_mean_ratio"], color=[GREY, BLUE])
    axis.axhline(1.0, color=INK, linestyle="--", linewidth=1)
    axis.set_ylabel("Selected/original mean-risk ratio")
    axis.set_title("Primary tail family versus benign control")
    fig.tight_layout()
    path = figure_dir / "figure_7_primary_vs_control.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    budget = summary.loc[
        np.isclose(summary["eta"], PRIMARY_ETA)
        & summary["s"].eq(PRIMARY_S)
        & summary["pair"].eq(PRIMARY_PAIR)
        & summary["estimator"].eq(PRIMARY_ESTIMATOR)
        & np.isclose(summary["epsilon"], PRIMARY_EPSILON)
        & summary["rank_scope"].eq("equal_rank")
    ].sort_values("budget")
    axis.plot(budget["budget"], budget["selected_mean_ratio"], marker="o", label="Empirical")
    axis.plot(budget["budget"], budget["paid_oracle_mean_ratio"], marker="o", label="Paid oracle")
    axis.axhline(1.0, color=INK, linestyle="--", linewidth=1)
    axis.set_xlabel("Total budget m")
    axis.set_ylabel("Mean-risk ratio")
    axis.set_title("Budget sensitivity")
    axis.legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "figure_8_budget_sensitivity.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    rank_budget = summary.loc[
        np.isclose(summary["eta"], PRIMARY_ETA)
        & summary["s"].eq(PRIMARY_S)
        & summary["pair"].eq(PRIMARY_PAIR)
        & summary["estimator"].eq(PRIMARY_ESTIMATOR)
        & np.isclose(summary["epsilon"], PRIMARY_EPSILON)
        & ~summary["rank_scope"].eq("equal_rank")
    ].assign(rank_number=lambda frame: frame["rank_scope"].astype(int))
    for rank, group in rank_budget.groupby("rank_number", sort=True):
        axis.plot(group["budget"], group["selected_mean_ratio"], marker="o", label=f"r*={rank}")
    axis.axhline(1.0, color=INK, linestyle="--", linewidth=1)
    axis.set_yscale("log")
    axis.set_xlabel("Total budget m")
    axis.set_ylabel("Mean-risk ratio (log scale)")
    axis.set_title("Per-rank budget sensitivity")
    axis.legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "figure_9_rank_budget_sensitivity.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return paths


def _write_proof_note(path):
    content = r"""# Budget-Aware Direct Certification: Lemma 15.1 and Theorems 16--17

**Classification:** `PROVED` on the explicit nested-construction, positive-denominator, and freshness domains below.

## Assumption ledger

Let $A\in\mathbb R^{d\times d}$ be symmetric. Let $Q_0$ and $Q_a$ be orthonormal bases constructed before certification. Let $q_x$ count sampled products, let $r_x$ be the accepted rank, and let $c_x=q_x+r_x$. Certification uses $s\ge1$ new products. Final residual probes are fresh after selection.

## Lemma 15.1: Nested-construction reuse

If $\operatorname{range}(Q_{\rm low})\subseteq\operatorname{range}(Q_{\rm high})$ and both bases have orthonormal columns, put $T=Q_{\rm high}^TQ_{\rm low}$. The higher-space projector fixes every column of the lower basis, so

$$
Q_{\rm low}=Q_{\rm high}Q_{\rm high}^TQ_{\rm low}=Q_{\rm high}T.
$$

Linearity then gives

$$
AQ_{\rm low}=(AQ_{\rm high})T.
$$

Thus a cached higher-prefix construction supplies the lower basis and its image without another oracle product. For the frozen incremental paths, constructing both actions costs $\max\{c_0,c_a\}$ rather than $c_0+c_a$. The conclusion does not extend to nonnested actions. $\blacksquare$

## Theorem 16: Common sunk-cost denominator

Once both actions have been constructed, the committed cost is $c_{\rm pre}=\max\{c_0,c_a\}$. Certification spends another $s$ products. Since spent queries cannot be refunded, either final basis has

$$
\ell_{\rm paid}=m-c_{\rm pre}-s
$$

fresh residual products. If $\ell_{\rm paid}>0$, the exact conditional risks are

$$
\mathcal R_{0,\rm paid}=\frac{\sigma_0^2}{\ell_{\rm paid}},
\qquad
\mathcal R_{a,\rm paid}=\frac{\sigma_a^2}{\ell_{\rm paid}}.
$$

The common positive denominator proves that the paid candidate is better exactly when $\sigma_a^2<\sigma_0^2$. $\blacksquare$

## Theorem 17: No-free-fallback lemma

The original baseline has $\ell_0=m-c_0$ residual products and risk $\sigma_0^2/\ell_0$. If $\sigma_0^2>0$ and $c_{\rm pre}+s>c_0$, then $\ell_{\rm paid}<\ell_0$, and

$$
\frac{\mathcal R_{0,\rm paid}}{\mathcal R_0^{\rm original}}
=\frac{\ell_0}{\ell_{\rm paid}}>1.
$$

Therefore rejection or abstention after construction and certification cannot restore the original baseline. If $\sigma_0^2=0$, both fallback risks are zero and the displayed ratio is undefined; this boundary must not be handled by division. $\blacksquare$

## Corollary 17.1: Net-benefit threshold

On the domain $\sigma_0^2>0$, the paid candidate beats the original baseline exactly when

$$
\frac{\sigma_a^2}{\sigma_0^2}
<\frac{\ell_{\rm paid}}{\ell_0}
=1-\frac{(c_{\rm pre}-c_0)+s}{\ell_0}.
$$

For the full-rank transition $r_\star\to r_\star+1$, the added construction cost is two, so the right side is $1-(s+2)/(m-2r_\star)$. $\blacksquare$

## Corollary 17.2: Paid two-action oracle

The smallest conditional risk available after both actions and certification have been paid is

$$
\mathcal R_{\rm oracle,paid}
=\frac{\min\{\sigma_0^2,\sigma_a^2\}}{\ell_{\rm paid}}.
$$

Every selector restricted to these bases is pointwise no better. If the finite frozen-population mean of this oracle is no smaller than the original-baseline mean, no selector over this pair can pay for this timing on that frozen population. This is not an orientation-universal impossibility theorem. $\blacksquare$

## Audit result

- The maximum-cost rule requires nested subspaces and cached higher-prefix products.
- Every division requires the displayed positive denominator.
- Paid-order truth $\sigma_a^2<\sigma_0^2$ is distinct from net-benefit truth $\sigma_a^2/\ell_{\rm paid}<\sigma_0^2/\ell_0$.
- Certification-based selection is followed by fresh residual probes; reuse of certification probes is not covered.
"""
    Path(path).write_text(content, encoding="utf-8")


def _write_report(path, verdict, summary, bootstrap, catastrophic):
    primary = summary.loc[_primary_mask(summary) & summary["rank_scope"].eq("equal_rank")].iloc[0]
    interval = bootstrap.loc[
        _primary_mask(bootstrap) & bootstrap["metric"].eq("selected_mean_ratio")
    ].iloc[0]
    unevaluable_summary_rows = int((~summary["evaluable"]).sum())
    rank_rows = summary.loc[
        _primary_mask(summary) & ~summary["rank_scope"].eq("equal_rank")
    ].assign(rank_number=lambda frame: frame["rank_scope"].astype(int)).sort_values(
        "rank_number"
    )
    rank_lines = []
    for row in rank_rows.itertuples(index=False):
        rank_lines.append(
            f"| {row.rank_scope} | {row.mean_net_numerator_threshold:.3f} | "
            f"{row.paid_baseline_mean_ratio:.6g} | {row.paid_oracle_mean_ratio:.6g} | "
            f"{row.selected_mean_ratio:.6g} | {row.median_pathwise_ratio:.6g} | "
            f"{row.fraction_paths_harmed:.2%} |"
        )
    s_rows = summary.loc[
        np.isclose(summary["eta"], PRIMARY_ETA)
        & summary["budget"].eq(PRIMARY_BUDGET)
        & summary["pair"].eq(PRIMARY_PAIR)
        & summary["estimator"].eq(PRIMARY_ESTIMATOR)
        & np.isclose(summary["epsilon"], PRIMARY_EPSILON)
        & summary["rank_scope"].eq("equal_rank")
    ].sort_values("s")
    s_lines = [
        f"| {row.s} | {row.paid_baseline_mean_ratio:.4f} | "
        f"{row.paid_oracle_mean_ratio:.4f} | {row.selected_mean_ratio:.4f} | "
        f"{row.accept_probability:.2%} | {row.abstain_probability:.2%} |"
        for row in s_rows.itertuples(index=False)
    ]
    content = rf"""# Phase 1B-A: Budget-Aware Direct Rademacher-Risk Certification

## Executive verdict

**{verdict.loc[0, 'verdict_label']}**

This is a preregistered empirical verdict conditional on the frozen Phase 1A orientations and paths. It is not a theorem-level confidence certificate.

## Accounting result

**PROVED.** Nested construction and cached higher-prefix products make the committed construction cost

$$
c_{{\rm pre}}=\max\{{q_0+r_0,q_a+r_a\}}.
$$

After $s$ certification queries, both final bases share

$$
\ell_{{\rm paid}}=m-c_{{\rm pre}}-s.
$$

If the original baseline numerator is positive, returning to that basis after paying cannot restore its original risk because $\ell_{{\rm paid}}<\ell_0$.

## Primary result

The primary selected/original mean-risk ratio is **{primary.selected_mean_ratio:.6f}**, with conditional 95% percentile interval **[{interval.ci_low:.6f}, {interval.ci_high:.6f}]**. The paid-oracle ratio is **{primary.paid_oracle_mean_ratio:.6f}**, while always falling back after paying has ratio **{primary.paid_baseline_mean_ratio:.6f}**.

| $r_\star$ | numerator threshold | paid fallback | paid oracle | empirical | median path | paths harmed |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rank_lines)}

## Certification-cost sensitivity

| $s$ | paid fallback | paid oracle | empirical | accept | abstain |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(s_lines)}

## Interpretation

**EMPIRICALLY ESTABLISHED.** The paid oracle separates architecture economics from decision quality. The empirical selector is evaluated only after candidate construction and certification are charged. Paid-order truth compares $\sigma_a^2$ and $\sigma_0^2$; net-benefit truth compares the complete paid candidate with never starting the procedure.

**EMPIRICAL DECISION RULE.** The multiplicative numerator guard is not a finite-sample certificate.

**OPEN.** A simultaneous confidence theorem and a genuinely online schedule remain open. Whether theorem work is warranted is determined by the paid-oracle and empirical results above.

## Limitations

The analysis reuses frozen Phase 1A probes and paths and performs no new matrix-vector experiment. Bootstrap intervals are conditional on that finite frozen population. The original comparator is the baseline action in each frozen adjacent pair, not a claim about every Standard Hutch++ allocation.

The complete preregistered secondary grid is retained. It contains **{unevaluable_summary_rows}** summary rows whose paid residual capacity is nonpositive; their paid risks are undefined and they are not silently dropped or used to renormalize equal-rank conclusions. The primary configuration is feasible in every rank.
"""
    Path(path).write_text(content, encoding="utf-8")


def _historical_checksums():
    checksums = {}
    for path in sorted((ROOT_DIR / "results").glob("*")):
        if path.is_file() and not path.name.startswith(OUTPUT_PREFIX):
            checksums[path.name] = _hash_file(path)
    return checksums


def _publish(staging_dir, output_dir, default_output):
    figure_target = output_dir / "figures" / OUTPUT_PREFIX
    report_target = (
        ROOT_DIR / "reports" / OUTPUTS["report"]
        if default_output
        else output_dir / OUTPUTS["report"]
    )
    proof_target = (
        ROOT_DIR / "docs" / "proof_budget_aware_certification.md"
        if default_output
        else output_dir / "proof_budget_aware_certification.md"
    )
    figure_target.mkdir(parents=True, exist_ok=False)
    for path in (staging_dir / "figures").iterdir():
        shutil.move(str(path), figure_target / path.name)
    for key, filename in OUTPUTS.items():
        if key in {"report", "manifest"}:
            continue
        shutil.move(str(staging_dir / filename), output_dir / filename)
    shutil.move(str(staging_dir / OUTPUTS["manifest"]), output_dir / OUTPUTS["manifest"])
    shutil.move(str(staging_dir / OUTPUTS["report"]), report_target)
    shutil.move(str(staging_dir / "proof_budget_aware_certification.md"), proof_target)
    (staging_dir / "figures").rmdir()
    staging_dir.rmdir()
    return report_target, proof_target, figure_target


def _validate_published_checksums(
    manifest_path, output_dir, report_path, proof_path, figure_dir
):
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 1:
        raise RuntimeError("Published Phase 1B-A manifest must have one row.")
    expected = json.loads(manifest.loc[0, "output_checksums_excluding_manifest"])
    mismatches = []
    for name, checksum in expected.items():
        if name == report_path.name:
            path = report_path
        elif name == proof_path.name:
            path = proof_path
        elif name.startswith(f"figures/{OUTPUT_PREFIX}/"):
            path = figure_dir / Path(name).name
        else:
            path = output_dir / name
        if not path.is_file() or _hash_file(path) != checksum:
            mismatches.append(name)
    if mismatches:
        raise RuntimeError(f"Published output checksum mismatch: {mismatches}")


def run_phase1b_budget(
    phase1a_trials=DEFAULT_PHASE1A_TRIALS,
    phase1a_truth=DEFAULT_PHASE1A_TRUTH,
    phase1a_manifest=DEFAULT_PHASE1A_MANIFEST,
    output_dir=DEFAULT_OUTPUT_DIR,
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
    max_paths_per_rank=None,
):
    phase1a_trials = Path(phase1a_trials).resolve()
    phase1a_truth = Path(phase1a_truth).resolve()
    phase1a_manifest = Path(phase1a_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    default_output = output_dir == DEFAULT_OUTPUT_DIR.resolve()
    if max_paths_per_rank is not None and default_output:
        raise ValueError("Reduced runs must use an alternate output directory.")
    bootstrap_samples = int(bootstrap_samples)
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")
    bootstrap_seed = int(bootstrap_seed)
    if bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be nonnegative.")
    destinations = [output_dir / name for key, name in OUTPUTS.items() if key != "report"]
    destinations.append(
        ROOT_DIR / "reports" / OUTPUTS["report"]
        if default_output
        else output_dir / OUTPUTS["report"]
    )
    destinations.append(
        ROOT_DIR / "docs" / "proof_budget_aware_certification.md"
        if default_output
        else output_dir / "proof_budget_aware_certification.md"
    )
    destinations.append(output_dir / "figures" / OUTPUT_PREFIX)
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing Phase 1B-A outputs: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    historical_before = _historical_checksums() if default_output else {}
    _, source_checksums = _validate_source_files(
        phase1a_trials, phase1a_truth, phase1a_manifest
    )
    core = _load_core(phase1a_trials, max_paths_per_rank=max_paths_per_rank)
    staging_dir = Path(tempfile.mkdtemp(prefix=".phase1b-budget-", dir=output_dir))
    try:
        paths = build_path_artifact(core)
        paths.to_parquet(staging_dir / OUTPUTS["paths"], index=False)
        summary = summarize_paths(paths)
        summary.to_csv(staging_dir / OUTPUTS["summary"], index=False)
        bootstrap = bootstrap_summaries(
            paths, summary, bootstrap_samples, bootstrap_seed
        )
        bootstrap.to_csv(staging_dir / OUTPUTS["bootstrap"], index=False)
        cost_columns = CONFIG_COLUMNS + [
            "rank_scope",
            "original_baseline_mean",
            "paid_baseline_mean",
            "paid_candidate_mean",
            "paid_oracle_mean",
            "paid_selected_mean",
            "selected_mean_ratio",
            "paid_baseline_mean_ratio",
            "paid_candidate_mean_ratio",
            "paid_oracle_mean_ratio",
            "decision_efficiency",
            "mean_cost_multiplier",
            "mean_net_numerator_threshold",
        ]
        summary[cost_columns].to_csv(staging_dir / OUTPUTS["cost"], index=False)
        decision_columns = CONFIG_COLUMNS + [
            "rank_scope",
            "accept_probability",
            "reject_probability",
            "abstain_probability",
            "correct_paid_order_probability",
            "false_candidate_probability",
            "missed_paid_candidate_probability",
            "net_beneficial_accept_probability",
            "net_harmful_accept_probability",
        ]
        summary[decision_columns].to_csv(
            staging_dir / OUTPUTS["decisions"], index=False
        )
        catastrophic = catastrophic_summary(paths)
        catastrophic.to_csv(staging_dir / OUTPUTS["catastrophic"], index=False)
        sensitivity = summary.loc[
            summary["pair"].eq(PRIMARY_PAIR)
            & summary["estimator"].eq(PRIMARY_ESTIMATOR)
            & np.isclose(summary["epsilon"], PRIMARY_EPSILON)
        ].copy()
        sensitivity.to_csv(staging_dir / OUTPUTS["sensitivity"], index=False)
        verdict = evaluate_verdict(summary, bootstrap)
        verdict.to_csv(staging_dir / OUTPUTS["verdict"], index=False)
        figures = _make_figures(summary, catastrophic, verdict, staging_dir)
        _write_proof_note(staging_dir / "proof_budget_aware_certification.md")
        _write_report(
            staging_dir / OUTPUTS["report"],
            verdict,
            summary,
            bootstrap,
            catastrophic,
        )
        historical_after = _historical_checksums() if default_output else {}
        if historical_after != historical_before:
            raise RuntimeError("Historical result artifacts changed during Phase 1B-A.")
        completed = datetime.now(timezone.utc).isoformat()
        output_checksums = {}
        for path in staging_dir.iterdir():
            if path.is_file() and path.name != OUTPUTS["manifest"]:
                output_checksums[path.name] = _hash_file(path)
        for path in figures:
            output_checksums[f"figures/{OUTPUT_PREFIX}/{path.name}"] = _hash_file(path)
        manifest = pd.DataFrame(
            [
                {
                    "configuration_version": "direct_rademacher_phase1b_budget_v1",
                    "started_at_utc": started,
                    "completed_at_utc": completed,
                    "source_checksums": json.dumps(source_checksums, sort_keys=True),
                    "source_rows": len(core),
                    "path_rows": len(paths),
                    "summary_rows": len(summary),
                    "bootstrap_rows": len(bootstrap),
                    "bootstrap_samples": bootstrap_samples,
                    "bootstrap_seed": bootstrap_seed,
                    "bootstrap_stream": "shared_rank_stratified_multinomial",
                    "max_paths_per_rank": max_paths_per_rank,
                    "feasible_path_rows": int(paths["accounting_feasible"].sum()),
                    "infeasible_path_rows": int(
                        (~paths["accounting_feasible"]).sum()
                    ),
                    "evaluable_summary_rows": int(summary["evaluable"].sum()),
                    "unevaluable_summary_rows": int((~summary["evaluable"]).sum()),
                    "primary_configuration": json.dumps(
                        {
                            "budget": PRIMARY_BUDGET,
                            "eta": PRIMARY_ETA,
                            "s": PRIMARY_S,
                            "epsilon": PRIMARY_EPSILON,
                            "pair": PRIMARY_PAIR,
                            "estimator": PRIMARY_ESTIMATOR,
                        },
                        sort_keys=True,
                    ),
                    "verdict": verdict.loc[0, "verdict_label"],
                    "python_version": platform.python_version(),
                    "numpy_version": np.__version__,
                    "pandas_version": pd.__version__,
                    "pyarrow_version": pa.__version__,
                    "historical_checksums": json.dumps(
                        historical_before, sort_keys=True
                    ),
                    "output_checksums_excluding_manifest": json.dumps(
                        output_checksums, sort_keys=True
                    ),
                }
            ]
        )
        manifest.to_csv(staging_dir / OUTPUTS["manifest"], index=False)
        report_path, proof_path, figure_dir = _publish(
            staging_dir, output_dir, default_output
        )
        _validate_published_checksums(
            output_dir / OUTPUTS["manifest"],
            output_dir,
            report_path,
            proof_path,
            figure_dir,
        )
    except Exception:
        raise
    return {
        "verdict": verdict.loc[0, "verdict_label"],
        "paths": len(paths),
        "report": report_path,
        "proof": proof_path,
        "figures": figure_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1a-trials", type=Path, default=DEFAULT_PHASE1A_TRIALS)
    parser.add_argument("--phase1a-truth", type=Path, default=DEFAULT_PHASE1A_TRUTH)
    parser.add_argument("--phase1a-manifest", type=Path, default=DEFAULT_PHASE1A_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--max-paths-per-rank", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    result = run_phase1b_budget(**vars(parse_args()))
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
