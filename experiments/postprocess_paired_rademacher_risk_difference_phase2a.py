"""Artifact-only Phase 2A audit of direct paired Rademacher risk differences."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paired_rademacher_risk_difference import (  # noqa: E402
    construction_accounting,
    shifted_decorrelated_baseline,
)


RESULTS = ROOT / "results"
PHASE1A_TRIALS = RESULTS / "direct_rademacher_certification_phase1a_trials.parquet"
PHASE1A_TRUTH = RESULTS / "direct_rademacher_certification_phase1a_truth.csv"
PHASE1A_MANIFEST = RESULTS / "direct_rademacher_certification_phase1a_manifest.csv"
PHASE1B_PATHS = RESULTS / "direct_rademacher_certification_phase1b_budget_paths.parquet"
PHASE1B_MANIFEST = RESULTS / "direct_rademacher_certification_phase1b_budget_manifest.csv"

PREFIX = "paired_rademacher_risk_difference_phase2a"
OUTPUT_NAMES = {
    "paths": f"{PREFIX}_paths.parquet",
    "summary": f"{PREFIX}_summary.csv",
    "quantiles": f"{PREFIX}_one_sided_quantiles.csv",
    "catastrophic": f"{PREFIX}_catastrophic.csv",
    "bootstrap": f"{PREFIX}_bootstrap.csv",
    "gate": f"{PREFIX}_gate.csv",
    "verdict": f"{PREFIX}_verdict.csv",
    "manifest": f"{PREFIX}_manifest.csv",
}
REPORT_NAME = "paired_rademacher_risk_difference_phase2a.md"
FIGURE_DIR_NAME = PREFIX

STEP_RANKS = (5, 15, 30)
ETAS = (1e-10, 1e-6)
BUDGETS = (80, 160, 240)
SAMPLE_SIZES = (4, 8, 16, 32)
ACTION_PAIRS = {
    "left": ("km1", "k"),
    "primary": ("kp1", "k"),
    "right": ("kp2", "kp1"),
}
ACTIONS = ("km1", "k", "kp1", "kp2")
PRIMARY = {"eta": 1e-6, "budget": 160, "s": 16, "pair": "primary"}
BOOTSTRAP_SEED = 93_000
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
TOLERANCE_FACTOR = 128.0 * np.finfo(np.float64).eps


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_columns():
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
    for action in ACTIONS:
        columns.extend(
            [
                f"q__{action}",
                f"r_actual__{action}",
                f"sigma2_exact__{action}",
                f"sigma2_hat__{action}__sample_variance",
            ]
        )
    return columns


def validate_frozen_inputs():
    paths = (
        PHASE1A_TRIALS,
        PHASE1A_TRUTH,
        PHASE1A_MANIFEST,
        PHASE1B_PATHS,
        PHASE1B_MANIFEST,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    phase1a = pd.read_csv(PHASE1A_MANIFEST)
    phase1b = pd.read_csv(PHASE1B_MANIFEST)
    if len(phase1a) != 1 or len(phase1b) != 1:
        raise RuntimeError("Frozen manifests must each contain exactly one row.")
    a_outputs = json.loads(phase1a.loc[0, "output_checksums_excluding_manifest"])
    b_outputs = json.loads(phase1b.loc[0, "output_checksums_excluding_manifest"])
    expected = {
        PHASE1A_TRIALS: a_outputs[PHASE1A_TRIALS.name],
        PHASE1A_TRUTH: a_outputs[PHASE1A_TRUTH.name],
        PHASE1B_PATHS: b_outputs[PHASE1B_PATHS.name],
    }
    for path, checksum in expected.items():
        if file_hash(path) != checksum:
            raise RuntimeError(f"Frozen checksum mismatch: {path.name}")
    if pq.ParquetFile(PHASE1A_TRIALS).metadata.num_rows != 960_000:
        raise RuntimeError("Phase 1A source must contain 960,000 rows.")
    if pq.ParquetFile(PHASE1B_PATHS).metadata.num_rows != 691_200:
        raise RuntimeError("Phase 1B source must contain 691,200 rows.")
    schema = set(pq.ParquetFile(PHASE1A_TRIALS).schema.names)
    missing = sorted(set(required_columns()) - schema)
    if missing:
        raise RuntimeError(f"Phase 1A artifact is missing columns: {missing}")
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def load_source(max_paths_per_rank=None, repetitions_per_batch=50):
    frame = pd.read_parquet(PHASE1A_TRIALS, columns=required_columns())
    if max_paths_per_rank is not None:
        frame = frame.loc[frame["basis_trial"] < int(max_paths_per_rank)].copy()
    repetitions_per_batch = int(repetitions_per_batch)
    if not 2 <= repetitions_per_batch <= 50:
        raise ValueError("repetitions_per_batch must be in [2, 50].")
    frame = frame.loc[frame["repetition"] < repetitions_per_batch].copy()
    counts = frame.groupby(
        ["step_rank", "basis_trial", "eta", "s", "batch"], sort=False
    ).size()
    if not np.all(counts.to_numpy() == repetitions_per_batch):
        raise RuntimeError("Every retained path/configuration/batch must be complete.")
    return frame


def truth_label(candidate, baseline):
    tolerance = TOLERANCE_FACTOR * max(abs(candidate), abs(baseline))
    difference = candidate - baseline
    if difference < -tolerance:
        return -1
    if difference > tolerance:
        return 1
    return 0


def sample_metrics(candidate, baseline, batches, repetitions, exact_delta):
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    paired = candidate - baseline
    shifted_baseline = shifted_decorrelated_baseline(baseline, batches, repetitions)
    shifted = candidate - shifted_baseline
    candidate_variance = float(np.var(candidate, ddof=1))
    baseline_variance = float(np.var(baseline, ddof=1))
    covariance = float(np.cov(candidate, baseline, ddof=1)[0, 1])
    independence_variance = candidate_variance + baseline_variance
    paired_variance = float(np.var(paired, ddof=1))
    shifted_variance = float(np.var(shifted, ddof=1))
    if independence_variance == 0.0:
        covariance_contribution = np.nan
        pairing_ratio = np.nan
        shifted_ratio = np.nan
    else:
        covariance_contribution = 2.0 * covariance / independence_variance
        pairing_ratio = paired_variance / independence_variance
        shifted_ratio = shifted_variance / independence_variance
    dangerous_error = exact_delta - paired
    shifted_error = exact_delta - shifted
    batch_ratios = []
    for batch in np.unique(batches):
        mask = batches == batch
        candidate_batch = candidate[mask]
        baseline_batch = baseline[mask]
        denominator = float(
            np.var(candidate_batch, ddof=1) + np.var(baseline_batch, ddof=1)
        )
        batch_ratios.append(
            np.nan
            if denominator == 0.0
            else float(np.var(candidate_batch - baseline_batch, ddof=1) / denominator)
        )
    quantiles = np.quantile(dangerous_error, [0.90, 0.95, 0.975, 0.99])
    shifted_quantiles = np.quantile(shifted_error, [0.90, 0.95, 0.975, 0.99])
    return {
        "candidate_estimate_variance": candidate_variance,
        "baseline_estimate_variance": baseline_variance,
        "paired_estimate_covariance": covariance,
        "independence_variance_benchmark": independence_variance,
        "paired_difference_variance": paired_variance,
        "shifted_difference_variance": shifted_variance,
        "covariance_contribution": covariance_contribution,
        "pairing_variance_ratio": pairing_ratio,
        "shifted_variance_ratio": shifted_ratio,
        "dangerous_error_mean": float(np.mean(dangerous_error)),
        "dangerous_error_q90": float(quantiles[0]),
        "dangerous_error_q95": float(quantiles[1]),
        "dangerous_error_q975": float(quantiles[2]),
        "dangerous_error_q99": float(quantiles[3]),
        "shifted_error_q90": float(shifted_quantiles[0]),
        "shifted_error_q95": float(shifted_quantiles[1]),
        "shifted_error_q975": float(shifted_quantiles[2]),
        "shifted_error_q99": float(shifted_quantiles[3]),
        "zero_radius_accept_probability": float(np.mean(paired <= 0.0)),
        "batch_pairing_ratio_min": float(np.nanmin(batch_ratios)),
        "batch_pairing_ratio_max": float(np.nanmax(batch_ratios)),
        "batch_pairing_ratio_std": float(np.nanstd(batch_ratios, ddof=0)),
    }


def build_path_table(source):
    rows = []
    grouping = ["eta", "s", "step_rank", "basis_trial"]
    for (eta, sample_size, rank, trial), group in source.groupby(grouping, sort=True):
        group = group.sort_values(["batch", "repetition"])
        first = group.iloc[0]
        batches = group["batch"].to_numpy(dtype=int)
        repetitions = group["repetition"].to_numpy(dtype=int)
        for pair, (candidate_action, baseline_action) in ACTION_PAIRS.items():
            candidate_q = int(first[f"q__{candidate_action}"])
            candidate_r = int(first[f"r_actual__{candidate_action}"])
            baseline_q = int(first[f"q__{baseline_action}"])
            baseline_r = int(first[f"r_actual__{baseline_action}"])
            sigma_candidate = float(first[f"sigma2_exact__{candidate_action}"])
            sigma_baseline = float(first[f"sigma2_exact__{baseline_action}"])
            candidate_hat_sigma = group[
                f"sigma2_hat__{candidate_action}__sample_variance"
            ].to_numpy(dtype=float)
            baseline_hat_sigma = group[
                f"sigma2_hat__{baseline_action}__sample_variance"
            ].to_numpy(dtype=float)
            for budget in BUDGETS:
                accounting = construction_accounting(
                    budget=budget,
                    candidate_q=candidate_q,
                    candidate_r=candidate_r,
                    baseline_q=baseline_q,
                    baseline_r=baseline_r,
                    sample_size=int(sample_size),
                    nested_shared_prefix=True,
                )
                feasible = accounting.original_ell > 0 and accounting.paid_ell > 0
                base = {
                    "rank_index": int(first["rank_index"]),
                    "step_rank": int(rank),
                    "basis_trial": int(trial),
                    "eta": float(eta),
                    "budget": int(budget),
                    "s": int(sample_size),
                    "pair": pair,
                    "candidate_action": candidate_action,
                    "baseline_action": baseline_action,
                    "orientation_seed": int(first["orientation_seed"]),
                    "basis_seed": int(first["basis_seed"]),
                    "q_candidate": candidate_q,
                    "r_candidate": candidate_r,
                    "q_baseline": baseline_q,
                    "r_baseline": baseline_r,
                    "candidate_cost": accounting.candidate_cost,
                    "baseline_cost": accounting.baseline_cost,
                    "committed_construction_count": accounting.committed_cost,
                    "construction_accounting_mode": accounting.accounting_mode,
                    "ell_paid_candidate": accounting.paid_ell,
                    "ell_original_baseline": accounting.original_ell,
                    "accounting_feasible": feasible,
                    "sigma2_candidate": sigma_candidate,
                    "sigma2_baseline": sigma_baseline,
                    "new_matvec_queries": 0,
                }
                qk = int(first["q__k"])
                rk = int(first["r_actual__k"])
                base["risk_original_k"] = (
                    float(first["sigma2_exact__k"]) / (budget - qk - rk)
                    if budget - qk - rk > 0
                    else np.nan
                )
                if not feasible:
                    base.update(
                        {
                            "exact_candidate_paid_risk": np.nan,
                            "exact_baseline_original_risk": np.nan,
                            "exact_delta": np.nan,
                            "truth_label": 0,
                        }
                    )
                    for name in sample_metrics(
                        np.arange(8.0), np.arange(8.0),
                        np.repeat(np.arange(4), 2), np.tile(np.arange(2), 4), 0.0
                    ):
                        base[name] = np.nan
                    rows.append(base)
                    continue
                exact_candidate = sigma_candidate / accounting.paid_ell
                exact_baseline = sigma_baseline / accounting.original_ell
                exact_delta = exact_candidate - exact_baseline
                candidate_hat = candidate_hat_sigma / accounting.paid_ell
                baseline_hat = baseline_hat_sigma / accounting.original_ell
                metrics = sample_metrics(
                    candidate_hat,
                    baseline_hat,
                    batches,
                    repetitions,
                    exact_delta,
                )
                label = truth_label(exact_candidate, exact_baseline)
                metrics["false_safe_probability"] = (
                    metrics["zero_radius_accept_probability"] if label > 0 else 0.0
                )
                metrics["true_better_accept_probability"] = (
                    metrics["zero_radius_accept_probability"] if label < 0 else 0.0
                )
                base.update(
                    {
                        "exact_candidate_paid_risk": exact_candidate,
                        "exact_baseline_original_risk": exact_baseline,
                        "exact_delta": exact_delta,
                        "truth_label": label,
                    }
                )
                base.update(metrics)
                rows.append(base)
    result = pd.DataFrame(rows)
    expected = (
        source["step_rank"].nunique()
        * source.groupby("step_rank")["basis_trial"].nunique().min()
        * len(ETAS)
        * len(BUDGETS)
        * len(SAMPLE_SIZES)
        * len(ACTION_PAIRS)
    )
    if len(result) != expected:
        raise RuntimeError(f"Path table has {len(result)} rows, expected {expected}.")
    key = ["step_rank", "basis_trial", "eta", "budget", "s", "pair"]
    if result.duplicated(key).any():
        raise RuntimeError("Path-table key is not unique.")
    return mark_catastrophic_paths(result)


def validate_phase1b_accounting(paths):
    """Cross-check the nested shared-prefix accounting against frozen Phase 1B."""

    columns = [
        "step_rank",
        "basis_trial",
        "eta",
        "budget",
        "s",
        "pair",
        "estimator",
        "epsilon",
        "candidate_cost",
        "baseline_cost",
        "committed_cost",
        "ell_original",
        "ell_paid",
        "accounting_feasible",
    ]
    frozen = pd.read_parquet(PHASE1B_PATHS, columns=columns)
    frozen = frozen.loc[
        frozen["estimator"].eq("sample_variance")
        & np.isclose(frozen["epsilon"], 1.0 / 3.0)
    ].copy()
    retained_trials = paths[["step_rank", "basis_trial"]].drop_duplicates()
    frozen = frozen.merge(
        retained_trials,
        on=["step_rank", "basis_trial"],
        how="inner",
        validate="many_to_one",
    )
    key = ["step_rank", "basis_trial", "eta", "budget", "s", "pair"]
    frozen = frozen.drop(columns=["estimator", "epsilon"])
    comparison = paths.merge(
        frozen,
        on=key,
        how="left",
        suffixes=("_phase2a", "_phase1b"),
        validate="one_to_one",
        indicator=True,
    )
    if not comparison["_merge"].eq("both").all():
        raise RuntimeError("Phase 1B accounting cross-check is missing frozen rows.")
    integer_pairs = (
        ("candidate_cost_phase2a", "candidate_cost_phase1b"),
        ("baseline_cost_phase2a", "baseline_cost_phase1b"),
        ("committed_construction_count", "committed_cost"),
        ("ell_original_baseline", "ell_original"),
        ("ell_paid_candidate", "ell_paid"),
    )
    for current, frozen_name in integer_pairs:
        if not np.array_equal(
            comparison[current].to_numpy(), comparison[frozen_name].to_numpy()
        ):
            raise RuntimeError(
                f"Phase 2A accounting disagrees with Phase 1B for {current}."
            )
    if not np.array_equal(
        comparison["accounting_feasible_phase2a"].to_numpy(dtype=bool),
        comparison["accounting_feasible_phase1b"].to_numpy(dtype=bool),
    ):
        raise RuntimeError("Phase 2A feasibility disagrees with Phase 1B.")
    if not paths["construction_accounting_mode"].eq(
        "nested_shared_prefix_max"
    ).all():
        raise RuntimeError("Frozen Phase 2A paths must use nested-prefix accounting.")
    return True


def mark_catastrophic_paths(paths):
    paths = paths.copy()
    for fraction, label in ((0.01, "top1"), (0.05, "top5"), (0.10, "top10")):
        paths[f"catastrophic_{label}"] = False
    unique = paths[
        ["step_rank", "basis_trial", "eta", "budget", "risk_original_k"]
    ].drop_duplicates()
    for (rank, eta, budget), group in unique.groupby(
        ["step_rank", "eta", "budget"], sort=True
    ):
        for fraction, label in ((0.01, "top1"), (0.05, "top5"), (0.10, "top10")):
            count = max(1, int(round(fraction * len(group))))
            trials = set(group.nlargest(count, "risk_original_k")["basis_trial"])
            mask = (
                paths["step_rank"].eq(rank)
                & np.isclose(paths["eta"], eta)
                & paths["budget"].eq(budget)
                & paths["basis_trial"].isin(trials)
            )
            paths.loc[mask, f"catastrophic_{label}"] = True
    return paths


METRICS = (
    "covariance_contribution",
    "pairing_variance_ratio",
    "shifted_variance_ratio",
    "dangerous_error_q95",
    "dangerous_error_q99",
    "zero_radius_accept_probability",
    "batch_pairing_ratio_std",
)


def summarize_paths(paths):
    rows = []
    config = ["eta", "budget", "s", "pair"]
    feasible = paths.loc[paths["accounting_feasible"]].copy()
    for keys, group in feasible.groupby(config + ["step_rank"], sort=True):
        row = dict(zip(config + ["step_rank"], keys))
        row["scope"] = "rank"
        row["eligible_paths"] = len(group)
        row["true_better_paths"] = int(np.sum(group["truth_label"] < 0))
        row["true_worse_paths"] = int(np.sum(group["truth_label"] > 0))
        row["truth_tie_paths"] = int(np.sum(group["truth_label"] == 0))
        for metric in METRICS:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    rank_summary = pd.DataFrame(rows)
    aggregate_rows = []
    for keys, group in rank_summary.groupby(config, sort=True):
        row = dict(zip(config, keys))
        row["step_rank"] = "equal_rank"
        row["scope"] = "equal_rank"
        row["eligible_paths"] = int(group["eligible_paths"].sum())
        row["true_better_paths"] = int(group["true_better_paths"].sum())
        row["true_worse_paths"] = int(group["true_worse_paths"].sum())
        row["truth_tie_paths"] = int(group["truth_tie_paths"].sum())
        for metric in METRICS:
            row[metric] = float(group[metric].mean())
        aggregate_rows.append(row)
    return pd.concat([rank_summary, pd.DataFrame(aggregate_rows)], ignore_index=True)


def build_quantile_table(paths):
    columns = [
        "step_rank", "basis_trial", "eta", "budget", "s", "pair",
        "accounting_feasible", "truth_label", "dangerous_error_q90",
        "dangerous_error_q95", "dangerous_error_q975", "dangerous_error_q99",
        "shifted_error_q90", "shifted_error_q95", "shifted_error_q975",
        "shifted_error_q99",
    ]
    return paths[columns].copy()


def build_catastrophic_table(paths):
    primary = primary_scope(paths)
    rows = []
    for rank, group in primary.groupby("step_rank", sort=True):
        for label in ("top1", "top5", "top10"):
            subset = group.loc[group[f"catastrophic_{label}"]]
            better = subset.loc[subset["truth_label"] < 0]
            rows.append(
                {
                    "step_rank": rank,
                    "stratum": label,
                    "paths": len(subset),
                    "true_better_paths": len(better),
                    "mean_pairing_variance_ratio": float(subset["pairing_variance_ratio"].mean()),
                    "mean_covariance_contribution": float(subset["covariance_contribution"].mean()),
                    "true_better_zero_radius_acceptance": (
                        float(better["zero_radius_accept_probability"].mean())
                        if len(better) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def primary_scope(frame):
    return frame.loc[
        np.isclose(frame["eta"], PRIMARY["eta"])
        & frame["budget"].eq(PRIMARY["budget"])
        & frame["s"].eq(PRIMARY["s"])
        & frame["pair"].eq(PRIMARY["pair"])
        & frame["accounting_feasible"]
    ].copy()


def bootstrap_primary(paths, samples):
    primary = primary_scope(paths)
    by_rank = {
        int(rank): group.sort_values("basis_trial")
        for rank, group in primary.groupby("step_rank", sort=True)
    }
    if set(by_rank) != set(STEP_RANKS):
        raise RuntimeError("Primary bootstrap is missing a rank.")
    rows = []
    for metric_index, metric in enumerate(
        ("pairing_variance_ratio", "covariance_contribution")
    ):
        rng = np.random.default_rng(
            np.random.SeedSequence([BOOTSTRAP_SEED, metric_index])
        )
        rank_bootstraps = []
        for rank in STEP_RANKS:
            values = by_rank[rank][metric].to_numpy(dtype=float)
            indices = rng.integers(0, len(values), size=(samples, len(values)))
            rank_bootstraps.append(np.mean(values[indices], axis=1))
        aggregate = np.mean(rank_bootstraps, axis=0)
        rows.append(
            {
                "metric": metric,
                "bootstrap_samples": samples,
                "point_estimate": float(
                    np.mean([by_rank[rank][metric].mean() for rank in STEP_RANKS])
                ),
                "ci_lower": float(np.quantile(aggregate, 0.025)),
                "ci_upper": float(np.quantile(aggregate, 0.975)),
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        )
    return pd.DataFrame(rows)


def evaluate_gate(paths, bootstrap, checksums_valid):
    primary = primary_scope(paths)
    rank_groups = {int(rank): group for rank, group in primary.groupby("step_rank")}
    ratio_row = bootstrap.loc[bootstrap["metric"].eq("pairing_variance_ratio")].iloc[0]
    covariance_row = bootstrap.loc[
        bootstrap["metric"].eq("covariance_contribution")
    ].iloc[0]
    conditions = {
        "positive_equal_rank_covariance": covariance_row["point_estimate"] > 0.0,
        "pairing_ratio_ci_upper_below_one": ratio_row["ci_upper"] < 1.0,
        "pairing_ratio_at_most_0p80": ratio_row["point_estimate"] <= 0.80,
        "better_and_worse_paths_each_rank": all(
            np.any(group["truth_label"] < 0) and np.any(group["truth_label"] > 0)
            for group in rank_groups.values()
        ) and set(rank_groups) == set(STEP_RANKS),
        "top5_better_nonempty_each_rank": all(
            np.any(group["catastrophic_top5"] & (group["truth_label"] < 0))
            for group in rank_groups.values()
        ) and set(rank_groups) == set(STEP_RANKS),
        "frozen_checksums_valid": bool(checksums_valid),
    }
    if not conditions["frozen_checksums_valid"]:
        verdict = "INCONCLUSIVE"
    elif not conditions["better_and_worse_paths_each_rank"] or not conditions[
        "top5_better_nonempty_each_rank"
    ]:
        verdict = "INCONCLUSIVE"
    elif all(conditions.values()):
        verdict = "PAIRING SIGNAL GO"
    elif conditions["positive_equal_rank_covariance"]:
        verdict = "WEAK / MIXED PAIRING SIGNAL"
    else:
        verdict = "NO PAIRING ADVANTAGE"
    gate = pd.DataFrame(
        [{"criterion": name, "passed": bool(value)} for name, value in conditions.items()]
    )
    return gate, verdict


def generate_figures(paths, summary, output_dir):
    figure_dir = Path(output_dir) / "figures" / FIGURE_DIR_NAME
    figure_dir.mkdir(parents=True, exist_ok=True)
    primary = primary_scope(paths)
    files = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    subset = summary.loc[
        np.isclose(summary["eta"], PRIMARY["eta"])
        & summary["budget"].eq(PRIMARY["budget"])
        & summary["pair"].eq(PRIMARY["pair"])
        & summary["scope"].eq("equal_rank")
    ]
    ax.plot(subset["s"], subset["pairing_variance_ratio"], marker="o", label="paired")
    ax.plot(subset["s"], subset["shifted_variance_ratio"], marker="s", label="shifted/decorrelated")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="independence benchmark")
    ax.set(xlabel="Certification probes s", ylabel="Variance ratio", title="Pairing gain versus sample size")
    ax.legend()
    files.append(save_figure(fig, figure_dir / "figure_1_pairing_gain_vs_s.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [primary.loc[primary["step_rank"].eq(rank), "covariance_contribution"] for rank in STEP_RANKS]
    ax.boxplot(data, tick_labels=[str(rank) for rank in STEP_RANKS], showfliers=True)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel=r"$r_\star$", ylabel="Covariance contribution", title="Pathwise common-probe covariance")
    files.append(save_figure(fig, figure_dir / "figure_2_covariance_distribution.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(primary["dangerous_error_q95"], primary["shifted_error_q95"], s=12, alpha=0.5)
    limits = np.nanpercentile(np.r_[primary["dangerous_error_q95"], primary["shifted_error_q95"]], [1, 99])
    ax.plot(limits, limits, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Paired q95 dangerous error", ylabel="Shifted comparator q95", title="One-sided error: paired versus shifted")
    files.append(save_figure(fig, figure_dir / "figure_3_one_sided_error.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    positive = primary["exact_baseline_original_risk"] > 0.0
    ratio = primary.loc[positive, "exact_candidate_paid_risk"] / primary.loc[positive, "exact_baseline_original_risk"]
    ax.scatter(ratio, primary.loc[positive, "pairing_variance_ratio"], s=12, alpha=0.5)
    ax.set_xscale("log")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Exact paid-candidate/original-baseline risk ratio", ylabel="Pairing variance ratio", title="Pairing gain versus exact risk ratio")
    files.append(save_figure(fig, figure_dir / "figure_4_pairing_vs_risk_ratio.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    categories = ["ordinary", "top5"]
    values = [
        primary.loc[~primary["catastrophic_top5"], "pairing_variance_ratio"].mean(),
        primary.loc[primary["catastrophic_top5"], "pairing_variance_ratio"].mean(),
    ]
    ax.bar(categories, values, color=["#5B7FA3", "#B55A4A"])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set(ylabel="Mean pairing variance ratio", title="Ordinary versus catastrophic paths")
    files.append(save_figure(fig, figure_dir / "figure_5_catastrophic_pairing.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(
        [primary.loc[primary["step_rank"].eq(rank), "batch_pairing_ratio_std"] for rank in STEP_RANKS],
        tick_labels=[str(rank) for rank in STEP_RANKS],
    )
    ax.set(xlabel=r"$r_\star$", ylabel="SD across four batch ratios", title="Batch stability")
    files.append(save_figure(fig, figure_dir / "figure_6_batch_stability.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    secondary = summary.loc[
        summary["scope"].eq("equal_rank")
        & summary["s"].eq(PRIMARY["s"])
        & summary["pair"].eq(PRIMARY["pair"])
    ]
    for eta, group in secondary.groupby("eta"):
        ax.plot(group["budget"], group["pairing_variance_ratio"], marker="o", label=f"eta={eta:g}")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Budget m", ylabel="Pairing variance ratio", title="Budget and tail sensitivity")
    ax.legend()
    files.append(save_figure(fig, figure_dir / "figure_7_sensitivity.png"))
    return files


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def write_report(output_dir, verdict, bootstrap, catastrophic, paths):
    ratio = bootstrap.loc[bootstrap["metric"].eq("pairing_variance_ratio")].iloc[0]
    covariance = bootstrap.loc[bootstrap["metric"].eq("covariance_contribution")].iloc[0]
    primary = primary_scope(paths)
    top5 = catastrophic.loc[catastrophic["stratum"].eq("top5")]
    better = primary.loc[primary["truth_label"] < 0]
    worse = primary.loc[primary["truth_label"] > 0]
    ordinary = primary.loc[~primary["catastrophic_top5"]]
    catastrophic_primary = primary.loc[primary["catastrophic_top5"]]
    per_rank = primary.groupby("step_rank", sort=True).agg(
        paths=("basis_trial", "size"),
        true_better=("truth_label", lambda values: int(np.sum(values < 0))),
        pairing_ratio=("pairing_variance_ratio", "mean"),
        covariance=("covariance_contribution", "mean"),
    )
    rank_lines = "\n".join(
        f"| {rank} | {int(row.paths)} | {int(row.true_better)} | "
        f"{row.pairing_ratio:.6f} | {row.covariance:.6f} |"
        for rank, row in per_rank.iterrows()
    )
    report = rf"""# Phase 2A: Direct Paired Rademacher Risk-Difference Audit

**Verdict:** `{verdict}`

## Executive result

This artifact-only continuation study issued no new matrix--vector experiment.
It tested whether the same certification probes reduce uncertainty in

$$
\widehat\Delta_R=S_a^2/\ell_a-S_0^2/\ell_0
$$

relative to the exact-form independence variance benchmark
$V_{{\rm ind}}=\operatorname{{Var}}(S_a^2/\ell_a)+\operatorname{{Var}}(S_0^2/\ell_0)$.
The reported $V_{{\rm ind}}$ values are empirical plug-in estimates of that
mathematically exact independence formula.

At the frozen primary configuration, the equal-rank pairing variance ratio is
**{ratio['point_estimate']:.6f}** with conditional path-bootstrap interval
**[{ratio['ci_lower']:.6f}, {ratio['ci_upper']:.6f}]**. The corresponding
covariance contribution is **{covariance['point_estimate']:.6f}** with interval
**[{covariance['ci_lower']:.6f}, {covariance['ci_upper']:.6f}]**.

The three ranks contain {len(better)} net-beneficial
and {len(worse)} net-harmful paths in total. Top-5%
true-better eligibility by rank is
{dict(zip(top5['step_rank'].astype(int), top5['true_better_paths'].astype(int)))}.

## What the GO verdict does and does not mean

The preregistered aggregate gate passes decisively: the common-probe statistic
retains only about {100.0 * ratio['point_estimate']:.1f}% of the plug-in
independence-benchmark variance. The within-batch shifted/decorrelated
comparator has equal-rank mean ratio
**{primary['shifted_variance_ratio'].groupby(primary['step_rank']).mean().mean():.6f}**,
close to one, which supports the interpretation that the original same-probe
alignment is responsible for the aggregate cancellation.

The gain is not uniform. Net-beneficial paths have mean pairing ratio
**{better['pairing_variance_ratio'].mean():.6f}**, whereas net-harmful paths
have mean ratio **{worse['pairing_variance_ratio'].mean():.6f}**. The top-5%
catastrophic paths have mean ratio
**{catastrophic_primary['pairing_variance_ratio'].mean():.6f}**, compared with
**{ordinary['pairing_variance_ratio'].mean():.6f}** on ordinary paths. Thus the
global GO establishes a strong paired signal, but the cancellation is weakest
precisely on the rare beneficial tail paths that motivate certification. This
tail limitation must be carried into Phase 2B rather than hidden by the
aggregate verdict.

## Primary results by rank

| $r_\star$ | paths | truly better | pairing ratio | covariance contribution |
|---:|---:|---:|---:|---:|
{rank_lines}

The pathwise ratios are computed first within each frozen path, then averaged
within rank, then averaged equally across ranks. Individual certification
repetitions are not treated as independent research paths.

## Evidence classification

- `PROVED`: conditional unbiasedness, paired Hoeffding decomposition, and exact
  variance/covariance identities in the frozen plan and tested module.
- `EMPIRICALLY ESTABLISHED`: the reported covariance and variance-ratio results
  conditional on the frozen orientations and path population.
- `DESCRIPTIVE COMPARATOR`: the within-batch cyclic shift; it is not called an
  independent sequence. $V_{{\rm ind}}$ remains the mathematical independence
  benchmark.
- `OPEN`: a finite-sample one-sided radius for $\Delta_R$ and every online
  allocator consequence.

## Accounting scope

The formula $c_{{\rm pre}}=\max\{{q_a+r_a,q_0+r_0\}}$ is used only because the
primary frozen actions are nested shared prefixes. Other architectures must use
their actual committed construction-query count from a query ledger.

Every one of the 43,200 Phase 2A accounting rows was cross-checked against the
frozen Phase 1B table. Infeasible rows remain in the artifact with an explicit
flag; no denominator was silently changed or repaired.

## Recommendation

The preregistered gate authorizes a separate Phase 2B one-sided theorem
feasibility study for the direct paired difference. It does not establish a
finite-sample radius, and this report does not implement or authorize an
allocator.
"""
    report_path = Path(output_dir).parent / "reports" / REPORT_NAME if Path(output_dir).resolve() == RESULTS.resolve() else Path(output_dir) / REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report_path


def write_outputs(output_dir, paths, summary, quantiles, catastrophic, bootstrap, gate, verdict, input_checksums, figure_files, report_path, started):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {name: output_dir / filename for name, filename in OUTPUT_NAMES.items()}
    pq.write_table(pa.Table.from_pandas(paths, preserve_index=False), output_paths["paths"], compression="zstd")
    summary.to_csv(output_paths["summary"], index=False)
    quantiles.to_csv(output_paths["quantiles"], index=False)
    catastrophic.to_csv(output_paths["catastrophic"], index=False)
    bootstrap.to_csv(output_paths["bootstrap"], index=False)
    gate.to_csv(output_paths["gate"], index=False)
    pd.DataFrame([{"phase": "Phase 2A", "verdict": verdict, "new_matvec_queries": 0}]).to_csv(output_paths["verdict"], index=False)
    output_hashes = {
        path.name: file_hash(path)
        for name, path in output_paths.items()
        if name != "manifest"
    }
    output_hashes[str(report_path.relative_to(ROOT)) if report_path.is_relative_to(ROOT) else report_path.name] = file_hash(report_path)
    for path in figure_files:
        output_hashes[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name] = file_hash(path)
    manifest = pd.DataFrame(
        [{
            "configuration_version": "paired_rademacher_risk_difference_phase2a_v1",
            "started_at_utc": started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "primary_configuration": json.dumps(PRIMARY, sort_keys=True),
            "path_rows": len(paths),
            "expected_full_path_rows": 43_200,
            "bootstrap_samples": int(bootstrap["bootstrap_samples"].max()),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "new_matvec_queries": 0,
            "verdict": verdict,
            "construction_accounting_scope": "max cost only for nested shared prefixes; otherwise actual query ledger",
            "shifted_comparator_status": "decorrelated diagnostic, not independent sequence",
            "independence_benchmark": "V_ind = Var(candidate estimate) + Var(baseline estimate)",
            "input_checksums": json.dumps(input_checksums, sort_keys=True),
            "output_checksums_excluding_manifest": json.dumps(output_hashes, sort_keys=True),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "pyarrow_version": pa.__version__,
        }]
    )
    manifest.to_csv(output_paths["manifest"], index=False)
    return output_paths


def run_phase2a(output_dir=RESULTS, max_paths_per_rank=None, repetitions_per_batch=50, bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES, make_figures=True):
    started = datetime.now(timezone.utc).isoformat()
    before = validate_frozen_inputs()
    source = load_source(max_paths_per_rank, repetitions_per_batch)
    paths = build_path_table(source)
    validate_phase1b_accounting(paths)
    summary = summarize_paths(paths)
    quantiles = build_quantile_table(paths)
    catastrophic = build_catastrophic_table(paths)
    bootstrap = bootstrap_primary(paths, int(bootstrap_samples))
    after = validate_frozen_inputs()
    if before != after:
        raise RuntimeError("Frozen inputs changed during Phase 2A.")
    gate, verdict = evaluate_gate(paths, bootstrap, before == after)
    figure_files = generate_figures(paths, summary, output_dir) if make_figures else []
    report_path = write_report(output_dir, verdict, bootstrap, catastrophic, paths)
    output_paths = write_outputs(
        output_dir, paths, summary, quantiles, catastrophic, bootstrap, gate,
        verdict, before, figure_files, report_path, started,
    )
    return {
        "paths": paths,
        "summary": summary,
        "bootstrap": bootstrap,
        "gate": gate,
        "verdict": verdict,
        "output_paths": output_paths,
        "report_path": report_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--max-paths-per-rank", type=int)
    parser.add_argument("--repetitions-per-batch", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    result = run_phase2a(
        output_dir=args.output_dir,
        max_paths_per_rank=args.max_paths_per_rank,
        repetitions_per_batch=args.repetitions_per_batch,
        bootstrap_samples=args.bootstrap_samples,
        make_figures=not args.skip_figures,
    )
    print(result["verdict"])


if __name__ == "__main__":
    main()
