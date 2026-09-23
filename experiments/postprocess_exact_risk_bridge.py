"""Read-only regret postprocessor for the completed exact-risk bridge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from risk_bridge_regret import (  # noqa: E402
    paired_bootstrap_minimizer_frequencies,
    regret_columns,
)


DEFAULT_TRIALS_CSV = ROOT_DIR / "results" / "risk_bridge_exact_trials.csv"
DEFAULT_BOOTSTRAP_SAMPLES = 20_000
DEFAULT_BOOTSTRAP_SEED_BASE = 20_260_815
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results"
RISK_COLUMNS = ("oracle_risk", "gaussian_risk", "rademacher_risk")


def validate_exact_trials(trials):
    required = {
        "setup_index",
        "spectrum_family",
        "setup_name",
        "matrix_seed",
        "basis_trial",
        "q",
        *RISK_COLUMNS,
    }
    missing = sorted(required.difference(trials.columns))
    if missing:
        raise ValueError(f"Exact bridge trial CSV is missing columns: {missing}")
    if trials.empty or trials.duplicated(["setup_index", "basis_trial", "q"]).any():
        raise ValueError("Exact bridge trials are empty or have duplicate keys.")
    numeric = trials[list(RISK_COLUMNS)].to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(numeric)) or np.any(numeric < 0.0):
        raise ValueError("Exact bridge risks must be finite and nonnegative.")
    for setup_index, selected in trials.groupby("setup_index", sort=True):
        q_values = np.sort(selected["q"].unique())
        trial_values = np.sort(selected["basis_trial"].unique())
        expected = len(q_values) * len(trial_values)
        if len(selected) != expected:
            raise ValueError(f"Setup {setup_index} has an incomplete paired grid.")
        counts = selected.groupby("basis_trial")["q"].nunique().to_numpy()
        if np.any(counts != len(q_values)):
            raise ValueError(f"Setup {setup_index} has an incomplete trial curve.")


def postprocess_exact_bridge(
    trials_csv=DEFAULT_TRIALS_CSV,
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed_base=DEFAULT_BOOTSTRAP_SEED_BASE,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    if isinstance(bootstrap_samples, (bool, np.bool_)) or not isinstance(
        bootstrap_samples, (int, np.integer)
    ) or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be a positive integer.")
    if isinstance(bootstrap_seed_base, (bool, np.bool_)) or not isinstance(
        bootstrap_seed_base, (int, np.integer)
    ) or bootstrap_seed_base < 0:
        raise ValueError("bootstrap_seed_base must be a nonnegative integer.")
    trials_csv = Path(trials_csv)
    trials = pd.read_csv(trials_csv)
    validate_exact_trials(trials)

    curve_rows = []
    frequency_rows = []
    keys = ["setup_index", "spectrum_family", "setup_name", "matrix_seed"]
    for key_values, selected in trials.groupby(keys, sort=True):
        info = dict(zip(keys, key_values))
        q_values = np.sort(selected["q"].unique())
        trial_indices = np.sort(selected["basis_trial"].unique())
        local_rows = {int(q): {**info, "q": int(q)} for q in q_values}
        for risk_name in RISK_COLUMNS:
            pivot = selected.pivot(index="basis_trial", columns="q", values=risk_name)
            pivot = pivot.reindex(index=trial_indices, columns=q_values)
            if pivot.isna().any().any():
                raise ValueError("Exact bridge contains an incomplete paired curve.")
            risk_matrix = pivot.to_numpy(dtype=np.float64)
            mean_curve = risk_matrix.mean(axis=0)
            standard_errors = (
                risk_matrix.std(axis=0, ddof=1) / np.sqrt(risk_matrix.shape[0])
                if risk_matrix.shape[0] > 1
                else np.zeros(q_values.size)
            )
            regrets = regret_columns(q_values, mean_curve)
            label = risk_name.removesuffix("_risk")
            for index, q in enumerate(q_values):
                row = local_rows[int(q)]
                row[f"mean_{risk_name}"] = float(mean_curve[index])
                row[f"se_{risk_name}"] = float(standard_errors[index])
                row[f"additive_regret_{label}"] = float(
                    regrets["additive_regret"][index]
                )
                row[f"baseline_normalized_additive_regret_{label}"] = float(
                    regrets["baseline_normalized_additive_regret"][index]
                )
                row[f"multiplicative_regret_{label}"] = float(
                    regrets["multiplicative_regret"][index]
                )
                row[f"minimum_q_smallest_{label}"] = regrets["minimum_q_smallest"]
                row[f"minimum_q_largest_{label}"] = regrets["minimum_q_largest"]
                row[f"minimum_q_count_{label}"] = regrets["minimum_q_count"]
                row[f"minimum_is_zero_{label}"] = regrets["minimum_is_zero"]
                row[f"minimum_is_plateau_{label}"] = regrets[
                    "minimum_is_plateau"
                ]
            bootstrap = paired_bootstrap_minimizer_frequencies(
                risk_matrix,
                q_values,
                int(bootstrap_samples),
                int(bootstrap_seed_base) + int(info["setup_index"]),
            )
            for q, count, frequency in zip(
                bootstrap["q_values"],
                bootstrap["counts"],
                bootstrap["frequencies"],
            ):
                frequency_rows.append(
                    {
                        **info,
                        "risk_name": risk_name,
                        "q": int(q),
                        "bootstrap_count": int(count),
                        "bootstrap_frequency": float(frequency),
                        "bootstrap_samples": int(bootstrap_samples),
                        "bootstrap_seed": int(bootstrap_seed_base)
                        + int(info["setup_index"]),
                    }
                )
        curve_rows.extend(local_rows[int(q)] for q in q_values)

    curves = pd.DataFrame(curve_rows)
    frequencies = pd.DataFrame(frequency_rows)
    sums = frequencies.groupby(["setup_index", "risk_name"])[
        "bootstrap_frequency"
    ].sum()
    if not np.allclose(sums.to_numpy(), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("Bootstrap minimizer frequencies do not sum to one.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "curves": output_dir / "risk_bridge_exact_regret_curves.csv",
        "frequencies": output_dir / "risk_bridge_exact_minimizer_frequencies.csv",
    }
    temporary = {
        name: path.with_name(f".{path.name}.tmp") for name, path in outputs.items()
    }
    curves.to_csv(temporary["curves"], index=False)
    frequencies.to_csv(temporary["frequencies"], index=False)
    for name in outputs:
        temporary[name].replace(outputs[name])
    return curves, frequencies


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-csv", type=Path, default=DEFAULT_TRIALS_CSV)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument(
        "--bootstrap-seed-base", type=int, default=DEFAULT_BOOTSTRAP_SEED_BASE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    postprocess_exact_bridge(**vars(parse_args()))
