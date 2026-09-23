import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
for path in (SRC_DIR, EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trace_baseline import (  # noqa: E402
    Adaptive_Hutch_pplus_SequentialPilot,
    Hutch_pplus,
    MatVecOracle,
)
from test_sequential_pilot import (  # noqa: E402
    GAMMA_GAP,
    generate_exponential_psd,
    generate_powerlaw_psd,
    generate_step_psd,
)


D = 500
M = 160
N_TRIALS = 50
MAX_Q_SHIFT = 4
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_813

STANDARD = "Hutch++ (Standard)"
UNGUARDED = "Sequential Pilot Safe (Ours)"
GUARDED = "Sequential Pilot Trust-Region (s=4)"


def _make_setups():
    rng = np.random.default_rng(42)
    return [
        generate_powerlaw_psd(D, 2.0, rng),
        generate_powerlaw_psd(D, 0.5, rng),
        generate_exponential_psd(D, 0.05, rng),
        generate_step_psd(D, r=10, eta=0.01, rng=rng),
        generate_step_psd(D, r=20, eta=0.01, rng=rng),
    ]


def _run_trial(A, algorithm, trial_seed):
    oracle = MatVecOracle(A, d=D)
    rng = np.random.default_rng(trial_seed)

    if algorithm == STANDARD:
        estimate = Hutch_pplus(oracle, M, D, rng=rng)
        q_standard = M // 3
        diagnostics = {
            "b_final": 0,
            "r_pilot_actual": 0,
            "q_0": q_standard,
            "q_adapt_raw": q_standard,
            "q_target": q_standard,
            "r_actual": q_standard,
            "ell_eff": M - 2 * q_standard,
            "guard_applied": False,
            "guard_relaxed_for_pilot_floor": False,
            "max_adjacent_log_gap": np.nan,
            "gap_location": 0,
            "stop_reason": "not_applicable",
        }
    else:
        max_q_shift = MAX_Q_SHIFT if algorithm == GUARDED else None
        estimate, diagnostics = Adaptive_Hutch_pplus_SequentialPilot(
            oracle,
            M,
            D,
            b_0=8,
            delta_b=4,
            tau_gap=GAMMA_GAP,
            rng=rng,
            return_diagnostics=True,
            max_q_shift=max_q_shift,
        )

    if oracle.query_count != M:
        raise RuntimeError(
            f"Query count mismatch for {algorithm}: {oracle.query_count} != {M}"
        )
    if (
        diagnostics["q_target"]
        + diagnostics["r_actual"]
        + diagnostics["ell_eff"]
        != M
    ):
        raise RuntimeError(f"Budget identity failed for {algorithm}: {diagnostics}")

    return float(estimate), diagnostics, oracle.query_count


def _paired_bootstrap_ratio(squared_errors, baseline_errors, seed):
    squared_errors = np.asarray(squared_errors, dtype=np.float64)
    baseline_errors = np.asarray(baseline_errors, dtype=np.float64)
    if squared_errors.shape != baseline_errors.shape:
        raise ValueError("Paired bootstrap inputs must have the same shape.")

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(squared_errors),
        size=(BOOTSTRAP_SAMPLES, len(squared_errors)),
    )
    denominator = baseline_errors[indices].mean(axis=1)
    if np.any(denominator <= 0.0):
        raise RuntimeError("Bootstrap baseline MSE must remain positive.")
    ratios = squared_errors[indices].mean(axis=1) / denominator
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def _summarize(trials):
    rows = []
    algorithms = [STANDARD, UNGUARDED, GUARDED]

    for setup_index, setup_name in enumerate(trials["setup"].drop_duplicates()):
        setup_trials = trials[trials["setup"] == setup_name]
        baseline = (
            setup_trials[setup_trials["algorithm"] == STANDARD]
            .sort_values("trial")
            .reset_index(drop=True)
        )

        for algorithm_index, algorithm in enumerate(algorithms):
            method = (
                setup_trials[setup_trials["algorithm"] == algorithm]
                .sort_values("trial")
                .reset_index(drop=True)
            )
            if not np.array_equal(method["trial"].to_numpy(), baseline["trial"].to_numpy()):
                raise RuntimeError(f"Trials are not paired for {setup_name}, {algorithm}.")

            mse = float(method["squared_error"].mean())
            baseline_mse = float(baseline["squared_error"].mean())
            ratio = mse / baseline_mse
            if algorithm == STANDARD:
                ci_low, ci_high = 1.0, 1.0
            else:
                ci_low, ci_high = _paired_bootstrap_ratio(
                    method["squared_error"].to_numpy(),
                    baseline["squared_error"].to_numpy(),
                    BOOTSTRAP_SEED + 10 * setup_index + algorithm_index,
                )

            rows.append(
                {
                    "setup": setup_name,
                    "algorithm": algorithm,
                    "mse": mse,
                    "median_relative_error": float(method["relative_error"].median()),
                    "mean_final_b": float(method["b_final"].mean()),
                    "mean_q_adapt_raw": float(method["q_adapt_raw"].mean()),
                    "mean_q_target": float(method["q_target"].mean()),
                    "mean_ell_eff": float(method["ell_eff"].mean()),
                    "guard_activation_rate": float(method["guard_applied"].mean()),
                    "mse_ratio_vs_standard": ratio,
                    "mse_ratio_ci_low": ci_low,
                    "mse_ratio_ci_high": ci_high,
                }
            )

    return pd.DataFrame(rows)


def _check_historical_reproduction(summary):
    historical_path = ROOT_DIR / "results" / "sequential_pilot_benchmark_results.csv"
    if not historical_path.exists():
        return

    historical = pd.read_csv(historical_path)
    for algorithm in (STANDARD, UNGUARDED):
        previous = historical[historical["algorithm"] == algorithm][
            ["setup", "mse", "median_rel_error", "mean_final_b"]
        ].rename(
            columns={
                "mse": "mse_previous",
                "median_rel_error": "median_previous",
                "mean_final_b": "b_previous",
            }
        )
        current = summary[summary["algorithm"] == algorithm][
            ["setup", "mse", "median_relative_error", "mean_final_b"]
        ]
        comparison = current.merge(previous, on="setup", validate="one_to_one")
        if len(comparison) != 5:
            raise RuntimeError(f"Historical comparison is incomplete for {algorithm}.")
        if not np.allclose(comparison["mse"], comparison["mse_previous"], rtol=1e-13):
            raise RuntimeError(f"Historical MSE regression detected for {algorithm}.")
        if not np.allclose(
            comparison["median_relative_error"],
            comparison["median_previous"],
            rtol=1e-13,
        ):
            raise RuntimeError(f"Historical median regression detected for {algorithm}.")
        if not np.allclose(comparison["mean_final_b"], comparison["b_previous"]):
            raise RuntimeError(f"Historical pilot-size regression detected for {algorithm}.")


def run_benchmark():
    algorithms = [STANDARD, UNGUARDED, GUARDED]
    records = []

    for A, setup_name in _make_setups():
        exact_trace = float(np.trace(A))
        print(f"Running {setup_name}", flush=True)

        for algorithm in algorithms:
            for trial in range(N_TRIALS):
                trial_seed = 1000 + trial
                estimate, diagnostics, query_count = _run_trial(
                    A,
                    algorithm,
                    trial_seed,
                )
                error = estimate - exact_trace
                records.append(
                    {
                        "setup": setup_name,
                        "algorithm": algorithm,
                        "trial": trial,
                        "trial_seed": trial_seed,
                        "exact_trace": exact_trace,
                        "estimate": estimate,
                        "squared_error": error**2,
                        "relative_error": abs(error) / exact_trace,
                        "b_final": diagnostics["b_final"],
                        "r_pilot_actual": diagnostics["r_pilot_actual"],
                        "q_0": diagnostics["q_0"],
                        "q_adapt_raw": diagnostics["q_adapt_raw"],
                        "q_target": diagnostics["q_target"],
                        "r_actual": diagnostics["r_actual"],
                        "ell_eff": diagnostics["ell_eff"],
                        "guard_applied": diagnostics["guard_applied"],
                        "guard_relaxed_for_pilot_floor": diagnostics[
                            "guard_relaxed_for_pilot_floor"
                        ],
                        "max_adjacent_log_gap": diagnostics["max_adjacent_log_gap"],
                        "gap_location": diagnostics["gap_location"],
                        "stop_reason": diagnostics["stop_reason"],
                        "query_count": query_count,
                    }
                )

    trials = pd.DataFrame(records)
    summary = _summarize(trials)
    _check_historical_reproduction(summary)

    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    trials_path = results_dir / "sequential_pilot_guard_trials.csv"
    summary_path = results_dir / "sequential_pilot_guard_summary.csv"
    trials.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)

    guarded = summary[summary["algorithm"] == GUARDED][
        [
            "setup",
            "mse_ratio_vs_standard",
            "mse_ratio_ci_low",
            "mse_ratio_ci_high",
            "mean_q_adapt_raw",
            "mean_q_target",
            "guard_activation_rate",
        ]
    ]
    print("\nGuarded results relative to Standard Hutch++:")
    print(guarded.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nSaved trial results to {trials_path}")
    print(f"Saved summary results to {summary_path}")


if __name__ == "__main__":
    run_benchmark()
