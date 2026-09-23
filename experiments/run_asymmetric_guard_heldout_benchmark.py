"""Frozen held-out benchmark for baseline-preserving asymmetric Adaptive Hutch++."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as la


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trace_baseline import (  # noqa: E402
    Adaptive_Hutch_pplus_SequentialPilot,
    Hutch_pplus,
    MatVecOracle,
)


STANDARD = "Hutch++ (Standard)"
FORCED = "Sequential Forced Baseline (q=q0)"
UNGUARDED = "Sequential Unguarded (B<=q0)"
SYMMETRIC = "Sequential Symmetric Guard [-4,+4]"
ASYMMETRIC = "Sequential Asymmetric Guard [0,+4]"
METHODS = (STANDARD, FORCED, UNGUARDED, SYMMETRIC, ASYMMETRIC)
SEQUENTIAL_METHODS = METHODS[1:]

DEFAULT_DIMENSION = 500
DEFAULT_BUDGET = 160
DEFAULT_TRIALS = 200
DEFAULT_BOOTSTRAP_SAMPLES = 20_000
TRIAL_SEED_BASE = 50_000
MATRIX_SEED_BASE = 42_000
BOOTSTRAP_SEED_BASE = 20_260_814
TAU_GAP = 1.5
B_0 = 8
DELTA_B = 4


def build_setup_manifest():
    """Return the frozen 24-setup manifest without constructing matrices."""
    setups = []

    def add(family, setup_name, **parameters):
        setup_index = len(setups)
        row = {
            "setup_index": setup_index,
            "spectrum_family": family,
            "setup_name": setup_name,
            "matrix_seed": MATRIX_SEED_BASE + setup_index,
            "spectral_parameters": json.dumps(parameters, sort_keys=True),
            "c": np.nan,
            "alpha": np.nan,
            "r_star": np.nan,
            "eta": np.nan,
            "variant": "",
        }
        row.update(parameters)
        setups.append(row)

    for c in (0.3, 0.7, 1.2, 1.7, 2.5):
        add("power", f"power_c_{c:g}", c=c)
    for alpha in (0.02, 0.08, 0.10, 0.15):
        add("exponential", f"exponential_alpha_{alpha:.2f}", alpha=alpha)
    for r_star in (5, 15, 25, 30):
        for eta in (0.001, 0.05, 0.1):
            add(
                "step",
                f"step_r_{r_star}_eta_{eta:g}",
                r_star=r_star,
                eta=eta,
            )
    add("misspecified", "smooth_elbow", variant="smooth_elbow")
    add("misspecified", "power_exponential_mixture", variant="mixture")
    add("misspecified", "lognormal_shaped_decay", variant="lognormal")

    manifest = pd.DataFrame(setups)
    if len(manifest) != 24 or manifest["setup_name"].nunique() != 24:
        raise RuntimeError("The frozen manifest must contain exactly 24 unique setups.")
    return manifest


def build_eigenvalues(setup, dimension):
    """Construct and validate the spectrum for one manifest row."""
    if isinstance(setup, pd.Series):
        setup = setup.to_dict()
    i = np.arange(1, dimension + 1, dtype=np.float64)
    family = setup["spectrum_family"]

    if family == "power":
        eigenvalues = i ** (-float(setup["c"]))
    elif family == "exponential":
        eigenvalues = np.exp(-float(setup["alpha"]) * i)
    elif family == "step":
        eigenvalues = np.full(dimension, float(setup["eta"]), dtype=np.float64)
        eigenvalues[: int(setup["r_star"])] = 1.0
    elif setup["variant"] == "smooth_elbow":
        eigenvalues = (1.0 + (i / 20.0) ** 4) ** (-0.5)
        eigenvalues /= eigenvalues[0]
    elif setup["variant"] == "mixture":
        eigenvalues = 0.5 * i ** (-1.2) + 0.5 * np.exp(-0.08 * (i - 1.0))
        eigenvalues /= eigenvalues[0]
    elif setup["variant"] == "lognormal":
        eigenvalues = np.exp(-0.35 * np.log(i) ** 2)
    else:
        raise ValueError(f"Unknown spectrum setup: {setup}")

    if eigenvalues.shape != (dimension,):
        raise RuntimeError("Spectrum has the wrong dimension.")
    if not np.all(np.isfinite(eigenvalues)):
        raise RuntimeError("Spectrum contains a nonfinite value.")
    if not np.all(eigenvalues > 0.0):
        raise RuntimeError("Spectrum must be strictly positive.")
    if np.any(np.diff(eigenvalues) > 1e-14):
        raise RuntimeError("Spectrum must be nonincreasing.")
    return eigenvalues


def build_oriented_matrix(eigenvalues, matrix_seed):
    """Construct A = (Q diag(lambda)) Q^T from a deterministic Gaussian QR."""
    dimension = len(eigenvalues)
    rng = np.random.default_rng(int(matrix_seed))
    orientation, _ = la.qr(rng.normal(size=(dimension, dimension)))
    matrix = (orientation * eigenvalues) @ orientation.T
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("Oriented matrix contains a nonfinite value.")
    return matrix


def _standard_diagnostics(budget):
    q_0 = budget // 3
    return {
        "b_final": 0,
        "r_pilot_actual": 0,
        "q_0": q_0,
        "q_adapt_raw": q_0,
        "q_target": q_0,
        "delta_q_raw": 0,
        "delta_q_safe": 0,
        "r_actual": q_0,
        "ell_eff": budget - 2 * q_0,
        "guard_applied": False,
        "guard_relaxed_for_pilot_floor": False,
        "pilot_cap_applied": False,
        "max_adjacent_log_gap": np.nan,
        "gap_location": 0,
        "stop_reason": "not_applicable",
    }


def _run_trial(matrix, method, trial_seed, dimension, budget):
    oracle = MatVecOracle(matrix, d=dimension)
    rng = np.random.default_rng(int(trial_seed))
    q_0 = min(dimension, (budget - 2) // 2, budget // 3)

    if method == STANDARD:
        estimate, standard_allocation = Hutch_pplus(
            oracle, budget, dimension, rng=rng, return_diagnostics=True
        )
        diagnostics = _standard_diagnostics(budget)
        diagnostics.update(standard_allocation)
    else:
        bounds = {
            FORCED: (0, 0),
            UNGUARDED: None,
            SYMMETRIC: (-4, 4),
            ASYMMETRIC: (0, 4),
        }[method]
        estimate, diagnostics = Adaptive_Hutch_pplus_SequentialPilot(
            oracle,
            budget,
            dimension,
            b_0=B_0,
            delta_b=DELTA_B,
            b_max=q_0,
            tau_gap=TAU_GAP,
            rng=rng,
            return_diagnostics=True,
            q_shift_bounds=bounds,
            preserve_baseline_feasibility=True,
        )

    if oracle.query_count != budget:
        raise RuntimeError(
            f"Query count mismatch for {method}: {oracle.query_count} != {budget}."
        )
    if (
        diagnostics["q_target"]
        + diagnostics["r_actual"]
        + diagnostics["ell_eff"]
        != budget
    ):
        raise RuntimeError(f"Budget identity failed for {method}: {diagnostics}")

    q_target = diagnostics["q_target"]
    if method == FORCED and q_target != diagnostics["q_0"]:
        raise RuntimeError("Forced-baseline method moved away from q_0.")
    if method == SYMMETRIC and not diagnostics["q_0"] - 4 <= q_target <= diagnostics["q_0"] + 4:
        raise RuntimeError("Symmetric guard bound failed.")
    if method == ASYMMETRIC and not diagnostics["q_0"] <= q_target <= diagnostics["q_0"] + 4:
        raise RuntimeError("Asymmetric guard bound failed.")
    if method in SEQUENTIAL_METHODS and diagnostics["b_final"] > diagnostics["q_0"]:
        raise RuntimeError("Baseline-preserving pilot exceeded q_0.")

    return float(estimate), diagnostics, oracle.query_count


def _paired_bootstrap_ratio(method_errors, baseline_errors, samples, seed):
    method_errors = np.asarray(method_errors, dtype=np.float64)
    baseline_errors = np.asarray(baseline_errors, dtype=np.float64)
    if method_errors.shape != baseline_errors.shape or method_errors.size == 0:
        raise ValueError("Paired bootstrap inputs must be nonempty and aligned.")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, method_errors.size, size=(samples, method_errors.size))
    denominator = baseline_errors[indices].mean(axis=1)
    if np.any(denominator <= 0.0):
        raise RuntimeError("Bootstrap baseline MSE must remain positive.")
    ratios = method_errors[indices].mean(axis=1) / denominator
    return tuple(float(value) for value in np.quantile(ratios, (0.025, 0.975)))


def summarize_trials(trials, bootstrap_samples):
    rows = []
    setup_names = trials.sort_values("setup_index")["setup_name"].drop_duplicates()
    for setup_index, setup_name in enumerate(setup_names):
        setup_trials = trials[trials["setup_name"] == setup_name]
        baseline = (
            setup_trials[setup_trials["method"] == STANDARD]
            .sort_values("trial_index")
            .reset_index(drop=True)
        )
        baseline_mse = float(baseline["squared_error"].mean())

        for method_index, method in enumerate(METHODS):
            selected = (
                setup_trials[setup_trials["method"] == method]
                .sort_values("trial_index")
                .reset_index(drop=True)
            )
            if not np.array_equal(
                selected["trial_index"].to_numpy(), baseline["trial_index"].to_numpy()
            ):
                raise RuntimeError(f"Trial indices are not paired for {setup_name}, {method}.")
            mse = float(selected["squared_error"].mean())
            ratio = mse / baseline_mse
            if method == STANDARD:
                ci_low, ci_high = 1.0, 1.0
            else:
                ci_low, ci_high = _paired_bootstrap_ratio(
                    selected["squared_error"],
                    baseline["squared_error"],
                    bootstrap_samples,
                    BOOTSTRAP_SEED_BASE + 10 * setup_index + method_index,
                )
            if ci_high < 1.0:
                comparison = "resolved better"
            elif ci_low > 1.0:
                comparison = "resolved worse"
            else:
                comparison = "unresolved"

            row = {
                "setup_index": int(selected["setup_index"].iloc[0]),
                "spectrum_family": selected["spectrum_family"].iloc[0],
                "setup_name": setup_name,
                "method": method,
                "observations": len(selected),
                "mse": mse,
                "median_relative_error": float(selected["relative_error"].median()),
                "guard_activation_rate": float(selected["guard_intervention"].mean()),
                "pilot_cap_activation_rate": float(selected["pilot_cap_applied"].mean()),
                "mse_ratio_vs_standard": ratio,
                "mse_ratio_ci_low": ci_low,
                "mse_ratio_ci_high": ci_high,
                "comparison_flag": comparison,
            }
            for field in ("b_final", "q_adapt_raw", "q_target", "r_actual", "ell_eff"):
                values = selected[field]
                row[f"mean_{field}"] = float(values.mean())
                row[f"std_{field}"] = float(values.std(ddof=0))
                row[f"min_{field}"] = float(values.min())
                row[f"max_{field}"] = float(values.max())
            rows.append(row)
    return pd.DataFrame(rows)


def stratify_delta_q(trials):
    sequential = trials[trials["method"].isin(SEQUENTIAL_METHODS)]
    rows = []
    for (setup_index, family, setup_name, method, category), selected in sequential.groupby(
        ["setup_index", "spectrum_family", "setup_name", "method", "delta_q_category"],
        sort=False,
    ):
        rows.append(
            {
                "setup_index": setup_index,
                "spectrum_family": family,
                "setup_name": setup_name,
                "method": method,
                "delta_q_category": category,
                "observations": len(selected),
                "mse": float(selected["squared_error"].mean()),
                "median_relative_error": float(selected["relative_error"].median()),
                "mean_delta_q_raw": float(selected["delta_q_raw"].mean()),
                "mean_delta_q_safe": float(selected["delta_q_safe"].mean()),
                "guard_activation_rate": float(selected["guard_intervention"].mean()),
            }
        )
    return pd.DataFrame(rows)


def validate_outputs(manifest, trials, summary, requested_trials, budget):
    expected_trials = 24 * len(METHODS) * requested_trials
    if len(manifest) != 24 or manifest["setup_name"].nunique() != 24:
        raise RuntimeError("Manifest validation failed.")
    if len(trials) != expected_trials:
        raise RuntimeError(f"Expected {expected_trials} trial rows, found {len(trials)}.")
    counts = trials.groupby(["setup_name", "method"]).size()
    if len(counts) != 24 * len(METHODS) or not np.all(counts == requested_trials):
        raise RuntimeError("Each method/setup must have exactly the requested trial count.")
    if len(summary) != 24 * len(METHODS):
        raise RuntimeError("Summary must contain one row per method/setup.")
    if trials.duplicated(["setup_name", "method", "trial_index"]).any():
        raise RuntimeError("Duplicate trial identifier detected.")
    numeric = trials[["exact_trace", "estimate", "squared_error", "relative_error"]]
    if not np.all(np.isfinite(numeric.to_numpy())):
        raise RuntimeError("Nonfinite estimate or error detected.")
    if not np.all(trials["query_count"].to_numpy() == budget):
        raise RuntimeError("Query-count validation failed.")
    if not np.all(
        trials["q_target"] + trials["r_actual"] + trials["ell_eff"] == budget
    ):
        raise RuntimeError("Budget-identity validation failed.")
    sequential = trials[trials["method"].isin(SEQUENTIAL_METHODS)]
    for field in (
        "b_final",
        "r_pilot_actual",
        "q_0",
        "q_adapt_raw",
        "max_adjacent_log_gap",
        "gap_location",
        "stop_reason",
    ):
        counts_per_trial = sequential.groupby(["setup_name", "trial_index"])[field].nunique(
            dropna=False
        )
        if not np.all(counts_per_trial == 1):
            raise RuntimeError(
                f"Sequential methods do not share the common pilot diagnostic {field}."
            )


def run_benchmark(trials_per_method, dimension, budget, bootstrap_samples, output_dir):
    if trials_per_method <= 0 or dimension <= 0 or budget <= 0 or bootstrap_samples <= 0:
        raise ValueError("All numeric benchmark settings must be positive.")
    q_0 = min(dimension, (budget - 2) // 2, budget // 3)
    if q_0 < B_0:
        raise ValueError(f"Configuration requires q_0 >= {B_0}; got q_0={q_0}.")

    manifest = build_setup_manifest()
    records = []
    for setup in manifest.to_dict("records"):
        eigenvalues = build_eigenvalues(setup, dimension)
        matrix = build_oriented_matrix(eigenvalues, setup["matrix_seed"])
        exact_trace = float(np.sum(eigenvalues))
        print(f"Running {setup['setup_name']} ({setup['setup_index'] + 1}/24)", flush=True)

        for method in METHODS:
            for trial_index in range(trials_per_method):
                trial_seed = TRIAL_SEED_BASE + trial_index
                estimate, diagnostics, query_count = _run_trial(
                    matrix, method, trial_seed, dimension, budget
                )
                error = estimate - exact_trace
                delta_raw = int(diagnostics["q_adapt_raw"] - diagnostics["q_0"])
                category = "negative" if delta_raw < 0 else "positive" if delta_raw > 0 else "zero"
                records.append(
                    {
                        "setup_index": setup["setup_index"],
                        "spectrum_family": setup["spectrum_family"],
                        "setup_name": setup["setup_name"],
                        "spectral_parameters": setup["spectral_parameters"],
                        "matrix_seed": setup["matrix_seed"],
                        "method": method,
                        "trial_index": trial_index,
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
                        "delta_q_raw": delta_raw,
                        "delta_q_safe": diagnostics["q_target"] - diagnostics["q_0"],
                        "delta_q_category": category,
                        "r_actual": diagnostics["r_actual"],
                        "ell_eff": diagnostics["ell_eff"],
                        "guard_intervention": bool(diagnostics["guard_applied"]),
                        "guard_relaxed_for_pilot_floor": bool(
                            diagnostics["guard_relaxed_for_pilot_floor"]
                        ),
                        "pilot_cap_applied": bool(diagnostics["pilot_cap_applied"]),
                        "max_adjacent_log_gap": diagnostics["max_adjacent_log_gap"],
                        "gap_location": diagnostics["gap_location"],
                        "stop_reason": diagnostics["stop_reason"],
                        "query_count": query_count,
                    }
                )

    trial_frame = pd.DataFrame(records)
    summary = summarize_trials(trial_frame, bootstrap_samples)
    strata = stratify_delta_q(trial_frame)
    validate_outputs(manifest, trial_frame, summary, trials_per_method, budget)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "manifest": output_dir / "asymmetric_guard_heldout_manifest.csv",
        "trials": output_dir / "asymmetric_guard_heldout_trials.csv",
        "summary": output_dir / "asymmetric_guard_heldout_summary.csv",
        "strata": output_dir / "asymmetric_guard_delta_q_strata.csv",
    }
    manifest.to_csv(artifacts["manifest"], index=False)
    trial_frame.to_csv(artifacts["trials"], index=False)
    summary.to_csv(artifacts["summary"], index=False)
    strata.to_csv(artifacts["strata"], index=False)

    asymmetric = summary[summary["method"] == ASYMMETRIC][
        [
            "setup_name",
            "mse_ratio_vs_standard",
            "mse_ratio_ci_low",
            "mse_ratio_ci_high",
            "comparison_flag",
            "mean_q_adapt_raw",
            "mean_q_target",
        ]
    ]
    print("\nAsymmetric guard relative to Standard Hutch++:")
    print(asymmetric.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    for path in artifacts.values():
        print(f"Saved {path}")
    return manifest, trial_frame, summary, strata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        trials_per_method=args.trials,
        dimension=args.dimension,
        budget=args.budget,
        bootstrap_samples=args.bootstrap_samples,
        output_dir=args.output_dir,
    )
