"""
Benchmark script comparing Adaptive_Hutch_pplus_TwoStageGated vs Standard Hutch++ and Hutchinson.
Evaluates:
1. Smooth power-law spectra (c in {0.5, 1.0, 2.0})
2. Step spectra (r_star in {5, 15}, eta=0.001)
3. Real-world dataset YearPredictionMSD
"""

import os
import sys
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.trace_baseline import (
    MatVecOracle,
    Hutchinson,
    Hutch_pplus,
    Adaptive_Hutch_pplus_TwoStageGated,
)
from src.data_loaders import load_year_prediction

RESULTS_DIR = os.path.join(project_dir, "results")


def generate_synthetic_psd(d: int, family: str, param: float, rng: np.random.Generator):
    if family == "power_law":
        c = param
        evals = np.array([float(i) ** (-c) for i in range(1, d + 1)])
    elif family == "step":
        r_star = int(param)
        eta = 0.001
        evals = np.ones(d) * eta
        evals[:r_star] = 1.0
    else:
        raise ValueError(f"Unknown family {family}")
    
    V, _ = np.linalg.qr(rng.normal(size=(d, d)))
    A = V @ np.diag(evals) @ V.T
    return A, float(np.sum(evals))


def run_benchmark(output_dir=None):
    print("Running TwoStageGated comparative benchmark...")
    rng = np.random.default_rng(2026)
    d = 100
    m_budget = 60
    n_trials = 50

    cases = [
        ("PowerLaw (c=0.5 - Flat)", "power_law", 0.5),
        ("PowerLaw (c=1.0 - Moderate)", "power_law", 1.0),
        ("PowerLaw (c=2.0 - Steep)", "power_law", 2.0),
        ("StepSpectrum (r*=5, eta=1e-3)", "step", 5.0),
        ("StepSpectrum (r*=15, eta=1e-3)", "step", 15.0),
    ]

    records = []

    for name, family, param in cases:
        A, exact_tr = generate_synthetic_psd(d, family, param, rng)
        print(f"Testing {name} (exact trace = {exact_tr:.4f})...")

        for trial in range(n_trials):
            # 1. Hutchinson
            o_hutch = MatVecOracle(A)
            est_hutch = Hutchinson(o_hutch, m=m_budget, d=d, rng=rng)
            records.append({
                "setup": name,
                "trial": trial,
                "algorithm": "Hutchinson",
                "rel_err": abs(est_hutch - exact_tr) / exact_tr,
                "gated_triggered": False,
            })

            # 2. Standard Hutch++ (q_0 = m // 3 = 20)
            o_hpp = MatVecOracle(A)
            est_hpp = Hutch_pplus(o_hpp, m=m_budget, d=d, rng=rng)
            records.append({
                "setup": name,
                "trial": trial,
                "algorithm": "Standard Hutch++",
                "rel_err": abs(est_hpp - exact_tr) / exact_tr,
                "gated_triggered": False,
            })

            # 3. TwoStageGated
            o_gated = MatVecOracle(A)
            est_gated, diag = Adaptive_Hutch_pplus_TwoStageGated(
                o_gated, m=m_budget, d=d, b_0=8, tau_gap=1.5, p_oversample=2, rng=rng, return_diagnostics=True
            )
            records.append({
                "setup": name,
                "trial": trial,
                "algorithm": "TwoStageGated (Ours)",
                "rel_err": abs(est_gated - exact_tr) / exact_tr,
                "gated_triggered": diag["is_gated_trigger"],
            })

    df = pd.DataFrame(records)

    # Summary table
    summary = df.groupby(["setup", "algorithm"])["rel_err"].agg(
        median="median",
        iqr=lambda x: np.percentile(x, 75) - np.percentile(x, 25),
        mean="mean"
    ).reset_index()
    print("\n=== Benchmark Summary ===")
    print(summary.to_string(index=False))

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "twostage_gated_benchmark_results.csv"), index=False)
        summary.to_csv(os.path.join(output_dir, "twostage_gated_benchmark_summary.csv"), index=False)
    return df, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    run_benchmark(output_dir=parser.parse_args().output_dir)
