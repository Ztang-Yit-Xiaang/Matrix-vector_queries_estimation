import sys
import numpy as np
import scipy.linalg as la
import pandas as pd
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trace_baseline import (
    MatVecOracle,
    Hutchinson,
    Hutch_pplus,
    Gaussian_Hutch_pplus,
    Adaptive_Hutch_pplus_ModelAveraged,
    Adaptive_Hutch_pplus_SequentialPilot,
    Hutch_pplus_CrossFitting
)

def generate_powerlaw_psd(d, c, rng):
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = (np.arange(1, d + 1, dtype=np.float64)) ** (-float(c))
    A = (Q_orth * eigenvals) @ Q_orth.T
    return A, f"Power-Law (c={c})"

def generate_exponential_psd(d, alpha, rng):
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = np.exp(-float(alpha) * np.arange(1, d + 1, dtype=np.float64))
    A = (Q_orth * eigenvals) @ Q_orth.T
    return A, f"Exponential (alpha={alpha})"

def generate_step_psd(d, r=20, eta=0.01, rng=None):
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = np.ones(d, dtype=np.float64) * float(eta)
    eigenvals[:r] = 1.0
    A = (Q_orth * eigenvals) @ Q_orth.T
    return A, f"Step (r={r}, eta={eta})"

def run_sequential_pilot_benchmark():
    print("==========================================================================", flush=True)
    print("DIRECTION 1 BENCHMARK: SEQUENTIAL ADAPTIVE PILOT STOPPING", flush=True)
    print("==========================================================================", flush=True)

    d = 500
    m = 160
    n_trials = 50
    rng = np.random.default_rng(42)

    setups = [
        generate_powerlaw_psd(d, 2.0, rng),
        generate_powerlaw_psd(d, 0.5, rng),
        generate_exponential_psd(d, 0.05, rng),
        generate_step_psd(d, r=10, eta=0.01, rng=rng),
        generate_step_psd(d, r=20, eta=0.01, rng=rng)
    ]

    estimators = {
        "Hutchinson": lambda o, m, d, rng: Hutchinson(o, m, d, rng=rng),
        "Hutch++ (Standard)": lambda o, m, d, rng: Hutch_pplus(o, m, d, rng=rng),
        "Gaussian-Hutch++": lambda o, m, d, rng: Gaussian_Hutch_pplus(o, m, d, rng=rng),
        "Fixed-Pilot Model-Avg (b=10)": lambda o, m, d, rng: Adaptive_Hutch_pplus_ModelAveraged(o, m, d, b=10, rng=rng),
        "Sequential Pilot (Ours)": lambda o, m, d, rng: Adaptive_Hutch_pplus_SequentialPilot(o, m, d, b_0=8, delta_b=4, rng=rng),
        "Cross-Fitting Reuse (Ours)": lambda o, m, d, rng: Hutch_pplus_CrossFitting(o, m, d, k=50, rng=rng)
    }

    seq_rows = []

    for A, setup_name in setups:
        print(f"\n--- Evaluating Setup: {setup_name} (m={m}, d={d}) ---", flush=True)
        exact_tr = float(np.trace(A))

        for alg_name, alg_fn in estimators.items():
            estimates = []
            final_b_list = []
            oracle = MatVecOracle(A, d=d)

            for t in range(n_trials):
                oracle.reset_query_count()
                trial_rng = np.random.default_rng(1000 + t)

                if "Sequential" in alg_name:
                    est, diag = Adaptive_Hutch_pplus_SequentialPilot(
                        oracle, m, d, b_0=8, delta_b=4, rng=trial_rng, return_diagnostics=True
                    )
                    final_b_list.append(diag["b_final"])
                elif "Fixed-Pilot" in alg_name:
                    est, diag = Adaptive_Hutch_pplus_ModelAveraged(
                        oracle, m, d, b=10, rng=trial_rng, return_diagnostics=True
                    )
                    final_b_list.append(10)
                else:
                    est = alg_fn(oracle, m, d, trial_rng)
                    final_b_list.append(0)

                assert oracle.query_count == m, f"Query count mismatch: {oracle.query_count} != {m}"
                estimates.append(est)

            estimates = np.array(estimates, dtype=np.float64)
            mse_val = float(np.mean((estimates - exact_tr) ** 2))
            rel_errors = np.abs(estimates - exact_tr) / exact_tr
            median_rel_err = float(np.median(rel_errors))
            mean_b = float(np.mean(final_b_list))

            seq_rows.append({
                "setup": setup_name,
                "algorithm": alg_name,
                "mse": mse_val,
                "median_rel_error": median_rel_err,
                "mean_final_b": mean_b
            })

            print(f"  {alg_name:30s} | MSE: {mse_val:.6e} | Mean Pilot b: {mean_b:.1f}", flush=True)

    df_seq = pd.DataFrame(seq_rows)
    out_csv = Path(__file__).resolve().parent.parent / "results" / "sequential_pilot_benchmark_results.csv"
    df_seq.to_csv(out_csv, index=False)

    print("\n==========================================================================", flush=True)
    print(f"SEQUENTIAL PILOT BENCHMARK COMPLETE: SAVED TO {out_csv}", flush=True)
    print("==========================================================================", flush=True)

if __name__ == "__main__":
    run_sequential_pilot_benchmark()
