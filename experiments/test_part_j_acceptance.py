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
    Adaptive_Hutch_pplus_RademacherResidual,
    Adaptive_Hutch_pplus_Soft_RademacherResidual,
    Adaptive_Hutch_pplus_ModelAveraged,
    _rank_aware_qr
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

def evaluate_estimator_mse(A, estimator_fn, m, d, n_trials=50):
    exact_tr = float(np.trace(A))
    sq_errors = []
    oracle = MatVecOracle(A, d=d)
    
    for t in range(n_trials):
        oracle.reset_query_count()
        trial_rng = np.random.default_rng(10000 + t)
        est = estimator_fn(oracle, m, d, rng=trial_rng)
        assert oracle.query_count == m, f"Query count mismatch: {oracle.query_count} != {m}"
        sq_errors.append((est - exact_tr) ** 2)
        
    return float(np.mean(sq_errors))

def run_acceptance_test():
    print("==========================================================================", flush=True)
    print("PART J: FOCUSED 3-CASE IMMEDIATE ACCEPTANCE BENCHMARK", flush=True)
    print("==========================================================================", flush=True)

    d = 500
    m = 160
    b = 10
    n_trials = 50
    rng = np.random.default_rng(42)

    # 3 Focused Cases
    cases = [
        ("Case 1: Power-Law (c=2.0)", generate_powerlaw_psd(d, 2.0, rng)[0], 62),
        ("Case 2: Exponential (alpha=0.05)", generate_exponential_psd(d, 0.05, rng)[0], 68),
        ("Case 3: Step Spectrum (r=20, eta=0.01)", generate_step_psd(d, r=20, eta=0.01, rng=rng)[0], 35)
    ]

    estimators = {
        "Hutchinson": Hutchinson,
        "Hutch++ (Standard)": Hutch_pplus,
        "Gaussian-Hutch++": Gaussian_Hutch_pplus,
        "Hard Single-c": Adaptive_Hutch_pplus_RademacherResidual,
        "Soft Single-c": Adaptive_Hutch_pplus_Soft_RademacherResidual,
        "Model-Averaged (Ours)": Adaptive_Hutch_pplus_ModelAveraged
    }

    acceptance_results = []

    for case_name, A, q_emp_target in cases:
        print(f"\n--- {case_name} (m={m}, d={d}, Target Empirical q*={q_emp_target}) ---", flush=True)
        oracle = MatVecOracle(A, d=d)

        # Inspect Diagnostics for Model-Averaged Estimator
        oracle.reset_query_count()
        _, diag = Adaptive_Hutch_pplus_ModelAveraged(
            oracle, m, d, b=b, rng=np.random.default_rng(42), return_diagnostics=True
        )

        weights = diag["weights"]
        gamma = diag["gamma"]
        q_adapt = diag["q_adapt"]
        q_final = diag["q_target"]
        q_0 = diag["q_0"]

        print(f"  Diagnostics for Model-Averaged Estimator:", flush=True)
        print(f"    - Model Weights: Power={weights['power']:.4f}, Exp={weights['exp']:.4f}, Step={weights['step']:.4f}", flush=True)
        print(f"    - Predicted q_adapt = {q_adapt} | Standard q_0 = {q_0} | Confidence gamma = {gamma:.4f}", flush=True)
        print(f"    - Final Allocation q_final = {q_final} | Target Empirical q* = {q_emp_target}", flush=True)

        case_row = {"case": case_name, "q_emp_target": q_emp_target, "q_final": q_final, "gamma": gamma, "w_power": weights["power"], "w_exp": weights["exp"], "w_step": weights["step"]}

        for alg_name, alg_fn in estimators.items():
            mse_val = evaluate_estimator_mse(A, alg_fn, m, d, n_trials=n_trials)
            case_row[alg_name] = mse_val
            print(f"    {alg_name:30s} | MSE: {mse_val:.6e}", flush=True)

        acceptance_results.append(case_row)

    df_acc = pd.DataFrame(acceptance_results)
    out_csv = Path(__file__).resolve().parent.parent / "results" / "part_j_acceptance_results.csv"
    df_acc.to_csv(out_csv, index=False)

    print("\n==========================================================================", flush=True)
    print("ACCEPTANCE TEST SUMMARY TABLE", flush=True)
    print("==========================================================================", flush=True)
    print(df_acc.to_string(), flush=True)
    print("\n==========================================================================", flush=True)
    print(f"PART J ACCEPTANCE TEST COMPLETE: SAVED TO {out_csv}", flush=True)
    print("==========================================================================", flush=True)

if __name__ == "__main__":
    run_acceptance_test()
