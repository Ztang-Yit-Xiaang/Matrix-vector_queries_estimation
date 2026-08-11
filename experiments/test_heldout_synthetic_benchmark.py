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

def generate_step_psd(d, r=5, eta=0.01, rng=None):
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = np.ones(d, dtype=np.float64) * float(eta)
    eigenvals[:r] = 1.0
    A = (Q_orth * eigenvals) @ Q_orth.T
    return A, f"Step (r={r}, eta={eta})"

def generate_elbow_psd(d, r_elbow=5, eta=0.005, rng=None):
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = (np.arange(1, d + 1, dtype=np.float64)) ** (-1.5)
    eigenvals[r_elbow:] = float(eta)
    A = (Q_orth * eigenvals) @ Q_orth.T
    return A, f"Elbow (r={r_elbow}, eta={eta})"

def generate_lognormal_psd(d, sigma=1.0, rng=None):
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    raw_vals = np.exp(-sigma * np.sort(rng.normal(size=d)))
    eigenvals = raw_vals / np.max(raw_vals)
    A = (Q_orth * eigenvals) @ Q_orth.T
    return A, f"Log-Normal Misspecified (sigma={sigma})"

def evaluate_actual_hutch_mse(A, q, m, n_trials=50, rng_seed=42):
    d = A.shape[0]
    exact_tr = float(np.trace(A))
    sq_errors = []
    
    for t in range(n_trials):
        trial_rng = np.random.default_rng(rng_seed + t * 1000)
        S = trial_rng.choice([-1.0, 1.0], size=(d, q))
        W = A @ S
        scale_W = float(la.norm(W, ord='fro'))
        Q, r_actual = _rank_aware_qr(W, reference_scale=scale_W)
        
        AQ = A @ Q if r_actual > 0 else np.empty((d, 0))
        l_eff = m - q - r_actual
        if l_eff < 1:
            l_eff = 1
            
        G = trial_rng.choice([-1.0, 1.0], size=(d, l_eff))
        B_G = G - Q @ (Q.T @ G) if r_actual > 0 else G
        ABG = A @ B_G
        
        tr_low = float(np.sum(Q * AQ)) if r_actual > 0 else 0.0
        tr_res = float(np.sum(B_G * ABG)) / l_eff
        tr_est = tr_low + tr_res
        
        sq_errors.append((tr_est - exact_tr) ** 2)
        
    return float(np.mean(sq_errors))

def run_heldout_benchmark():
    print("==========================================================================", flush=True)
    print("PARTS F, G, H: HELD-OUT SYNTHETIC BENCHMARK & ALLOCATION REGRET", flush=True)
    print("==========================================================================", flush=True)

    d = 500
    m = 160
    b = 10
    n_trials = 50
    rng = np.random.default_rng(42)

    heldout_families = [
        generate_powerlaw_psd(d, 0.3, rng),
        generate_powerlaw_psd(d, 0.7, rng),
        generate_powerlaw_psd(d, 1.2, rng),
        generate_powerlaw_psd(d, 1.7, rng),
        generate_powerlaw_psd(d, 2.5, rng),
        generate_powerlaw_psd(d, 3.0, rng),
        generate_exponential_psd(d, 0.02, rng),
        generate_exponential_psd(d, 0.08, rng),
        generate_exponential_psd(d, 0.15, rng),
        generate_step_psd(d, r=5, eta=0.01, rng=rng),
        generate_step_psd(d, r=15, eta=0.01, rng=rng),
        generate_elbow_psd(d, r_elbow=5, eta=0.005, rng=rng),
        generate_lognormal_psd(d, sigma=1.0, rng=rng)
    ]

    estimators = {
        "Hutchinson": lambda o, m, d, rng: Hutchinson(o, m, d, rng=rng),
        "Hutch++ (Standard)": lambda o, m, d, rng: Hutch_pplus(o, m, d, rng=rng),
        "Gaussian-Hutch++": lambda o, m, d, rng: Gaussian_Hutch_pplus(o, m, d, rng=rng),
        "Hard Single-c": lambda o, m, d, rng: Adaptive_Hutch_pplus_RademacherResidual(o, m, d, b=b, rng=rng),
        "Soft Single-c": lambda o, m, d, rng: Adaptive_Hutch_pplus_Soft_RademacherResidual(o, m, d, b=b, rng=rng),
        "Model-Averaged (Ours)": lambda o, m, d, rng: Adaptive_Hutch_pplus_ModelAveraged(o, m, d, b=b, use_safety_shrinkage=True, rng=rng)
    }

    benchmark_rows = []

    q_max = min(d, (m - 2) // 2)

    for A, family_name in heldout_families:
        print(f"\n--- Benchmark Family: {family_name} (m={m}, d={d}) ---", flush=True)
        exact_tr = float(np.trace(A))

        # Independent Monte Carlo Estimation of Empirical Oracle Allocation q*
        candidate_q = np.arange(b, q_max + 1)
        mse_map = {}
        for q_cand in candidate_q:
            mse_map[q_cand] = evaluate_actual_hutch_mse(A, q_cand, m, n_trials=50, rng_seed=42 + q_cand)
        q_star_emp = min(mse_map, key=mse_map.get)
        mse_star_emp = mse_map[q_star_emp]

        # Commitment Regret (from pilot floor b)
        mse_q_b = mse_map.get(b, mse_star_emp)
        r_commitment = float((mse_q_b / (mse_star_emp + 1e-15)) - 1.0)

        for alg_name, alg_fn in estimators.items():
            estimates = []
            q_selected_list = []
            weights_list = []
            gamma_list = []
            oracle = MatVecOracle(A, d=d)

            for t in range(n_trials):
                oracle.reset_query_count()
                trial_rng = np.random.default_rng(20000 + t)

                if "Model-Averaged" in alg_name:
                    est, diag = Adaptive_Hutch_pplus_ModelAveraged(
                        oracle, m, d, b=b, use_safety_shrinkage=True, rng=trial_rng, return_diagnostics=True
                    )
                    q_selected_list.append(diag["q_target"])
                    weights_list.append(diag["weights"])
                    gamma_list.append(diag["gamma"])
                else:
                    est = alg_fn(oracle, m, d, trial_rng)
                    q_selected_list.append(m // 3 if "Hutch++" in alg_name else b)

                assert oracle.query_count == m, f"Query count mismatch: {oracle.query_count} != {m}"
                estimates.append(est)

            estimates = np.array(estimates, dtype=np.float64)
            mse_val = float(np.mean((estimates - exact_tr) ** 2))
            rel_errors = np.abs(estimates - exact_tr) / exact_tr
            median_rel_err = float(np.median(rel_errors))
            iqr_rel_err = float(np.percentile(rel_errors, 75) - np.percentile(rel_errors, 25))
            estimated_bias = float(np.mean(estimates) - exact_tr)

            q_final_mean = float(np.mean(q_selected_list))
            q_final_std = float(np.std(q_selected_list))

            # Allocation & Decision Regret
            closest_q = int(round(q_final_mean))
            mse_selected = mse_map.get(closest_q, mse_val)
            r_alloc = float((mse_selected / (mse_star_emp + 1e-15)) - 1.0)
            r_decision = float((mse_selected / (mse_q_b + 1e-15)) - 1.0)

            w_pow_avg = float(np.mean([w["power"] for w in weights_list])) if weights_list else 0.0
            w_exp_avg = float(np.mean([w["exp"] for w in weights_list])) if weights_list else 0.0
            w_step_avg = float(np.mean([w["step"] for w in weights_list])) if weights_list else 0.0
            gamma_avg = float(np.mean(gamma_list)) if gamma_list else 0.0

            benchmark_rows.append({
                "family": family_name,
                "algorithm": alg_name,
                "mse": mse_val,
                "median_rel_error": median_rel_err,
                "iqr_rel_error": iqr_rel_err,
                "estimated_bias": estimated_bias,
                "q_emp_target": q_star_emp,
                "q_final_mean": q_final_mean,
                "q_final_std": q_final_std,
                "r_alloc": r_alloc,
                "r_commitment": r_commitment,
                "r_decision": r_decision,
                "w_power_avg": w_pow_avg,
                "w_exp_avg": w_exp_avg,
                "w_step_avg": w_step_avg,
                "gamma_avg": gamma_avg
            })

            print(f"  {alg_name:25s} | MSE: {mse_val:.6e} | Mean q: {q_final_mean:.1f} | Regret R_alloc: {r_alloc:.4f}", flush=True)

    df_bench = pd.DataFrame(benchmark_rows)
    out_csv = Path(__file__).resolve().parent.parent / "results" / "heldout_synthetic_benchmark_results.csv"
    df_bench.to_csv(out_csv, index=False)

    print("\n==========================================================================", flush=True)
    print(f"HELDOUT BENCHMARK COMPLETE: SAVED TO {out_csv}", flush=True)
    print("==========================================================================", flush=True)

if __name__ == "__main__":
    run_heldout_benchmark()
