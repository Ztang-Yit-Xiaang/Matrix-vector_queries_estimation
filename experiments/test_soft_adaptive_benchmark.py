import numpy as np
import scipy.linalg as la
import pandas as pd
from pathlib import Path
from trace_baseline import (
    MatVecOracle,
    Hutchinson,
    Hutch_pplus,
    Gaussian_Hutch_pplus,
    Adaptive_Hutch_pplus_RademacherResidual,
    Adaptive_Hutch_pplus_Soft_RademacherResidual,
    _rank_aware_qr
)

def generate_powerlaw_psd(d, c, rng):
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = (np.arange(1, d + 1, dtype=np.float64)) ** (-float(c))
    A = (Q_orth * eigenvals) @ Q_orth.T
    return A, "Power-Law (c=" + str(c) + ")"

def generate_exponential_psd(d, alpha, rng):
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = np.exp(-float(alpha) * np.arange(1, d + 1, dtype=np.float64))
    A = (Q_orth * eigenvals) @ Q_orth.T
    return A, "Exponential (alpha=" + str(alpha) + ")"

def generate_step_psd(d, r=20, eta=0.01, rng=None):
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = np.ones(d, dtype=np.float64) * float(eta)
    eigenvals[:r] = 1.0
    A = (Q_orth * eigenvals) @ Q_orth.T
    return A, f"Step (r={r}, eta={eta})"

def generate_elbow_psd(d, r_elbow=10, eta=0.01, rng=None):
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = (np.arange(1, d + 1, dtype=np.float64)) ** (-2.0)
    eigenvals[r_elbow:] = float(eta)
    A = (Q_orth * eigenvals) @ Q_orth.T
    return A, f"Elbow (r={r_elbow}, eta={eta})"

def evaluate_actual_hutch_mse(A, q, m, n_trials=50, rng_seed=42):
    d = A.shape[0]
    exact_tr = np.trace(A)
    errors = []
    
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
        
        errors.append((tr_est - exact_tr) ** 2)
        
    return float(np.mean(errors))

def run_soft_adaptive_benchmark():
    print("==========================================================================", flush=True)
    print("SOFT ADAPTIVE HUTCH++ BENCHMARK & REGRET ANALYSIS", flush=True)
    print("==========================================================================", flush=True)
    
    d = 500
    b = 10
    budgets = [40, 80, 160, 320]
    n_trials = 50
    
    rng = np.random.default_rng(42)
    
    # 4 Expanded Synthetic Matrix Families
    dataset_generators = [
        generate_powerlaw_psd(d, 0.5, rng),
        generate_powerlaw_psd(d, 2.0, rng),
        generate_exponential_psd(d, 0.05, rng),
        generate_step_psd(d, r=20, eta=0.01, rng=rng),
        generate_elbow_psd(d, r_elbow=10, eta=0.01, rng=rng)
    ]
    
    algorithms = {
        "Hutchinson": Hutchinson,
        "Hutch++ (Standard)": Hutch_pplus,
        "Gaussian-Hutch++": Gaussian_Hutch_pplus,
        "Hard-Adaptive-Rademacher": Adaptive_Hutch_pplus_RademacherResidual,
        "Soft-Adaptive-Rademacher": Adaptive_Hutch_pplus_Soft_RademacherResidual
    }
    
    results = []
    
    for A, ds_name in dataset_generators:
        exact_tr = float(np.trace(A))
        fro_norm = float(la.norm(A, ord='fro'))
        r_eff = (exact_tr ** 2) / (fro_norm ** 2 + 1e-12)
        
        print(f"\n--- Testing Dataset Family: {ds_name} (d={d}, tr(A)={exact_tr:.2f}, r_eff={r_eff:.2f}) ---", flush=True)
        oracle = MatVecOracle(A, d=d)
        
        for m in budgets:
            q_max = min(d, (m - 2) // 2)
            
            # Precompute exhaustive MSE mapping over candidate q in [1, q_max]
            mse_q_map = {}
            for q_cand in range(1, q_max + 1):
                mse_q_map[q_cand] = evaluate_actual_hutch_mse(A, q_cand, m, n_trials=n_trials)
                
            q_star = min(mse_q_map, key=mse_q_map.get)
            mse_q_star = mse_q_map[q_star]
            
            # Restricted oracle choice q_b_star in [b, q_max]
            mse_q_b_map = {q: mse_q_map[q] for q in range(b, q_max + 1) if q in mse_q_map}
            q_b_star = min(mse_q_b_map, key=mse_q_b_map.get) if len(mse_q_b_map) > 0 else q_star
            mse_q_b_star = mse_q_map[q_b_star]
            
            # Commitment Regret
            R_commitment = (mse_q_b_star / (mse_q_star + 1e-12)) - 1.0
            
            for alg_idx, (alg_name, alg_fn) in enumerate(algorithms.items()):
                sq_errors = []
                rel_errors = []
                
                for trial in range(n_trials):
                    trial_seed = abs(hash(ds_name)) % 100000 + m * 1000 + alg_idx * 10000 + trial
                    trial_rng = np.random.default_rng(trial_seed)
                    
                    oracle.reset_query_count()
                    est = alg_fn(oracle, m, d, rng=trial_rng)
                    
                    assert oracle.query_count == m, f"{alg_name} query count mismatch: {oracle.query_count} != {m}"
                    
                    sq_err = (est - exact_tr) ** 2
                    rel_err = abs(est - exact_tr) / exact_tr
                    sq_errors.append(sq_err)
                    rel_errors.append(rel_err)
                    
                mse_val = float(np.mean(sq_errors))
                median_rel_err = float(np.median(rel_errors))
                
                # Regret Metrics
                R_alloc = (mse_val / (mse_q_star + 1e-12)) - 1.0
                R_decision = (mse_val / (mse_q_b_star + 1e-12)) - 1.0
                
                results.append({
                    "dataset": ds_name,
                    "d": d,
                    "effective_rank": r_eff,
                    "m": m,
                    "algorithm": alg_name,
                    "mse": mse_val,
                    "median_rel_err": median_rel_err,
                    "q_star": q_star,
                    "q_b_star": q_b_star,
                    "R_alloc": R_alloc,
                    "R_commitment": R_commitment,
                    "R_decision": R_decision
                })
                
                print(f"  m={m:3d} | {alg_name:30s} | MSE: {mse_val:.6e} | RelErr: {median_rel_err:.4e} | R_alloc: {R_alloc:+.2f}", flush=True)
                
    df_res = pd.DataFrame(results)
    out_csv = Path(__file__).parent / "soft_adaptive_benchmark_results.csv"
    df_res.to_csv(out_csv, index=False)
    
    print("\n==========================================================================", flush=True)
    print("SUMMARY TABLE (MSE & ALLOCATION REGRET)", flush=True)
    print("==========================================================================", flush=True)
    pivoted_mse = df_res.pivot(index=["dataset", "m"], columns="algorithm", values="mse")
    print("\n--- MSE COMPARISON ---")
    print(pivoted_mse.to_string(), flush=True)
    
    pivoted_regret = df_res.pivot(index=["dataset", "m"], columns="algorithm", values="R_alloc")
    print("\n--- ALLOCATION REGRET (R_alloc) COMPARISON ---")
    print(pivoted_regret.to_string(), flush=True)
    
    print("\n==========================================================================", flush=True)
    print(f"SOFT ADAPTIVE BENCHMARK COMPLETED AND SAVED TO {out_csv}", flush=True)
    print("==========================================================================", flush=True)

if __name__ == "__main__":
    run_soft_adaptive_benchmark()
