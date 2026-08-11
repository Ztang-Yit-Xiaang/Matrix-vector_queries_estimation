import numpy as np
import pandas as pd
from pathlib import Path
from trace_baseline import (
    MatVecOracle,
    Hutchinson,
    Hutch_pplus,
    Gaussian_Hutch_pplus,
    Adaptive_Hutch_pplus_GaussianResidual,
    Adaptive_Hutch_pplus_RademacherResidual
)
from data_loaders import load_synthetic_decay, load_year_prediction, load_wiki_vote

def run_adaptive_tests():
    print("==========================================================================", flush=True)
    print("1. VALIDATING ADAPTIVE QUERY COUNTS & DIAGNOSTICS", flush=True)
    print("==========================================================================", flush=True)
    d = 100
    rng = np.random.default_rng(42)
    M = rng.normal(size=(d, d))
    A_mat = M.T @ M
    oracle = MatVecOracle(A_mat)
    
    m_values = [30, 60, 120, 300]
    all_query_counts_pass = True
    
    for m in m_values:
        for mode_name, mode_fn in [("Adaptive-Hutch++-GaussianResidual", Adaptive_Hutch_pplus_GaussianResidual), 
                                   ("Adaptive-Hutch++-RademacherResidual", Adaptive_Hutch_pplus_RademacherResidual)]:
            oracle.reset_query_count()
            est, diag = mode_fn(oracle, m, d, b=10, rng=rng, return_diagnostics=True)
            q_count = oracle.query_count
            if q_count != m:
                print(f"FAIL: {mode_name} (m={m}): expected {m} queries, got {q_count}", flush=True)
                all_query_counts_pass = False
            else:
                c_str = f"{diag['c_hat']:.2f}" if diag['c_hat'] is not None else "N/A"
                print(f"PASS: {mode_name} (m={m}): queries={q_count} | c_hat={c_str}, q_target={diag['q_target']}, ell_eff={diag['ell_eff']}", flush=True)
                
    if all_query_counts_pass:
        print("--> SUCCESS: 100% OF ADAPTIVE TRIALS SATISFY oracle.query_count == m EXACTLY!\n", flush=True)

    print("==========================================================================", flush=True)
    print("2. REAL-WORLD & SYNTHETIC BENCHMARK COMPARISON", flush=True)
    print("==========================================================================", flush=True)
    print("Loading benchmark datasets...", flush=True)
    datasets = [
        load_synthetic_decay(d=500, c=0.5),
        load_synthetic_decay(d=500, c=2.0),
        load_year_prediction(nrows=25000),
        load_wiki_vote()
    ]
    print("Datasets successfully loaded.", flush=True)
    
    budgets = [40, 80, 160, 320]
    n_trials = 50
    
    algorithms = {
        "Hutchinson": Hutchinson,
        "Hutch++": Hutch_pplus,
        "Gaussian-Hutch++": Gaussian_Hutch_pplus,
        "Adaptive-Hutch++-GaussianResidual": Adaptive_Hutch_pplus_GaussianResidual,
        "Adaptive-Hutch++-RademacherResidual": Adaptive_Hutch_pplus_RademacherResidual
    }
    
    results = []
    total_trials_checked = 0
    
    for ds in datasets:
        reff_str = f"{ds.effective_rank:.2f}" if ds.effective_rank is not None else "N/A"
        print(f"\n--- Testing Dataset: {ds.name} (d={ds.d}, tr(A)={ds.exact_trace:.2f}, r_eff={reff_str}) ---", flush=True)
        oracle = MatVecOracle(ds.matvec_fn, d=ds.d)
        
        for m in budgets:
            for alg_idx, (alg_name, alg_fn) in enumerate(algorithms.items()):
                rel_errors = []
                for trial in range(n_trials):
                    trial_seed = abs(hash(ds.name)) % 100000 + m * 1000 + alg_idx * 10000 + trial
                    trial_rng = np.random.default_rng(trial_seed)
                    
                    oracle.reset_query_count()
                    est = alg_fn(oracle, m, ds.d, rng=trial_rng)
                    
                    # Enforce strict query budget check
                    assert oracle.query_count == m, f"{alg_name} query count mismatch: {oracle.query_count} != {m}"
                    total_trials_checked += 1
                    
                    rel_err = abs(est - ds.exact_trace) / ds.exact_trace
                    rel_errors.append(rel_err)
                    
                rel_errors = np.array(rel_errors)
                median_err = np.median(rel_errors)
                q25, q75 = np.percentile(rel_errors, [25, 75])
                
                results.append({
                    "dataset": ds.name,
                    "d": ds.d,
                    "effective_rank": ds.effective_rank,
                    "m": m,
                    "algorithm": alg_name,
                    "median_rel_err": median_err,
                    "iqr_rel_err": q75 - q25
                })
                print(f"  m={m:3d} | {alg_name:35s} | Median Rel Err: {median_err:.6e} | IQR: {(q75 - q25):.6e}", flush=True)
                
    print(f"\nVerified query compliance oracle.query_count == m across all {total_trials_checked} benchmark trials.", flush=True)
    df_res = pd.DataFrame(results)
    out_csv = Path(__file__).parent / "adaptive_benchmark_results.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\n==========================================================================", flush=True)
    print("SUMMARY TABLE (MEDIAN RELATIVE ERROR)", flush=True)
    print("==========================================================================", flush=True)
    pivoted = df_res.pivot(index=["dataset", "m"], columns="algorithm", values="median_rel_err")
    print(pivoted.to_string(), flush=True)
    print("\n==========================================================================", flush=True)
    print(f"BENCHMARK COMPLETED AND SAVED TO {out_csv}", flush=True)
    print("==========================================================================", flush=True)

if __name__ == "__main__":
    run_adaptive_tests()

