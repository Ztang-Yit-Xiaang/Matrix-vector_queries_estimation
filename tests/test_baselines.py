import numpy as np
from trace_baseline import (
    MatVecOracle,
    Hutchinson,
    Hutch_pplus,
    Gaussian_Hutch_pplus,
    NA_Hutch_pplus
)

def run_tests():
    print("==========================================")
    print("1. VALIDATING QUERY COUNTS")
    print("==========================================")
    d = 100
    rng = np.random.default_rng(42)
    M = rng.normal(size=(d, d))
    A_mat = M.T @ M  # PSD matrix
    exact_trace = np.trace(A_mat)
    
    oracle = MatVecOracle(A_mat)
    
    m_test_values = [30, 60, 120, 300]
    estimators = {
        "Hutchinson": Hutchinson,
        "Hutch++": Hutch_pplus,
        "Gaussian-Hutch++": Gaussian_Hutch_pplus,
        "NA-Hutch++": NA_Hutch_pplus
    }
    
    all_query_counts_pass = True
    for name, est_fn in estimators.items():
        for m in m_test_values:
            oracle.reset_query_count()
            _ = est_fn(oracle, m, d, rng=rng)
            queries = oracle.query_count
            if queries != m:
                print(f"FAIL: {name} (m={m}): expected {m} queries, got {queries}")
                all_query_counts_pass = False
            else:
                print(f"PASS: {name} (m={m}): query count = {queries}")
                
    if all_query_counts_pass:
        print("--> ALL QUERY COUNTS VERIFIED EXACTLY MATCHING BUDGET m!\n")

    print("==========================================")
    print("2. VALIDATING UNBIASEDNESS & RELATIVE ERROR")
    print("==========================================")
    n_trials = 100
    m = 60
    print(f"Matrix dim d={d}, exact trace={exact_trace:.4f}, budget m={m}, trials={n_trials}")
    
    for name, est_fn in estimators.items():
        estimates = []
        for trial in range(n_trials):
            trial_rng = np.random.default_rng(1000 + trial)
            oracle.reset_query_count()
            est = est_fn(oracle, m, d, rng=trial_rng)
            estimates.append(est)
        
        estimates = np.array(estimates)
        mean_est = np.mean(estimates)
        rel_bias = (mean_est - exact_trace) / exact_trace
        rel_errors = np.abs(estimates - exact_trace) / exact_trace
        median_rel_err = np.median(rel_errors)
        q25, q75 = np.percentile(rel_errors, [25, 75])
        iqr = q75 - q25
        
        print(f"Estimator: {name:18s} | Mean Rel. Bias: {rel_bias:+.4e} | Median Rel. Err: {median_rel_err:.4e} (IQR: {iqr:.4e})")

if __name__ == "__main__":
    run_tests()
