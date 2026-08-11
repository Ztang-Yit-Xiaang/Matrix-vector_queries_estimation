import csv
import optuna
import numpy as np
from trace_baseline import MatVecOracle

# Suppress detailed Optuna logs for clean console output
optuna.logging.set_verbosity(optuna.logging.WARNING)

def generate_psd_matrix(d, c, seed=42):
    """
    Generate synthetic PSD matrix A = U diag(lambda) U^T where lambda_i = i^(-c).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(X)
    i_vals = np.arange(1, d + 1, dtype=np.float64)
    lambdas = i_vals ** (-c)
    A = Q @ (lambdas[:, None] * Q.T)
    exact_trace = np.sum(lambdas)
    return A, exact_trace

def hutch_pplus_allocation(oracle, k, l, d, rng=None):
    """
    Hutch++ with explicit sketch rank k (m1=k, m2=k) and residual queries l (m3=l).
    Total query count = 2*k + l.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    S = rng.choice([-1.0, 1.0], size=(d, k))
    AS = oracle(S)
    
    Q, _ = np.linalg.qr(AS)
    AQ = oracle(Q)
    
    G = rng.choice([-1.0, 1.0], size=(d, l))
    B_G = G - Q @ (Q.T @ G)  # (I - Q Q^T) G
    ABG = oracle(B_G)
    
    trace_low_rank = np.trace(Q.T @ AQ)
    trace_residual = np.trace(B_G.T @ ABG) / l
    
    return trace_low_rank + trace_residual

def main():
    d = 500
    m_total = 300
    decay_powers = [0.5, 1.0, 1.5, 2.0]
    k_values = list(range(10, 141, 10))
    n_repetitions = 50

    all_csv_rows = []
    best_summary = []
    
    total_query_verifications = 0
    failed_query_verifications = 0

    print("==========================================================================================")
    print("OPTUNA QUERY ALLOCATION OPTIMIZATION FOR HUTCH++")
    print(f"Dimension d={d}, Fixed Budget m={m_total}, Repetitions per trial={n_repetitions}")
    print(f"Search space for sketch rank k: {k_values} with residual queries l = {m_total} - 2*k")
    print("==========================================================================================\n")

    for c in decay_powers:
        mat_seed = int(1000 * c) + 42
        A, exact_trace = generate_psd_matrix(d, c, seed=mat_seed)
        oracle = MatVecOracle(A)

        def objective(trial):
            nonlocal total_query_verifications, failed_query_verifications
            k = trial.suggest_int("k", 10, 140, step=10)
            l = m_total - 2 * k
            
            rel_errors = []
            for rep in range(n_repetitions):
                rep_seed = int(c * 100000) + k * 1000 + rep
                trial_rng = np.random.default_rng(rep_seed)
                
                oracle.reset_query_count()
                est_trace = hutch_pplus_allocation(oracle, k, l, d, rng=trial_rng)
                
                q_count = oracle.query_count
                if q_count != m_total:
                    failed_query_verifications += 1
                    print(f"ERROR: (c={c}, k={k}, l={l}, rep={rep}) query count = {q_count}, expected {m_total}")
                else:
                    total_query_verifications += 1
                assert q_count == m_total, f"Query count mismatch: got {q_count}, expected {m_total}"
                
                rel_err = abs(est_trace - exact_trace) / exact_trace
                rel_errors.append(rel_err)
                
            rel_errors = np.array(rel_errors)
            median_err = np.median(rel_errors)
            q25, q75 = np.percentile(rel_errors, [25, 75])
            iqr_err = q75 - q25
            mean_err = np.mean(rel_errors)
            std_err = np.std(rel_errors)

            trial.set_user_attr("l", l)
            trial.set_user_attr("median_rel_err", median_err)
            trial.set_user_attr("iqr_rel_err", iqr_err)
            trial.set_user_attr("mean_rel_err", mean_err)
            trial.set_user_attr("std_rel_err", std_err)
            
            return median_err

        search_space = {"k": k_values}
        sampler = optuna.samplers.GridSampler(search_space)
        study = optuna.create_study(study_name=f"decay_c_{c}", direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=len(k_values))

        best_k = study.best_params["k"]
        best_l = m_total - 2 * best_k
        best_median_err = study.best_value

        # Find standard Hutch++ trial (k=100, l=100)
        std_trial = [t for t in study.trials if t.params["k"] == 100][0]
        std_median_err = std_trial.user_attrs["median_rel_err"]
        
        reduction_ratio = std_median_err / best_median_err if best_median_err > 0 else 1.0
        pct_reduction = (1.0 - best_median_err / std_median_err) * 100.0 if std_median_err > 0 else 0.0

        best_summary.append({
            "c": c,
            "best_k": best_k,
            "best_l": best_l,
            "best_median_err": best_median_err,
            "std_median_err": std_median_err,
            "reduction_ratio": reduction_ratio,
            "pct_reduction": pct_reduction,
            "exact_trace": exact_trace
        })

        for trial in study.trials:
            k_val = trial.params["k"]
            l_val = trial.user_attrs["l"]
            med_err = trial.user_attrs["median_rel_err"]
            iqr_err = trial.user_attrs["iqr_rel_err"]
            mean_err = trial.user_attrs["mean_rel_err"]
            std_err = trial.user_attrs["std_rel_err"]
            is_best = (k_val == best_k)
            is_std = (k_val == 100)
            rel_ratio_vs_std = std_median_err / med_err if med_err > 0 else 1.0

            all_csv_rows.append({
                "c": c,
                "k": k_val,
                "l": l_val,
                "median_rel_err": med_err,
                "iqr_rel_err": iqr_err,
                "mean_rel_err": mean_err,
                "std_rel_err": std_err,
                "is_optuna_best": is_best,
                "is_standard_hutchpp": is_std,
                "error_reduction_ratio_vs_std": rel_ratio_vs_std,
                "exact_trace": exact_trace
            })

    # Save to CSV
    csv_file = "optuna_allocation_results.csv"
    fieldnames = [
        "c", "k", "l", "median_rel_err", "iqr_rel_err", 
        "mean_rel_err", "std_rel_err", "is_optuna_best", 
        "is_standard_hutchpp", "error_reduction_ratio_vs_std", "exact_trace"
    ]
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_csv_rows:
            writer.writerow(row)

    print("==========================================================================")
    print("QUERY COUNT VERIFICATION SUMMARY")
    print("==========================================================================")
    print(f"Total trials evaluated: {len(all_csv_rows)} (14 rank values per decay * 4 decays)")
    print(f"Total query verifications checked: {total_query_verifications}")
    print(f"Passed oracle.query_count == 300: {total_query_verifications}")
    print(f"Failed query count != 300: {failed_query_verifications}")
    if failed_query_verifications == 0:
        print("--> SUCCESS: For EVERY repetition of EVERY trial, oracle.query_count == 300 holds exactly!\n")
    else:
        print("--> WARNING: Query count mismatch detected in some trials!\n")

    print("=======================================================================================================================")
    print("OPTUNA OPTIMAL ALLOCATION SUMMARY vs STANDARD HUTCH++ (k=100, l=100)")
    print("=======================================================================================================================")
    print(f"{'Decay (c)':<10} | {'Best Split (k*, l*)':<20} | {'Optuna Median Err':<18} | {'Std (100,100) Err':<18} | {'Err Red. Ratio':<15} | {'% Error Red.':<12}")
    print("-----------------------------------------------------------------------------------------------------------------------")
    for s in best_summary:
        split_str = f"({s['best_k']}, {s['best_l']})"
        print(f"{s['c']:<10.1f} | {split_str:<20s} | {s['best_median_err']:<18.4e} | {s['std_median_err']:<18.4e} | {s['reduction_ratio']:<15.3f}x | {s['pct_reduction']:<11.2f}%")
    print("=======================================================================================================================\n")

    print("DETAILED SWEEP RESULTS:")
    print("==========================================================================================")
    print(f"{'c':<6} | {'k':<5} | {'l':<5} | {'Median Rel Err':<16} | {'IQR Rel Err':<16} | {'Status':<15}")
    print("==========================================================================================")
    for r in all_csv_rows:
        status = ""
        if r["is_optuna_best"] and r["is_standard_hutchpp"]:
            status = "Best & Standard"
        elif r["is_optuna_best"]:
            status = "OPTUNA BEST (*)"
        elif r["is_standard_hutchpp"]:
            status = "Standard (100,100)"
        print(f"{r['c']:<6.1f} | {r['k']:<5d} | {r['l']:<5d} | {r['median_rel_err']:<16.4e} | {r['iqr_rel_err']:<16.4e} | {status:<15s}")
    print("==========================================================================================")
    print(f"Results successfully saved to {csv_file}")

if __name__ == "__main__":
    main()
