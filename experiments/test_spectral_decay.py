import csv
import numpy as np
from trace_baseline import (
    MatVecOracle,
    Hutchinson,
    Hutch_pplus,
    Gaussian_Hutch_pplus,
    NA_Hutch_pplus
)

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

def main():
    d = 500
    decay_powers = [0.5, 1.0, 1.5, 2.0]
    budgets = [20, 40, 80, 160, 320]
    n_trials = 50

    algorithms = {
        "Hutchinson": Hutchinson,
        "Hutch++": Hutch_pplus,
        "Gaussian-Hutch++": Gaussian_Hutch_pplus,
        "NA-Hutch++": NA_Hutch_pplus
    }

    results = []
    total_query_verifications = 0
    failed_query_verifications = 0

    print("==========================================================================")
    print("RUNNING SPECTRAL DECAY EXPERIMENTS")
    print(f"Dimension d={d}, Decay powers c={decay_powers}, Budgets m={budgets}, Trials={n_trials}")
    print("==========================================================================\n")

    for c in decay_powers:
        mat_seed = int(1000 * c) + 42
        A, exact_trace = generate_psd_matrix(d, c, seed=mat_seed)
        oracle = MatVecOracle(A)

        for m in budgets:
            for alg_idx, (alg_name, alg_fn) in enumerate(algorithms.items()):
                rel_errors = []
                for trial in range(n_trials):
                    trial_seed = int(c * 100000) + m * 1000 + alg_idx * 10000 + trial
                    trial_rng = np.random.default_rng(trial_seed)

                    oracle.reset_query_count()
                    est_trace = alg_fn(oracle, m, d, rng=trial_rng)
                    
                    q_count = oracle.query_count
                    if q_count != m:
                        failed_query_verifications += 1
                        print(f"ERROR: {alg_name} (c={c}, m={m}, trial={trial}) used {q_count} queries, expected {m}")
                    else:
                        total_query_verifications += 1

                    rel_err = abs(est_trace - exact_trace) / exact_trace
                    rel_errors.append(rel_err)

                rel_errors = np.array(rel_errors)
                median_err = np.median(rel_errors)
                q25, q75 = np.percentile(rel_errors, [25, 75])
                iqr_err = q75 - q25

                results.append({
                    "c": c,
                    "m": m,
                    "algorithm": alg_name,
                    "median_rel_err": median_err,
                    "iqr_rel_err": iqr_err,
                    "exact_trace": exact_trace
                })

    csv_file = "spectral_decay_results.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["c", "m", "algorithm", "median_rel_err", "iqr_rel_err", "exact_trace"])
        for r in results:
            writer.writerow([r["c"], r["m"], r["algorithm"], r["median_rel_err"], r["iqr_rel_err"], r["exact_trace"]])

    print("==========================================================================")
    print("QUERY COUNT VERIFICATION SUMMARY")
    print("==========================================================================")
    print(f"Total trials checked: {total_query_verifications + failed_query_verifications}")
    print(f"Passed query count == m: {total_query_verifications}")
    print(f"Failed query count != m: {failed_query_verifications}")
    if failed_query_verifications == 0:
        print("--> SUCCESS: For EVERY trial, oracle.query_count == m holds exactly!\n")
    else:
        print("--> WARNING: Query count mismatch detected in some trials!\n")

    print("==========================================================================================")
    print(f"{'c':<6} | {'m':<6} | {'Algorithm':<18} | {'Median Rel Err':<16} | {'IQR Rel Err':<16}")
    print("==========================================================================================")
    for r in results:
        print(f"{r['c']:<6.1f} | {r['m']:<6d} | {r['algorithm']:<18s} | {r['median_rel_err']:<16.4e} | {r['iqr_rel_err']:<16.4e}")
    print("==========================================================================================")
    print(f"Results successfully saved to {csv_file}")

if __name__ == "__main__":
    main()
