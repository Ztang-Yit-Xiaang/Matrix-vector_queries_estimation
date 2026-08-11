import numpy as np
import pandas as pd
import time
from pathlib import Path
from trace_baseline import MatVecOracle, Hutchinson, Hutch_pplus, Gaussian_Hutch_pplus
from data_loaders import load_year_prediction, load_wiki_vote

def main():
    print("=" * 80)
    print("RUNNING REAL-WORLD MATRIX TRACE ESTIMATION BENCHMARK")
    print("=" * 80)

    # 1. Load Datasets
    print("\n[1/3] Loading real-world datasets...")
    datasets = [
        load_year_prediction(nrows=50000),
        load_wiki_vote()
    ]
    for ds in datasets:
        print("-" * 50)
        print(ds.summary())

    # 2. Experiment Setup
    algorithms = {
        "Hutchinson": Hutchinson,
        "Hutch_pplus": Hutch_pplus,
        "Gaussian_Hutch_pplus": Gaussian_Hutch_pplus
    }
    budgets = [10, 20, 40, 80, 160, 320]
    num_trials = 50
    base_seed = 1000

    results_rows = []
    total_runs = len(datasets) * len(algorithms) * len(budgets) * num_trials
    run_counter = 0

    print(f"\n[2/3] Executing {total_runs} estimation runs (50 trials per setup)...")
    start_time = time.time()

    for ds in datasets:
        print(f"\nEvaluating dataset: {ds.name} (d={ds.d}, exact_trace={ds.exact_trace:.4f})")
        for alg_name, alg_fn in algorithms.items():
            for m in budgets:
                rel_errors = []
                for trial in range(num_trials):
                    seed = base_seed + trial
                    rng = np.random.default_rng(seed)
                    
                    oracle = MatVecOracle(ds.matvec_fn, d=ds.d)
                    
                    # Run estimation
                    est_trace = alg_fn(oracle, m, ds.d, rng=rng)
                    
                    # Verification check as required
                    assert oracle.query_count == m, (
                        f"Query count mismatch! Expected {m}, got {oracle.query_count} "
                        f"for {alg_name} on {ds.name}"
                    )
                    
                    rel_err = abs(est_trace - ds.exact_trace) / ds.exact_trace
                    rel_errors.append(rel_err)
                    
                    run_counter += 1

                rel_errors = np.array(rel_errors)
                median_err = float(np.median(rel_errors))
                q25_err = float(np.percentile(rel_errors, 25))
                q75_err = float(np.percentile(rel_errors, 75))
                iqr_err = q75_err - q25_err
                mean_err = float(np.mean(rel_errors))
                std_err = float(np.std(rel_errors))

                results_rows.append({
                    "dataset": ds.name,
                    "d": ds.d,
                    "exact_trace": ds.exact_trace,
                    "effective_rank": ds.effective_rank,
                    "algorithm": alg_name,
                    "budget_m": m,
                    "median_rel_error": median_err,
                    "q25_rel_error": q25_err,
                    "q75_rel_error": q75_err,
                    "iqr_rel_error": iqr_err,
                    "mean_rel_error": mean_err,
                    "std_rel_error": std_err
                })
                
                print(f"  {alg_name:22s} | m={m:3d} | Median Rel Err: {median_err:.6e} | IQR: {iqr_err:.6e}")

    elapsed = time.time() - start_time
    print(f"\nAll benchmark runs completed in {elapsed:.2f} seconds.")

    # 3. Save Results & Print Summary Table
    print("\n[3/3] Saving results and generating summary table...")
    df_res = pd.DataFrame(results_rows)
    output_path = Path(__file__).parent / "real_world_benchmark_results.csv"
    df_res.to_csv(output_path, index=False)
    print(f"Results saved to {output_path.resolve()}")

    print("\n" + "=" * 100)
    print("SUMMARY TABLE: MEDIAN RELATIVE ERROR & [25th, 75th PERCENTILES]")
    print("=" * 100)
    
    summary_display = []
    for row in results_rows:
        summary_display.append({
            "Dataset": row["dataset"].split()[0],
            "Algorithm": row["algorithm"],
            "m": row["budget_m"],
            "Median Rel Error": f"{row['median_rel_error']:.4e}",
            "25th Pct": f"{row['q25_rel_error']:.4e}",
            "75th Pct": f"{row['q75_rel_error']:.4e}",
            "IQR": f"{row['iqr_rel_error']:.4e}"
        })
    df_summary = pd.DataFrame(summary_display)
    print(df_summary.to_string(index=False))
    print("=" * 100)

if __name__ == "__main__":
    main()
