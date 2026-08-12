import sys
import numpy as np
import scipy.linalg as la
import pandas as pd
from pathlib import Path

# Add src/ directory to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trace_baseline import MatVecOracle, _rank_aware_qr, Adaptive_Hutch_pplus_ModelAveraged

def generate_powerlaw_psd(d, c, rng):
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = (np.arange(1, d + 1, dtype=np.float64)) ** (-float(c))
    A = (Q_orth * eigenvals) @ Q_orth.T
    return A, eigenvals, f"Power-Law (c={c})"

def generate_exponential_psd(d, alpha, rng):
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = np.exp(-float(alpha) * np.arange(1, d + 1, dtype=np.float64))
    A = (Q_orth * eigenvals) @ Q_orth.T
    return A, eigenvals, f"Exponential (alpha={alpha})"

def generate_step_psd(d, r=20, eta=0.01, rng=None):
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = np.ones(d, dtype=np.float64) * float(eta)
    eigenvals[:r] = 1.0
    A = (Q_orth * eigenvals) @ Q_orth.T
    return A, eigenvals, f"Step (r={r}, eta={eta})"

def run_experiment_1_tail_predicticability():
    print("==========================================================================", flush=True)
    print("EXPERIMENT 1: IS PILOT FIT LOSS PREDICTIVE OF OUT-OF-SAMPLE TAIL ACCURACY?", flush=True)
    print("==========================================================================", flush=True)

    d = 500
    b = 10
    n_trials = 100
    rng = np.random.default_rng(42)

    setups = [
        generate_exponential_psd(d, 0.05, rng),
        generate_exponential_psd(d, 0.08, rng),
        generate_exponential_psd(d, 0.15, rng),
        generate_powerlaw_psd(d, 1.5, rng),
        generate_powerlaw_psd(d, 2.0, rng),
        generate_powerlaw_psd(d, 2.5, rng)
    ]

    eval_q_list = [20, 40, 60, 80]
    exp1_rows = []

    for A, true_eigenvals, setup_name in setups:
        print(f"\n--- Analyzing Setup: {setup_name} ---", flush=True)
        # Compute exact true tail energy T_true(q) = sum_{i > q} lambda_i^2
        sq_true_vals = true_eigenvals ** 2
        T_true_map = {q: float(np.sum(sq_true_vals[q:])) for q in eval_q_list}

        for trial_idx in range(n_trials):
            trial_rng = np.random.default_rng(1000 + trial_idx)

            # Pilot Phase
            S_0 = trial_rng.choice([-1.0, 1.0], size=(d, b))
            W_0 = A @ S_0
            scale_0 = float(la.norm(W_0, ord='fro'))
            Q_0, r_0 = _rank_aware_qr(W_0, reference_scale=scale_0)
            Z_0 = A @ Q_0 if r_0 > 0 else np.empty((d, 0))

            if r_0 >= 4:
                M_0 = 0.5 * (Q_0.T @ Z_0 + Z_0.T @ Q_0)
                ritz_vals = la.eigvalsh(M_0)[::-1]
                theta_max = float(ritz_vals[0]) if len(ritz_vals) > 0 else 0.0
                pos_ritz = ritz_vals[ritz_vals > 1e-12 * theta_max] if theta_max > 0 else np.empty(0)
            else:
                pos_ritz = np.empty(0)

            if len(pos_ritz) >= 4:
                j_indices = np.arange(1, len(pos_ritz) + 1, dtype=np.float64)
                log_j = np.log(j_indices)
                log_theta = np.log(pos_ritz)
                i_vals = np.arange(1, d + 1, dtype=np.float64)

                # Fit Power-Law
                slope_p, intercept_p = np.polyfit(log_j, log_theta, 1)
                c_hat = float(max(0.0, -slope_p))
                fit_p = intercept_p + slope_p * log_j
                loss_p = float(np.mean((log_theta - fit_p) ** 2))
                w_pow = i_vals ** (-2.0 * c_hat)
                T_cum_p = np.cumsum(w_pow[::-1])[::-1]

                # Fit Exponential
                slope_e, intercept_e = np.polyfit(j_indices, log_theta, 1)
                alpha_hat = float(max(1e-5, -slope_e))
                fit_e = intercept_e + slope_e * j_indices
                loss_e = float(np.mean((log_theta - fit_e) ** 2))
                w_exp = np.exp(-2.0 * alpha_hat * i_vals)
                T_cum_e = np.cumsum(w_exp[::-1])[::-1]

                for target_q in eval_q_list:
                    T_true_val = T_true_map[target_q]
                    T_pred_p = float(T_cum_p[target_q - 1])
                    T_pred_e = float(T_cum_e[target_q - 1])

                    err_p = abs(T_pred_p - T_true_val) / (T_true_val + 1e-15)
                    err_e = abs(T_pred_e - T_true_val) / (T_true_val + 1e-15)

                    exp1_rows.append({
                        "setup": setup_name,
                        "trial": trial_idx,
                        "target_q": target_q,
                        "loss_power": loss_p,
                        "loss_exp": loss_e,
                        "rel_err_tail_power": err_p,
                        "rel_err_tail_exp": err_e,
                        "amplification_power": err_p / (loss_p + 1e-12),
                        "amplification_exp": err_e / (loss_e + 1e-12)
                    })

    df_exp1 = pd.DataFrame(exp1_rows)
    out_csv1 = Path(__file__).resolve().parent.parent / "results" / "rq1_pilot_vs_tail_accuracy.csv"
    df_exp1.to_csv(out_csv1, index=False)

    print(f"\nSaved Experiment 1 results to {out_csv1}", flush=True)

    # Compute Correlations
    print("\n--- Correlation Summary: In-Sample Fit Loss vs. Out-of-Sample Tail Error ---", flush=True)
    summary_rows = []
    for setup_name in df_exp1["setup"].unique():
        sub_df = df_exp1[df_exp1["setup"] == setup_name]
        for target_q in sub_df["target_q"].unique():
            q_df = sub_df[sub_df["target_q"] == target_q]
            
            corr_p_pearson = float(q_df["loss_power"].corr(q_df["rel_err_tail_power"], method="pearson"))
            corr_p_spearman = float(q_df["loss_power"].corr(q_df["rel_err_tail_power"], method="spearman"))
            corr_e_pearson = float(q_df["loss_exp"].corr(q_df["rel_err_tail_exp"], method="pearson"))
            corr_e_spearman = float(q_df["loss_exp"].corr(q_df["rel_err_tail_exp"], method="spearman"))
            
            mean_amp_p = float(np.mean(q_df["amplification_power"]))
            mean_amp_e = float(np.mean(q_df["amplification_exp"]))

            summary_rows.append({
                "setup": setup_name,
                "target_q": target_q,
                "corr_power_spearman": corr_p_spearman,
                "corr_exp_spearman": corr_e_spearman,
                "mean_amp_power": mean_amp_p,
                "mean_amp_exp": mean_amp_e
            })

    df_sum1 = pd.DataFrame(summary_rows)
    print(df_sum1.to_string(index=False), flush=True)


def run_experiment_2_pilot_horizon():
    print("\n==========================================================================", flush=True)
    print("EXPERIMENT 2: PILOT HORIZON VS. FEATURE LOCATION (b vs. r)", flush=True)
    print("==========================================================================", flush=True)

    d = 500
    m = 160
    n_trials = 50
    rng = np.random.default_rng(42)

    step_ranks = [5, 10, 15, 20, 25, 30]
    pilot_budgets = [5, 10, 15, 20, 25, 30, 40, 50]

    exp2_rows = []

    for r_step in step_ranks:
        A, true_eigenvals, setup_name = generate_step_psd(d, r=r_step, eta=0.01, rng=rng)
        print(f"\n--- Analyzing Step Rank r={r_step} ---", flush=True)

        for b in pilot_budgets:
            if 2 * b + 2 > m:
                continue

            detection_successes = 0
            q_targets = []

            for trial_idx in range(n_trials):
                trial_rng = np.random.default_rng(2000 + trial_idx)

                # Pilot Phase
                S_0 = trial_rng.choice([-1.0, 1.0], size=(d, b))
                W_0 = A @ S_0
                scale_0 = float(la.norm(W_0, ord='fro'))
                Q_0, r_0 = _rank_aware_qr(W_0, reference_scale=scale_0)
                Z_0 = A @ Q_0 if r_0 > 0 else np.empty((d, 0))

                if r_0 >= 3:
                    M_0 = 0.5 * (Q_0.T @ Z_0 + Z_0.T @ Q_0)
                    ritz_vals = la.eigvalsh(M_0)[::-1]
                    theta_max = float(ritz_vals[0]) if len(ritz_vals) > 0 else 0.0
                    pos_ritz = ritz_vals[ritz_vals > 1e-12 * theta_max] if theta_max > 0 else np.empty(0)

                    if len(pos_ritz) >= 2:
                        ratios = pos_ritz[:-1] / (pos_ritz[1:] + 1e-12)
                        max_ratio = float(np.max(ratios)) if len(ratios) > 0 else 1.0
                        r_elbow = int(np.argmax(ratios) + 1) if len(ratios) > 0 else 1
                        
                        # Correctly detected step drop at rank r
                        if max_ratio > 3.0 and abs(r_elbow - r_step) <= 2:
                            detection_successes += 1

                # Benchmark estimator with pilot size b
                oracle = MatVecOracle(A, d=d)
                est, diag = Adaptive_Hutch_pplus_ModelAveraged(
                    oracle, m, d, b=b, rng=trial_rng, return_diagnostics=True
                )
                q_targets.append(diag["q_target"])

            detection_rate = detection_successes / float(n_trials)
            mean_q = float(np.mean(q_targets))
            std_q = float(np.std(q_targets))
            pilot_ratio = float(b / r_step)

            exp2_rows.append({
                "r_step": r_step,
                "pilot_b": b,
                "pilot_ratio": pilot_ratio,
                "detection_rate": detection_rate,
                "mean_q_target": mean_q,
                "std_q_target": std_q
            })

            print(f"  Step r={r_step:2d} | Pilot b={b:2d} (b/r={pilot_ratio:.2f}) | Detection Rate: {detection_rate*100:5.1f}% | Mean q: {mean_q:.1f}", flush=True)

    df_exp2 = pd.DataFrame(exp2_rows)
    out_csv2 = Path(__file__).resolve().parent.parent / "results" / "rq2_pilot_horizon_vs_step_rank.csv"
    df_exp2.to_csv(out_csv2, index=False)

    print(f"\nSaved Experiment 2 results to {out_csv2}", flush=True)

if __name__ == "__main__":
    run_experiment_1_tail_predicticability()
    run_experiment_2_pilot_horizon()
