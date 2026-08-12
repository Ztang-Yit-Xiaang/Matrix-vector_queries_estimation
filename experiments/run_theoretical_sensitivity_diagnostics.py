import sys
import numpy as np
import scipy.linalg as la
import pandas as pd
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trace_baseline import MatVecOracle, _rank_aware_qr

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

def run_logarithmic_tail_sensitivity():
    print("==========================================================================", flush=True)
    print("1. LOGARITHMIC TAIL ERROR & EXTRAPOLATION DISTANCE ANALYSIS", flush=True)
    print("==========================================================================", flush=True)

    d = 500
    b = 10
    n_trials = 100
    rng = np.random.default_rng(42)

    alphas = [0.05, 0.08, 0.15]
    eval_q_list = [20, 30, 40, 50, 60, 70, 80]
    log_err_rows = []

    for alpha in alphas:
        A, true_eigenvals, setup_name = generate_exponential_psd(d, alpha, rng)
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
                log_theta = np.log(pos_ritz)
                i_vals = np.arange(1, d + 1, dtype=np.float64)

                # Fit Exponential Slope
                slope_e, intercept_e = np.polyfit(j_indices, log_theta, 1)
                alpha_hat = float(max(1e-5, -slope_e))
                delta_alpha = alpha_hat - alpha
                fit_e = intercept_e + slope_e * j_indices
                loss_e = float(np.mean((log_theta - fit_e) ** 2))

                w_exp = np.exp(-2.0 * alpha_hat * i_vals)
                T_cum_e = np.cumsum(w_exp[::-1])[::-1]

                for target_q in eval_q_list:
                    T_true_val = T_true_map[target_q]
                    T_pred_e = float(T_cum_e[target_q - 1])
                    extrapolation_dist = target_q - b

                    # Logarithmic Tail Error |ln T_hat(q) - ln T(q)|
                    log_tail_error = abs(np.log(T_pred_e + 1e-300) - np.log(T_true_val + 1e-300))

                    log_err_rows.append({
                        "alpha": alpha,
                        "trial": trial_idx,
                        "extrapolation_dist": extrapolation_dist,
                        "target_q": target_q,
                        "alpha_hat": alpha_hat,
                        "delta_alpha": delta_alpha,
                        "loss_pilot": loss_e,
                        "log_tail_error": log_tail_error,
                        "theoretical_exponent": 2.0 * abs(delta_alpha) * target_q
                    })

    df_log = pd.DataFrame(log_err_rows)
    out_csv = Path(__file__).resolve().parent.parent / "results" / "logarithmic_tail_sensitivity_results.csv"
    df_log.to_csv(out_csv, index=False)

    print(f"\nSaved Logarithmic Sensitivity results to {out_csv}", flush=True)

    print("\n--- Summary: Mean Logarithmic Tail Error |ln T_hat(q) - ln T(q)| vs. Extrapolation Distance (q - b) ---", flush=True)
    summary_df = df_log.groupby(["alpha", "extrapolation_dist"])[["loss_pilot", "log_tail_error", "delta_alpha"]].mean().reset_index()
    print(summary_df.to_string(index=False), flush=True)


def run_oversampling_post_knee_analysis():
    print("\n==========================================================================", flush=True)
    print("2. PILOT OVERSAMPLING POST-KNEE ANALYSIS (b = r + p)", flush=True)
    print("==========================================================================", flush=True)

    d = 500
    n_trials = 100
    rng = np.random.default_rng(42)

    step_ranks = [10, 15, 20]
    oversampling_p_list = [0, 1, 2, 3, 4, 5, 8, 10]
    oversampling_rows = []

    for r_step in step_ranks:
        A, true_eigenvals, setup_name = generate_step_psd(d, r=r_step, eta=0.01, rng=rng)
        print(f"\n--- Analyzing Step Rank r={r_step} with Oversampling p ---", flush=True)

        for p in oversampling_p_list:
            b = r_step + p

            detection_successes = 0

            for trial_idx in range(n_trials):
                trial_rng = np.random.default_rng(3000 + trial_idx)

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
                        
                        if max_ratio > 3.0 and abs(r_elbow - r_step) <= 2:
                            detection_successes += 1

            detection_rate = detection_successes / float(n_trials)

            oversampling_rows.append({
                "r_step": r_step,
                "p_oversample": p,
                "b_pilot": b,
                "detection_rate": detection_rate
            })

            print(f"  Step r={r_step:2d} | p={p:2d} (b={b:2d}) | Detection Rate: {detection_rate*100:5.1f}%", flush=True)

    df_over = pd.DataFrame(oversampling_rows)
    out_csv2 = Path(__file__).resolve().parent.parent / "results" / "post_knee_oversampling_results.csv"
    df_over.to_csv(out_csv2, index=False)

    print(f"\nSaved Post-Knee Oversampling results to {out_csv2}", flush=True)

if __name__ == "__main__":
    run_logarithmic_tail_sensitivity()
    run_oversampling_post_knee_analysis()
