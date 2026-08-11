import sys
import numpy as np
import scipy.linalg as la
import pandas as pd
from pathlib import Path

# Add src/ directory to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trace_baseline import MatVecOracle, _rank_aware_qr

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

def generate_elbow_psd(d, r_elbow=10, eta=0.01, rng=None):
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = (np.arange(1, d + 1, dtype=np.float64)) ** (-2.0)
    eigenvals[r_elbow:] = float(eta)
    A = (Q_orth * eigenvals) @ Q_orth.T
    return A, f"Elbow (r={r_elbow}, eta={eta})"

def evaluate_actual_hutch_mse(A, q, m, n_trials=100, rng_seed=42):
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

def run_diagnostics():
    print("==========================================================================", flush=True)
    print("PART A: HARD MULTIMODEL FAILURE MODE DIAGNOSTIC EXPERIMENT", flush=True)
    print("==========================================================================", flush=True)

    d = 500
    m = 160
    b = 10
    n_diagnostic_trials = 100
    rng = np.random.default_rng(42)

    q_max = min(d, (m - 2) // 2)
    q_0 = min(q_max, max(b, m // 3))

    families = [
        generate_powerlaw_psd(d, 2.0, rng),
        generate_powerlaw_psd(d, 0.5, rng),
        generate_exponential_psd(d, 0.05, rng),
        generate_step_psd(d, r=20, eta=0.01, rng=rng),
        generate_elbow_psd(d, r_elbow=10, eta=0.01, rng=rng)
    ]

    diagnostic_rows = []

    for A, family_name in families:
        print(f"\n--- Diagnosing Family: {family_name} (m={m}, d={d}) ---", flush=True)

        # 1. Compute empirical risk curve & oracle q_star_emp
        candidate_q = np.arange(b, q_max + 1)
        mse_map = {}
        for q_cand in candidate_q:
            mse_map[q_cand] = evaluate_actual_hutch_mse(A, q_cand, m, n_trials=50, rng_seed=42 + q_cand)
        q_star_emp = min(mse_map, key=mse_map.get)
        mse_star_emp = mse_map[q_star_emp]

        # Classify expected true model family
        if "Power-Law" in family_name:
            true_family = "power_law"
        elif "Exponential" in family_name:
            true_family = "exponential"
        else:
            true_family = "step_gap"

        fail_A_count = 0
        fail_B_count = 0
        fail_C_count = 0
        success_count = 0

        for trial_idx in range(n_diagnostic_trials):
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

                # 1. Power-Law Fit
                slope_p, intercept_p = np.polyfit(log_j, log_theta, 1)
                c_hat = float(max(0.0, -slope_p))
                fit_p = intercept_p + slope_p * log_j
                r2_power = float(1.0 - np.sum((log_theta - fit_p) ** 2) / (np.sum((log_theta - np.mean(log_theta)) ** 2) + 1e-12))

                i_vals = np.arange(1, d + 1, dtype=np.float64)
                w_pow = i_vals ** (-2.0 * c_hat)
                T_cum_p = np.cumsum(w_pow[::-1])[::-1]
                q_power = b
                best_risk_p = float("inf")
                for q_cand in range(b, q_max + 1):
                    l_cand = m - 2 * q_cand
                    if l_cand > 0:
                        risk = 2.0 * T_cum_p[q_cand - 1] / l_cand
                        if risk < best_risk_p:
                            best_risk_p = risk
                            q_power = q_cand

                # 2. Exponential Fit
                slope_e, intercept_e = np.polyfit(j_indices, log_theta, 1)
                alpha_hat = float(max(1e-5, -slope_e))
                fit_e = intercept_e + slope_e * j_indices
                r2_exp = float(1.0 - np.sum((log_theta - fit_e) ** 2) / (np.sum((log_theta - np.mean(log_theta)) ** 2) + 1e-12))

                w_exp = np.exp(-2.0 * alpha_hat * i_vals)
                T_cum_e = np.cumsum(w_exp[::-1])[::-1]
                q_exp = b
                best_risk_e = float("inf")
                for q_cand in range(b, q_max + 1):
                    l_cand = m - 2 * q_cand
                    if l_cand > 0:
                        risk = 2.0 * T_cum_e[q_cand - 1] / l_cand
                        if risk < best_risk_e:
                            best_risk_e = risk
                            q_exp = q_cand

                # 3. Step/Gap Model
                ratios = pos_ritz[:-1] / (pos_ritz[1:] + 1e-12)
                max_ratio = float(np.max(ratios)) if len(ratios) > 0 else 1.0
                r_elbow = int(np.argmax(ratios) + 1) if len(ratios) > 0 else 1
                r2_step = 0.95 if max_ratio > 3.0 else 0.0
                q_step = min(q_max, max(b, r_elbow + 2))

                # Hard Selection Choice
                best_r2 = max(r2_power, r2_exp, r2_step)
                if r2_step >= 0.95 and max_ratio > 4.0:
                    selected_model = "step_gap"
                    q_multi = q_step
                elif r2_exp > r2_power:
                    selected_model = "exponential"
                    q_multi = q_exp
                else:
                    selected_model = "power_law"
                    q_multi = q_power
            else:
                r2_power, r2_exp, r2_step = 0.0, 0.0, 0.0
                q_power, q_exp, q_step = q_0, q_0, q_0
                selected_model = "fallback"
                q_multi = q_0

            # Failure Mode Classification
            if selected_model != true_family and selected_model != "fallback":
                failure_mode = "Failure_A (Wrong Family Selected)"
                fail_A_count += 1
            elif abs(q_multi - q_star_emp) > 10:
                failure_mode = "Failure_B (Extrapolated T(q) Allocation Error)"
                fail_B_count += 1
            elif mse_map.get(q_multi, 0) > 2.0 * mse_star_emp:
                failure_mode = "Failure_C (Oracle-vs-Randomized Risk Mismatch)"
                fail_C_count += 1
            else:
                failure_mode = "Success (Near-Optimal Allocation)"
                success_count += 1

            diagnostic_rows.append({
                "family": family_name,
                "trial": trial_idx,
                "r2_power": r2_power,
                "r2_exp": r2_exp,
                "r2_step": r2_step,
                "selected_model": selected_model,
                "true_family": true_family,
                "q_power": q_power,
                "q_exp": q_exp,
                "q_step": q_step,
                "q_multi": q_multi,
                "q_0": q_0,
                "q_star_emp": q_star_emp,
                "failure_mode": failure_mode
            })

        print(f"  Diagnostics Summary for {family_name}:", flush=True)
        print(f"    - Success Rate: {success_count}/{n_diagnostic_trials} ({success_count}%)", flush=True)
        print(f"    - Failure A (Wrong Family): {fail_A_count}%", flush=True)
        print(f"    - Failure B (Wrong Allocation under Correct Family): {fail_B_count}%", flush=True)
        print(f"    - Failure C (Oracle Mismatch): {fail_C_count}%", flush=True)

    df_diag = pd.DataFrame(diagnostic_rows)
    out_csv = Path(__file__).resolve().parent.parent / "results" / "multimodel_diagnostic_trials.csv"
    df_diag.to_csv(out_csv, index=False)

    print("\n==========================================================================", flush=True)
    print(f"DIAGNOSTICS COMPLETE AND SAVED TO {out_csv}", flush=True)
    print("==========================================================================", flush=True)

if __name__ == "__main__":
    run_diagnostics()
