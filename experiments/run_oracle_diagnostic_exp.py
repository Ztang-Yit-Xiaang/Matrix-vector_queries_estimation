import numpy as np
import scipy.linalg as la
import pandas as pd
from pathlib import Path
from trace_baseline import _rank_aware_qr

def generate_synthetic_powerlaw_psd(d, c, rng):
    """
    Generates a synthetic PSD matrix A = U diag(i^-c) U^T.
    """
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = (np.arange(1, d + 1, dtype=np.float64)) ** (-float(c))
    A = (Q_orth * eigenvals) @ Q_orth.T
    return A, eigenvals

def compute_q_model(c_val, m, d, b=10):
    """
    Computes q_model(c) = argmin_q 2/(m - 2q) * sum_{i=q+1}^d i^(-2c)
    using the exact spectrum decay parameter c.
    """
    q_max = min(d, (m - 2) // 2)
    if b >= q_max:
        return b
    
    i_vals = np.arange(1, d + 1, dtype=np.float64)
    w_i = i_vals ** (-2.0 * float(c_val))
    T_cum = np.zeros(d + 1, dtype=np.float64)
    T_cum[:-1] = np.cumsum(w_i[::-1])[::-1]
    
    best_risk = float("inf")
    q_best = b
    for q in range(b, q_max + 1):
        l_cand = m - 2 * q
        if l_cand <= 0:
            continue
        risk = 2.0 * T_cum[q] / l_cand
        if risk < best_risk:
            best_risk = risk
            q_best = q
    return q_best

def estimate_c_pilot(A, b=10, rng=None, min_fit_points=4, r2_min=0.8):
    """
    Runs Phase 1 pilot on A to estimate c_hat via log-log Ritz linear regression.
    """
    if rng is None:
        rng = np.random.default_rng()
    d = A.shape[0]
    S_0 = rng.choice([-1.0, 1.0], size=(d, b))
    W_0 = A @ S_0
    scale_0 = float(la.norm(W_0, ord='fro'))
    
    Q_0, r_0 = _rank_aware_qr(W_0, reference_scale=scale_0)
    if r_0 < min_fit_points:
        return None, 0.0, False
        
    Z_0 = A @ Q_0
    M_0 = 0.5 * (Q_0.T @ Z_0 + Z_0.T @ Q_0)
    ritz_vals = la.eigvalsh(M_0)[::-1]
    theta_max = float(ritz_vals[0]) if len(ritz_vals) > 0 else 0.0
    
    if theta_max <= 0.0:
        return None, 0.0, False
        
    pos_ritz = ritz_vals[ritz_vals > 1e-12 * theta_max]
    if len(pos_ritz) < min_fit_points:
        return None, 0.0, False
        
    j_indices = np.arange(1, len(pos_ritz) + 1, dtype=np.float64)
    log_j = np.log(j_indices)
    log_theta = np.log(pos_ritz)
    
    slope, intercept = np.polyfit(log_j, log_theta, 1)
    c_hat = float(max(0.0, -slope))
    
    fit_vals = intercept + slope * log_j
    ss_res = np.sum((log_theta - fit_vals) ** 2)
    ss_tot = np.sum((log_theta - np.mean(log_theta)) ** 2)
    r_squared = float(1.0 - (ss_res / (ss_tot + 1e-12)))
    
    fit_is_reliable = (r_squared >= r2_min)
    return c_hat, r_squared, fit_is_reliable

def evaluate_actual_hutch_mse(A, q, m, n_trials=50, rng_seed=42):
    """
    Evaluates actual empirical MSE of randomized Hutch++ at fixed split q over n_trials.
    """
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

def compute_q_emp(A, m, b=10, n_trials=50):
    """
    Exhaustively searches over candidate q in [b, floor((m-2)/2)] to find q_emp = argmin MSE(q).
    """
    d = A.shape[0]
    q_max = min(d, (m - 2) // 2)
    if b >= q_max:
        return b, evaluate_actual_hutch_mse(A, b, m, n_trials=n_trials)
        
    best_mse = float("inf")
    q_emp = b
    
    for q_cand in range(b, q_max + 1):
        mse = evaluate_actual_hutch_mse(A, q_cand, m, n_trials=n_trials)
        if mse < best_mse:
            best_mse = mse
            q_emp = q_cand
            
    return q_emp, best_mse

def run_diagnostic_experiment():
    print("==========================================================================", flush=True)
    print("ORACLE DIAGNOSTIC EXPERIMENT: q_model(c_true) vs q_pilot(c_hat) vs q_emp", flush=True)
    print("==========================================================================", flush=True)
    
    d = 500
    b = 10
    c_values = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    m_values = [40, 80, 160, 320]
    n_trials_emp = 50
    
    results = []
    
    for c_true in c_values:
        print(f"\n--- Testing Power-Law Decay c_true = {c_true} (d={d}) ---", flush=True)
        gen_rng = np.random.default_rng(int(c_true * 1000) + 1234)
        A, _ = generate_synthetic_powerlaw_psd(d, c_true, gen_rng)
        
        for m in m_values:
            # 1. Oracle Model Allocation (using exact c_true)
            q_model = compute_q_model(c_true, m, d, b=b)
            
            # 2. Pilot Allocation (using c_hat estimated from b=10 queries)
            pilot_rng = np.random.default_rng(int(c_true * 1000) + m + 5678)
            c_hat, r2, fit_reliable = estimate_c_pilot(A, b=b, rng=pilot_rng)
            
            if fit_reliable and c_hat is not None:
                q_pilot = compute_q_model(c_hat, m, d, b=b)
            else:
                q_pilot = min(min(d, (m - 2) // 2), max(b, m // 3))
                
            # 3. Empirical Optimal Allocation (Exhaustive search over actual randomized Hutch++ MSE)
            q_emp, mse_q_emp = compute_q_emp(A, m, b=b, n_trials=n_trials_emp)
            
            # Evaluate MSEs at q_model and q_pilot
            mse_q_model = evaluate_actual_hutch_mse(A, q_model, m, n_trials=n_trials_emp)
            mse_q_pilot = evaluate_actual_hutch_mse(A, q_pilot, m, n_trials=n_trials_emp)
            
            # Calculate Allocation Gaps
            gap_model_emp = abs(q_model - q_emp)
            gap_pilot_model = abs(q_pilot - q_model) if fit_reliable else None
            
            # Record diagnosis for this setup
            if gap_model_emp > 5:
                diag_state = "STATE_1: Model Discrepancy (q_model != q_emp)"
            elif gap_pilot_model is not None and gap_pilot_model > 5:
                diag_state = "STATE_2: Pilot Error (q_pilot != q_model)"
            else:
                diag_state = "STATE_3: Adaptation Working (q_pilot ~ q_model ~ q_emp)"
                
            results.append({
                "c_true": c_true,
                "m": m,
                "d": d,
                "b": b,
                "c_hat": c_hat if fit_reliable else np.nan,
                "r2_fit": r2 if fit_reliable else np.nan,
                "fit_reliable": fit_reliable,
                "q_model": q_model,
                "q_pilot": q_pilot,
                "q_emp": q_emp,
                "gap_model_emp": gap_model_emp,
                "gap_pilot_model": gap_pilot_model,
                "mse_q_model": mse_q_model,
                "mse_q_pilot": mse_q_pilot,
                "mse_q_emp": mse_q_emp,
                "diag_state": diag_state
            })
            
            c_hat_str = f"{c_hat:.2f}" if fit_reliable and c_hat is not None else "N/A"
            print(f"  m={m:3d} | q_model={q_model:3d} | c_hat={c_hat_str:5s} | q_pilot={q_pilot:3d} | q_emp={q_emp:3d} | State: {diag_state}", flush=True)
            
    df_res = pd.DataFrame(results)
    out_csv = Path(__file__).parent / "oracle_diagnostic_results.csv"
    df_res.to_csv(out_csv, index=False)
    
    print("\n==========================================================================", flush=True)
    print("DIAGNOSTIC SUMMARY TABLE (q_model vs q_pilot vs q_emp)", flush=True)
    print("==========================================================================", flush=True)
    summary_cols = ["c_true", "m", "q_model", "q_pilot", "q_emp", "gap_model_emp", "mse_q_model", "mse_q_emp", "diag_state"]
    print(df_res[summary_cols].to_string(), flush=True)
    print(f"\n==========================================================================", flush=True)
    print(f"DIAGNOSTIC BENCHMARK COMPLETED AND SAVED TO {out_csv}", flush=True)
    print("==========================================================================", flush=True)

if __name__ == "__main__":
    run_diagnostic_experiment()
