import sys
import numpy as np
import scipy.linalg as la
import pandas as pd
from pathlib import Path

# Add src/ directory to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trace_baseline import (
    MatVecOracle,
    Hutchinson,
    Hutch_pplus,
    Gaussian_Hutch_pplus,
    Adaptive_Hutch_pplus_RademacherResidual,
    Adaptive_Hutch_pplus_Soft_RademacherResidual,
    _rank_aware_qr
)


def Adaptive_Hutch_pplus_MultiModel(
    oracle,
    m,
    d,
    b=10,
    min_fit_points=4,
    r2_min=0.8,
    probe_mode='rademacher',
    rng=None,
    return_diagnostics=False
):
    """
    Multi-Model Adaptive Hutch++ Trace Estimator.
    Fits Power-Law (log-log), Exponential (semilog), and Step/Gap models to pilot Ritz values.
    Selects the best structural family via R^2 comparison, eliminating single power-law model mismatch.
    """
    if rng is None:
        rng = np.random.default_rng()

    queries_before = oracle.query_count
    q_max = min(d, (m - 2) // 2)
    q_0 = min(q_max, max(b, m // 3)) if q_max >= b else b
    min_budget = 2 * b + 2

    if m < min_budget or b > q_max:
        q_target = q_0
        gamma = 0.0
        selected_model = "fallback_budget"
        fallback_used = True
        fallback_reason = "budget_too_small_for_pilot"
    else:
        fallback_used = False
        fallback_reason = None
        gamma = 0.0
        selected_model = "standard_hutchpp"

    # PHASE 1: Pilot Stage
    if not fallback_used:
        S_0 = rng.choice([-1.0, 1.0], size=(d, b))
        W_0 = oracle(S_0)
        scale_0 = float(la.norm(W_0, ord='fro'))
        
        Q_0, r_0 = _rank_aware_qr(W_0, reference_scale=scale_0)
        Z_0 = oracle(Q_0) if r_0 > 0 else np.empty((d, 0), dtype=W_0.dtype)

        if r_0 >= min_fit_points:
            M_0 = 0.5 * (Q_0.T @ Z_0 + Z_0.T @ Q_0)
            ritz_vals = la.eigvalsh(M_0)[::-1]
            theta_max = float(ritz_vals[0]) if len(ritz_vals) > 0 else 0.0

            if theta_max > 0.0:
                ritz_cutoff = 1e-12 * theta_max
                pos_ritz = ritz_vals[ritz_vals > ritz_cutoff]

                if len(pos_ritz) >= min_fit_points:
                    j_indices = np.arange(1, len(pos_ritz) + 1, dtype=np.float64)
                    log_j = np.log(j_indices)
                    log_theta = np.log(pos_ritz)

                    # 1. Power-Law Fit (log-log)
                    slope_p, intercept_p = np.polyfit(log_j, log_theta, 1)
                    c_hat = float(max(0.0, -slope_p))
                    fit_p = intercept_p + slope_p * log_j
                    r2_power = float(1.0 - np.sum((log_theta - fit_p) ** 2) / (np.sum((log_theta - np.mean(log_theta)) ** 2) + 1e-12))

                    # 2. Exponential Fit (semilog)
                    slope_e, intercept_e = np.polyfit(j_indices, log_theta, 1)
                    alpha_hat = float(max(1e-5, -slope_e))
                    fit_e = intercept_e + slope_e * j_indices
                    r2_exp = float(1.0 - np.sum((log_theta - fit_e) ** 2) / (np.sum((log_theta - np.mean(log_theta)) ** 2) + 1e-12))

                    # 3. Spectral Gap / Step Model Fit
                    ratios = pos_ritz[:-1] / (pos_ritz[1:] + 1e-12)
                    max_ratio = float(np.max(ratios)) if len(ratios) > 0 else 1.0
                    r_elbow = int(np.argmax(ratios) + 1) if len(ratios) > 0 else 1
                    r2_step = 0.95 if max_ratio > 3.0 else 0.0

                    # Model Selection
                    best_r2 = max(r2_power, r2_exp, r2_step)

                    if best_r2 >= r2_min:
                        if r2_step >= 0.95 and max_ratio > 4.0:
                            selected_model = "step_gap"
                            q_adapt = min(q_max, max(b, r_elbow + 2))
                            gamma = 0.85
                        elif r2_exp > r2_power:
                            selected_model = "exponential"
                            # Compute T_exp(q) = sum_{i > q} exp(-2 * alpha * i)
                            i_vals = np.arange(1, d + 1, dtype=np.float64)
                            w_exp = np.exp(-2.0 * alpha_hat * i_vals)
                            T_cum = np.zeros(d + 1, dtype=np.float64)
                            T_cum[:-1] = np.cumsum(w_exp[::-1])[::-1]

                            best_risk = float("inf")
                            q_adapt = b
                            for q_cand in range(b, q_max + 1):
                                l_cand = m - 2 * q_cand
                                if l_cand <= 0:
                                    continue
                                risk = 2.0 * T_cum[q_cand] / l_cand
                                if risk < best_risk:
                                    best_risk = risk
                                    q_adapt = q_cand

                            h = min(4, len(pos_ritz))
                            c_h = float(np.sum(pos_ritz[:h]**2) / (np.sum(pos_ritz**2) + 1e-12))
                            gamma = float(np.clip(r2_exp * c_h, 0.0, 1.0))
                        else:
                            selected_model = "power_law"
                            i_vals = np.arange(1, d + 1, dtype=np.float64)
                            w_pow = i_vals ** (-2.0 * c_hat)
                            T_cum = np.zeros(d + 1, dtype=np.float64)
                            T_cum[:-1] = np.cumsum(w_pow[::-1])[::-1]

                            best_risk = float("inf")
                            q_adapt = b
                            for q_cand in range(b, q_max + 1):
                                l_cand = m - 2 * q_cand
                                if l_cand <= 0:
                                    continue
                                risk = 2.0 * T_cum[q_cand] / l_cand
                                if risk < best_risk:
                                    best_risk = risk
                                    q_adapt = q_cand

                            h = min(4, len(pos_ritz))
                            c_h = float(np.sum(pos_ritz[:h]**2) / (np.sum(pos_ritz**2) + 1e-12))
                            gamma = float(np.clip(r2_power * c_h, 0.0, 1.0))

                        # Soft combination allocation
                        q_soft_float = (1.0 - gamma) * float(q_0) + gamma * float(q_adapt)
                        q_target = int(np.clip(round(q_soft_float), b, q_max))
                    else:
                        q_target = q_0
                        gamma = 0.0
                        fallback_used = True
                        fallback_reason = "low_fit_quality_all_models"

    # PHASE 3: Pilot Basis Extension
    k_extra = q_target - b
    if k_extra > 0:
        S_1 = rng.choice([-1.0, 1.0], size=(d, k_extra))
        W_1 = oracle(S_1)
        scale_1 = float(la.norm(W_1, ord='fro'))

        if r_0 > 0:
            W1_tilde = W_1 - Q_0 @ (Q_0.T @ W_1)
            W1_tilde = W1_tilde - Q_0 @ (Q_0.T @ W1_tilde)
        else:
            W1_tilde = W_1

        Q_1, r_1 = _rank_aware_qr(W1_tilde, reference_scale=scale_1)

        if r_1 > 0 and r_0 > 0:
            Q_1 = Q_1 - Q_0 @ (Q_0.T @ Q_1)
            Q_1, r_1 = _rank_aware_qr(Q_1, reference_scale=1.0)

        Z_1 = oracle(Q_1) if r_1 > 0 else np.empty((d, 0), dtype=W_1.dtype)

        if r_0 > 0 and r_1 > 0:
            Q = np.column_stack([Q_0, Q_1])
            Z = np.column_stack([Z_0, Z_1])
        elif r_0 > 0:
            Q, Z = Q_0, Z_0
        else:
            Q, Z = Q_1, Z_1
    else:
        Q, Z = Q_0, Z_0
        r_1 = 0

    r_actual = Q.shape[1]

    # PHASE 4: Exact Budget Accounting
    ell_eff = m - q_target - r_actual
    if ell_eff < 2:
        raise RuntimeError(
            f"Invalid query allocation: fewer than 2 residual probes remain. "
            f"m={m}, q_target={q_target}, r_actual={r_actual}, ell_eff={ell_eff}"
        )

    # PHASE 5: Double Residual Projection
    if probe_mode == 'gaussian':
        G = rng.normal(loc=0.0, scale=1.0, size=(d, ell_eff))
    else:
        G = rng.choice([-1.0, 1.0], size=(d, ell_eff))

    if r_actual > 0:
        RG = G - Q @ (Q.T @ G)
    else:
        RG = G

    ARG = oracle(RG)

    queries_used = oracle.query_count - queries_before
    if queries_used != m:
        raise RuntimeError(
            f"Query budget mismatch: expected {m} queries, actually used {queries_used}."
        )

    trace_low = float(np.sum(Q * Z)) if r_actual > 0 else 0.0
    trace_res = float(np.sum(RG * ARG)) / ell_eff
    trace_est = trace_low + trace_res

    if return_diagnostics:
        diag = {
            "selected_model": selected_model,
            "gamma": gamma,
            "q_0": q_0,
            "q_target": q_target,
            "r_actual": r_actual,
            "ell_eff": ell_eff,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "probe_mode": probe_mode
        }
        return trace_est, diag

    return trace_est

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

def run_multimodel_benchmark():
    print("==========================================================================", flush=True)
    print("MULTI-MODEL ADAPTIVE HUTCH++ BENCHMARK & REGRET ANALYSIS", flush=True)
    print("==========================================================================", flush=True)

    d = 500
    b = 10
    budgets = [40, 80, 160, 320]
    n_trials = 50
    rng = np.random.default_rng(42)

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
        "Hard-Adaptive (Single-c)": Adaptive_Hutch_pplus_RademacherResidual,
        "Soft-Adaptive (Single-c)": Adaptive_Hutch_pplus_Soft_RademacherResidual,
        "MultiModel-Adaptive (Ours)": Adaptive_Hutch_pplus_MultiModel
    }

    results = []

    for A, ds_name in dataset_generators:
        exact_tr = float(np.trace(A))
        fro_norm = float(la.norm(A, ord='fro'))
        r_eff = (exact_tr ** 2) / (fro_norm ** 2 + 1e-12)

        print(f"\n--- Testing Dataset Family: {ds_name} (d={d}, tr(A)={exact_tr:.2f}, r_eff={r_eff:.2f}) ---", flush=True)
        oracle = MatVecOracle(A, d=d)

        for m in budgets:
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

                results.append({
                    "dataset": ds_name,
                    "d": d,
                    "effective_rank": r_eff,
                    "m": m,
                    "algorithm": alg_name,
                    "mse": mse_val,
                    "median_rel_err": median_rel_err
                })

                print(f"  m={m:3d} | {alg_name:30s} | MSE: {mse_val:.6e} | RelErr: {median_rel_err:.4e}", flush=True)

    df_res = pd.DataFrame(results)
    out_csv = Path(__file__).resolve().parent.parent / "results" / "multimodel_benchmark_results.csv"
    df_res.to_csv(out_csv, index=False)


    print("\n==========================================================================", flush=True)
    print("MULTI-MODEL SUMMARY TABLE (MSE COMPARISON)", flush=True)
    print("==========================================================================", flush=True)
    pivoted_mse = df_res.pivot(index=["dataset", "m"], columns="algorithm", values="mse")
    print(pivoted_mse.to_string(), flush=True)

    print("\n==========================================================================", flush=True)
    print(f"MULTI-MODEL ADAPTIVE BENCHMARK COMPLETED AND SAVED TO {out_csv}", flush=True)
    print("==========================================================================", flush=True)

if __name__ == "__main__":
    run_multimodel_benchmark()
