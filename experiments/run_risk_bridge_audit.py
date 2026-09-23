import sys
import numpy as np
import scipy.linalg as la
import pandas as pd
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trace_baseline import (
    MatVecOracle,
    _rank_aware_qr
)

def generate_spectrum_matrix(spectrum_type, params, d=500, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))

    if spectrum_type == "power_law":
        c = params["c"]
        lambdas = (np.arange(1, d + 1, dtype=np.float64)) ** (-c)
    elif spectrum_type == "exponential":
        alpha = params["alpha"]
        lambdas = np.exp(-alpha * np.arange(0, d, dtype=np.float64))
    elif spectrum_type == "step":
        r_star = params["r_star"]
        eta = params["eta"]
        lambdas = np.ones(d, dtype=np.float64) * eta
        lambdas[:r_star] = 1.0
    elif spectrum_type == "spiked":
        r_spikes = params["r_spikes"]
        spike_val = params["spike_val"]
        c_tail = params["c_tail"]
        lambdas = (np.arange(1, d + 1, dtype=np.float64)) ** (-c_tail)
        lambdas[:r_spikes] = spike_val
    elif spectrum_type == "log_normal":
        sigma = params["sigma"]
        idx = np.arange(1, d + 1, dtype=np.float64)
        lambdas = np.exp(-sigma * (np.log(idx)) ** 2)
    else:
        raise ValueError(f"Unknown spectrum_type: {spectrum_type}")

    A = (Q_orth * lambdas) @ Q_orth.T
    return A, lambdas

def oracle_risk_full(q, lambdas, m, d):
    if q >= d or (m - 2 * q) <= 0:
        return float("inf")
    T_q = float(np.sum(lambdas[q:] ** 2))
    return (2.0 * T_q) / (m - 2 * q)

def run_risk_bridge_audit():
    print("=== Running Offline Risk Bridge Audit (R_oracle vs R_real vs MSE_emp) ===")
    d = 500
    m = 160
    n_mc_trials = 100

    q_grid = np.arange(8, 77, 4)  # 8, 12, 16, ..., 76

    spectra_to_audit = [
        ("power_law", {"c": 0.5}, "Power-Law c=0.5"),
        ("power_law", {"c": 2.0}, "Power-Law c=2.0"),
        ("exponential", {"alpha": 0.02}, "Exponential alpha=0.02"),
        ("exponential", {"alpha": 0.08}, "Exponential alpha=0.08"),
        ("step", {"r_star": 10, "eta": 0.01}, "Step r*=10, eta=0.01"),
        ("step", {"r_star": 20, "eta": 0.01}, "Step r*=20, eta=0.01"),
    ]

    summary_records = []

    for spec_type, params, label in spectra_to_audit:
        rng_master = np.random.default_rng(42)
        A_fixed, lambdas = generate_spectrum_matrix(spec_type, params, d=d, rng=rng_master)
        true_trace = float(np.trace(A_fixed))

        # 1. Oracle Risk Curve & q*_oracle
        R_oracle_curve = [oracle_risk_full(q, lambdas, m, d) for q in q_grid]
        q_star_oracle = int(q_grid[np.argmin(R_oracle_curve)])

        # 2. Realized Basis Risk Curve R_real(q) & Empirical MSE curves for Gaussian/Rademacher
        R_real_curve_mean = []
        mse_gauss_curve = []
        mse_rademacher_curve = []

        for q in q_grid:
            R_real_list = []
            err_gauss_list = []
            err_rad_list = []

            for mc_idx in range(n_mc_trials):
                rng_trial = np.random.default_rng(100000 + mc_idx)
                
                # Range-finding with sketch size q
                S = rng_trial.choice([-1.0, 1.0], size=(d, q))
                oracle = MatVecOracle(A_fixed, d=d)
                W = oracle(S)
                scale_W = float(la.norm(W, ord='fro'))
                Q, r_act = _rank_aware_qr(W, reference_scale=scale_W)
                Z = oracle(Q)

                ell_eff = m - q - r_act
                if ell_eff < 2:
                    ell_eff = 2
                    q = m - r_act - ell_eff

                # Compute R_real = 2 * ||R A R||_F^2 / (m - q - r_act)
                P = Q @ Q.T
                R_proj = np.eye(d) - P
                RAR = R_proj @ A_fixed @ R_proj
                fro_sq = float(np.sum(RAR ** 2))
                R_real_val = (2.0 * fro_sq) / ell_eff
                R_real_list.append(R_real_val)

                tr_low = float(np.sum(Q * Z))

                # Gaussian probes
                G_gauss = rng_trial.normal(loc=0.0, scale=1.0, size=(d, ell_eff))
                RG_gauss = G_gauss - Q @ (Q.T @ G_gauss)
                ARG_gauss = oracle(RG_gauss)
                tr_res_gauss = float(np.sum(RG_gauss * ARG_gauss)) / ell_eff
                tr_est_gauss = tr_low + tr_res_gauss
                err_gauss_list.append((tr_est_gauss - true_trace) ** 2)

                # Rademacher probes
                G_rad = rng_trial.choice([-1.0, 1.0], size=(d, ell_eff))
                RG_rad = G_rad - Q @ (Q.T @ G_rad)
                ARG_rad = oracle(RG_rad)
                tr_res_rad = float(np.sum(RG_rad * ARG_rad)) / ell_eff
                tr_est_rad = tr_low + tr_res_rad
                err_rad_list.append((tr_est_rad - true_trace) ** 2)

            R_real_curve_mean.append(float(np.mean(R_real_list)))
            mse_gauss_curve.append(float(np.mean(err_gauss_list)))
            mse_rademacher_curve.append(float(np.mean(err_rad_list)))

        q_star_real = int(q_grid[np.argmin(R_real_curve_mean)])
        q_star_gauss = int(q_grid[np.argmin(mse_gauss_curve)])
        q_star_rad = int(q_grid[np.argmin(mse_rademacher_curve)])

        print(f"[{label:25s}] q*_oracle={q_star_oracle:2d} | q*_real={q_star_real:2d} | q*_MSE(Gaussian)={q_star_gauss:2d} | q*_MSE(Rademacher)={q_star_rad:2d}")

        summary_records.append({
            "spectrum_type": spec_type,
            "label": label,
            "q_star_oracle": q_star_oracle,
            "q_star_real": q_star_real,
            "q_star_gauss": q_star_gauss,
            "q_star_rad": q_star_rad
        })

    df_summary = pd.DataFrame(summary_records)
    out_path = Path(__file__).resolve().parent.parent / "results" / "risk_bridge_audit_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(out_path, index=False)
    print(f"\n--> Saved risk bridge audit summary to {out_path}")

if __name__ == "__main__":
    run_risk_bridge_audit()
