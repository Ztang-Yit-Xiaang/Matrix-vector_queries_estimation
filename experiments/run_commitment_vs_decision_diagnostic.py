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
    Adaptive_Hutch_pplus_SequentialPilot
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

def run_commitment_vs_decision_diagnostic():
    print("=== Running Commitment vs Decision Regret Diagnostic Across 24 Held-Out Spectra ===")
    d = 500
    m = 160
    n_trials = 50

    # 24 Held-Out Spectra Definitions
    spectra_manifest = [
        # Power-Law (6)
        ("power_law", {"c": 0.3}, "Power-Law c=0.3"),
        ("power_law", {"c": 0.7}, "Power-Law c=0.7"),
        ("power_law", {"c": 1.2}, "Power-Law c=1.2"),
        ("power_law", {"c": 1.7}, "Power-Law c=1.7"),
        ("power_law", {"c": 2.5}, "Power-Law c=2.5"),
        ("power_law", {"c": 3.0}, "Power-Law c=3.0"),
        # Exponential (4)
        ("exponential", {"alpha": 0.02}, "Exponential alpha=0.02"),
        ("exponential", {"alpha": 0.05}, "Exponential alpha=0.05"),
        ("exponential", {"alpha": 0.08}, "Exponential alpha=0.08"),
        ("exponential", {"alpha": 0.15}, "Exponential alpha=0.15"),
        # Step (8)
        ("step", {"r_star": 5, "eta": 0.001}, "Step r*=5, eta=0.001"),
        ("step", {"r_star": 5, "eta": 0.05}, "Step r*=5, eta=0.05"),
        ("step", {"r_star": 10, "eta": 0.01}, "Step r*=10, eta=0.01"),
        ("step", {"r_star": 15, "eta": 0.001}, "Step r*=15, eta=0.001"),
        ("step", {"r_star": 20, "eta": 0.01}, "Step r*=20, eta=0.01"),
        ("step", {"r_star": 25, "eta": 0.01}, "Step r*=25, eta=0.01"),
        ("step", {"r_star": 30, "eta": 0.001}, "Step r*=30, eta=0.001"),
        ("step", {"r_star": 30, "eta": 0.05}, "Step r*=30, eta=0.05"),
        # Spiked (4)
        ("spiked", {"r_spikes": 3, "spike_val": 10.0, "c_tail": 1.0}, "Spiked r=3, val=10"),
        ("spiked", {"r_spikes": 5, "spike_val": 20.0, "c_tail": 1.5}, "Spiked r=5, val=20"),
        ("spiked", {"r_spikes": 10, "spike_val": 5.0, "c_tail": 0.5}, "Spiked r=10, val=5"),
        ("spiked", {"r_spikes": 15, "spike_val": 15.0, "c_tail": 2.0}, "Spiked r=15, val=15"),
        # Log-Normal (2)
        ("log_normal", {"sigma": 0.5}, "Log-Normal sigma=0.5"),
        ("log_normal", {"sigma": 1.0}, "Log-Normal sigma=1.0"),
    ]

    results = []

    for spec_type, params, label in spectra_manifest:
        q_max = (m - 2) // 2
        # Compute exact lambdas for this spectrum
        rng_seed = np.random.default_rng(12345)
        _, lambdas = generate_spectrum_matrix(spec_type, params, d=d, rng=rng_seed)

        # Unrestricted oracle allocation q*
        q_grid = np.arange(1, q_max + 1)
        risks_unrestricted = [oracle_risk_full(q, lambdas, m, d) for q in q_grid]
        best_q_idx = int(np.argmin(risks_unrestricted))
        q_star = int(q_grid[best_q_idx])
        R_star = float(risks_unrestricted[best_q_idx])

        trial_records = []
        for trial_idx in range(n_trials):
            rng_trial = np.random.default_rng(20000 + trial_idx)
            A, _ = generate_spectrum_matrix(spec_type, params, d=d, rng=rng_trial)
            oracle = MatVecOracle(A, d=d)

            _, diag = Adaptive_Hutch_pplus_SequentialPilot(
                oracle, m, d, b_0=8, delta_b=4, rng=rng_trial, return_diagnostics=True
            )

            B = diag["b_final"]
            q_sel = diag["q_target"]

            # Pilot-constrained oracle q_B*
            q_grid_B = np.arange(B, q_max + 1)
            risks_constrained = [oracle_risk_full(q, lambdas, m, d) for q in q_grid_B]
            best_B_idx = int(np.argmin(risks_constrained))
            q_B_star = int(q_grid_B[best_B_idx])
            R_B_star = float(risks_constrained[best_B_idx])

            R_sel = oracle_risk_full(q_sel, lambdas, m, d)

            C_commit = (R_B_star / R_star) - 1.0
            C_decision = (R_sel / R_B_star) - 1.0
            C_total = (R_sel / R_star) - 1.0

            trial_records.append({
                "B": B,
                "q_star": q_star,
                "q_B_star": q_B_star,
                "q_sel": q_sel,
                "R_star": R_star,
                "R_B_star": R_B_star,
                "R_sel": R_sel,
                "C_commit": C_commit,
                "C_decision": C_decision,
                "C_total": C_total
            })

        df_trials = pd.DataFrame(trial_records)
        mean_B = df_trials["B"].mean()
        mean_q_B_star = df_trials["q_B_star"].mean()
        mean_q_sel = df_trials["q_sel"].mean()
        mean_C_commit = df_trials["C_commit"].mean()
        mean_C_decision = df_trials["C_decision"].mean()
        mean_C_total = df_trials["C_total"].mean()

        print(f"[{label:25s}] q*={q_star:2d} | mean(B)={mean_B:4.1f} | mean(q_B*)={mean_q_B_star:4.1f} | mean(q_sel)={mean_q_sel:4.1f} | C_commit={mean_C_commit:+6.3f} | C_decision={mean_C_decision:+6.3f} | C_total={mean_C_total:+6.3f}")

        results.append({
            "spectrum_type": spec_type,
            "label": label,
            "q_star": q_star,
            "mean_B": mean_B,
            "mean_q_B_star": mean_q_B_star,
            "mean_q_sel": mean_q_sel,
            "mean_C_commit": mean_C_commit,
            "mean_C_decision": mean_C_decision,
            "mean_C_total": mean_C_total
        })

    df_summary = pd.DataFrame(results)
    out_path = Path(__file__).resolve().parent.parent / "results" / "commitment_vs_decision_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(out_path, index=False)
    print(f"\n--> Saved commitment vs decision summary to {out_path}")

if __name__ == "__main__":
    run_commitment_vs_decision_diagnostic()
