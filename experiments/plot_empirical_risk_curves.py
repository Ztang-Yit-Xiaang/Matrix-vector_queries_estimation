import sys
import numpy as np
import scipy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add src/ directory to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trace_baseline import (
    MatVecOracle,
    Adaptive_Hutch_pplus_RademacherResidual,
    Adaptive_Hutch_pplus_Soft_RademacherResidual,
    _rank_aware_qr
)


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

def evaluate_actual_hutch_mse(A, q, m, n_trials=200, rng_seed=42):
    """Computes high-precision Monte Carlo MSE over n_trials repetitions."""
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

def run_risk_curve_audit():
    print("==========================================================================", flush=True)
    print("HIGH-PRECISION EMPIRICAL RISK CURVE AUDIT (200 Trials / Allocation)", flush=True)
    print("==========================================================================", flush=True)
    
    d = 500
    m = 160
    b = 10
    n_trials = 200
    rng = np.random.default_rng(42)
    
    q_max = min(d, (m - 2) // 2)
    q_0 = min(q_max, max(b, m // 3))
    
    families = [
        generate_powerlaw_psd(d, 2.0, rng),
        generate_powerlaw_psd(d, 0.5, rng),
        generate_exponential_psd(d, 0.05, rng),
        generate_step_psd(d, r=20, eta=0.01, rng=rng)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes_flat = axes.flatten()
    
    csv_rows = []
    
    for idx, (A, family_name) in enumerate(families):
        ax = axes_flat[idx]
        print(f"\n[{idx+1}/4] Auditing Risk Curve for: {family_name} (m={m}, d={d})", flush=True)
        
        oracle = MatVecOracle(A, d=d)
        
        # 1. Get Hard Adaptive decision q_hard
        oracle.reset_query_count()
        _, diag_hard = Adaptive_Hutch_pplus_RademacherResidual(
            oracle, m, d, b=b, rng=np.random.default_rng(123), return_diagnostics=True
        )
        q_hard = diag_hard["q_target"]
        
        # 2. Get Soft Adaptive decision q_soft
        oracle.reset_query_count()
        _, diag_soft = Adaptive_Hutch_pplus_Soft_RademacherResidual(
            oracle, m, d, b=b, rng=np.random.default_rng(123), return_diagnostics=True
        )
        q_soft = diag_soft["q_target"]
        
        # 3. Evaluate empirical MSE curve over q in [b, q_max]
        candidate_q = np.arange(b, q_max + 1)
        mse_curve = []
        
        for q_cand in candidate_q:
            mse_val = evaluate_actual_hutch_mse(A, q_cand, m, n_trials=n_trials, rng_seed=42 + q_cand)
            mse_curve.append(mse_val)
            csv_rows.append({
                "family": family_name,
                "m": m,
                "q": q_cand,
                "mse": mse_val
            })
            
        mse_curve = np.array(mse_curve)
        best_idx = np.argmin(mse_curve)
        q_star_emp = candidate_q[best_idx]
        mse_star_emp = mse_curve[best_idx]
        
        print(f"  Standard Hutch++ q_0   = {q_0:2d} | MSE: {evaluate_actual_hutch_mse(A, q_0, m, n_trials=n_trials):.4e}", flush=True)
        print(f"  Hard Adaptive q_hard   = {q_hard:2d} | MSE: {evaluate_actual_hutch_mse(A, q_hard, m, n_trials=n_trials):.4e}", flush=True)
        print(f"  Soft Adaptive q_soft   = {q_soft:2d} | MSE: {evaluate_actual_hutch_mse(A, q_soft, m, n_trials=n_trials):.4e}", flush=True)
        print(f"  Empirical Optimal q*   = {q_star_emp:2d} | MSE: {mse_star_emp:.4e}", flush=True)
        
        # Plotting Risk Curve
        ax.plot(candidate_q, mse_curve, 'o-', color='#2b5c8f', linewidth=2, markersize=4, label='Empirical MSE Risk Curve')
        
        # Annotate Key Allocations
        ax.axvline(q_0, color='#e74c3c', linestyle='--', linewidth=1.8, label=f'Standard Hutch++ (q_0={q_0})')
        ax.axvline(q_hard, color='#e67e22', linestyle=':', linewidth=2.0, label=f'Hard Adaptive (q={q_hard})')
        ax.axvline(q_soft, color='#27ae60', linestyle='-.', linewidth=2.0, label=f'Soft Adaptive (q={q_soft})')
        ax.plot(q_star_emp, mse_star_emp, '*', color='#9b59b6', markersize=14, label=f'Empirical Optimal (q*={q_star_emp})')
        
        ax.set_title(f"{family_name} (m={m})", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Target Low-Rank Sketch Rank (q)", fontsize=10)
        ax.set_ylabel("Mean Squared Error (MSE)", fontsize=10)
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(fontsize=8, loc='best')
        
    plt.tight_layout()
    out_png = Path(__file__).resolve().parent.parent / "figures" / "empirical_risk_curves.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    
    out_csv = Path(__file__).resolve().parent.parent / "results" / "empirical_risk_curves_data.csv"
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    
    print("\n==========================================================================", flush=True)
    print(f"EMPIRICAL RISK CURVES AUDIT COMPLETE: SAVED TO {out_png} AND {out_csv}", flush=True)
    print("==========================================================================", flush=True)

if __name__ == "__main__":
    run_risk_curve_audit()
