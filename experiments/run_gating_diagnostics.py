"""Pilot Feature Gating Diagnostics & Predictability Map for Adaptive Hutch++.

Constructs an observable diagnostic mapping from pilot spectral features
(top Ritz gap gamma_gap, Ritz decay slope, Ritz condition proxy kappa) to
optimal query allocation decisions (q* vs q_0).

Demonstrates:
1. Which spectral regimes are "obviously adapt" (high pilot gap, large oracle gain,
   near-100% trigger, zero regret).
2. Which regimes are "benign baseline" (smooth power law, flat spectrum, zero trigger,
   safe fallback to q_0 = m // 3 with zero penalty).
3. The transition boundary where mild curvature transitions to sharp subspace isolation.
"""

from __future__ import annotations

import os
import sys
import math
import numpy as np
import scipy.linalg as la
import pandas as pd
from collections.abc import Sequence

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.trace_baseline import (
    MatVecOracle,
    Hutchinson,
    Hutch_pplus,
    Gaussian_Hutch_pplus,
    Adaptive_Hutch_pplus_TwoStageGated,
    _rank_aware_qr,
)

RESULTS_DIR = os.path.join(project_dir, "results")


def sample_haar_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    """Sample Haar-random orthogonal matrix."""
    Z = rng.normal(loc=0.0, scale=1.0, size=(d, d))
    Q, R = la.qr(Z)
    d_diag = np.diagonal(R)
    ph = d_diag / np.where(np.abs(d_diag) > 1e-14, np.abs(d_diag), 1.0)
    return Q * ph


def build_test_matrices(d: int = 100, seed: int = 42) -> list[dict]:
    """Construct diverse spectra spanning step, power-law, exponential, and neural regimes."""
    rng = np.random.default_rng(seed)
    U = sample_haar_orthogonal(d, rng)
    matrices = []

    # 1. Step Spectra: Varying step drop at r=4
    drops = [1.5, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    for drop in drops:
        lam = np.ones(d, dtype=float)
        lam[4:] = 1.0 / drop
        A = U @ np.diag(lam) @ U.T
        matrices.append({
            "name": f"Step (r=4, drop={drop})",
            "family": "Step Height Sweep",
            "param": drop,
            "eigenvalues": lam,
            "matrix": A,
            "true_trace": float(np.sum(lam)),
        })

    # 2. Step Spectra: Varying knee location (drop=100)
    for r_knee in [2, 4, 6]:
        lam = np.ones(d, dtype=float)
        lam[r_knee:] = 0.01
        A = U @ np.diag(lam) @ U.T
        matrices.append({
            "name": f"Step (r={r_knee}, drop=100)",
            "family": "Step Location Sweep",
            "param": r_knee,
            "eigenvalues": lam,
            "matrix": A,
            "true_trace": float(np.sum(lam)),
        })

    # 3. Power-Law Spectra: Varying exponent alpha
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for alpha in alphas:
        if alpha == 0.0:
            lam = np.ones(d, dtype=float)
        else:
            lam = (np.arange(1, d + 1, dtype=float)) ** (-alpha)
        A = U @ np.diag(lam) @ U.T
        matrices.append({
            "name": f"PowerLaw (alpha={alpha})",
            "family": "Power Law Sweep",
            "param": alpha,
            "eigenvalues": lam,
            "matrix": A,
            "true_trace": float(np.sum(lam)),
        })

    # 4. Exponential Spectra: Varying decay rate beta
    betas = [0.03, 0.08, 0.15, 0.30]
    for beta in betas:
        lam = np.exp(-beta * np.arange(d, dtype=float))
        A = U @ np.diag(lam) @ U.T
        matrices.append({
            "name": f"Exponential (beta={beta})",
            "family": "Exponential Sweep",
            "param": beta,
            "eigenvalues": lam,
            "matrix": A,
            "true_trace": float(np.sum(lam)),
        })

    # 5. Neural Network Spiked / Outlier Spectrum
    # Top 4 large outliers, followed by power-law bulk
    lam_neural = np.zeros(d, dtype=float)
    lam_neural[0] = 100.0
    lam_neural[1] = 50.0
    lam_neural[2] = 25.0
    lam_neural[3] = 10.0
    tail_idx = np.arange(1, d - 3, dtype=float)
    lam_neural[4:] = 1.0 / (tail_idx ** 1.2)
    A_neural = U @ np.diag(lam_neural) @ U.T
    matrices.append({
        "name": "Neural Spiked Bulk",
        "family": "ML Curvature",
        "param": 4,
        "eigenvalues": lam_neural,
        "matrix": A_neural,
        "true_trace": float(np.sum(lam_neural)),
    })

    return matrices


def extract_pilot_features(
    oracle: MatVecOracle,
    d: int,
    b_0: int = 8,
    rng: np.random.Generator | None = None,
) -> dict:
    """Extract observable pilot spectral features using exactly b_0 queries."""
    if rng is None:
        rng = np.random.default_rng()

    S_pilot = rng.choice([-1.0, 1.0], size=(d, b_0))
    W_pilot = oracle(S_pilot)
    scale = float(la.norm(W_pilot, ord="fro"))
    Q_pilot, r_pilot = _rank_aware_qr(W_pilot, reference_scale=scale)
    Z_pilot = oracle(Q_pilot) if r_pilot > 0 else np.empty((d, 0))

    if r_pilot >= 4:
        M = 0.5 * (Q_pilot.T @ Z_pilot + Z_pilot.T @ Q_pilot)
        ritz = la.eigvalsh(M)[::-1]
        theta_max = float(ritz[0]) if len(ritz) > 0 else 0.0
        pos_ritz = ritz[ritz > 1e-12 * theta_max] if theta_max > 0 else np.array([])
    else:
        pos_ritz = np.array([])

    if len(pos_ritz) >= 4:
        log_gaps = np.log(pos_ritz[:-1]) - np.log(pos_ritz[1:])
        gamma_gap = float(np.max(log_gaps))
        gap_loc = int(np.argmax(log_gaps) + 1)
        kappa = float(pos_ritz[0] / max(pos_ritz[-1], 1e-12))
        
        # Fit log-log decay slope
        k_fit = len(pos_ritz)
        idx_log = np.log(np.arange(1, k_fit + 1))
        val_log = np.log(pos_ritz)
        slope = float(np.polyfit(idx_log, val_log, deg=1)[0])

        top4_energy = float(np.sum(pos_ritz[:min(4, k_fit)]) / np.sum(pos_ritz))
    else:
        gamma_gap = 0.0
        gap_loc = 0
        kappa = 1.0
        slope = 0.0
        top4_energy = 0.5

    return {
        "gamma_gap": gamma_gap,
        "tau_ratio": float(np.exp(gamma_gap)),
        "gap_loc": gap_loc,
        "kappa": kappa,
        "decay_slope": slope,
        "top4_energy": top4_energy,
    }


def evaluate_fixed_q_hutch_pplus(
    oracle: MatVecOracle,
    m: int,
    d: int,
    q: int,
    rng: np.random.Generator,
) -> float:
    """Hutch++ with explicit subspace allocation q and ell = m - 2q residual probes."""
    ell = m - 2 * q
    if ell < 2 or q < 1:
        raise ValueError(f"Invalid q={q} for budget m={m}.")

    S = rng.choice([-1.0, 1.0], size=(d, q))
    W = oracle(S)
    scale = float(la.norm(W, ord="fro"))
    Q, r = _rank_aware_qr(W, reference_scale=scale)

    if r > 0:
        AQ = oracle(Q)
        tr_low = float(np.sum(Q * AQ))
    else:
        tr_low = 0.0

    G = rng.choice([-1.0, 1.0], size=(d, ell))
    RG = G - Q @ (Q.T @ G) if r > 0 else G
    ARG = oracle(RG)
    tr_res = float(np.sum(RG * ARG)) / ell

    return tr_low + tr_res


def run_gating_diagnostics(
    m: int = 60,
    d: int = 100,
    b_0: int = 8,
    tau_gap: float = 1.2,
    p_oversample: int = 2,
    n_pilot_trials: int = 50,
    n_eval_trials: int = 60,
    seed: int = 2026,
    output_dir: str | os.PathLike | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full gating diagnostics sweep."""
    print("=" * 70)
    print("STARTING PILOT FEATURE GATING DIAGNOSTICS & PREDICTABILITY MAP")
    print(f"Parameters: d={d}, m={m}, b_0={b_0}, tau_gap={tau_gap}, p_oversample={p_oversample}")
    print("=" * 70)

    matrices = build_test_matrices(d=d, seed=seed)
    rng = np.random.default_rng(seed)

    candidate_qs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    q_standard = m // 3  # 20

    pilot_records = []
    q_curve_records = []
    summary_records = []

    for mat_dict in matrices:
        mat_name = mat_dict["name"]
        family = mat_dict["family"]
        param = mat_dict["param"]
        A = mat_dict["matrix"]
        exact_tr = mat_dict["true_trace"]
        lam = mat_dict["eigenvalues"]
        oracle = MatVecOracle(A)

        # 1. Pilot Feature Sampling
        gaps, locs, kappas, slopes, energies = [], [], [], [], []
        for trial in range(n_pilot_trials):
            oracle.query_count = 0
            feats = extract_pilot_features(oracle, d=d, b_0=b_0, rng=rng)
            gaps.append(feats["gamma_gap"])
            locs.append(feats["gap_loc"])
            kappas.append(feats["kappa"])
            slopes.append(feats["decay_slope"])
            energies.append(feats["top4_energy"])

            pilot_records.append({
                "matrix_name": mat_name,
                "family": family,
                "param": param,
                "trial": trial,
                **feats,
            })

        mean_gap = float(np.mean(gaps))
        mean_kappa = float(np.mean(kappas))
        mean_slope = float(np.mean(slopes))
        mean_energy = float(np.mean(energies))

        # 2. Oracle Q-Risk Curve Sweep
        q_mse_dict = {}
        for q_cand in candidate_qs:
            sq_errs = []
            for _ in range(n_eval_trials):
                oracle.query_count = 0
                est = evaluate_fixed_q_hutch_pplus(oracle, m=m, d=d, q=q_cand, rng=rng)
                sq_errs.append((est - exact_tr) ** 2)
            mse_q = float(np.mean(sq_errs))
            q_mse_dict[q_cand] = mse_q

            q_curve_records.append({
                "matrix_name": mat_name,
                "family": family,
                "param": param,
                "q": q_cand,
                "mse": mse_q,
                "rmse": math.sqrt(mse_q),
                "rel_rmse": math.sqrt(mse_q) / exact_tr,
            })

        q_star = min(q_mse_dict.keys(), key=lambda q: q_mse_dict[q])
        mse_star = q_mse_dict[q_star]
        mse_std = q_mse_dict[q_standard]
        oracle_gain = mse_std / max(mse_star, 1e-14)

        # 3. TwoStageGated Performance Evaluation
        gated_sq_errs = []
        gated_triggers = []
        gated_qs = []
        for _ in range(n_eval_trials):
            oracle.query_count = 0
            est, diag = Adaptive_Hutch_pplus_TwoStageGated(
                oracle,
                m=m,
                d=d,
                b_0=b_0,
                tau_gap=tau_gap,
                p_oversample=p_oversample,
                rng=rng,
                return_diagnostics=True,
            )
            gated_sq_errs.append((est - exact_tr) ** 2)
            gated_triggers.append(bool(diag["is_gated_trigger"]))
            gated_qs.append(int(diag["q_target"]))

        mse_gated = float(np.mean(gated_sq_errs))
        trigger_rate = float(np.mean(gated_triggers))
        mean_q_gated = float(np.mean(gated_qs))
        achieved_gain = mse_std / max(mse_gated, 1e-14)

        # 4. Regime Classification
        if oracle_gain >= 1.30 and q_star <= 12:
            if trigger_rate >= 0.70:
                regime = "Obvious Adapt (Captured)"
            else:
                regime = "Under-triggered"
        elif oracle_gain <= 1.15:
            if trigger_rate <= 0.10:
                regime = "Benign Baseline (Protected)"
            else:
                regime = "False Trigger Risk"
        else:
            regime = "Transition / Mild Decay"

        print(
            f"[{mat_name:<30}] Mean Gap={mean_gap:.2f} | q*={q_star:>2} (Gain={oracle_gain:.2f}x) "
            f"| Gated Trigger={trigger_rate*100:>5.1f}% (Mean q={mean_q_gated:.1f}, Gain={achieved_gain:.2f}x) | {regime}"
        )

        summary_records.append({
            "matrix_name": mat_name,
            "family": family,
            "param": param,
            "true_trace": exact_tr,
            "mean_pilot_gamma_gap": mean_gap,
            "mean_pilot_tau_ratio": float(np.exp(mean_gap)),
            "mean_pilot_kappa": mean_kappa,
            "mean_pilot_decay_slope": mean_slope,
            "mean_pilot_energy_top4": mean_energy,
            "q_oracle_star": q_star,
            "mse_oracle_star": mse_star,
            "mse_standard_hutch": mse_std,
            "oracle_gain_ratio": oracle_gain,
            "gated_trigger_rate": trigger_rate,
            "mean_q_gated": mean_q_gated,
            "mse_gated": mse_gated,
            "achieved_gain_ratio": achieved_gain,
            "adaptation_efficiency": float(
                (mse_std - mse_gated) / max(mse_std - mse_star, 1e-14)
            ) if mse_std > mse_star else 1.0,
            "regime": regime,
        })

    df_pilot = pd.DataFrame(pilot_records)
    df_curves = pd.DataFrame(q_curve_records)
    df_summary = pd.DataFrame(summary_records)

    if output_dir is None:
        return df_pilot, df_curves, df_summary
    os.makedirs(output_dir, exist_ok=True)
    p_pilot = os.path.join(output_dir, "gating_diagnostics_pilot_features.csv")
    p_curves = os.path.join(output_dir, "gating_diagnostics_q_curves.csv")
    p_summary = os.path.join(output_dir, "gating_diagnostics_summary.csv")

    df_pilot.to_csv(p_pilot, index=False)
    df_curves.to_csv(p_curves, index=False)
    df_summary.to_csv(p_summary, index=False)

    print("\nSaved diagnostics artifacts:")
    print(f"  - {p_pilot}")
    print(f"  - {p_curves}")
    print(f"  - {p_summary}")

    return df_pilot, df_curves, df_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    run_gating_diagnostics(output_dir=parser.parse_args().output_dir)
