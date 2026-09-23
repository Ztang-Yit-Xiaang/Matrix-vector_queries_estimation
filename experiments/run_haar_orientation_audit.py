"""Haar Orientation Robustness Audit for Adaptive TwoStageGated Hutch++.

Evaluates whether the Stage 1 pilot screening, knee detection, and MSE performance
of Adaptive_Hutch_pplus_TwoStageGated are invariant to coordinate-basis orientation.

Experimental Protocol:
1. Freeze spectra Lambda across representative families:
   - Step spectrum (r* = 5, eta = 1e-3, abrupt knee)
   - Step spectrum (r* = 15, eta = 1e-2, intermediate knee)
   - Flat power law (c = 0.5, no knee)
   - Moderate power law (c = 1.0, smooth decay)
   - Steep power law (c = 2.0, rapid decay)
   - Exponential decay (alpha = 0.08)

2. For each spectrum, evaluate across:
   - Coordinate-aligned basis: U = I_d (extreme anisotropic case)
   - N_orient independent Haar-distributed orthogonal matrices U_j ~ Haar(O(d))

3. For each orientation and matrix A_j = U_j Lambda U_j^T, evaluate:
   - Gate trigger rate: does gamma_gap >= tau_gap trigger consistently across U?
   - Selected low-rank dimension q_target
   - Relative error and MSE for:
     * Classical Hutchinson
     * Standard Hutch++ (Rademacher)
     * TwoStageGated (Ours, Rademacher)
     * Standard Gaussian Hutch++ (as rotation-invariant baseline)
   - Off-diagonal vs Diagonal residual energy: sum_i (H_0)_{ii}^2 / ||H_0||_F^2
     (measuring the Rademacher-vs-Gaussian variance advantage across orientations).
"""

from __future__ import annotations

import os
import sys
import math
import numpy as np
import pandas as pd

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
)

RESULTS_DIR = os.path.join(project_dir, "results")


def sample_haar_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random orthogonal matrix uniformly distributed according to Haar measure on O(d).

    Follows the standard Mezzadri (2007) algorithm: QR decomposition of a standard
    Gaussian matrix with diagonal sign normalization.
    """
    Z = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(Z)
    diag_r = np.diagonal(R)
    phases = diag_r / np.abs(diag_r)
    return Q * phases


def build_spectrum(d: int, family: str, param: float) -> np.ndarray:
    """Construct eigenvalue vector for dimension d."""
    if family == "step":
        r_star = int(param)
        eta = 0.001 if r_star == 5 else 0.01
        evals = np.ones(d) * eta
        evals[:r_star] = 1.0
        return evals
    elif family == "power_law":
        c = float(param)
        return np.array([float(i) ** (-c) for i in range(1, d + 1)])
    elif family == "exponential":
        alpha = float(param)
        return np.array([math.exp(-alpha * (i - 1)) for i in range(1, d + 1)])
    else:
        raise ValueError(f"Unknown family {family}")


def run_haar_orientation_audit(
    d: int = 100,
    m_budget: int = 60,
    n_orientations: int = 30,
    n_trials_per_orientation: int = 10,
    seed: int = 2026,
    output_dir: str | os.PathLike | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the audit; write artifacts only to an explicitly supplied directory."""
    rng = np.random.default_rng(seed)

    cases = [
        ("Step (r*=5, eta=1e-3)", "step", 5.0),
        ("Step (r*=15, eta=1e-2)", "step", 15.0),
        ("PowerLaw (c=0.5 - Flat)", "power_law", 0.5),
        ("PowerLaw (c=1.0 - Moderate)", "power_law", 1.0),
        ("PowerLaw (c=2.0 - Steep)", "power_law", 2.0),
        ("Exponential (alpha=0.08)", "exponential", 0.08),
    ]

    records = []

    for case_name, family, param in cases:
        evals = build_spectrum(d, family, param)
        exact_trace = float(np.sum(evals))
        print(f"\n--- Testing Spectrum: {case_name} (tr(A) = {exact_trace:.4f}) ---")

        # Define orientations: Orientation 0 is Coordinate-Aligned (U = I), Orientations 1..n are Haar
        for orient_idx in range(n_orientations + 1):
            if orient_idx == 0:
                orient_type = "Coordinate-Aligned (U=I)"
                U = np.eye(d)
            else:
                orient_type = "Haar-Random"
                U = sample_haar_orthogonal(d, rng)

            A = U @ np.diag(evals) @ U.T

            # Measure baseline residual diagonal concentration on fixed q_0 = m // 3
            q_0 = m_budget // 3
            # We compute H_0 for the exact top-q_0 eigenspace to measure coordinate concentration
            # (I - U_q0 U_q0^T) A (I - U_q0 U_q0^T)
            H_exact_0 = U[:, q_0:] @ np.diag(evals[q_0:]) @ U[:, q_0:].T
            frob_sq = float(np.sum(H_exact_0 ** 2))
            diag_sq = float(np.sum(np.diagonal(H_exact_0) ** 2))
            diag_concentration = diag_sq / frob_sq if frob_sq > 0 else 0.0

            for trial in range(n_trials_per_orientation):
                # 1. Classical Hutchinson
                o_hutch = MatVecOracle(A)
                est_hutch = Hutchinson(o_hutch, m=m_budget, d=d, rng=rng)
                rel_err_hutch = abs(est_hutch - exact_trace) / exact_trace

                # 2. Standard Hutch++ (Rademacher)
                o_hpp = MatVecOracle(A)
                est_hpp = Hutch_pplus(o_hpp, m=m_budget, d=d, rng=rng)
                rel_err_hpp = abs(est_hpp - exact_trace) / exact_trace

                # 3. Standard Gaussian Hutch++ (Rotation-Invariant Control)
                o_ghpp = MatVecOracle(A)
                est_ghpp = Gaussian_Hutch_pplus(o_ghpp, m=m_budget, d=d, rng=rng)
                rel_err_ghpp = abs(est_ghpp - exact_trace) / exact_trace

                # 4. TwoStageGated (Ours, Rademacher)
                o_gated = MatVecOracle(A)
                est_gated, diag = Adaptive_Hutch_pplus_TwoStageGated(
                    o_gated,
                    m=m_budget,
                    d=d,
                    b_0=8,
                    tau_gap=1.5,
                    p_oversample=2,
                    rng=rng,
                    return_diagnostics=True,
                )
                rel_err_gated = abs(est_gated - exact_trace) / exact_trace

                assert all(o.query_count == m_budget for o in (o_hutch, o_hpp, o_ghpp, o_gated))

                records.append({
                    "case_name": case_name,
                    "orient_idx": orient_idx,
                    "orient_type": orient_type,
                    "trial": trial,
                    "seed": seed,
                    "dimension": d,
                    "budget_m": m_budget,
                    "query_count_per_method": m_budget,
                    "diag_concentration": diag_concentration,
                    "gate_triggered": bool(diag["is_gated_trigger"]),
                    "r_knee_detected": int(diag["gap_location"]),
                    "q_target": int(diag["q_target"]),
                    "rel_err_hutch": rel_err_hutch,
                    "rel_err_hpp": rel_err_hpp,
                    "rel_err_ghpp": rel_err_ghpp,
                    "rel_err_gated": rel_err_gated,
                    "sq_err_hutch": (est_hutch - exact_trace) ** 2,
                    "sq_err_hpp": (est_hpp - exact_trace) ** 2,
                    "sq_err_ghpp": (est_ghpp - exact_trace) ** 2,
                    "sq_err_gated": (est_gated - exact_trace) ** 2,
                })

    df = pd.DataFrame(records)

    # Compute aggregation across (case_name, orient_type)
    summary_rows = []
    for (case_name, orient_type), group in df.groupby(["case_name", "orient_type"]):
        mse_hpp = group["sq_err_hpp"].mean()
        mse_gated = group["sq_err_gated"].mean()
        mse_ghpp = group["sq_err_ghpp"].mean()
        mse_hutch = group["sq_err_hutch"].mean()

        summary_rows.append({
            "case_name": case_name,
            "orient_type": orient_type,
            "n_orientations": group["orient_idx"].nunique(),
            "n_total_trials": len(group),
            "gate_trigger_rate": group["gate_triggered"].mean(),
            "mean_q_target": group["q_target"].mean(),
            "mean_diag_concentration": group["diag_concentration"].mean(),
            "median_rel_err_hpp": group["rel_err_hpp"].median(),
            "median_rel_err_gated": group["rel_err_gated"].median(),
            "mse_hpp": mse_hpp,
            "mse_gated": mse_gated,
            "gated_vs_hpp_mse_ratio": mse_gated / mse_hpp if mse_hpp > 0 else np.nan,
            "rademacher_vs_gaussian_hpp_ratio": mse_hpp / mse_ghpp if mse_ghpp > 0 else np.nan,
        })

    df_summary = pd.DataFrame(summary_rows)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "haar_orientation_audit_trials.csv"), index=False)
        df_summary.to_csv(os.path.join(output_dir, "haar_orientation_audit_summary.csv"), index=False)

    return df, df_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    run_haar_orientation_audit(output_dir=parser.parse_args().output_dir)
