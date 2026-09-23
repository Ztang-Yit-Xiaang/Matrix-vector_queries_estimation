"""
Unit and regression tests for Adaptive_Hutch_pplus_TwoStageGated.
"""

import math
import os
import sys
import numpy as np
import pytest

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.trace_baseline import MatVecOracle, Adaptive_Hutch_pplus_TwoStageGated, Hutch_pplus


def test_twostage_gated_exact_query_budget():
    """Verify that TwoStageGated strictly satisfies query_count == m across multiple budgets."""
    d = 100
    rng = np.random.default_rng(123)
    M = rng.normal(size=(d, d))
    A = M.T @ M

    for m in [40, 60, 90, 120]:
        oracle = MatVecOracle(A)
        tr_est, diag = Adaptive_Hutch_pplus_TwoStageGated(
            oracle, m=m, d=d, b_0=8, tau_gap=1.5, p_oversample=1, rng=rng, return_diagnostics=True
        )
        assert oracle.query_count == m
        assert diag["q_target"] + diag["r_actual"] + diag["ell_eff"] == m


def test_twostage_gated_unbiasedness():
    """Verify zero systematic bias on synthetic PSD matrix over 200 trials."""
    d = 50
    rng = np.random.default_rng(42)
    evals = np.array([float(i) ** (-1.5) for i in range(1, d + 1)])
    V, _ = np.linalg.qr(rng.normal(size=(d, d)))
    A = V @ np.diag(evals) @ V.T
    exact_trace = float(np.trace(A))

    m = 40
    estimates = []
    for _ in range(200):
        oracle = MatVecOracle(A)
        est = Adaptive_Hutch_pplus_TwoStageGated(oracle, m=m, d=d, b_0=8, rng=rng)
        estimates.append(est)
    
    mean_est = float(np.mean(estimates))
    rel_bias = abs(mean_est - exact_trace) / exact_trace
    assert rel_bias < 0.05, f"Relative bias {rel_bias} is unexpectedly large"


def test_twostage_gated_step_knee_trigger():
    """Verify that a sharp step spectrum triggers Stage 2 and sets q_target to r_knee + p > b_0."""
    d = 100
    rng = np.random.default_rng(999)
    r_star = 6
    eta = 0.001
    evals = np.ones(d) * eta
    evals[:r_star] = 1.0
    V, _ = np.linalg.qr(rng.normal(size=(d, d)))
    A = V @ np.diag(evals) @ V.T

    m = 60
    oracle = MatVecOracle(A)
    est, diag = Adaptive_Hutch_pplus_TwoStageGated(
        oracle, m=m, d=d, b_0=8, tau_gap=1.5, p_oversample=3, rng=rng, return_diagnostics=True
    )
    assert diag["is_gated_trigger"] is True
    assert diag["gap_location"] == r_star
    assert diag["q_target"] == r_star + 3  # 6 + 3 = 9 > b_0 = 8



def test_twostage_gated_benign_smooth_fallback():
    """Verify that a smooth power-law spectrum triggers benign fallback to standard q_0."""
    d = 100
    rng = np.random.default_rng(777)
    evals = np.array([float(i) ** (-0.5) for i in range(1, d + 1)])
    V, _ = np.linalg.qr(rng.normal(size=(d, d)))
    A = V @ np.diag(evals) @ V.T

    m = 60
    oracle = MatVecOracle(A)
    est, diag = Adaptive_Hutch_pplus_TwoStageGated(
        oracle, m=m, d=d, b_0=8, tau_gap=1.5, p_oversample=1, rng=rng, return_diagnostics=True
    )
    assert diag["is_gated_trigger"] is False
    assert diag["q_target"] == diag["q_0"]  # Seamlessly falls back to m // 3 = 20
