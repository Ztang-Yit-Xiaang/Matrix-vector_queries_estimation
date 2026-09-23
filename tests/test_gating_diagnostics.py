"""Tests for Pilot Feature Gating Diagnostics and Predictability Map."""

import numpy as np
import pytest
from experiments.run_gating_diagnostics import (
    build_test_matrices,
    extract_pilot_features,
    evaluate_fixed_q_hutch_pplus,
    run_gating_diagnostics,
)
from src.trace_baseline import MatVecOracle


def test_build_test_matrices_shape_and_trace():
    matrices = build_test_matrices(d=20, seed=42)
    assert len(matrices) > 10
    for m in matrices:
        assert m["matrix"].shape == (20, 20)
        assert np.isfinite(m["true_trace"])
        assert len(m["eigenvalues"]) == 20


def test_extract_pilot_features_properties():
    A = np.diag([10.0, 10.0, 1.0, 0.1, 0.01, 0.01, 0.01, 0.01])
    oracle = MatVecOracle(A)
    feats = extract_pilot_features(oracle, d=8, b_0=6, rng=np.random.default_rng(42))
    assert feats["gamma_gap"] >= 0.0
    assert feats["tau_ratio"] >= 1.0
    assert feats["kappa"] >= 1.0
    assert 0.0 <= feats["top4_energy"] <= 1.0


def test_evaluate_fixed_q_hutch_pplus():
    A = np.eye(30)
    oracle = MatVecOracle(A)
    est = evaluate_fixed_q_hutch_pplus(oracle, m=20, d=30, q=6, rng=np.random.default_rng(42))
    assert np.isfinite(est)
    assert abs(est - 30.0) < 5.0
