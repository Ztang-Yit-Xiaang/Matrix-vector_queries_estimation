"""Unit tests for Haar orientation audit."""

import numpy as np
import pytest

from experiments.run_haar_orientation_audit import (
    sample_haar_orthogonal,
    build_spectrum,
    run_haar_orientation_audit,
)


def test_sample_haar_orthogonal():
    """Verify that sample_haar_orthogonal generates true orthonormal matrices."""
    rng = np.random.default_rng(42)
    d = 25
    U = sample_haar_orthogonal(d, rng)

    assert U.shape == (d, d)
    # Check orthogonality U^T U = I
    assert np.allclose(U.T @ U, np.eye(d), atol=1e-10)
    assert np.allclose(U @ U.T, np.eye(d), atol=1e-10)

    # Check determinant is +/- 1
    det = np.linalg.det(U)
    assert np.isclose(abs(det), 1.0, atol=1e-8)


def test_build_spectrum():
    """Verify spectrum construction for all supported families."""
    d = 50

    # Step spectrum
    ev_step = build_spectrum(d, "step", 5.0)
    assert len(ev_step) == d
    assert np.allclose(ev_step[:5], 1.0)
    assert np.allclose(ev_step[5:], 0.001)

    # Power law
    ev_pl = build_spectrum(d, "power_law", 1.0)
    assert len(ev_pl) == d
    assert ev_pl[0] == 1.0
    assert ev_pl[-1] == 1.0 / 50.0

    # Exponential
    ev_exp = build_spectrum(d, "exponential", 0.1)
    assert len(ev_exp) == d
    assert np.isclose(ev_exp[0], 1.0)
    assert np.all(np.diff(ev_exp) < 0)  # Monotonically decreasing


def test_haar_audit_smoke(tmp_path):
    """Smoke test running a tiny audit to verify dataframes and pipeline integrity."""
    d = 20
    m_budget = 30
    n_orientations = 2
    n_trials = 2

    df, df_summary = run_haar_orientation_audit(
        d=d,
        m_budget=m_budget,
        n_orientations=n_orientations,
        n_trials_per_orientation=n_trials,
        seed=123,
        output_dir=tmp_path,
    )

    assert not df.empty
    assert (tmp_path / "haar_orientation_audit_trials.csv").is_file()
    assert len(df) == 36
    assert not df_summary.empty
    assert "gated_vs_hpp_mse_ratio" in df_summary.columns
    assert "gate_trigger_rate" in df_summary.columns
    assert set(df["orient_type"].unique()) == {"Coordinate-Aligned (U=I)", "Haar-Random"}
