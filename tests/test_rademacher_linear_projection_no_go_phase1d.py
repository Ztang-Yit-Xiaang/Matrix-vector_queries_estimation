"""
Test suite for Phase 1D: Rademacher Linear Projection Lower-Tail Analytic No-Go.
"""

import itertools
import math
import os
import sys
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.rademacher_linear_projection_no_go import (
    FROZEN_SAMPLE_SIZES,
    FROZEN_JOINT_DELTAS,
    FROZEN_ALLOCATIONS,
    FROZEN_CAP_GRID,
    FROZEN_KAPPAS,
    FROZEN_EPSILONS,
    FOURTH_MOMENT_BOUND,
    VARIANCE_FLOOR_LOWER,
    MINIMUM_FROZEN_CAP,
    ANALYTIC_EXPONENT_DIVISOR,
    cortinovis_kressner_tail_bound,
    truncation_bias_upper_bound,
    capped_variance_envelope,
    one_sided_bernstein_lower_tail,
    analytic_probability_floor,
    directed_component_delta,
    linear_lower_tail_no_go_audit,
)


def test_exact_variance_formula_small_dim():
    """Exhaustive check of Var(g^T H g) == 2 ||C||_F^2 over all 2^d Rademacher vectors."""
    d = 4
    rng = np.random.default_rng(42)
    # Generate random symmetric matrix
    M = rng.normal(size=(d, d))
    H = (M + M.T) / 2.0
    C = H - np.diag(np.diag(H))
    exact_sigma2 = 2.0 * np.sum(C ** 2)

    all_g = list(itertools.product([-1.0, 1.0], repeat=d))
    values = []
    for g_tuple in all_g:
        g = np.array(g_tuple)
        values.append(float(g.T @ H @ g))

    emp_mean = np.mean(values)
    emp_var = np.mean((values - emp_mean) ** 2)

    assert math.isclose(emp_var, exact_sigma2, rel_tol=1e-10)


def test_zero_variance_branch():
    """When C is diagonal, off-diagonal elements are 0, so Var(g^T H g) == 0."""
    d = 4
    H = np.diag([1.0, 2.0, 3.0, 4.0])
    C = H - np.diag(np.diag(H))
    assert np.allclose(C, 0.0)
    exact_sigma2 = 2.0 * np.sum(C ** 2)
    assert exact_sigma2 == 0.0


def test_normalized_tail_substitution():
    """The dimensionless formula must equal the dimensional CK expression."""
    frobenius_norm = 3.25
    kappa = 0.4
    spectral_norm = kappa * frobenius_norm
    sigma = math.sqrt(2.0) * frobenius_norm
    for u in (0.25, 1.0, 3.0):
        t = u * sigma
        dimensional = min(
            1.0,
            2.0 * math.exp(
                -(t * t)
                / (8.0 * frobenius_norm**2 + 8.0 * t * spectral_norm)
            ),
        )
        normalized = cortinovis_kressner_tail_bound(u * u, kappa)
        assert math.isclose(dimensional, normalized, rel_tol=1e-14)


def test_hoeffding_linear_factor_identity():
    """Protect L_s=(1/s)sum(Z_i^2/sigma^2-1), including the factor two."""
    z = np.array([1.0, -2.0, 3.0, -4.0])
    sigma2 = 5.0
    h1 = (z**2 - sigma2) / 2.0
    from_projection = (2.0 / len(z)) * np.sum(h1) / sigma2
    from_normalized_squares = np.mean(z**2 / sigma2 - 1.0)
    assert math.isclose(from_projection, from_normalized_squares, rel_tol=1e-14)


def test_cortinovis_kressner_tail_bounds():
    """Verify tail bound is in [0, 1] and strictly decreases with v."""
    for kappa in FROZEN_KAPPAS:
        prev_p = 1.0
        for v in [0.1, 1.0, 5.0, 20.0, 100.0]:
            p = cortinovis_kressner_tail_bound(v, kappa)
            assert 0.0 <= p <= 1.0
            assert p <= prev_p
            prev_p = p


def test_truncation_bias_properties():
    """Verify truncation bias beta_kappa(T) > 0 and strictly decreases in T."""
    for kappa in FROZEN_KAPPAS:
        prev_beta = float("inf")
        for T in FROZEN_CAP_GRID:
            beta = truncation_bias_upper_bound(T, kappa)
            assert beta > 0.0
            assert beta < prev_beta
            prev_beta = beta


def test_truncation_bias_closed_form_integral():
    """The implemented beta is the exact integral of the exponential envelope."""
    for T, kappa in ((64.0, 1.0), (256.0, 0.25), (16384.0, 0.1)):
        sqrt_T = math.sqrt(T)
        a = 1.0 / (4.0 * math.sqrt(2.0) * kappa + 4.0 / sqrt_T)
        # Substitute sqrt(v)=sqrt(T)+w and remove exp(-a*sqrt(T)).
        # This avoids an inaccurate absolute-error calculation when beta is
        # around 1e-90 on the largest frozen caps.
        numerical_scaled, _ = quad(
            lambda w: 4.0 * (sqrt_T + w) * math.exp(-a * w),
            0.0,
            np.inf,
        )
        closed = truncation_bias_upper_bound(T, kappa)
        closed_scaled = closed / math.exp(-a * sqrt_T)
        assert math.isclose(numerical_scaled, closed_scaled, rel_tol=1e-10)


def test_variance_envelope_floor():
    """Verify 80.0 <= nu_kappa(T) <= 81.0 across all frozen caps and kappas."""
    for T in FROZEN_CAP_GRID:
        for kappa in FROZEN_KAPPAS:
            nu = capped_variance_envelope(T, kappa)
            assert VARIANCE_FLOOR_LOWER <= nu <= FOURTH_MOMENT_BOUND


def test_analytic_probability_floor():
    """Verify probability floor decreases strictly with s and equals exp(-s/160)."""
    assert math.isclose(analytic_probability_floor(32), math.exp(-0.2), rel_tol=1e-10)
    assert math.isclose(analytic_probability_floor(16), math.exp(-0.1), rel_tol=1e-10)
    assert analytic_probability_floor(32) < analytic_probability_floor(16) < analytic_probability_floor(4)


def test_one_sided_bernstein_dominates_floor():
    """Verify D_{s, kappa}(eps, T) > exp(-s/160) for all admissible (s, eps, T, kappa)."""
    for s in FROZEN_SAMPLE_SIZES:
        floor = analytic_probability_floor(s)
        for T in FROZEN_CAP_GRID:
            for kappa in FROZEN_KAPPAS:
                for eps in FROZEN_EPSILONS:
                    val = one_sided_bernstein_lower_tail(s, eps, T, kappa)
                    assert val > floor


def test_inadmissible_cap_reports_one():
    """A cap whose bias already exceeds epsilon provides no certificate."""
    assert one_sided_bernstein_lower_tail(32, 0.1, 64.0, 1.0) == 1.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: cortinovis_kressner_tail_bound(float("nan"), 0.5),
        lambda: cortinovis_kressner_tail_bound(1.0, 0.0),
        lambda: truncation_bias_upper_bound(float("inf"), 0.5),
        lambda: capped_variance_envelope(63.0, 0.5),
        lambda: one_sided_bernstein_lower_tail(True, 0.5, 64.0, 0.5),
        lambda: one_sided_bernstein_lower_tail(32, 1.0, 64.0, 0.5),
        lambda: directed_component_delta(float("nan"), "optimistic"),
        lambda: directed_component_delta(0.05, "unknown"),
    ],
)
def test_parameter_validation(call):
    with pytest.raises(ValueError):
        call()


def test_directed_component_delta_split():
    """Verify directed failure probability allocation."""
    assert directed_component_delta(0.05, "final_compatible") == 0.0125
    assert directed_component_delta(0.05, "optimistic") == 0.025
    assert directed_component_delta(0.10, "final_compatible") == 0.025
    assert directed_component_delta(0.10, "optimistic") == 0.05


def test_linear_lower_tail_no_go_audit_verdict():
    """Verify all 24 theorem combinations deterministically yield STRONG LINEAR NO-GO."""
    for s in FROZEN_SAMPLE_SIZES:
        for delta in FROZEN_JOINT_DELTAS:
            for alloc in FROZEN_ALLOCATIONS:
                audit = linear_lower_tail_no_go_audit(s, delta, alloc)
                assert audit.verdict == "STRONG LINEAR NO-GO"
                assert audit.theorem_status == "PROVED"
                assert audit.analytic_probability_floor > audit.directed_component_delta
                assert audit.analytic_probability_floor > audit.maximum_declared_component_delta


def test_artifacts_on_disk():
    """Verify the 4 generated CSV files exist with exact row counts."""
    results_dir = os.path.join(project_dir, "results")

    p_manifest = os.path.join(results_dir, "rademacher_linear_projection_no_go_phase1d_manifest.csv")
    p_theorem = os.path.join(results_dir, "rademacher_linear_projection_no_go_phase1d_theorem_grid.csv")
    p_cap = os.path.join(results_dir, "rademacher_linear_projection_no_go_phase1d_cap_regression.csv")
    p_verdict = os.path.join(results_dir, "rademacher_linear_projection_no_go_phase1d_verdict.csv")

    assert os.path.exists(p_manifest), f"Missing {p_manifest}"
    assert os.path.exists(p_theorem), f"Missing {p_theorem}"
    assert os.path.exists(p_cap), f"Missing {p_cap}"
    assert os.path.exists(p_verdict), f"Missing {p_verdict}"

    df_theorem = pd.read_csv(p_theorem)
    assert len(df_theorem) == 24

    df_cap = pd.read_csv(p_cap)
    assert len(df_cap) == 900

    df_verdict = pd.read_csv(p_verdict)
    assert len(df_verdict) == 1
    assert df_verdict["verdict"].iloc[0] == "STRONG LINEAR NO-GO"

    df_manifest = pd.read_csv(p_manifest)
    assert len(df_manifest) == 1
    assert df_manifest["new_matvec_queries"].iloc[0] == 0
    assert "frozen_input_checksums" in df_manifest.columns
