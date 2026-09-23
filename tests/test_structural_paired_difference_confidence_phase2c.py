"""Automated unit and regression tests for Phase 2C structural priors.

Verifies:
1. Theorems 16, 17, and 18 from proof_structural_paired_difference_confidence.md.
2. Exact mathematical scaling laws (M0^2 scaling, 1/sqrt(n) scaling).
3. Zero-variance boundary conditions.
4. Input validation and error handling.
5. End-to-end certification under realistic residual norm envelopes.
"""

import math
import pytest
import numpy as np

from src.structural_paired_difference_confidence import (
    PROVED_UNDER_PRIOR,
    norm_envelope_variance_bound,
    cantelli_structural_one_sided_radius,
    structural_chaos_fourth_moment_factor,
    chebyshev_scale_feasibility_threshold,
    independent_signed_pair_observations,
    structural_paired_difference_certificate,
)


def test_input_validation_errors():
    """Verify strict parameter validation."""
    with pytest.raises(ValueError, match="M0 must be finite and nonnegative"):
        norm_envelope_variance_bound(-1.0, 100, 100)

    with pytest.raises(ValueError, match="ell_candidate must be positive"):
        norm_envelope_variance_bound(1.0, 0, 100)

    with pytest.raises(ValueError, match="ell_baseline must be positive"):
        norm_envelope_variance_bound(1.0, 100, -5)

    with pytest.raises(ValueError, match="delta must lie strictly between zero and one"):
        cantelli_structural_one_sided_radius(16, 0.0, 1.0, 100, 100)

    with pytest.raises(ValueError, match="delta must lie strictly between zero and one"):
        cantelli_structural_one_sided_radius(16, 1.0, 1.0, 100, 100)

    with pytest.raises(ValueError, match="sample_size must be at least 2"):
        cantelli_structural_one_sided_radius(1, 0.05, 1.0, 100, 100)

    with pytest.raises(ValueError, match="kappa must lie in"):
        structural_chaos_fourth_moment_factor(0.0)

    with pytest.raises(ValueError, match="kappa must lie in"):
        structural_chaos_fourth_moment_factor(1.5)


def test_zero_variance_boundary():
    """Verify exact behavior at the M0 = 0 boundary."""
    var_bound = norm_envelope_variance_bound(0.0, 100, 100)
    assert var_bound == 0.0

    radius = cantelli_structural_one_sided_radius(16, 0.05, 0.0, 100, 100)
    assert radius == 0.0

    cands = [0.0, 0.0, 0.0, 0.0]
    bases = [0.0, 0.0, 0.0, 0.0]
    cert = structural_paired_difference_certificate(
        cands, bases, ell_candidate=100, ell_baseline=100, delta=0.05, norm_envelope_M0=0.0
    )
    assert cert.one_sided_radius == 0.0
    assert cert.upper_confidence_bound == cert.point_estimate
    assert cert.nonvacuous is True


def test_mathematical_scaling_laws():
    """Verify analytical scaling: radius scales quadratically with M0 and inversely with sqrt(n)."""
    ell_a, ell_0 = 80, 80
    delta = 0.05

    # 1. Quadratic scaling in M0
    r_1 = cantelli_structural_one_sided_radius(16, delta, 1.0, ell_a, ell_0)
    r_2 = cantelli_structural_one_sided_radius(16, delta, 2.0, ell_a, ell_0)
    assert math.isclose(r_2, 4.0 * r_1, rel_tol=1e-12)

    r_half = cantelli_structural_one_sided_radius(16, delta, 0.5, ell_a, ell_0)
    assert math.isclose(r_half, 0.25 * r_1, rel_tol=1e-12)

    # 2. 1/sqrt(n) scaling with sample size (s = 2n)
    r_s8 = cantelli_structural_one_sided_radius(8, delta, 1.0, ell_a, ell_0)  # n = 4
    r_s32 = cantelli_structural_one_sided_radius(32, delta, 1.0, ell_a, ell_0)  # n = 16
    # sqrt(16/4) = 2
    assert math.isclose(r_s8, 2.0 * r_s32, rel_tol=1e-12)

    # 3. Monotonicity in failure probability delta
    r_delta_01 = cantelli_structural_one_sided_radius(16, 0.01, 1.0, ell_a, ell_0)
    r_delta_05 = cantelli_structural_one_sided_radius(16, 0.05, 1.0, ell_a, ell_0)
    r_delta_10 = cantelli_structural_one_sided_radius(16, 0.10, 1.0, ell_a, ell_0)
    assert r_delta_01 > r_delta_05 > r_delta_10


def test_theorem17_chaos_fourth_moment_factor():
    """Verify Theorem 17: E[Z^4] <= (3 + 12*kappa^2) sigma^4."""
    # Worst case kappa = 1 gives 15.0
    assert math.isclose(structural_chaos_fourth_moment_factor(1.0), 15.0)

    # Asymptotic Gaussian limit kappa -> 0 gives 3.0
    assert math.isclose(structural_chaos_fourth_moment_factor(1e-6), 3.0, abs_tol=1e-10)

    # Strict monotonicity
    kappas = [0.1, 0.2, 0.5, 0.8, 1.0]
    factors = [structural_chaos_fourth_moment_factor(k) for k in kappas]
    for i in range(len(factors) - 1):
        assert factors[i] < factors[i + 1]
        assert 3.0 < factors[i] <= 15.0


def test_theorem18_chebyshev_feasibility_boundary():
    """Verify the strict sample-variance/Chebyshev boundary, including equality."""
    # Phase 2B data-only worst case: K = 6561, delta_scale = 0.05
    n_star_2b = chebyshev_scale_feasibility_threshold(6561.0, 0.05)
    assert n_star_2b == 131201

    # Low-coherence Gaussian limit: K = 3, delta_scale = 0.05
    n_star_gauss = chebyshev_scale_feasibility_threshold(3.0, 0.05)
    # At n=41 the bound equals 0.05 exactly; strict feasibility starts at 42.
    assert n_star_gauss == 42

    # Since 41 > 16 (for s = 32), Chebyshev data-only scale estimation is provably vacuous
    assert n_star_gauss > 16


def test_threshold_is_minimal_for_exact_decimal_inputs():
    from fractions import Fraction
    for K in (1, 1.2, 3, 15, 6561):
        for delta in (0.01, 0.025, 0.05, 0.1):
            n = chebyshev_scale_feasibility_threshold(K, delta)
            k, probability = Fraction(str(K)), Fraction(str(delta))
            def feasible(j):
                return (k - 1) * (j - 1) + 2 < probability * j * (j - 1)
            assert feasible(n)
            assert n == 2 or not feasible(n - 1)


def test_fourth_moment_expansion_by_exhaustive_rademacher_enumeration():
    from itertools import product
    rng = np.random.default_rng(192)
    for d in (2, 3, 4, 6):
        for _ in range(4):
            C = rng.normal(size=(d, d))
            C = (C + C.T) / 2
            np.fill_diagonal(C, 0)
            G = np.array(list(product((-1., 1.), repeat=d)))
            Z = np.einsum("bi,ij,bj->b", G, C, G)
            f2 = np.sum(C*C)
            row4 = np.sum(np.sum(C*C, axis=1)**2)
            exact = 12*f2*f2 + 48*np.trace(C@C@C@C) - 96*row4 + 32*np.sum(C**4)
            assert np.isclose(np.mean(Z**2), 2*f2)
            assert np.isclose(np.mean(Z**4), exact)
            kappa = np.linalg.norm(C, 2) / np.sqrt(f2)
            assert exact <= structural_chaos_fourth_moment_factor(kappa)*(2*f2)**2 + 1e-9


def test_independent_signed_pair_vector_checks():
    """Verify signed pair construction and arithmetic."""
    cands = [10.0, 6.0, 8.0, 4.0]
    bases = [12.0, 2.0, 14.0, 6.0]
    ell_a, ell_0 = 100, 100

    D = independent_signed_pair_observations(cands, bases, ell_a, ell_0)
    assert len(D) == 2

    # Pair 1: (10 - 6)^2 / 200 - (12 - 2)^2 / 200 = 16/200 - 100/200 = -84/200 = -0.42
    assert math.isclose(D[0], -0.42, rel_tol=1e-12)

    # Pair 2: (8 - 4)^2 / 200 - (14 - 6)^2 / 200 = 16/200 - 64/200 = -48/200 = -0.24
    assert math.isclose(D[1], -0.24, rel_tol=1e-12)


def test_end_to_end_certificate_evaluation():
    """Check API arithmetic; these synthetic arrays are not a coverage experiment."""
    np.random.seed(42)
    s = 16
    n = s // 2

    # Generate synthetic observations where candidate is significantly better than baseline
    # candidate residual ~ Normal(0, 0.01^2), baseline residual ~ Normal(0, 0.1^2)
    cands = np.random.normal(0, 0.01, size=s) ** 2
    bases = np.random.normal(0, 0.1, size=s) ** 2

    ell_a = 70
    ell_0 = 100
    delta = 0.05
    M0 = 0.05

    cert = structural_paired_difference_certificate(
        cands, bases, ell_candidate=ell_a, ell_baseline=ell_0, delta=delta, norm_envelope_M0=M0
    )

    assert cert.sample_size == s
    assert cert.independent_pair_count == n
    assert cert.delta == delta
    assert cert.nonvacuous is True
    assert cert.proof_status == PROVED_UNDER_PRIOR
    assert cert.one_sided_radius > 0.0
    assert cert.upper_confidence_bound == cert.point_estimate + cert.one_sided_radius

    # If point estimate is sufficiently negative and radius is small, it certifies safe
    if cert.upper_confidence_bound <= 0.0:
        assert cert.certified_safe is True
    else:
        assert cert.certified_safe is False
