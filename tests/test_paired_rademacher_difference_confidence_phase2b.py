"""Mathematical and API tests for the Phase 2B paired confidence audit."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.paired_rademacher_difference_confidence import (
    COMPLETE_U_METHOD,
    ELEMENTARY_METHOD,
    FOURTH_MOMENT_FACTOR,
    FROZEN_JOINT_DELTAS,
    FROZEN_SAMPLE_SIZES,
    HYPERCONTRACTIVE_L4_L2_FACTOR,
    INCOMPLETE,
    METHOD_PRIORITY,
    PAIRWISE_SCALE_RELATIVE_VARIANCE_FACTOR,
    PROVED_BUT_BUDGET_VACUOUS,
    ROBUST_MOM_METHOD,
    direct_paired_net_safe_decision,
    elementary_scale_relative_radius,
    exact_signed_pair_target,
    hypercontractive_fourth_moment_factor,
    hypercontractive_l4_l2_factor,
    independent_signed_pair_observations,
    minimum_scale_block_size,
    paired_risk_difference_certificate,
    pairwise_scale_relative_variance_bound,
    phase2b_final_verdict,
    phase2b_route_audit,
    sample_variance_relative_variance_bound,
)
from src.paired_rademacher_risk_difference import (
    hoeffding_kernel_components,
    paired_u_statistic,
    sample_variance_difference,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _quadratic_values(matrix: np.ndarray) -> np.ndarray:
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=matrix.shape[0])))
    return np.einsum("bi,ij,bj->b", signs, matrix, signs)


def test_signed_pair_exact_conditional_mean_by_enumeration():
    candidate_matrix = np.array([[1.0, 0.7], [0.7, 2.0]])
    baseline_matrix = np.array([[0.5, -0.2], [-0.2, 1.5]])
    xa = _quadratic_values(candidate_matrix)
    x0 = _quadratic_values(baseline_matrix)
    ell_a, ell_0 = 7, 11
    exact_target = exact_signed_pair_target(
        np.var(xa), np.var(x0), ell_a, ell_0
    )
    pair_values = []
    for left in range(xa.size):
        for right in range(xa.size):
            pair_values.append(
                (xa[left] - xa[right]) ** 2 / (2.0 * ell_a)
                - (x0[left] - x0[right]) ** 2 / (2.0 * ell_0)
            )
    assert math.isclose(np.mean(pair_values), exact_target, rel_tol=1e-14, abs_tol=1e-14)


def test_disjoint_signed_pair_constructor_and_common_probe_dependence():
    candidate = np.array([1.0, 3.0, 2.0, 8.0])
    baseline = np.array([4.0, 5.0, -1.0, 2.0])
    observed = independent_signed_pair_observations(candidate, baseline, 2, 3)
    expected = np.array([(1.0 - 3.0) ** 2 / 4.0 - (4.0 - 5.0) ** 2 / 6.0,
                         (2.0 - 8.0) ** 2 / 4.0 - (-1.0 - 2.0) ** 2 / 6.0])
    assert np.allclose(observed, expected)


def test_hypercontractive_constant_and_exact_small_polynomial_check():
    assert hypercontractive_l4_l2_factor(4) == HYPERCONTRACTIVE_L4_L2_FACTOR == 9.0
    assert hypercontractive_fourth_moment_factor(4) == FOURTH_MOMENT_FACTOR == 6561.0
    # An explicit centered degree-at-most-four polynomial over four signs.
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=4)))
    values = (
        signs[:, 0] * signs[:, 1]
        + 0.5 * signs[:, 2] * signs[:, 3]
        + 0.25 * np.prod(signs, axis=1)
    )
    centered = values - np.mean(values)
    assert np.mean(centered**4) <= FOURTH_MOMENT_FACTOR * np.mean(centered**2) ** 2


def test_sample_variance_scale_audit_is_vacuous_on_full_grid():
    for sample_size in FROZEN_SAMPLE_SIZES:
        n = sample_size // 2
        for joint_delta in FROZEN_JOINT_DELTAS:
            radius = elementary_scale_relative_radius(n, joint_delta / 2.0)
            assert radius > 1.0
            audit = phase2b_route_audit(sample_size, joint_delta, ELEMENTARY_METHOD)
            assert audit.proof_status == PROVED_BUT_BUDGET_VACUOUS
            assert not audit.nonvacuous
            assert math.isinf(audit.observable_radius)


def test_pairwise_scale_moment_derivation_by_enumeration():
    centered = np.array([-3.0, -1.0, 1.0, 3.0])
    variance = np.mean(centered**2)
    fourth = np.mean(centered**4)
    kurtosis = fourth / variance**2
    scale_values = [
        (left - right) ** 2 / 2.0 for left in centered for right in centered
    ]
    exact_relative_variance = np.var(scale_values) / variance**2
    assert exact_relative_variance <= pairwise_scale_relative_variance_bound(kurtosis) + 1e-14
    assert pairwise_scale_relative_variance_bound() == PAIRWISE_SCALE_RELATIVE_VARIANCE_FACTOR


def test_robust_mom_scale_requirement_exceeds_available_budget():
    assert minimum_scale_block_size() == 13125
    for sample_size in FROZEN_SAMPLE_SIZES:
        audit = phase2b_route_audit(sample_size, 0.05, ROBUST_MOM_METHOD)
        assert audit.proof_status == PROVED_BUT_BUDGET_VACUOUS
        assert audit.minimum_scale_block_size == 13125
        assert sample_size // 4 < audit.minimum_scale_block_size


def test_complete_u_route_stops_before_canonical_theorem():
    audit = phase2b_route_audit(16, 0.05, COMPLETE_U_METHOD)
    assert audit.proof_status == INCOMPLETE
    assert "linear projection" in audit.stopping_reason
    assert "K2" in audit.stopping_reason


def test_complete_u_equals_sample_variance_difference():
    candidate = np.array([1.0, 4.0, -2.0, 7.0, 3.0])
    baseline = np.array([2.0, -1.0, 5.0, 0.0, 6.0])
    left = paired_u_statistic(candidate, baseline, 13, 17)
    right = sample_variance_difference(candidate, baseline, 13, 17)
    assert math.isclose(left, right, rel_tol=1e-14, abs_tol=1e-14)


def test_complete_hoeffding_kernel_degeneracy_and_nonzero_linear_part():
    za = np.array([-2.0, -1.0, 1.0, 2.0])
    z0 = np.array([-3.0, -1.0, 1.0, 3.0])
    sigma_a = np.mean(za**2)
    sigma_0 = np.mean(z0**2)
    linear_values = []
    conditional_k2 = []
    fixed_left = 0
    for right in range(za.size):
        k1, _, k2 = hoeffding_kernel_components(
            za[fixed_left], z0[fixed_left], za[right], z0[right],
            sigma_a, sigma_0, 7, 11,
        )
        linear_values.append(k1)
        conditional_k2.append(k2)
    assert not np.allclose(linear_values, 0.0)
    assert math.isclose(np.mean(conditional_k2), 0.0, abs_tol=1e-14)


def test_observed_zero_sample_does_not_claim_population_zero():
    certificate = paired_risk_difference_certificate(
        np.zeros(8), joint_delta=0.05, method=ROBUST_MOM_METHOD
    )
    assert certificate.zero_variance_observed_only
    assert certificate.proof_status == PROVED_BUT_BUDGET_VACUOUS
    assert math.isinf(certificate.upper_confidence_bound)
    assert direct_paired_net_safe_decision(certificate) == "abstain"


def test_all_frozen_routes_issue_no_replay_verdict():
    assert phase2b_final_verdict() == PROVED_BUT_BUDGET_VACUOUS
    for method in METHOD_PRIORITY:
        for sample_size in FROZEN_SAMPLE_SIZES:
            for joint_delta in FROZEN_JOINT_DELTAS:
                audit = phase2b_route_audit(sample_size, joint_delta, method)
                assert audit.new_matvec_queries == 0


@pytest.mark.parametrize(
    "call",
    [
        lambda: hypercontractive_l4_l2_factor(True),
        lambda: sample_variance_relative_variance_bound(1),
        lambda: sample_variance_relative_variance_bound(8, 0.5),
        lambda: elementary_scale_relative_radius(8, 1.0),
        lambda: minimum_scale_block_size(0.0),
        lambda: minimum_scale_block_size(0.25, 1.1),
        lambda: phase2b_route_audit(3, 0.05, ELEMENTARY_METHOD),
        lambda: phase2b_route_audit(16, 0.05, "unknown"),
        lambda: independent_signed_pair_observations([1, 2, 3], [1, 2, 3], 2, 2),
        lambda: exact_signed_pair_target(-1.0, 1.0, 2, 2),
    ],
)
def test_parameter_validation(call):
    with pytest.raises(ValueError):
        call()


def test_generated_artifacts():
    results = PROJECT_DIR / "results"
    route_path = results / "paired_rademacher_difference_confidence_phase2b_route_grid.csv"
    verdict_path = results / "paired_rademacher_difference_confidence_phase2b_verdict.csv"
    manifest_path = results / "paired_rademacher_difference_confidence_phase2b_manifest.csv"
    assert route_path.exists() and verdict_path.exists() and manifest_path.exists()
    route_grid = pd.read_csv(route_path)
    assert len(route_grid) == 36
    assert set(route_grid["method"]) == set(METHOD_PRIORITY)
    verdict = pd.read_csv(verdict_path)
    assert len(verdict) == 1
    assert verdict.loc[0, "verdict"] == PROVED_BUT_BUDGET_VACUOUS
    manifest = pd.read_csv(manifest_path)
    assert int(manifest.loc[0, "new_matvec_queries"]) == 0
