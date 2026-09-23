from __future__ import annotations

import dataclasses
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
for directory in (SRC_DIR, EXPERIMENTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from rademacher_sample_variance_confidence import (  # noqa: E402
    CHEBYSHEV_METHOD,
    INCOMPLETE,
    PROVED,
    PROVED_BUT_BUDGET_VACUOUS,
    SHARPER_HOEFFDING_METHOD,
    assert_degenerate_theorem_scope,
    chebyshev_joint_relative_radius,
    chebyshev_pointwise_relative_radius,
    chebyshev_radius_is_decreasing,
    hoeffding_decomposition,
    hoeffding_degenerate_kernel,
    hoeffding_kernel,
    hoeffding_linear_projection,
    hypercontractive_sample_variance_relative_variance_bound,
    net_safe_decision,
    sample_variance_exact_moments,
    select_gate_sample_size,
    sharper_hoeffding_theorem_audit,
    two_action_sample_variance_certificate,
)
from postprocess_rademacher_sample_variance_confidence_phase1c import (  # noqa: E402
    JOINT_DELTA_GRID,
    SAMPLE_SIZE_GRID,
    SOURCE_PATHS,
    VERDICT_BUDGET_VACUOUS,
    build_certificate_grid,
    build_verdict,
    evaluate_gate,
    load_frozen_context,
    run_audit,
    validate_frozen_sources,
)


def _quadratic_form_distribution(matrix):
    matrix = np.asarray(matrix, dtype=float)
    assert np.allclose(matrix, matrix.T)
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=matrix.shape[0]):
        vector = np.asarray(signs)
        values.append(float(vector @ matrix @ vector))
    return np.asarray(values)


def _enumerated_sample_variances(population, sample_size):
    result = []
    for indices in itertools.product(range(len(population)), repeat=sample_size):
        sample = population[np.asarray(indices)]
        result.append(float(np.var(sample, ddof=1)))
    return np.asarray(result)


def test_exact_sample_variance_expectation_and_variance_by_enumeration():
    matrix = np.array([[1.0, 0.75], [0.75, -0.5]])
    population = _quadratic_form_distribution(matrix)
    mean = population.mean()
    variance = population.var()
    fourth = np.mean((population - mean) ** 4)
    for sample_size in (2, 3, 4):
        samples = _enumerated_sample_variances(population, sample_size)
        expected_mean, expected_variance = sample_variance_exact_moments(
            variance, fourth, sample_size
        )
        assert samples.mean() == pytest.approx(expected_mean, rel=1e-13, abs=1e-13)
        assert samples.var() == pytest.approx(
            expected_variance, rel=1e-13, abs=1e-13
        )


def test_degree_two_rademacher_chaos_obeys_retained_81_factor():
    matrices = (
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array(
            [
                [0.0, 1.0, -0.5, 0.25],
                [1.0, 0.0, 0.75, -1.5],
                [-0.5, 0.75, 0.0, 0.4],
                [0.25, -1.5, 0.4, 0.0],
            ]
        ),
    )
    for matrix in matrices:
        values = _quadratic_form_distribution(matrix)
        centered = values - values.mean()
        sigma2 = np.mean(centered**2)
        mu4 = np.mean(centered**4)
        assert sigma2 > 0.0
        assert mu4 <= 81.0 * sigma2**2 + 1e-12


def test_chebyshev_formula_joint_split_monotonicity_and_grid_vacuity():
    sizes = tuple(range(2, 33))
    assert chebyshev_radius_is_decreasing(sizes)
    for sample_size in sizes:
        pointwise = chebyshev_pointwise_relative_radius(sample_size, 0.025)
        joint = chebyshev_joint_relative_radius(sample_size, 0.05)
        assert joint == pytest.approx(pointwise)
        assert joint > 1.0
    assert chebyshev_joint_relative_radius(32, 0.05) > 10.0
    assert select_gate_sample_size(SAMPLE_SIZE_GRID, 0.05) is None


def test_exact_relative_variance_bound_matches_formula():
    for sample_size in (2, 4, 8, 16, 32):
        expected = (80.0 + 2.0 / (sample_size - 1.0)) / sample_size
        assert hypercontractive_sample_variance_relative_variance_bound(
            sample_size
        ) == pytest.approx(expected)


def test_hoeffding_decomposition_holds_term_by_term():
    population = np.array([-2.0, 0.0, 1.0, 5.0])
    mean = float(population.mean())
    variance = float(population.var())
    sample = np.array([-2.0, 1.0, 5.0])
    decomposition = hoeffding_decomposition(sample, mean, variance)
    assert decomposition["total"] == pytest.approx(
        decomposition["sample_variance_minus_variance"]
    )
    for left in population:
        for right in population:
            reconstructed = (
                variance
                + hoeffding_linear_projection(left, mean, variance)
                + hoeffding_linear_projection(right, mean, variance)
                + hoeffding_degenerate_kernel(left, right, mean)
            )
            assert reconstructed == pytest.approx(hoeffding_kernel(left, right))


def test_h2_is_degenerate_while_h1_can_be_nondegenerate():
    population = np.array([-2.0, 0.0, 1.0, 5.0])
    mean = float(population.mean())
    variance = float(population.var())
    for fixed in population:
        conditional_mean = np.mean(
            [hoeffding_degenerate_kernel(fixed, value, mean) for value in population]
        )
        assert conditional_mean == pytest.approx(0.0, abs=1e-14)
    h1_values = np.array(
        [hoeffding_linear_projection(value, mean, variance) for value in population]
    )
    assert h1_values.var() > 0.0
    assert_degenerate_theorem_scope("h2")
    for invalid in ("full_kernel", "h", "h1"):
        with pytest.raises(ValueError, match="only to h2"):
            assert_degenerate_theorem_scope(invalid)


def test_joint_delta_api_and_component_allocations_are_unambiguous():
    certificate = two_action_sample_variance_certificate(16, 0.05)
    assert certificate.joint_delta == 0.05
    assert certificate.action_count == 2
    assert certificate.per_action_delta == 0.025
    assert certificate.method_selected == CHEBYSHEV_METHOD
    assert certificate.proof_status == PROVED_BUT_BUDGET_VACUOUS
    assert "0.94999999999999996" in certificate.simultaneous_event
    large_sample = two_action_sample_variance_certificate(100_000, 0.05)
    assert large_sample.proof_status == PROVED
    assert large_sample.nonvacuous
    audit = sharper_hoeffding_theorem_audit(0.05)
    assert audit.proof_status == INCOMPLETE
    assert 2 * audit.linear_component_delta + 2 * audit.degenerate_component_delta == pytest.approx(0.05)
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        two_action_sample_variance_certificate(
            16, 0.05, method=SHARPER_HOEFFDING_METHOD
        )


def test_tiny_two_action_simultaneous_event_meets_the_conservative_bound():
    population_a = _quadratic_form_distribution(
        np.array([[0.0, 0.5], [0.5, 0.0]])
    )
    population_0 = _quadratic_form_distribution(
        np.array([[0.0, 1.0], [1.0, 0.0]])
    )
    sample_size = 2
    epsilon = chebyshev_joint_relative_radius(sample_size, 0.5)
    sigma_a = population_a.var()
    sigma_0 = population_0.var()
    failures = 0
    total = 0
    for ia in itertools.product(range(4), repeat=sample_size):
        sample_a = population_a[np.asarray(ia)]
        sa = np.var(sample_a, ddof=1)
        for i0 in itertools.product(range(4), repeat=sample_size):
            sample_0 = population_0[np.asarray(i0)]
            s0 = np.var(sample_0, ddof=1)
            event = (
                abs(sa - sigma_a) <= epsilon * sigma_a
                and abs(s0 - sigma_0) <= epsilon * sigma_0
            )
            failures += int(not event)
            total += 1
    assert failures / total <= 0.5


def test_net_safe_decision_uses_original_baseline_denominator():
    base = two_action_sample_variance_certificate(16, 0.05)
    useful = dataclasses.replace(
        base,
        proof_status=PROVED,
        epsilon=0.2,
        nonvacuous=True,
    )
    # rho=2/3 and ell_paid/ell_original=0.8, so the threshold is 0.5333.
    assert net_safe_decision(0.5, 1.0, 80, 100, useful) == "accept"
    assert net_safe_decision(0.6, 1.0, 80, 100, useful) == "abstain"
    # Without the paid/original factor, 0.6 would have passed the paid-order test.
    assert 0.6 <= ((1.0 - 0.2) / (1.0 + 0.2)) * 1.0


@pytest.mark.parametrize(
    "candidate,baseline,ell_paid,ell_original",
    [
        (0.0, 0.0, 80, 100),
        (-1.0, 1.0, 80, 100),
        (np.nan, 1.0, 80, 100),
        (1.0, np.inf, 80, 100),
        (1.0, 1.0, 0, 100),
        (1.0, 1.0, 80, -1),
        (1.0, 1.0, True, 100),
    ],
)
def test_net_safe_decision_abstains_on_boundaries(
    candidate, baseline, ell_paid, ell_original
):
    certificate = two_action_sample_variance_certificate(16, 0.05)
    assert (
        net_safe_decision(
            candidate, baseline, ell_paid, ell_original, certificate
        )
        == "abstain"
    )


def test_deterministic_grid_and_gate_do_not_search_empirical_outcomes():
    grid = build_certificate_grid()
    assert len(grid) == len(SAMPLE_SIZE_GRID) * len(JOINT_DELTA_GRID) * 2
    assert grid.loc[grid["selected_proved_method"], "nonvacuous"].sum() == 0
    # Deliberately favorable empirical context cannot produce a gate size.
    summary = pd.Series({"selected_mean_ratio": 0.001})
    bootstrap = pd.Series({"ci_low": 0.0, "ci_high": 0.01})
    catastrophic = pd.Series({"net_beneficial_accept_probability": 1.0})
    gate = evaluate_gate(grid, (summary, bootstrap, catastrophic))
    assert not bool(gate.loc[0, "gate_evaluable"])
    assert gate.loc[0, "verdict"] == VERDICT_BUDGET_VACUOUS
    verdict = build_verdict(grid, gate)
    assert verdict.loc[0, "verdict"] == VERDICT_BUDGET_VACUOUS


def test_input_validation_and_no_ambiguous_public_delta_argument():
    for invalid in (True, 1, 1.5, np.nan, np.inf):
        with pytest.raises(ValueError):
            two_action_sample_variance_certificate(invalid, 0.05)
    for invalid in (True, 0.0, 1.0, -0.1, np.nan, np.inf):
        with pytest.raises(ValueError):
            two_action_sample_variance_certificate(16, invalid)
    import inspect

    parameters = inspect.signature(two_action_sample_variance_certificate).parameters
    assert "joint_delta" in parameters
    assert "delta" not in parameters
    helper = inspect.signature(chebyshev_pointwise_relative_radius).parameters
    assert "pointwise_delta" in helper


def test_frozen_source_checksums_and_context_are_valid():
    checksums, phase1a, phase1b = validate_frozen_sources(
        SOURCE_PATHS, validate_historical=True
    )
    assert checksums
    assert phase1a["configuration_version"]
    assert phase1b["configuration_version"]
    summary, bootstrap, catastrophic = load_frozen_context(SOURCE_PATHS)
    assert summary["selected_mean_ratio"] == pytest.approx(0.036329, rel=2e-5)
    assert bootstrap["ci_high"] < 1.0
    assert catastrophic["net_beneficial_accept_probability"] > 0.75


def test_isolated_audit_artifacts_have_stable_vacuous_schema(tmp_path):
    result = run_audit(tmp_path, SOURCE_PATHS, validate_historical=True)
    assert len(result["grid"]) == 24
    assert len(result["gate"]) == 1
    assert result["bootstrap"].empty
    assert result["verdict"].loc[0, "verdict"] == VERDICT_BUDGET_VACUOUS
    assert result["manifest"].loc[0, "new_matvec_queries"] == 0
    files = {path.name for path in tmp_path.iterdir()}
    assert files == {
        "rademacher_sample_variance_confidence_phase1c_manifest.csv",
        "rademacher_sample_variance_confidence_phase1c_certificate_grid.csv",
        "rademacher_sample_variance_confidence_phase1c_gate.csv",
        "rademacher_sample_variance_confidence_phase1c_bootstrap.csv",
        "rademacher_sample_variance_confidence_phase1c_verdict.csv",
    }
