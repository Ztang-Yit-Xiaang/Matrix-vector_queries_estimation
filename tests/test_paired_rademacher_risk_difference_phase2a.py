from itertools import product

import numpy as np
import pandas as pd
import pytest

from experiments.postprocess_paired_rademacher_risk_difference_phase2a import (
    build_path_table,
    load_source,
    validate_phase1b_accounting,
)
from src.paired_rademacher_risk_difference import (
    construction_accounting,
    exact_paired_variance_components,
    hoeffding_kernel_components,
    paired_u_statistic,
    sample_variance_difference,
    shifted_decorrelated_baseline,
)


def test_nested_shared_prefix_uses_maximum_cost_only():
    accounting = construction_accounting(
        budget=160,
        candidate_q=16,
        candidate_r=16,
        baseline_q=15,
        baseline_r=15,
        sample_size=16,
        nested_shared_prefix=True,
    )
    assert accounting.candidate_cost == 32
    assert accounting.baseline_cost == 30
    assert accounting.committed_cost == 32
    assert accounting.original_ell == 130
    assert accounting.paid_ell == 112
    assert accounting.accounting_mode == "nested_shared_prefix_max"


def test_nonnested_architecture_requires_actual_query_ledger():
    with pytest.raises(ValueError, match="actual committed"):
        construction_accounting(
            budget=160,
            candidate_q=16,
            candidate_r=16,
            baseline_q=15,
            baseline_r=15,
            sample_size=16,
            nested_shared_prefix=False,
        )
    accounting = construction_accounting(
        budget=160,
        candidate_q=16,
        candidate_r=16,
        baseline_q=15,
        baseline_r=15,
        sample_size=16,
        nested_shared_prefix=False,
        actual_committed_count=62,
    )
    assert accounting.committed_cost == 62
    assert accounting.paid_ell == 82
    assert accounting.accounting_mode == "actual_query_ledger"


@pytest.mark.parametrize("bad_value", [True, 2.5, -1])
def test_construction_accounting_rejects_invalid_counts(bad_value):
    with pytest.raises(ValueError):
        construction_accounting(
            budget=160,
            candidate_q=bad_value,
            candidate_r=1,
            baseline_q=1,
            baseline_r=1,
            sample_size=4,
            nested_shared_prefix=True,
        )


def test_direct_u_statistic_equals_sample_variance_difference():
    candidate = np.array([1.0, 2.0, 5.0, 7.0, 8.0])
    baseline = np.array([0.0, 4.0, 3.0, 6.0, 9.0])
    direct = paired_u_statistic(candidate, baseline, 11, 13)
    difference = sample_variance_difference(candidate, baseline, 11, 13)
    assert direct == pytest.approx(difference, rel=1e-14, abs=1e-14)


def test_hoeffding_decomposition_holds_term_by_term():
    za = np.array([-2.0, -1.0, 1.0, 2.0])
    z0 = np.array([-3.0, 0.0, 1.0, 2.0])
    sigma_a = float(np.mean(za**2))
    sigma_0 = float(np.mean(z0**2))
    ell_a, ell_0 = 7, 9
    delta = sigma_a / ell_a - sigma_0 / ell_0
    k1_values = []
    conditional_k2 = []
    for i in range(len(za)):
        current = []
        for j in range(len(za)):
            k1, k1_prime, k2 = hoeffding_kernel_components(
                za[i], z0[i], za[j], z0[j], sigma_a, sigma_0, ell_a, ell_0
            )
            kernel = (
                (za[i] - za[j]) ** 2 / (2.0 * ell_a)
                - (z0[i] - z0[j]) ** 2 / (2.0 * ell_0)
            )
            assert kernel - delta == pytest.approx(k1 + k1_prime + k2)
            current.append(k2)
        k1_values.append(k1)
        conditional_k2.append(np.mean(current))
    assert np.var(k1_values) > 0.0
    assert np.allclose(conditional_k2, 0.0)


def test_exact_u_statistic_variance_identity_by_enumeration():
    za = np.array([-2.0, -1.0, 1.0, 2.0])
    z0 = np.array([-1.5, -0.5, 0.5, 1.5])
    sample_size = 3
    components = exact_paired_variance_components(
        za, z0, ell_candidate=7, ell_baseline=11, sample_size=sample_size
    )
    statistics = []
    for indices in product(range(len(za)), repeat=sample_size):
        indices = np.asarray(indices)
        statistics.append(
            sample_variance_difference(za[indices], z0[indices], 7, 11)
        )
    assert np.var(statistics, ddof=0) == pytest.approx(
        components.u_statistic_variance, rel=1e-13, abs=1e-15
    )


def test_positive_and_negative_common_probe_covariance_are_both_possible():
    positive = exact_paired_variance_components(
        np.array([-2.0, -1.0, 1.0, 2.0]),
        np.array([-3.0, -1.0, 1.0, 3.0]),
        7,
        9,
        4,
    )
    negative = exact_paired_variance_components(
        np.array([-2.0, -1.0, 1.0, 2.0]),
        np.array([-1.0, -2.0, 2.0, 1.0]),
        7,
        9,
        4,
    )
    assert positive.squared_chaos_covariance > 0.0
    assert negative.squared_chaos_covariance < 0.0


def test_shifted_comparator_is_cyclic_within_batch_not_independent():
    values = np.array([10.0, 11.0, 12.0, 20.0, 21.0, 22.0])
    batches = np.array([0, 0, 0, 1, 1, 1])
    repetitions = np.array([0, 1, 2, 0, 1, 2])
    shifted = shifted_decorrelated_baseline(values, batches, repetitions)
    assert np.array_equal(shifted, [11.0, 12.0, 10.0, 21.0, 22.0, 20.0])
    assert "not an independent" in shifted_decorrelated_baseline.__doc__


def test_phase2a_accounting_matches_frozen_phase1b_on_reduced_grid():
    source = load_source(max_paths_per_rank=1, repetitions_per_batch=2)
    paths = build_path_table(source)
    assert len(paths) == 216
    assert validate_phase1b_accounting(paths)
    assert paths["construction_accounting_mode"].eq(
        "nested_shared_prefix_max"
    ).all()
    assert paths["new_matvec_queries"].eq(0).all()
    assert not paths.duplicated(
        ["step_rank", "basis_trial", "eta", "budget", "s", "pair"]
    ).any()


def test_shifted_comparator_rejects_nonconsecutive_repetitions():
    with pytest.raises(ValueError, match="consecutive"):
        shifted_decorrelated_baseline(
            np.array([1.0, 2.0]), np.array([0, 0]), np.array([0, 2])
        )

