"""Exact identities and accounting for paired Rademacher risk differences."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers

import numpy as np


@dataclass(frozen=True)
class ConstructionAccounting:
    """Committed construction and residual capacities for one fixed pair."""

    candidate_cost: int
    baseline_cost: int
    committed_cost: int
    original_ell: int
    paid_ell: int
    accounting_mode: str


@dataclass(frozen=True)
class PairedVarianceComponents:
    """Exact finite-population moments under equally weighted probe outcomes."""

    linear_kernel_variance: float
    canonical_kernel_second_moment: float
    u_statistic_variance: float
    squared_chaos_covariance: float
    chaos_covariance: float


def _query_count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, not a boolean.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def construction_accounting(
    *,
    budget,
    candidate_q,
    candidate_r,
    baseline_q,
    baseline_r,
    sample_size,
    nested_shared_prefix,
    actual_committed_count=None,
):
    """Return paid/original capacities under an explicit construction contract.

    ``max(candidate_cost, baseline_cost)`` is used only for nested shared
    prefixes.  Every other architecture must provide the actual committed
    construction-query count from its query ledger.
    """

    budget = _query_count(budget, "budget")
    candidate_q = _query_count(candidate_q, "candidate_q")
    candidate_r = _query_count(candidate_r, "candidate_r")
    baseline_q = _query_count(baseline_q, "baseline_q")
    baseline_r = _query_count(baseline_r, "baseline_r")
    sample_size = _query_count(sample_size, "sample_size")
    if sample_size == 0:
        raise ValueError("sample_size must be positive.")
    if candidate_r > candidate_q or baseline_r > baseline_q:
        raise ValueError("Accepted rank cannot exceed attempted sketch queries.")
    if not isinstance(nested_shared_prefix, (bool, np.bool_)):
        raise ValueError("nested_shared_prefix must be Boolean.")

    candidate_cost = candidate_q + candidate_r
    baseline_cost = baseline_q + baseline_r
    if nested_shared_prefix:
        if actual_committed_count is not None:
            raise ValueError(
                "actual_committed_count must be omitted for nested shared prefixes."
            )
        committed_cost = max(candidate_cost, baseline_cost)
        mode = "nested_shared_prefix_max"
    else:
        if actual_committed_count is None:
            raise ValueError(
                "Nonnested accounting requires the actual committed query count."
            )
        committed_cost = _query_count(actual_committed_count, "actual_committed_count")
        if committed_cost < max(candidate_cost, baseline_cost):
            raise ValueError(
                "The committed count cannot be smaller than either available action cost."
            )
        mode = "actual_query_ledger"

    original_ell = budget - baseline_cost
    paid_ell = budget - committed_cost - sample_size
    return ConstructionAccounting(
        candidate_cost=candidate_cost,
        baseline_cost=baseline_cost,
        committed_cost=committed_cost,
        original_ell=original_ell,
        paid_ell=paid_ell,
        accounting_mode=mode,
    )


def _positive_denominator(value, name):
    value = _query_count(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def paired_u_statistic(candidate_values, baseline_values, ell_candidate, ell_baseline):
    """Compute the direct order-two U-statistic for a fixed action pair."""

    candidate = np.asarray(candidate_values, dtype=float)
    baseline = np.asarray(baseline_values, dtype=float)
    if candidate.ndim != 1 or baseline.ndim != 1 or candidate.shape != baseline.shape:
        raise ValueError("Candidate and baseline observations must be matching vectors.")
    if candidate.size < 2:
        raise ValueError("At least two paired observations are required.")
    if np.any(~np.isfinite(candidate)) or np.any(~np.isfinite(baseline)):
        raise ValueError("Observations must be finite.")
    ell_candidate = _positive_denominator(ell_candidate, "ell_candidate")
    ell_baseline = _positive_denominator(ell_baseline, "ell_baseline")
    total = 0.0
    pair_count = 0
    for i in range(candidate.size - 1):
        candidate_diff = candidate[i] - candidate[i + 1 :]
        baseline_diff = baseline[i] - baseline[i + 1 :]
        total += float(
            np.sum(candidate_diff**2 / (2.0 * ell_candidate))
            - np.sum(baseline_diff**2 / (2.0 * ell_baseline))
        )
        pair_count += candidate.size - i - 1
    return total / pair_count


def sample_variance_difference(
    candidate_values,
    baseline_values,
    ell_candidate,
    ell_baseline,
):
    """Difference of common-probe unbiased sample variances."""

    candidate = np.asarray(candidate_values, dtype=float)
    baseline = np.asarray(baseline_values, dtype=float)
    if candidate.ndim != 1 or baseline.ndim != 1 or candidate.shape != baseline.shape:
        raise ValueError("Candidate and baseline observations must be matching vectors.")
    if candidate.size < 2:
        raise ValueError("At least two paired observations are required.")
    ell_candidate = _positive_denominator(ell_candidate, "ell_candidate")
    ell_baseline = _positive_denominator(ell_baseline, "ell_baseline")
    return float(
        np.var(candidate, ddof=1) / ell_candidate
        - np.var(baseline, ddof=1) / ell_baseline
    )


def hoeffding_kernel_components(
    za,
    z0,
    za_prime,
    z0_prime,
    sigma2_candidate,
    sigma2_baseline,
    ell_candidate,
    ell_baseline,
):
    """Return K1(Y), K1(Y'), and canonical K2(Y,Y')."""

    values = (za, z0, za_prime, z0_prime, sigma2_candidate, sigma2_baseline)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("Kernel inputs must be finite.")
    if sigma2_candidate < 0.0 or sigma2_baseline < 0.0:
        raise ValueError("Variances must be nonnegative.")
    ell_candidate = _positive_denominator(ell_candidate, "ell_candidate")
    ell_baseline = _positive_denominator(ell_baseline, "ell_baseline")
    alpha = 1.0 / ell_candidate
    beta = 1.0 / ell_baseline
    k1 = 0.5 * (
        alpha * (za**2 - sigma2_candidate)
        - beta * (z0**2 - sigma2_baseline)
    )
    k1_prime = 0.5 * (
        alpha * (za_prime**2 - sigma2_candidate)
        - beta * (z0_prime**2 - sigma2_baseline)
    )
    k2 = -alpha * za * za_prime + beta * z0 * z0_prime
    return float(k1), float(k1_prime), float(k2)


def exact_paired_variance_components(
    centered_candidate,
    centered_baseline,
    ell_candidate,
    ell_baseline,
    sample_size,
):
    """Evaluate the exact component identities on equally weighted outcomes."""

    za = np.asarray(centered_candidate, dtype=float)
    z0 = np.asarray(centered_baseline, dtype=float)
    if za.ndim != 1 or z0.ndim != 1 or za.shape != z0.shape or za.size == 0:
        raise ValueError("Centered outcomes must be nonempty matching vectors.")
    if np.any(~np.isfinite(za)) or np.any(~np.isfinite(z0)):
        raise ValueError("Centered outcomes must be finite.")
    if not np.isclose(np.mean(za), 0.0) or not np.isclose(np.mean(z0), 0.0):
        raise ValueError("Inputs must be centered under the empirical distribution.")
    ell_candidate = _positive_denominator(ell_candidate, "ell_candidate")
    ell_baseline = _positive_denominator(ell_baseline, "ell_baseline")
    sample_size = _query_count(sample_size, "sample_size")
    if sample_size < 2:
        raise ValueError("sample_size must be at least two.")

    alpha = 1.0 / ell_candidate
    beta = 1.0 / ell_baseline
    sigma2_candidate = float(np.mean(za**2))
    sigma2_baseline = float(np.mean(z0**2))
    squared_covariance = float(
        np.mean((za**2 - sigma2_candidate) * (z0**2 - sigma2_baseline))
    )
    chaos_covariance = float(np.mean(za * z0))
    linear_variable = alpha * za**2 - beta * z0**2
    linear_kernel_variance = 0.25 * float(np.var(linear_variable, ddof=0))
    canonical_second_moment = float(
        alpha**2 * sigma2_candidate**2
        + beta**2 * sigma2_baseline**2
        - 2.0 * alpha * beta * chaos_covariance**2
    )
    u_variance = (
        4.0 * linear_kernel_variance / sample_size
        + 2.0 * canonical_second_moment / (sample_size * (sample_size - 1))
    )
    return PairedVarianceComponents(
        linear_kernel_variance=linear_kernel_variance,
        canonical_kernel_second_moment=canonical_second_moment,
        u_statistic_variance=float(u_variance),
        squared_chaos_covariance=squared_covariance,
        chaos_covariance=chaos_covariance,
    )


def shifted_decorrelated_baseline(values, batches, repetitions):
    """Apply the declared within-batch cyclic shift to baseline estimates.

    This is a deterministic decorrelation diagnostic, not an independent
    sample sequence.  Every batch must contain repetition labels 0,...,n-1.
    """

    values = np.asarray(values, dtype=float)
    batches = np.asarray(batches)
    repetitions = np.asarray(repetitions)
    if values.ndim != 1 or batches.shape != values.shape or repetitions.shape != values.shape:
        raise ValueError("Values, batches, and repetitions must be matching vectors.")
    shifted = np.empty_like(values)
    for batch in np.unique(batches):
        indices = np.flatnonzero(batches == batch)
        order = indices[np.argsort(repetitions[indices])]
        expected = np.arange(order.size)
        if not np.array_equal(repetitions[order], expected):
            raise ValueError("Each batch must have consecutive zero-based repetitions.")
        shifted[order] = np.roll(values[order], -1)
    return shifted
