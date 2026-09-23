"""Finite-sample confidence utilities for Phase 1C.

This module certifies one fixed candidate--baseline pair.  It does not choose
actions, modify Hutch++, or claim that an abstaining candidate-first policy can
recover the original baseline after construction and certification costs have
been paid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np


PROVED = "PROVED"
PROVED_BUT_BUDGET_VACUOUS = "PROVED BUT BUDGET-VACUOUS"
INCOMPLETE = "INCOMPLETE"
REFUTED = "REFUTED"

CHEBYSHEV_METHOD = "chebyshev_81"
SHARPEST_PROVED_METHOD = "sharpest_proved_scale_free"
SHARPER_HOEFFDING_METHOD = "hoeffding_sharper_attempt"
METHOD_PRIORITY = (CHEBYSHEV_METHOD,)
PROVED_STATUSES = frozenset({PROVED, PROVED_BUT_BUDGET_VACUOUS})


@dataclass(frozen=True)
class TwoActionCertificate:
    """A simultaneous relative-error certificate for exactly two actions."""

    sample_size: int
    joint_delta: float
    action_count: int
    per_action_delta: float
    linear_component_delta: float | None
    degenerate_component_delta: float | None
    method_requested: str
    method_selected: str
    theorem_name: str
    proof_status: str
    constants: tuple[tuple[str, float], ...]
    epsilon: float
    nonvacuous: bool
    simultaneous_event: str


@dataclass(frozen=True)
class SharperTheoremAudit:
    """Status of the non-Chebyshev Hoeffding-decomposition route."""

    proof_status: str
    joint_delta: float
    per_action_delta: float
    linear_component_delta: float
    degenerate_component_delta: float
    linear_component_status: str
    degenerate_component_status: str
    unresolved_step: str


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean.")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if integer != value or integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _probability(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a probability, not a boolean.")
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a probability.") from exc
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one.")
    return probability


def sample_variance_exact_moments(
    population_variance: float,
    fourth_central_moment: float,
    sample_size: int,
) -> tuple[float, float]:
    """Return E[S^2] and Var(S^2) for iid data with a finite fourth moment."""

    sample_size = _positive_integer(sample_size, "sample_size")
    if sample_size < 2:
        raise ValueError("sample_size must be at least two.")
    variance = float(population_variance)
    fourth = float(fourth_central_moment)
    if not math.isfinite(variance) or variance < 0.0:
        raise ValueError("population_variance must be finite and nonnegative.")
    if not math.isfinite(fourth) or fourth < 0.0:
        raise ValueError("fourth_central_moment must be finite and nonnegative.")
    if fourth + 64.0 * np.finfo(float).eps * max(1.0, variance**2) < variance**2:
        raise ValueError("A fourth central moment cannot be smaller than variance squared.")
    variance_of_sample_variance = (
        fourth
        - ((sample_size - 3.0) / (sample_size - 1.0)) * variance**2
    ) / sample_size
    if variance_of_sample_variance < 0.0 and abs(variance_of_sample_variance) < 1e-14:
        variance_of_sample_variance = 0.0
    return variance, float(variance_of_sample_variance)


def hypercontractive_sample_variance_relative_variance_bound(sample_size: int) -> float:
    """Return the 81-hypercontractive upper bound on Var(S^2)/sigma^4."""

    sample_size = _positive_integer(sample_size, "sample_size")
    if sample_size < 2:
        raise ValueError("sample_size must be at least two.")
    return (80.0 + 2.0 / (sample_size - 1.0)) / sample_size


def chebyshev_pointwise_relative_radius(
    sample_size: int,
    pointwise_delta: float,
) -> float:
    """Pointwise relative radius; ``pointwise_delta`` is not a joint level."""

    pointwise_delta = _probability(pointwise_delta, "pointwise_delta")
    bound = hypercontractive_sample_variance_relative_variance_bound(sample_size)
    return math.sqrt(bound / pointwise_delta)


def chebyshev_joint_relative_radius(sample_size: int, joint_delta: float) -> float:
    """Two-action radius after the explicit equal union-bound split."""

    joint_delta = _probability(joint_delta, "joint_delta")
    return chebyshev_pointwise_relative_radius(sample_size, joint_delta / 2.0)


def chebyshev_radius_is_decreasing(sample_sizes: Iterable[int]) -> bool:
    """Check strict decrease over a supplied increasing integer grid."""

    sizes = tuple(_positive_integer(value, "sample_size") for value in sample_sizes)
    if any(value < 2 for value in sizes):
        raise ValueError("Every sample size must be at least two.")
    if any(right <= left for left, right in zip(sizes, sizes[1:])):
        raise ValueError("sample_sizes must be strictly increasing.")
    values = [hypercontractive_sample_variance_relative_variance_bound(s) for s in sizes]
    return all(right < left for left, right in zip(values, values[1:]))


def sharper_hoeffding_theorem_audit(joint_delta: float) -> SharperTheoremAudit:
    """Record why the sharper route is not yet an implementation-ready theorem."""

    joint_delta = _probability(joint_delta, "joint_delta")
    component_delta = joint_delta / 4.0
    return SharperTheoremAudit(
        proof_status=INCOMPLETE,
        joint_delta=joint_delta,
        per_action_delta=joint_delta / 2.0,
        linear_component_delta=component_delta,
        degenerate_component_delta=component_delta,
        linear_component_status=(
            "Moment growth is identified, but no verified explicit scale-free tail "
            "constant closes a radius below one on the frozen grid."
        ),
        degenerate_component_status=(
            "The canonical kernel is identified, but the audited Adamczak route "
            "contains theorem constants/norm reductions not closed numerically here."
        ),
        unresolved_step=(
            "A numerical simultaneous radius requires explicit bounds for both the "
            "nondegenerate linear projection and the canonical remainder."
        ),
    )


def _chebyshev_certificate(
    sample_size: int,
    joint_delta: float,
    method_requested: str,
) -> TwoActionCertificate:
    epsilon = chebyshev_joint_relative_radius(sample_size, joint_delta)
    proof_status = PROVED if epsilon < 1.0 else PROVED_BUT_BUDGET_VACUOUS
    return TwoActionCertificate(
        sample_size=sample_size,
        joint_delta=joint_delta,
        action_count=2,
        per_action_delta=joint_delta / 2.0,
        linear_component_delta=None,
        degenerate_component_delta=None,
        method_requested=method_requested,
        method_selected=CHEBYSHEV_METHOD,
        theorem_name="81-hypercontractive Chebyshev two-action certificate",
        proof_status=proof_status,
        constants=(("fourth_moment_factor", 81.0), ("action_count", 2.0)),
        epsilon=epsilon,
        nonvacuous=epsilon < 1.0,
        simultaneous_event=(
            f"With conditional probability at least {1.0 - joint_delta:.17g}, both "
            f"fixed actions satisfy |S_x^2-sigma_x^2| <= {epsilon:.17g}*sigma_x^2."
        ),
    )


def two_action_sample_variance_certificate(
    sample_size: int,
    joint_delta: float,
    method: str = SHARPEST_PROVED_METHOD,
) -> TwoActionCertificate:
    """Construct a simultaneous certificate for one fixed two-action comparison."""

    sample_size = _positive_integer(sample_size, "sample_size")
    if sample_size < 2:
        raise ValueError("sample_size must be at least two.")
    joint_delta = _probability(joint_delta, "joint_delta")
    if method not in {SHARPEST_PROVED_METHOD, CHEBYSHEV_METHOD, SHARPER_HOEFFDING_METHOD}:
        raise ValueError("Unknown certificate method.")
    if method == SHARPER_HOEFFDING_METHOD:
        audit = sharper_hoeffding_theorem_audit(joint_delta)
        raise RuntimeError(
            f"{SHARPER_HOEFFDING_METHOD} is {audit.proof_status}: "
            f"{audit.unresolved_step}"
        )
    # Chebyshev is currently the only method with a complete numerical theorem.
    return _chebyshev_certificate(sample_size, joint_delta, method)


def select_gate_sample_size(
    sample_sizes: Iterable[int],
    joint_delta: float,
) -> int | None:
    """Select the first analytically nonvacuous size, never using empirical data."""

    sizes = tuple(_positive_integer(value, "sample_size") for value in sample_sizes)
    if any(value < 2 for value in sizes):
        raise ValueError("Every sample size must be at least two.")
    if any(right <= left for left, right in zip(sizes, sizes[1:])):
        raise ValueError("sample_sizes must be strictly increasing.")
    for sample_size in sizes:
        certificate = two_action_sample_variance_certificate(sample_size, joint_delta)
        if certificate.nonvacuous:
            return sample_size
    return None


def hoeffding_kernel(x: float, y: float) -> float:
    return 0.5 * (float(x) - float(y)) ** 2


def hoeffding_linear_projection(x: float, mean: float, variance: float) -> float:
    variance = float(variance)
    if variance < 0.0 or not math.isfinite(variance):
        raise ValueError("variance must be finite and nonnegative.")
    return 0.5 * ((float(x) - float(mean)) ** 2 - variance)


def hoeffding_degenerate_kernel(x: float, y: float, mean: float) -> float:
    return -(float(x) - float(mean)) * (float(y) - float(mean))


def hoeffding_decomposition(values: Iterable[float], mean: float, variance: float):
    """Return the linear, canonical, and total centered U-statistic terms."""

    values = np.asarray(tuple(values), dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite one-dimensional sample of size at least two.")
    variance = float(variance)
    linear = (2.0 / values.size) * sum(
        hoeffding_linear_projection(value, mean, variance) for value in values
    )
    degenerate_sum = 0.0
    for left in range(values.size):
        for right in range(left + 1, values.size):
            degenerate_sum += hoeffding_degenerate_kernel(
                values[left], values[right], mean
            )
    degenerate = degenerate_sum / math.comb(values.size, 2)
    total = linear + degenerate
    sample_variance = float(np.var(values, ddof=1))
    return {
        "linear": float(linear),
        "degenerate": float(degenerate),
        "total": float(total),
        "sample_variance_minus_variance": sample_variance - variance,
    }


def assert_degenerate_theorem_scope(component: str) -> None:
    """Prevent a canonical-kernel theorem from being assigned to the full statistic."""

    if component != "h2":
        raise ValueError(
            "A completely degenerate U-statistic theorem applies only to h2, "
            "not to the full sample-variance kernel or h1."
        )


def net_safe_decision(
    candidate_variance_hat: float,
    baseline_variance_hat: float,
    ell_paid: int,
    ell_original: int,
    certificate: TwoActionCertificate,
) -> Literal["accept", "abstain"]:
    """Certify a paid candidate against the original, unstarted baseline."""

    try:
        candidate = float(candidate_variance_hat)
        baseline = float(baseline_variance_hat)
    except (TypeError, ValueError, OverflowError):
        return "abstain"
    if not math.isfinite(candidate) or not math.isfinite(baseline):
        return "abstain"
    if candidate < 0.0 or baseline <= 0.0:
        return "abstain"
    if isinstance(ell_paid, (bool, np.bool_)) or isinstance(ell_original, (bool, np.bool_)):
        return "abstain"
    try:
        paid = int(ell_paid)
        original = int(ell_original)
    except (TypeError, ValueError, OverflowError):
        return "abstain"
    if paid != ell_paid or original != ell_original or paid <= 0 or original <= 0:
        return "abstain"
    if certificate.proof_status not in PROVED_STATUSES:
        return "abstain"
    epsilon = float(certificate.epsilon)
    if not math.isfinite(epsilon) or epsilon < 0.0 or epsilon >= 1.0:
        return "abstain"
    threshold = ((1.0 - epsilon) / (1.0 + epsilon)) * (paid / original) * baseline
    return "accept" if candidate <= threshold else "abstain"
