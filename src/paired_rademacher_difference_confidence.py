"""Phase 2B confidence audits for a direct paired Rademacher risk difference.

The public contract concerns one candidate--baseline pair fixed before fresh
common Rademacher probes are drawn.  This module contains no matrix--vector
queries and no allocation logic.  Its implemented theorem routes deliberately
early-stop when their observable scale bounds are vacuous on the frozen
``s <= 32`` grid.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
from typing import Literal

import numpy as np


PROVED = "PROVED"
PROVED_BUT_BUDGET_VACUOUS = "PROVED BUT BUDGET-VACUOUS"
INCOMPLETE = "INCOMPLETE"
UNVERIFIED = "UNVERIFIED"

ROBUST_MOM_METHOD = "independent_pair_robust_mom"
ELEMENTARY_METHOD = "independent_pair_elementary"
COMPLETE_U_METHOD = "complete_u_statistic"
METHOD_PRIORITY = (ROBUST_MOM_METHOD, ELEMENTARY_METHOD, COMPLETE_U_METHOD)

FROZEN_SAMPLE_SIZES = (4, 8, 16, 32)
FROZEN_JOINT_DELTAS = (0.01, 0.05, 0.10)
POLYNOMIAL_DEGREE = 4
HYPERCONTRACTIVE_L4_L2_FACTOR = 9.0
FOURTH_MOMENT_FACTOR = 6561.0
PAIRWISE_SCALE_RELATIVE_VARIANCE_FACTOR = (FOURTH_MOMENT_FACTOR + 1.0) / 2.0


@dataclass(frozen=True)
class Phase2BRouteAudit:
    """Analytic status of one preregistered confidence route."""

    method: str
    sample_size: int
    independent_pair_count: int
    joint_delta: float
    proof_status: str
    observable_radius: float
    nonvacuous: bool
    theorem_source: str
    stopping_reason: str
    fourth_moment_factor: float
    scale_relative_variance_factor: float | None
    scale_relative_radius: float | None
    minimum_scale_block_size: int | None
    new_matvec_queries: int


@dataclass(frozen=True)
class PairedDifferenceCertificate:
    """One-sided certificate record for one fixed paired comparison."""

    method_requested: str
    method_selected: str
    sample_size: int
    independent_pair_count: int
    joint_delta: float
    point_estimate: float
    one_sided_radius: float
    upper_confidence_bound: float
    proof_status: str
    nonvacuous: bool
    zero_variance_observed_only: bool
    guaranteed_event: str
    stopping_reason: str


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, not a boolean.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


def _probability(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a probability, not a boolean.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a probability.") from exc
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one.")
    return value


def _positive_denominator(value, name: str) -> int:
    value = _positive_integer(value, name)
    return value


def hypercontractive_l4_l2_factor(degree: int = POLYNOMIAL_DEGREE) -> float:
    """Return ``3**(degree/2)`` for a degree-at-most ``degree`` Rademacher polynomial."""

    degree = _positive_integer(degree, "degree")
    return 3.0 ** (degree / 2.0)


def hypercontractive_fourth_moment_factor(
    degree: int = POLYNOMIAL_DEGREE,
) -> float:
    """Return the fourth-moment factor implied by the L4--L2 inequality."""

    return hypercontractive_l4_l2_factor(degree) ** 4


def independent_signed_pair_observations(
    candidate_values,
    baseline_values,
    ell_candidate: int,
    ell_baseline: int,
) -> np.ndarray:
    """Build conditionally iid signed-pair observations from disjoint probe pairs."""

    candidate = np.asarray(candidate_values, dtype=float)
    baseline = np.asarray(baseline_values, dtype=float)
    if candidate.ndim != 1 or baseline.ndim != 1 or candidate.shape != baseline.shape:
        raise ValueError("Candidate and baseline observations must be matching vectors.")
    if candidate.size < 2 or candidate.size % 2:
        raise ValueError("An even number of at least two common-probe observations is required.")
    if np.any(~np.isfinite(candidate)) or np.any(~np.isfinite(baseline)):
        raise ValueError("Observations must be finite.")
    ell_candidate = _positive_denominator(ell_candidate, "ell_candidate")
    ell_baseline = _positive_denominator(ell_baseline, "ell_baseline")
    candidate_differences = candidate[0::2] - candidate[1::2]
    baseline_differences = baseline[0::2] - baseline[1::2]
    return (
        candidate_differences**2 / (2.0 * ell_candidate)
        - baseline_differences**2 / (2.0 * ell_baseline)
    )


def exact_signed_pair_target(
    candidate_variance: float,
    baseline_variance: float,
    ell_candidate: int,
    ell_baseline: int,
) -> float:
    """Return ``E[D]`` from the two exact quadratic-form variances."""

    candidate_variance = float(candidate_variance)
    baseline_variance = float(baseline_variance)
    if (
        not math.isfinite(candidate_variance)
        or not math.isfinite(baseline_variance)
        or candidate_variance < 0.0
        or baseline_variance < 0.0
    ):
        raise ValueError("Variances must be finite and nonnegative.")
    ell_candidate = _positive_denominator(ell_candidate, "ell_candidate")
    ell_baseline = _positive_denominator(ell_baseline, "ell_baseline")
    return candidate_variance / ell_candidate - baseline_variance / ell_baseline


def sample_variance_relative_variance_bound(
    independent_pair_count: int,
    fourth_moment_factor: float = FOURTH_MOMENT_FACTOR,
) -> float:
    """Bound ``Var(S_D^2) / Var(D)^2`` from the exact sample-variance identity."""

    n = _positive_integer(independent_pair_count, "independent_pair_count")
    if n < 2:
        raise ValueError("independent_pair_count must be at least two.")
    factor = float(fourth_moment_factor)
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError("fourth_moment_factor must be finite and at least one.")
    return (factor - (n - 3.0) / (n - 1.0)) / n


def elementary_scale_relative_radius(
    independent_pair_count: int,
    pointwise_delta: float,
    fourth_moment_factor: float = FOURTH_MOMENT_FACTOR,
) -> float:
    """Chebyshev relative radius for the observable scale audit."""

    pointwise_delta = _probability(pointwise_delta, "pointwise_delta")
    relative_variance = sample_variance_relative_variance_bound(
        independent_pair_count, fourth_moment_factor
    )
    return math.sqrt(relative_variance / pointwise_delta)


def pairwise_scale_relative_variance_bound(
    fourth_moment_factor: float = FOURTH_MOMENT_FACTOR,
) -> float:
    """Bound ``Var((D-D')^2/2) / Var(D)^2`` for centered iid copies."""

    factor = float(fourth_moment_factor)
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError("fourth_moment_factor must be finite and at least one.")
    return (factor + 1.0) / 2.0


def minimum_scale_block_size(
    block_failure_probability: float = 0.25,
    relative_error: float = 1.0,
    fourth_moment_factor: float = FOURTH_MOMENT_FACTOR,
) -> int:
    """First integer block size making the Chebyshev scale bound strictly smaller.

    A strict inequality is used because a block must have failure probability
    below the requested threshold before a median-of-means majority argument
    can amplify it.
    """

    block_failure_probability = _probability(
        block_failure_probability, "block_failure_probability"
    )
    relative_error = float(relative_error)
    if not math.isfinite(relative_error) or not 0.0 < relative_error <= 1.0:
        raise ValueError("relative_error must lie in (0, 1].")
    factor = pairwise_scale_relative_variance_bound(fourth_moment_factor)
    threshold = factor / (block_failure_probability * relative_error**2)
    return math.floor(threshold) + 1


def _elementary_route_audit(sample_size: int, joint_delta: float) -> Phase2BRouteAudit:
    n = sample_size // 2
    # Fixed equal split: scale control and one-sided mean control.
    scale_delta = joint_delta / 2.0
    radius = elementary_scale_relative_radius(n, scale_delta)
    return Phase2BRouteAudit(
        method=ELEMENTARY_METHOD,
        sample_size=sample_size,
        independent_pair_count=n,
        joint_delta=joint_delta,
        proof_status=PROVED_BUT_BUDGET_VACUOUS,
        observable_radius=math.inf,
        nonvacuous=False,
        theorem_source="Exact sample-variance variance identity plus Chebyshev",
        stopping_reason=(
            "The observable variance upper bound requires a relative scale radius "
            f"below one, but the proved radius is {radius:.17g}."
        ),
        fourth_moment_factor=FOURTH_MOMENT_FACTOR,
        scale_relative_variance_factor=sample_variance_relative_variance_bound(n),
        scale_relative_radius=radius,
        minimum_scale_block_size=None,
        new_matvec_queries=0,
    )


def _robust_mom_route_audit(sample_size: int, joint_delta: float) -> Phase2BRouteAudit:
    n = sample_size // 2
    available_scale_pairs = n // 2
    required = minimum_scale_block_size()
    return Phase2BRouteAudit(
        method=ROBUST_MOM_METHOD,
        sample_size=sample_size,
        independent_pair_count=n,
        joint_delta=joint_delta,
        proof_status=PROVED_BUT_BUDGET_VACUOUS,
        observable_radius=math.inf,
        nonvacuous=False,
        theorem_source=(
            "Self-contained pairwise-scale Chebyshev blocks with median amplification"
        ),
        stopping_reason=(
            "Even granting every independent D observation to scale estimation, "
            f"only {available_scale_pairs} pairwise scale observations are available; "
            f"a single 25%-failure block with relative error at most one needs "
            f"at least {required}."
        ),
        fourth_moment_factor=FOURTH_MOMENT_FACTOR,
        scale_relative_variance_factor=PAIRWISE_SCALE_RELATIVE_VARIANCE_FACTOR,
        scale_relative_radius=None,
        minimum_scale_block_size=required,
        new_matvec_queries=0,
    )


def _complete_u_route_audit(sample_size: int, joint_delta: float) -> Phase2BRouteAudit:
    return Phase2BRouteAudit(
        method=COMPLETE_U_METHOD,
        sample_size=sample_size,
        independent_pair_count=sample_size // 2,
        joint_delta=joint_delta,
        proof_status=INCOMPLETE,
        observable_radius=math.inf,
        nonvacuous=False,
        theorem_source="Gine--Latala--Zinn canonical U-statistic route (scope audit only)",
        stopping_reason=(
            "Canonical concentration applies only to the degenerate Hoeffding term. "
            "The nondegenerate linear projection still lacks an observable small-sample "
            "scale bound, so the preregistered stopping rule forbids continuing with K2."
        ),
        fourth_moment_factor=FOURTH_MOMENT_FACTOR,
        scale_relative_variance_factor=None,
        scale_relative_radius=None,
        minimum_scale_block_size=None,
        new_matvec_queries=0,
    )


def phase2b_route_audit(
    sample_size: int,
    joint_delta: float,
    method: str,
) -> Phase2BRouteAudit:
    """Return the deterministic analytic audit for one frozen route/grid point."""

    sample_size = _positive_integer(sample_size, "sample_size")
    if sample_size < 4 or sample_size % 2:
        raise ValueError("sample_size must be an even integer of at least four.")
    joint_delta = _probability(joint_delta, "joint_delta")
    if method == ELEMENTARY_METHOD:
        return _elementary_route_audit(sample_size, joint_delta)
    if method == ROBUST_MOM_METHOD:
        return _robust_mom_route_audit(sample_size, joint_delta)
    if method == COMPLETE_U_METHOD:
        return _complete_u_route_audit(sample_size, joint_delta)
    raise ValueError("Unknown Phase 2B theorem method.")


def paired_risk_difference_certificate(
    paired_observations,
    *,
    joint_delta: float,
    method: str = ROBUST_MOM_METHOD,
) -> PairedDifferenceCertificate:
    """Construct the audited data-only record; current proved routes abstain.

    A zero empirical variance is *not* treated as proof of zero population
    variance.  The exact-zero theorem branch is valid only with structural
    knowledge unavailable under the Phase 2B data-only contract.
    """

    values = np.asarray(paired_observations, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
        raise ValueError("paired_observations must be a finite vector of size at least two.")
    audit = phase2b_route_audit(values.size * 2, joint_delta, method)
    point_estimate = float(np.mean(values))
    observed_zero = bool(np.all(values == values[0]))
    return PairedDifferenceCertificate(
        method_requested=method,
        method_selected=method,
        sample_size=values.size * 2,
        independent_pair_count=values.size,
        joint_delta=audit.joint_delta,
        point_estimate=point_estimate,
        one_sided_radius=audit.observable_radius,
        upper_confidence_bound=math.inf,
        proof_status=audit.proof_status,
        nonvacuous=False,
        zero_variance_observed_only=observed_zero,
        guaranteed_event=(
            "No finite observable one-sided event is returned on the frozen grid; "
            "the data-only decision must abstain."
        ),
        stopping_reason=audit.stopping_reason,
    )


def direct_paired_net_safe_decision(
    certificate: PairedDifferenceCertificate,
) -> Literal["accept", "abstain"]:
    """Accept only a proved, finite direct upper bound not exceeding zero."""

    if certificate.proof_status != PROVED or not certificate.nonvacuous:
        return "abstain"
    if not math.isfinite(certificate.upper_confidence_bound):
        return "abstain"
    return "accept" if certificate.upper_confidence_bound <= 0.0 else "abstain"


def phase2b_final_verdict() -> str:
    """Return the preregistered no-replay verdict for the frozen theorem grid."""

    audits = [
        phase2b_route_audit(s, delta, method)
        for method in METHOD_PRIORITY
        for s in FROZEN_SAMPLE_SIZES
        for delta in FROZEN_JOINT_DELTAS
    ]
    if any(audit.proof_status == PROVED and audit.nonvacuous for audit in audits):
        return PROVED
    # The two explicit iid routes are proved and uniformly budget-vacuous.
    # The full-U route is stopped, not treated as a refutation of every theorem.
    return PROVED_BUT_BUDGET_VACUOUS
