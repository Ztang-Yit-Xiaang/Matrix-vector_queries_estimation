"""Phase 2C confidence certificates for direct paired Rademacher risk differences under structural priors.

This module implements Theorems 16, 17, and 18 from
``docs/proof_structural_paired_difference_confidence.md``.

It provides:
1. An observable, finite one-sided confidence certificate for Delta_R under a
   residual norm envelope prior (||H_0||_F <= M_0).
2. The refined chaos fourth-moment factor as a function of structural ratio
   kappa = ||C||_2 / ||C||_F.
3. The exact Chebyshev sample size feasibility boundary n*(K, delta_scale).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from fractions import Fraction
import numbers
from typing import Literal

import numpy as np


PROVED_UNDER_PRIOR = "PROVED UNDER EXPLICIT STRUCTURAL PRIOR"
PROVED_ANALYTIC_LIMIT = "PROVED ANALYTIC LIMIT"
BUDGET_VACUOUS = "BUDGET-VACUOUS"

ROUTE_NORM_ENVELOPE = "norm_envelope_cantelli"
ROUTE_STRUCTURAL_KURTOSIS = "structural_kurtosis_sample_variance"
ROUTE_FEASIBILITY_BOUNDARY = "chebyshev_feasibility_boundary"


@dataclass(frozen=True)
class StructuralCertificate:
    """Certificate conditional on the norm prior and nested action bases.

    The legacy `nonvacuous` field denotes a finite radius only, not useful power.
    `certified_safe` denotes whether the observed upper bound permits acceptance.
    """

    sample_size: int
    independent_pair_count: int
    delta: float
    norm_envelope_M0: float
    point_estimate: float
    one_sided_radius: float
    upper_confidence_bound: float
    certified_safe: bool
    proof_status: str
    nonvacuous: bool
    theorem_source: str


@dataclass(frozen=True)
class StructuralRouteAudit:
    """Audit record for structural prior feasibility analysis."""

    route_name: str
    sample_size: int
    independent_pair_count: int
    delta: float
    proof_status: str
    nonvacuous: bool
    radius: float
    kurtosis_factor: float
    minimum_sample_size: int | None
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
    return _positive_integer(value, name)


def _nonnegative_float(value, name: str) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real number.") from exc
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return val


def norm_envelope_variance_bound(
    M0: float,
    ell_candidate: int,
    ell_baseline: int,
) -> float:
    """Compute the unconditional upper bound on Var(D_j | G) under ||H_0||_F <= M_0.

    Theorem 16 proves:
        Var(D_j | G) <= 164 * M_0^4 * (1/ell_candidate + 1/ell_baseline)^2.
    """
    M0 = _nonnegative_float(M0, "M0")
    ell_a = _positive_denominator(ell_candidate, "ell_candidate")
    ell_0 = _positive_denominator(ell_baseline, "ell_baseline")

    if M0 == 0.0:
        return 0.0

    return 164.0 * (M0**4) * ((1.0 / ell_a + 1.0 / ell_0) ** 2)


def cantelli_structural_one_sided_radius(
    sample_size: int,
    delta: float,
    M0: float,
    ell_candidate: int,
    ell_baseline: int,
) -> float:
    """Compute the finite one-sided Cantelli confidence radius C_n^norm(delta; M0).

    Theorem 16 proves:
        C_n^norm = sqrt( (v_bar_D / n) * ((1 - delta) / delta) ),
    where n = floor(sample_size / 2).
    """
    s = _positive_integer(sample_size, "sample_size")
    n = s // 2
    if n < 1:
        raise ValueError("sample_size must be at least 2 to form an independent pair.")
    delta = _probability(delta, "delta")
    var_bound = norm_envelope_variance_bound(M0, ell_candidate, ell_baseline)

    if var_bound == 0.0:
        return 0.0

    variance_of_mean = var_bound / float(n)
    cantelli_factor = (1.0 - delta) / delta
    return math.sqrt(variance_of_mean * cantelli_factor)


def structural_chaos_fourth_moment_factor(kappa: float) -> float:
    """Compute the refined Rademacher chaos kurtosis factor for structural ratio kappa.

    Theorem 17 proves:
        E[Z^4] <= (3 + 12 * kappa^2) * sigma^4 <= 15 * sigma^4,
    where kappa = ||C||_2 / ||C||_F in (0, 1].
    """
    kappa = float(kappa)
    if not math.isfinite(kappa) or not 0.0 < kappa <= 1.0:
        raise ValueError("kappa must lie in (0, 1].")
    return 3.0 + 12.0 * (kappa**2)


def chebyshev_scale_feasibility_threshold(
    kurtosis_factor: float,
    delta_scale: float,
) -> int:
    """Smallest integer n >= 2 with [(K-1)+2/(n-1)]/n < delta_scale.

    Decimal input values are interpreted exactly for the strict inequality, so
    K=3, delta_scale=0.05 correctly returns 42 (n=41 is equality).
    This is a threshold for this bound, not a universal sample lower bound.
    """
    if isinstance(kurtosis_factor, (bool, np.bool_)):
        raise ValueError("kurtosis_factor must be at least 1.0, not boolean.")
    K = float(kurtosis_factor)
    if not math.isfinite(K) or K < 1.0:
        raise ValueError("kurtosis_factor must be at least 1.0.")
    delta_scale = _probability(delta_scale, "delta_scale")
    k_exact = Fraction(str(K))
    delta_exact = Fraction(str(delta_scale))

    def feasible(n):
        return (k_exact - 1) * (n - 1) + 2 < delta_exact * n * (n - 1)

    lower, upper = 1, 2
    while not feasible(upper):
        lower, upper = upper, 2 * upper
    while upper - lower > 1:
        middle = (upper + lower) // 2
        if feasible(middle):
            upper = middle
        else:
            lower = middle
    return upper


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


def structural_paired_difference_certificate(
    candidate_values,
    baseline_values,
    ell_candidate: int,
    ell_baseline: int,
    delta: float,
    norm_envelope_M0: float,
) -> StructuralCertificate:
    """Compute Theorem 16 under a valid norm envelope AND nested bases.

    These matrix assumptions cannot be verified from the supplied observations.
    The caller must establish them before treating the result as a certificate.
    """
    D = independent_signed_pair_observations(
        candidate_values, baseline_values, ell_candidate, ell_baseline
    )
    s = len(candidate_values)
    n = len(D)
    delta = _probability(delta, "delta")
    M0 = _nonnegative_float(norm_envelope_M0, "norm_envelope_M0")

    point_estimate = float(np.mean(D))
    radius = cantelli_structural_one_sided_radius(
        sample_size=s,
        delta=delta,
        M0=M0,
        ell_candidate=ell_candidate,
        ell_baseline=ell_baseline,
    )
    ucb = point_estimate + radius
    certified_safe = bool(ucb <= 0.0)

    return StructuralCertificate(
        sample_size=s,
        independent_pair_count=n,
        delta=delta,
        norm_envelope_M0=M0,
        point_estimate=point_estimate,
        one_sided_radius=radius,
        upper_confidence_bound=ucb,
        certified_safe=certified_safe,
        proof_status=PROVED_UNDER_PRIOR,
        nonvacuous=True,
        theorem_source="Theorem 16: Norm-Envelope Cantelli Certificate",
    )
