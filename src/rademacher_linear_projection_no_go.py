"""
Phase 1D: Rademacher Linear Projection Lower-Tail Analytic No-Go.

Isolates the nondegenerate linear Hoeffding component L_s = (1/s) sum_i (Z_i^2 - sigma^2)/sigma^2
under Cortinovis-Kressner Theorem 2 large deviations and the frozen truncation-Bernstein family.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np

# Frozen specification constants
FOURTH_MOMENT_BOUND: Final[float] = 81.0
VARIANCE_FLOOR_LOWER: Final[float] = 80.0
MINIMUM_FROZEN_CAP: Final[float] = 64.0
ANALYTIC_EXPONENT_DIVISOR: Final[float] = 160.0

FROZEN_SAMPLE_SIZES: Final[tuple[int, ...]] = (4, 8, 16, 32)
FROZEN_JOINT_DELTAS: Final[tuple[float, ...]] = (0.01, 0.05, 0.10)
FROZEN_ALLOCATIONS: Final[tuple[str, ...]] = ("final_compatible", "optimistic")
FROZEN_CAP_GRID: Final[tuple[float, ...]] = (
    64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0
)
FROZEN_KAPPAS: Final[tuple[float, ...]] = (1.0, 0.5, 0.25, 0.1, float(1.0 / np.sqrt(500)))
FROZEN_EPSILONS: Final[tuple[float, ...]] = (0.1, 0.25, 0.5, 0.75, 0.99)


@dataclass(frozen=True)
class LinearLowerTailNoGoAudit:
    sample_size: int
    joint_delta: float
    allocation: str
    directed_component_delta: float
    cap_minimum: float
    variance_envelope_floor: float
    analytic_probability_floor: float
    maximum_declared_component_delta: float
    theorem_status: str
    verdict: str


def cortinovis_kressner_tail_bound(v: float, kappa: float) -> float:
    """
    Cortinovis-Kressner Theorem 2 tail bound for V = Z^2 / sigma^2.
    Pr(V >= v) <= min(1.0, 2.0 * exp(-v / (4 + 4*sqrt(2*v)*kappa))).
    """
    if not math.isfinite(v):
        raise ValueError(f"v must be finite, got {v}")
    if v <= 0.0:
        return 1.0
    if not math.isfinite(kappa) or not (0.0 < kappa <= 1.0):
        raise ValueError(f"kappa must be in (0, 1], got {kappa}")

    denom = 4.0 + 4.0 * math.sqrt(2.0 * v) * kappa
    exponent = -v / denom
    raw = 2.0 * math.exp(exponent)
    return min(1.0, raw)


def truncation_bias_upper_bound(T: float, kappa: float) -> float:
    """
    Analytical upper bound on truncation bias b_T = E[(V - T)_+].
    beta_kappa(T) = 4 * exp(-a * sqrt(T)) * (sqrt(T)/a + 1/a^2),
    where a = 1 / (4*sqrt(2)*kappa + 4/sqrt(T)).
    """
    if not math.isfinite(T) or T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    if not math.isfinite(kappa) or not (0.0 < kappa <= 1.0):
        raise ValueError(f"kappa must be in (0, 1], got {kappa}")

    sqrt_T = math.sqrt(T)
    a = 1.0 / (4.0 * math.sqrt(2.0) * kappa + 4.0 / sqrt_T)
    factor = sqrt_T / a + 1.0 / (a * a)
    return 4.0 * math.exp(-a * sqrt_T) * factor


def capped_variance_envelope(T: float, kappa: float) -> float:
    """
    Variance envelope nu_kappa(T) for V^(T) = min(V, T).
    nu_kappa(T) = min(81, T^2 / 4, 81 - max(0, 1 - beta_kappa(T))^2).
    """
    if not math.isfinite(T) or T < MINIMUM_FROZEN_CAP:
        raise ValueError(f"T must be >= {MINIMUM_FROZEN_CAP}, got {T}")

    beta = truncation_bias_upper_bound(T, kappa)
    term1 = FOURTH_MOMENT_BOUND
    term2 = (T * T) / 4.0
    term3 = FOURTH_MOMENT_BOUND - max(0.0, 1.0 - beta) ** 2
    return min(term1, term2, term3)


def one_sided_bernstein_lower_tail(s: int, eps: float, T: float, kappa: float) -> float:
    """
    One-sided Bernstein lower-tail bound D_{s, kappa}(eps, T) for Pr(L_s <= -eps).
    Returns 1.0 if beta_kappa(T) >= eps (inadmissible cap).
    """
    if isinstance(s, bool) or not isinstance(s, numbers.Integral) or s < 2:
        raise ValueError(f"sample_size must be >= 2, got {s}")
    if not math.isfinite(eps) or not (0.0 < eps < 1.0):
        raise ValueError(f"eps must be in (0, 1), got {eps}")

    beta = truncation_bias_upper_bound(T, kappa)
    if beta >= eps:
        return 1.0

    x = eps - beta
    nu = capped_variance_envelope(T, kappa)
    denom = 2.0 * (nu + x / 3.0)
    exponent = -(s * (x * x)) / denom
    return math.exp(exponent)


def analytic_probability_floor(s: int) -> float:
    """
    Analytical lower bound on D_{s, kappa}(eps, T) > exp(-s / 160).
    """
    if isinstance(s, bool) or not isinstance(s, numbers.Integral) or s <= 0:
        raise ValueError(f"s must be positive, got {s}")
    return math.exp(-float(s) / ANALYTIC_EXPONENT_DIVISOR)


def directed_component_delta(joint_delta: float, allocation: str) -> float:
    """
    Computes directed linear component failure probability delta_linear.
    - 'final_compatible': joint_delta / 4
    - 'optimistic': joint_delta / 2
    """
    if not math.isfinite(joint_delta) or not (0.0 < joint_delta < 1.0):
        raise ValueError(f"joint_delta must be in (0, 1), got {joint_delta}")
    if allocation == "final_compatible":
        return joint_delta / 4.0
    elif allocation == "optimistic":
        return joint_delta / 2.0
    else:
        raise ValueError(f"Unknown allocation '{allocation}'. Must be 'final_compatible' or 'optimistic'.")


def linear_lower_tail_no_go_audit(
    sample_size: int,
    joint_delta: float,
    allocation: str = "final_compatible",
) -> LinearLowerTailNoGoAudit:
    """
    Executes the deterministic Phase 1D Linear Lower-Tail Analytic No-Go Audit.
    """
    if sample_size not in FROZEN_SAMPLE_SIZES:
        raise ValueError(f"sample_size {sample_size} not in frozen set {FROZEN_SAMPLE_SIZES}")
    if joint_delta not in FROZEN_JOINT_DELTAS:
        raise ValueError(f"joint_delta {joint_delta} not in frozen set {FROZEN_JOINT_DELTAS}")
    if allocation not in FROZEN_ALLOCATIONS:
        raise ValueError(f"allocation {allocation} not in frozen set {FROZEN_ALLOCATIONS}")

    directed_delta = directed_component_delta(joint_delta, allocation)
    prob_floor = analytic_probability_floor(sample_size)

    # This numerical field is a regression summary only.  It searches the
    # complete frozen grid but cannot affect the analytic verdict.
    min_cap_bound = 1.0
    for T in FROZEN_CAP_GRID:
        for kappa in FROZEN_KAPPAS:
            for eps in FROZEN_EPSILONS:
                val = one_sided_bernstein_lower_tail(sample_size, eps, T, kappa)
                if val < min_cap_bound:
                    min_cap_bound = val

    # Maximum declared component delta across optimistic allocation at largest joint_delta (0.10 / 2 = 0.05)
    max_declared = 0.10 / 2.0

    # Since prob_floor >= exp(-32/160) = exp(-0.2) approx 0.8187 > 0.05, verdict is strictly STRONG LINEAR NO-GO
    return LinearLowerTailNoGoAudit(
        sample_size=sample_size,
        joint_delta=joint_delta,
        allocation=allocation,
        directed_component_delta=directed_delta,
        cap_minimum=min_cap_bound,
        variance_envelope_floor=VARIANCE_FLOOR_LOWER,
        analytic_probability_floor=prob_floor,
        maximum_declared_component_delta=max_declared,
        theorem_status="PROVED",
        verdict="STRONG LINEAR NO-GO",
    )
