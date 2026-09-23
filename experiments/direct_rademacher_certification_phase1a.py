"""Core utilities for the offline Phase 1A Rademacher-risk audit.

The functions in this module do not modify an estimator.  They reconstruct
pre-existing actions, evaluate fresh external certification probes, and expose
small pure functions for the preregistered statistical analysis.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as la


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trace_baseline import MatVecOracle, _rank_aware_qr  # noqa: E402

from run_rank_deficient_risk_bridge import (  # noqa: E402
    DEFAULT_QR_ATOL,
    DEFAULT_QR_RTOL,
    _energy_state,
    apply_structured_step,
)


ACTION_OFFSETS = {
    "km1": -1,
    "k": 0,
    "kp1": 1,
    "kp2": 2,
}
ACTION_PAIRS = {
    "left": ("km1", "k"),
    "primary": ("kp1", "k"),
    "right": ("kp2", "kp1"),
}
ESTIMATORS = (
    "sample_variance",
    "paired_mean",
    "mom_w1",
    "mom_w2",
)
VERDICT_ESTIMATORS = (
    "sample_variance",
    "mom_w1",
    "mom_w2",
)
SAMPLE_SIZES = (4, 8, 16, 32)
EPSILONS = (0.0, 0.2, 1.0 / 3.0, 0.5)
EPSILON_TAGS = {
    0.0: "0",
    0.2: "0p2",
    1.0 / 3.0: "1over3",
    0.5: "0p5",
}
TRUTH_BETTER = -1
TRUTH_TIE = 0
TRUTH_WORSE = 1
DECISION_REJECT = -1
DECISION_ABSTAIN = 0
DECISION_ACCEPT = 1
MASTER_CERT_SEED = 91_000
MASTER_BOOTSTRAP_SEED = 92_000
PROJECTOR_VALIDATION_ATOL = 1e-5
FROZEN_ENERGY_RTOL = 1e-7
FROZEN_ENERGY_ATOL = 1e-26


@dataclass(frozen=True)
class ActionState:
    """One reconstructed, already-paid-for Hutch++ action."""

    label: str
    q: int
    r_actual: int
    basis: np.ndarray
    image_basis: np.ndarray
    reconstruction_query_count: int
    exact_sigma2: float
    gaussian_energy: float
    rademacher_energy: float
    projector_error_op: float
    orthogonality_error: float


def epsilon_tag(epsilon):
    epsilon = float(epsilon)
    for value, tag in EPSILON_TAGS.items():
        if epsilon == value:
            return tag
    raise ValueError(f"Unsupported epsilon {epsilon!r}.")


def certification_seed(rank_index, basis_trial, batch, repetition):
    """Return the exact preregistered scalar certification seed."""
    components = (
        MASTER_CERT_SEED,
        int(rank_index),
        int(basis_trial),
        int(batch),
        int(repetition),
    )
    if any(value < 0 for value in components):
        raise ValueError("Certification seed components must be nonnegative.")
    sequence = np.random.SeedSequence(list(components))
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def certification_probes(
    dimension,
    rank_index,
    basis_trial,
    batch,
    repetition,
):
    """Generate the shared 32-column Rademacher certification matrix."""
    dimension = int(dimension)
    if dimension < 1:
        raise ValueError("dimension must be positive.")
    seed = certification_seed(rank_index, basis_trial, batch, repetition)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(dimension, 32), dtype=np.int8)
    probes_int8 = 2 * bits - 1
    digest = hashlib.sha256(
        np.ascontiguousarray(probes_int8).tobytes()
    ).hexdigest()
    return probes_int8.astype(np.float64), seed, digest


def sample_variance(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("sample variance requires at least two values.")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite.")
    return float(np.var(values, ddof=1))


def all_pairs_variance(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("all-pairs variance requires at least two values.")
    differences = values[:, None] - values[None, :]
    upper = np.triu_indices(values.size, k=1)
    return float(np.sum(differences[upper] ** 2) / (values.size * (values.size - 1)))


def paired_observations(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or values.size % 2:
        raise ValueError("paired observations require a positive even sample size.")
    return 0.5 * (values[0::2] - values[1::2]) ** 2


def certification_estimators(values):
    """Return all four preregistered estimates of the variance numerator."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size not in SAMPLE_SIZES:
        raise ValueError(f"sample size must be one of {SAMPLE_SIZES}.")
    paired = paired_observations(values)
    if paired.size % 2:
        raise RuntimeError("mom_w2 requires an even number of paired observations.")
    block_means = paired.reshape(-1, 2).mean(axis=1)
    estimates = {
        "sample_variance": sample_variance(values),
        "paired_mean": float(np.mean(paired)),
        "mom_w1": float(np.median(paired)),
        "mom_w2": float(np.median(block_means)),
    }
    if any((not np.isfinite(value) or value < 0.0) for value in estimates.values()):
        raise RuntimeError("Certification variance estimates must be finite and nonnegative.")
    return estimates


def truth_label(candidate_risk, baseline_risk):
    """Classify exact risk difference with the frozen roundoff tolerance."""
    candidate_risk = float(candidate_risk)
    baseline_risk = float(baseline_risk)
    if (
        not np.isfinite(candidate_risk)
        or not np.isfinite(baseline_risk)
        or candidate_risk < 0.0
        or baseline_risk < 0.0
    ):
        raise ValueError("Exact risks must be finite and nonnegative.")
    difference = candidate_risk - baseline_risk
    tolerance = (
        128.0
        * np.finfo(np.float64).eps
        * max(candidate_risk, baseline_risk)
    )
    if difference < -tolerance:
        return TRUTH_BETTER
    if difference > tolerance:
        return TRUTH_WORSE
    return TRUTH_TIE


def empirical_decision(candidate_risk_hat, baseline_risk_hat, epsilon):
    """Apply the preregistered accept/reject/abstain empirical guard."""
    candidate_risk_hat = float(candidate_risk_hat)
    baseline_risk_hat = float(baseline_risk_hat)
    epsilon = float(epsilon)
    if (
        not np.isfinite(candidate_risk_hat)
        or not np.isfinite(baseline_risk_hat)
        or candidate_risk_hat < 0.0
        or baseline_risk_hat < 0.0
    ):
        raise ValueError("Estimated risks must be finite and nonnegative.")
    if epsilon not in EPSILON_TAGS:
        raise ValueError("epsilon is outside the frozen grid.")
    if candidate_risk_hat == baseline_risk_hat:
        return DECISION_ABSTAIN
    rho = (1.0 - epsilon) / (1.0 + epsilon)
    if candidate_risk_hat <= rho * baseline_risk_hat:
        return DECISION_ACCEPT
    if baseline_risk_hat <= rho * candidate_risk_hat:
        return DECISION_REJECT
    return DECISION_ABSTAIN


def _frozen_incremental_reference_basis(
    signal_basis,
    eta,
    sketch_prefix,
    qr_rtol,
    qr_atol,
):
    """Replay the frozen incremental QR contract without cached AQ products.

    A one-shot batch QR is not a numerically invariant reference when eta is
    tiny: its tail directions can rotate substantially even when the frozen
    incremental construction is reproduced correctly.  This replay follows the
    historical one-column, twice-residualized acceptance rule and is compared
    through its projector rather than raw column signs.
    """
    dimension, q_max = sketch_prefix.shape
    accepted = np.empty((dimension, q_max), dtype=np.float64)
    accepted_rank = 0
    reference_scale_sq = 0.0
    for column_index in range(q_max):
        sample = apply_structured_step(
            sketch_prefix[:, column_index],
            signal_basis,
            eta,
        )
        reference_scale_sq += float(np.dot(sample, sample))
        if accepted_rank:
            current = accepted[:, :accepted_rank]
            residual = sample - current @ (current.T @ sample)
            residual = residual - current @ (current.T @ residual)
        else:
            residual = sample.copy()
        candidate, candidate_rank = _rank_aware_qr(
            residual[:, None],
            reference_scale=float(np.sqrt(reference_scale_sq)),
            rtol=qr_rtol,
            atol=qr_atol,
        )
        if candidate_rank == 1:
            accepted[:, accepted_rank] = candidate[:, 0]
            accepted_rank += 1
    return accepted[:, :accepted_rank].copy()


def _projector_distance_from_angles(left, right):
    if left.shape[1] != right.shape[1]:
        raise ValueError("Projector comparison requires equal ranks.")
    if left.shape[1] == 0:
        return 0.0
    angles = la.subspace_angles(left, right)
    return float(np.sin(np.max(angles)))


def reconstruct_actions(
    signal_basis,
    eta,
    sketch_matrix,
    step_rank,
    qr_rtol=DEFAULT_QR_RTOL,
    qr_atol=DEFAULT_QR_ATOL,
):
    """Reconstruct the four frozen adjacent actions and cache every AQ."""
    signal_basis = np.asarray(signal_basis, dtype=np.float64)
    sketch_matrix = np.asarray(sketch_matrix, dtype=np.float64)
    dimension = signal_basis.shape[0]
    q_by_label = {
        label: int(step_rank + offset)
        for label, offset in ACTION_OFFSETS.items()
    }
    if min(q_by_label.values()) < 1:
        raise ValueError("All Phase 1A action sizes must be positive.")
    q_max = max(q_by_label.values())
    if sketch_matrix.shape[0] != dimension or sketch_matrix.shape[1] < q_max:
        raise ValueError("Sketch matrix does not cover all requested actions.")

    oracle = MatVecOracle(
        lambda vector: apply_structured_step(vector, signal_basis, eta),
        d=dimension,
    )
    accepted_basis = np.empty((dimension, q_max), dtype=np.float64)
    accepted_images = np.empty((dimension, q_max), dtype=np.float64)
    accepted_rank = 0
    reference_scale_sq = 0.0
    projected_signal = signal_basis.copy()
    basis_row_norm_sq = np.zeros(dimension, dtype=np.float64)
    states = {}

    for column_index in range(q_max):
        sample = oracle(sketch_matrix[:, column_index])
        reference_scale_sq += float(np.dot(sample, sample))
        if accepted_rank:
            current = accepted_basis[:, :accepted_rank]
            residual = sample - current @ (current.T @ sample)
            residual = residual - current @ (current.T @ residual)
        else:
            residual = sample.copy()
        candidate, candidate_rank = _rank_aware_qr(
            residual[:, None],
            reference_scale=float(np.sqrt(reference_scale_sq)),
            rtol=qr_rtol,
            atol=qr_atol,
        )
        if candidate_rank == 1:
            direction = candidate[:, 0]
            accepted_basis[:, accepted_rank] = direction
            accepted_images[:, accepted_rank] = oracle(direction)
            accepted_rank += 1
            coefficients = direction @ signal_basis
            projected_signal -= np.outer(direction, coefficients)
            basis_row_norm_sq += direction**2

        q = column_index + 1
        matching = [label for label, action_q in q_by_label.items() if action_q == q]
        if not matching:
            continue
        label = matching[0]
        basis = accepted_basis[:, :accepted_rank].copy()
        image_basis = accepted_images[:, :accepted_rank].copy()
        energy = _energy_state(
            signal_basis,
            projected_signal,
            basis_row_norm_sq,
            float(eta),
            accepted_rank,
        )
        reference = _frozen_incremental_reference_basis(
            signal_basis,
            float(eta),
            sketch_matrix[:, :q],
            qr_rtol,
            qr_atol,
        )
        if reference.shape[1] != accepted_rank:
            raise RuntimeError("Independent frozen QR replay returned a different rank.")
        projector_error = _projector_distance_from_angles(basis, reference)
        orthogonality_error = (
            float(la.norm(basis.T @ basis - np.eye(accepted_rank), ord=2))
            if accepted_rank
            else 0.0
        )
        states[label] = ActionState(
            label=label,
            q=q,
            r_actual=accepted_rank,
            basis=basis,
            image_basis=image_basis,
            reconstruction_query_count=oracle.query_count,
            exact_sigma2=2.0 * float(energy["rademacher_energy"]),
            gaussian_energy=float(energy["gaussian_energy"]),
            rademacher_energy=float(energy["rademacher_energy"]),
            projector_error_op=projector_error,
            orthogonality_error=orthogonality_error,
        )

    if set(states) != set(ACTION_OFFSETS):
        raise RuntimeError("Phase 1A action reconstruction is incomplete.")
    for state in states.values():
        if state.reconstruction_query_count != state.q + state.r_actual:
            raise RuntimeError("Reconstruction query accounting failed.")
        if state.projector_error_op > PROJECTOR_VALIDATION_ATOL:
            raise RuntimeError(
                "Incremental reconstruction disagrees with the frozen replay "
                f"for {state.label}: projector error={state.projector_error_op:.3e}."
            )
        if state.orthogonality_error > 5e-10:
            raise RuntimeError("Reconstructed basis is not sufficiently orthonormal.")
        if not np.allclose(
            state.image_basis,
            apply_structured_step(state.basis, signal_basis, eta),
            rtol=2e-12,
            atol=2e-13,
        ):
            raise RuntimeError("Cached AQ does not match the structured operator.")
    return states


def certification_quadratic_forms(signal_basis, eta, actions, probes):
    """Evaluate all action quadratic forms using one shared queried AG."""
    probes = np.asarray(probes, dtype=np.float64)
    if probes.ndim != 2 or probes.shape[1] != 32:
        raise ValueError("Certification requires exactly 32 probe columns.")
    oracle = MatVecOracle(
        lambda vector: apply_structured_step(vector, signal_basis, eta),
        d=probes.shape[0],
    )
    image_probes = oracle(probes)
    if oracle.query_count != 32:
        raise RuntimeError("Certification oracle must count exactly 32 queries.")
    values = {}
    for label, state in actions.items():
        coefficients = state.basis.T @ probes
        residual_probes = probes - state.basis @ coefficients
        image_residuals = image_probes - state.image_basis @ coefficients
        quadratic = np.sum(residual_probes * image_residuals, axis=0)
        if not np.all(np.isfinite(quadratic)):
            raise RuntimeError("Certification quadratic forms must be finite.")
        values[label] = quadratic
    return values, oracle.query_count


def bootstrap_seed(
    estimator_index,
    sample_size_index,
    epsilon_index,
    eta_index,
    budget_index,
    pair_index,
    metric_index,
    batch_scope_index,
):
    components = [
        MASTER_BOOTSTRAP_SEED,
        int(estimator_index),
        int(sample_size_index),
        int(epsilon_index),
        int(eta_index),
        int(budget_index),
        int(pair_index),
        int(metric_index),
        int(batch_scope_index),
    ]
    if any(value < 0 for value in components):
        raise ValueError("Bootstrap seed components must be nonnegative.")
    return int(
        np.random.SeedSequence(components).generate_state(
            1, dtype=np.uint64
        )[0]
    )


def equal_rank_point(path_rates_by_rank):
    """Average eligible path rates within rank and then equally across ranks."""
    rank_values = []
    for rank in sorted(path_rates_by_rank):
        values = np.asarray(path_rates_by_rank[rank], dtype=np.float64)
        if values.size == 0:
            return np.nan
        rank_values.append(float(np.mean(values)))
    if not rank_values:
        return np.nan
    return float(np.mean(rank_values))


def conditional_cluster_bootstrap(
    path_rates_by_rank,
    samples,
    seed,
):
    """Bootstrap directly inside each metric's eligible path population."""
    samples = int(samples)
    if samples < 1:
        raise ValueError("samples must be positive.")
    prepared = {}
    for rank, values in path_rates_by_rank.items():
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            return {
                "evaluable": False,
                "point": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
            }
        if not np.all(np.isfinite(values)):
            raise ValueError("Eligible path rates must be finite.")
        prepared[int(rank)] = values
    if not prepared:
        return {
            "evaluable": False,
            "point": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    rng = np.random.default_rng(int(seed))
    rank_bootstraps = []
    for rank in sorted(prepared):
        values = prepared[rank]
        indices = rng.integers(0, values.size, size=(samples, values.size))
        rank_bootstraps.append(np.mean(values[indices], axis=1))
    draws = np.mean(np.stack(rank_bootstraps, axis=1), axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "evaluable": True,
        "point": equal_rank_point(prepared),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def exhaustive_rademacher_variance(matrix):
    """Exact variance of g^TBg by enumeration for tiny matrices."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-13):
        raise ValueError("Rademacher identity requires a symmetric matrix.")
    dimension = matrix.shape[0]
    if dimension > 16:
        raise ValueError("Exhaustive enumeration is restricted to tiny matrices.")
    integers = np.arange(2**dimension, dtype=np.uint64)[:, None]
    shifts = np.arange(dimension, dtype=np.uint64)[None, :]
    bits = ((integers >> shifts) & 1).astype(np.float64)
    probes = 2.0 * bits - 1.0
    values = np.einsum("bi,ij,bj->b", probes, matrix, probes)
    return float(np.var(values, ddof=0))
