"""Pure helpers for risk-bridge regret and minimizer summaries."""

from __future__ import annotations

import numpy as np


PLATEAU_RTOL = 128.0 * np.finfo(np.float64).eps


def _as_curve(q_values, risk_values):
    q_values = np.asarray(q_values)
    risk_values = np.asarray(risk_values, dtype=np.float64)
    if q_values.ndim != 1 or risk_values.ndim != 1:
        raise ValueError("q_values and risk_values must be one-dimensional.")
    if q_values.size == 0 or q_values.size != risk_values.size:
        raise ValueError("q_values and risk_values must be nonempty and aligned.")
    if np.any(~np.isfinite(risk_values)) or np.any(risk_values < 0.0):
        raise ValueError("risk_values must be finite and nonnegative.")
    if np.any(np.diff(q_values.astype(np.float64)) <= 0.0):
        raise ValueError("q_values must be strictly increasing.")
    return q_values, risk_values


def minimum_plateau(q_values, risk_values):
    """Return deterministic minimum/plateau metadata for one ordered curve."""
    q_values, risk_values = _as_curve(q_values, risk_values)
    minimum = float(np.min(risk_values))
    if minimum == 0.0:
        mask = risk_values == 0.0
    else:
        mask = np.isclose(risk_values, minimum, rtol=PLATEAU_RTOL, atol=0.0)
    minimizing_q = q_values[mask]
    return {
        "minimum_risk": minimum,
        "representative_q": int(minimizing_q[0]),
        "minimum_q_smallest": int(minimizing_q[0]),
        "minimum_q_largest": int(minimizing_q[-1]),
        "minimum_q_count": int(minimizing_q.size),
        "minimum_is_zero": bool(minimum == 0.0),
        "minimum_is_plateau": bool(minimizing_q.size > 1),
    }


def regret_columns(q_values, risk_values):
    """Compute exact additive and documented normalized regret columns."""
    q_values, risk_values = _as_curve(q_values, risk_values)
    metadata = minimum_plateau(q_values, risk_values)
    minimum = metadata["minimum_risk"]
    additive = np.maximum(0.0, risk_values - minimum)
    baseline_positions = np.flatnonzero(q_values == 0)
    if baseline_positions.size != 1:
        raise ValueError("Every curve must contain exactly one q=0 baseline.")
    baseline = float(risk_values[baseline_positions[0]])
    if baseline > 0.0:
        normalized = additive / baseline
        zero_reference = False
    else:
        normalized = np.full(risk_values.shape, np.nan)
        zero_reference = True
    if minimum > 0.0:
        multiplicative = risk_values / minimum - 1.0
    else:
        multiplicative = np.full(risk_values.shape, np.nan)
    return {
        **metadata,
        "baseline_risk_q0": baseline,
        "zero_reference": zero_reference,
        "additive_regret": additive,
        "baseline_normalized_additive_regret": normalized,
        "multiplicative_regret": multiplicative,
    }


def paired_bootstrap_minimizer_frequencies(
    risk_matrix,
    q_values,
    bootstrap_samples,
    seed,
    chunk_size=1_000,
):
    """Bootstrap paired trials and return minimizer support and frequencies."""
    risk_matrix = np.asarray(risk_matrix, dtype=np.float64)
    q_values = np.asarray(q_values)
    if risk_matrix.ndim != 2 or risk_matrix.shape[1] != q_values.size:
        raise ValueError("risk_matrix must have shape (trials, len(q_values)).")
    if risk_matrix.shape[0] == 0 or np.any(~np.isfinite(risk_matrix)):
        raise ValueError("risk_matrix must be nonempty and finite.")
    if isinstance(bootstrap_samples, (bool, np.bool_)) or not isinstance(
        bootstrap_samples, (int, np.integer)
    ):
        raise ValueError("bootstrap_samples must be an integer.")
    if bootstrap_samples <= 0 or chunk_size <= 0:
        raise ValueError("bootstrap_samples and chunk_size must be positive.")

    trial_count = risk_matrix.shape[0]
    probabilities = np.full(trial_count, 1.0 / trial_count)
    counts = np.zeros(q_values.size, dtype=np.int64)
    minimizers = np.empty(bootstrap_samples, dtype=q_values.dtype)
    rng = np.random.default_rng(seed)
    start = 0
    while start < bootstrap_samples:
        stop = min(bootstrap_samples, start + chunk_size)
        weights = rng.multinomial(
            trial_count, probabilities, size=stop - start
        )
        curves = weights @ risk_matrix / trial_count
        indices = np.argmin(curves, axis=1)
        counts += np.bincount(indices, minlength=q_values.size)
        minimizers[start:stop] = q_values[indices]
        start = stop
    frequencies = counts.astype(np.float64) / bootstrap_samples
    if not np.isclose(float(np.sum(frequencies)), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("Bootstrap minimizer frequencies do not sum to one.")
    return {
        "q_values": q_values.copy(),
        "counts": counts,
        "frequencies": frequencies,
        "minimizers": minimizers,
        "ci_low": float(np.quantile(minimizers, 0.025)),
        "ci_high": float(np.quantile(minimizers, 0.975)),
    }
