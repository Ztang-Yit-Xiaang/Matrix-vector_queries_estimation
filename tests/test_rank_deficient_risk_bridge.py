import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.linalg as la


EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from postprocess_exact_risk_bridge import postprocess_exact_bridge  # noqa: E402
from risk_bridge_regret import (  # noqa: E402
    paired_bootstrap_minimizer_frequencies,
    regret_columns,
)
from run_rank_deficient_risk_bridge import (  # noqa: E402
    RISK_COLUMNS,
    _energy_state,
    _expand_path_for_budget,
    build_spectrum_manifest,
    compute_nested_rank_path,
    make_signal_basis,
    q_max_for_budget,
    run_rank_deficient_bridge,
    spectral_tail_energy,
    validate_configuration,
)


def test_default_manifest_has_12_spectra_and_paired_orientations():
    config = validate_configuration()
    manifest = build_spectrum_manifest(config)
    assert len(manifest) == 36
    assert manifest["spectrum_index"].nunique() == 12
    assert set(config["q_maxima"].values()) == {36, 76, 116}
    for _, selected in manifest.groupby("step_rank"):
        assert selected["orientation_seed"].nunique() == 1
        assert selected["orientation_group_id"].nunique() == 1
        assert selected["eta"].nunique() == 4


@pytest.mark.parametrize(
    "overrides",
    [
        {"budgets": (True, 160)},
        {"budgets": (80, 80)},
        {"primary_budget": 81},
        {"trials": False},
        {"dimension": 4, "step_ranks": (5,)},
        {"tail_levels": (0.0, np.nan)},
        {"tail_levels": (0.0, 1.0)},
        {"qr_rtol": np.inf},
        {"qr_atol": -1.0},
        {"min_residual_probes": 0},
    ],
)
def test_invalid_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        validate_configuration(**overrides)


def test_q_max_is_derived_and_requires_residual_capacity():
    assert q_max_for_budget(500, 80, 8) == 36
    assert q_max_for_budget(500, 160, 8) == 76
    assert q_max_for_budget(500, 240, 8) == 116
    with pytest.raises(ValueError):
        q_max_for_budget(10, 3, 8)


def test_structure_aware_energies_match_explicit_dense_residual():
    rng = np.random.default_rng(84)
    dimension = 9
    step_rank = 3
    signal_basis, _ = la.qr(rng.normal(size=(dimension, step_rank)), mode="economic")
    full_basis, _ = la.qr(rng.normal(size=(dimension, 2)), mode="economic")
    projected_signal = signal_basis - full_basis @ (full_basis.T @ signal_basis)
    row_norm_sq = np.sum(full_basis**2, axis=1)
    eta = 1e-14
    energy = _energy_state(
        signal_basis, projected_signal, row_norm_sq, eta, rank=2
    )
    matrix = eta * np.eye(dimension) + (1.0 - eta) * signal_basis @ signal_basis.T
    residual_projector = np.eye(dimension) - full_basis @ full_basis.T
    residual = residual_projector @ matrix @ residual_projector
    gaussian = float(np.sum(residual**2))
    rademacher = gaussian - float(np.sum(np.diag(residual) ** 2))
    assert np.isclose(energy["raw_gaussian_energy"], gaussian, rtol=2e-13, atol=1e-28)
    assert np.isclose(energy["rademacher_energy"], rademacher, rtol=2e-13, atol=1e-28)


def test_exact_rank_path_accepts_then_rejects_and_counts_queries():
    signal_basis = np.eye(6)[:, :2]
    sketch = np.array(
        [
            [1.0, 1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0],
        ]
    )
    path = compute_nested_rank_path(signal_basis, 0.0, sketch)
    assert path["r_actual"].tolist() == [0, 1, 2, 2, 2]
    assert path["rejected_query_count"].tolist() == [0, 0, 0, 1, 2]
    assert np.all(path["constructed_basis_queries"] == path["q"] + path["r_actual"])
    assert np.isnan(path.loc[0, "rank_efficiency"])
    assert np.allclose(path.loc[1:, "rank_efficiency"], [1.0, 1.0, 2 / 3, 0.5])
    assert path.loc[2:, "exact_zero_canonicalized"].all()
    assert np.all(path.loc[2:, ["gaussian_energy", "rademacher_energy"]] == 0.0)


def test_tiny_positive_tail_is_not_canonicalized_to_zero():
    signal_basis = np.eye(6)[:, :2]
    sketch = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    path = compute_nested_rank_path(signal_basis, 1e-14, sketch)
    assert not path["exact_zero_canonicalized"].any()
    assert np.all(path["gaussian_energy"] > 0.0)


def test_rank_aware_and_realized_transition_identities_hold():
    signal_basis = np.eye(6)[:, :2]
    sketch = np.array(
        [
            [1.0, 1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0],
        ]
    )
    path = compute_nested_rank_path(signal_basis, 1e-6, sketch, qr_rtol=1e-4)
    expanded = _expand_path_for_budget(
        path, dimension=6, step_rank=2, eta=1e-6, budget=12, q_max=4
    )
    for column in (
        "rank_marginal_identity_residual",
        "gaussian_marginal_identity_residual",
        "rademacher_marginal_identity_residual",
    ):
        assert np.allclose(expanded[column].dropna(), 0.0, rtol=0.0, atol=1e-14)
    failed = expanded.loc[~expanded["rank_gain_accepted"] & (expanded["q"] > 0)]
    assert np.all(failed["failed_penalty_rank_transition"] >= 0.0)


def test_spectral_tail_energy_covers_rank_boundary():
    assert spectral_tail_energy(10, 3, 0.1, 0) == pytest.approx(3.07)
    assert spectral_tail_energy(10, 3, 0.1, 3) == pytest.approx(0.07)
    assert spectral_tail_energy(10, 3, 0.1, 10) == 0.0


def test_regret_zero_minimum_uses_nan_without_epsilon():
    summary = regret_columns(np.array([0, 1, 2]), np.array([2.0, 0.0, 0.0]))
    assert np.allclose(summary["additive_regret"], [2.0, 0.0, 0.0])
    assert np.allclose(
        summary["baseline_normalized_additive_regret"], [1.0, 0.0, 0.0]
    )
    assert np.isnan(summary["multiplicative_regret"]).all()
    assert summary["minimum_q_smallest"] == 1
    assert summary["minimum_q_largest"] == 2
    assert summary["minimum_is_plateau"]


def test_bootstrap_complete_support_sums_to_one():
    risks = np.array([[3.0, 1.0, 2.0], [2.0, 1.5, 1.0], [4.0, 1.0, 3.0]])
    result = paired_bootstrap_minimizer_frequencies(
        risks, np.array([0, 1, 2]), bootstrap_samples=200, seed=19
    )
    assert result["frequencies"].shape == (3,)
    assert np.sum(result["counts"]) == 200
    assert np.sum(result["frequencies"]) == pytest.approx(1.0)


def test_rank_deficient_runner_smoke_and_schema(tmp_path):
    manifest, trials, curves, minimizers, frequencies = run_rank_deficient_bridge(
        output_dir=tmp_path,
        budgets=(16, 20),
        primary_budget=20,
        trials=2,
        dimension=20,
        step_ranks=(2, 5),
        tail_levels=(0.0, 1e-10),
        min_residual_probes=4,
        bootstrap_samples=50,
    )
    assert len(manifest) == 8
    assert len(trials) == 4 * 2 * ((6 + 1) + (8 + 1))
    assert len(curves) == 4 * ((6 + 1) + (8 + 1))
    assert len(minimizers) == 4 * 2 * len(RISK_COLUMNS)
    assert len(frequencies) == len(RISK_COLUMNS) * len(curves)
    assert np.all(trials["q"] + trials["r_actual"] + trials["ell"] == trials["budget"])
    for filename in (
        "manifest",
        "trials",
        "curves",
        "minimizers",
        "minimizer_frequencies",
    ):
        assert (tmp_path / f"risk_bridge_rank_deficient_{filename}.csv").exists()


def test_existing_bridge_postprocessor_is_read_only(tmp_path):
    rows = []
    for trial in range(3):
        for q in range(3):
            rows.append(
                {
                    "setup_index": 0,
                    "spectrum_family": "fixture",
                    "setup_name": "fixture",
                    "matrix_seed": 1,
                    "basis_trial": trial,
                    "q": q,
                    "oracle_risk": [3.0, 1.0, 2.0][q],
                    "gaussian_risk": [3.0, 1.0 + 0.1 * trial, 2.0][q],
                    "rademacher_risk": [2.0, 1.0, 1.5 + 0.1 * trial][q],
                }
            )
    input_path = tmp_path / "trials.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    before = input_path.read_bytes()
    curves, frequencies = postprocess_exact_bridge(
        input_path,
        bootstrap_samples=50,
        bootstrap_seed_base=22,
        output_dir=tmp_path,
    )
    assert input_path.read_bytes() == before
    assert len(curves) == 3
    assert len(frequencies) == 9
    assert np.allclose(
        frequencies.groupby("risk_name")["bootstrap_frequency"].sum(), 1.0
    )
