import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.linalg as la


ROOT_DIR = Path(__file__).resolve().parent.parent
for directory in (ROOT_DIR / "experiments", ROOT_DIR / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from direct_rademacher_certification_phase1a import (  # noqa: E402
    ACTION_OFFSETS,
    DECISION_ABSTAIN,
    DECISION_ACCEPT,
    DECISION_REJECT,
    FROZEN_ENERGY_ATOL,
    FROZEN_ENERGY_RTOL,
    PROJECTOR_VALIDATION_ATOL,
    TRUTH_BETTER,
    TRUTH_TIE,
    TRUTH_WORSE,
    all_pairs_variance,
    certification_estimators,
    certification_probes,
    certification_quadratic_forms,
    conditional_cluster_bootstrap,
    empirical_decision,
    exhaustive_rademacher_variance,
    paired_observations,
    reconstruct_actions,
    sample_variance,
    truth_label,
)
from run_direct_rademacher_certification_phase1a import (  # noqa: E402
    OUTPUT_FILENAMES,
    run_phase1a,
)
from run_rank_deficient_risk_bridge import (  # noqa: E402
    apply_structured_step,
    make_signal_basis,
)


def _small_actions(dimension=12, step_rank=3, eta=1e-6):
    signal_basis = make_signal_basis(dimension, step_rank, 52000)
    rng = np.random.default_rng(70000)
    sketch = rng.choice(
        [-1.0, 1.0],
        size=(dimension, step_rank + 2),
    )
    actions = reconstruct_actions(
        signal_basis,
        eta,
        sketch,
        step_rank,
    )
    return signal_basis, actions


def test_exact_rademacher_variance_identity_requires_symmetric_residual():
    rng = np.random.default_rng(14)
    matrix = rng.normal(size=(5, 5))
    matrix = 0.5 * (matrix + matrix.T)
    basis, _ = la.qr(rng.normal(size=(5, 2)), mode="economic")
    projector = np.eye(5) - basis @ basis.T
    residual = projector @ matrix @ projector
    assert np.allclose(residual, residual.T, rtol=0.0, atol=1e-13)
    enumerated = exhaustive_rademacher_variance(residual)
    formula = 4.0 * np.sum(np.triu(residual, k=1) ** 2)
    assert enumerated == pytest.approx(formula, rel=2e-13, abs=2e-13)
    nonsymmetric = residual.copy()
    nonsymmetric[0, 1] += 0.1
    with pytest.raises(ValueError, match="symmetric"):
        exhaustive_rademacher_variance(nonsymmetric)


def test_sample_variance_all_pairs_and_paired_expectation_identities():
    values = np.array([1.0, -2.0, 4.0, 7.0])
    assert sample_variance(values) == pytest.approx(all_pairs_variance(values))
    centered = values - np.mean(values)
    pair_grid = 0.5 * (centered[:, None] - centered[None, :]) ** 2
    assert np.mean(pair_grid) == pytest.approx(np.var(values, ddof=0))
    paired = paired_observations(values)
    assert np.array_equal(
        paired,
        0.5 * (values[0::2] - values[1::2]) ** 2,
    )


def test_mom_definitions_and_declared_degeneracies():
    values4 = np.array([0.0, 2.0, 3.0, 7.0])
    estimates4 = certification_estimators(values4)
    assert estimates4["mom_w1"] == pytest.approx(estimates4["paired_mean"])
    assert estimates4["mom_w2"] == pytest.approx(estimates4["paired_mean"])

    values8 = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 9.0, 12.0, 16.0])
    estimates8 = certification_estimators(values8)
    assert estimates8["mom_w2"] == pytest.approx(estimates8["paired_mean"])

    values16 = np.arange(16.0) ** 2
    estimates16 = certification_estimators(values16)
    assert estimates16["mom_w1"] != pytest.approx(estimates16["paired_mean"])
    assert estimates16["mom_w2"] != pytest.approx(estimates16["paired_mean"])


def test_seed_map_is_reproducible_and_excludes_analysis_fields():
    first, seed, digest = certification_probes(20, 1, 7, 2, 11)
    second, seed_again, digest_again = certification_probes(20, 1, 7, 2, 11)
    assert seed == seed_again
    assert digest == digest_again
    assert np.array_equal(first, second)
    assert set(np.unique(first)) == {-1.0, 1.0}
    for changed in (
        (0, 7, 2, 11),
        (1, 8, 2, 11),
        (1, 7, 3, 11),
        (1, 7, 2, 12),
    ):
        other, other_seed, other_digest = certification_probes(20, *changed)
        assert other_seed != seed
        assert other_digest != digest
        assert not np.array_equal(first, other)


def test_reconstruction_caches_aq_and_is_subspace_invariant():
    signal_basis, actions = _small_actions()
    assert set(actions) == set(ACTION_OFFSETS)
    for action in actions.values():
        assert action.reconstruction_query_count == action.q + action.r_actual
        assert action.projector_error_op < PROJECTOR_VALIDATION_ATOL
        assert action.orthogonality_error < 5e-10
        assert np.allclose(
            action.image_basis,
            apply_structured_step(action.basis, signal_basis, 1e-6),
            rtol=2e-12,
            atol=2e-13,
        )

    state = actions["k"]
    signs = np.ones(state.r_actual)
    signs[::2] = -1.0
    rotated = state.basis * signs
    projector_error = la.norm(
        state.basis @ state.basis.T - rotated @ rotated.T,
        ord=2,
    )
    assert projector_error < 1e-13
    assert not np.array_equal(state.basis, rotated)


def test_full_width_sketch_reproduces_frozen_phase1a_rows():
    step_rank = 5
    signal_basis = make_signal_basis(500, step_rank, 52000)
    rng = np.random.default_rng(70000)
    full_frozen_sketch = rng.choice([-1.0, 1.0], size=(500, 116))
    actions = reconstruct_actions(
        signal_basis,
        1e-6,
        full_frozen_sketch,
        step_rank,
    )
    frozen = pd.read_csv(
        ROOT_DIR / "results" / "risk_bridge_rank_deficient_trials.csv",
        usecols=[
            "step_rank",
            "basis_trial",
            "eta",
            "budget",
            "q",
            "r_actual",
            "ell",
            "rademacher_energy",
            "rademacher_risk",
        ],
    )
    frozen = frozen.loc[
        frozen["step_rank"].eq(step_rank)
        & frozen["basis_trial"].eq(0)
        & np.isclose(frozen["eta"], 1e-6, rtol=1e-12, atol=0.0)
        & frozen["budget"].eq(160)
        & frozen["q"].isin([4, 5, 6, 7])
    ].set_index("q")
    assert len(frozen) == 4
    for state in actions.values():
        row = frozen.loc[state.q]
        assert state.r_actual == int(row["r_actual"])
        assert state.rademacher_energy == pytest.approx(
            float(row["rademacher_energy"]),
            rel=FROZEN_ENERGY_RTOL,
            abs=FROZEN_ENERGY_ATOL,
        )
        assert state.exact_sigma2 / (160 - state.q - state.r_actual) == pytest.approx(
            float(row["rademacher_risk"]),
            rel=FROZEN_ENERGY_RTOL,
            abs=FROZEN_ENERGY_ATOL,
        )


def test_matrix_free_common_probe_forms_match_dense_and_count_32():
    signal_basis, actions = _small_actions()
    probes, _, _ = certification_probes(12, 0, 0, 0, 0)
    forms, query_count = certification_quadratic_forms(
        signal_basis, 1e-6, actions, probes
    )
    assert query_count == 32
    matrix = (
        1e-6 * np.eye(12)
        + (1.0 - 1e-6) * signal_basis @ signal_basis.T
    )
    for label, state in actions.items():
        residual = np.eye(12) - state.basis @ state.basis.T
        dense = np.sum(probes * (residual @ matrix @ residual @ probes), axis=0)
        assert np.allclose(forms[label], dense, rtol=2e-11, atol=2e-12)


def test_truth_and_empirical_decision_boundaries():
    assert truth_label(1.0, 2.0) == TRUTH_BETTER
    assert truth_label(2.0, 1.0) == TRUTH_WORSE
    assert truth_label(1.0, 1.0) == TRUTH_TIE
    assert empirical_decision(0.0, 0.0, 1.0 / 3.0) == DECISION_ABSTAIN
    assert empirical_decision(1.0, 1.0, 1.0 / 3.0) == DECISION_ABSTAIN
    assert empirical_decision(0.5, 1.0, 1.0 / 3.0) == DECISION_ACCEPT
    assert empirical_decision(1.0, 0.5, 1.0 / 3.0) == DECISION_REJECT
    assert empirical_decision(0.75, 1.0, 1.0 / 3.0) == DECISION_ABSTAIN
    with pytest.raises(ValueError):
        empirical_decision(-1.0, 1.0, 0.2)


def test_conditional_bootstrap_preserves_eligible_populations():
    path_rates = {
        5: np.array([0.0, 0.5]),
        15: np.array([0.25, 0.75, 1.0]),
        30: np.array([0.2]),
    }
    result = conditional_cluster_bootstrap(
        path_rates,
        samples=1000,
        seed=19,
    )
    expected = np.mean([0.25, 2.0 / 3.0, 0.2])
    assert result["evaluable"]
    assert result["point"] == pytest.approx(expected)
    assert result["ci_low"] <= result["point"] <= result["ci_high"]
    empty = conditional_cluster_bootstrap(
        {5: np.array([0.1]), 15: np.array([]), 30: np.array([0.2])},
        samples=20,
        seed=4,
    )
    assert not empty["evaluable"]
    assert np.isnan(empty["point"])


def test_phase1a_smoke_writes_unique_schema_and_preserves_accounting(tmp_path):
    result = run_phase1a(
        output_dir=tmp_path,
        dimension=60,
        step_ranks=(5, 15, 30),
        etas=(1e-10, 1e-6),
        budgets=(80, 160, 240),
        trials=2,
        batches=1,
        repetitions=2,
        bootstrap_samples=20,
    )
    expected = 3 * 2 * 2 * 1 * 2 * 4
    assert result["trial_rows"] == expected
    trials = pd.read_parquet(tmp_path / OUTPUT_FILENAMES["trials"])
    assert len(trials) == expected
    keys = ["step_rank", "basis_trial", "eta", "batch", "repetition", "s"]
    assert not trials.duplicated(keys).any()
    assert np.all(trials["certification_query_count"] == 32)
    assert np.all(trials["s"].isin((4, 8, 16, 32)))
    assert (
        tmp_path / "figures" / "direct_rademacher_certification_phase1a"
    ).is_dir()
    for name, filename in OUTPUT_FILENAMES.items():
        assert (tmp_path / filename).exists(), name

    manifest = pd.read_csv(tmp_path / OUTPUT_FILENAMES["manifest"])
    assert int(manifest.iloc[0]["expected_trial_rows"]) == expected
    truth = pd.read_csv(tmp_path / OUTPUT_FILENAMES["truth"])
    assert not truth.duplicated(
        ["step_rank", "basis_trial", "eta", "budget", "q"]
    ).any()
