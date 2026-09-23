import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
for directory in (ROOT_DIR / "src", EXPERIMENTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from run_q_rank_vs_realized_risk_mechanism import (  # noqa: E402
    DEFAULT_BASIS_SEED_BASE,
    DEFAULT_DIMENSION,
    DEFAULT_ETA,
    DEFAULT_ORIENTATION_SEED_BASE,
    add_budget_and_local_fields,
    reconstruct_mechanism_path,
    run_sensitivity,
    validate_against_frozen,
)
from run_rank_deficient_risk_bridge import make_signal_basis  # noqa: E402


def _representative_path(step_rank=5, trial=0):
    signal_basis = make_signal_basis(
        DEFAULT_DIMENSION, step_rank, DEFAULT_ORIENTATION_SEED_BASE
    )
    rng = np.random.default_rng(DEFAULT_BASIS_SEED_BASE + trial)
    sketch = rng.choice([-1.0, 1.0], size=(DEFAULT_DIMENSION, 116))
    path = reconstruct_mechanism_path(
        signal_basis,
        DEFAULT_ETA,
        sketch,
        range(step_rank - 1, step_rank + 2),
    )
    path["dimension"] = DEFAULT_DIMENSION
    path["step_rank"] = step_rank
    path["eta"] = DEFAULT_ETA
    path["basis_trial"] = trial
    return path


def test_reconstruction_matches_frozen_path_and_accounting():
    expanded = add_budget_and_local_fields(_representative_path(), budgets=(160,))
    validate_against_frozen(expanded)
    assert np.all(expanded["r_actual"] == expanded["q"])
    assert np.all(expanded["ell"] == 160 - 2 * expanded["q"])
    assert np.all(
        expanded["constructed_basis_queries"]
        == expanded["q"] + expanded["r_actual"]
    )


def test_square_graph_identity_and_nested_capture_monotonicity():
    step_rank = 5
    path = _representative_path(step_rank=step_rank)
    square = path.loc[path["q"] == step_rank].iloc[0]
    oversampled = path.loc[path["q"] == step_rank + 1].iloc[0]
    assert square["s1_full_row_rank"]
    assert square["graph_identity_error"] < 1e-10
    assert oversampled["s1_sigma_min"] >= square["s1_sigma_min"] - 1e-12
    assert oversampled["s1_pinv_norm"] <= square["s1_pinv_norm"] + 1e-12
    assert (
        oversampled["subspace_projection_op"]
        <= square["subspace_projection_op"] + 1e-12
    )


def test_local_energy_ratio_is_equivalent_to_risk_improvement():
    expanded = add_budget_and_local_fields(_representative_path(), budgets=(160,))
    valid = expanded["next_q"] == expanded["q"] + 1
    for risk in ("gaussian", "rademacher"):
        lhs = expanded.loc[valid, f"{risk}_energy_ratio_next"]
        rhs = expanded.loc[valid, "local_energy_ratio_threshold"]
        observed = expanded.loc[valid, f"{risk}_local_improves"].astype(bool)
        assert np.array_equal((lhs < rhs).to_numpy(), observed.to_numpy())


def test_sensitivity_smoke_is_finite_and_reports_shifts():
    frame = run_sensitivity(
        dimensions=(40,),
        step_ranks=(3,),
        etas=(1e-6, 1e-3),
        budgets=(20, 30),
        batches=1,
        trials=2,
    )
    assert len(frame) == 4
    assert np.all(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()))
    assert {
        "q_rank_star",
        "q_gaussian_star",
        "q_rademacher_star",
        "gaussian_shift_from_rank",
        "rademacher_shift_from_rank",
    }.issubset(frame.columns)
