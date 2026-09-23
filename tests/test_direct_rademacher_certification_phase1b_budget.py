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
    DECISION_ABSTAIN,
    DECISION_ACCEPT,
    DECISION_REJECT,
    EPSILONS,
    ESTIMATORS,
)
from postprocess_direct_rademacher_certification_phase1b_budget import (  # noqa: E402
    CONFIG_COLUMNS,
    PaidAccounting,
    bootstrap_summaries,
    budget_decision_array,
    build_path_artifact,
    evaluate_verdict,
    nested_reuse_errors,
    paid_accounting,
    paid_risk_decomposition,
    summarize_paths,
)


def test_nested_reuse_lemma_and_nonnested_rejection():
    rng = np.random.default_rng(19)
    q_high, _ = la.qr(rng.normal(size=(12, 5)), mode="economic")
    rotation, _ = la.qr(rng.normal(size=(5, 3)), mode="economic")
    q_low = q_high @ rotation
    matrix = rng.normal(size=(12, 12))
    aq_high = matrix @ q_high
    aq_low = matrix @ q_low
    errors = nested_reuse_errors(q_low, aq_low, q_high, aq_high)
    assert errors["projector_error"] < 1e-12
    assert errors["cached_aq_error"] < 1e-12

    q_bad, _ = la.qr(rng.normal(size=(12, 3)), mode="economic")
    bad = nested_reuse_errors(q_bad, matrix @ q_bad, q_high, aq_high)
    assert bad["projector_error"] > 1e-3
    with pytest.raises(ValueError, match="nested"):
        paid_accounting(160, 16, 16, 15, 15, 16, nested=False)


def test_paid_accounting_is_rank_aware_and_charges_sunk_cost():
    full_rank = paid_accounting(160, 16, 16, 15, 15, 16)
    assert full_rank == PaidAccounting(32, 30, 32, 130, 112)
    assert full_rank.extra_cost == 2
    assert full_rank.cost_multiplier == pytest.approx(130 / 112)
    assert full_rank.net_numerator_threshold == pytest.approx(112 / 130)

    rank_deficient = paid_accounting(80, 12, 5, 10, 5, 8)
    assert rank_deficient.candidate_cost == 17
    assert rank_deficient.baseline_cost == 15
    assert rank_deficient.paid_ell == 55
    with pytest.raises(ValueError, match="no residual capacity"):
        paid_accounting(20, 8, 8, 7, 7, 5)


def test_no_free_fallback_and_exact_net_threshold_with_zero_boundary():
    accounting = paid_accounting(160, 16, 16, 15, 15, 16)
    risks = paid_risk_decomposition(80.0, 100.0, accounting)
    assert risks["paid_baseline"] / risks["original_baseline"] == pytest.approx(
        accounting.original_ell / accounting.paid_ell
    )
    assert risks["paid_candidate"] < risks["original_baseline"]
    assert 80.0 / 100.0 < accounting.paid_ell / accounting.original_ell
    assert risks["paid_oracle"] <= risks["paid_candidate"]
    assert risks["paid_oracle"] <= risks["paid_baseline"]

    zero = paid_risk_decomposition(0.0, 0.0, accounting)
    assert all(float(value) == 0.0 for value in zero.values())


def test_budget_decision_uses_common_numerator_denominator():
    candidate_hat = np.array([90.0, 40.0, 100.0])
    baseline_hat = np.array([100.0, 100.0, 100.0])
    decisions = budget_decision_array(candidate_hat, baseline_hat, 1.0 / 3.0)
    assert np.array_equal(
        decisions,
        np.array([DECISION_ABSTAIN, DECISION_ACCEPT, DECISION_ABSTAIN]),
    )
    equal = budget_decision_array(np.zeros(2), np.zeros(2), 0.0)
    assert np.all(equal == DECISION_ABSTAIN)

    # Separate Phase 1A denominators can reverse a near-boundary comparison.
    old_candidate_risk = 90.0 / 128.0
    old_baseline_risk = 100.0 / 130.0
    assert old_candidate_risk < old_baseline_risk
    assert decisions[0] == DECISION_ABSTAIN


def _mini_core(paths_per_rank=2):
    rows = []
    for basis_trial in range(paths_per_rank):
        for batch in range(4):
            for repetition in range(50):
                row = {
                    "rank_index": 0,
                    "step_rank": 5,
                    "basis_trial": basis_trial,
                    "eta": 1e-6,
                    "batch": batch,
                    "repetition": repetition,
                    "s": 4,
                    "orientation_seed": 52000,
                    "basis_seed": 70000 + basis_trial,
                }
                q_values = {"km1": 4, "k": 5, "kp1": 6, "kp2": 7}
                sigma_values = {
                    "km1": 130.0,
                    "k": 100.0 + basis_trial,
                    "kp1": 40.0 + basis_trial,
                    "kp2": 35.0 + basis_trial,
                }
                for action, q in q_values.items():
                    row[f"q__{action}"] = q
                    row[f"r_actual__{action}"] = q
                    row[f"reconstruction_queries__{action}"] = 2 * q
                    row[f"sigma2_exact__{action}"] = sigma_values[action]
                    for estimator_index, estimator in enumerate(ESTIMATORS):
                        perturbation = 1.0 + 0.02 * estimator_index
                        row[f"sigma2_hat__{action}__{estimator}"] = (
                            sigma_values[action] * perturbation
                        )
                rows.append(row)
    return pd.DataFrame(rows)


def test_path_artifact_count_keys_and_decision_mapping():
    core = _mini_core(paths_per_rank=2)
    paths = build_path_artifact(core, budgets=(80,))
    expected = 2 * 1 * 1 * 1 * 3 * len(ESTIMATORS) * len(EPSILONS)
    assert len(paths) == expected
    key = [
        "step_rank",
        "basis_trial",
        "eta",
        "budget",
        "s",
        "pair",
        "estimator",
        "epsilon",
    ]
    assert not paths.duplicated(key).any()
    primary = paths.loc[
        paths["pair"].eq("primary")
        & paths["estimator"].eq("sample_variance")
        & np.isclose(paths["epsilon"], 1.0 / 3.0)
    ]
    assert np.all(primary["ell_original"] == 70)
    assert np.all(primary["ell_paid"] == 64)
    assert np.all(primary["accept_probability"] == 1.0)
    assert np.allclose(
        primary["risk_paid_selected_mean"],
        primary["risk_paid_candidate"],
        rtol=1e-13,
        atol=1e-13,
    )


def test_infeasible_secondary_configurations_are_retained_and_not_averaged():
    core = _mini_core(paths_per_rank=2)
    paths = build_path_artifact(core, budgets=(18,))
    primary = paths.loc[
        paths["pair"].eq("primary")
        & paths["estimator"].eq("sample_variance")
        & np.isclose(paths["epsilon"], 1.0 / 3.0)
    ]
    assert len(primary) == 2
    assert primary["accounting_feasible"].all()
    assert np.all(primary["ell_paid"] == 2)

    right = paths.loc[
        paths["pair"].eq("right")
        & paths["estimator"].eq("sample_variance")
        & np.isclose(paths["epsilon"], 1.0 / 3.0)
    ]
    assert not right["accounting_feasible"].any()
    assert np.all(right["ell_paid"] == 0)
    assert right["risk_paid_selected_mean"].isna().all()
    summary = summarize_paths(paths)
    row = summary.loc[
        summary["rank_scope"].eq("equal_rank")
        & summary["pair"].eq("right")
        & summary["estimator"].eq("sample_variance")
        & np.isclose(summary["epsilon"], 1.0 / 3.0)
    ].iloc[0]
    assert not bool(row.evaluable)
    assert np.isnan(row.selected_mean_ratio)


def test_ratio_of_means_and_equal_rank_aggregation_are_explicit():
    core = _mini_core(paths_per_rank=2)
    paths = build_path_artifact(core, budgets=(80,))
    summary = summarize_paths(paths)
    row = summary.loc[
        summary["rank_scope"].eq("equal_rank")
        & summary["pair"].eq("primary")
        & summary["estimator"].eq("sample_variance")
        & np.isclose(summary["epsilon"], 1.0 / 3.0)
    ].iloc[0]
    selected = paths.loc[
        paths["pair"].eq("primary")
        & paths["estimator"].eq("sample_variance")
        & np.isclose(paths["epsilon"], 1.0 / 3.0),
        "risk_paid_selected_mean",
    ]
    baseline = paths.loc[
        paths["pair"].eq("primary")
        & paths["estimator"].eq("sample_variance")
        & np.isclose(paths["epsilon"], 1.0 / 3.0),
        "risk_original_baseline",
    ]
    assert row.selected_mean_ratio == pytest.approx(selected.mean() / baseline.mean())


def test_cluster_bootstrap_is_deterministic_and_verdict_logic_is_preregistered():
    core = _mini_core(paths_per_rank=2)
    paths = build_path_artifact(core, budgets=(80,))
    summary = summarize_paths(paths)
    first = bootstrap_summaries(paths, summary, samples=20, master_seed=93000)
    second = bootstrap_summaries(paths, summary, samples=20, master_seed=93000)
    pd.testing.assert_frame_equal(first, second)

    primary_summary = pd.DataFrame(
        [
            {
                "eta": 1e-6,
                "budget": 160,
                "s": 16,
                "pair": "primary",
                "estimator": "sample_variance",
                "epsilon": 1.0 / 3.0,
                "rank_scope": "equal_rank",
                "selected_mean_ratio": 0.9,
                "paid_oracle_mean_ratio": 0.8,
                "paid_baseline_mean_ratio": 1.1,
                "median_pathwise_ratio": 1.05,
                "fraction_paths_harmed": 0.7,
                "accept_probability": 0.2,
                "abstain_probability": 0.7,
            }
        ]
    )
    primary_bootstrap = pd.DataFrame(
        [
            {
                "eta": 1e-6,
                "budget": 160,
                "s": 16,
                "pair": "primary",
                "estimator": "sample_variance",
                "epsilon": 1.0 / 3.0,
                "metric": "selected_mean_ratio",
                "point": 0.9,
                "ci_low": 0.85,
                "ci_high": 0.95,
            }
        ]
    )
    verdict = evaluate_verdict(primary_summary, primary_bootstrap)
    assert verdict.loc[0, "verdict"] == "NET BENEFIT"
    assert verdict.loc[0, "tail_insurance_tradeoff"]
    assert "TAIL-INSURANCE" in verdict.loc[0, "verdict_label"]
