import sys
import numpy as np
import scipy.linalg as la
import pytest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trace_baseline import (
    MatVecOracle,
    Hutch_pplus,
    Adaptive_Hutch_pplus_SequentialPilot,
    Adaptive_Hutch_pplus_MarginalRisk,
    marginal_certificate_decision,
    marginal_risk_quantity,
    _rank_aware_qr
)


def test_standard_hutchpp_reports_rank_aware_allocation():
    d = 20
    m = 18
    matrix = np.diag([3.0, 1.0] + [0.0] * (d - 2))
    oracle = MatVecOracle(matrix, d=d)

    estimate, diagnostics = Hutch_pplus(
        oracle,
        m,
        d,
        rng=np.random.default_rng(3),
        return_diagnostics=True,
    )

    assert np.isfinite(estimate)
    assert diagnostics["q_target"] == m // 3
    assert diagnostics["r_actual"] == 2
    assert diagnostics["q_target"] + diagnostics["r_actual"] + diagnostics["ell_eff"] == m
    assert oracle.query_count == m

def test_exact_budget_identity_and_diagnostics():
    print("--- Test 1: Exact Budget Identity q + r + ell == m ---")
    d = 500
    m = 160
    rng = np.random.default_rng(42)
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = (np.arange(1, d + 1, dtype=np.float64)) ** (-2.0)
    A = (Q_orth * eigenvals) @ Q_orth.T
    oracle = MatVecOracle(A, d=d)

    est, diag = Adaptive_Hutch_pplus_SequentialPilot(
        oracle, m, d, b_0=8, delta_b=4, rng=np.random.default_rng(42), return_diagnostics=True
    )
    print(f"Diagnostics: q={diag['q_target']}, r={diag['r_actual']}, ell={diag['ell_eff']}, b_final={diag['b_final']}")
    assert diag['q_target'] + diag['r_actual'] + diag['ell_eff'] == m, "Budget identity q + r + ell == m violated!"
    assert oracle.query_count == m, "Oracle query count mismatch!"
    assert np.isclose(diag['tau_ratio_threshold'], np.exp(diag['gamma_gap_threshold']))
    print("--> PASS: Exact budget identity q + r + ell == m strictly satisfied!")


def test_step_spectrum_post_knee_noise_floor():
    print("\n--- Test 2: Theorem 7 Post-Knee Ritz Noise Floor (theta_{r+1} ... theta_{r+p} == eta) ---")
    d = 200
    r_step = 10
    p_over = 5
    b = r_step + p_over
    eta = 0.01
    rng = np.random.default_rng(42)

    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = np.ones(d, dtype=np.float64) * eta
    eigenvals[:r_step] = 1.0
    A = (Q_orth * eigenvals) @ Q_orth.T

    S = rng.normal(size=(d, b))
    Y = A @ S
    Q_mat, _ = _rank_aware_qr(Y, reference_scale=la.norm(Y, 'fro'))
    M = 0.5 * (Q_mat.T @ (A @ Q_mat) + (A @ Q_mat).T @ Q_mat)
    ritz_vals = la.eigvalsh(M)[::-1]

    post_knee_vals = ritz_vals[r_step:]
    print(f"Top {r_step} Ritz values min: {ritz_vals[r_step-1]:.6f}")
    print(f"Post-knee {p_over} Ritz values: {post_knee_vals}")
    
    assert len(post_knee_vals) == p_over, "Incorrect post-knee Ritz count!"
    assert np.allclose(post_knee_vals, eta, atol=1e-10), "Post-knee Ritz values are not exactly equal to eta!"
    print("--> PASS: Theorem 7 verified! Post-knee Ritz values equal eta exactly!")


def test_exponential_tail_ratio_against_numerical_sum():
    print("\n--- Test 3: Theorem 5 Exponential Tail Log-Ratio Formula vs Numerical Sum ---")
    d = 1000
    b = 10
    q = 50
    alpha = 0.08
    delta = 0.01
    theta_b = 0.5

    # Direct numerical summation for finite d
    i_indices = np.arange(q + 1, d + 1, dtype=np.float64)
    true_tail_num = float(np.sum((theta_b * np.exp(-alpha * (i_indices - b))) ** 2))
    est_tail_num = float(np.sum((theta_b * np.exp(-(alpha + delta) * (i_indices - b))) ** 2))

    log_ratio_num = np.log(est_tail_num / true_tail_num)
    
    # Theorem 5 Theoretical Formula (Infinite Tail)
    log_ratio_theo = -2.0 * delta * (q + 1 - b) + np.log((1.0 - np.exp(-2.0 * alpha)) / (1.0 - np.exp(-2.0 * (alpha + delta))))

    print(f"Numerical Log-Ratio (d={d}): {log_ratio_num:.8f}")
    print(f"Theorem 5 Log-Ratio Formula: {log_ratio_theo:.8f}")
    assert abs(log_ratio_num - log_ratio_theo) < 1e-6, "Theorem 5 formula mismatch!"
    print("--> PASS: Theorem 5 exponential tail log-ratio formula verified!")


def test_sequential_stopping_rule_regression():
    print("\n--- Test 4: Sequential Stopping Rule Regression ---")
    d = 500
    m = 160
    rng = np.random.default_rng(42)

    # Setup A: Unresolved head plateau (Step r=20, start b_0=8)
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals_step = np.ones(d, dtype=np.float64) * 0.01
    eigenvals_step[:20] = 1.0
    A_step = (Q_orth * eigenvals_step) @ Q_orth.T
    oracle_step = MatVecOracle(A_step, d=d)

    _, diag_step = Adaptive_Hutch_pplus_SequentialPilot(
        oracle_step, m, d, b_0=8, delta_b=4, return_diagnostics=True
    )
    print(f"Unresolved Step r=20: Final b={diag_step['b_final']} (Expected >= 20)")
    assert diag_step['b_final'] >= 20, "Sequential pilot stopped too early before knee!"

    print("--> PASS: Stopping rule regression test passed!")


def test_sequential_trust_region_guard_and_default_regression():
    d = 160
    m = 80
    shift = 4
    matrix_rng = np.random.default_rng(314)
    Q_orth, _ = la.qr(matrix_rng.normal(size=(d, d)))
    eigenvals = np.arange(1, d + 1, dtype=np.float64) ** (-2.0)
    A = (Q_orth * eigenvals) @ Q_orth.T

    oracle_default = MatVecOracle(A, d=d)
    est_default, diag_default = Adaptive_Hutch_pplus_SequentialPilot(
        oracle_default,
        m,
        d,
        b_0=8,
        delta_b=4,
        rng=np.random.default_rng(2718),
        return_diagnostics=True,
    )

    oracle_none = MatVecOracle(A, d=d)
    est_none, diag_none = Adaptive_Hutch_pplus_SequentialPilot(
        oracle_none,
        m,
        d,
        b_0=8,
        delta_b=4,
        rng=np.random.default_rng(2718),
        return_diagnostics=True,
        max_q_shift=None,
    )

    assert est_default == est_none
    assert diag_default == diag_none
    assert diag_default["q_target"] == diag_default["q_adapt_raw"]
    assert diag_default["max_q_shift"] is None
    assert not diag_default["guard_applied"]
    assert oracle_default.query_count == oracle_none.query_count == m

    oracle_guarded = MatVecOracle(A, d=d)
    _, diag_guarded = Adaptive_Hutch_pplus_SequentialPilot(
        oracle_guarded,
        m,
        d,
        b_0=8,
        delta_b=4,
        rng=np.random.default_rng(2718),
        return_diagnostics=True,
        max_q_shift=shift,
    )

    q_max = min(d, (m - 2) // 2)
    lower = max(diag_guarded["b_final"], diag_guarded["q_0"] - shift)
    upper = min(q_max, diag_guarded["q_0"] + shift)
    expected = int(np.clip(diag_guarded["q_adapt_raw"], lower, upper))

    assert not diag_guarded["guard_relaxed_for_pilot_floor"]
    assert diag_guarded["q_target"] == expected
    assert lower <= diag_guarded["q_target"] <= upper
    assert diag_guarded["guard_applied"] == (expected != diag_guarded["q_adapt_raw"])
    assert diag_guarded["q_target"] + diag_guarded["r_actual"] + diag_guarded["ell_eff"] == m
    assert diag_guarded["ell_eff"] >= 2
    assert oracle_guarded.query_count == m


def test_legacy_and_explicit_symmetric_guard_are_equivalent():
    d = 80
    m = 60
    eigenvals = np.arange(1, d + 1, dtype=np.float64) ** (-1.4)
    A = np.diag(eigenvals)

    oracle_legacy = MatVecOracle(A, d=d)
    est_legacy, diag_legacy = Adaptive_Hutch_pplus_SequentialPilot(
        oracle_legacy,
        m,
        d,
        rng=np.random.default_rng(91),
        return_diagnostics=True,
        max_q_shift=4,
    )
    oracle_explicit = MatVecOracle(A, d=d)
    est_explicit, diag_explicit = Adaptive_Hutch_pplus_SequentialPilot(
        oracle_explicit,
        m,
        d,
        rng=np.random.default_rng(91),
        return_diagnostics=True,
        q_shift_bounds=(-4, 4),
    )

    assert est_legacy == est_explicit
    for key in (
        "b_final",
        "r_pilot_actual",
        "q_0",
        "q_adapt_raw",
        "q_target",
        "r_actual",
        "ell_eff",
        "guard_lower_effective",
        "guard_upper_effective",
    ):
        assert diag_legacy[key] == diag_explicit[key]
    assert oracle_legacy.query_count == oracle_explicit.query_count == m


def test_conflicting_guard_apis_are_rejected():
    oracle = MatVecOracle(np.eye(20), d=20)
    with pytest.raises(ValueError, match="at most one"):
        Adaptive_Hutch_pplus_SequentialPilot(
            oracle,
            m=20,
            d=20,
            b_0=4,
            max_q_shift=4,
            q_shift_bounds=(-4, 4),
        )


@pytest.mark.parametrize(
    "invalid_bounds",
    [
        4,
        (-4,),
        (-4, 4, 5),
        [-4, 4],
        (False, 4),
        (-4, True),
        (-4.0, 4),
        (-4, 4.0),
        (np.nan, 4),
        (-4, np.inf),
        (5, 4),
    ],
)
def test_explicit_guard_rejects_invalid_bounds(invalid_bounds):
    oracle = MatVecOracle(np.eye(20), d=20)
    with pytest.raises(ValueError, match="q_shift_bounds"):
        Adaptive_Hutch_pplus_SequentialPilot(
            oracle,
            m=20,
            d=20,
            b_0=4,
            q_shift_bounds=invalid_bounds,
        )


@pytest.mark.parametrize("bounds", [(0, 0), (0, 4)])
def test_nonnegative_guard_respects_standard_anchor(bounds):
    d = 100
    m = 80
    A = np.diag(np.exp(-0.08 * np.arange(1, d + 1, dtype=np.float64)))
    oracle = MatVecOracle(A, d=d)

    _, diag = Adaptive_Hutch_pplus_SequentialPilot(
        oracle,
        m,
        d,
        b_0=8,
        delta_b=4,
        b_max=26,
        rng=np.random.default_rng(17),
        return_diagnostics=True,
        q_shift_bounds=bounds,
        preserve_baseline_feasibility=True,
    )

    assert diag["q_target"] >= diag["q_0"]
    assert diag["q_target"] <= diag["q_0"] + bounds[1]
    if bounds == (0, 0):
        assert diag["q_target"] == diag["q_0"]
    assert diag["delta_q_safe"] == diag["q_target"] - diag["q_0"]
    assert diag["q_target"] + diag["r_actual"] + diag["ell_eff"] == m
    assert oracle.query_count == m


def test_baseline_preservation_caps_pilot_and_stage_grid_ends_at_52():
    d = 80
    m = 160
    q_0 = 53
    oracle = MatVecOracle(np.eye(d), d=d)

    _, diag = Adaptive_Hutch_pplus_SequentialPilot(
        oracle,
        m,
        d,
        b_0=8,
        delta_b=4,
        b_max=70,
        max_extrapolation_dist=-1,
        rng=np.random.default_rng(9),
        return_diagnostics=True,
        q_shift_bounds=(0, 4),
        preserve_baseline_feasibility=True,
    )

    assert diag["q_0"] == q_0
    assert diag["b_max_requested"] == 70
    assert diag["b_max_effective"] == q_0
    assert diag["pilot_cap_applied"]
    assert diag["b_final"] == 52
    assert diag["b_final"] <= q_0
    assert oracle.query_count == m


def test_baseline_preservation_rejects_pilot_above_standard_anchor():
    oracle = MatVecOracle(np.eye(20), d=20)
    with pytest.raises(ValueError, match=r"Standard-Hutch\+\+ anchor"):
        Adaptive_Hutch_pplus_SequentialPilot(
            oracle,
            m=20,
            d=20,
            b_0=8,
            preserve_baseline_feasibility=True,
        )


@pytest.mark.parametrize("invalid_shift", [True, -1, 1.5, np.nan, np.inf])
def test_sequential_trust_region_rejects_invalid_radius(invalid_shift):
    A = np.eye(12)
    oracle = MatVecOracle(A, d=12)
    with pytest.raises(ValueError, match="max_q_shift"):
        Adaptive_Hutch_pplus_SequentialPilot(
            oracle,
            m=12,
            d=12,
            rng=np.random.default_rng(7),
            max_q_shift=invalid_shift,
        )


def test_sequential_rank_deficient_pilot_uses_query_count_not_rank():
    d = 20
    m = 20
    A = np.diag([3.0, 1.0] + [0.0] * (d - 2))
    oracle = MatVecOracle(A, d=d)

    _, diag = Adaptive_Hutch_pplus_SequentialPilot(
        oracle,
        m=m,
        d=d,
        b_0=4,
        delta_b=2,
        b_max=4,
        rng=np.random.default_rng(1),
        return_diagnostics=True,
        max_q_shift=1,
    )

    assert diag["b_final"] == 4
    assert diag["r_pilot_actual"] == 2
    assert diag["q_target"] + diag["r_actual"] + diag["ell_eff"] == m
    assert oracle.query_count == m


def test_sequential_rejects_infeasible_initial_pilot():
    A = np.eye(12)
    oracle = MatVecOracle(A, d=12)
    with pytest.raises(ValueError, match="b_0"):
        Adaptive_Hutch_pplus_SequentialPilot(
            oracle,
            m=12,
            d=12,
            b_0=8,
            rng=np.random.default_rng(7),
        )


def test_marginal_risk_sequential_allocator():
    print("\n--- Test 5: Marginal Risk Sequential Allocator ---")
    d = 500
    m = 160
    rng = np.random.default_rng(42)

    # Step spectrum r*=10
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = np.ones(d, dtype=np.float64) * 0.01
    eigenvals[:10] = 1.0
    A = (Q_orth * eigenvals) @ Q_orth.T
    oracle = MatVecOracle(A, d=d)

    est, diag = Adaptive_Hutch_pplus_MarginalRisk(
        oracle, m, d, b_0=8, delta_b=4, return_diagnostics=True
    )
    print(f"Marginal Risk Allocator: q_target={diag['q_target']}, r_actual={diag['r_actual']}, ell_eff={diag['ell_eff']}, stop_reason={diag['stop_reason']}")
    assert diag['q_target'] + diag['r_actual'] + diag['ell_eff'] == m, "Marginal Risk budget identity violated!"
    assert oracle.query_count == m, "Marginal Risk oracle query count mismatch!"
    print("--> PASS: Marginal Risk Sequential Allocator unit test passed!")


def _full_rank_oracle_risk(eigenvalues, m, q):
    return 2.0 * float(np.sum(np.asarray(eigenvalues)[q:] ** 2)) / (m - 2 * q)


def test_marginal_identity_and_single_crossing_on_monotone_spectra():
    rng = np.random.default_rng(20260814)
    for _ in range(10):
        eigenvalues = np.sort(rng.uniform(0.0, 2.0, size=30))[::-1]
        m = 40
        marginals = []
        for q in range(0, 19):
            marginal = marginal_risk_quantity(eigenvalues, m, q)
            difference = _full_rank_oracle_risk(eigenvalues, m, q + 1) - _full_rank_oracle_risk(eigenvalues, m, q)
            assert np.sign(difference) == -np.sign(marginal)
            marginals.append(marginal)
        assert np.all(np.diff(marginals) <= 1e-12)


def test_corrected_step_spectrum_minimizer_and_boundary_plateau():
    d = 50
    r_star = 8
    eta = 0.1
    eigenvalues = np.full(d, eta)
    eigenvalues[:r_star] = 1.0

    m = 30
    risks = [_full_rank_oracle_risk(eigenvalues, m, q) for q in range((m - 2) // 2 + 1)]
    assert int(np.argmin(risks)) == r_star

    boundary_eta = 0.5
    boundary_eigenvalues = np.full(d, boundary_eta)
    boundary_eigenvalues[:r_star] = 1.0
    boundary_budget = int(2 * r_star + 2 * (d - r_star) * boundary_eta**2)
    assert np.isclose(
        _full_rank_oracle_risk(boundary_eigenvalues, boundary_budget, 0),
        _full_rank_oracle_risk(boundary_eigenvalues, boundary_budget, r_star),
    )

    counterexample = np.full(500, 0.5)
    counterexample[:30] = 1.0
    counterexample_risks = [_full_rank_oracle_risk(counterexample, 160, q) for q in range(80)]
    assert int(np.argmin(counterexample_risks)) == 0


@pytest.mark.parametrize(
    ("marginal_hat", "radius", "expected"),
    [(3.0, 1.0, "increase"), (-3.0, 1.0, "stop"), (0.5, 1.0, "learn_more"), (1.0, 1.0, "learn_more")],
)
def test_marginal_certificate_three_state_decision(marginal_hat, radius, expected):
    assert marginal_certificate_decision(marginal_hat, radius) == expected


@pytest.mark.parametrize(
    ("marginal_hat", "radius"),
    [(0.0, -1.0), (np.nan, 1.0), (0.0, np.inf), (True, 1.0), (0.0, False)],
)
def test_marginal_certificate_rejects_invalid_inputs(marginal_hat, radius):
    with pytest.raises(ValueError):
        marginal_certificate_decision(marginal_hat, radius)


def test_marginal_rank_deficient_accounting_separates_queries_and_rank():
    d = 20
    m = 20
    matrix = np.diag([3.0, 1.0] + [0.0] * (d - 2))
    oracle = MatVecOracle(matrix, d=d)
    _, diagnostics = Adaptive_Hutch_pplus_MarginalRisk(
        oracle,
        m,
        d,
        b_0=4,
        delta_b=2,
        b_max=4,
        rng=np.random.default_rng(1),
        return_diagnostics=True,
    )
    assert diagnostics["b_final"] == 4
    assert diagnostics["r_pilot_actual"] == 2
    assert diagnostics["q_target"] + diagnostics["r_actual"] + diagnostics["ell_eff"] == m
    assert oracle.query_count == m


def test_marginal_resolved_knee_uses_committed_width_not_power_extrapolation():
    d = 500
    m = 160
    orientation_rng = np.random.default_rng(42)
    orientation, _ = la.qr(orientation_rng.normal(size=(d, d)))
    eigenvalues = np.full(d, 0.01)
    eigenvalues[:10] = 1.0
    matrix = (orientation * eigenvalues) @ orientation.T
    oracle = MatVecOracle(matrix, d=d)

    _, diagnostics = Adaptive_Hutch_pplus_MarginalRisk(
        oracle,
        m,
        d,
        b_0=12,
        delta_b=4,
        b_max=12,
        rng=np.random.default_rng(123),
        return_diagnostics=True,
    )
    assert diagnostics["heuristic_intervention_reason"] == "resolved_knee"
    assert diagnostics["q_adapt_raw"] > diagnostics["b_final"]
    assert diagnostics["q_target"] == diagnostics["b_final"] == 12
    assert oracle.query_count == m


def test_marginal_baseline_preservation_caps_stage_grid_at_52():
    d = 80
    m = 160
    oracle = MatVecOracle(np.zeros((d, d)), d=d)
    _, diagnostics = Adaptive_Hutch_pplus_MarginalRisk(
        oracle,
        m,
        d,
        b_0=8,
        delta_b=4,
        b_max=70,
        preserve_baseline_feasibility=True,
        rng=np.random.default_rng(4),
        return_diagnostics=True,
    )
    assert diagnostics["q_0"] == 53
    assert diagnostics["b_max_requested"] == 70
    assert diagnostics["b_max_effective"] == 53
    assert diagnostics["b_final"] == 52
    assert diagnostics["pilot_cap_applied"]
    assert diagnostics["q_target"] + diagnostics["r_actual"] + diagnostics["ell_eff"] == m
    assert oracle.query_count == m


@pytest.mark.parametrize(
    "kwargs",
    [
        {"b_0": 0},
        {"delta_b": 0},
        {"b_max": 3},
        {"tau_gap": 0.0},
        {"p_min": 0},
        {"probe_mode": "invalid"},
        {"preserve_baseline_feasibility": 1},
    ],
)
def test_marginal_rejects_invalid_configuration(kwargs):
    oracle = MatVecOracle(np.eye(20), d=20)
    configuration = {"b_0": 4, "b_max": 4}
    configuration.update(kwargs)
    with pytest.raises(ValueError):
        Adaptive_Hutch_pplus_MarginalRisk(
            oracle,
            m=20,
            d=20,
            **configuration,
        )


if __name__ == "__main__":
    test_exact_budget_identity_and_diagnostics()
    test_step_spectrum_post_knee_noise_floor()
    test_exponential_tail_ratio_against_numerical_sum()
    test_sequential_stopping_rule_regression()
    test_marginal_risk_sequential_allocator()

    test_sequential_trust_region_guard_and_default_regression()
    test_sequential_rank_deficient_pilot_uses_query_count_not_rank()
    test_sequential_rejects_infeasible_initial_pilot()
