import sys
import numpy as np
import scipy.linalg as la
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trace_baseline import (
    MatVecOracle,
    Adaptive_Hutch_pplus_SequentialPilot,
    _rank_aware_qr
)

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


if __name__ == "__main__":
    test_exact_budget_identity_and_diagnostics()
    test_step_spectrum_post_knee_noise_floor()
    test_exponential_tail_ratio_against_numerical_sum()
    test_sequential_stopping_rule_regression()
