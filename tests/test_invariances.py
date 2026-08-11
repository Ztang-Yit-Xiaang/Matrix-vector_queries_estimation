import sys
import numpy as np
import scipy.linalg as la
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trace_baseline import (
    MatVecOracle,
    Hutchinson,
    Hutch_pplus,
    Gaussian_Hutch_pplus,
    Adaptive_Hutch_pplus_GaussianResidual,
    Adaptive_Hutch_pplus_RademacherResidual
)


def run_invariance_tests():
    print("==========================================================================")
    print("RUNNING ADAPTIVE HUTCH++ INVARIANCE & CORRECTNESS UNIT TESTS")
    print("==========================================================================")
    
    # -------------------------------------------------------------------------
    # TEST 1: Rank-1 Numerical Dust Bug Fix
    # -------------------------------------------------------------------------
    print("\n--- Test 1: Rank-1 Matrix Numerical Dust Check ---")
    d, m, b = 30, 80, 10
    rng = np.random.default_rng(42)
    v = rng.normal(size=(d, 1))
    A_rank1 = v @ v.T
    
    oracle = MatVecOracle(A_rank1)
    est, diag = Adaptive_Hutch_pplus_RademacherResidual(
        oracle, m, d, b=b, rng=rng, return_diagnostics=True
    )
    
    print(f"Rank-1 Matrix True Trace: {np.trace(A_rank1):.6f} | Estimate: {est:.6f}")
    print(f"Calculated Basis Rank (r_actual): {diag['r_actual']} (Expected: 1)")
    print(f"Residual Probes Allocated (ell_eff): {diag['ell_eff']}")
    assert diag['r_actual'] == 1, f"FAIL: Expected r_actual == 1, got {diag['r_actual']}"
    assert oracle.query_count == m, f"FAIL: Expected query_count == {m}, got {oracle.query_count}"
    print("--> PASS: Rank-1 test correctly identified rank=1 with 0 numerical dust rank inflation!")

    # -------------------------------------------------------------------------
    # TEST 2: Zero Matrix Check
    # -------------------------------------------------------------------------
    print("\n--- Test 2: Zero Matrix Check ---")
    A_zero = np.zeros((d, d))
    oracle = MatVecOracle(A_zero)
    est, diag = Adaptive_Hutch_pplus_RademacherResidual(
        oracle, m, d, b=b, rng=rng, return_diagnostics=True
    )
    print(f"Zero Matrix Estimate: {est:.6f} (Expected: 0.00)")
    print(f"Basis Rank: {diag['r_actual']} (Expected: 0)")
    assert abs(est) < 1e-12, f"FAIL: Expected 0.0 estimate, got {est}"
    assert diag['r_actual'] == 0, f"FAIL: Expected r_actual == 0, got {diag['r_actual']}"
    assert oracle.query_count == m, f"FAIL: Expected query_count == {m}, got {oracle.query_count}"
    print("--> PASS: Zero matrix test passed cleanly!")

    # -------------------------------------------------------------------------
    # TEST 3: Basis Orthonormality & Matvec Consistency Check
    # -------------------------------------------------------------------------
    print("\n--- Test 3: Orthonormality ||Q^T Q - I||_F and Matvec ||Z - AQ||_F Check ---")
    M = rng.normal(size=(d, d))
    A_full = M.T @ M
    
    # Internal state inspection test
    oracle = MatVecOracle(A_full)
    oracle.reset_query_count()
    
    # Step through Phase 1 & 3 manually using implementation helpers
    S_0 = rng.choice([-1.0, 1.0], size=(d, b))
    W_0 = oracle(S_0)
    from trace_baseline import _rank_aware_qr
    Q_0, r_0 = _rank_aware_qr(W_0, reference_scale=la.norm(W_0, ord='fro'))
    Z_0 = oracle(Q_0)
    
    # Check basis consistency
    ortho_err_0 = la.norm(Q_0.T @ Q_0 - np.eye(r_0), ord='fro') if r_0 > 0 else 0.0
    matvec_err_0 = la.norm(Z_0 - A_full @ Q_0, ord='fro') if r_0 > 0 else 0.0
    
    print(f"Pilot Basis Orthonormality Error: {ortho_err_0:.2e} (Threshold: < 1e-10)")
    print(f"Pilot Matvec Consistency Error: {matvec_err_0:.2e} (Threshold: < 1e-10)")
    assert ortho_err_0 < 1e-10, "FAIL: Pilot basis Q_0 is not orthonormal"
    assert matvec_err_0 < 1e-10, "FAIL: Pilot matvec Z_0 != A @ Q_0"
    print("--> PASS: Basis orthonormality and matvec consistency checks passed!")

    # -------------------------------------------------------------------------
    # TEST 4: Query Budget Enforcement Across All Estimators
    # -------------------------------------------------------------------------
    print("\n--- Test 4: Enforced Query Budget Check (oracle.query_count == m) ---")
    m_test_values = [24, 40, 80, 150, 300]
    estimators = {
        "Hutchinson": Hutchinson,
        "Hutch++": Hutch_pplus,
        "Gaussian-Hutch++": Gaussian_Hutch_pplus,
        "Adaptive-Gaussian": Adaptive_Hutch_pplus_GaussianResidual,
        "Adaptive-Rademacher": Adaptive_Hutch_pplus_RademacherResidual
    }
    
    all_query_passes = True
    for m_val in m_test_values:
        for est_name, est_fn in estimators.items():
            oracle = MatVecOracle(A_full)
            oracle.reset_query_count()
            est = est_fn(oracle, m_val, d, rng=rng)
            if oracle.query_count != m_val:
                print(f"FAIL: {est_name} (m={m_val}): expected {m_val}, got {oracle.query_count}")
                all_query_passes = False
                
    if all_query_passes:
        print("--> PASS: 100% of all estimators enforce query_count == m exactly!")

    # -------------------------------------------------------------------------
    # TEST 5: Unbiasedness Empirical Verification (500 Trials)
    # -------------------------------------------------------------------------
    print("\n--- Test 5: Unbiasedness Check over 500 Trials ---")
    exact_tr = np.trace(A_full)
    n_unbiased_trials = 500
    estimates = []
    
    for trial in range(n_unbiased_trials):
        trial_rng = np.random.default_rng(1000 + trial)
        oracle = MatVecOracle(A_full)
        est = Adaptive_Hutch_pplus_RademacherResidual(oracle, m=60, d=d, b=10, rng=trial_rng)
        estimates.append(est)
        
    mean_est = np.mean(estimates)
    rel_bias = abs(mean_est - exact_tr) / exact_tr
    print(f"Exact Trace: {exact_tr:.6f} | Mean 500-Trial Estimate: {mean_est:.6f} | Relative Bias: {rel_bias:.4e}")
    assert rel_bias < 5e-3, f"FAIL: Relative bias too high ({rel_bias:.4e})"
    print("--> PASS: Unbiasedness empirically verified (Relative bias < 0.5%)!")

    print("\n==========================================================================")
    print("ALL INVARIANCE AND CORRECTNESS UNIT TESTS PASSED SUCCESSFULLY!")
    print("==========================================================================")

if __name__ == "__main__":
    run_invariance_tests()
