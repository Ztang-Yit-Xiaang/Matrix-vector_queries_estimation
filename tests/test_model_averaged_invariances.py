import sys
import numpy as np
import scipy.linalg as la
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trace_baseline import MatVecOracle, Adaptive_Hutch_pplus_ModelAveraged

def run_model_averaged_invariance_tests():
    print("==========================================================================")
    print("RUNNING MODEL-AVERAGED ADAPTIVE HUTCH++ INVARIANCE TESTS (PART K)")
    print("==========================================================================")

    d = 500
    m = 160
    b = 10
    rng = np.random.default_rng(42)

    # TEST 1: Model Weights Properties (w_j >= 0 and sum(w_j) == 1.0)
    print("\n--- Test 1: Model Weights Validity (w_j >= 0, sum(w_j) == 1.0) ---")
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = (np.arange(1, d + 1, dtype=np.float64)) ** (-2.0)
    A_pow = (Q_orth * eigenvals) @ Q_orth.T
    oracle_pow = MatVecOracle(A_pow, d=d)

    _, diag = Adaptive_Hutch_pplus_ModelAveraged(
        oracle_pow, m, d, b=b, rng=np.random.default_rng(42), return_diagnostics=True
    )
    weights = diag["weights"]
    w_sum = sum(weights.values())
    w_nonneg = all(w >= 0.0 for w in weights.values())
    print(f"Weights for Power-Law c=2: {weights} | Sum: {w_sum:.6f}")
    assert w_nonneg, "Model weights contain negative values!"
    assert abs(w_sum - 1.0) < 1e-6, f"Model weights do not sum to 1.0! Sum={w_sum}"
    print("--> PASS: Model weights satisfy non-negativity and sum-to-one!")

    # TEST 2: Two-Layer Confidence Gamma in [0, 1]
    print("\n--- Test 2: Two-Layer Confidence Gamma Range (gamma in [0, 1]) ---")
    gamma = diag["gamma"]
    print(f"Confidence gamma: {gamma:.4f}")
    assert 0.0 <= gamma <= 1.0, f"Gamma out of range [0, 1]: {gamma}"
    print("--> PASS: Gamma strictly lies in [0, 1]!")

    # TEST 3: Allocation Feasibility (b <= q_final <= q_max)
    print("\n--- Test 3: Allocation Feasibility Bounds ---")
    q_max = min(d, (m - 2) // 2)
    q_final = diag["q_target"]
    print(f"Allocation q_final = {q_final} | Feasible Range: [{b}, {q_max}]")
    assert b <= q_final <= q_max, f"Allocation q_final={q_final} violates feasibility bounds [{b}, {q_max}]"
    print("--> PASS: Allocation respects pilot floor and maximum feasibility!")

    # TEST 4: Exact Query Budget Enforcement (oracle.query_count == m)
    print("\n--- Test 4: Exact Query Budget Enforcement (oracle.query_count == m) ---")
    oracle_pow.reset_query_count()
    est = Adaptive_Hutch_pplus_ModelAveraged(oracle_pow, m, d, b=b, rng=np.random.default_rng(42))
    assert oracle_pow.query_count == m, f"Query count mismatch: used {oracle_pow.query_count}, expected {m}"
    print(f"--> PASS: Exact query budget enforced! ({oracle_pow.query_count} == {m})")

    # TEST 5: Determinism under Fixed Seed
    print("\n--- Test 5: Determinism Check ---")
    oracle_pow.reset_query_count()
    est1 = Adaptive_Hutch_pplus_ModelAveraged(oracle_pow, m, d, b=b, rng=np.random.default_rng(123))
    oracle_pow.reset_query_count()
    est2 = Adaptive_Hutch_pplus_ModelAveraged(oracle_pow, m, d, b=b, rng=np.random.default_rng(123))
    assert est1 == est2, f"Non-deterministic estimates under fixed seed: {est1} != {est2}"
    print("--> PASS: Estimator is 100% deterministic under fixed seed!")

    # TEST 6: Zero Matrix & Pathological Spectra Handling
    print("\n--- Test 6: Zero Matrix Pathological Safety ---")
    A_zero = np.zeros((d, d))
    oracle_zero = MatVecOracle(A_zero, d=d)
    est_zero, diag_zero = Adaptive_Hutch_pplus_ModelAveraged(
        oracle_zero, m, d, b=b, rng=np.random.default_rng(42), return_diagnostics=True
    )
    assert oracle_zero.query_count == m, "Zero matrix failed query budget enforcement!"
    assert est_zero == 0.0, f"Zero matrix trace non-zero: {est_zero}"
    assert np.isfinite(est_zero), "Zero matrix produced NaN/Inf!"
    print("--> PASS: Pathological zero matrix handled with zero error and valid query accounting!")

    print("\n==========================================================================")
    print("ALL PART K MODEL-AVERAGED INVARIANCE TESTS PASSED CLEANLY!")
    print("==========================================================================")

if __name__ == "__main__":
    run_model_averaged_invariance_tests()
