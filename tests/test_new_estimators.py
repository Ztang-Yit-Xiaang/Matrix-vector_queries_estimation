import sys
import numpy as np
import scipy.linalg as la
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trace_baseline import (
    MatVecOracle,
    Adaptive_Hutch_pplus_SequentialPilot
)

def run_new_estimators_unit_tests():
    print("==========================================================================")
    print("RUNNING DIRECTION 1 UPGRADED SEQUENTIAL PILOT UNIT TESTS")
    print("==========================================================================")

    d = 500
    m = 160
    rng = np.random.default_rng(42)

    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    eigenvals = (np.arange(1, d + 1, dtype=np.float64)) ** (-2.0)
    A = (Q_orth * eigenvals) @ Q_orth.T
    oracle = MatVecOracle(A, d=d)

    # TEST 1: Sequential Pilot Query Budget Enforcement
    print("\n--- Test 1: Sequential Pilot Query Count (oracle.query_count == m) ---")
    oracle.reset_query_count()
    est_seq, diag_seq = Adaptive_Hutch_pplus_SequentialPilot(
        oracle, m, d, b_0=8, delta_b=4, tau_plateau=1.15, rng=np.random.default_rng(42), return_diagnostics=True
    )
    print(f"Sequential Pilot Estimate: {est_seq:.6f} | Final Pilot Size b: {diag_seq['b_final']} | Stop Reason: {diag_seq['stop_reason']}")
    assert oracle.query_count == m, f"Sequential pilot query mismatch: used {oracle.query_count}, expected {m}"
    print("--> PASS: Upgraded sequential pilot strictly enforces query budget m!")

    print("\n==========================================================================")
    print("ALL DIRECTION 1 UNIT TESTS PASSED SUCCESSFULLY!")
    print("==========================================================================")

if __name__ == "__main__":
    run_new_estimators_unit_tests()
