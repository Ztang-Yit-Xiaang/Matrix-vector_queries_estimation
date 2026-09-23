import sys
import numpy as np
import scipy.linalg as la
import pytest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trace_baseline import MatVecOracle, _rank_aware_qr

@pytest.mark.parametrize("matrix_rank", [None, 2])
def test_probe_identical_sequential_vs_standard(matrix_rank):
    """
    Deterministic Regression Test:
    When Standard Hutch++ and Sequential forced-q_0 pilot use the EXACT SAME
    sketch matrix S in R^{d x q_0} (presented in block-by-block chunks for sequential)
    and EXACT SAME residual probes G in R^{d x ell}, the computed trace estimates
    MUST match up to floating point precision.
    """
    print("--- Test: Probe-Identical Standard vs Sequential Forced-q_0 Equivalence ---")
    d = 200
    m = 160
    q_0 = 53
    b_0 = 8
    delta_b = 4

    rng = np.random.default_rng(42)

    # 1. Generate either a full-rank or rank-deficient PSD matrix.
    Q_orth, _ = la.qr(rng.normal(size=(d, d)))
    if matrix_rank is None:
        eigenvals = (np.arange(1, d + 1, dtype=np.float64)) ** (-1.5)
    else:
        eigenvals = np.zeros(d, dtype=np.float64)
        eigenvals[:matrix_rank] = np.arange(matrix_rank, 0, -1, dtype=np.float64)
    A = (Q_orth * eigenvals) @ Q_orth.T

    # 2. Draw Shared Sketch Matrix S (d x q_0) and Shared Residual Probes G (d x ell)
    S_shared = rng.choice([-1.0, 1.0], size=(d, q_0))
    
    # Compute basis for Standard Hutch++
    oracle_std = MatVecOracle(A, d=d)
    W_std = oracle_std(S_shared)
    scale_std = float(la.norm(W_std, ord='fro'))
    Q_std, r_std = _rank_aware_qr(W_std, reference_scale=scale_std)
    Z_std = oracle_std(Q_std)
    
    ell_std = m - q_0 - r_std
    G_shared = rng.choice([-1.0, 1.0], size=(d, ell_std))

    # Standard Hutch++ Estimate using shared G
    RG_std = G_shared - Q_std @ (Q_std.T @ G_shared)
    ARG_std = oracle_std(RG_std)
    tr_low_std = float(np.sum(Q_std * Z_std))
    tr_res_std = float(np.sum(RG_std * ARG_std)) / ell_std
    tr_est_std = tr_low_std + tr_res_std

    # 3. Sequential Pilot using identical block presentation of S_shared and shared G
    # Block decomposition of S_shared: S_{1:8}, S_{9:12}, ..., S_{49:53}
    oracle_seq = MatVecOracle(A, d=d)

    # Re-construct sequential pilot using exact blocks of S_shared
    S_pilot_0 = S_shared[:, :b_0]
    W_pilot_0 = oracle_seq(S_pilot_0)
    scale_0 = float(la.norm(W_pilot_0, ord='fro'))
    Q_seq, r_seq_0 = _rank_aware_qr(W_pilot_0, reference_scale=scale_0)
    Z_seq = oracle_seq(Q_seq)

    curr_col = b_0
    while curr_col < q_0:
        next_col = min(curr_col + delta_b, q_0)
        S_chunk = S_shared[:, curr_col:next_col]
        W_chunk = oracle_seq(S_chunk)
        scale_chunk = float(la.norm(W_chunk, ord='fro'))

        W_tilde = W_chunk - Q_seq @ (Q_seq.T @ W_chunk)
        W_tilde = W_tilde - Q_seq @ (Q_seq.T @ W_tilde)

        Q_delta, r_delta = _rank_aware_qr(W_tilde, reference_scale=scale_chunk)
        if r_delta > 0:
            Q_delta = Q_delta - Q_seq @ (Q_seq.T @ Q_delta)
            Q_delta, r_delta = _rank_aware_qr(Q_delta, reference_scale=1.0)
            Z_delta = oracle_seq(Q_delta)
            Q_seq = np.column_stack([Q_seq, Q_delta])
            Z_seq = np.column_stack([Z_seq, Z_delta])

        curr_col = next_col

    r_seq = Q_seq.shape[1]
    ell_seq = m - q_0 - r_seq

    # Residual estimation with identical G_shared
    RG_seq = G_shared - Q_seq @ (Q_seq.T @ G_shared)
    ARG_seq = oracle_seq(RG_seq)

    tr_low_seq = float(np.sum(Q_seq * Z_seq))
    tr_res_seq = float(np.sum(RG_seq * ARG_seq)) / ell_seq
    tr_est_seq = tr_low_seq + tr_res_seq

    # 4. Compare Orthogonal Projectors and Estimates
    P_std = Q_std @ Q_std.T
    P_seq = Q_seq @ Q_seq.T
    proj_diff = float(la.norm(P_std - P_seq, ord='fro'))

    print(f"Standard Hutch++ Trace Estimate:   {tr_est_std:.10f}")
    print(f"Sequential Forced-q_0 Trace Est:   {tr_est_seq:.10f}")
    print(f"Absolute Difference in Estimate:  {abs(tr_est_std - tr_est_seq):.2e}")
    print(f"Frobenius Norm Projector Diff:    {proj_diff:.2e}")

    assert np.allclose(P_std, P_seq, atol=1e-12), "Projectors P_std and P_seq do not match!"
    assert np.isclose(tr_est_std, tr_est_seq, atol=1e-12), "Trace estimates do not match under identical probes!"
    assert r_std == r_seq
    assert ell_std == ell_seq
    assert oracle_std.query_count == m, "Standard oracle query count mismatch!"
    assert oracle_seq.query_count == m, "Sequential oracle query count mismatch!"
    print("--> PASS: Probe-identical Standard vs Sequential forced-q_0 equivalence verified perfectly!")

if __name__ == "__main__":
    test_probe_identical_sequential_vs_standard()
