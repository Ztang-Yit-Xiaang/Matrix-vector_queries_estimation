import itertools
import numpy as np
import pandas as pd
import pytest

from experiments.audit_coordinate_gate_failure import (
    ROOT, RecordingOracle, integer_nullspace, step_energies, direct_energies, replay, run,
)
from src.trace_baseline import MatVecOracle


def test_exact_nullspace_integer_not_tolerance():
    S = np.array([[1, -1, 1], [-1, 1, -1], [1, 1, 1]])
    rank, vectors = integer_nullspace(S.T)
    assert rank == 2 and len(vectors) == 1
    assert np.all(S.T @ np.array(vectors[0], dtype=object) == 0)
    assert integer_nullspace(np.eye(3, dtype=int))[0] == 3
    with pytest.raises(ValueError):
        integer_nullspace([[0.5]])


def test_recording_preserves_count_and_inputs():
    A = np.diag([1., 2., 3.])
    oracle = RecordingOracle(A)
    V = np.ones((3, 2))
    out = oracle(V)
    V[:] = 0
    np.testing.assert_array_equal(oracle.calls[0][0], np.ones((3, 2)))
    np.testing.assert_array_equal(out, [[1,1],[2,2],[3,3]])
    assert oracle.query_count == 2


@pytest.mark.parametrize('eta', [0., .001, 1.])
def test_block_energies_and_query_cost(eta):
    rng = np.random.default_rng(123)
    U = np.linalg.qr(rng.normal(size=(7, 3)))[0]
    Q = np.linalg.qr(rng.normal(size=(7, 4)))[0]
    A = eta*np.eye(7)+(1-eta)*U@U.T
    R = np.eye(7)-Q@Q.T
    H = R@A@R
    np.testing.assert_allclose(H, H.T, atol=1e-14)
    expected_g = np.sum(H**2)
    H = H.copy()
    np.fill_diagonal(H, 0)
    expected_r = np.sum(H**2)
    eg, er, _ = step_energies(Q, U, eta, block_size=2)
    np.testing.assert_allclose([eg,er], [expected_g,expected_r], atol=1e-14)
    oracle = MatVecOracle(A)
    np.testing.assert_allclose(direct_energies(oracle,Q,2), [eg,er], atol=1e-14)
    assert oracle.query_count == 7


def test_missing_mixed_direction_has_nonzero_rademacher_risk():
    # Capturing e1+e2 leaves (e1-e2)/sqrt(2): even diagonal A can yield off-diagonal H.
    Q = np.array([[1.],[1.]])/np.sqrt(2)
    eg, er, _ = step_energies(Q,np.eye(2),0.)
    np.testing.assert_allclose([eg,er],[1.,.5])
    H = np.eye(2)-Q@Q.T
    np.testing.assert_allclose(H,H.T)
    probes = np.array(list(itertools.product([-1.,1.],repeat=2)))
    vals = np.einsum('bi,ij,bj->b',probes,H,probes)
    np.testing.assert_allclose(np.var(vals),2*er)
    # Original diagonal identity has zero Rademacher variance; projecting changed it.
    assert np.var(np.sum(probes**2,axis=1)) == 0


def test_coordinate_replay_reproduces_frozen_errors_and_accounting():
    reference = pd.read_csv(ROOT/'results/haar_orientation_audit_trials.csv')
    df, witnesses, states, counts = replay(reference,0)
    assert len(df) == 20 and len(states) == 20
    assert (df.q+df.r+df.ell == 60).all()
    assert counts['replay_queries'] == 4*10*60
    assert counts['diagnostic_validation_queries'] == 2*10*100
    bad = df[(df.method == 'gated') & (df.trial == 7)].iloc[0]
    assert bad.exact_signal_rank == 4 and bad.r == bad.q == 8
    assert bad.projection_error_op > .999999
    assert witnesses and all(w['basis_overlap'] < 1e-12 for w in witnesses)


def test_refuses_existing_output_before_any_replay(tmp_path):
    with pytest.raises(FileExistsError):
        run(tmp_path)
