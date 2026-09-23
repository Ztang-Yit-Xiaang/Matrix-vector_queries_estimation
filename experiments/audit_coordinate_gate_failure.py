"""Read-only replay of the recovered rank-five step experiment; no allocator edits.

Replays the FIRST spectrum in the original shared RNG stream. All four original
methods run in their original order. Only Standard and gated bases are diagnosed.
Outputs must go to a new directory. Existing results and source are hash guarded.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_haar_orientation_audit import sample_haar_orthogonal
from src.trace_baseline import (MatVecOracle, Hutchinson, Hutch_pplus,
    Gaussian_Hutch_pplus, Adaptive_Hutch_pplus_TwoStageGated)


class RecordingOracle(MatVecOracle):
    """Observe unchanged estimator calls, retaining the actual S, Q, AQ, and RG."""
    def __init__(self, A):
        super().__init__(A)
        self.calls = []

    def __call__(self, V):
        out = super().__call__(V)
        self.calls.append((np.array(V, copy=True), np.array(out, copy=True)))
        return out


def integer_nullspace(A):
    """Exact rational nullspace for a small INTEGER matrix, no SVD tolerance."""
    A = np.asarray(A)
    if A.ndim != 2 or not np.all(np.isfinite(A)) or not np.all(A == np.round(A)):
        raise ValueError('Expected a finite integer matrix.')
    rows = [[Fraction(int(x)) for x in row] for row in A]
    pivots = []
    lead = 0
    for col in range(A.shape[1]):
        pivot = next((i for i in range(lead, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[lead], rows[pivot] = rows[pivot], rows[lead]
        scale = rows[lead][col]
        rows[lead] = [x / scale for x in rows[lead]]
        for i in range(len(rows)):
            if i != lead:
                scale = rows[i][col]
                rows[i] = [x - scale*y for x, y in zip(rows[i], rows[lead])]
        pivots.append(col)
        lead += 1
        if lead == len(rows):
            break
    vectors = []
    for free in sorted(set(range(A.shape[1])) - set(pivots)):
        v = [Fraction(0)] * A.shape[1]
        v[free] = Fraction(1)
        for row, pivot in enumerate(pivots):
            v[pivot] = -rows[row][free]
        vectors.append(v)
    return len(pivots), vectors


def step_energies(Q, U_signal, eta, block_size=8):
    """Nonnegative block sums of H=eta R+(1-eta)ZZ'; never form dense H or R.

    Uses known synthetic structure, not additional oracle products. Off-diagonal
    summation avoids subtracting nearly equal Frobenius/diagonal energies.
    """
    d, r = Q.shape
    Z = U_signal - Q @ (Q.T @ U_signal)
    e_g = e_r = 0.0
    for start in range(0, d, block_size):
        ids = np.arange(start, min(start + block_size, d))
        Rcols = -(Q @ Q[ids].T)
        Rcols[ids, np.arange(len(ids))] += 1
        Hcols = eta * Rcols + (1 - eta) * Z @ Z[ids].T
        e_g += float(np.sum(Hcols**2))
        Hcols[ids, np.arange(len(ids))] = 0
        e_r += float(np.sum(Hcols**2))
    gram = np.einsum('ij,ik->jk', Z, Z)
    formula = (eta**2 * (d-r) + 2*eta*(1-eta)*np.sum(Z**2)
               + (1-eta)**2*np.sum(gram**2))
    np.testing.assert_allclose(e_g, formula, rtol=1e-9, atol=1e-15)
    return e_g, e_r, Z


def direct_energies(oracle, Q, block_size=8):
    """Independent block oracle check: exactly d diagnostic queries per basis."""
    d = Q.shape[0]
    e_g = e_r = 0.0
    for start in range(0, d, block_size):
        ids = np.arange(start, min(start + block_size, d))
        Rcols = -(Q @ Q[ids].T)
        Rcols[ids, np.arange(len(ids))] += 1
        AR = oracle(Rcols)
        Hcols = AR - Q @ (Q.T @ AR)
        e_g += float(np.sum(Hcols**2))
        Hcols[ids, np.arange(len(ids))] = 0
        e_r += float(np.sum(Hcols**2))
    return e_g, e_r


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def replay(reference, n_orientations=30):
    """Original seed 2026, d=100, m=60, ten trials per orientation; no new trials."""
    if not isinstance(n_orientations, int) or not 0 <= n_orientations <= 30:
        raise ValueError('Replay only 0..30 original Haar orientations.')
    d, m, k, eta = 100, 60, 5, 0.001
    rng = np.random.default_rng(2026)
    evals = np.r_[np.ones(k), np.full(d-k, eta)]
    trace = float(evals.sum())
    reference = reference[reference.case_name == 'Step (r*=5, eta=1e-3)']
    assert not reference.duplicated(['orient_idx', 'trial']).any()
    reference = reference.set_index(['orient_idx', 'trial'])
    records, witnesses, states = [], [], []
    replay_queries = validation_queries = 0
    max_error_difference = 0.0
    for orientation in range(n_orientations+1):
        U = np.eye(d) if orientation == 0 else sample_haar_orthogonal(d, rng)
        # Reproduce the original input matrix, including its floating arithmetic.
        A = U @ np.diag(evals) @ U.T
        for trial in range(10):
            ref = reference.loc[(orientation, trial)]
            for name, method in [('hutch', Hutchinson), ('hpp', Hutch_pplus),
                                 ('ghpp', Gaussian_Hutch_pplus),
                                 ('gated', Adaptive_Hutch_pplus_TwoStageGated)]:
                state = rng.bit_generator.state
                oracle = RecordingOracle(A)
                diag = {}
                if name == 'gated':
                    estimate, diag = method(oracle, m=m, d=d, b_0=8, tau_gap=1.5,
                        p_oversample=2, rng=rng, return_diagnostics=True)
                    assert diag['q_target'] == int(ref.q_target)
                    assert diag['gap_location'] == int(ref.r_knee_detected)
                    assert diag['is_gated_trigger'] == bool(ref.gate_triggered)
                else:
                    estimate = method(oracle, m=m, d=d, rng=rng)
                assert oracle.query_count == m
                replay_queries += oracle.query_count
                squared_error = (estimate - trace)**2
                expected = float(ref['sq_err_' + name])
                np.testing.assert_allclose(squared_error, expected, rtol=2e-7, atol=1e-17)
                max_error_difference = max(max_error_difference, abs(squared_error-expected))
                if name not in ('hpp', 'gated'):
                    continue
                # This frozen first-spectrum run uses no gated extension. Fail
                # explicitly rather than silently interpreting a changed call layout.
                assert len(oracle.calls) == 3
                S, Y = oracle.calls[0]
                Q, AQ = oracle.calls[1]
                q, r = S.shape[1], Q.shape[1]
                ell = m-q-r
                assert oracle.calls[-1][0].shape[1] == ell and ell > 0
                if name == 'gated':
                    assert (q, r, ell) == (diag['q_target'], diag['r_actual'], diag['ell_eff'])
                S1 = U[:, :k].T @ S
                singular = np.linalg.svd(S1, compute_uv=False)
                signal_rank = int(np.sum(singular > 1e-12*np.linalg.norm(S1)))
                e_g, e_r, Z = step_energies(Q, U[:, :k], eta)
                proj_error = float(np.linalg.norm(Z, 2))
                orth_error = float(np.linalg.norm(Q.T@Q-np.eye(r), 2))
                assert orth_error < 1e-12
                exact_rank = None
                if orientation == 0:
                    exact_rank, null = integer_nullspace(S1.T)
                    assert exact_rank == signal_rank
                    checker = MatVecOracle(A)
                    actual = direct_energies(checker, Q)
                    validation_queries += checker.query_count
                    assert checker.query_count == d
                    np.testing.assert_allclose(actual, (e_g, e_r), rtol=1e-9, atol=1e-15)
                    for v in null:
                        raw = np.array([float(x) for x in v])
                        w = U[:, :k] @ (raw / np.linalg.norm(raw))
                        assert all(sum(Fraction(int(S1[i,j])) * v[i] for i in range(k)) == 0
                                   for j in range(q))
                        witnesses.append(dict(orientation=orientation, trial=trial, method=name,
                            signal_sketch=S1.astype(int).tolist(), rational_null_vector=list(map(str, v)),
                            basis_overlap=float(np.linalg.norm(Q.T@w)),
                            eigendirection_error=float(np.linalg.norm(evals*w-w))))
                ritz = np.linalg.eigvalsh((Q.T@AQ+AQ.T@Q)/2)[::-1]
                records.append(dict(orientation=orientation, trial=trial, method=name,
                    q=q, r=r, ell=ell, query_count=oracle.query_count,
                    signal_rank=signal_rank, exact_signal_rank=exact_rank,
                    signal_sigma_min=float(singular[-1]), signal_sigma_max=float(singular[0]),
                    sketch_sigma_min=float(np.linalg.svd(Y, compute_uv=False)[-1]),
                    projection_error_op=proj_error, projection_error_fro=float(np.linalg.norm(Z)),
                    largest_angle_degrees=float(np.degrees(np.arcsin(min(1.,proj_error)))),
                    orthogonality_error=orth_error, energy_gaussian=e_g, energy_rademacher=e_r,
                    risk_gaussian=2*e_g/ell, risk_rademacher=2*e_r/ell,
                    estimate=estimate, squared_error=squared_error,
                    squared_error_over_risk=squared_error/(2*e_r/ell),
                    detected_knee=diag.get('gap_location'), gap=diag.get('max_adjacent_log_gap'),
                    contrast=diag.get('contrast'), ritz_values=json.dumps(ritz.tolist())))
                states.append(dict(orientation=orientation, trial=trial, method=name, rng_state=state))
    return pd.DataFrame(records), witnesses, states, dict(
        replay_queries=replay_queries, diagnostic_validation_queries=validation_queries,
        total_queries=replay_queries+validation_queries,
        max_squared_error_reproduction_difference=max_error_difference)


def run(output_dir, n_orientations=30):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('Use a new output directory; existing artifacts are immutable.')
    protected = list((ROOT/'results').rglob('*')) + [ROOT/'src/trace_baseline.py',
                 ROOT/'experiments/run_haar_orientation_audit.py']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected if p.is_file()}
    reference = pd.read_csv(ROOT/'results/haar_orientation_audit_trials.csv')
    df, witnesses, states, counts = replay(reference, n_orientations)
    assert len(df) == 20*(n_orientations+1)
    assert not df.duplicated(['orientation','trial','method']).any()
    assert np.isfinite(df[['energy_gaussian','energy_rademacher','risk_gaussian',
                          'risk_rademacher','squared_error']].to_numpy()).all()
    assert all(sha(ROOT/p) == digest for p,digest in hashes.items())
    output_dir.mkdir(parents=True)
    df.to_csv(output_dir/'paths.csv', index=False)
    (output_dir/'exact_nullspace_witnesses.json').write_text(json.dumps(witnesses, indent=2)+'\n')
    (output_dir/'rng_states.json').write_text(json.dumps(states, indent=2)+'\n')
    summary = []
    for (coordinate, method), group in df.groupby([df.orientation.eq(0), 'method']):
        se, risk = group.squared_error, group.risk_rademacher
        summary.append(dict(orientation_group='coordinate' if coordinate else 'haar', method=method,
            paths=len(group), mse=float(se.mean()), median_squared_error=float(se.median()),
            mean_conditional_risk=float(risk.mean()), median_conditional_risk=float(risk.median()),
            maximum_conditional_risk=float(risk.max()), worst_error_fraction=float(se.max()/se.sum()),
            worst_risk_fraction=float(risk.max()/risk.sum()), signal_deficient_paths=int((group.signal_rank<5).sum())))
    pd.DataFrame(summary).to_csv(output_dir/'summary.csv', index=False)
    manifest = dict(protocol='Recovered first step spectrum, seed 2026, unchanged four-method order',
        dimension=100, budget=60, r_star=5, eta=.001, haar_orientations=n_orientations,
        trials_per_orientation=10, rank_rtol=1e-12, rank_atol=0.,
        replay_is_not_new_independent_evidence=True, methods_have_different_not_paired_probes=True,
        estimator_modified=False, historical_files_preserved=len(hashes), protected_sha256=hashes,
        diagnostic_rows=len(df), witness_count=len(witnesses), **counts,
        python=sys.version, numpy=np.__version__, pandas=pd.__version__,
        output_sha256={p.name:sha(p) for p in output_dir.iterdir() if p.is_file()})
    (output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(summary=summary, counts=counts, witnesses=witnesses), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--haar-orientations', type=int, default=30)
    args = parser.parse_args()
    run(args.output_dir, args.haar_orientations)
