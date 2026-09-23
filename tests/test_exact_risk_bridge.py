import sys
from pathlib import Path

import numpy as np
import scipy.linalg as la


EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from run_exact_risk_bridge import (  # noqa: E402
    compute_nested_conditional_risks,
    run_exact_bridge,
)


def test_fast_nested_formulas_match_explicit_residual_matrix():
    rng = np.random.default_rng(17)
    dimension = 8
    budget = 12
    eigenvalues = np.linspace(2.0, 0.2, dimension)
    orientation, _ = la.qr(rng.normal(size=(dimension, dimension)))
    matrix = (orientation * eigenvalues) @ orientation.T
    sketch = rng.choice([-1.0, 1.0], size=(dimension, 3))
    curves = compute_nested_conditional_risks(matrix, eigenvalues, budget, sketch)
    full_basis, _ = la.qr(matrix @ sketch, mode="economic")

    for q in range(4):
        basis = full_basis[:, :q]
        projector = basis @ basis.T
        residual = (np.eye(dimension) - projector) @ matrix @ (np.eye(dimension) - projector)
        ell = budget - 2 * q
        frobenius_sq = float(np.sum(residual**2))
        diagonal_sq = float(np.sum(np.diag(residual) ** 2))
        row = curves.iloc[q]
        assert np.isclose(row["residual_frobenius_sq"], frobenius_sq, rtol=1e-11, atol=1e-12)
        assert np.isclose(row["residual_diagonal_sq"], diagonal_sq, rtol=1e-11, atol=1e-12)
        assert np.isclose(row["gaussian_risk"], 2.0 * frobenius_sq / ell)
        assert np.isclose(row["rademacher_risk"], 2.0 * (frobenius_sq - diagonal_sq) / ell)


def test_rademacher_conditional_risk_is_zero_for_diagonal_residual():
    eigenvalues = np.array([4.0, 3.0, 2.0, 1.0])
    matrix = np.diag(eigenvalues)
    sketch = np.eye(4)[:, :2]
    curves = compute_nested_conditional_risks(matrix, eigenvalues, budget=10, sketch_matrix=sketch)
    assert np.allclose(curves["rademacher_risk"], 0.0, atol=1e-14)


def test_gaussian_formula_matches_residual_probe_monte_carlo():
    rng = np.random.default_rng(29)
    dimension = 6
    budget = 12
    eigenvalues = np.linspace(1.5, 0.25, dimension)
    orientation, _ = la.qr(rng.normal(size=(dimension, dimension)))
    matrix = (orientation * eigenvalues) @ orientation.T
    sketch = rng.normal(size=(dimension, 2))
    curves = compute_nested_conditional_risks(matrix, eigenvalues, budget, sketch)
    basis, _ = la.qr(matrix @ sketch, mode="economic")
    residual = (np.eye(dimension) - basis @ basis.T) @ matrix @ (np.eye(dimension) - basis @ basis.T)
    ell = budget - 4

    probes = rng.normal(size=(30_000, ell, dimension))
    quadratic = np.einsum("nld,de,nle->nl", probes, residual, probes)
    estimates = quadratic.mean(axis=1)
    empirical_variance = float(np.var(estimates, ddof=1))
    exact_variance = float(curves.loc[curves["q"] == 2, "gaussian_risk"].iloc[0])
    assert np.isclose(empirical_variance, exact_variance, rtol=0.06)


def test_exact_bridge_smoke_writes_valid_frozen_artifacts(tmp_path):
    manifest, trials, curves, minimizers = run_exact_bridge(
        trials_per_setup=2,
        dimension=20,
        budget=16,
        bootstrap_samples=100,
        q_max=4,
        output_dir=tmp_path,
    )
    assert len(manifest) == 24
    assert len(trials) == 24 * 2 * 5
    assert len(curves) == 24 * 5
    assert len(minimizers) == 24 * 3
    assert not trials.duplicated(["setup_name", "basis_trial", "q"]).any()
    assert np.all(trials["q"] + trials["r_actual"] + trials["ell"] == 16)
    for name in ("manifest", "trials", "curves", "minimizers"):
        assert (tmp_path / f"risk_bridge_exact_{name}.csv").exists()
