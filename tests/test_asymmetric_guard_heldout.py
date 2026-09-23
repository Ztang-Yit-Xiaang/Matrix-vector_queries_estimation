import sys
from pathlib import Path

import numpy as np


EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from run_asymmetric_guard_heldout_benchmark import (  # noqa: E402
    build_eigenvalues,
    build_setup_manifest,
)


def test_frozen_manifest_has_exactly_24_unique_configurations():
    manifest = build_setup_manifest()

    assert len(manifest) == 24
    assert manifest["setup_name"].nunique() == 24
    assert manifest["setup_index"].tolist() == list(range(24))
    assert manifest["matrix_seed"].tolist() == list(range(42_000, 42_024))
    assert (manifest["spectrum_family"] == "power").sum() == 5
    assert (manifest["spectrum_family"] == "exponential").sum() == 4
    assert (manifest["spectrum_family"] == "step").sum() == 12
    assert (manifest["spectrum_family"] == "misspecified").sum() == 3


def test_all_frozen_spectra_are_positive_finite_and_nonincreasing():
    dimension = 100
    manifest = build_setup_manifest()

    for setup in manifest.to_dict("records"):
        eigenvalues = build_eigenvalues(setup, dimension)
        assert eigenvalues.shape == (dimension,)
        assert np.all(np.isfinite(eigenvalues))
        assert np.all(eigenvalues > 0.0)
        assert np.all(np.diff(eigenvalues) <= 1e-14)


def test_frozen_spectral_parameterization():
    manifest = build_setup_manifest()

    power = manifest[manifest["spectrum_family"] == "power"]
    exponential = manifest[manifest["spectrum_family"] == "exponential"]
    steps = manifest[manifest["spectrum_family"] == "step"]

    assert power["c"].tolist() == [0.3, 0.7, 1.2, 1.7, 2.5]
    assert exponential["alpha"].tolist() == [0.02, 0.08, 0.10, 0.15]
    assert set(zip(steps["r_star"], steps["eta"])) == {
        (r_star, eta)
        for r_star in (5, 15, 25, 30)
        for eta in (0.001, 0.05, 0.1)
    }

    smooth = manifest[manifest["variant"] == "smooth_elbow"].iloc[0]
    mixture = manifest[manifest["variant"] == "mixture"].iloc[0]
    lognormal = manifest[manifest["variant"] == "lognormal"].iloc[0]
    assert np.isclose(build_eigenvalues(smooth, 100)[0], 1.0)
    assert np.isclose(build_eigenvalues(mixture, 100)[0], 1.0)
    assert np.isclose(build_eigenvalues(lognormal, 100)[0], 1.0)
