"""No-query regressions for the source-backed report figures."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'experiments/build_validated_urop_figures.py'
spec = importlib.util.spec_from_file_location('validated_urop_figures', SCRIPT)
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


@pytest.fixture(scope='module')
def reviewed():
    return figures.load_reviewed_data()


def test_capture_witness_and_reference_selection(reviewed):
    assert reviewed['bad'].trial == 7
    assert reviewed['typical'].trial == 2
    assert np.linalg.matrix_rank(reviewed['S1']) == 4
    assert (reviewed['coords'].query("method == 'gated'").r == 8).all()


def test_truth_conditioning_populations(reviewed):
    for row in reviewed['cat'].itertuples():
        assert json.loads(row.eligible_by_rank) == {'5':10, '15':10, '30':10}
    assert set(reviewed['cert'].estimator) == set(figures.ESTIMATORS)


def test_risk_estimands_are_not_conflated(reviewed):
    paths = reviewed['paths']
    assert int((paths.pathwise_net_ratio > 1).sum()) == 575
    point = reviewed['budget'].query("metric == 'selected_mean_ratio'").point.item()
    assert point < .04
    assert paths.pathwise_net_ratio.mean() > 1
    assert not np.isclose(point, paths.pathwise_net_ratio.mean())


def test_existing_output_is_never_overwritten(tmp_path):
    marker = tmp_path/'keep.txt'
    marker.write_text('original')
    with pytest.raises(FileExistsError):
        figures.build(tmp_path)
    assert marker.read_text() == 'original'


def test_full_export_integrity(tmp_path):
    out = tmp_path/'new'
    figures.build(out)
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['matrix_vector_queries'] == 0
    assert manifest['new_bootstrap_replicates'] == 0
    assert len(list(out.glob('*.png'))) == 3
    assert len(list(out.glob('*.svg'))) == 3
    assert len(list(out.glob('*.csv'))) == 8
    assert not list(out.glob('*.pdf'))
    for name, checksum in manifest['output_sha256'].items():
        assert figures.sha(out/name) == checksum
    for name, checksum in manifest['source_sha256'].items():
        assert figures.sha(figures.ROOT/'results'/name) == checksum
