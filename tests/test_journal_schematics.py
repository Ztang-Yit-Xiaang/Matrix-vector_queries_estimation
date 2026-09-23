"""Editorial regressions for the journal figures and source revision."""
from pathlib import Path
import importlib.util
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('schematics', ROOT / 'experiments/build_journal_schematics.py')
schematics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schematics)


def test_schematic_labels_fit_canvas():
    for factory in [schematics.workflow, schematics.risk_models]:
        fig = factory()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            for label in ax.texts:
                bounds = label.get_window_extent(renderer)
                assert bounds.x0 >= 0 and bounds.y0 >= 0, label.get_text()
                assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height, label.get_text()
        schematics.plt.close(fig)


def test_trace_split_does_not_assert_matrix_split():
    a = np.array([[2., 1.], [1., 3.]])
    q = np.array([[1.], [0.]])
    p = q @ q.T
    r = np.eye(2) - p
    assert np.isclose(np.trace(a), np.trace(q.T @ a @ q) + np.trace(r @ a @ r))
    assert not np.allclose(a, p @ a @ p + r @ a @ r)


def test_ideal_tail_not_rademacher_energy():
    a = np.diag([2., 1.])
    r = np.diag([0., 1.])
    h = r @ a @ r
    assert np.sum(h*h) == 1
    assert np.sum((h - np.diag(np.diag(h)))**2) == 0


def test_compiled_source_preserves_results_without_phase_codes():
    package = ROOT / 'reports/latex/adaptive_hutchpp_20260919'
    main = (package / 'main.tex').read_text()
    included = re.findall(r'\\input\{([^}]+)\}', main)
    text = main + '\n'.join((package / (p + '.tex')).read_text() for p in included)
    assert not re.search(r'\bPhase\s+[12][A-Z]', text)
    assert text.count('\\begin{proof}') == 17
    assert len(re.findall(r'\\begin\{(?:theorem|lemma|proposition|corollary)\}', text)) == 17
    assert len(re.findall(r'\\label\{exp:E[1-6]\}', text)) == 6
    assert 'Statement A' not in text
    assert 'ideal Gaussian/Frobenius risk model' in text


def test_square_matrix_masks_are_symmetric():
    for mode in ['all', 'offdiag']:
        fig, ax = schematics.canvas(3)
        schematics.matrix(ax, 1, 1, 8, 8, 4, 4, mode=mode)
        colors = np.array([p.get_facecolor() for p in ax.patches[:-1]]).reshape(4, 4, 4)
        np.testing.assert_allclose(colors, colors.transpose(1, 0, 2))
        schematics.plt.close(fig)
