"""Original manuscript schematics; no empirical data or oracle calls.

All shapes are explanatory, not measured geometry. SVG text stays editable.
Use the project's Python plotting environment. Existing figures are untouched.
"""
from pathlib import Path
import argparse
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'urop-journal-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch

INK = '#26343D'
BLUE = '#326A8D'
TEAL = '#438D88'
AMBER = '#BC8136'
GREY = '#77858C'
LIGHT = '#E7ECEF'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                     'mathtext.fontset': 'dejavusans', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'lines.linewidth': .8})


def canvas(height):
    fig = plt.figure(figsize=(7.2, height), facecolor='white')
    ax = fig.add_axes((.015, .015, .97, .97))
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.set_axis_off()
    return fig, ax


def text(ax, x, y, value, size=8.5, color=INK, ha='center', **kw):
    return ax.text(x, y, value, fontsize=size, color=color, ha=ha,
                   va='center', **kw)


def arrow(ax, start, end, color=INK, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                                mutation_scale=9, linewidth=.85,
                                linestyle='--' if dashed else '-', color=color))


def panel(ax, letter, title, y):
    text(ax, 1, y, letter, 10, ha='left', weight='bold')
    text(ax, 5, y, title, 9.5, ha='left', weight='bold')


def matrix(ax, x, y, w, h, rows, cols, color=BLUE, mode='all'):
    for i in range(rows):
        for j in range(cols):
            fill = color
            alpha = .12 + .5 * ((i + j) % 4) / 3
            if mode == 'offdiag' and i == j:
                fill, alpha = 'white', 1
            if mode == 'columns':
                alpha = .18 + .13 * (j % 3)
            ax.add_patch(Rectangle((x+j*w/cols, y+(rows-1-i)*h/rows),
                                   w/cols, h/rows, facecolor=fill,
                                   edgecolor='white', linewidth=.5, alpha=alpha))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=.85))


def ledger(ax, x, y, widths, labels, colors):
    for width, label, color in zip(widths, labels, colors):
        ax.add_patch(Rectangle((x, y), width, 4.2, facecolor=color,
                               edgecolor='white', linewidth=1, alpha=.2))
        ax.add_patch(Rectangle((x, y), width, 4.2, fill=False,
                               edgecolor=color, linewidth=.65))
        text(ax, x+width/2, y+2.1, label, 8, color)
        x += width


def workflow():
    fig, ax = canvas(5.8)
    panel(ax, 'a', 'Build a small basis from random directions', 97)
    matrix(ax, 5, 75, 13, 16, 6, 6)
    text(ax, 11.5, 71.5, r'$A\;(d\times d)$')
    text(ax, 21, 83, r'$\times$', 12)
    matrix(ax, 25, 75, 7, 16, 6, 3, TEAL, 'columns')
    text(ax, 28.5, 71.5, r'$S\;(d\times q)$')
    text(ax, 36.5, 83, r'$=$', 12)
    matrix(ax, 41, 75, 7, 16, 6, 3, BLUE, 'columns')
    text(ax, 44.5, 71.5, r'$Y=AS$')
    arrow(ax, (50.5, 83), (66, 83))
    text(ax, 58.5, 87.5, 'orthogonalize', 7.5)
    text(ax, 58.5, 78.5, 'keep rank r', 7.5)
    matrix(ax, 70, 75, 5, 16, 6, 2, TEAL, 'columns')
    text(ax, 72.5, 71.5, r'$Q\;(d\times r)$')
    arrow(ax, (78, 83), (86, 83))
    text(ax, 82, 88, r'$A\times$', 8)
    matrix(ax, 89, 75, 5, 16, 6, 2, BLUE, 'columns')
    text(ax, 91.5, 71.5, r'$AQ$')
    text(ax, 28, 67, r'$q$ products to sample the matrix', 8, BLUE)
    text(ax, 82, 67, r'$r$ products to evaluate the basis', 8, TEAL)
    ax.plot([1, 99], [63, 63], color=LIGHT, lw=.8)

    panel(ax, 'b', 'Measure the captured part; sample what remains', 59.5)
    # Euclidean 2-D schematic: h is perpendicular to the drawn Q axis.
    ax.add_patch(Polygon([(5, 39), (32, 39), (32, 43), (5, 43)],
                         facecolor=TEAL, alpha=.14, edgecolor='none'))
    ax.plot([5, 32], [41, 41], color=TEAL, lw=1)
    arrow(ax, (10, 41), (28, 41), TEAL)
    arrow(ax, (10, 41), (28, 52), INK)
    arrow(ax, (10, 41), (10, 52), AMBER)
    ax.plot([10, 28, 28], [52, 52, 41], color=GREY, ls=':', lw=.8)
    ax.plot([10, 11.7, 11.7], [42.7, 42.7, 41], color=GREY, lw=.7)
    text(ax, 29.5, 52, r'$g$', ha='left')
    text(ax, 7.5, 49, r'$h$', color=AMBER)
    text(ax, 22, 36.5, r'captured subspace $\mathrm{span}(Q)$', 7.5, TEAL)
    text(ax, 17, 32.5, r'$h=(I-QQ^\top)g$', 8)

    matrix(ax, 41, 43, 8, 8, 2, 2, TEAL)
    text(ax, 45, 54, 'Evaluate exactly', 8, TEAL, weight='bold')
    text(ax, 45, 39.5, r'$\mathrm{tr}(Q^\top AQ)$', 9)
    text(ax, 57, 46, '+', 14)
    text(ax, 63, 46, r'$h$', 8, AMBER)
    arrow(ax, (65, 46), (70, 46), AMBER)
    matrix(ax, 72, 42, 8, 10, 4, 4, GREY)
    text(ax, 76, 47, r'$A$', 10)
    arrow(ax, (82, 46), (87, 46), AMBER)
    text(ax, 91, 46, r'$h^\top Ah$', 8)
    text(ax, 79, 55, r'Average $\ell$ fresh probe values', 8, GREY, weight='bold')
    text(ax, 79, 37.5, r'estimates $\mathrm{tr}(H_Q)$', 8)
    text(ax, 69, 32.5, r'$\mathrm{tr}(A)=\mathrm{tr}(Q^\top AQ)+\mathrm{tr}(H_Q)$', 9)
    ax.plot([1, 99], [28.5, 28.5], color=LIGHT, lw=.8)

    panel(ax, 'c', 'Every query has a cost', 25)
    text(ax, 2, 19, 'Basic estimator', 8, ha='left')
    text(ax, 2, 15.5, r'$q+r+\ell=m$', 7.5, ha='left')
    ledger(ax, 31, 16.9, [18, 14, 36], [r'$q$: sketch', r'$r$: basis', r'$\ell$: fresh final probes'], [BLUE, TEAL, GREY])
    text(ax, 2, 12, 'With certification', 8, ha='left')
    text(ax, 2, 8.3, r'$c_{\rm pre}+s+\ell_{\rm paid}=m$', 7.5, ha='left')
    ledger(ax, 31, 9.9, [32, 13, 23], [r'$c_{\rm pre}$: construction', r'$s$: check', r'$\ell_{\rm paid}$: final'], [BLUE, AMBER, GREY])
    text(ax, 65, 6.5, r'Same total budget $m$; segments are schematic.', 7.5, GREY)
    text(ax, 50, 1.8, 'Certification uses separate probes. Abstaining does not return spent queries.', 8)
    return fig


def model_box(ax, x, y, w, title, formula, note, color):
    ax.add_patch(Rectangle((x, y), w, 17, fill=False, edgecolor=color, linewidth=.8))
    text(ax, x+w/2, y+13.5, title, 8, color, weight='bold')
    text(ax, x+w/2, y+7.8, formula, 11)
    text(ax, x+w/2, y+2.5, note, 7)


def risk_models():
    fig, ax = canvas(5.5)
    text(ax, 50, 96, r'One matrix $A$, budget $m$, and attempted sketch width $q$', 10, weight='bold')
    text(ax, 50, 91, 'Which information do we use to predict the error?', 9, GREY)
    arrow(ax, (46, 87.5), (25, 80))
    arrow(ax, (54, 87.5), (75, 80))
    text(ax, 25, 76.5, 'Ideal spectral capture', 9.5, BLUE, weight='bold')
    text(ax, 75, 76.5, 'The computed basis', 9.5, TEAL, weight='bold')
    ax.plot([50, 50], [4, 79], color=LIGHT, lw=1)
    # Deliberately schematic step spectrum, without quantitative tick marks.
    for j in range(10):
        height = 11 if j < 4 else 3
        ax.add_patch(Rectangle((9+j*3.25, 55), 2.1, height,
                               facecolor=BLUE if j < 4 else GREY, alpha=.8 if j < 4 else .5))
    ax.plot([7, 43], [54.6, 54.6], color=GREY, lw=.7)
    ax.plot([21, 21], [54, 67], color=GREY, ls=':', lw=.7)
    text(ax, 21, 52.3, r'$j$', 8, GREY)
    text(ax, 43, 67, r'$\lambda_i^2$', 8, GREY)
    text(ax, 14, 69, 'leading directions', 7.5, BLUE)
    text(ax, 35, 61, 'tail', 8, GREY)
    text(ax, 25, 48, r'$T(j)=\sum_{i>j}\lambda_i^2$', 10)
    text(ax, 25, 44, 'Assume the best eigendirections were captured.', 7.5)
    # Actual residual and variance masks: not an operation on A.
    matrix(ax, 58, 57, 5, 12, 5, 2, TEAL, 'columns')
    text(ax, 60.5, 53.5, r'$Q_q$', 8)
    arrow(ax, (65, 63), (70, 63), TEAL)
    matrix(ax, 73, 57, 12, 12, 5, 5, GREY)
    text(ax, 79, 53.5, r'$H_Q=R_QAR_Q$', 9)
    text(ax, 75, 47.5, r'$r_q=\mathrm{rank}(Q_q),\quad \ell=m-q-r_q$', 8.5)
    text(ax, 75, 42.5, 'Measure the residual left by this basis.', 7.5)

    arrow(ax, (25, 40), (13, 34), BLUE)
    arrow(ax, (25, 40), (37, 34), BLUE)
    model_box(ax, 1, 15, 22, 'Full-rank ideal', r'$\frac{2T(q)}{m-2q}$', r'assume $r=q$', BLUE)
    model_box(ax, 26, 15, 22, 'Rank-aware ideal', r'$\frac{2T(r_q)}{m-q-r_q}$', 'use accepted rank', BLUE)
    text(ax, 25, 8, 'Spectral surrogates', 9, BLUE)

    arrow(ax, (75, 40.5), (62.5, 37.5), TEAL)
    arrow(ax, (75, 40.5), (87.5, 37.5), TEAL)
    text(ax, 62.5, 35.5, 'all entries', 7, TEAL)
    text(ax, 87.5, 35.5, 'off-diagonal entries', 7, TEAL)
    matrix(ax, 59, 26, 7, 7, 4, 4, TEAL)
    matrix(ax, 84, 26, 7, 7, 4, 4, TEAL, 'offdiag')
    text(ax, 62.5, 22, 'Gaussian probes', 8, weight='bold')
    text(ax, 87.5, 22, 'Rademacher probes', 8, weight='bold')
    text(ax, 62.5, 15, r'$\frac{2\|H_Q\|_F^2}{\ell}$', 11)
    text(ax, 87.5, 15, r'$\frac{2\sum_{i\ne j}(H_Q)_{ij}^2}{\ell}$', 10.5)
    text(ax, 75, 7.5, 'Exact conditional mean-squared errors', 8.5, TEAL)
    text(ax, 50, 1.5, 'Branches distinguish assumptions, not an ordering of the four risks.', 7.5, GREY)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, factory in [('workflow', workflow), ('models', risk_models)]:
        fig = factory()
        for ext in ['svg', 'png']:
            fig.savefig(args.output_dir / f'{name}.{ext}', dpi=450, facecolor='white')
        plt.close(fig)
    (args.output_dir / 'schematic_contract.json').write_text(json.dumps({
        'kind': 'original explanatory schematics; no measurements',
        'backend': 'Python matplotlib', 'oracle_queries': 0,
        'editable_text': 'SVG text, not paths',
        'risk_domains': 'all displayed denominators strictly positive',
        'trace_split': 'trace identity only; no cross-block deletion from A',
        'query_bars': 'schematic widths, not numerical examples',
        'probe_masks': 'variance functionals; diagonal still contributes to trace',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
