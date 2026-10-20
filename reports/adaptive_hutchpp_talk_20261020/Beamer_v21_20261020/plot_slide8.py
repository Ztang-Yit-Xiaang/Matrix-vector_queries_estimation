"""Select the approved lower-median held-out case and replot saved data only."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import math
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
args = parser.parse_args()
root = args.root.resolve()
here = Path(__file__).resolve().parent
fig_dir = here / 'figures'
data_dir = here / 'data'
fig_dir.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(here / '.mpl-cache'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

v1 = root / 'q_finding/results_20261019_v1/main'
curve_file = v1 / 'risk_curves_by_q.csv'
policy_file = v1 / 'allocation_policy_comparison.csv'
summary_file = v1 / 'policy_heldout_summary.csv'
manifest_file = v1 / 'manifest.json'
phase4_file = root / 'results/spectral_floor_mechanism_20261019_v4/figures/source_data/figure2_actual_and_fixed_basis_risk_curves.csv'
manifest = json.loads(manifest_file.read_text())
configurations = {r['id']: r for r in manifest['configurations']}
heldout_ids = {r['id'] for r in manifest['configurations'] if r['split'] == 'test'}
policy_rows = [r for r in csv.DictReader(policy_file.open()) if r['method'] == 'offline_tuned_group_holdout']
assert len(heldout_ids) == 186 and len(policy_rows) == 744
assert {r['id'] for r in policy_rows} == heldout_ids
assert {r['split'] for r in policy_rows} == {'test'}
summary_rows = list(csv.DictReader(summary_file.open()))
aggregate = next(r for r in summary_rows if r['method'] == 'offline_tuned_group_holdout' and r['stratum'] == 'ALL')
assert (int(aggregate['matrix_configurations']), int(aggregate['matrix_budget_cases']), int(aggregate['spectral_groups'])) == (186, 744, 16)
assert round(float(aggregate['saving_vs_standard_pct']), 3) == 19.793

keys = {(r['id'], int(r['m'])) for r in policy_rows}
curves = {}
for row in csv.DictReader(curve_file.open()):
    key = row['id'], int(row['m'])
    if key in keys:
        q = int(row['q'])
        assert q not in curves.setdefault(key, {})
        curves[key][q] = row
assert len(curves) == 744

cases = []
for row in policy_rows:
    m = int(row['m'])
    q_rule = math.floor(.24 * m + .5)
    q_std = math.floor(m / 3)
    curve = curves[row['id'], m]
    rule = float(curve[q_rule]['mean_exact_conditional_mse'])
    std = float(curve[q_std]['mean_exact_conditional_mse'])
    assert std > 0 and np.isfinite(rule) and np.isfinite(std)
    assert math.isclose(rule, float(row['mean_mse']), rel_tol=1e-12)
    assert math.isclose(std, float(row['standard_mse']), rel_tol=1e-12)
    cases.append({
        'id': row['id'], 'family': row['family'], 'param': float(row['param']),
        'tail_floor': float(row['tail_floor']), 'd': int(row['d']),
        'orientation': row['orientation'], 'group_id': row['group_id'],
        'sketch_trials': int(row['sketch_trials']), 'm': m,
        'q_rule': q_rule, 'q_standard': q_std, 'mse_rule': rule,
        'mse_standard': std, 'gain': 1 - rule / std,
    })

# The user approved excluding both power-law labels and choosing the lower
# central observation when the 144-case sample has an even size.
excluded_families = {'power', 'signed_power'}
eligible = sorted([r for r in cases if r['m'] == 96 and r['family'] not in excluded_families],
                  key=lambda r: (r['gain'], r['id']))
assert len(eligible) == 144
rank_index = (len(eligible) - 1) // 2
selected = eligible[rank_index]
assert selected['id'] == '11cfff958145affe'
configuration = configurations[selected['id']]
selected_rows = [curves[selected['id'], 96][q] for q in sorted(curves[selected['id'], 96])]
assert len(selected_rows) == 48 and [int(r['q']) for r in selected_rows] == list(range(48))
right_q = np.array([int(r['q']) for r in selected_rows])
right_mse = np.array([float(r['mean_exact_conditional_mse']) for r in selected_rows])
right_best = int(np.argmin(right_mse))
assert int(right_q[right_best]) == 23
gain_values = np.array([r['gain'] for r in cases])
assert np.all(np.isfinite(gain_values))
family_totals = collections.Counter(r['family'] for r in cases)
family_losses = collections.Counter(r['family'] for r in cases if r['gain'] < 0)
distribution = {
    'cases': 744, 'matrices': 186, 'budgets': sorted({r['m'] for r in cases}),
    'positive_count': int(np.sum(gain_values > 0)),
    'positive_fraction': float(np.mean(gain_values > 0)),
    'standard_better_count': int(np.sum(gain_values < 0)),
    'standard_better_fraction': float(np.mean(gain_values < 0)),
    'tie_count': int(np.sum(gain_values == 0)),
    'p10': float(np.quantile(gain_values, .1, method='linear')),
    'median': float(np.median(gain_values)),
    'p90': float(np.quantile(gain_values, .9, method='linear')),
    'percentile_method': 'NumPy linear empirical quantiles; unweighted over cases',
    'standard_better_by_family': {
        f: {'count': n, 'family_cases': family_totals[f], 'within_family_fraction': n / family_totals[f]}
        for f, n in sorted(family_losses.items())
    },
}

left_rows = sorted([r for r in csv.DictReader(phase4_file.open()) if float(r['eta']) == .005], key=lambda r: int(r['q']))
assert len(left_rows) == 48
assert all(r['d'] == '512' and r['a'] == '1.8' and r['m'] == '96' and r['sketch_paths'] == '256' for r in left_rows)
left_q = np.array([int(r['q']) for r in left_rows])
left_mse = np.array([float(r['mean_mse']) for r in left_rows])
left_best = int(np.argmin(left_mse))
assert int(left_q[left_best]) == 22

NAVY = '#12263A'
TEAL = '#246B94'
GRAY = '#617080'
LINE = '#DFE7ED'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': NAVY,
    'axes.labelcolor': NAVY, 'axes.edgecolor': LINE,
    'xtick.color': GRAY, 'ytick.color': GRAY,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'mathtext.fontset': 'cm', 'axes.titlepad': 9,
    'savefig.facecolor': 'white',
})
fig, axes = plt.subplots(1, 2, figsize=(6.35, 1.55), sharey=False)
left, right = axes
left.semilogy(left_q, left_mse, color=TEAL, lw=2)
left.axvline(24, color=GRAY, lw=1, ls=(0, (3, 3)))
left.scatter(left_q[left_best], left_mse[left_best], color=TEAL, s=22, zorder=5)
left.annotate(r'$q^*=22$', (22, left_mse[left_best]), xytext=(-24, 9),
              textcoords='offset points', ha='center', fontsize=10, color=TEAL)
left.set(xlim=(0, 48), ylim=(1e-6, .1), xticks=[0, 12, 24, 36, 48],
         yticks=[1e-6, 1e-4, 1e-2], xlabel=r'Sketch queries $q$')
left.set_ylabel('Mean MSE (log scale)')
left.set_title(r'$\eta=0.005$' + '\nMSE increase under cap: 0%', fontsize=10, color=TEAL, fontweight='bold')

right.semilogy(right_q, right_mse, color=TEAL, lw=2)
right.scatter([23], [selected['mse_rule']], color=TEAL, s=25, zorder=5)
right.scatter([32], [selected['mse_standard']], color=GRAY, marker='s', s=25, zorder=5)
for label, q, mse, offset, color in [
    ('24% rule', 23, selected['mse_rule'], (-18, 12), TEAL),
    (r'standard $m/3$', 32, selected['mse_standard'], (7, 28), GRAY),
]:
    right.annotate(label, (q, mse), xytext=offset, textcoords='offset points',
                   ha='center', fontsize=7.5, color=color,
                   arrowprops={'arrowstyle': '-', 'color': color, 'lw': .6, 'shrinkB': 4})
if int(right_q[right_best]) not in {23, 32}:
    right.scatter(right_q[right_best], right_mse[right_best], facecolors='white', edgecolors=NAVY, s=24, zorder=5)
    right.annotate(r'$q^*=' + str(right_q[right_best]) + '$', (right_q[right_best], right_mse[right_best]),
                   xytext=(0, 10), textcoords='offset points', ha='center', color=NAVY, fontsize=8)
right.set(xlim=(0, 48), ylim=(1e-2, 1), xticks=[0, 12, 24, 36, 48],
          yticks=[1e-2, 1e-1, 1], xlabel=r'Sketch queries $q$')
assert np.all(right_mse > .01) and np.all(right_mse < 1)
right.set_title(r'Step spectrum ($d=512,k=8$, tail 0.1)' + '\n' +
                r'24% rule vs $m/3$: $-' + f'{100 * selected["gain"]:.1f}' + r'\%$ MSE',
                fontsize=9, color=TEAL, fontweight='bold')
right.text(.5, -.68, 'Median held-out case outside power laws', transform=right.transAxes,
           ha='center', va='top', color=GRAY, fontsize=7.5)
for ax in axes:
    ax.grid(axis='y', color=LINE, lw=.5)
    ax.set_axisbelow(True)
fig.subplots_adjust(left=.105, right=.99, bottom=.26, top=.72, wspace=.36)
fig.savefig(fig_dir / 'allocation_examples.pdf', bbox_inches='tight', pad_inches=.05)
plt.close(fig)

with (fig_dir / 'allocation_examples.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['panel', 'id', 'family', 'eta', 'q', 'mean_mse'])
    writer.writerows(['left', r['id'], 'power', .005, r['q'], r['mean_mse']] for r in left_rows)
    writer.writerows(['right', r['id'], r['family'], '', r['q'], r['mean_exact_conditional_mse']] for r in selected_rows)
with (data_dir / 'heldout_case_gains.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(cases[0]))
    writer.writeheader()
    writer.writerows(cases)
with (data_dir / 'median_eligible_cases.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['ascending_rank', *cases[0]])
    writer.writeheader()
    writer.writerows({'ascending_rank': i + 1, **r} for i, r in enumerate(eligible))
evidence = {
    'sources': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [curve_file, policy_file, summary_file, manifest_file, phase4_file]},
    'selection': {'budget': 96, 'heldout_matrices': 186, 'excluded_families': sorted(excluded_families),
                  'eligible_cases': 144, 'convention': 'User-approved lower-middle observed case',
                  'ascending_rank': 72, 'tie_break': 'matrix ID ascending',
                  'arithmetic_median_gain': float(np.median([r['gain'] for r in eligible])),
                  'selected_case': selected, 'configuration': configuration,
                  'curve_q_star': int(right_q[right_best]), 'curve_mse_min': float(right_mse[right_best]),
                  'full_curve_points': len(selected_rows)},
    'distribution_all_744': distribution,
    'retained_aggregate': aggregate,
    'left_panel': {'q_star': 22, 'cap_q': 24, 'cap_excess_percent': 0, 'points': 48},
    'notes': ['No experiments run; all values read from saved mean-risk curves',
              'Casewise gain distribution is distinct from aggregate normalized-MSE saving',
              'Panels have independent logarithmic vertical ranges',
              'All 744 matrix-budget cases include all held-out families and four repeated budgets'],
}
(data_dir / 'slide8_evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
print(json.dumps({'selection': evidence['selection'], 'distribution': distribution}, indent=2))
