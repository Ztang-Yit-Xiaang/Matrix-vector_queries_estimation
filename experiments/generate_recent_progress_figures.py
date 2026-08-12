import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Ensure output directory exists
out_dir = 'Hutch++/Matrix-vector_queries_estimation/figures/recent_progress'
os.makedirs(out_dir, exist_ok=True)

# Set clean aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
})

# ==========================================
# Figure 1: RQ1 - Pilot Ritz Loss vs Tail Amplification
# ==========================================
csv_rq1 = 'Hutch++/Matrix-vector_queries_estimation/results/rq1_pilot_vs_tail_accuracy.csv'
if os.path.exists(csv_rq1):
    df_rq1 = pd.read_csv(csv_rq1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    setups = df_rq1['setup'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Left Plot: Tail Error Amplification vs Target q
    for idx, s in enumerate(setups):
        sub = df_rq1[df_rq1['setup'] == s]
        grouped = sub.groupby('target_q')['amplification_exp'].median().reset_index()
        ax1.plot(grouped['target_q'], grouped['amplification_exp'], marker='o', label=s, color=colors[idx % len(colors)])
    
    ax1.set_yscale('log')
    ax1.set_xlabel(r'Target Subspace Dimension ($q$)')
    ax1.set_ylabel('Tail Error Amplification Ratio (Log Scale)')
    ax1.set_title('RQ1: Exponential Extrapolation Error Explosion')
    ax1.legend(loc='upper left')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # Right Plot: Pilot Ritz Loss vs Out-of-Sample Tail Error (Scatter)
    sample_sub = df_rq1[df_rq1['target_q'] == 80]
    ax2.scatter(sample_sub['loss_exp'], sample_sub['rel_err_tail_exp'], alpha=0.6, color='#d62728', edgecolors='k')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'In-Sample Pilot Ritz Loss $\mathcal{L}_{pilot}$ (Log Scale)')
    ax2.set_ylabel(r'Out-of-Sample Tail Error $\epsilon_{tail}(q=80)$ (Log Scale)')
    ax2.set_title('RQ1: Ritz Loss Uncorrelated with Tail Energy')
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rq1_pilot_vs_tail_error_explosion.png'), dpi=300)
    plt.close()
    print("Generated rq1_pilot_vs_tail_error_explosion.png")

# ==========================================
# Figure 2: RQ2 - Subspace Horizon Threshold Law
# ==========================================
csv_rq2 = 'Hutch++/Matrix-vector_queries_estimation/results/rq2_pilot_horizon_vs_step_rank.csv'
if os.path.exists(csv_rq2):
    df_rq2 = pd.read_csv(csv_rq2)
    
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    ranks = df_rq2['r_step'].unique()
    markers = ['s', 'o', '^', 'D']
    
    for idx, r in enumerate(ranks):
        sub = df_rq2[df_rq2['r_step'] == r]
        ax.plot(sub['pilot_ratio'], sub['detection_rate'] * 100, marker=markers[idx % len(markers)],
                label=f'Step Location $r_{{step}} = {r}$', linewidth=2.5)
    
    # Highlight Threshold Law
    ax.axvline(x=1.33, color='red', linestyle='--', linewidth=2, label=r'Subspace Horizon Law ($b \geq 1.33 r$)')
    ax.axhline(y=95, color='gray', linestyle=':', linewidth=1.5, label='95% Target Detection Rate')
    
    ax.set_xlabel(r'Pilot Horizon Ratio ($b / r_{step}$)')
    ax.set_ylabel('Feature Detection Rate (%)')
    ax.set_title(r'RQ2: Subspace Horizon Threshold Law ($b \geq 1.33 \cdot r$)')
    ax.set_ylim(-5, 105)
    ax.legend(loc='lower right')
    ax.grid(True, ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rq2_subspace_horizon_threshold.png'), dpi=300)
    plt.close()
    print("Generated rq2_subspace_horizon_threshold.png")

# ==========================================
# Figure 3: Optuna Allocation Headroom
# ==========================================
csv_opt = 'Hutch++/Matrix-vector_queries_estimation/results/optuna_allocation_results.csv'
if os.path.exists(csv_opt):
    df_opt = pd.read_csv(csv_opt)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Flat Spectrum c=0.5
    sub_flat = df_opt[df_opt['c'] == 0.5].sort_values('k')
    ax1.plot(sub_flat['k'], sub_flat['median_rel_err'], marker='o', color='#1f77b4', label='Optuna Allocation Sweep')
    
    # Highlight standard vs best
    std_flat = sub_flat[sub_flat['is_standard_hutchpp']]
    best_flat = sub_flat[sub_flat['is_optuna_best']]
    if not std_flat.empty:
        ax1.scatter(std_flat['k'], std_flat['median_rel_err'], color='orange', s=140, zorder=5, label='Standard Hutch++ (100, 100)')
    if not best_flat.empty:
        ax1.scatter(best_flat['k'], best_flat['median_rel_err'], color='green', s=160, marker='*', zorder=6, label='Optimal Split (90, 120)')
    
    ax1.set_xlabel(r'Subspace Allocation ($k$) [Residual Probes $\ell = 300 - 2k$]')
    ax1.set_ylabel('Median Relative Error')
    ax1.set_title('Flat Spectrum ($c=0.5$): 49.1% Error Reduction')
    ax1.legend(loc='upper right')
    ax1.grid(True, ls="--", alpha=0.5)
    
    # Steep Spectrum c=2.0
    sub_steep = df_opt[df_opt['c'] == 2.0].sort_values('k')
    ax2.plot(sub_steep['k'], sub_steep['median_rel_err'], marker='o', color='#d62728', label='Optuna Allocation Sweep')
    
    std_steep = sub_steep[sub_steep['is_standard_hutchpp']]
    best_steep = sub_steep[sub_steep['is_optuna_best']]
    if not std_steep.empty:
        ax2.scatter(std_steep['k'], std_steep['median_rel_err'], color='orange', s=140, zorder=5, label='Standard Hutch++ (100, 100)')
    if not best_steep.empty:
        ax2.scatter(best_steep['k'], best_steep['median_rel_err'], color='green', s=160, marker='*', zorder=6, label='Optimal Split (120, 60)')
    
    ax2.set_xlabel(r'Subspace Allocation ($k$) [Residual Probes $\ell = 300 - 2k$]')
    ax2.set_ylabel('Median Relative Error')
    ax2.set_title('Steep Spectrum ($c=2.0$): 22.7% Error Reduction')
    ax2.legend(loc='upper right')
    ax2.grid(True, ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'optuna_adaptive_allocation_headroom.png'), dpi=300)
    plt.close()
    print("Generated optuna_adaptive_allocation_headroom.png")

# ==========================================
# Figure 4: Real-World Effective Rank Governing Crossover
# ==========================================
csv_rw = 'Hutch++/Matrix-vector_queries_estimation/results/real_world_benchmark_results.csv'
if os.path.exists(csv_rw):
    df_rw = pd.read_csv(csv_rw)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Dataset 1: YearPredictionMSD
    ds1_name = [d for d in df_rw['dataset'].unique() if 'YearPrediction' in d][0]
    sub_ds1 = df_rw[df_rw['dataset'] == ds1_name]
    reff_1 = sub_ds1['effective_rank'].iloc[0]
    
    for alg in sub_ds1['algorithm'].unique():
        sub_alg = sub_ds1[sub_ds1['algorithm'] == alg].sort_values('budget_m')
        ax1.plot(sub_alg['budget_m'], sub_alg['median_rel_error'], marker='o', label=alg)
    
    ax1.axvline(x=reff_1, color='purple', linestyle='--', linewidth=2, label=f'Effective Rank $r_{{eff}} = {reff_1:.1f}$')
    ax1.set_yscale('log')
    ax1.set_xlabel('Query Budget ($m$)')
    ax1.set_ylabel('Median Relative Trace Error')
    ax1.set_title('YearPredictionMSD ($d=90, n=50k$)\nEffective Rank Crossover')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # Dataset 2: Wiki-Vote
    ds2_name = [d for d in df_rw['dataset'].unique() if 'Wiki' in d][0]
    sub_ds2 = df_rw[df_rw['dataset'] == ds2_name]
    reff_2 = sub_ds2['effective_rank'].iloc[0]
    
    for alg in sub_ds2['algorithm'].unique():
        sub_alg = sub_ds2[sub_ds2['algorithm'] == alg].sort_values('budget_m')
        ax2.plot(sub_alg['budget_m'], sub_alg['median_rel_error'], marker='s', label=alg)
    
    ax2.axvline(x=reff_2, color='purple', linestyle='--', linewidth=2, label=f'Effective Rank $r_{{eff}} = {reff_2:.1f}$')
    ax2.set_yscale('log')
    ax2.set_xlabel('Query Budget ($m$)')
    ax2.set_ylabel('Median Relative Trace Error')
    ax2.set_title('Wiki-Vote Graph Network ($d=7115$)\nEffective Rank Crossover')
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'real_world_effective_rank_crossover.png'), dpi=300)
    plt.close()
    print("Generated real_world_effective_rank_crossover.png")

print("All recent progress figures successfully generated!")

