"""Postprocess and evaluate Phase 2C structural priors across sample size and delta grids.

Produces two publication-ready CSV artifacts in results/:
1. results/structural_certificate_grid_phase2c.csv
   - Evaluates Cantelli one-sided certificate radius across (s, delta, M0, ell_a, ell_0).
   - Evaluates finite radii under a supplied prior, without claiming useful acceptance.
2. results/structural_kurtosis_boundary_phase2c.csv
   - Evaluates refined chaos kurtosis factor and Chebyshev sample size feasibility boundary
     across structural ratio kappa in (0, 1].
"""

import os
import sys
import numpy as np
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.structural_paired_difference_confidence import (
    cantelli_structural_one_sided_radius,
    norm_envelope_variance_bound,
    structural_chaos_fourth_moment_factor,
    chebyshev_scale_feasibility_threshold,
)


def run_structural_grid_evaluation(output_dir=None):
    sample_sizes = [4, 8, 16, 32]
    deltas = [0.01, 0.05, 0.10]
    M0_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    denominators = [(70, 80), (80, 100), (100, 100)]

    rows = []
    for s in sample_sizes:
        n = s // 2
        for delta in deltas:
            for M0 in M0_values:
                for ell_a, ell_0 in denominators:
                    var_bound = norm_envelope_variance_bound(M0, ell_a, ell_0)
                    radius = cantelli_structural_one_sided_radius(s, delta, M0, ell_a, ell_0)
                    rows.append({
                        "sample_size_s": s,
                        "independent_pairs_n": n,
                        "delta": delta,
                        "norm_envelope_M0": M0,
                        "ell_candidate": ell_a,
                        "ell_baseline": ell_0,
                        "variance_bound": var_bound,
                        "one_sided_radius": radius,
                        "radius_finite": bool(np.isfinite(radius)),
                        "useful_acceptance_established": False,
                        "phase2b_data_only_radius": "inf",
                    })

    df_grid = pd.DataFrame(rows)
    if output_dir is None:
        return df_grid
    results_dir = os.fspath(output_dir)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "structural_certificate_grid_phase2c.csv")
    df_grid.to_csv(out_path, index=False)
    print(f"Saved {len(df_grid)} rows to {out_path}")
    return df_grid


def run_kurtosis_boundary_evaluation(output_dir=None):
    kappas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    deltas = [0.01, 0.025, 0.05, 0.10]

    rows = []
    for kappa in kappas:
        effective_rank = 1.0 / (kappa ** 2)
        kurtosis_factor = structural_chaos_fourth_moment_factor(kappa)
        for delta_scale in deltas:
            n_star = chebyshev_scale_feasibility_threshold(kurtosis_factor, delta_scale)
            rows.append({
                "structural_ratio_kappa": kappa,
                "effective_rank": effective_rank,
                "chaos_kurtosis_factor": kurtosis_factor,
                "delta_scale": delta_scale,
                "min_sample_size_n_star": n_star,
                "min_probes_s_star": n_star * 2,
                "feasible_at_s32": bool(n_star <= 16),
                "scope": "illustrative K input; degree-two bound is not a signed-pair kurtosis bound",
            })

    df_kurt = pd.DataFrame(rows)
    if output_dir is None:
        return df_kurt
    results_dir = os.fspath(output_dir)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "structural_kurtosis_boundary_phase2c.csv")
    df_kurt.to_csv(out_path, index=False)
    print(f"Saved {len(df_kurt)} rows to {out_path}")
    return df_kurt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_structural_grid_evaluation(args.output_dir)
    run_kurtosis_boundary_evaluation(args.output_dir)
