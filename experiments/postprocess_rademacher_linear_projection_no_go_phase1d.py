"""
Postprocessing script for Phase 1D: Rademacher Linear Projection Lower-Tail Analytic No-Go.

Generates:
1. results/rademacher_linear_projection_no_go_phase1d_manifest.csv
2. results/rademacher_linear_projection_no_go_phase1d_theorem_grid.csv (24 rows)
3. results/rademacher_linear_projection_no_go_phase1d_cap_regression.csv (900 rows)
4. results/rademacher_linear_projection_no_go_phase1d_verdict.csv
"""

import hashlib
import json
import os
import platform
import sys
import pandas as pd
import numpy as np

# Ensure parent directory is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.rademacher_linear_projection_no_go import (
    FROZEN_SAMPLE_SIZES,
    FROZEN_JOINT_DELTAS,
    FROZEN_ALLOCATIONS,
    FROZEN_CAP_GRID,
    FROZEN_KAPPAS,
    FROZEN_EPSILONS,
    FOURTH_MOMENT_BOUND,
    VARIANCE_FLOOR_LOWER,
    MINIMUM_FROZEN_CAP,
    ANALYTIC_EXPONENT_DIVISOR,
    linear_lower_tail_no_go_audit,
    truncation_bias_upper_bound,
    capped_variance_envelope,
    one_sided_bernstein_lower_tail,
    analytic_probability_floor,
)

RESULTS_DIR = os.path.join(project_dir, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FROZEN_INPUTS = (
    os.path.join(RESULTS_DIR, "rademacher_sample_variance_confidence_phase1c_manifest.csv"),
    os.path.join(RESULTS_DIR, "rademacher_sample_variance_confidence_phase1c_certificate_grid.csv"),
    os.path.join(RESULTS_DIR, "rademacher_sample_variance_confidence_phase1c_gate.csv"),
    os.path.join(RESULTS_DIR, "rademacher_sample_variance_confidence_phase1c_verdict.csv"),
)


def compute_file_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "MISSING"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def generate_phase1d_artifacts():
    print("Generating Phase 1D artifacts...")
    frozen_input_checksums = {
        os.path.relpath(path, project_dir): compute_file_sha256(path)
        for path in FROZEN_INPUTS
    }
    if any(value == "MISSING" for value in frozen_input_checksums.values()):
        raise RuntimeError("A frozen Phase 1C input is missing.")

    # 1. Generate Theorem Grid (24 rows)
    theorem_rows = []
    for s in FROZEN_SAMPLE_SIZES:
        for delta in FROZEN_JOINT_DELTAS:
            for alloc in FROZEN_ALLOCATIONS:
                audit = linear_lower_tail_no_go_audit(s, delta, alloc)
                theorem_rows.append({
                    "sample_size": audit.sample_size,
                    "joint_delta": audit.joint_delta,
                    "allocation": audit.allocation,
                    "directed_component_delta": audit.directed_component_delta,
                    "cap_minimum": audit.cap_minimum,
                    "variance_envelope_floor": audit.variance_envelope_floor,
                    "analytic_probability_floor": audit.analytic_probability_floor,
                    "maximum_declared_component_delta": audit.maximum_declared_component_delta,
                    "theorem_status": audit.theorem_status,
                    "verdict": audit.verdict,
                })

    df_theorem = pd.DataFrame(theorem_rows)
    assert len(df_theorem) == 24, f"Expected 24 theorem rows, got {len(df_theorem)}"
    theorem_path = os.path.join(RESULTS_DIR, "rademacher_linear_projection_no_go_phase1d_theorem_grid.csv")
    df_theorem.to_csv(theorem_path, index=False)
    print(f"Saved {theorem_path} (24 rows)")

    # 2. Generate Cap Regression Grid (900 rows)
    cap_rows = []
    for s in FROZEN_SAMPLE_SIZES:
        prob_floor = analytic_probability_floor(s)
        for T in FROZEN_CAP_GRID:
            for kappa in FROZEN_KAPPAS:
                beta = truncation_bias_upper_bound(T, kappa)
                nu = capped_variance_envelope(T, kappa)
                for eps in FROZEN_EPSILONS:
                    D_val = one_sided_bernstein_lower_tail(s, eps, T, kappa)
                    is_admissible = bool(beta < eps)
                    strictly_exceeds_floor = bool(D_val > prob_floor)
                    cap_rows.append({
                        "sample_size": s,
                        "T_cap": T,
                        "kappa": kappa,
                        "epsilon": eps,
                        "truncation_bias_beta": beta,
                        "variance_envelope_nu": nu,
                        "bernstein_lower_tail_D": D_val,
                        "analytic_probability_floor": prob_floor,
                        "is_admissible": is_admissible,
                        "strictly_exceeds_analytic_floor": strictly_exceeds_floor,
                    })

    df_cap = pd.DataFrame(cap_rows)
    assert len(df_cap) == 900, f"Expected 900 cap rows, got {len(df_cap)}"
    # Verify invariants
    assert (df_cap["variance_envelope_nu"] >= 80.0).all() and (df_cap["variance_envelope_nu"] <= 81.0).all(), "Variance floor violated!"
    assert (df_cap["strictly_exceeds_analytic_floor"]).all(), "Strict analytic floor violated!"

    cap_path = os.path.join(RESULTS_DIR, "rademacher_linear_projection_no_go_phase1d_cap_regression.csv")
    df_cap.to_csv(cap_path, index=False)
    print(f"Saved {cap_path} (900 rows)")

    # 3. Generate Verdict Summary (1 row)
    verdict_path = os.path.join(RESULTS_DIR, "rademacher_linear_projection_no_go_phase1d_verdict.csv")
    df_verdict = pd.DataFrame([{
        "phase": "Phase 1D",
        "title": "Rademacher Linear Projection Lower-Tail Analytic No-Go",
        "verdict": "STRONG LINEAR NO-GO",
        "theorem_status": "PROVED",
        "analytic_probability_floor_max_s32": float(np.exp(-32.0 / 160.0)),
        "max_declared_component_delta": 0.05,
        "new_matvec_queries": 0,
        "implication": "Linear Hoeffding component cannot close small-sample certification under truncation-Bernstein; route is budget-vacuous for s <= 32."
    }])
    df_verdict.to_csv(verdict_path, index=False)
    print(f"Saved {verdict_path} (1 row)")

    frozen_input_checksums_after = {
        os.path.relpath(path, project_dir): compute_file_sha256(path)
        for path in FROZEN_INPUTS
    }
    if frozen_input_checksums_after != frozen_input_checksums:
        raise RuntimeError("A frozen Phase 1C input changed during Phase 1D generation.")

    phase1d_outputs = (
        theorem_path,
        cap_path,
        verdict_path,
        os.path.join(project_dir, "src", "rademacher_linear_projection_no_go.py"),
        os.path.join(project_dir, "docs", "proof_rademacher_sample_variance_confidence.md"),
        os.path.join(project_dir, "reports", "rademacher_linear_projection_no_go_phase1d.md"),
    )
    output_checksums = {
        os.path.relpath(path, project_dir): compute_file_sha256(path)
        for path in phase1d_outputs
    }
    if any(value == "MISSING" for value in output_checksums.values()):
        raise RuntimeError("A required Phase 1D output is missing.")

    # 4. Generate Manifest
    manifest_path = os.path.join(RESULTS_DIR, "rademacher_linear_projection_no_go_phase1d_manifest.csv")
    df_manifest = pd.DataFrame([{
        "source_doi": "https://doi.org/10.1007/s10208-021-09525-9",
        "source_theorem": "Cortinovis and Kressner (2022) Theorem 2, equation (8)",
        "fourth_moment_factor": FOURTH_MOMENT_BOUND,
        "variance_envelope_floor": VARIANCE_FLOOR_LOWER,
        "minimum_frozen_cap": MINIMUM_FROZEN_CAP,
        "analytic_exponent_divisor": ANALYTIC_EXPONENT_DIVISOR,
        "analytic_floor_s32": float(np.exp(-0.2)),
        "new_matvec_queries": 0,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "theorem_grid_rows": len(df_theorem),
        "cap_regression_rows": len(df_cap),
        "verdict": "STRONG LINEAR NO-GO",
        "frozen_input_checksums": json.dumps(frozen_input_checksums, sort_keys=True),
        "output_checksums_excluding_manifest": json.dumps(output_checksums, sort_keys=True),
        "theorem_grid_sha256": compute_file_sha256(theorem_path),
        "cap_regression_sha256": compute_file_sha256(cap_path),
        "verdict_sha256": compute_file_sha256(verdict_path),
    }])
    df_manifest.to_csv(manifest_path, index=False)
    print(f"Saved {manifest_path} (1 row)")
    print("All Phase 1D artifacts successfully generated!")


if __name__ == "__main__":
    generate_phase1d_artifacts()
