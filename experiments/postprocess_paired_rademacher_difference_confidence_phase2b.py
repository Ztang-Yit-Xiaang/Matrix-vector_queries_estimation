"""Generate the isolated, zero-query Phase 2B theorem-audit artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.paired_rademacher_difference_confidence import (  # noqa: E402
    ELEMENTARY_METHOD,
    FOURTH_MOMENT_FACTOR,
    FROZEN_JOINT_DELTAS,
    FROZEN_SAMPLE_SIZES,
    METHOD_PRIORITY,
    PAIRWISE_SCALE_RELATIVE_VARIANCE_FACTOR,
    PROVED_BUT_BUDGET_VACUOUS,
    elementary_scale_relative_radius,
    minimum_scale_block_size,
    phase2b_final_verdict,
    phase2b_route_audit,
    sample_variance_relative_variance_bound,
)


RESULTS_DIR = PROJECT_DIR / "results"
FROZEN_INPUTS = (
    RESULTS_DIR / "paired_rademacher_risk_difference_phase2a_manifest.csv",
    RESULTS_DIR / "paired_rademacher_risk_difference_phase2a_summary.csv",
    RESULTS_DIR / "paired_rademacher_risk_difference_phase2a_gate.csv",
    RESULTS_DIR / "paired_rademacher_risk_difference_phase2a_verdict.csv",
    RESULTS_DIR / "rademacher_linear_projection_no_go_phase1d_manifest.csv",
    RESULTS_DIR / "rademacher_linear_projection_no_go_phase1d_verdict.csv",
)


def sha256(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_phase2b_artifacts() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frozen_before = {str(path.relative_to(PROJECT_DIR)): sha256(path) for path in FROZEN_INPUTS}
    if any(value == "MISSING" for value in frozen_before.values()):
        raise RuntimeError("A required frozen Phase 1D/2A input is missing.")

    route_rows = []
    for method in METHOD_PRIORITY:
        for sample_size in FROZEN_SAMPLE_SIZES:
            for joint_delta in FROZEN_JOINT_DELTAS:
                audit = phase2b_route_audit(sample_size, joint_delta, method)
                route_rows.append(audit.__dict__)
    route_grid = pd.DataFrame(route_rows)
    if len(route_grid) != 36:
        raise RuntimeError("The Phase 2B route grid must contain exactly 36 rows.")
    route_path = RESULTS_DIR / "paired_rademacher_difference_confidence_phase2b_route_grid.csv"
    route_grid.to_csv(route_path, index=False)

    scale_rows = []
    for sample_size in FROZEN_SAMPLE_SIZES:
        n = sample_size // 2
        for joint_delta in FROZEN_JOINT_DELTAS:
            scale_delta = joint_delta / 2.0
            scale_rows.append(
                {
                    "sample_size": sample_size,
                    "independent_pair_count": n,
                    "joint_delta": joint_delta,
                    "scale_delta": scale_delta,
                    "sample_variance_relative_variance_bound": (
                        sample_variance_relative_variance_bound(n)
                    ),
                    "elementary_scale_relative_radius": (
                        elementary_scale_relative_radius(n, scale_delta)
                    ),
                    "pairwise_scale_relative_variance_factor": (
                        PAIRWISE_SCALE_RELATIVE_VARIANCE_FACTOR
                    ),
                    "minimum_25pct_scale_block_size": minimum_scale_block_size(),
                    "available_signed_pairs": n,
                    "available_pairwise_scale_observations": n // 2,
                    "elementary_nonvacuous": False,
                    "robust_mom_nonvacuous": False,
                }
            )
    scale_grid = pd.DataFrame(scale_rows)
    if len(scale_grid) != 12:
        raise RuntimeError("The Phase 2B analytic scale grid must contain exactly 12 rows.")
    if not (scale_grid["elementary_scale_relative_radius"] > 1.0).all():
        raise RuntimeError("The analytic scale-vacuity invariant failed.")
    scale_path = RESULTS_DIR / "paired_rademacher_difference_confidence_phase2b_scale_grid.csv"
    scale_grid.to_csv(scale_path, index=False)

    verdict = phase2b_final_verdict()
    if verdict != PROVED_BUT_BUDGET_VACUOUS:
        raise RuntimeError("The deterministic Phase 2B early-stop verdict changed.")
    verdict_path = RESULTS_DIR / "paired_rademacher_difference_confidence_phase2b_verdict.csv"
    pd.DataFrame(
        [
            {
                "phase": "Phase 2B",
                "verdict": verdict,
                "primary_method": METHOD_PRIORITY[0],
                "primary_sample_size": 16,
                "primary_joint_delta": 0.05,
                "fourth_moment_factor": FOURTH_MOMENT_FACTOR,
                "minimum_robust_scale_block_size": minimum_scale_block_size(),
                "frozen_replay_performed": False,
                "new_matvec_queries": 0,
                "allocator_modified": False,
                "scope": (
                    "Two explicit data-only independent-pair theorem routes; "
                    "not an impossibility theorem for all certificates."
                ),
            }
        ]
    ).to_csv(verdict_path, index=False)

    frozen_after = {str(path.relative_to(PROJECT_DIR)): sha256(path) for path in FROZEN_INPUTS}
    if frozen_before != frozen_after:
        raise RuntimeError("A frozen historical artifact changed during Phase 2B.")

    outputs = {
        str(path.relative_to(PROJECT_DIR)): sha256(path)
        for path in (route_path, scale_path, verdict_path)
    }
    manifest_path = RESULTS_DIR / "paired_rademacher_difference_confidence_phase2b_manifest.csv"
    pd.DataFrame(
        [
            {
                "phase": "Phase 2B",
                "design_spec": (
                    "docs/superpowers/specs/"
                    "2026-08-19-direct-paired-risk-difference-confidence-phase2b-design.md"
                ),
                "hypercontractive_source": (
                    "O'Donnell Analysis of Boolean Functions Lecture 16 Corollary 1.3"
                ),
                "hypercontractive_source_url": (
                    "https://www.cs.cmu.edu/~odonnell/boolean-analysis/lecture16.pdf"
                ),
                "canonical_u_source_url": "https://arxiv.org/abs/math/0003228",
                "empirical_bernstein_scope_source_url": "https://arxiv.org/abs/0907.3740",
                "catoni_audit_source_url": "https://arxiv.org/abs/0909.5366",
                "fourth_moment_factor": FOURTH_MOMENT_FACTOR,
                "sample_sizes": json.dumps(FROZEN_SAMPLE_SIZES),
                "joint_deltas": json.dumps(FROZEN_JOINT_DELTAS),
                "method_priority": json.dumps(METHOD_PRIORITY),
                "route_grid_rows": len(route_grid),
                "scale_grid_rows": len(scale_grid),
                "verdict": verdict,
                "frozen_replay_performed": False,
                "new_matvec_queries": 0,
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "frozen_input_checksums": json.dumps(frozen_before, sort_keys=True),
                "output_checksums_excluding_manifest": json.dumps(outputs, sort_keys=True),
            }
        ]
    ).to_csv(manifest_path, index=False)


if __name__ == "__main__":
    generate_phase2b_artifacts()
