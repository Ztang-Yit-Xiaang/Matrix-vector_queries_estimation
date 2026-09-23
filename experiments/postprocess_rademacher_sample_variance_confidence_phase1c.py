"""Phase 1C theorem/usefulness audit over frozen Phase 1A/1B artifacts.

No matrix-vector query, range sketch, or certification probe is generated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rademacher_sample_variance_confidence import (  # noqa: E402
    CHEBYSHEV_METHOD,
    INCOMPLETE,
    METHOD_PRIORITY,
    PROVED_BUT_BUDGET_VACUOUS,
)
from rademacher_sample_variance_confidence import (  # noqa: E402
    select_gate_sample_size,
    sharper_hoeffding_theorem_audit,
    two_action_sample_variance_certificate,
)


DEFAULT_RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR
PROOF_PATH = ROOT_DIR / "docs" / "proof_rademacher_sample_variance_confidence.md"
REPORT_PATH = ROOT_DIR / "reports" / "rademacher_sample_variance_confidence_phase1c.md"
SAMPLE_SIZE_GRID = (4, 8, 16, 32)
JOINT_DELTA_GRID = (0.01, 0.05, 0.10)
PRIMARY_JOINT_DELTA = 0.05
PRIMARY_PHASE1B = {
    "eta": 1e-6,
    "budget": 160,
    "s": 16,
    "pair": "primary",
    "estimator": "sample_variance",
    "epsilon": 1.0 / 3.0,
}
VERDICT_BUDGET_VACUOUS = "THEOREM ONLY / BUDGET-VACUOUS"
OUTPUT_PREFIX = "rademacher_sample_variance_confidence_phase1c"
OUTPUTS = {
    "manifest": f"{OUTPUT_PREFIX}_manifest.csv",
    "grid": f"{OUTPUT_PREFIX}_certificate_grid.csv",
    "gate": f"{OUTPUT_PREFIX}_gate.csv",
    "bootstrap": f"{OUTPUT_PREFIX}_bootstrap.csv",
    "verdict": f"{OUTPUT_PREFIX}_verdict.csv",
}

SOURCE_PATHS = {
    "phase1a_manifest": DEFAULT_RESULTS_DIR
    / "direct_rademacher_certification_phase1a_manifest.csv",
    "phase1a_trials": DEFAULT_RESULTS_DIR
    / "direct_rademacher_certification_phase1a_trials.parquet",
    "phase1b_manifest": DEFAULT_RESULTS_DIR
    / "direct_rademacher_certification_phase1b_budget_manifest.csv",
    "phase1b_paths": DEFAULT_RESULTS_DIR
    / "direct_rademacher_certification_phase1b_budget_paths.parquet",
    "phase1b_summary": DEFAULT_RESULTS_DIR
    / "direct_rademacher_certification_phase1b_budget_summary.csv",
    "phase1b_bootstrap": DEFAULT_RESULTS_DIR
    / "direct_rademacher_certification_phase1b_budget_bootstrap.csv",
    "phase1b_catastrophic": DEFAULT_RESULTS_DIR
    / "direct_rademacher_certification_phase1b_budget_catastrophic.csv",
}


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _single_manifest(path):
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise RuntimeError(f"{path} must contain exactly one manifest row.")
    return frame.iloc[0].to_dict()


def validate_frozen_sources(source_paths=SOURCE_PATHS, validate_historical=True):
    """Validate every source checksum and optionally all historical CSV hashes."""

    source_paths = {name: Path(path) for name, path in source_paths.items()}
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen Phase 1C source files: {missing}")

    phase1a = _single_manifest(source_paths["phase1a_manifest"])
    phase1b = _single_manifest(source_paths["phase1b_manifest"])
    phase1a_outputs = json.loads(phase1a["output_checksums_excluding_manifest"])
    phase1b_outputs = json.loads(phase1b["output_checksums_excluding_manifest"])
    expected = {
        "phase1a_trials": phase1a_outputs[source_paths["phase1a_trials"].name],
        "phase1b_paths": phase1b_outputs[source_paths["phase1b_paths"].name],
        "phase1b_summary": phase1b_outputs[source_paths["phase1b_summary"].name],
        "phase1b_bootstrap": phase1b_outputs[source_paths["phase1b_bootstrap"].name],
        "phase1b_catastrophic": phase1b_outputs[
            source_paths["phase1b_catastrophic"].name
        ],
    }
    actual = {name: _hash_file(path) for name, path in source_paths.items()}
    for name, checksum in expected.items():
        if actual[name] != checksum:
            raise RuntimeError(f"Frozen source checksum mismatch for {name}.")

    if validate_historical:
        historical = json.loads(phase1b["historical_checksums"])
        for filename, checksum in historical.items():
            path = DEFAULT_RESULTS_DIR / filename
            if not path.is_file() or _hash_file(path) != checksum:
                raise RuntimeError(f"Historical artifact changed: {filename}")
    return actual, phase1a, phase1b


def build_certificate_grid(
    sample_sizes=SAMPLE_SIZE_GRID,
    joint_deltas=JOINT_DELTA_GRID,
):
    """Build the preregistered analytic grid; empirical data never enters."""

    rows = []
    for joint_delta in joint_deltas:
        for sample_size in sample_sizes:
            certificate = two_action_sample_variance_certificate(
                sample_size=sample_size,
                joint_delta=joint_delta,
                method=CHEBYSHEV_METHOD,
            )
            rows.append(
                {
                    "sample_size": sample_size,
                    "joint_delta": joint_delta,
                    "method": CHEBYSHEV_METHOD,
                    "selected_proved_method": True,
                    "proof_status": certificate.proof_status,
                    "epsilon": certificate.epsilon,
                    "nonvacuous": certificate.nonvacuous,
                    "action_count": certificate.action_count,
                    "per_action_delta": certificate.per_action_delta,
                    "linear_component_delta": np.nan,
                    "degenerate_component_delta": np.nan,
                    "fourth_moment_factor": 81.0,
                    "unresolved_step": "",
                }
            )
            audit = sharper_hoeffding_theorem_audit(joint_delta)
            rows.append(
                {
                    "sample_size": sample_size,
                    "joint_delta": joint_delta,
                    "method": "hoeffding_sharper_attempt",
                    "selected_proved_method": False,
                    "proof_status": audit.proof_status,
                    "epsilon": np.nan,
                    "nonvacuous": False,
                    "action_count": 2,
                    "per_action_delta": audit.per_action_delta,
                    "linear_component_delta": audit.linear_component_delta,
                    "degenerate_component_delta": audit.degenerate_component_delta,
                    "fourth_moment_factor": 81.0,
                    "unresolved_step": audit.unresolved_step,
                }
            )
    grid = pd.DataFrame(rows)
    if len(grid) != len(sample_sizes) * len(joint_deltas) * 2:
        raise RuntimeError("Certificate grid row count is inconsistent.")
    return grid


def _primary_filter(frame):
    mask = np.ones(len(frame), dtype=bool)
    for name, value in PRIMARY_PHASE1B.items():
        if isinstance(value, float):
            mask &= np.isclose(frame[name].to_numpy(dtype=float), value)
        else:
            mask &= frame[name].eq(value).to_numpy()
    return frame.loc[mask].copy()


def load_frozen_context(source_paths=SOURCE_PATHS):
    """Load Phase 1B numbers for context only, not for theorem selection."""

    summary = _primary_filter(pd.read_csv(source_paths["phase1b_summary"]))
    summary = summary.loc[summary["rank_scope"].eq("equal_rank")]
    if len(summary) != 1:
        raise RuntimeError("Expected one equal-rank primary Phase 1B summary row.")
    bootstrap = _primary_filter(pd.read_csv(source_paths["phase1b_bootstrap"]))
    selected_bootstrap = bootstrap.loc[
        bootstrap["metric"].eq("selected_mean_ratio")
    ]
    if len(selected_bootstrap) != 1:
        raise RuntimeError("Expected one primary selected-risk bootstrap row.")
    catastrophic = _primary_filter(pd.read_csv(source_paths["phase1b_catastrophic"]))
    catastrophic = catastrophic.loc[np.isclose(catastrophic["fraction"], 0.05)]
    if len(catastrophic) != 1:
        raise RuntimeError("Expected one primary top-five-percent Phase 1B row.")
    return summary.iloc[0], selected_bootstrap.iloc[0], catastrophic.iloc[0]


def evaluate_gate(grid, frozen_context):
    """Apply the analytic sample-size rule before reading empirical outcomes."""

    summary, selected_bootstrap, catastrophic = frozen_context
    s_gate = select_gate_sample_size(SAMPLE_SIZE_GRID, PRIMARY_JOINT_DELTA)
    gate_evaluable = s_gate is not None
    if gate_evaluable:
        raise RuntimeError(
            "A nonvacuous proved certificate now exists; implement the frozen practical "
            "gate rather than silently reusing the budget-vacuous branch."
        )
    primary_rows = grid.loc[
        np.isclose(grid["joint_delta"], PRIMARY_JOINT_DELTA)
        & grid["selected_proved_method"]
    ]
    if len(primary_rows) != len(SAMPLE_SIZE_GRID):
        raise RuntimeError("Primary analytic grid is incomplete.")
    if primary_rows["nonvacuous"].any():
        raise RuntimeError("The frozen s<=32 Chebyshev vacuity theorem was violated.")
    gate = pd.DataFrame(
        [
            {
                "joint_delta": PRIMARY_JOINT_DELTA,
                "sample_size_grid": json.dumps(list(SAMPLE_SIZE_GRID)),
                "s_gate": np.nan,
                "gate_evaluable": False,
                "practical_bootstrap_run": False,
                "analytic_selection_used_empirical_data": False,
                "smallest_grid_epsilon": float(primary_rows["epsilon"].min()),
                "frozen_phase1b_selected_original_ratio": float(
                    summary["selected_mean_ratio"]
                ),
                "frozen_phase1b_ratio_ci_low": float(selected_bootstrap["ci_low"]),
                "frozen_phase1b_ratio_ci_high": float(selected_bootstrap["ci_high"]),
                "frozen_phase1b_top5_detection": float(
                    catastrophic["net_beneficial_accept_probability"]
                ),
                "verdict": VERDICT_BUDGET_VACUOUS,
                "reason": (
                    "No proved method has epsilon<1 on the preregistered grid; "
                    "empirical outcomes were not searched to select s."
                ),
            }
        ]
    )
    return gate


def build_verdict(grid, gate):
    chebyshev = grid.loc[grid["method"].eq(CHEBYSHEV_METHOD)]
    sharper = grid.loc[grid["method"].eq("hoeffding_sharper_attempt")]
    if not chebyshev["proof_status"].eq(PROVED_BUT_BUDGET_VACUOUS).all():
        raise RuntimeError("Chebyshev theorem classification changed unexpectedly.")
    if not sharper["proof_status"].eq(INCOMPLETE).all():
        raise RuntimeError("Sharper theorem audit classification changed unexpectedly.")
    return pd.DataFrame(
        [
            {
                "verdict": gate.loc[0, "verdict"],
                "proved_theorem": "81-hypercontractive Chebyshev",
                "proved_theorem_status": PROVED_BUT_BUDGET_VACUOUS,
                "sharper_route_status": INCOMPLETE,
                "primary_joint_delta": PRIMARY_JOINT_DELTA,
                "s_gate": np.nan,
                "online_allocator_implemented": False,
                "new_matvec_queries": 0,
                "interpretation": (
                    "The finite-sample theorem is valid, but no multiplicative lower "
                    "confidence bound exists on s<=32; the sharper Hoeffding route is "
                    "not numerically closed."
                ),
            }
        ]
    )


def _empty_bootstrap():
    return pd.DataFrame(
        columns=[
            "sample_size",
            "joint_delta",
            "metric",
            "point",
            "ci_low",
            "ci_high",
            "bootstrap_samples",
            "status",
        ]
    )


def _atomic_write_frames(output_dir, frames):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="phase1c_", dir=output_dir.parent))
    try:
        for key, frame in frames.items():
            frame.to_csv(staging / OUTPUTS[key], index=False)
        for key in frames:
            shutil.move(str(staging / OUTPUTS[key]), str(output_dir / OUTPUTS[key]))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def run_audit(
    output_dir=DEFAULT_OUTPUT_DIR,
    source_paths=SOURCE_PATHS,
    validate_historical=True,
):
    started = datetime.now(timezone.utc).isoformat()
    source_checksums, phase1a, phase1b = validate_frozen_sources(
        source_paths=source_paths,
        validate_historical=validate_historical,
    )
    grid = build_certificate_grid()
    context = load_frozen_context(source_paths)
    gate = evaluate_gate(grid, context)
    verdict = build_verdict(grid, gate)
    bootstrap = _empty_bootstrap()
    frames = {
        "grid": grid,
        "gate": gate,
        "bootstrap": bootstrap,
        "verdict": verdict,
    }
    _atomic_write_frames(output_dir, frames)

    output_dir = Path(output_dir)
    output_checksums = {
        OUTPUTS[key]: _hash_file(output_dir / OUTPUTS[key]) for key in frames
    }
    for path in (PROOF_PATH, REPORT_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
        output_checksums[str(path.relative_to(ROOT_DIR))] = _hash_file(path)
    # Revalidate after output generation so accidental source mutation cannot hide.
    ending_checksums, _, _ = validate_frozen_sources(
        source_paths=source_paths,
        validate_historical=validate_historical,
    )
    if ending_checksums != source_checksums:
        raise RuntimeError("A frozen input changed during the Phase 1C audit.")
    manifest = pd.DataFrame(
        [
            {
                "configuration_version": "rademacher_sample_variance_phase1c_v1",
                "started_at_utc": started,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "sample_size_grid": json.dumps(list(SAMPLE_SIZE_GRID)),
                "joint_delta_grid": json.dumps(list(JOINT_DELTA_GRID)),
                "primary_joint_delta": PRIMARY_JOINT_DELTA,
                "method_priority": json.dumps(list(METHOD_PRIORITY)),
                "public_delta_semantics": "joint failure probability over two fixed actions",
                "source_checksums": json.dumps(source_checksums, sort_keys=True),
                "phase1a_configuration_version": phase1a["configuration_version"],
                "phase1b_configuration_version": phase1b["configuration_version"],
                "historical_checksums_verified": bool(validate_historical),
                "certificate_grid_rows": len(grid),
                "gate_rows": len(gate),
                "bootstrap_rows": len(bootstrap),
                "verdict": verdict.loc[0, "verdict"],
                "new_matvec_queries": 0,
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "output_checksums_excluding_manifest": json.dumps(
                    output_checksums, sort_keys=True
                ),
            }
        ]
    )
    manifest.to_csv(output_dir / OUTPUTS["manifest"], index=False)
    return {
        "grid": grid,
        "gate": gate,
        "bootstrap": bootstrap,
        "verdict": verdict,
        "manifest": manifest,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-historical-checks",
        action="store_true",
        help="Use only for isolated smoke fixtures, never for production.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_audit(
        output_dir=args.output_dir,
        validate_historical=not args.skip_historical_checks,
    )
    print(result["verdict"].to_string(index=False))


if __name__ == "__main__":
    main()
