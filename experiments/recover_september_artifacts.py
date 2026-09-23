"""Recover the two overwritten September runs with provenance and isolation checks.

No allocator parameters are changed. Recomputed artifacts are identified as reruns,
not byte-identical restoration of unavailable pre-inspection files.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import os
import shutil
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RECOVERY = ROOT / "results/recovery_20260914"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot():
    return {str(p.relative_to(ROOT)): digest(p)
            for p in sorted((ROOT / "results").rglob("*"))
            if p.is_file() and RECOVERY not in p.parents}


def finalize():
    """Publish only validated reruns after checking every existing result hash."""
    import pandas as pd
    before = json.loads((RECOVERY / "before_checksums.json").read_text())
    current = snapshot()
    assert current == before, "Production artifacts changed during recovery; investigate before publishing."
    suite = (RECOVERY / "full_suite.log").read_text()
    assert re.search(r"\d+ passed", suite) and not re.search(r"\d+ (failed|errors?)", suite)
    specs = {
        "haar_orientation_audit_trials.csv": (1860, ["case_name", "orient_idx", "trial"]),
        "haar_orientation_audit_summary.csv": (12, ["case_name", "orient_type"]),
        "pytorch_hessian_benchmark_trials.csv": (480, ["budget_m", "algorithm", "trial"]),
        "pytorch_hessian_benchmark_summary.csv": (16, ["budget_m", "algorithm"]),
        "structural_certificate_grid_phase2c.csv": (252, ["sample_size_s", "delta", "norm_envelope_M0", "ell_candidate", "ell_baseline"]),
        "structural_kurtosis_boundary_phase2c.csv": (36, ["structural_ratio_kappa", "delta_scale"]),
    }
    for name, (rows, key) in specs.items():
        df = pd.read_csv(RECOVERY / "staged" / name)
        assert len(df) == rows and not df.duplicated(key).any(), name
    for stage in ("haar", "hessian"):
        metadata = json.loads((RECOVERY / f"{stage}_manifest.json").read_text())
        for name, expected in metadata["files"].items():
            assert digest(RECOVERY / "staged" / name) == expected
    for name in specs:
        destination = ROOT / "results" / name
        temporary = destination.with_suffix(".recovery-tmp")
        shutil.copy2(RECOVERY / "staged" / name, temporary)
        os.replace(temporary, destination)
    after = snapshot()
    changed = sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))
    assert changed == sorted("results/" + name for name in specs)
    sources = ["src/trace_baseline.py", "src/pytorch_matvec_oracle.py",
               "src/structural_paired_difference_confidence.py",
               "experiments/run_haar_orientation_audit.py", "experiments/run_pytorch_hessian_benchmark.py"]
    manifest = dict(completed=datetime.now(timezone.utc).isoformat(),
                    baseline_commit="8a983e5", tests=re.findall(r"\d+ passed[^\n]*", suite)[-1],
                    changed_artifacts=changed, preserved_artifact_count=len(before)-len(changed),
                    after_checksums=after,
                    source_checksums={path: digest(ROOT / path) for path in sources},
                    haar_estimator_calls=1860*4, haar_matvec_queries=1860*4*60,
                    hessian_estimator_calls=480, hessian_estimator_hvps=30*4*(30+60+90+120),
                    hessian_reference_hvps=4254,
                    limitations=["Original overwritten CSV hashes unavailable; rerun recovery only.",
                                 "Timing columns are environment-dependent.",
                                 "Gated heuristic remains exploratory; no thresholds were retuned."])
    (RECOVERY / "recovery_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k:v for k,v in manifest.items() if not k.endswith("checksums")}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("snapshot", "haar", "hessian", "finalize"))
    args = parser.parse_args()
    RECOVERY.mkdir(parents=True, exist_ok=True)
    if args.stage == "finalize":
        finalize()
        return
    if args.stage == "snapshot":
        path = RECOVERY / "before_checksums.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(snapshot(), indent=2) + "\n")
        return
    import numpy as np
    import pandas as pd
    import torch
    started = datetime.now(timezone.utc).isoformat()
    out = RECOVERY / "staged"
    if args.stage == "haar":
        from experiments.run_haar_orientation_audit import run_haar_orientation_audit
        raw, summary = run_haar_orientation_audit(output_dir=out)
        assert len(raw) == 1860 and len(summary) == 12
        assert not raw.duplicated(["case_name", "orient_idx", "trial"]).any()
        assert (raw.query_count_per_method == 60).all()
    else:
        from experiments.run_pytorch_hessian_benchmark import run_pytorch_hessian_benchmark
        raw, summary = run_pytorch_hessian_benchmark(output_dir=out)
        assert len(raw) == 480 and len(summary) == 16
        assert not raw.duplicated(["budget_m", "algorithm", "trial"]).any()
        assert (raw.query_count == raw.budget_m).all()
        assert (raw.ground_truth_hvp_count == 4254).all()
    assert np.isfinite(raw.select_dtypes(include="number").to_numpy()).all()
    metadata = dict(stage=args.stage, started=started,
                    completed=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(), numpy=np.__version__,
                    pandas=pd.__version__, torch=torch.__version__,
                    torch_threads=torch.get_num_threads(), rows=len(raw),
                    summary_rows=len(summary), seed=2026,
                    recovery_type="deterministic protocol rerun; original CSV hashes unavailable",
                    files={p.name: digest(p) for p in sorted(out.glob(f"{args.stage if args.stage == 'haar' else 'pytorch'}*.csv"))})
    (RECOVERY / f"{args.stage}_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
