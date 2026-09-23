"""Validate and package the source manuscript; optionally archive/sync vault copies."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
PACKAGE = ROOT / "reports/latex/adaptive_hutchpp_20260917"
ZIP = ROOT / "reports/adaptive_hutchpp_latex_20260917.zip"
VAULT = Path("/Users/chenyixin/Documents/Obsidian_Vault/Swati_Summer_Research")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(log):
    old = json.loads((ROOT / "reports/theorem_report_preservation_20260917.json").read_text())
    assert all(sha(ROOT / name) == digest for name, digest in old.items())
    assembly = json.loads((PACKAGE / "assembly_manifest.json").read_text())
    assert sha(ROOT / assembly["source_report"]) == assembly["source_sha256"]
    assert sha(PACKAGE / "provenance/source_report.md") == assembly["source_sha256"]
    for name, digest in assembly["figures"].items():
        assert sha(PACKAGE / "figures" / name) == digest
    sources = [PACKAGE / "main.tex", *sorted((PACKAGE / "sections").glob("*.tex"))]
    body = "\n".join(p.read_text() for p in sources)
    labels = re.findall(r"\\label\{([^}]+)\}", body)
    refs = re.findall(r"\\ref\{([^}]+)\}", body)
    assert len(labels) == len(set(labels))
    assert set(refs) <= set(labels)
    expected = {f"thm:T{n}" for n in range(1, 16)} | {"thm:T5b", "cor:ideal"}
    assert expected <= set(labels)
    assert body.count(r"\begin{proof}") == 17
    assert len(re.findall(r"\\begin\{(?:theorem|lemma|proposition|corollary)\}", body)) == 17
    assert all(f"exp:E{n}" in labels for n in range(1, 7))
    for source in sources:
        stack = []
        for mode, env in re.findall(r"\\(begin|end)\{([^}]+)\}", source.read_text()):
            if mode == "begin":
                stack.append(env)
            else:
                assert stack and stack.pop() == env, (source, env)
        assert not stack, source
    for name in re.findall(r"\\input\{([^}]+)\}", body):
        assert (PACKAGE / (name + ".tex")).is_file()
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
        assert (PACKAGE / name).is_file()
    assert r"\Omega" not in body
    assert "^{,2}" not in body
    assert "LATEXMATH" not in body
    assert not list(PACKAGE.rglob("*.pdf"))
    log_text = log.read_text()
    for problem in ("undefined", "Overfull", "LaTeX Error", "Fatal error", "Rerun to"):
        assert problem not in log_text, problem
    assert "pdfdraftmode enabled" in log_text
    manifest = {
        "date": "2026-09-17",
        "writing_reference": "https://arxiv.org/abs/2010.09649v5",
        "protected_historical_files": len(old),
        "historical_files_unchanged": True,
        "original_report_unchanged": True,
        "formal_results": 17, "proofs": 17, "empirical_findings": 6,
        "figures": 6, "algorithms": 2,
        "compile_mode": "pdfLaTeX draftmode + BibTeX + resolving passes",
        "compile_log_sha256": sha(log),
        "undefined_references_or_citations": 0, "overfull_boxes": 0,
        "rendered_page_visual_review": False,
        "focused_math_and_figure_tests_passed": 25,
        "full_estimator_suite_rerun": False,
        "new_oracle_queries": 0, "new_bootstrap_samples": 0,
        "new_pdf": False,
        "files": {str(p.relative_to(PACKAGE)): sha(p)
                  for p in sorted(PACKAGE.rglob("*"))
                  if p.is_file() and p.name != "validation_manifest.json"},
    }
    (PACKAGE / "validation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(PACKAGE.rglob("*")):
            if p.is_file():
                archive.write(p, arcname=str(p.relative_to(PACKAGE)))
    with zipfile.ZipFile(ZIP) as archive:
        assert archive.testzip() is None
        assert "main.tex" in archive.namelist()
        assert not any(name.endswith(".pdf") for name in archive.namelist())
    print(json.dumps({k: manifest[k] for k in
                      ("protected_historical_files", "formal_results", "proofs",
                       "empirical_findings", "new_pdf")}))
    print(f"Source ZIP: {ZIP}")


def sync():
    sources = {
        "UROP_TRACKER.md": WORKSPACE / "UROP_TRACKER.md",
        "memory.md": WORKSPACE / "memory.md",
        "UROP_Research_Memory.md": WORKSPACE / "memory.md",
        "CURRENT_STATE.md": ROOT / "CURRENT_STATE.md",
        ZIP.name: ZIP,
    }
    for p in PACKAGE.rglob("*"):
        if p.is_file():
            sources["latex/adaptive_hutchpp_20260917/" + str(p.relative_to(PACKAGE))] = p
    archive = VAULT / "latex_manuscript_archive_20260917"
    archive.mkdir(exist_ok=False)
    records = []
    for name, source in sources.items():
        target = VAULT / name
        previous = sha(target) if target.exists() else None
        if target.exists():
            backup = archive / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        records.append({"destination": name, "old_sha256": previous, "sha256": sha(target)})
    payload = {"archive": str(archive), "files": records}
    (ROOT / "reports/latex_manuscript_sync_20260917.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Synced {len(records)} files; existing copies archived at {archive}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-log", type=Path, required=True)
    parser.add_argument("--sync", action="store_true")
    args = parser.parse_args()
    validate(args.compile_log)
    if args.sync:
        sync()
