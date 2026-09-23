"""Copy reviewed recovery records into the user-designated Obsidian directory.

Run only with filesystem approval for the destination. Existing vault versions
are copied into a dated archive first; no files are deleted.
"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
VAULT = Path('/Users/chenyixin/Documents/Obsidian_Vault/Swati_Summer_Research')
RECOVERY = ROOT / 'results/recovery_20260914'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert (RECOVERY / 'recovery_manifest.json').is_file(), 'Complete recovery first.'
    sources = {
        'UROP_TRACKER.md': WORKSPACE / 'UROP_TRACKER.md',
        'memory.md': WORKSPACE / 'memory.md',
        'UROP_Research_Memory.md': WORKSPACE / 'memory.md',
        'CURRENT_STATE.md': ROOT / 'CURRENT_STATE.md',
        'proof_structural_paired_difference_confidence.md': ROOT / 'docs/proof_structural_paired_difference_confidence.md',
        'README.md': RECOVERY / 'obsidian/README.md',
        'Hutchpp_Adaptive_Trace_Estimation.md': RECOVERY / 'obsidian/Hutchpp_Adaptive_Trace_Estimation.md',
    }
    for name in ('recovery_audit_20260914.md', 'structural_paired_difference_confidence_phase2c.md',
                 'haar_orientation_audit.md', 'pytorch_hessian_benchmark.md',
                 'gating_diagnostics_predictability_map.md'):
        sources[name] = ROOT / 'reports' / name
    archive = VAULT / 'recovery_archive_20260914'
    archive.mkdir(exist_ok=False)
    rows = []
    for name, source in sources.items():
        assert source.is_file()
        target = VAULT / name
        previous = None
        if target.exists():
            previous = sha(target)
            shutil.copy2(target, archive / name)
            assert sha(archive / name) == previous
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        rows.append(dict(name=name, before=previous, after=sha(target)))
    (RECOVERY / 'obsidian_sync_manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'Synchronized and hash-verified {len(rows)} records; previous vault versions archived.')


if __name__ == '__main__':
    main()
