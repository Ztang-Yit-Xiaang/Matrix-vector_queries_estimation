"""Archive prior Obsidian versions, then synchronize the reviewed audit records.

Requires write approval for the explicitly user-designated vault. No deletion.
"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
VAULT = Path('/Users/chenyixin/Documents/Obsidian_Vault/Swati_Summer_Research')
AUDIT = ROOT / 'results/coordinate_gate_failure_audit_20260916'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert (AUDIT/'manifest.json').is_file(), 'Complete the audit before syncing.'
    sources = {
        'UROP_TRACKER.md': WORKSPACE/'UROP_TRACKER.md',
        'memory.md': WORKSPACE/'memory.md',
        'UROP_Research_Memory.md': WORKSPACE/'memory.md',
        'CURRENT_STATE.md': ROOT/'CURRENT_STATE.md',
        'coordinate_gate_failure_audit_20260916.md': ROOT/'reports/coordinate_gate_failure_audit_20260916.md',
    }
    assert all(p.is_file() for p in sources.values())
    archive = VAULT/'coordinate_gate_audit_archive_20260916'
    archive.mkdir(exist_ok=False)
    rows = []
    for name, source in sources.items():
        target = VAULT/name
        previous = None
        if target.exists():
            previous = sha(target)
            shutil.copy2(target, archive/name)
            assert sha(archive/name) == previous
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        rows.append(dict(name=name, previous_sha256=previous, current_sha256=sha(target)))
    (AUDIT/'obsidian_sync_manifest.json').write_text(json.dumps(rows, indent=2)+'\n')
    print('Five records synchronized and hash-verified; previous versions archived.')


if __name__ == '__main__':
    main()
