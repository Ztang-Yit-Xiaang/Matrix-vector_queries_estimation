"""Archive old vault notes and synchronize the reviewed September report.

Document-copy utility only: no estimators, experiments, or result files touched.
Requires approval for the explicitly user-designated Obsidian vault.
"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
VAULT = Path('/Users/chenyixin/Documents/Obsidian_Vault/Swati_Summer_Research')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = {
        'UROP_TRACKER.md': WORKSPACE/'UROP_TRACKER.md',
        'memory.md': WORKSPACE/'memory.md',
        'UROP_Research_Memory.md': WORKSPACE/'memory.md',
        'CURRENT_STATE.md': ROOT/'CURRENT_STATE.md',
    }
    for name in ('urop_research_report_20260916.md', 'urop_claims_register_20260916.md',
                 'urop_research_progress_report_aug2026.md'):
        sources[name] = ROOT/'reports'/name
    assert all(source.is_file() for source in sources.values())
    archive = VAULT/'report_consolidation_archive_20260916'
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
    manifest = ROOT/'reports/urop_report_consolidation_sync_20260916.json'
    manifest.write_text(json.dumps(rows, indent=2)+'\n')
    print('Seven notes synchronized and hash-verified; previous vault versions archived.')


if __name__ == '__main__':
    main()
