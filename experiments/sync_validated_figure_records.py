"""Recoverable, scoped synchronization of reviewed figure notes and assets."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
VAULT = Path('/Users/chenyixin/Documents/Obsidian_Vault/Swati_Summer_Research')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    sources = {
        'UROP_TRACKER.md': WORKSPACE/'UROP_TRACKER.md',
        'memory.md': WORKSPACE/'memory.md',
        'UROP_Research_Memory.md': WORKSPACE/'memory.md',
        'CURRENT_STATE.md': ROOT/'CURRENT_STATE.md',
        'urop_research_report_20260916.md': ROOT/'reports/urop_research_report_20260916.md',
        'urop_figure_guide_20260916.md': ROOT/'reports/urop_figure_guide_20260916.md',
        'docs/figure_plan_20260916.md': ROOT/'docs/figure_plan_20260916.md',
    }
    figures = ROOT/'figures/urop_validated_20260916'
    manifest = json.loads((figures/'manifest.json').read_text())
    for name, checksum in manifest['output_sha256'].items():
        assert digest((figures/name).read_bytes()) == checksum
    for name, checksum in manifest['source_sha256'].items():
        assert digest((ROOT/'results'/name).read_bytes()) == checksum
    for path in figures.iterdir():
        if path.is_file():
            sources[str(path.relative_to(ROOT))] = path
    assert len(sources) == 22 and all(p.is_file() for p in sources.values())
    archive = VAULT/'figure_package_archive_20260916'
    archive.mkdir(exist_ok=False)
    rows = []
    for name, source in sources.items():
        target = VAULT/name
        previous = None
        if target.exists():
            previous = digest(target.read_bytes())
            archived = archive/name
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, archived)
            assert digest(archived.read_bytes()) == previous
        original = source.read_bytes()
        payload = original
        transformed = False
        if name in ('urop_research_report_20260916.md', 'urop_figure_guide_20260916.md'):
            text = original.decode()
            text = text.replace('../figures/urop_validated_20260916/', 'figures/urop_validated_20260916/')
            text = text.replace('../docs/figure_plan_20260916.md', 'docs/figure_plan_20260916.md')
            payload = text.encode()
            transformed = payload != original
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        assert target.read_bytes() == payload
        rows.append(dict(name=name, previous_sha256=previous,
                         source_sha256=digest(original), current_sha256=digest(payload),
                         figure_links_rebased_for_flat_vault=transformed))
    (ROOT/'reports/urop_figure_sync_20260916.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('22 notes/assets verified; prior notes archived; new image links rebased for the vault.')


if __name__ == '__main__':
    main()
