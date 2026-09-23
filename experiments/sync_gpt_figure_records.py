"""Archive prior notes and sync separate GPT presentation assets, without replacing plots."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
VAULT = Path('/Users/chenyixin/Documents/Obsidian_Vault/Swati_Summer_Research')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    sources = {
        'UROP_TRACKER.md': WORKSPACE/'UROP_TRACKER.md',
        'memory.md': WORKSPACE/'memory.md',
        'UROP_Research_Memory.md': WORKSPACE/'memory.md',
        'CURRENT_STATE.md': ROOT/'CURRENT_STATE.md',
    }
    for name in ('urop_figure_guide_20260916.md', 'urop_research_report_20260916.md',
                 'urop_gpt_polished_figures_20260916.md'):
        sources[name] = ROOT/'reports'/name
    package = ROOT/'figures/urop_gpt_polished_20260916'
    manifest = json.loads((package/'manifest.json').read_text())
    for row in manifest['images']:
        assert sha((package/row['file']).read_bytes()) == row['sha256']
    assert sha((package/'prompts.md').read_bytes()) == manifest['prompts_sha256']
    for p in package.iterdir():
        if p.is_file():
            sources[str(p.relative_to(ROOT))] = p
    assert len(sources) == 12
    archive = VAULT/'gpt_figure_archive_20260916'
    archive.mkdir(exist_ok=False)
    records = []
    for name, source in sources.items():
        target = VAULT/name
        previous = None
        if target.exists():
            previous = sha(target.read_bytes())
            backup = archive/name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            assert sha(backup.read_bytes()) == previous
        original = source.read_bytes()
        payload = original
        if name.endswith('.md') and '/' not in name:
            text = original.decode()
            for folder in ('urop_validated_20260916', 'urop_gpt_polished_20260916'):
                text = text.replace('../figures/'+folder+'/', 'figures/'+folder+'/')
            text = text.replace('../docs/figure_plan_20260916.md', 'docs/figure_plan_20260916.md')
            payload = text.encode()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        assert target.read_bytes() == payload
        records.append(dict(file=name, source_sha256=sha(original), vault_sha256=sha(payload),
                            previous_sha256=previous, relative_links_adjusted=payload != original))
    (ROOT/'reports/urop_gpt_figure_sync_20260916.json').write_text(json.dumps(records,indent=2)+'\n')
    print('12 notes/assets synchronized; previous notes archived; original validated plots not touched.')


if __name__ == '__main__':
    main()
