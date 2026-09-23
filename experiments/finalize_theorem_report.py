"""Validate the editorial package; optionally archive and sync selected vault notes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
VAULT = Path('/Users/chenyixin/Documents/Obsidian_Vault/Swati_Summer_Research')
REPORT = ROOT / 'reports/urop_theorem_proof_report_20260917.md'
PACKAGE = ROOT / 'figures/urop_structure_20260917'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate():
    frozen = json.loads((ROOT / 'reports/theorem_report_preservation_20260917.json').read_text())
    changed = [name for name, digest in frozen.items()
               if not (ROOT/name).is_file() or sha(ROOT/name) != digest]
    assert not changed, f'Protected artifacts changed: {changed}'
    text = REPORT.read_text()
    links = re.findall(r'\]\(([^)]+)\)', text)
    local_links = [x for x in links if not x.startswith('http')]
    assert all((REPORT.parent/x.split('#')[0]).is_file() for x in local_links)
    assert text.count('$$') % 2 == 0
    assert re.findall(r'^### T(\d+)\.', text, re.M) == [str(x) for x in range(1, 16)]
    assert len(re.findall(r'^### Empirical finding E\d\.', text, re.M)) == 6
    assert len(re.findall(r'!\[', text)) == 6
    assert r'\Omega' not in text
    images = []
    for name in ('workflow', 'dependencies', 'models'):
        path = PACKAGE/f'{name}.png'
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            width, height = im.size
        assert width >= 1500 and height >= 800
        images.append(dict(file=path.name, width=width, height=height, sha256=sha(path)))
    outputs = [REPORT, ROOT/'reports/theorem_report_audit_20260917.md',
               ROOT/'tests/test_theorem_report_identities.py',
               PACKAGE/'prompts_and_contract.md', Path(__file__).resolve()]
    manifest = dict(date='2026-09-17', protected_files=len(frozen), protected_files_unchanged=True,
                    new_oracle_queries=0, new_bootstrap_samples=0, new_pdf=False,
                    full_algorithm_suite_rerun=False, focused_math_cases=20,
                    existing_figure_tests=5, words=len(text.split()), local_links=len(local_links),
                    images=images, files={str(p.relative_to(ROOT)): sha(p) for p in outputs})
    (PACKAGE/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ('protected_files', 'words', 'local_links', 'new_oracle_queries')}))
    return manifest


def sync():
    sources = {
        'UROP_TRACKER.md': WORKSPACE/'UROP_TRACKER.md',
        'memory.md': WORKSPACE/'memory.md',
        'UROP_Research_Memory.md': WORKSPACE/'memory.md',
        'CURRENT_STATE.md': ROOT/'CURRENT_STATE.md',
        REPORT.name: REPORT,
        'theorem_report_audit_20260917.md': ROOT/'reports/theorem_report_audit_20260917.md',
    }
    for path in PACKAGE.iterdir():
        if path.is_file():
            sources[str(path.relative_to(ROOT))] = path
    payloads = {}
    for name, source in sources.items():
        payload = source.read_bytes()
        if source == REPORT:
            body = payload.decode()
            body = body.replace('../figures/', 'figures/').replace('../docs/', '')
            body = body.replace('../../../UROP_TRACKER.md', 'UROP_TRACKER.md')
            body = body.replace('../../../memory.md', 'memory.md')
            payload = body.encode()
        elif name == 'theorem_report_audit_20260917.md':
            payload = payload.decode().replace('../figures/', 'figures/').encode()
        payloads[name] = payload
    # Validate direct report links against existing or planned vault destinations.
    for link in re.findall(r'\]\(([^)]+)\)', payloads[REPORT.name].decode()):
        if not link.startswith('http'):
            target = link.split('#')[0]
            assert target in sources or (VAULT/target).is_file(), target
    archive = VAULT/'theorem_report_archive_20260917'
    # Refuse accidental overwrite of an earlier archive.
    archive.mkdir(exist_ok=False)
    records = []
    for name, source in sources.items():
        target = VAULT/name
        old = None
        if target.exists():
            old = sha(target)
            backup = archive/name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            assert sha(backup) == old
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[name])
        assert target.read_bytes() == payloads[name]
        records.append(dict(file=name, source_sha256=sha(source), vault_sha256=sha(target),
                            previous_sha256=old, link_rebased=payloads[name] != source.read_bytes()))
    (ROOT/'reports/theorem_report_sync_20260917.json').write_text(json.dumps(records, indent=2)+'\n')
    print(f'{len(records)} notes/assets synchronized; previous notes archived at {archive}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sync', action='store_true')
    args = parser.parse_args()
    validate()
    if args.sync:
        sync()
