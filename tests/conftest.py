"""Keep maintained research artifacts immutable during the test session."""
from pathlib import Path
import hashlib
import pytest


@pytest.fixture(scope="session", autouse=True)
def protect_research_results():
    root = Path(__file__).resolve().parents[1] / "results"

    def snapshot():
        hashes = {}
        for path in sorted(root.rglob("*")):
            # Recovery staging is an explicitly separate, concurrently running job.
            if not path.is_file() or "recovery_20260914" in path.parts:
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[str(path.relative_to(root))] = digest.hexdigest()
        return hashes

    before = snapshot()
    yield
    after = snapshot()
    changed = sorted(key for key in before.keys() | after.keys()
                     if before.get(key) != after.get(key))
    assert not changed, f"Tests changed production research artifacts: {changed}"
