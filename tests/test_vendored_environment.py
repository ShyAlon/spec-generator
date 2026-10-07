"""The development install uses a versioned, hashed local contracts wheel."""
from pathlib import Path
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def test_vendored_contracts_version_hash_and_no_private_git_install():
    wheel = ROOT / "vendor/vmodel_contracts-0.27.2-py3-none-any.whl"
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert digest == "7081549ad8cb67f392d755f01fc9342bfa2223b369ba0dc3c393df9cd3e2a98d"
    assert digest in (ROOT / "uv.lock").read_text()
    assert digest in (ROOT / "vendor/README.md").read_text()
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read("vmodel_contracts-0.27.2.dist-info/METADATA").decode()
    assert "Version: 0.27.2" in metadata
    assert "git+https://github.com/ShyAlon/vmodel-contracts" not in (ROOT / "pyproject.toml").read_text()


def test_test_runner_enforces_the_lock():
    runner = (ROOT / "scripts/test-local.sh").read_text()
    assert 'uv sync --locked --extra dev --python "$PYTHON"' in runner
    assert "pip install -e" not in runner
