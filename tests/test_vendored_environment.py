"""The development install uses a versioned, hashed local contracts wheel."""
from pathlib import Path
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def test_vendored_contracts_version_hash_and_no_private_git_install():
    wheel = ROOT / "vendor/vmodel_contracts-0.27.0-py3-none-any.whl"
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert digest == "ddbda9ddf6885c3b326e0777a22f785218ee919cc7f0b0cd599d01c429573bad"
    assert digest in (ROOT / "uv.lock").read_text()
    assert digest in (ROOT / "vendor/README.md").read_text()
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read("vmodel_contracts-0.27.0.dist-info/METADATA").decode()
    assert "Version: 0.27.0" in metadata
    assert "git+https://github.com/ShyAlon/vmodel-contracts" not in (ROOT / "pyproject.toml").read_text()


def test_test_runner_enforces_the_lock():
    runner = (ROOT / "scripts/test-local.sh").read_text()
    assert 'uv sync --locked --extra dev --python "$PYTHON"' in runner
    assert "pip install -e" not in runner
