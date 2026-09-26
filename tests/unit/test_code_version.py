"""Code version stamping and the fingerprint-to-commit lookup."""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

from app.core import code_version as cv

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location("code_fingerprint", REPO_ROOT / "scripts" / "code_fingerprint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repo laid out like this one, with the module pointed at it."""
    files = {
        "backend/app/main.py": "print('app')\n",
        "backend/app/core/x.py": "X = 1\n",
        "scripts/run.py": "#!/usr/bin/env python3\n",
        "config/registry.json": '{"v": 1}',
        "README.md": "not shipped\n",
    }
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")

    monkeypatch.setattr(cv, "APP_DIR", tmp_path / "backend" / "app")
    monkeypatch.setattr(cv, "SCRIPTS_DIR", tmp_path / "scripts")
    monkeypatch.setattr(cv, "CONFIG_DIR", tmp_path / "config")
    return tmp_path


class TestFingerprint:
    def test_ignores_order(self):
        files = [("app/a.py", b"a"), ("app/b.py", b"b")]
        assert cv.fingerprint(files) == cv.fingerprint(list(reversed(files)))

    def test_changes_with_content(self):
        assert cv.fingerprint([("app/a.py", b"a")]) != cv.fingerprint([("app/a.py", b"b")])

    def test_changes_with_path(self):
        assert cv.fingerprint([("app/a.py", b"a")]) != cv.fingerprint([("app/b.py", b"a")])

    @pytest.mark.parametrize("path,shipped", [
        ("app/main.py", True),
        ("config/speaker_registry.json", True),
        ("app/__pycache__/main.cpython-312.pyc", False),
        ("scripts/old.pyc", False),
    ])
    def test_is_shipped(self, path, shipped):
        assert cv.is_shipped(path) is shipped


class TestCodeVersion:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        cv.code_version.cache_clear()
        yield
        cv.code_version.cache_clear()

    def test_setting_wins(self, monkeypatch):
        monkeypatch.setattr(cv.settings, "code_version", "img-2026.09.23")
        assert cv.code_version() == "img-2026.09.23"

    def test_git_commit_in_a_checkout(self, repo):
        assert cv._git_version(repo) == f"git:{_git(repo, 'rev-parse', '--short=12', 'HEAD')}"

    def test_dirty_only_for_shipped_changes(self, repo):
        (repo / "README.md").write_text("edited\n")
        assert not cv._git_version(repo).endswith("-dirty")
        (repo / "config" / "registry.json").write_text('{"v": 2}')
        assert cv._git_version(repo).endswith("-dirty")

    def test_no_git_is_none(self, tmp_path):
        assert cv._git_version(tmp_path) is None

    def test_tree_hash_without_git(self, repo, monkeypatch):
        monkeypatch.setattr(cv.settings, "code_version", "")
        monkeypatch.setattr(cv, "PROJECT_ROOT", repo / "no-git-here")
        assert cv.code_version() == f"tree:{cv.fingerprint(cv.shipped_files())[:12]}"


class TestLookupScript:
    def test_commit_matches_the_shipped_tree(self, repo, monkeypatch):
        script = _load_script()
        monkeypatch.setattr(script, "ROOT", repo)
        (repo / "backend" / "app" / "__pycache__").mkdir()
        (repo / "backend" / "app" / "__pycache__" / "main.pyc").write_bytes(b"\0")

        assert script.rev_fingerprint("HEAD") == cv.fingerprint(cv.shipped_files())

    def test_find_returns_the_matching_commit(self, repo, monkeypatch):
        script = _load_script()
        monkeypatch.setattr(script, "ROOT", repo)
        first = _git(repo, "rev-parse", "HEAD")
        target = cv.fingerprint(cv.shipped_files())[:12]

        (repo / "scripts" / "run.py").write_text("changed\n")
        _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "change")

        assert script.find(target, limit=10) == first
        assert script.find("000000000000", limit=10) is None
