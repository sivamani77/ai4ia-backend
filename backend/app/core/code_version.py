"""Identify the code a process is running, so jobs can record it.

The image ships without .git, so outside a checkout the version is a hash of
the shipped code. scripts/code_fingerprint.py maps that hash back to a commit.
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from app.core.paths import CONFIG_DIR, PROJECT_ROOT, SCRIPTS_DIR
from app.core.settings import settings

logger = logging.getLogger("ai4ia.code_version")

APP_DIR = Path(__file__).resolve().parents[1]

# Where each shipped directory lives in the repo. The image copies them to
# /app/app, /app/scripts and /app/config.
REPO_DIRS = {"app": "backend/app", "scripts": "scripts", "config": "config"}


def is_shipped(rel_path: str) -> bool:
    """False for build artefacts that are never part of a commit."""
    parts = rel_path.split("/")
    return "__pycache__" not in parts and not rel_path.endswith((".pyc", ".pyo"))


def fingerprint(files: Iterable[tuple[str, bytes]]) -> str:
    """Order-independent hash of (path, content) pairs."""
    digest = hashlib.sha256()
    for path, content in sorted((p, hashlib.sha256(c).hexdigest()) for p, c in files):
        digest.update(f"{path}\0{content}\n".encode())
    return digest.hexdigest()


def shipped_files() -> Iterable[tuple[str, bytes]]:
    roots = {"app": APP_DIR, "scripts": SCRIPTS_DIR, "config": CONFIG_DIR}
    for label, root in roots.items():
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            rel = f"{label}/{path.relative_to(root).as_posix()}"
            if path.is_file() and is_shipped(rel):
                yield rel, path.read_bytes()


def _git_version(root: Path) -> str | None:
    """`git:<sha>`, with `-dirty` if shipped files differ from HEAD."""
    if not (root / ".git").exists():
        return None
    try:
        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip()

        sha = git("rev-parse", "--short=12", "HEAD")
        dirty = git("status", "--porcelain", "--", *REPO_DIRS.values())
    except (OSError, subprocess.SubprocessError):
        return None
    return f"git:{sha}-dirty" if dirty else f"git:{sha}"


@lru_cache(maxsize=1)
def code_version() -> str:
    """CODE_VERSION if set, else the git commit, else `tree:<hash>`."""
    if settings.code_version:
        return settings.code_version
    version = _git_version(PROJECT_ROOT)
    if version:
        return version
    try:
        return f"tree:{fingerprint(shipped_files())[:12]}"
    except OSError:
        logger.warning("Could not fingerprint the shipped code", exc_info=True)
        return "unknown"
