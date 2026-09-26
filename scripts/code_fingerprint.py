#!/usr/bin/env python3
"""Map a job's `tree:<hash>` code version back to a git commit.

A deployed image has no .git, so jobs record a hash of the shipped code
(app/core/code_version.py). This recomputes that hash from git history.

Usage:
    python scripts/code_fingerprint.py                     # this checkout's working tree
    python scripts/code_fingerprint.py --rev origin/uat    # a commit
    python scripts/code_fingerprint.py --find tree:0123456789ab [--max 300]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.code_version import REPO_DIRS, fingerprint, is_shipped, shipped_files  # noqa: E402

PREFIXES = {f"{repo}/": f"{label}/" for label, repo in REPO_DIRS.items()}


def _git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True).stdout


def rev_fingerprint(rev: str) -> str:
    """Fingerprint of the shipped directories at a commit."""
    listing = _git("ls-tree", "-r", "-z", rev, "--", *REPO_DIRS.values())
    blobs: list[tuple[str, str]] = []
    for entry in filter(None, listing.split(b"\0")):
        meta, path = entry.decode().split("\t", 1)
        mode, kind, sha = meta.split()
        if kind != "blob" or mode == "120000":  # skip symlinks and submodules
            continue
        for repo_prefix, label_prefix in PREFIXES.items():
            if path.startswith(repo_prefix):
                rel = label_prefix + path[len(repo_prefix):]
                if is_shipped(rel):
                    blobs.append((rel, sha))
                break

    batch = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "--batch"],
        input="".join(f"{sha}\n" for _, sha in blobs).encode(),
        capture_output=True, check=True,
    ).stdout
    files, pos = [], 0
    for rel, _ in blobs:
        header_end = batch.index(b"\n", pos)
        size = int(batch[pos:header_end].split()[2])
        start = header_end + 1
        files.append((rel, batch[start:start + size]))
        pos = start + size + 1  # content is followed by a newline
    return fingerprint(files)


def find(target: str, limit: int) -> str | None:
    """Most recent commit whose fingerprint starts with `target`."""
    seen: set[bytes] = set()
    for rev in _git("rev-list", "--all", f"--max-count={limit}").decode().split():
        try:
            trees = _git("rev-parse", *(f"{rev}:{d}" for d in REPO_DIRS.values()))
        except subprocess.CalledProcessError:  # a directory missing at this commit
            trees = rev.encode()
        if trees in seen:  # same shipped code as a commit already checked
            continue
        seen.add(trees)
        if rev_fingerprint(rev).startswith(target):
            return rev
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rev", help="print the fingerprint of this commit")
    parser.add_argument("--find", help="tree:<hash> from a job record")
    parser.add_argument("--max", type=int, default=300, help="commits to search (default 300)")
    args = parser.parse_args()

    if args.find:
        target = args.find.removeprefix("tree:")
        rev = find(target, args.max)
        if rev is None:
            print(f"No commit in the last {args.max} matches tree:{target}", file=sys.stderr)
            return 1
        print(_git("log", "-1", "--format=%h %ad %s", "--date=short", rev).decode().strip())
        return 0

    value = rev_fingerprint(args.rev) if args.rev else fingerprint(shipped_files())
    print(f"tree:{value[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
