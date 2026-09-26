"""Directory contracts between consecutive ingest pipeline scripts.

The ingest runner invokes scripts/01-06 back-to-back as subprocesses, and each
stage hands off to the next through a local directory. Nothing at runtime
checks that the producer's output directory is the consumer's input directory,
so a drifting path silently breaks the chain: commit bb24260 repointed
05_classify_fragments at data/fragments_cleaned (a directory only the separate
manual reprocess flow writes) without mentioning it, and from then on every
fragment 04_create_fragments produced was a dead end.

The directory constants are read out of each script's AST rather than by
importing it. Importing would drag in heavy optional deps (03_transcribe needs
faster_whisper) and would run each script's module-level mkdir of its own
output directory.
"""
from __future__ import annotations

import ast
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _flatten_chain(node: ast.AST) -> tuple[str, ...] | None:
    """Turn a `PROJECT_ROOT / "a" / "b"` AST node into ("a", "b")."""
    parts: list[str] = []
    current = node
    while isinstance(current, ast.BinOp) and isinstance(current.op, ast.Div):
        if not isinstance(current.right, ast.Constant) or not isinstance(current.right.value, str):
            return None
        parts.append(current.right.value)
        current = current.left
    if isinstance(current, ast.Name) and current.id == "PROJECT_ROOT":
        return tuple(reversed(parts))
    return None


def _path_under_project_root(filename: str, const_name: str) -> tuple[str, ...]:
    """The path components of a module-level `X = PROJECT_ROOT / ...` constant.

    Tolerates the constant being wrapped, as 03_transcribe's AUDIO_DIR is in an
    os.getenv default — the first PROJECT_ROOT-rooted chain in the assignment
    is the one that matters.
    """
    tree = ast.parse((SCRIPTS_DIR / filename).read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == const_name for t in node.targets):
            continue
        for sub in ast.walk(node.value):
            chain = _flatten_chain(sub)
            if chain:
                return chain
    raise AssertionError(f"{filename}: no PROJECT_ROOT-rooted {const_name} found")


class TestIngestStageHandoffs:
    def test_extract_audio_writes_where_transcribe_reads(self):
        assert _path_under_project_root("02_extract_audio.py", "AUDIO_DIR") == \
               _path_under_project_root("03_transcribe.py", "AUDIO_DIR")

    def test_transcribe_writes_where_create_fragments_reads(self):
        assert _path_under_project_root("03_transcribe.py", "TRANSCRIPT_DIR") == \
               _path_under_project_root("04_create_fragments.py", "TRANSCRIPT_DIR")

    def test_create_fragments_writes_where_classify_reads(self):
        """The bb24260 regression: 04 wrote data/fragments, 05 read
        data/fragments_cleaned, so classify never saw fresh fragments."""
        assert _path_under_project_root("04_create_fragments.py", "FRAGMENT_DIR") == \
               _path_under_project_root("05_classify_fragments.py", "FRAGMENT_DIR")

    def test_classify_writes_where_embed_reads(self):
        assert _path_under_project_root("05_classify_fragments.py", "CLASSIFIED_DIR") == \
               _path_under_project_root("06_generate_embeddings.py", "CLASSIFIED_DIR")

    def test_download_writes_where_extract_audio_reads(self):
        assert _path_under_project_root("01_download_videos.py", "RAW_DIR") == \
               _path_under_project_root("02_extract_audio.py", "VIDEO_DIR")


class TestReprocessFlowStaysSeparate:
    def test_regenerate_writes_its_own_directory(self):
        """The manual reprocess flow (reprocess_audio_transcripts.py) owns
        data/fragments_cleaned and must not collide with the ingest chain —
        it feeds 06 directly and never invokes 05."""
        regenerate = _path_under_project_root(
            "04_regenerate_fragments_from_clean_transcripts.py", "FRAGMENT_DIR"
        )
        assert regenerate != _path_under_project_root("04_create_fragments.py", "FRAGMENT_DIR")
        assert regenerate[-1] == "fragments_cleaned"
