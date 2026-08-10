from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Any

from app.core.paths import PROJECT_ROOT

RAW_VIDEO_DIR = Path(os.environ.get("AI4IA_RAW_VIDEO_DIR", str(PROJECT_ROOT / "data" / "raw" / "youtube" / "ai4ia_channel")))


def slugify(text: str, max_len: int = 80) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "video"


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}. Run scripts/08_generate_assets.py first.")

    return json.loads(path.read_text(encoding="utf-8"))


_VIDEO_ID_PREFIX_RE = re.compile(r"^(\d{4}_[A-Za-z0-9_-]{6,})")


def _video_id_prefix(name: str) -> str | None:
    """Stable '0254_RXXzSGoo0KQ' id prefix — see stage_09_export._video_id_prefix
    for why title text (source_video/fragment_id suffix) can drift from the real
    filename (LLM round-trips normalize fullwidth Unicode punctuation in titles)
    while this id prefix never does."""
    m = _VIDEO_ID_PREFIX_RE.match(name)
    return m.group(1) if m else None


def find_source_video(source_video: str | None, fragment_id: str | None = None) -> Path | None:
    candidates = [c for c in (source_video, fragment_id and re.sub(r"_\d{4}$", "", fragment_id)) if c]

    for name in candidates:
        exact = RAW_VIDEO_DIR / f"{name}.mp4"
        if exact.exists():
            return exact
        matches = sorted(RAW_VIDEO_DIR.glob(f"{name}*.mp4"))
        if matches:
            return matches[0]

    # Fallback: match on the stable numeric/video-id prefix alone, which
    # survives title text that was normalized somewhere upstream.
    for name in candidates:
        prefix = _video_id_prefix(name)
        if not prefix:
            continue
        for path in RAW_VIDEO_DIR.glob(f"{prefix}*.mp4"):
            return path

    return None


def get_story(manifest: dict[str, Any]) -> dict[str, Any]:
    embedded_storyline = manifest.get("storyline")
    if isinstance(embedded_storyline, dict):
        return embedded_storyline

    storyline_path = Path(manifest["storyline_path"])
    payload = json.loads(storyline_path.read_text(encoding="utf-8"))
    return payload["storyline"]


def section_voiceover_map(manifest: dict[str, Any]) -> dict[str, Path]:
    mapping = {}

    voice = manifest.get("voice", {})
    for item in voice.get("section_voiceovers", []):
        section_id = item.get("section_id")
        audio_path = item.get("audio_path")

        if section_id and audio_path:
            path = Path(audio_path)
            if path.exists():
                mapping[section_id] = path

    return mapping


def section_voiceover_by_type_and_index(manifest: dict[str, Any]) -> dict[str, Path]:
    """
    Extended voiceover lookup: keys are section_id, section_type, section_role, AND
    positional index (str(0), str(1), ...) so that canonical timeline items with only
    section_role can still be matched.
    """
    mapping: dict[str, Path] = {}
    voice = manifest.get("voice", {})
    svlist = voice.get("section_voiceovers", [])

    for pos_idx, item in enumerate(svlist):
        audio_path = item.get("audio_path")
        if not audio_path:
            continue
        path = Path(audio_path)
        if not path.exists():
            continue
        section_id = item.get("section_id", "")
        section_type = item.get("section_type", "")
        section_role = item.get("section_role", "")
        if section_id:
            mapping[section_id] = path
        if section_type:
            mapping[section_type] = path
        if section_role:
            mapping[section_role] = path
        # positional index (0-based): canonical timeline fragment items are also 0-based
        mapping[str(pos_idx)] = path

    return mapping


def section_card_map(manifest: dict[str, Any]) -> dict[str, Path]:
    cards = manifest.get("cards", {})
    mapping = {}

    for item in cards.get("section_cards", []):
        section_id = item.get("section_id")
        path = item.get("path")

        if section_id and path:
            card_path = Path(path)
            if card_path.exists():
                mapping[section_id] = card_path

    return mapping


def source_video_lookup(manifest: dict[str, Any]) -> dict[str, str]:
    lookup = {}

    for item in manifest.get("archive_fragments", []):
        fragment_id = item.get("fragment_id")
        source_video = item.get("source_video")

        if fragment_id and source_video:
            lookup[fragment_id] = source_video

    return lookup
