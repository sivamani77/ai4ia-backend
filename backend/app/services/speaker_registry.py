"""Runtime access to config/speaker_registry.json, the canonical speaker list.

The registry is built and reviewed offline (scripts/build_speaker_registry.py)
and shipped in the image. Lookups here are deterministic only: exact
normalized match, then a unique initial-plus-surname match. Fuzzy matching
happens at build time, where a human reviews it.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.paths import CONFIG_DIR
from app.services.speaker_names import is_placeholder, norm_key
from app.services.video_naming import parse_source_video

logger = logging.getLogger("ai4ia.speaker_registry")

REGISTRY_PATH = CONFIG_DIR / "speaker_registry.json"
MULTIPLE_SPEAKERS = "Multiple speakers"
VIDEO_LEVEL = "video-level"


@dataclass(frozen=True)
class VideoSpeakers:
    kind: str  # "single" or "panel"
    speakers: tuple[tuple[str, str], ...]  # (name, role)
    source: str | None

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.speakers]


class SpeakerRegistry:
    def __init__(self, data: dict[str, Any]) -> None:
        self.version: str = data.get("version") or ""
        self.speakers: dict[str, dict[str, Any]] = data.get("speakers") or {}
        self._videos: dict[str, dict[str, Any]] = data.get("videos") or {}
        self._index: dict[str, str] = {}
        for canonical, entry in self.speakers.items():
            self._index[norm_key(canonical)] = canonical
            for alias in entry.get("aliases") or []:
                self._index.setdefault(norm_key(alias), canonical)
        self._by_surname: dict[str, list[str]] = {}
        for canonical in self.speakers:
            tokens = norm_key(canonical).split()
            if len(tokens) >= 2:
                self._by_surname.setdefault(tokens[-1], []).append(canonical)
        # Single-word aliases are left out: they match ordinary words.
        self._alias_lookup = {alias.lower(): canonical for alias, canonical in self.aliases().items()
                              if len(alias.split()) >= 2 and alias != canonical}
        self._alias_pattern = re.compile(
            r"\b(" + "|".join(re.escape(a) for a in sorted(self._alias_lookup, key=len, reverse=True)) + r")\b",
            re.IGNORECASE,
        ) if self._alias_lookup else None

    @classmethod
    def load(cls, path: Path = REGISTRY_PATH) -> "SpeakerRegistry":
        try:
            return cls(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            logger.warning("Speaker registry not found at %s; names pass through unchanged", path)
            return cls({})

    def aliases(self) -> dict[str, str]:
        """alias -> canonical, for every alias in the registry."""
        return {alias: canonical for canonical, entry in self.speakers.items()
                for alias in entry.get("aliases") or []}

    def canonical(self, name: str | None) -> str | None:
        """Canonical name for a known speaker, or None."""
        key = norm_key(name)
        if not key:
            return None
        if key in self._index:
            return self._index[key]
        tokens = key.split()
        if len(tokens) >= 2 and len(tokens[0]) == 1:
            hits = [c for c in self._by_surname.get(tokens[-1], [])
                    if norm_key(c).startswith(tokens[0])]
            if len(hits) == 1:
                return hits[0]
        return None

    def fix_names_in_text(self, text: str) -> str:
        """Replace multi-word alias spellings in free text with canonical names."""
        if not text or not self._alias_pattern:
            return text
        return self._alias_pattern.sub(lambda m: self._alias_lookup[m.group(0).lower()], text)

    def canonicalize(self, name: str | None) -> str | None:
        """Canonical name if known, the name unchanged if not, None for placeholders."""
        if name is None or is_placeholder(name):
            return None
        return self.canonical(name) or name

    def video_ids(self) -> list[str]:
        return list(self._videos)

    def video(self, video_id: str | None) -> VideoSpeakers | None:
        entry = self._videos.get(video_id or "")
        if not entry:
            return None
        speakers = tuple((s["name"], s.get("role") or "Speaker") for s in entry.get("speakers") or [])
        return VideoSpeakers(entry.get("kind") or "single", speakers, entry.get("source"))

    def speaker_fields(self, source_video: str | None, speaker: str | None,
                       attribution: str | None, video_id: str | None = None) -> dict[str, str]:
        """Correct speaker fields for one fragment. Videos in the registry take
        their roster from it; others keep their speaker, canonicalized."""
        parsed_id, _ = parse_source_video(source_video or "")
        video_id = parsed_id or video_id or ""
        entry = self.video(video_id)
        if entry is not None and entry.kind == "panel":
            return {"video_id": video_id, "speaker": "", "speaker_attribution": MULTIPLE_SPEAKERS,
                    "speaker_roster": "; ".join(entry.names)}
        if entry is not None and entry.speakers:
            name = entry.speakers[0][0]
            return {"video_id": video_id, "speaker": name, "speaker_attribution": VIDEO_LEVEL,
                    "speaker_roster": name}
        name = self.canonicalize(speaker) or ""
        if attribution == MULTIPLE_SPEAKERS:
            return {"video_id": video_id, "speaker": name, "speaker_attribution": attribution,
                    "speaker_roster": ""}
        return {"video_id": video_id, "speaker": name,
                "speaker_attribution": (attribution or VIDEO_LEVEL) if name else (attribution or ""),
                "speaker_roster": name}


@lru_cache(maxsize=1)
def get_registry() -> SpeakerRegistry:
    return SpeakerRegistry.load()
