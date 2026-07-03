"""Loads and validates the theme/chapter taxonomy from config/taxonomy.json.

Single source of truth for the hierarchy, shared by scripts/05_classify_fragments.py
(classification keywords) and backend/app/core/taxonomy.py (API-facing display data).
Fails fast at import time if the file is missing, malformed, or internally
inconsistent (dangling theme_id, chapter listed under the wrong theme, etc.).
"""
from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from app.core.settings import settings

CONFIG_PATH = settings.project_root / "config" / "taxonomy.json"


class ThemeModel(BaseModel):
    name: str
    chapters: list[str]
    keywords: list[str]


class ChapterModel(BaseModel):
    theme_id: str
    name: str
    keywords: list[str]


class TaxonomyModel(BaseModel):
    themes: dict[str, ThemeModel]
    chapters: dict[str, ChapterModel]


def _load() -> TaxonomyModel:
    if not CONFIG_PATH.exists():
        raise ValueError(f"Taxonomy config not found: {CONFIG_PATH}")

    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Taxonomy config is not valid JSON: {CONFIG_PATH}") from exc

    try:
        taxonomy = TaxonomyModel.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Taxonomy config failed schema validation: {exc}") from exc

    for cid, chapter in taxonomy.chapters.items():
        if chapter.theme_id not in taxonomy.themes:
            raise ValueError(
                f"Taxonomy config inconsistent: {cid} references unknown theme_id {chapter.theme_id}"
            )

    referenced: dict[str, int] = {}
    for tid, theme in taxonomy.themes.items():
        for cid in theme.chapters:
            if cid not in taxonomy.chapters:
                raise ValueError(
                    f"Taxonomy config inconsistent: theme {tid} lists unknown chapter {cid}"
                )
            if taxonomy.chapters[cid].theme_id != tid:
                raise ValueError(
                    f"Taxonomy config inconsistent: {cid} listed under {tid} but "
                    f"has theme_id={taxonomy.chapters[cid].theme_id}"
                )
            referenced[cid] = referenced.get(cid, 0) + 1

    orphans = set(taxonomy.chapters) - set(referenced)
    if orphans:
        raise ValueError(f"Taxonomy config inconsistent: orphan chapters not listed under any theme: {orphans}")
    duplicates = {cid: n for cid, n in referenced.items() if n > 1}
    if duplicates:
        raise ValueError(f"Taxonomy config inconsistent: chapters listed under multiple themes: {duplicates}")

    return taxonomy


_TAXONOMY = _load()

THEMES: dict[str, dict] = {
    tid: theme.model_dump() for tid, theme in _TAXONOMY.themes.items()
}
CHAPTERS: dict[str, dict] = {
    cid: chapter.model_dump() for cid, chapter in _TAXONOMY.chapters.items()
}
