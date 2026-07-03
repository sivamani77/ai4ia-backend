"""Theme/chapter taxonomy — single source of truth for the hierarchy.

Loaded from config/taxonomy.json via taxonomy_loader (also consumed by
scripts/05_classify_fragments.py for classification keywords). Themes are the
top level (T1-T7); each chapter (Ch 1-Ch 25) belongs to exactly one theme via
`theme_id`. The frontend uses this to render a nested theme -> chapters tree.
"""

from app.core import taxonomy_loader

THEMES: dict[str, dict] = {
    tid: {"name": theme["name"], "chapters": theme["chapters"]}
    for tid, theme in taxonomy_loader.THEMES.items()
}

# chapter_id -> (theme_id, display name), display name prefixed with the
# chapter id (e.g. "Ch 1: Foundations of AI Ethics & Philosophy") to match
# the live classified data.
CHAPTERS: dict[str, dict] = {
    cid: {"theme_id": chapter["theme_id"], "name": f"{cid}: {chapter['name']}"}
    for cid, chapter in taxonomy_loader.CHAPTERS.items()
}


def chapter_id_from_name(name: str) -> str | None:
    """'Ch 12: Youth...' or 'Ch 12' -> 'Ch 12'."""
    if not name:
        return None
    head = name.split(":", 1)[0].strip()
    return head if head in CHAPTERS else None


def theme_id_from_name(name: str) -> str | None:
    if not name:
        return None
    for tid, t in THEMES.items():
        if t["name"] == name:
            return tid
    return None
