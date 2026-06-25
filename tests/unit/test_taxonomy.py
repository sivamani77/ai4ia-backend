"""Unit tests for app.core.taxonomy — THEMES, CHAPTERS, helper functions."""
from __future__ import annotations

import pytest

from app.core.taxonomy import (
    CHAPTERS,
    THEMES,
    chapter_id_from_name,
    theme_id_from_name,
)

# ---------------------------------------------------------------------------
# Structure invariants
# ---------------------------------------------------------------------------

class TestStructure:
    def test_seven_themes(self):
        assert len(THEMES) == 7

    def test_twenty_five_chapters(self):
        assert len(CHAPTERS) == 25

    def test_theme_ids_sequential(self):
        assert set(THEMES.keys()) == {"T1", "T2", "T3", "T4", "T5", "T6", "T7"}

    def test_chapter_ids_sequential(self):
        expected = {f"Ch {i}" for i in range(1, 26)}
        assert set(CHAPTERS.keys()) == expected

    def test_every_theme_has_name_and_chapters(self):
        for tid, theme in THEMES.items():
            assert "name" in theme, f"{tid} missing 'name'"
            assert isinstance(theme["chapters"], list), f"{tid} 'chapters' not a list"
            assert len(theme["chapters"]) > 0, f"{tid} has no chapters"

    def test_every_chapter_has_name_and_theme_id(self):
        for cid, ch in CHAPTERS.items():
            assert "name" in ch, f"{cid} missing 'name'"
            assert "theme_id" in ch, f"{cid} missing 'theme_id'"

    def test_all_chapters_reference_valid_theme(self):
        for cid, ch in CHAPTERS.items():
            assert ch["theme_id"] in THEMES, f"{cid} references unknown theme {ch['theme_id']}"

    def test_chapters_in_themes_match_chapters_dict(self):
        """Every chapter listed under THEMES also exists in CHAPTERS."""
        all_chapter_refs: list[str] = []
        for theme in THEMES.values():
            all_chapter_refs.extend(theme["chapters"])
        assert set(all_chapter_refs) == set(CHAPTERS.keys())

    def test_no_orphan_chapters(self):
        """Every chapter in CHAPTERS is referenced by exactly one theme."""
        refs: dict[str, int] = {}
        for theme in THEMES.values():
            for cid in theme["chapters"]:
                refs[cid] = refs.get(cid, 0) + 1
        for cid, count in refs.items():
            assert count == 1, f"{cid} appears {count} times across themes"

    def test_chapter_names_start_with_id(self):
        """Chapter names follow the 'Ch N: ...' convention."""
        for cid, ch in CHAPTERS.items():
            assert ch["name"].startswith(cid), (
                f"{cid} name '{ch['name']}' does not start with its ID"
            )

    def test_theme_chapter_back_reference(self):
        """CHAPTERS[cid]['theme_id'] matches the theme that lists cid."""
        for tid, theme in THEMES.items():
            for cid in theme["chapters"]:
                assert CHAPTERS[cid]["theme_id"] == tid, (
                    f"{cid} listed under {tid} but theme_id={CHAPTERS[cid]['theme_id']}"
                )


# ---------------------------------------------------------------------------
# chapter_id_from_name
# ---------------------------------------------------------------------------

class TestChapterIdFromName:
    def test_full_name_with_colon(self):
        assert chapter_id_from_name("Ch 1: Foundations of AI Ethics & Philosophy") == "Ch 1"

    def test_plain_id_no_colon(self):
        assert chapter_id_from_name("Ch 12") == "Ch 12"

    def test_last_chapter(self):
        assert chapter_id_from_name("Ch 25: AI & Intellectual Property") == "Ch 25"

    def test_unknown_chapter_returns_none(self):
        assert chapter_id_from_name("Ch 99: Nonexistent") is None

    def test_empty_string_returns_none(self):
        assert chapter_id_from_name("") is None

    def test_none_returns_none(self):
        assert chapter_id_from_name(None) is None  # type: ignore[arg-type]

    def test_whitespace_only_returns_none(self):
        assert chapter_id_from_name("   ") is None

    def test_colon_in_chapter_name_stripped_correctly(self):
        # "Ch 3: AI Policy, Regulation & Legislation" → "Ch 3"
        assert chapter_id_from_name("Ch 3: AI Policy, Regulation & Legislation") == "Ch 3"

    def test_chapter_with_extra_whitespace(self):
        # Leading whitespace before colon
        result = chapter_id_from_name("Ch 5: Open Data")
        assert result == "Ch 5"

    @pytest.mark.parametrize("cid", list(CHAPTERS.keys()))
    def test_all_chapter_full_names_resolve(self, cid):
        full_name = CHAPTERS[cid]["name"]
        assert chapter_id_from_name(full_name) == cid


# ---------------------------------------------------------------------------
# theme_id_from_name
# ---------------------------------------------------------------------------

class TestThemeIdFromName:
    def test_known_theme(self):
        assert theme_id_from_name("AI Ethics, Governance & Policy") == "T1"

    def test_another_known_theme(self):
        assert theme_id_from_name("Disinformation, Deepfakes & AI Safety") == "T5"

    def test_last_theme(self):
        assert theme_id_from_name("AI, Law & Human Rights") == "T7"

    def test_unknown_theme_returns_none(self):
        assert theme_id_from_name("Completely Unknown Theme") is None

    def test_empty_string_returns_none(self):
        assert theme_id_from_name("") is None

    def test_none_returns_none(self):
        assert theme_id_from_name(None) is None  # type: ignore[arg-type]

    def test_partial_name_returns_none(self):
        assert theme_id_from_name("AI Ethics") is None

    @pytest.mark.parametrize("tid,theme", list(THEMES.items()))
    def test_all_theme_names_resolve(self, tid, theme):
        assert theme_id_from_name(theme["name"]) == tid
