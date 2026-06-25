"""Unit tests for app.services.enrichment — derive_year, derive_edition, etc."""
from __future__ import annotations

import pytest

from app.services.enrichment import (
    derive_edition,
    derive_facets,
    derive_session_type,
    derive_year,
)


class TestDeriveYear:
    def test_year_from_title_ai4ia(self):
        assert derive_year("AI4IA 2025 Opening", "") == 2025

    def test_year_from_source_video(self):
        assert derive_year("", "0001_abc_AI4IA_2024_speaker") == 2024

    def test_any_year_fallback(self):
        assert derive_year("Conference 2023", "") == 2023

    def test_source_video_any_year_fallback(self):
        assert derive_year("", "video_2022_session") == 2022

    def test_no_year_returns_none(self):
        assert derive_year("No date here", "also_no_date") is None

    def test_title_takes_priority_over_source(self):
        # AI4IA pattern wins; title checked first
        assert derive_year("AI4IA 2025", "2020_old") == 2025

    def test_ai4ia_hyphen_separator(self):
        assert derive_year("AI4IA-2026 Conference", "") == 2026

    def test_ai4ia_underscore_separator(self):
        assert derive_year("AI4IA_2025_talk", "") == 2025

    def test_case_insensitive(self):
        assert derive_year("ai4ia 2025", "") == 2025

    def test_empty_strings_returns_none(self):
        assert derive_year("", "") is None


class TestDeriveEdition:
    def test_returns_ai4ia_year_format(self):
        assert derive_edition("AI4IA 2025 session", "") == "AI4IA 2025"

    def test_returns_none_when_no_year(self):
        assert derive_edition("no year here", "also none") is None

    def test_year_from_title(self):
        result = derive_edition("AI4IA 2023", "")
        assert result == "AI4IA 2023"


class TestDeriveSessionType:
    def test_default_talk(self):
        assert derive_session_type("Regular session", "0001_abc_speaker") == "talk"

    def test_panel(self):
        assert derive_session_type("Panel Discussion on AI", "") == "panel"

    def test_keynote(self):
        assert derive_session_type("Keynote address", "") == "keynote"

    def test_workshop(self):
        assert derive_session_type("Workshop: AI tools", "") == "workshop"

    def test_satellite(self):
        assert derive_session_type("Satellite event Africa", "") == "satellite_event"

    def test_regional(self):
        assert derive_session_type("Regional conference", "") == "regional_session"

    def test_opening(self):
        assert derive_session_type("Opening ceremony", "") == "opening"

    def test_closing(self):
        assert derive_session_type("Closing remarks", "") == "closing"

    def test_case_insensitive(self):
        assert derive_session_type("KEYNOTE speech", "") == "keynote"

    def test_keyword_in_source_video(self):
        assert derive_session_type("Normal title", "keynote_session") == "keynote"

    def test_empty_strings_returns_talk(self):
        assert derive_session_type("", "") == "talk"


class TestDeriveFacets:
    def test_returns_all_keys(self):
        result = derive_facets("AI4IA 2025 Keynote", "")
        assert "year" in result
        assert "edition" in result
        assert "session_type" in result

    def test_year_and_edition_consistent(self):
        result = derive_facets("AI4IA 2025 talk", "")
        assert result["year"] == 2025
        assert result["edition"] == "AI4IA 2025"

    def test_session_type_detected(self):
        result = derive_facets("Keynote address 2025", "")
        assert result["session_type"] == "keynote"

    def test_no_year_gives_none_edition(self):
        result = derive_facets("Regular talk", "")
        assert result["year"] is None
        assert result["edition"] is None
