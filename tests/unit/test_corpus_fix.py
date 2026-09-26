"""Corpus fix (2026-09) and the conference year every feature reads."""
from __future__ import annotations

import pytest

from app.services import corpus_fix
from app.services.conference_year import conference_year, known_years
from app.services.enrichment import derive_facets
from app.services.speaker_registry import get_registry

OPENING_2022 = "0243_srXR5cfLK4I_2022 - 3rd Annual AI4IA Conference - Live Opening Session"


class TestConferenceYear:
    def test_shipped_map_covers_every_year(self):
        assert set(known_years().values()) == {2021, 2022, 2023, 2024, 2025}
        assert known_years()["srXR5cfLK4I"] == 2022          # uploaded 2023-06
        assert "ea96KAytzSg" not in known_years()            # duplicate upload, removed

    @pytest.mark.parametrize("video_id,source_video,fallback,expected", [
        ("srXR5cfLK4I", OPENING_2022, 2023, 2022),            # the map beats the upload year
        (None, OPENING_2022, None, 2022),                     # id parsed from source_video
        ("newVideo123", "0400_newVideo123_Jane Doe - AI4IA 2026", 2025, 2026),   # title year
        ("newVideo123", "0400_newVideo123_Jane Doe - A talk", 2027, 2027),      # upload year last
        ("newVideo123", "0400_newVideo123_Jane Doe - predictions for 2030", "x", None),
    ])
    def test_order(self, video_id, source_video, fallback, expected):
        assert conference_year(video_id, source_video, fallback) == expected

    def test_postgres_facets_use_it(self):
        assert derive_facets("2022 - 3rd Annual AI4IA Conference - Live Opening Session", OPENING_2022)["year"] == 2022
        assert derive_facets("", "0147_B-B0ilBRyLo_AI4IA Conference 2023 - Opening Session")["edition"] == "AI4IA 2023"


def row(fid, video_id, source_video, speaker="Cordel Green", year=None, edition=None):
    return {"fragment_id": fid, "video_id": video_id, "source_video": source_video, "speaker": speaker,
            "speaker_attribution": "video-level", "extra": {"speaker_roster": speaker}, "year": year,
            "edition": edition}


class TestPlan:
    def test_panel_year_and_duplicates(self):
        rows = [row("a", "srXR5cfLK4I", OPENING_2022, year=2022, edition="AI4IA 2022"),
                row("b", "ea96KAytzSg", "0145_ea96KAytzSg_Ntsibane Nntllapa - x", speaker="Ntsibane Ntatlapa"),
                row("c", "F-HMw2lKYhQ", "0167_F-HMw2lKYhQ_Cordel Green - Future Proofing", year=2023,
                    edition="AI4IA 2023")]
        changes = corpus_fix.plan(rows, get_registry(), known_years())
        assert changes["deletes"] == ["b"]
        assert set(changes["updates"]) == {"a"}               # c is already right
        new = changes["updates"]["a"]
        assert (new["speaker"], new["speaker_attribution"]) == ("", "Multiple speakers")
        assert "Andrea Millwood Hargrave" in new["speaker_roster"] and new["year"] == 2022

    def test_a_second_plan_changes_nothing(self):
        first = corpus_fix.plan([row("a", "llAJjuaCCVE", "0098_llAJjuaCCVE_Chris Taylor - x", "Chris Taylor")],
                                get_registry(), known_years())["updates"]["a"]
        fixed = row("a", "llAJjuaCCVE", "0098_llAJjuaCCVE_Chris Taylor - x", first["speaker"],
                    first["year"], first["edition"])
        fixed.update(speaker_attribution=first["speaker_attribution"], extra={"speaker_roster": first["speaker_roster"]})
        assert corpus_fix.plan([fixed], get_registry(), known_years())["updates"] == {}

    def test_ships_as_a_dry_run(self):
        assert corpus_fix.RUN_CORPUS_FIX and not corpus_fix.CORPUS_FIX_APPLY
