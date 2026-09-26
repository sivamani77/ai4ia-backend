"""Chroma stores a missing year as "" (or 0); stage 07 must treat it as no year."""
from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.storyline_gen import retrieval
from app.pipeline.storyline_gen.selection import _assign_relay_sections_by_year
from app.services.conference_year import as_year

UNKNOWN = "0001_ZZZZZZZZZZZ_Some Talk"  # id not in the curated year map


def frag(fid, year, source_video=UNKNOWN):
    return {"fragment_id": fid, "video_year": year, "source_video": source_video,
            "speaker_name": "A Speaker", "text": "t", "start": 0, "end": 5}


def structure(n):
    return [{"section_type": "speaker", "role": f"s{i}"} for i in range(n)]


class TestAsYear:
    @pytest.mark.parametrize("value", ["", None, 0, "0", "abc", 1999, 2100, "  "])
    def test_not_a_year(self, value):
        assert as_year(value) is None

    @pytest.mark.parametrize("value", [2023, "2023", 2023.0])
    def test_year(self, value):
        assert as_year(value) == 2023


class TestRelaySections:
    def test_empty_and_zero_years_are_skipped(self):
        candidates = [frag("a", ""), frag("b", 2022), frag("c", 0), frag("d", None), frag("e", "2024")]
        out = _assign_relay_sections_by_year(structure(2), candidates)
        assert [s["year_slot"] for s in out] == [2022, 2024]
        used = {f["fragment_id"] for s in out for f in s["speaker_soundbites"]}
        assert used == {"b", "e"}

    def test_no_usable_year_returns_structure_unchanged(self):
        sections = structure(2)
        assert _assign_relay_sections_by_year(sections, [frag("a", ""), frag("b", 0)]) == sections


class FakeCollection:
    def __init__(self, metas):
        self.metas = metas

    def count(self):
        return len(self.metas)

    def query(self, **kwargs):
        ids = [m["fragment_id"] for m in self.metas]
        return {"ids": [ids], "distances": [[0.1] * len(ids)],
                "metadatas": [[{k: v for k, v in m.items() if k != "fragment_id"} for m in self.metas]]}


class TestSemanticRankYear:
    def test_chroma_year_is_normalized(self, monkeypatch):
        metas = [frag("a", ""), frag("b", 0), frag("c", 2023),
                 frag("d", "", source_video="0003_YYYYYYYYYYY_AI4IA 2022 Keynote")]
        monkeypatch.setattr(retrieval, "_get_chroma_collection", lambda: FakeCollection(metas))
        monkeypatch.setattr(retrieval, "_query_vector", lambda prompt: np.zeros(3))
        ranked = retrieval._semantic_rank_via_chroma("prompt", top_k=10)
        assert {r["fragment_id"]: r["video_year"] for r in ranked} == {"a": None, "b": None, "c": 2023, "d": 2022}

    def test_merged_records_carry_int_or_none_years(self, monkeypatch):
        metas = [frag("a", ""), frag("b", 2023)]
        corpus = [frag("a", None), frag("b", 2023), frag("c", None)]  # already normalized by the full load
        monkeypatch.setattr(retrieval, "_chroma_mode", lambda: True)
        monkeypatch.setattr(retrieval, "_FILTERED_RECORDS", corpus)
        monkeypatch.setattr(retrieval, "_get_chroma_collection", lambda: FakeCollection(metas))
        monkeypatch.setattr(retrieval, "_query_vector", lambda prompt: np.zeros(3))
        scored = retrieval.score_all_records("prompt", semantic_k=10)
        assert [(r["fragment_id"], r["video_year"]) for r in scored] == [("a", None), ("b", 2023), ("c", None)]
