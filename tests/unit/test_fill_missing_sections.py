"""Regression test: _fill_missing_sections must preserve a donor candidate's
validated type (speaker_clip) instead of hardcoding narration_broll.

Root cause found via a real local pipeline run: the narrative-assembly LLM call
in assemble_narrative_with_llm sometimes returns a final_timeline that omits
sections entirely (a structured-JSON failure mode of the 20B model under load),
even though 07b's earlier per-fragment validation pass had already approved
several fragments as speaker_clip. The old _fill_missing_sections backfilled
every missing section as narration_broll unconditionally, silently discarding
speaker audio the LLM had explicitly approved. Observed impact: a run where 6
speaker-approved fragments all turned into narration_broll/archive_visual, so
the final video had zero real speaker clips despite the validator wanting them.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))


@pytest.fixture(scope="module")
def validate_mod():
    path = PROJECT_ROOT / "scripts" / "07b_validate_storyline_with_llm.py"
    spec = importlib.util.spec_from_file_location("validate_07b_fill", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sections(*ids):
    return [{"section_id": sid, "title": sid, "purpose": "", "card_text": ""} for sid in ids]


def _approved_speaker(fid: str, speaker: str = "Jane Doe"):
    return {
        "fragment_id": fid,
        "approved": True,
        "decision": "speaker_clip",
        "speaker_label": speaker,
        "quote_start": 10.0,
        "quote_end": 25.0,
        "quote_text": "a real quote",
        "quote_complete": True,
        "source_video": f"video_{fid}",
    }


def _approved_broll(fid: str):
    return {
        "fragment_id": fid,
        "approved": True,
        "decision": "narration_broll",
        "start": 5.0,
        "end": 15.0,
        "source_video": f"video_{fid}",
    }


class TestFillMissingSectionsPreservesSpeakerType:
    def test_approved_speaker_candidate_fills_as_speaker_clip(self, validate_mod):
        timeline = [{"type": "title_card", "duration": 5.0}, {"type": "closing_card", "duration": 5.0}]
        sections = _sections("S1_HOOK")
        validated = [_approved_speaker("frag_1")]

        result = validate_mod._fill_missing_sections(timeline, sections, validated, [])

        filled = [i for i in result if i.get("section_id") == "S1_HOOK"]
        assert len(filled) == 1
        item = filled[0]
        assert item["type"] == "speaker_clip"
        assert item["segment_type"] == "speaker_clip"
        assert item["audio_mode"] == "source_audio"
        assert item["muted"] is False
        assert item["render_start"] == 10.0
        assert item["render_end"] == 25.0
        assert item["speaker"] == "Jane Doe"
        assert item["_auto_filled"] is True

    def test_speaker_donor_preferred_over_broll_donor(self, validate_mod):
        timeline = [{"type": "closing_card", "duration": 5.0}]
        sections = _sections("S1_HOOK")
        # Both an approved broll and an approved speaker candidate available —
        # the speaker clip must win so real speaker audio isn't discarded.
        validated = [_approved_broll("frag_broll"), _approved_speaker("frag_speaker")]

        result = validate_mod._fill_missing_sections(timeline, sections, validated, [])
        filled = [i for i in result if i.get("section_id") == "S1_HOOK"][0]
        assert filled["type"] == "speaker_clip"
        assert filled["fragment_id"] == "frag_speaker"

    def test_falls_back_to_broll_when_no_speaker_candidates(self, validate_mod):
        timeline = [{"type": "closing_card", "duration": 5.0}]
        sections = _sections("S1_HOOK")
        validated = [_approved_broll("frag_broll")]

        result = validate_mod._fill_missing_sections(timeline, sections, validated, [])
        filled = [i for i in result if i.get("section_id") == "S1_HOOK"][0]
        assert filled["type"] == "narration_broll"
        assert filled["fragment_id"] == "frag_broll"

    def test_falls_back_to_original_timeline_when_no_approved_candidates(self, validate_mod):
        timeline = [{"type": "closing_card", "duration": 5.0}]
        sections = _sections("S1_HOOK")
        original = [{"fragment_id": "frag_orig", "source_video": "v1", "start": 0.0, "end": 10.0}]

        result = validate_mod._fill_missing_sections(timeline, sections, [], original)
        filled = [i for i in result if i.get("section_id") == "S1_HOOK"][0]
        assert filled["type"] == "narration_broll"
        assert filled["fragment_id"] == "frag_orig"

    def test_each_missing_section_gets_a_distinct_donor(self, validate_mod):
        timeline = [{"type": "closing_card", "duration": 5.0}]
        sections = _sections("S1_HOOK", "S2_CONTEXT", "S3_RESPONSE")
        validated = [_approved_speaker("frag_a"), _approved_speaker("frag_b"), _approved_broll("frag_c")]

        result = validate_mod._fill_missing_sections(timeline, sections, validated, [])
        filled = [i for i in result if i.get("section_id")]
        assert len(filled) == 3
        fids = {i["fragment_id"] for i in filled}
        assert fids == {"frag_a", "frag_b", "frag_c"}
        types = {i["section_id"]: i["type"] for i in filled}
        assert types["S1_HOOK"] == "speaker_clip"
        assert types["S2_CONTEXT"] == "speaker_clip"
        assert types["S3_RESPONSE"] == "narration_broll"

    def test_already_used_fragment_id_not_reused_as_donor(self, validate_mod):
        timeline = [
            {"type": "speaker_clip", "section_id": "S1_HOOK", "fragment_id": "frag_used"},
            {"type": "closing_card", "duration": 5.0},
        ]
        sections = _sections("S1_HOOK", "S2_CONTEXT")
        validated = [_approved_speaker("frag_used"), _approved_speaker("frag_new")]

        result = validate_mod._fill_missing_sections(timeline, sections, validated, [])
        filled = [i for i in result if i.get("section_id") == "S2_CONTEXT"][0]
        assert filled["fragment_id"] == "frag_new"

    def test_no_missing_sections_returns_timeline_unchanged(self, validate_mod):
        timeline = [
            {"type": "speaker_clip", "section_id": "S1_HOOK", "fragment_id": "frag_1"},
            {"type": "closing_card", "duration": 5.0},
        ]
        sections = _sections("S1_HOOK")
        result = validate_mod._fill_missing_sections(timeline, sections, [_approved_speaker("frag_2")], [])
        assert result == timeline

    def test_insert_happens_before_closing_card(self, validate_mod):
        timeline = [
            {"type": "title_card", "duration": 5.0},
            {"type": "closing_card", "duration": 5.0},
        ]
        sections = _sections("S1_HOOK")
        result = validate_mod._fill_missing_sections(timeline, sections, [_approved_speaker("frag_1")], [])
        assert result[-1]["type"] == "closing_card"
        assert result[-2]["section_id"] == "S1_HOOK"

    def test_rejected_candidate_not_used_as_donor(self, validate_mod):
        timeline = [{"type": "closing_card", "duration": 5.0}]
        sections = _sections("S1_HOOK")
        rejected = {**_approved_speaker("frag_rejected"), "approved": False, "decision": "reject"}

        result = validate_mod._fill_missing_sections(timeline, sections, [rejected], [])
        filled = [i for i in result if i.get("section_id") == "S1_HOOK"]
        # no approved donor at all -> falls through to empty narration_broll stub
        assert len(filled) == 1
        assert filled[0]["fragment_id"] is None
