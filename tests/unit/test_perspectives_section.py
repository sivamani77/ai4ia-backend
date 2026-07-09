"""Tests for 07b's 'Additional Perspectives' duration-fill framing.

When the core narrative falls short of the requested duration,
fill_timeline_to_target already tops up with real speaker clips (no filler).
frame_duration_fill_as_perspectives_section groups those additions into an
explicit, labeled closing section (divider + section_id) instead of leaving
them as anonymous tail items, per the product decision that all remaining
eligible speaker voices on the topic should read as an intentional
"Additional Perspectives" compilation, not a silent patch.
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
    spec = importlib.util.spec_from_file_location("validate_storyline_07b", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fill_added_clip(i: int) -> dict:
    return {
        "type": "speaker_clip",
        "segment_type": "speaker_clip",
        "fragment_id": f"frag_{i:03d}",
        "source_video": f"video_{i:03d}",
        "added_by_duration_fill": True,
        "fill_stage": "supporting",
        "duration": 20.0,
    }


class TestFrameDurationFillAsPerspectivesSection:
    def test_noop_when_nothing_was_added(self, validate_mod):
        timeline = [
            {"type": "title_card", "segment_type": "title_card"},
            {"type": "speaker_clip", "segment_type": "speaker_clip", "fragment_id": "core_1"},
            {"type": "closing_card", "segment_type": "closing_card"},
        ]
        before = list(timeline)
        n = validate_mod.frame_duration_fill_as_perspectives_section(timeline)
        assert n == 0
        assert timeline == before

    def test_inserts_divider_and_tags_added_clips(self, validate_mod):
        timeline = [
            {"type": "title_card", "segment_type": "title_card"},
            {"type": "speaker_clip", "segment_type": "speaker_clip", "fragment_id": "core_1"},
            _fill_added_clip(1),
            _fill_added_clip(2),
            {"type": "closing_card", "segment_type": "closing_card"},
        ]
        n = validate_mod.frame_duration_fill_as_perspectives_section(timeline)
        assert n == 2

        types = [i.get("type") for i in timeline]
        assert types.count("title_card") == 2  # original + divider
        divider_idx = next(
            idx for idx, i in enumerate(timeline)
            if i.get("label") == "additional_perspectives_divider"
        )
        # Divider sits immediately before the first added clip.
        assert timeline[divider_idx + 1]["fragment_id"] == "frag_001"
        assert timeline[divider_idx]["section_id"] == "S_PERSPECTIVES"
        assert timeline[divider_idx]["subtitle_text"] == "Additional Perspectives on This Topic"

        for item in timeline:
            if item.get("added_by_duration_fill"):
                assert item["section_id"] == "S_PERSPECTIVES"
        # The original core clip must be untouched.
        core = next(i for i in timeline if i.get("fragment_id") == "core_1")
        assert "section_id" not in core

    def test_never_drops_or_duplicates_content(self, validate_mod):
        timeline = [
            {"type": "speaker_clip", "segment_type": "speaker_clip", "fragment_id": "core_1"},
            _fill_added_clip(1),
            {"type": "closing_card", "segment_type": "closing_card"},
        ]
        fragment_ids_before = {i.get("fragment_id") for i in timeline if i.get("fragment_id")}
        validate_mod.frame_duration_fill_as_perspectives_section(timeline)
        fragment_ids_after = {i.get("fragment_id") for i in timeline if i.get("fragment_id")}
        assert fragment_ids_before == fragment_ids_after
