"""Regression test for a real deployed failure (job vjob_20260708_102016_cc72f881):

09_export failed with "Failed to resolve 6 source video(s)" for names like
'narration_S2_CONTEXT' and 'speaker_S1_HOOK' — not real video filenames, but
placeholder strings matching the {type}_{section_id} pattern shown in 07b's
own narrative-assembly prompt example. The LLM (openai/gpt-oss-20b) invented
a fragment_id in that shape when no real fragment was available for a slot;
enrich_final_timeline then blindly derived a source_video from it via
string-splitting, producing an unresolvable S3 key that only surfaced as a
failure much later, in export.

Fix: a fragment_id is trustworthy only if it matches a real candidate in
candidate_map / orig_by_fid / validated_map. If it matches none of them,
treat the item as if it never had a fragment — speaker_clip is dropped (no
fabricated footage), narration_broll degrades to footage-less (already
supported: Step 08 renders those over a generated visual).
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
    spec = importlib.util.spec_from_file_location("validate_storyline_07b_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestHallucinatedFragmentGuard:
    def test_hallucinated_speaker_clip_is_dropped(self, validate_mod):
        final_timeline = [
            {
                "type": "speaker_clip",
                "segment_type": "speaker_clip",
                "section_id": "S1_HOOK",
                "fragment_id": "speaker_S1_HOOK",
            },
        ]
        result = validate_mod.enrich_final_timeline(
            final_timeline, candidate_map={}, original_timeline=[], validated_map={},
        )
        assert result == []

    def test_hallucinated_narration_broll_degrades_to_footage_less(self, validate_mod):
        final_timeline = [
            {
                "type": "narration_broll",
                "segment_type": "narration_broll",
                "section_id": "S2_CONTEXT",
                "fragment_id": "narration_S2_CONTEXT",
                "voiceover_text": "Some narration about the topic.",
            },
        ]
        result = validate_mod.enrich_final_timeline(
            final_timeline, candidate_map={}, original_timeline=[], validated_map={},
        )
        assert len(result) == 1
        assert result[0]["fragment_id"] is None
        assert result[0]["source_video"] is None
        # Real content (the narration text) is preserved — only the bogus
        # fragment link is stripped.
        assert result[0]["voiceover_text"] == "Some narration about the topic."

    def test_real_fragment_id_is_preserved(self, validate_mod):
        final_timeline = [
            {
                "type": "speaker_clip",
                "segment_type": "speaker_clip",
                "section_id": "S1_HOOK",
                "fragment_id": "0100_real_video_0007",
                "start": 10.0,
                "end": 20.0,
            },
        ]
        candidate_map = {
            "0100_real_video_0007": {"source_video": "0100_real_video", "speaker": "Jane Doe"},
        }
        result = validate_mod.enrich_final_timeline(
            final_timeline, candidate_map=candidate_map, original_timeline=[], validated_map={},
        )
        assert len(result) == 1
        assert result[0]["fragment_id"] == "0100_real_video_0007"
        assert result[0]["source_video"] == "0100_real_video"

    def test_intentionally_footage_less_item_is_untouched(self, validate_mod):
        """An item that never had a fragment_id (Step 08 generates a visual
        for it) must not be affected by the hallucination guard."""
        final_timeline = [
            {
                "type": "narration_broll",
                "segment_type": "narration_broll",
                "section_id": "S3_EVIDENCE",
                "fragment_id": None,
                "voiceover_text": "Bridging narration with no footage.",
            },
        ]
        result = validate_mod.enrich_final_timeline(
            final_timeline, candidate_map={}, original_timeline=[], validated_map={},
        )
        assert len(result) == 1
        assert result[0]["fragment_id"] is None
        assert result[0]["source_video"] is None
