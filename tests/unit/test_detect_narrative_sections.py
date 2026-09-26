"""detect_narrative_sections must never repeat the opening (hook, context) or
closing (synthesis) sections when a longer video needs more than 6 sections.

Confirmed incident this guards against: cycling the whole 6-section arc
(hook->...->synthesis->hook->...) put a second "opening hook" and "context"
section mid/late in the video (e.g. vjob_20260922_172613_d266276c rendered
S1_HOOK_2/S2_CONTEXT_2 as segments 12-13 of 14, right before the closing
card) — the burned-in video content read like a second open near the end.
Only the middle (evidence/challenge/response) should cycle to fill extra
slots on longer videos.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.pipeline.validate_gen.topic_detection import detect_narrative_sections  # noqa: E402


def test_short_video_unaffected():
    secs = detect_narrative_sections(120)
    assert [s["section_id"] for s in secs] == [
        "S1_HOOK", "S2_CONTEXT", "S3_EVIDENCE", "S4_CHALLENGE", "S5_RESPONSE", "S6_SYNTHESIS",
    ]


def test_long_video_never_repeats_hook_context_or_synthesis():
    for target_duration in (180, 240, 360, 480):
        secs = detect_narrative_sections(target_duration)
        ids = [s["section_id"] for s in secs]
        assert ids[0] == "S1_HOOK"
        assert ids[1] == "S2_CONTEXT"
        assert ids[-1] == "S6_SYNTHESIS"
        assert ids.count("S1_HOOK") == 1
        assert ids.count("S2_CONTEXT") == 1
        assert ids.count("S6_SYNTHESIS") == 1
        assert "S1_HOOK_2" not in ids
        assert "S2_CONTEXT_2" not in ids


def test_long_video_cycles_middle_sections_to_fill_extra_slots():
    secs = detect_narrative_sections(360)
    ids = [s["section_id"] for s in secs]
    assert "S3_EVIDENCE_2" in ids
    assert "S4_CHALLENGE_2" in ids
    assert "S5_RESPONSE_2" in ids
