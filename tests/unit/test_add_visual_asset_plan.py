"""_add_visual_asset_plan must resolve a repeated section's ("S1_HOOK_2") visual
plan by stripping the pass suffix before the template lookup — otherwise the
lookup misses (the plan table only has base ids) and falls back to a raw-slug
headline like "S1 Hook 2" instead of the on-brand template title.

Confirmed incident this guards against: vjob_20260922_172613_d266276c's
auto-filled S1_HOOK_2/S2_CONTEXT_2 sections rendered "S1 Hook 2" as the card's
big headline instead of the template's "The Central Question".
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.pipeline.validate_gen.quality_gates import _add_visual_asset_plan  # noqa: E402


def test_repeated_section_gets_same_template_as_base_section():
    tl = [
        {"type": "narration_broll", "section_id": "S1_HOOK"},
        {"type": "narration_broll", "section_id": "S1_HOOK_2"},
    ]
    out = _add_visual_asset_plan(tl, [])
    assert out[0]["visual_title"] == out[1]["visual_title"]
    assert out[1]["visual_title"] != "S1 Hook 2"


def test_cycled_middle_section_resolves_template():
    tl = [{"type": "narration_broll", "section_id": "S3_EVIDENCE_3"}]
    out = _add_visual_asset_plan(tl, [])
    assert out[0]["visual_title"] == "Real-World Impact"


def test_unknown_section_id_still_falls_back_to_slug():
    tl = [{"type": "narration_broll", "section_id": "SOME_UNKNOWN_ID"}]
    out = _add_visual_asset_plan(tl, [])
    assert out[0]["visual_title"] == "Some Unknown Id"
