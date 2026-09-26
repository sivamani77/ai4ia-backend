"""Node labels on generated narration visuals (diagram/mindmap/risk_map) must
never overflow their shape, regardless of how long the source word is.

Confirmed incident this guards against: labels were sized by blind
character-count truncation (lbl[:10]) drawn at a fixed font size — neither
step measures real pixel width, so a long or wide word still overflowed its
circle/box while a short one wasted the space. _fit_node_label replaces that
with measure-then-shrink (matching the pattern concept_card's keyword pills
already used), falling back to word-boundary and then character truncation
only if the label still doesn't fit at the smallest acceptable font size.
"""

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.pipeline.assets_gen.cards import _fit_node_label, create_narration_visual_asset  # noqa: E402


@pytest.fixture
def draw():
    return ImageDraw.Draw(Image.new("RGB", (10, 10)))


class TestFitNodeLabel:
    def test_short_label_untouched(self, draw):
        font, text = _fit_node_label(draw, "SCOPE", max_width=80, base_size=16)
        bbox = draw.textbbox((0, 0), text, font=font)
        assert text == "SCOPE"
        assert bbox[2] - bbox[0] <= 80

    def test_long_word_never_overflows(self, draw):
        font, text = _fit_node_label(
            draw, "MULTISTAKEHOLDERISM", max_width=80, base_size=16, min_size=12,
        )
        bbox = draw.textbbox((0, 0), text, font=font)
        assert bbox[2] - bbox[0] <= 80

    def test_multi_word_falls_back_to_word_boundary(self, draw):
        font, text = _fit_node_label(
            draw, "REGULATORY COMPLIANCE MECHANISMS", max_width=90, base_size=16, min_size=12,
        )
        bbox = draw.textbbox((0, 0), text, font=font)
        assert bbox[2] - bbox[0] <= 90
        assert text.endswith("…") or text == "REGULATORY COMPLIANCE MECHANISMS"

    def test_empty_text(self, draw):
        font, text = _fit_node_label(draw, "", max_width=80, base_size=16)
        assert text == ""


class TestCreateNarrationVisualAssetDoesNotCrash:
    """Real render smoke test across all six visual types with deliberately
    long keywords — the exact shape of the reported overflow."""

    LONG_KEYWORDS = [
        "Interoperability",
        "Accountability Framework",
        "Regulatory Compliance Mechanisms",
        "Extraterritoriality",
    ]

    @pytest.mark.parametrize(
        "visual_asset_type",
        ["concept_card", "diagram", "risk_map", "mindmap", "policy_card", "closing_synthesis"],
    )
    def test_renders_without_error(self, tmp_path, visual_asset_type):
        out = tmp_path / f"{visual_asset_type}.png"
        create_narration_visual_asset(
            output_path=out,
            visual_title="A Deliberately Long Section Title For Overflow Testing",
            visual_asset_type=visual_asset_type,
            visual_keywords=self.LONG_KEYWORDS,
            section_id="S2_CONTEXT",
        )
        assert out.exists()
