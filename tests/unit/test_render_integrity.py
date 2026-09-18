"""Phase 4/5 hardening tests: clip windows are never shrunk, narration durations
come from real TTS audio, and unrenderable timelines fail loudly before render."""

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.pipeline.assets_gen import voiceover as voiceover_mod  # noqa: E402


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen_assets():
    return _load_script("08_generate_assets")


class TestQuoteRenderBounds:
    def test_locked_validation_window_passes_through_verbatim(self, gen_assets):
        item = {
            "start": 630.8, "end": 679.744,
            "render_start": 658.6, "render_end": 666.8,
            "quote_complete": True,
        }
        bounds = gen_assets.compute_quote_render_bounds(item, {}, None)
        assert bounds["render_start"] == 658.6
        assert bounds["render_end"] == 666.8
        assert bounds["render_duration"] == 8.2
        assert bounds["quote_boundary_method"] == "locked_by_validation"
        assert bounds["should_demote"] is False

    def test_long_clip_is_not_clamped_to_22s(self, gen_assets):
        # Regression: CLIP_HARD_MAX used to shrink a 58s validated clip to 22s
        item = {
            "start": 630.8, "end": 679.744,
            "moment_text": "A full validated statement that ends properly.",
            "moment_scores": {"quote_completeness_score": 0.9, "incomplete_sentence_penalty": 0.0},
        }
        bounds = gen_assets.compute_quote_render_bounds(item, {}, None)
        # full window + pre/post roll, nothing shaved off
        assert bounds["render_duration"] >= (679.744 - 630.8)

    def test_short_clip_is_not_stretched_into_unvalidated_footage(self, gen_assets):
        item = {
            "start": 100.0, "end": 103.0,
            "moment_text": "Short but complete.",
            "moment_scores": {"quote_completeness_score": 0.9, "incomplete_sentence_penalty": 0.0},
        }
        bounds = gen_assets.compute_quote_render_bounds(item, {}, None)
        # old CLIP_MIN_DURATION would have forced this to 7.0s
        assert bounds["render_duration"] < 5.0


class TestNarrationReconciliation:
    def test_audio_longer_than_fragment_window_wins(self, gen_assets, monkeypatch, tmp_path):
        audio = tmp_path / "vo.mp3"
        audio.write_bytes(b"fake")
        monkeypatch.setattr(voiceover_mod, "get_media_duration", lambda p: 60.0)
        item = {"start": 0.0, "end": 45.0, "duration": 45.0}
        gen_assets.reconcile_narration_duration(item, str(audio))
        assert item["render_duration"] == 60.4  # audio + 0.4 tail, never trimmed
        assert item["duration"] == 60.4
        assert item["tts_audio_seconds"] == 60.0

    def test_audio_always_wins_over_fragment_window_when_present(self, gen_assets, monkeypatch, tmp_path):
        # Real TTS audio always wins over any inherited start/end window, even a
        # larger one — a narration_broll segment is audio over a still image, so
        # honouring a stray, larger window would pad it with dead silence (see
        # reconcile_narration_duration's docstring). The window is only a
        # last-resort fallback when there's no usable audio at all.
        audio = tmp_path / "vo.mp3"
        audio.write_bytes(b"fake")
        monkeypatch.setattr(voiceover_mod, "get_media_duration", lambda p: 20.0)
        item = {"duration": 49.0}
        gen_assets.reconcile_narration_duration(item, str(audio))
        assert item["render_duration"] == 20.4

    def test_missing_audio_keeps_fragment_duration(self, gen_assets):
        item = {"duration": 30.0}
        gen_assets.reconcile_narration_duration(item, "/nonexistent/vo.mp3")
        assert item["render_duration"] == 30.0


class TestPreflightAudit:
    def _manifest(self, tmp_path, timeline):
        title = tmp_path / "title.png"
        closing = tmp_path / "closing.png"
        title.write_bytes(b"png")
        closing.write_bytes(b"png")
        return {
            "cards": {"title_card": str(title), "closing_card": str(closing)},
            "timeline": timeline,
        }

    def test_clean_manifest_passes(self, gen_assets, tmp_path):
        audio = tmp_path / "vo.mp3"
        audio.write_bytes(b"mp3")
        visual = tmp_path / "visual.png"
        visual.write_bytes(b"png")
        manifest = self._manifest(tmp_path, [
            {"type": "title_card", "duration": 5.0},
            {"segment_type": "speaker_clip", "fragment_id": "f1", "source_video": "v1",
             "render_start": 10.0, "render_end": 58.0},
            {"segment_type": "narration_broll", "audio_mode": "ai_voiceover",
             "audio_path": str(audio), "render_visual_path": str(visual), "duration": 30.0},
            {"type": "closing_card", "duration": 5.0},
        ])
        assert gen_assets.preflight_audit_manifest(manifest) == []

    def test_missing_voiceover_audio_fails(self, gen_assets, tmp_path):
        manifest = self._manifest(tmp_path, [
            {"segment_type": "narration_broll", "audio_mode": "ai_voiceover",
             "audio_path": str(tmp_path / "gone.mp3"), "source_video": "v1", "duration": 30.0},
        ])
        failures = gen_assets.preflight_audit_manifest(manifest)
        assert len(failures) == 1 and "voiceover audio missing" in failures[0]

    def test_footageless_narration_without_visual_fails(self, gen_assets, tmp_path):
        manifest = self._manifest(tmp_path, [
            {"segment_type": "narration_broll", "audio_mode": "external", "duration": 30.0},
        ])
        failures = gen_assets.preflight_audit_manifest(manifest, skip_tts=True)
        assert any("no generated visual" in f for f in failures)

    def test_zero_duration_content_fails(self, gen_assets, tmp_path):
        manifest = self._manifest(tmp_path, [
            {"segment_type": "speaker_clip", "fragment_id": "f1", "source_video": "v1"},
        ])
        failures = gen_assets.preflight_audit_manifest(manifest)
        assert any("no measurable duration" in f for f in failures)

    def test_missing_card_png_fails(self, gen_assets, tmp_path):
        manifest = {"cards": {"title_card": str(tmp_path / "gone.png")}, "timeline": []}
        failures = gen_assets.preflight_audit_manifest(manifest)
        assert any("title_card" in f for f in failures)
        assert any("closing_card" in f for f in failures)

    def test_skip_tts_suppresses_audio_checks_only(self, gen_assets, tmp_path):
        manifest = self._manifest(tmp_path, [
            {"segment_type": "narration_broll", "audio_mode": "ai_voiceover",
             "source_video": "v1", "duration": 30.0},
        ])
        assert gen_assets.preflight_audit_manifest(manifest, skip_tts=True) == []
        failures = gen_assets.preflight_audit_manifest(manifest, skip_tts=False)
        assert any("no audio_path" in f for f in failures)
