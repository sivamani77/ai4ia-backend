"""friendly_message must never leak raw exception text to a client — it either
uses an exception's own curated user_message, or falls back to the error_type
table / a generic default. This is the one place deciding what users see for
failures whose real cause (ffmpeg stderr, tracebacks, fragment ids) is
deliberately kept in logs only."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.core.error_messages import DEFAULT_MESSAGE, friendly_message  # noqa: E402


def test_uses_exception_user_message_when_present():
    class Boom(Exception):
        user_message = "Something specific and friendly."

    assert friendly_message("pipeline_error", Boom("raw ffmpeg stderr blah")) == "Something specific and friendly."


def test_falls_back_to_error_type_table():
    msg = friendly_message("model_unavailable", None)
    assert "AI models" in msg


def test_unknown_error_type_falls_back_to_default():
    assert friendly_message("some_never_seen_type", None) == DEFAULT_MESSAGE
    assert friendly_message(None, None) == DEFAULT_MESSAGE


def test_never_returns_raw_exception_text():
    exc = RuntimeError("ffprobe -select_streams v:0 exited 1: stream not found")
    msg = friendly_message("pipeline_error", exc)
    assert "ffprobe" not in msg
