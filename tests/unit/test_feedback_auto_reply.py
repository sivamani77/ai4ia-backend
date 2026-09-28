"""Feedback auto-reply (issue #30): the LLM classifies and drafts an answer;
code appends the resolution-timing language, never the LLM."""
from __future__ import annotations

import pytest

from app.services import feedback_auto_reply
from app.services.feedback_auto_reply import generate


@pytest.fixture(autouse=True)
def _no_job_lookup(monkeypatch):
    """Keep this a unit test: no real job history lookup, empty by default."""
    import app.services.job_store as job_store

    class _EmptyRepo:
        def list_by_user(self, user_id):
            return []

    monkeypatch.setattr(job_store, "get_job_repo", lambda: _EmptyRepo())


class TestGenerate:
    def test_a_valid_reply_gets_the_category_suffix(self, monkeypatch):
        monkeypatch.setattr(feedback_auto_reply, "_ask",
                            lambda *a, **k: {"category": "bug", "reply": "Sounds like the export button is stuck."})
        result = generate("user-1", "the export button does nothing")
        assert result.category == "bug"
        assert result.reply.startswith("Sounds like the export button is stuck.")
        assert "few business days" in result.reply

    def test_question_and_other_get_no_suffix(self, monkeypatch):
        monkeypatch.setattr(feedback_auto_reply, "_ask",
                            lambda *a, **k: {"category": "question", "reply": "You can find that under Account Settings."})
        result = generate("user-1", "where do I change my language?")
        assert result.reply == "You can find that under Account Settings."

    def test_an_invalid_category_falls_back_to_other(self, monkeypatch):
        monkeypatch.setattr(feedback_auto_reply, "_ask",
                            lambda *a, **k: {"category": "praise", "reply": "Thank you for the kind words!"})
        result = generate("user-1", "this is great!")
        assert result.category == "other"

    def test_an_empty_reply_is_treated_as_failure(self, monkeypatch):
        monkeypatch.setattr(feedback_auto_reply, "_ask", lambda *a, **k: {"category": "bug", "reply": "   "})
        assert generate("user-1", "it's broken") is None

    def test_non_dict_output_is_treated_as_failure(self, monkeypatch):
        monkeypatch.setattr(feedback_auto_reply, "_ask", lambda *a, **k: "not json")
        assert generate("user-1", "hello") is None

    def test_an_exception_from_ask_never_raises(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("host down")
        monkeypatch.setattr(feedback_auto_reply, "_ask", boom)
        assert generate("user-1", "hello") is None

    def test_disabled_short_circuits_before_any_call(self, monkeypatch):
        monkeypatch.setattr(feedback_auto_reply.settings, "feedback_auto_reply_enabled", False)

        def boom(*a, **k):
            raise AssertionError("should not be called when disabled")
        monkeypatch.setattr(feedback_auto_reply, "_ask", boom)
        assert generate("user-1", "hello") is None

    def test_a_reply_over_the_length_cap_is_truncated_not_dropped(self, monkeypatch):
        long_reply = "x" * 2000
        monkeypatch.setattr(feedback_auto_reply, "_ask", lambda *a, **k: {"category": "other", "reply": long_reply})
        result = generate("user-1", "hello")
        assert result is not None
        assert len(result.reply) <= feedback_auto_reply.MAX_REPLY_CHARS
