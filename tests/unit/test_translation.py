"""Unit tests for locale resolution, translation_service, and the
translation middleware walker."""
from __future__ import annotations

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# app.core.locale
# ---------------------------------------------------------------------------

class _FakeRequest:
    def __init__(self, query: dict | None = None, headers: dict | None = None):
        self.query_params = query or {}
        self.headers = headers or {}


@pytest.fixture(autouse=True)
def _enable_translation():
    from app.core.settings import settings
    with patch.object(settings, "translation_enabled", True), \
         patch.object(settings, "translation_provider", "vllm"):
        yield


class TestResolveLocale:
    def test_query_lang_only(self):
        from app.core.locale import resolve_locale
        assert resolve_locale(_FakeRequest({"lang": "fr"})) == "fr"

    def test_query_lang_and_region(self):
        from app.core.locale import resolve_locale
        assert resolve_locale(_FakeRequest({"lang": "pt", "region": "br"})) == "pt-BR"

    def test_language_param_alias(self):
        from app.core.locale import resolve_locale
        assert resolve_locale(_FakeRequest({"language": "de"})) == "de"

    def test_english_is_noop(self):
        from app.core.locale import resolve_locale
        assert resolve_locale(_FakeRequest({"lang": "en"})) is None

    def test_unsupported_language(self):
        from app.core.locale import resolve_locale
        assert resolve_locale(_FakeRequest({"lang": "xx"})) is None

    def test_invalid_input_rejected(self):
        from app.core.locale import resolve_locale
        assert resolve_locale(_FakeRequest({"lang": "fr; DROP"})) is None
        assert resolve_locale(_FakeRequest({"lang": "fr", "region": "123"})) == "fr"

    def test_accept_language_header(self):
        from app.core.locale import resolve_locale
        req = _FakeRequest(headers={"accept-language": "fr-CA,fr;q=0.9,en;q=0.8"})
        assert resolve_locale(req) == "fr-CA"

    def test_accept_language_wildcard_ignored(self):
        from app.core.locale import resolve_locale
        assert resolve_locale(_FakeRequest(headers={"accept-language": "*"})) is None

    def test_query_wins_over_header(self):
        from app.core.locale import resolve_locale
        req = _FakeRequest({"lang": "de"}, {"accept-language": "fr"})
        assert resolve_locale(req) == "de"

    def test_disabled_flag(self):
        from app.core.locale import resolve_locale
        from app.core.settings import settings
        with patch.object(settings, "translation_enabled", False):
            assert resolve_locale(_FakeRequest({"lang": "fr"})) is None

    def test_no_locale_requested(self):
        from app.core.locale import resolve_locale
        assert resolve_locale(_FakeRequest()) is None


# ---------------------------------------------------------------------------
# translation_service.translate_batch
# ---------------------------------------------------------------------------

@pytest.fixture
def no_redis():
    import app.services.translation_service as ts
    with patch.object(ts, "_get_redis", return_value=None):
        yield ts


class TestTranslateBatch:
    def test_empty_input(self, no_redis):
        assert no_redis.translate_batch([], "fr") == ([], True)

    def test_provider_called_and_order_preserved(self, no_redis):
        with patch.object(no_redis, "_vllm_translate",
                          side_effect=lambda texts, loc: [f"FR:{t}" for t in texts]) as prov:
            out, complete = no_redis.translate_batch(["hello", "world"], "fr")
        assert out == ["FR:hello", "FR:world"]
        assert complete is True
        prov.assert_called_once()

    def test_duplicates_translated_once(self, no_redis):
        calls: list[list[str]] = []

        def fake(texts, loc):
            calls.append(list(texts))
            return [f"X:{t}" for t in texts]

        with patch.object(no_redis, "_vllm_translate", side_effect=fake):
            out, _ = no_redis.translate_batch(["a", "a", "b"], "fr")
        assert out == ["X:a", "X:a", "X:b"]
        assert calls == [["a", "b"]]

    def test_provider_failure_fails_open(self, no_redis):
        with patch.object(no_redis, "_vllm_translate", side_effect=RuntimeError("down")):
            out, complete = no_redis.translate_batch(["hello"], "fr")
        assert out == ["hello"]
        assert complete is False

    def test_unknown_provider_passes_through(self, no_redis):
        from app.core.settings import settings
        with patch.object(settings, "translation_provider", "nope"):
            out, complete = no_redis.translate_batch(["hello"], "fr")
        assert out == ["hello"]
        assert complete is False

    def test_whitespace_strings_skipped(self, no_redis):
        with patch.object(no_redis, "_vllm_translate",
                          side_effect=lambda texts, loc: [f"X:{t}" for t in texts]) as prov:
            out, complete = no_redis.translate_batch(["  ", ""], "fr")
        assert out == ["  ", ""]
        assert complete is True
        prov.assert_not_called()

    def test_chunking_large_batches(self, no_redis):
        calls: list[int] = []

        def fake(texts, loc):
            calls.append(len(texts))
            return list(texts)

        many = [f"string {i}" for i in range(130)]
        with patch.object(no_redis, "_vllm_translate", side_effect=fake), \
             patch.object(no_redis, "_MAX_ITEMS_PER_CALL", 60):
            no_redis.translate_batch(many, "fr")
        assert sum(calls) == 130
        assert all(n <= 60 for n in calls)


class TestVllmProviderValidation:
    def test_shape_mismatch_raises(self):
        import app.services.translation_service as ts
        fake_resp = {"choices": [{"message": {"content": '["only one"]'}}]}
        with patch.object(ts, "_post_with_retries", return_value=(fake_resp, 0)):
            with pytest.raises(ValueError, match="shape mismatch"):
                ts._vllm_translate(["a", "b"], "fr")

    def test_fenced_output_repaired(self):
        import app.services.translation_service as ts
        fake_resp = {"choices": [{"message": {"content": '```json\n["bonjour"]\n```'}}]}
        with patch.object(ts, "_post_with_retries", return_value=(fake_resp, 0)):
            assert ts._vllm_translate(["hello"], "fr") == ["bonjour"]

    def test_non_string_items_raise(self):
        import app.services.translation_service as ts
        fake_resp = {"choices": [{"message": {"content": '[42]'}}]}
        with patch.object(ts, "_post_with_retries", return_value=(fake_resp, 0)):
            with pytest.raises(ValueError, match="non-string"):
                ts._vllm_translate(["hello"], "fr")


# ---------------------------------------------------------------------------
# middleware walker
# ---------------------------------------------------------------------------

class TestWalker:
    def test_collect_and_apply_nested(self):
        from app.middleware.translation import apply_translations, collect_strings
        payload = {
            "id": "sl_123",
            "status": "completed",
            "title": "The Future of AI",
            "chapters": [
                {"title": "Opening", "narration": "Welcome to the conference.",
                 "fragment_id": "fr_9", "duration": 42.5},
                {"title": "Opening", "narration": ""},
            ],
            "created_at": "2026-07-03T00:00:00Z",
        }
        texts: list[str] = []
        collect_strings(payload, texts)
        assert texts == ["The Future of AI", "Opening", "Welcome to the conference.", "Opening"]

        mapping = {t: f"FR:{t}" for t in texts}
        out = apply_translations(payload, mapping)
        assert out["title"] == "FR:The Future of AI"
        assert out["chapters"][0]["narration"] == "FR:Welcome to the conference."
        # Non-allowlisted and non-string fields untouched
        assert out["id"] == "sl_123"
        assert out["status"] == "completed"
        assert out["chapters"][0]["fragment_id"] == "fr_9"
        assert out["chapters"][0]["duration"] == 42.5
        assert out["created_at"] == "2026-07-03T00:00:00Z"
        # Empty string untouched
        assert out["chapters"][1]["narration"] == ""

    def test_non_dict_root(self):
        from app.middleware.translation import collect_strings
        texts: list[str] = []
        collect_strings([{"title": "A"}, {"title": "B"}], texts)
        assert texts == ["A", "B"]

    def test_list_of_strings_under_allowlisted_key(self):
        from app.middleware.translation import apply_translations, collect_strings
        payload = {
            "key_points": ["First point", "Second point"],
            "fragment_ids": ["fr_1", "fr_2"],
            "synthesis": {"main_ideas": ["One idea"], "sources": ["v_9"]},
        }
        texts: list[str] = []
        collect_strings(payload, texts)
        assert texts == ["First point", "Second point", "One idea"]

        mapping = {t: f"FR:{t}" for t in texts}
        out = apply_translations(payload, mapping)
        assert out["key_points"] == ["FR:First point", "FR:Second point"]
        assert out["synthesis"]["main_ideas"] == ["FR:One idea"]
        # Non-allowlisted lists untouched
        assert out["fragment_ids"] == ["fr_1", "fr_2"]
        assert out["synthesis"]["sources"] == ["v_9"]

    def test_domain_specific_keys_covered(self):
        from app.middleware.translation import collect_strings
        payload = {
            "relay_title": "Passing the torch",
            "relay_steps": [{"role_in_relay": "opens the theme", "bridge": "links to policy"}],
            "synthesis_narration": "Across speakers...",
            "voiceover_text": "In this chapter...",
            "quote_text": "AI must serve everyone.",
            "video_mode_rationale": "Strong speaker evidence available.",
            "status": "completed",
            "job_id": "vj_1",
        }
        texts: list[str] = []
        collect_strings(payload, texts)
        assert set(texts) == {
            "Passing the torch", "opens the theme", "links to policy",
            "Across speakers...", "In this chapter...",
            "AI must serve everyone.", "Strong speaker evidence available.",
        }


# ---------------------------------------------------------------------------
# middleware end-to-end (stubbed translator)
# ---------------------------------------------------------------------------

class TestMiddlewareIntegration:
    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.middleware.translation import TranslationMiddleware

        app = FastAPI()
        app.add_middleware(TranslationMiddleware)

        @app.get("/thing")
        def thing():
            return {"id": "t_1", "title": "Hello world", "status": "ok"}

        @app.get("/health")
        def health():
            return {"title": "should not translate"}

        return TestClient(app)

    def test_translates_when_lang_present(self, client):
        with patch("app.services.translation_service.translate_batch",
                   return_value=(["Bonjour le monde"], True)):
            resp = client.get("/thing?lang=fr")
        assert resp.status_code == 200
        assert resp.json() == {"id": "t_1", "title": "Bonjour le monde", "status": "ok"}
        assert resp.headers["x-translation"] == "full"

    def test_no_lang_passthrough(self, client):
        resp = client.get("/thing")
        assert resp.json()["title"] == "Hello world"
        assert "x-translation" not in resp.headers

    def test_excluded_path(self, client):
        resp = client.get("/health?lang=fr")
        assert resp.json()["title"] == "should not translate"
        assert "x-translation" not in resp.headers

    def test_partial_translation_header(self, client):
        with patch("app.services.translation_service.translate_batch",
                   return_value=(["Hello world"], False)):
            resp = client.get("/thing?lang=fr")
        assert resp.headers["x-translation"] == "partial"

    def test_translator_crash_fails_open(self, client):
        with patch("app.services.translation_service.translate_batch",
                   side_effect=RuntimeError("boom")):
            resp = client.get("/thing?lang=fr")
        assert resp.status_code == 200
        assert resp.json()["title"] == "Hello world"
        assert resp.headers["x-translation"] == "skipped"

    def test_oversized_body_skipped(self, client):
        from app.core.settings import settings
        with patch.object(settings, "translation_max_body_bytes", 10):
            resp = client.get("/thing?lang=fr")
        assert resp.json()["title"] == "Hello world"
        assert resp.headers["x-translation"] == "skipped"
