"""Unit tests for app.services.llm_service."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest

import app.services.llm_service as llm


# ---------------------------------------------------------------------------
# _post_with_retries
# ---------------------------------------------------------------------------

class TestPostWithRetries:
    def _mock_response(self, json_data: dict, status_code: int = 200) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = json_data
        resp.status_code = status_code
        if status_code >= 400:
            from requests import HTTPError
            resp.raise_for_status.side_effect = HTTPError(f"{status_code}")
        else:
            resp.raise_for_status.return_value = None
        return resp

    def test_success_first_attempt(self):
        resp = self._mock_response({"result": "ok"})
        with patch("app.services.llm_service.requests.post", return_value=resp) as mock_post:
            result, retry_count = llm._post_with_retries("http://ollama/api/test", {"key": "val"})
        assert result == {"result": "ok"}
        assert retry_count == 0
        mock_post.assert_called_once()

    def test_retries_on_exception(self):
        good = self._mock_response({"result": "ok"})
        with patch("app.services.llm_service.requests.post",
                   side_effect=[Exception("conn refused"), good]) as mock_post, \
             patch("app.services.llm_service.time.sleep"):
            result, retry_count = llm._post_with_retries("http://x", {})
        assert result == {"result": "ok"}
        assert retry_count == 1
        assert mock_post.call_count == 2

    def test_raises_after_max_retries(self):
        with patch("app.services.llm_service.requests.post",
                   side_effect=Exception("always fails")), \
             patch("app.services.llm_service.time.sleep"):
            with pytest.raises(RuntimeError, match="retries"):
                llm._post_with_retries("http://x", {})

    def test_no_sleep_on_last_attempt(self):
        with patch("app.services.llm_service.requests.post",
                   side_effect=Exception("fail")), \
             patch("app.services.llm_service.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError):
                llm._post_with_retries("http://x", {})
        # MAX_RETRIES=3: sleep called between attempts 1→2 and 2→3 = 2 times
        assert mock_sleep.call_count == llm.MAX_RETRIES - 1

    def test_http_error_retries(self):
        from requests import HTTPError
        bad_resp = MagicMock()
        bad_resp.raise_for_status.side_effect = HTTPError("500")
        good_resp = self._mock_response({"ok": True})
        with patch("app.services.llm_service.requests.post",
                   side_effect=[bad_resp, good_resp]), \
             patch("app.services.llm_service.time.sleep"):
            result, retry_count = llm._post_with_retries("http://x", {})
        assert result == {"ok": True}
        assert retry_count == 1


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------

class TestEmbedQuery:
    def _vec(self):
        return [0.1, 0.2, 0.3]

    def test_adds_search_query_prefix(self):
        with patch.object(llm, "_post_with_retries",
                          return_value=({"embedding": self._vec()}, 0)) as mock_post:
            llm.embed_query("AI ethics")
        call_kwargs = mock_post.call_args[0][1]
        assert call_kwargs["input"].startswith("search_query:")

    def test_no_double_prefix(self):
        with patch.object(llm, "_post_with_retries",
                          return_value=({"embedding": self._vec()}, 0)) as mock_post:
            llm.embed_query("search_query: already prefixed")
        call_kwargs = mock_post.call_args[0][1]
        assert not call_kwargs["input"].startswith("search_query: search_query:")

    def test_returns_embedding_from_new_endpoint(self):
        with patch.object(llm, "_post_with_retries",
                          return_value=({"embedding": self._vec()}, 0)):
            result = llm.embed_query("test")
        assert result == self._vec()

    def test_returns_embeddings_array_from_new_endpoint(self):
        with patch.object(llm, "_post_with_retries",
                          return_value=({"embeddings": [self._vec()]}, 0)):
            result = llm.embed_query("test")
        assert result == self._vec()

    def test_falls_back_to_old_endpoint(self):
        responses = [
            Exception("new endpoint down"),
            ({"embedding": self._vec()}, 0),
        ]
        with patch.object(llm, "_post_with_retries", side_effect=responses):
            result = llm.embed_query("test")
        assert result == self._vec()

    def test_raises_when_both_endpoints_fail(self):
        with patch.object(llm, "_post_with_retries", side_effect=Exception("down")):
            with pytest.raises(RuntimeError, match="Embedding failed"):
                llm.embed_query("test")

    def test_raises_when_no_embedding_key(self):
        with patch.object(llm, "_post_with_retries", return_value=({"unexpected": "keys"}, 0)):
            with pytest.raises(RuntimeError, match="Embedding failed"):
                llm.embed_query("test")

    def test_strips_whitespace_from_query(self):
        with patch.object(llm, "_post_with_retries",
                          return_value=({"embedding": self._vec()}, 0)) as mock_post:
            llm.embed_query("  whitespace  ")
        call_kwargs = mock_post.call_args[0][1]
        assert "whitespace" in call_kwargs["input"]
        assert not call_kwargs["input"].startswith("search_query:  ")


# ---------------------------------------------------------------------------
# generate_text
# ---------------------------------------------------------------------------

class TestGenerateText:
    def _chat_response(self, content: str) -> dict:
        return {"choices": [{"message": {"content": content}}]}

    def test_returns_response_string(self):
        with patch.object(llm, "_post_with_retries",
                          return_value=(self._chat_response("Here is my answer."), 0)):
            result = llm.generate_text("tell me about AI")
        assert result == "Here is my answer."

    def test_empty_response_returns_empty_string(self):
        with patch.object(llm, "_post_with_retries",
                          return_value=(self._chat_response(""), 0)):
            result = llm.generate_text("prompt")
        assert result == ""

    def test_uses_custom_model(self):
        with patch.object(llm, "_post_with_retries",
                          return_value=(self._chat_response("ok"), 0)) as mock_post:
            llm.generate_text("prompt", model="custom-model")
        payload = mock_post.call_args[0][1]
        assert payload["model"] == "custom-model"

    def test_uses_custom_temperature(self):
        with patch.object(llm, "_post_with_retries",
                          return_value=(self._chat_response("ok"), 0)) as mock_post:
            llm.generate_text("prompt", temperature=0.9)
        payload = mock_post.call_args[0][1]
        assert payload["temperature"] == 0.9

    def test_stream_false_in_payload(self):
        with patch.object(llm, "_post_with_retries",
                          return_value=(self._chat_response("ok"), 0)) as mock_post:
            llm.generate_text("prompt")
        payload = mock_post.call_args[0][1]
        assert payload["stream"] is False


# ---------------------------------------------------------------------------
# generate_json
# ---------------------------------------------------------------------------

class TestGenerateJson:
    def test_parses_valid_json(self):
        raw = json.dumps({"key": "value"})
        with patch.object(llm, "_post_with_retries", return_value=({"response": raw}, 0)):
            result = llm.generate_json("prompt")
        assert result == {"key": "value"}

    def test_salvages_json_from_surrounding_text(self):
        raw = 'Some preamble {"key": "value"} and trailing text'
        with patch.object(llm, "_post_with_retries", return_value=({"response": raw}, 0)):
            result = llm.generate_json("prompt")
        assert result == {"key": "value"}

    def test_raises_on_unparseable_response(self):
        with patch.object(llm, "_post_with_retries",
                          return_value=({"response": "not json at all"}, 0)):
            with pytest.raises(json.JSONDecodeError):
                llm.generate_json("prompt")

    def test_format_json_in_payload(self):
        with patch.object(llm, "_post_with_retries",
                          return_value=({"response": "{}"}, 0)) as mock_post:
            llm.generate_json("prompt")
        payload = mock_post.call_args[0][1]
        assert payload.get("format") == "json"

    def test_uses_custom_model(self):
        with patch.object(llm, "_post_with_retries",
                          return_value=({"response": "{}"}, 0)) as mock_post:
            llm.generate_json("prompt", model="special-model")
        payload = mock_post.call_args[0][1]
        assert payload["model"] == "special-model"

    def test_empty_response_raises(self):
        with patch.object(llm, "_post_with_retries", return_value=({"response": ""}, 0)):
            with pytest.raises(json.JSONDecodeError):
                llm.generate_json("prompt")

    def test_whitespace_only_response_raises(self):
        with patch.object(llm, "_post_with_retries", return_value=({"response": "  "}, 0)):
            with pytest.raises(json.JSONDecodeError):
                llm.generate_json("prompt")

    def test_nested_json_parsed(self):
        data = {"outer": {"inner": [1, 2, 3]}}
        with patch.object(llm, "_post_with_retries",
                          return_value=({"response": json.dumps(data)}, 0)):
            result = llm.generate_json("prompt")
        assert result["outer"]["inner"] == [1, 2, 3]
