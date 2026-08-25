"""Regression coverage for the additive `system` parameter added to the three
LLM client entry points (llm_service.generate_text/generate_json,
storyline_gen/llm_client.call_strong_llm, and validate_gen/llm_client's
_to_ollama_generate_payload role-splitting) — confirms system=None preserves
the exact pre-existing payload shape, and system="..." produces a real
system-role message / Ollama "system" field."""
from __future__ import annotations

from unittest.mock import patch

import app.services.llm_service as llm_service
import app.pipeline.storyline_gen.llm_client as storyline_llm_client
from app.pipeline.validate_gen.llm_client import _to_ollama_generate_payload, _to_vllm_chat_payload


class TestGenerateTextSystemParam:
    def test_no_system_sends_single_user_message(self):
        with patch.object(llm_service, "_post_with_retries",
                           return_value=({"choices": [{"message": {"content": "ok"}}]}, 0)) as mock:
            llm_service.generate_text("hello")
        payload = mock.call_args[0][1]
        assert payload["messages"] == [{"role": "user", "content": "hello"}]

    def test_system_prepends_system_message(self):
        with patch.object(llm_service, "_post_with_retries",
                           return_value=({"choices": [{"message": {"content": "ok"}}]}, 0)) as mock:
            llm_service.generate_text("hello", system="be helpful")
        payload = mock.call_args[0][1]
        assert payload["messages"] == [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
        ]


class TestGenerateJsonSystemParam:
    def test_vllm_no_system_unchanged_shape(self):
        with patch.object(llm_service, "_post_with_retries",
                           return_value=({"choices": [{"message": {"content": "{}"}}]}, 0)) as mock:
            llm_service.generate_json("hello", model=llm_service.VLLM_MODEL)
        payload = mock.call_args[0][1]
        assert payload["messages"] == [{"role": "user", "content": "hello"}]

    def test_vllm_system_adds_system_message(self):
        with patch.object(llm_service, "_post_with_retries",
                           return_value=({"choices": [{"message": {"content": "{}"}}]}, 0)) as mock:
            llm_service.generate_json("hello", model=llm_service.VLLM_MODEL, system="classify this")
        payload = mock.call_args[0][1]
        assert payload["messages"][0] == {"role": "system", "content": "classify this"}

    def test_ollama_no_system_omits_system_key(self):
        with patch.object(llm_service, "_post_with_retries",
                           return_value=({"response": "{}"}, 0)) as mock:
            llm_service.generate_json("hello", model="some-fast-model")
        payload = mock.call_args[0][1]
        assert "system" not in payload
        assert payload["prompt"] == "hello"

    def test_ollama_system_sets_native_system_field(self):
        with patch.object(llm_service, "_post_with_retries",
                           return_value=({"response": "{}"}, 0)) as mock:
            llm_service.generate_json("hello", model="some-fast-model", system="classify this")
        payload = mock.call_args[0][1]
        assert payload["system"] == "classify this"
        assert payload["prompt"] == "hello"


class TestCallStrongLlmSystemParam:
    def test_ollama_no_system_omits_system_key(self):
        storyline_llm_client.STORYLINE_LLM_BACKEND = "ollama"
        with patch.object(storyline_llm_client, "post_with_retries",
                           return_value={"response": "ok"}) as mock:
            storyline_llm_client.call_strong_llm("hello", timeout=10)
        payload = mock.call_args[0][1]
        assert "system" not in payload

    def test_ollama_system_sets_native_system_field(self):
        storyline_llm_client.STORYLINE_LLM_BACKEND = "ollama"
        with patch.object(storyline_llm_client, "post_with_retries",
                           return_value={"response": "ok"}) as mock:
            storyline_llm_client.call_strong_llm("hello", timeout=10, system="be a producer")
        payload = mock.call_args[0][1]
        assert payload["system"] == "be a producer"
        assert payload["prompt"] == "hello"

    def test_vllm_system_prepends_message(self):
        storyline_llm_client.STORYLINE_LLM_BACKEND = "vllm"
        try:
            with patch.object(
                storyline_llm_client, "post_with_retries",
                return_value={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            ) as mock:
                storyline_llm_client.call_strong_llm("hello", timeout=10, system="be a producer")
            payload = mock.call_args[0][1]
            assert payload["messages"][0] == {"role": "system", "content": "be a producer"}
            assert payload["messages"][1] == {"role": "user", "content": "hello"}
        finally:
            storyline_llm_client.STORYLINE_LLM_BACKEND = "ollama"


class TestOllamaPayloadRoleSplitting:
    def test_single_user_message_unchanged_behavior(self):
        payload = _to_ollama_generate_payload({
            "model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.3,
        })
        assert payload["prompt"] == "hi"
        assert "system" not in payload

    def test_system_and_user_messages_split_correctly(self):
        payload = _to_ollama_generate_payload({
            "model": "m",
            "messages": [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "hi"},
            ],
            "temperature": 0.3,
        })
        assert payload["system"] == "you are helpful"
        assert payload["prompt"] == "hi"

    def test_vllm_payload_passes_messages_through_unchanged(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]
        payload = _to_vllm_chat_payload({"model": "m", "messages": messages, "temperature": 0.3})
        assert payload["messages"] == messages
