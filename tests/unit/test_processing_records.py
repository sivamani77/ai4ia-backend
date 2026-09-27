"""Request and LLM-exchange recording (prompt_resolutions, llm_call_payloads)."""
from __future__ import annotations

import json
import os

import pytest

from app.pipeline import llm_exchange_log
from app.services import processing_records


class TestRequestRecord:
    def test_start_opens_an_in_progress_record(self, processing_records_store):
        processing_records.start_request("vjob_1", "user_1", "What did Cordel Green say?")

        row = processing_records_store.resolutions["vjob_1"]
        assert row["layer_outcome"] == "in_progress"
        assert row["raw_prompt"] == "What did Cordel Green say?"
        assert row["user_id"] == "user_1"
        assert row["code_version"]
        assert row["expires_at"].startswith("2026-12-31T23:59:59")

    def test_finish_does_not_overwrite_a_layer_decision(self, processing_records_store):
        processing_records.start_request("vjob_1", "u", "p")
        processing_records.record_layer("vjob_1", outcome="l1_rejected", reason_code="off_topic")
        processing_records.finish("vjob_1", "failed")

        assert processing_records_store.resolutions["vjob_1"]["layer_outcome"] == "l1_rejected"

    def test_finish_sets_the_outcome_of_an_open_record(self, processing_records_store):
        processing_records.start_request("vjob_1", "u", "p")
        processing_records.record_layer("vjob_1", flags={"l1_failed_open": True}, timings_ms={"l1": 812})
        processing_records.finish("vjob_1", "completed")

        row = processing_records_store.resolutions["vjob_1"]
        assert row["layer_outcome"] == "completed"
        assert row["flags"] == {"l1_failed_open": True}
        assert row["timings_ms"] == {"l1": 812}

    def test_a_database_error_never_reaches_the_caller(self, monkeypatch):
        def broken():
            raise RuntimeError("db down")

        monkeypatch.setattr(processing_records, "_resolutions", broken)
        monkeypatch.setattr(processing_records, "_payloads", broken)
        processing_records.start_request("vjob_1", "u", "p")
        processing_records.record_layer("vjob_1", outcome="l1_rejected")
        processing_records.finish("vjob_1", "failed")
        processing_records.store_llm_exchange({"id": "x", "request_text": "{}"})


class TestLLMExchanges:
    def test_payload_switch_off_stores_nothing(self, processing_records_store, monkeypatch):
        monkeypatch.setattr(processing_records.settings, "record_llm_payloads", False)
        processing_records.store_llm_exchange({"id": "x", "request_text": "{}"})
        assert processing_records_store.payloads == []

    def test_script_log_round_trip(self, processing_records_store, tmp_path, monkeypatch):
        log_path = tmp_path / "exchanges.jsonl"
        monkeypatch.setenv(llm_exchange_log.ENV_VAR, str(log_path))
        body = {"model": "m", "messages": [{"role": "user", "content": "Cordel Green — ümlaut"}]}
        llm_exchange_log.record("m", body, '{"not": valid json', latency_ms=1200)
        llm_exchange_log.record("m", body, None, latency_ms=5, error="timeout")
        log_path.write_text(log_path.read_text() + "not json\n")

        stored = processing_records.import_exchange_log(log_path, job_id="vjob_1", user_id="u", stage="07b")

        assert stored == 2
        assert not log_path.exists()
        first, second = processing_records_store.payloads
        assert first["request_text"] == json.dumps(body)       # exactly what was sent
        assert first["response_text"] == '{"not": valid json'  # invalid JSON kept as-is
        assert (first["job_id"], first["user_id"], first["stage"]) == ("vjob_1", "u", "07b")
        assert second["error"] == "timeout" and second["response_text"] is None

    def test_no_log_file_without_the_env_var(self, tmp_path, monkeypatch):
        monkeypatch.delenv(llm_exchange_log.ENV_VAR, raising=False)
        monkeypatch.chdir(tmp_path)
        llm_exchange_log.record("m", {}, "x", latency_ms=1)
        assert list(tmp_path.iterdir()) == []

    def test_exchange_log_imports_even_when_the_script_fails(self, processing_records_store, mock_job_store):
        env: dict[str, str] = {}
        with pytest.raises(RuntimeError):
            with processing_records.exchange_log("vjob_2", "07_storyline", env):
                os.environ[llm_exchange_log.ENV_VAR] = env[llm_exchange_log.ENV_VAR]
                try:
                    llm_exchange_log.record("m", {"k": 1}, "partial", latency_ms=3)
                finally:
                    del os.environ[llm_exchange_log.ENV_VAR]
                raise RuntimeError("script exited 1")

        assert [p["response_text"] for p in processing_records_store.payloads] == ["partial"]
        assert not processing_records.exchange_log_path("vjob_2", "07_storyline").exists()

    def test_log_file_is_owner_only(self, tmp_path, monkeypatch):
        log_path = tmp_path / "exchanges.jsonl"
        monkeypatch.setenv(llm_exchange_log.ENV_VAR, str(log_path))
        llm_exchange_log.record("m", {}, "x", latency_ms=1)
        assert oct(log_path.stat().st_mode & 0o777) == oct(0o600)

    def test_no_log_file_when_payload_recording_is_off(self, monkeypatch):
        monkeypatch.setattr(processing_records.settings, "record_llm_payloads", False)
        env: dict[str, str] = {}
        with processing_records.exchange_log("vjob_4", "07_storyline", env):
            pass
        assert env == {}

    @pytest.mark.parametrize("data,expected", [
        ({"choices": [{"message": {"content": "vllm text"}}]}, "vllm text"),
        ({"response": "ollama text"}, "ollama text"),
        ({"choices": [{"message": {"content": None}}]}, ""),
    ])
    def test_response_content(self, data, expected):
        assert llm_exchange_log.response_content(data) == expected


def test_insert_rejects_unknown_columns():
    from app.db.repositories.prompt_resolution_repository import PromptResolutionRepository

    with pytest.raises(ValueError):
        PromptResolutionRepository().create({"id": "pr_1", "job_id": "j", "raw_prompt": "p",
                                             "surface": "video", "id) values ('x'); --": 1})


class TestInProcessCalls:
    def test_traced_generate_json_stores_request_and_raw_response(self, processing_records_store, monkeypatch):
        from app.db.repositories.llm_trace_repository import LLMTraceRepository
        from app.services import llm_service

        raw = '{"allowed": true, "category": null}'
        monkeypatch.setattr(llm_service, "_post_with_retries",
                            lambda url, body, **kw: ({"choices": [{"message": {"content": raw}}]}, 0))
        monkeypatch.setattr(LLMTraceRepository, "create", lambda self, trace: None)
        monkeypatch.setattr(llm_service.threading, "Thread",
                            lambda target, daemon: type("T", (), {"start": staticmethod(target)})())

        llm_service.generate_json("classify this", model=llm_service.VLLM_MODEL, system="policy",
                                  _trace={"job_id": "vjob_3", "user_id": "u", "stage": "prompt_guardrail",
                                          "agent": "prompt_guardrail"})

        (payload,) = processing_records_store.payloads
        sent = json.loads(payload["request_text"])
        assert sent["messages"] == [{"role": "system", "content": "policy"},
                                    {"role": "user", "content": "classify this"}]
        assert payload["response_text"] == raw
        assert payload["trace_id"] == payload["id"]
        assert (payload["job_id"], payload["stage"]) == ("vjob_3", "prompt_guardrail")


class TestPipelineHooks:
    def test_job_lifecycle(self, processing_records_store, mock_job_store):
        from app.services import video_job_runner as runner

        job_id = runner.create_video_job({"prompt": "AI ethics", "target_duration": 60}, user_id="u9")
        assert processing_records_store.resolutions[job_id]["layer_outcome"] == "in_progress"
        runner._complete_job(job_id)
        assert processing_records_store.resolutions[job_id]["layer_outcome"] == "completed"

    def test_guardrail_rejection_is_recorded(self, processing_records_store, mock_job_store, monkeypatch):
        from app.pipeline import stage_guardrail
        from app.services import job_store
        from app.services import video_job_runner as runner
        from app.services.guardrail_service import GuardrailVerdict

        # The stage holds its own references, which mock_job_store does not patch.
        monkeypatch.setattr(stage_guardrail, "append_event", job_store.append_event)
        monkeypatch.setattr(stage_guardrail, "load_job", job_store.load_job)
        job_id = runner.create_video_job({"prompt": "weather in Kingston", "target_duration": 60}, user_id="u")
        monkeypatch.setattr(stage_guardrail.settings, "prompt_guardrail_enabled", True)
        monkeypatch.setattr(stage_guardrail, "classify",
                            lambda *a, **k: GuardrailVerdict(allowed=False, category="off_topic", reason="r"))
        result = stage_guardrail.run(job_id, {"prompt": "weather in Kingston"}, context=None)
        runner._fail_job(job_id, result.stage, result.data["error_code"], result.message,
                         user_message=result.data["user_message"])

        row = processing_records_store.resolutions[job_id]
        assert (row["layer_outcome"], row["reason_code"]) == ("l1_rejected", "off_topic")
        assert row["user_message"] == stage_guardrail.GUARDRAIL_OFF_TOPIC_MESSAGE
        error = mock_job_store["jobs"][job_id]["error"]
        assert error["user_message"] == stage_guardrail.GUARDRAIL_OFF_TOPIC_MESSAGE
        assert "l1" in row["timings_ms"]
