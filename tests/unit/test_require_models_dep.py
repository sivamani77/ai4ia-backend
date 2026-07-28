"""Unit tests for app.api.deps.require_models — the interactive-endpoint gate.

Business logic must never run when a named target is cold: FastAPI never
calls the route handler if a Depends() raises, so these tests exercise the
dependency function directly (calling the route handler would defeat the
point — it would prove the handler works, not that the gate blocks it).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.api.deps import require_models
from app.services import model_gate


class TestRequireModels:
    def test_passes_through_when_gate_disabled(self, monkeypatch):
        monkeypatch.setattr("app.api.deps.settings.model_gate_enabled", False)
        with patch("app.api.deps.model_gate.ensure_ready") as mock_ensure:
            require_models(model_gate.LLM)()
        mock_ensure.assert_not_called()

    def test_passes_through_when_all_targets_ready(self, monkeypatch):
        monkeypatch.setattr("app.api.deps.settings.model_gate_enabled", True)
        with patch("app.api.deps.model_gate.ensure_ready", return_value={model_gate.LLM: True}):
            require_models(model_gate.LLM)()  # must not raise

    def test_raises_503_with_retry_after_when_cold(self, monkeypatch):
        monkeypatch.setattr("app.api.deps.settings.model_gate_enabled", True)
        with patch(
            "app.api.deps.model_gate.ensure_ready",
            return_value={model_gate.EMBEDDING: False, model_gate.LLM: True},
        ), patch("app.api.deps.model_gate.state", return_value={"state": "warming"}):
            with pytest.raises(HTTPException) as exc_info:
                require_models(model_gate.EMBEDDING, model_gate.LLM)()

        assert exc_info.value.status_code == 503
        assert exc_info.value.headers["Retry-After"] == "20"
        assert exc_info.value.detail["code"] == "model_warming"
        assert exc_info.value.detail["targets"] == [model_gate.EMBEDDING]

    def test_passes_the_configured_inline_wait_to_ensure_ready(self, monkeypatch):
        monkeypatch.setattr("app.api.deps.settings.model_gate_enabled", True)
        monkeypatch.setattr("app.api.deps.settings.model_gate_inline_wait_seconds", 7)
        with patch(
            "app.api.deps.model_gate.ensure_ready", return_value={model_gate.LLM: True}
        ) as mock_ensure:
            require_models(model_gate.LLM)()
        mock_ensure.assert_called_once_with(model_gate.LLM, wait_seconds=7)
