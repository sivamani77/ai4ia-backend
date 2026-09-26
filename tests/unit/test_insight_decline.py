"""Structural decline path for insights — GAP-005.

"No relevant evidence in the archive" is a legitimate outcome, not a client
error. It used to surface as a 422 ValueError, which gave the safeguards gate
nothing stable to assert against and gave the UI a validation error to render
where it wanted an empty result. It is now a 200 carrying
outcome="insufficient_evidence" in the same shape as a generated insight.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestDeclineOutcome:
    def _declined(self, **overrides):
        from app.services import insight_service
        with patch.object(insight_service.guardrail_service, "assert_query_allowed", MagicMock()), \
             patch.object(insight_service, "build_evidence_pool", return_value=[]), \
             patch.object(insight_service, "InsightRepository") as repo, \
             patch.object(insight_service, "generate_json") as gen, \
             patch.object(insight_service, "_emit_insight_decision_record", MagicMock()) as rec:
            result = insight_service.generate_insight(
                overrides.get("query", "care instructions for a bonsai juniper"),
                filters=overrides.get("filters"),
            )
        return result, repo, gen, rec

    def test_returns_a_result_instead_of_raising(self):
        result, _, _, _ = self._declined()
        assert isinstance(result, dict)

    def test_outcome_is_insufficient_evidence(self):
        from app.services.insight_service import OUTCOME_INSUFFICIENT_EVIDENCE
        result, _, _, _ = self._declined()
        assert result["outcome"] == OUTCOME_INSUFFICIENT_EVIDENCE

    def test_carries_a_human_readable_reason(self):
        result, _, _, _ = self._declined()
        assert result["decline_reason"]
        assert isinstance(result["decline_reason"], str)

    def test_llm_is_never_called(self):
        """Answering with no evidence is the failure this path prevents."""
        _, _, gen, _ = self._declined()
        gen.assert_not_called()

    def test_decline_is_not_stored(self):
        _, repo, _, _ = self._declined()
        repo.assert_not_called()

    def test_decision_record_is_still_emitted(self):
        """The record is the only durable trace of a declined run."""
        _, _, _, rec = self._declined()
        rec.assert_called_once()

    def test_query_and_filters_are_echoed_back(self):
        result, _, _, _ = self._declined(query="Q", filters={"theme": "T1"})
        assert result["query"] == "Q"
        assert result["filters"] == {"theme": "T1"}

    def test_shape_matches_a_generated_insight(self):
        """A client that does not know about `outcome` must still render this."""
        result, _, _, _ = self._declined()
        for key in ("id", "insight_id", "title", "query", "filters", "summary",
                    "patterns", "contrasts", "recommended_clips",
                    "unsupported_claims", "evidence_coverage", "evidence_pool",
                    "models", "created_at", "_explain"):
            assert key in result, f"declined payload is missing {key}"

    def test_collections_are_empty_not_absent(self):
        result, _, _, _ = self._declined()
        assert result["patterns"] == []
        assert result["contrasts"] == []
        assert result["recommended_clips"] == []
        assert result["unsupported_claims"] == []
        assert result["evidence_pool"] == []

    def test_coverage_is_zero(self):
        result, _, _, _ = self._declined()
        assert result["evidence_coverage"] == 0.0

    def test_explain_records_that_generation_was_skipped(self):
        result, _, _, _ = self._declined()
        assert result["_explain"]["generation"]["skipped"] is True
        assert result["_explain"]["retrieval"]["fragments_retrieved"] == 0


class TestGeneratedOutcome:
    def _generated(self):
        from app.services import insight_service
        with patch.object(insight_service.guardrail_service, "assert_query_allowed", MagicMock()), \
             patch.object(insight_service, "build_evidence_pool",
                          return_value=[{"ref": 1, "fragment_id": "f1", "text": "t",
                                         "stream_url": "/s", "start": 0, "end": 5}]), \
             patch.object(insight_service, "generate_json",
                          return_value={"title": "T", "summary": "S",
                                        "patterns": [{"statement": "p", "fragment_refs": [1]}],
                                        "contrasts": [], "recommended_clips": []}), \
             patch.object(insight_service, "InsightRepository", MagicMock()), \
             patch.object(insight_service, "_emit_insight_decision_record", MagicMock()):
            return insight_service.generate_insight("AI safety")

    def test_success_carries_the_generated_outcome(self):
        """A client branches on `outcome` alone, never on empty lists."""
        from app.services.insight_service import OUTCOME_GENERATED
        assert self._generated()["outcome"] == OUTCOME_GENERATED

    def test_success_has_a_null_decline_reason(self):
        assert self._generated()["decline_reason"] is None


class TestStoredInsightsReportAnOutcome:
    """POST and GET must agree, though `outcome` is not a stored column."""

    def test_get_insight_defaults_to_generated(self):
        from app.services import insight_service
        from app.services.insight_service import OUTCOME_GENERATED
        repo = MagicMock()
        repo.return_value.get.return_value = {"id": "insight_1", "title": "T"}
        with patch.object(insight_service, "InsightRepository", repo):
            result = insight_service.get_insight("insight_1")
        assert result["outcome"] == OUTCOME_GENERATED
        assert result["decline_reason"] is None

    def test_get_insight_still_returns_none_when_missing(self):
        from app.services import insight_service
        repo = MagicMock()
        repo.return_value.get.return_value = None
        with patch.object(insight_service, "InsightRepository", repo):
            assert insight_service.get_insight("nope") is None


class TestDecisionRecordOutcome:
    def _record_for(self, insight):
        from app.services import insight_service
        captured = {}

        class _Repo:
            def create(self, data):
                captured.update(data)

        with patch.dict("sys.modules"), \
             patch("app.db.repositories.decision_record_repository.DecisionRecordRepository", _Repo):
            thread = MagicMock()
            with patch.object(insight_service.threading, "Thread",
                              side_effect=lambda target, daemon: _Immediate(target)):
                insight_service._emit_insight_decision_record(insight, "user-1")
        return captured

    def test_declined_run_is_recorded_as_declined(self):
        from app.services.insight_service import OUTCOME_INSUFFICIENT_EVIDENCE
        rec = self._record_for({"id": "i1", "outcome": OUTCOME_INSUFFICIENT_EVIDENCE,
                                "query": "q", "evidence_pool": [], "patterns": [],
                                "contrasts": [], "models": {}, "_explain": {}})
        assert rec["outcome"] == "insight_declined"

    def test_generated_run_is_recorded_as_generated(self):
        from app.services.insight_service import OUTCOME_GENERATED
        rec = self._record_for({"id": "i2", "outcome": OUTCOME_GENERATED,
                                "query": "q", "evidence_pool": [], "patterns": [],
                                "contrasts": [], "models": {}, "_explain": {}})
        assert rec["outcome"] == "insight_generated"


class _Immediate:
    """Stand-in for threading.Thread that runs the target synchronously."""

    def __init__(self, target):
        self._target = target

    def start(self):
        self._target()
