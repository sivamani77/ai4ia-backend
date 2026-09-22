"""Relevance gate and the decline paths it makes reachable.

Guards the UAT defect where gibberish produced a confident relay: the query
services' `if not fragments` checks could never fire, because ANN search
always returns its top-k however poor the match.

Thresholds below come from measurements over the live corpus (top-1,
1 - cosine): genuine queries 0.735-0.815, off-topic real language
0.564-0.658.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.core.settings import settings  # noqa: E402
from app.services import comparison_service, evidence_gate  # noqa: E402
from app.services.insight_service import (  # noqa: E402
    OUTCOME_GENERATED,
    OUTCOME_INSUFFICIENT_EVIDENCE,
)


def frag(score=None, *, key="similarity", **extra):
    record = dict(extra)
    if score is not None:
        record[key] = score
    return record


class TestTopSimilarity:
    def test_reads_the_similarity_field(self):
        assert evidence_gate.top_similarity([frag(0.5), frag(0.8), frag(0.6)]) == 0.8

    def test_falls_back_to_semantic_score(self):
        """embedding_index.search sets `similarity`, chroma ranking sets the other."""
        assert evidence_gate.top_similarity([frag(0.77, key="semantic_score")]) == 0.77

    def test_prefers_similarity_when_both_present(self):
        assert evidence_gate.top_similarity([{"similarity": 0.9, "semantic_score": 0.1}]) == 0.9

    def test_unscored_pool_is_none(self):
        assert evidence_gate.top_similarity([{"fragment_id": "f1"}]) is None

    def test_empty_and_none_are_none(self):
        assert evidence_gate.top_similarity([]) is None
        assert evidence_gate.top_similarity(None) is None

    def test_unparseable_scores_are_skipped_not_fatal(self):
        assert evidence_gate.top_similarity([frag("bad"), frag(0.7)]) == 0.7


class TestEvidenceIsSufficient:
    def test_genuine_query_score_passes(self):
        ok, diag = evidence_gate.evidence_is_sufficient([frag(0.735)], surface="t")
        assert ok
        assert diag["reason"] == "above_floor"

    def test_off_topic_query_score_declines(self):
        ok, diag = evidence_gate.evidence_is_sufficient([frag(0.658)], surface="t")
        assert not ok
        assert diag["reason"] == "below_floor"

    def test_empty_pool_declines(self):
        ok, diag = evidence_gate.evidence_is_sufficient([], surface="t")
        assert not ok
        assert diag["reason"] == "no_fragments_retrieved"

    def test_unscored_pool_passes_rather_than_blocking(self):
        """A facet-retrieved pool has no similarity to judge; don't block it."""
        ok, diag = evidence_gate.evidence_is_sufficient([{"fragment_id": "f1"}], surface="t")
        assert ok
        assert diag["reason"] == "no_similarity_scores_to_judge"

    def test_judges_on_top_one_not_the_mean(self):
        """One strong hit is enough; the mean overlaps between query classes."""
        pool = [frag(0.80)] + [frag(0.40) for _ in range(19)]
        assert evidence_gate.evidence_is_sufficient(pool, surface="t")[0]

    def test_floor_is_configurable_per_call(self):
        assert evidence_gate.evidence_is_sufficient([frag(0.5)], surface="t", floor=0.4)[0]

    def test_diagnostics_carry_what_the_decline_payload_needs(self):
        _, diag = evidence_gate.evidence_is_sufficient([frag(0.5), frag(0.4)], surface="t")
        assert diag["top_similarity"] == 0.5
        assert diag["relevance_floor"] == settings.archive_relevance_floor
        assert diag["fragments_considered"] == 2


class TestRelayDeclines:
    """The exact UAT defect: gibberish must not produce a narrated relay."""

    def _run(self, scores):
        with patch.object(comparison_service.embedding_index, "search",
                          return_value=[frag(s, fragment_id=f"f{i}") for i, s in enumerate(scores)]), \
             patch.object(comparison_service.guardrail_service, "assert_query_allowed"), \
             patch.object(comparison_service, "generate_json") as llm:
            return comparison_service.build_relay("anything"), llm

    def test_weak_evidence_declines_without_calling_the_llm(self):
        result, llm = self._run([0.60, 0.58])
        assert result["outcome"] == OUTCOME_INSUFFICIENT_EVIDENCE
        assert result["relay_steps"] == []
        assert result["step_count"] == 0
        llm.assert_not_called()

    def test_declined_relay_has_the_same_keys_as_a_generated_one(self):
        declined, _ = self._run([0.60])
        with patch.object(comparison_service.embedding_index, "search",
                          return_value=[frag(0.80, fragment_id="f1", speaker="A", text="t")]), \
             patch.object(comparison_service.guardrail_service, "assert_query_allowed"), \
             patch.object(comparison_service, "generate_json",
                          return_value={"relay_title": "T", "relay_narrative": "N", "relay_steps": []}):
            generated = comparison_service.build_relay("anything")
        assert set(declined) >= set(generated), (
            f"declined payload is missing keys a client reads: {set(generated) - set(declined)}"
        )

    def test_strong_evidence_still_generates(self):
        with patch.object(comparison_service.embedding_index, "search",
                          return_value=[frag(0.80, fragment_id="f1", speaker="A", text="t")]), \
             patch.object(comparison_service.guardrail_service, "assert_query_allowed"), \
             patch.object(comparison_service, "generate_json",
                          return_value={"relay_title": "T", "relay_narrative": "N",
                                        "relay_steps": []}) as llm:
            result = comparison_service.build_relay("deepfakes and copyright law")
        assert result["outcome"] == OUTCOME_GENERATED
        llm.assert_called_once()


class TestThemeSynthesisDeclines:
    def test_weak_evidence_declines_without_calling_the_llm(self):
        with patch.object(comparison_service.embedding_index, "search",
                          return_value=[frag(0.55, fragment_id="f1")]), \
             patch.object(comparison_service.guardrail_service, "assert_query_allowed"), \
             patch.object(comparison_service, "generate_json") as llm:
            result = comparison_service.synthesize_theme("anything")
        assert result["outcome"] == OUTCOME_INSUFFICIENT_EVIDENCE
        assert result["key_points"] == []
        assert result["evidence_pool"] == []
        llm.assert_not_called()

    def test_declines_with_a_200_shape_not_a_valueerror(self):
        """It used to raise ValueError, which the router surfaced as a 422."""
        with patch.object(comparison_service.embedding_index, "search", return_value=[]), \
             patch.object(comparison_service.guardrail_service, "assert_query_allowed"), \
             patch.object(comparison_service, "generate_json"):
            result = comparison_service.synthesize_theme("anything")
        assert result["outcome"] == OUTCOME_INSUFFICIENT_EVIDENCE
