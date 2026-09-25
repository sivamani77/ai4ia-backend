"""Evidence gate: declines only when retrieval returns nothing.

The similarity floor was removed; weak matches now reach the LLM.
Gibberish is rejected earlier by the guardrail.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

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
    def test_weak_match_passes(self):
        """"youth empowerment by AI" was declined by the old floor."""
        ok, diag = evidence_gate.evidence_is_sufficient([frag(0.60)], surface="t")
        assert ok
        assert diag["reason"] == "fragments_retrieved"

    def test_empty_pool_declines(self):
        ok, diag = evidence_gate.evidence_is_sufficient([], surface="t")
        assert not ok
        assert diag["reason"] == "no_fragments_retrieved"

    def test_unscored_pool_passes(self):
        assert evidence_gate.evidence_is_sufficient([{"fragment_id": "f1"}], surface="t")[0]

    def test_diagnostics_carry_what_the_decline_payload_needs(self):
        _, diag = evidence_gate.evidence_is_sufficient([frag(0.5), frag(0.4)], surface="t")
        assert diag["top_similarity"] == 0.5
        assert diag["fragments_considered"] == 2


class TestRelay:
    def test_weak_evidence_still_generates(self):
        with patch.object(comparison_service.embedding_index, "search",
                          return_value=[frag(0.60, fragment_id="f1", speaker="A", text="t")]), \
             patch.object(comparison_service.guardrail_service, "assert_query_allowed"), \
             patch.object(comparison_service, "generate_json",
                          return_value={"relay_title": "T", "relay_narrative": "N",
                                        "relay_steps": []}) as llm:
            result = comparison_service.build_relay("youth empowerment by AI")
        assert result["outcome"] == OUTCOME_GENERATED
        llm.assert_called_once()

    def test_empty_retrieval_declines_without_calling_the_llm(self):
        with patch.object(comparison_service.embedding_index, "search", return_value=[]), \
             patch.object(comparison_service.guardrail_service, "assert_query_allowed"), \
             patch.object(comparison_service, "generate_json") as llm:
            result = comparison_service.build_relay("anything")
        assert result["outcome"] == OUTCOME_INSUFFICIENT_EVIDENCE
        assert result["relay_steps"] == []
        llm.assert_not_called()


class TestThemeSynthesisDeclines:
    def test_weak_semantic_fallback_still_generates(self):
        # Facet search finds nothing, fallback returns a weak match.
        with patch.object(comparison_service.embedding_index, "search",
                          side_effect=[[], [frag(0.55, fragment_id="f1")]]), \
             patch.object(comparison_service.guardrail_service, "assert_query_allowed"), \
             patch.object(comparison_service, "generate_json",
                          return_value={"title": "T", "summary": "S",
                                        "key_points": [], "representative_clips": []}) as llm:
            result = comparison_service.synthesize_theme("anything")
        assert result["outcome"] == OUTCOME_GENERATED
        llm.assert_called_once()

    def test_facet_match_orders_by_similarity(self):
        """Theme options come from the taxonomy; a classified fragment is
        evidence even when the label scores low against transcript text."""
        facet_hits = [frag(0.40, fragment_id="low"), frag(0.45, fragment_id="high")]
        with patch.object(comparison_service.embedding_index, "search",
                          return_value=facet_hits) as search, \
             patch.object(comparison_service.guardrail_service, "assert_query_allowed"), \
             patch.object(comparison_service, "generate_json",
                          return_value={"title": "T", "summary": "S",
                                        "key_points": [], "representative_clips": []}) as llm:
            result = comparison_service.synthesize_theme("AI in Health, Science & Environment")
        assert result["outcome"] == OUTCOME_GENERATED
        assert [p["fragment_id"] for p in result["evidence_pool"]] == ["high", "low"]
        assert result["_explain"]["retrieval"]["method"] == "theme_facet_match"
        search.assert_called_once()
        llm.assert_called_once()

    def test_declines_with_a_200_shape_not_a_valueerror(self):
        """It used to raise ValueError, which the router surfaced as a 422."""
        with patch.object(comparison_service.embedding_index, "search", return_value=[]), \
             patch.object(comparison_service.guardrail_service, "assert_query_allowed"), \
             patch.object(comparison_service, "generate_json"):
            result = comparison_service.synthesize_theme("anything")
        assert result["outcome"] == OUTCOME_INSUFFICIENT_EVIDENCE
