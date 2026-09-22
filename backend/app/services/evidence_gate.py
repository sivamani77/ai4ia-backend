"""Relevance check for archive queries.

ANN search always returns its top-k however poor the match, so the query
services' `if not fragments` checks never fire. This is the gate that makes
them fire. Gibberish is out of scope — it scores like a genuine query and is
handled by the guardrail's no-subject rule.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from app.core.settings import settings

logger = logging.getLogger("ai4ia.services.evidence_gate")


def top_similarity(fragments: Sequence[Mapping[str, Any]]) -> float | None:
    """Highest similarity in the pool, or None if nothing is scored.

    embedding_index.search sets `similarity`, _semantic_rank_via_chroma sets
    `semantic_score`; both are 1 - cosine distance.
    """
    scores = []
    for fragment in fragments or []:
        value = fragment.get("similarity")
        if value is None:
            value = fragment.get("semantic_score")
        if value is not None:
            try:
                scores.append(float(value))
            except (TypeError, ValueError):
                continue
    return max(scores) if scores else None


def evidence_is_sufficient(
    fragments: Sequence[Mapping[str, Any]],
    *,
    surface: str,
    floor: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    """(sufficient, diagnostics). Logs top-1 either way so the floor is tunable."""
    threshold = settings.archive_relevance_floor if floor is None else floor
    best = top_similarity(fragments)

    diagnostics: dict[str, Any] = {
        "top_similarity": best,
        "relevance_floor": threshold,
        "fragments_considered": len(fragments or []),
    }

    if not fragments:
        diagnostics["reason"] = "no_fragments_retrieved"
        logger.info("Evidence gate: %s retrieved nothing", surface)
        return False, diagnostics

    # An unscored pool can't be judged, so let it through rather than block a
    # caller that retrieved by facet instead of similarity.
    if best is None:
        diagnostics["reason"] = "no_similarity_scores_to_judge"
        return True, diagnostics

    sufficient = best >= threshold
    diagnostics["reason"] = "above_floor" if sufficient else "below_floor"
    logger.info(
        "Evidence gate: %s top_similarity=%.4f floor=%.2f -> %s",
        surface, best, threshold, "pass" if sufficient else "decline",
    )
    return sufficient, diagnostics


__all__ = ["evidence_is_sufficient", "top_similarity"]
