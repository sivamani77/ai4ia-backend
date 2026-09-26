"""Evidence check for archive queries.

Declines only when retrieval returns nothing. A similarity floor used to
decline weak matches too, but it refused genuine queries ("youth
empowerment by AI"), so it was removed. Top-1 is still logged. Gibberish
is the guardrail's job (its no-subject rule).
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence


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
) -> tuple[bool, dict[str, Any]]:
    """(sufficient, diagnostics). Logs top-1 so retrieval quality stays visible."""
    best = top_similarity(fragments)
    diagnostics: dict[str, Any] = {
        "top_similarity": best,
        "fragments_considered": len(fragments or []),
    }

    if not fragments:
        diagnostics["reason"] = "no_fragments_retrieved"
        logger.info("Evidence gate: %s retrieved nothing", surface)
        return False, diagnostics

    diagnostics["reason"] = "fragments_retrieved"
    logger.info("Evidence gate: %s top_similarity=%s -> pass", surface, best)
    return True, diagnostics


__all__ = ["evidence_is_sufficient", "top_similarity"]
