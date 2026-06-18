"""Insight generation — evidence-grounded cross-fragment synthesis.

Citation-first design: the LLM may only cite from a numbered evidence pool we
build via semantic retrieval, and every generated claim must carry
`fragment_refs` into that pool. After generation we validate the refs against
the real fragments, drop dangling ones, and report an `evidence_coverage`
score. This is what separates a grounded synthesis from a plausible paragraph.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import INSIGHT_MODEL, INSIGHT_DEFAULT_LIMIT
from app.core.paths import INSIGHTS_DIR
from app.services import embedding_index, fragment_store
from app.services.llm_service import generate_json


def _truncate(text: str, n: int = 600) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "..."


def build_evidence_pool(query: str, filters: dict | None, limit: int) -> list[dict[str, Any]]:
    """Semantic-retrieve fragments and number them [1..N] as the allowed citation pool."""
    frags = embedding_index.search(query, top_k=limit, filters=filters)
    pool = []
    for i, f in enumerate(frags, start=1):
        pool.append({
            "ref": i,
            "fragment_id": f.get("fragment_id"),
            "speaker": f.get("speaker"),
            "speaker_attribution": f.get("speaker_attribution"),
            "edition": f.get("edition"),
            "year": f.get("year"),
            "primary_theme": f.get("primary_theme"),
            "primary_chapter": f.get("primary_chapter"),
            "start": f.get("start"),
            "end": f.get("end"),
            "duration": f.get("duration"),
            "stream_url": f.get("stream_url"),
            "similarity": f.get("similarity"),
            "text": f.get("text"),
        })
    return pool


_PROMPT = """You are a research analyst synthesizing insights from the AI4IA conference archive.

Question: {query}

You are given NUMBERED evidence fragments. You may ONLY use these fragments.
EVERY statement you make MUST cite the fragment numbers that support it via a
"fragment_refs" array (e.g. [1, 4]). Do NOT use any number not listed below.
Do NOT invent facts. If the evidence is thin, say so and cite what exists.

EVIDENCE:
{evidence}

Return JSON exactly in this shape:
{{
  "title": "<short evocative title>",
  "summary": "<2-4 sentence overview>",
  "patterns":  [{{"statement": "<recurring idea>", "fragment_refs": [1,2]}}],
  "contrasts": [{{"statement": "<tension or disagreement>", "fragment_refs": [3,4]}}],
  "recommended_clips": [{{"ref": 1, "reason": "<why this clip is representative>"}}]
}}"""


def _format_evidence(pool: list[dict]) -> str:
    lines = []
    for p in pool:
        meta = " | ".join(filter(None, [
            p.get("speaker"), p.get("edition"),
            p.get("primary_theme"),
        ]))
        lines.append(f"[{p['ref']}] ({meta}) {_truncate(p['text'])}")
    return "\n".join(lines)


def _validate(claims: list[dict], valid_refs: set[int]) -> tuple[list[dict], list[dict]]:
    """Partition claims into supported (>=1 resolvable ref) and unsupported."""
    supported, unsupported = [], []
    for c in claims or []:
        refs = [r for r in (c.get("fragment_refs") or []) if isinstance(r, int) and r in valid_refs]
        c = {**c, "fragment_refs": refs, "support_count": len(refs)}
        (supported if refs else unsupported).append(c)
    return supported, unsupported


def generate_insight(query: str, filters: dict | None = None, limit: int | None = None) -> dict:
    limit = limit or INSIGHT_DEFAULT_LIMIT
    pool = build_evidence_pool(query, filters, limit)
    if not pool:
        raise ValueError("No fragments retrieved for this query/filters.")

    valid_refs = {p["ref"] for p in pool}
    raw = generate_json(
        _PROMPT.format(query=query, evidence=_format_evidence(pool)),
        model=INSIGHT_MODEL,
        temperature=0.2,
    )

    patterns, p_bad = _validate(raw.get("patterns"), valid_refs)
    contrasts, c_bad = _validate(raw.get("contrasts"), valid_refs)
    unsupported = p_bad + c_bad

    # recommended_clips reference the pool by ref -> resolve to fragment_ids
    rec = []
    for r in raw.get("recommended_clips") or []:
        ref = r.get("ref")
        if isinstance(ref, int) and ref in valid_refs:
            frag = next(p for p in pool if p["ref"] == ref)
            rec.append({"fragment_id": frag["fragment_id"], "ref": ref,
                        "reason": r.get("reason"), "stream_url": frag["stream_url"],
                        "start": frag["start"], "end": frag["end"]})

    total_claims = len(patterns) + len(contrasts) + len(unsupported)
    coverage = round((len(patterns) + len(contrasts)) / total_claims, 3) if total_claims else 0.0

    insight_id = f"insight_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    insight = {
        "insight_id": insight_id,
        "title": raw.get("title") or query[:80],
        "query": query,
        "filters": filters or {},
        "summary": raw.get("summary"),
        "patterns": patterns,
        "contrasts": contrasts,
        "recommended_clips": rec,
        "unsupported_claims": unsupported,
        "evidence_coverage": coverage,
        "evidence_pool": pool,
        "models": {"llm": INSIGHT_MODEL, "embedding": "nomic-embed-text:v1.5"},
        "created_at": datetime.utcnow().isoformat(),
    }
    _save(insight)
    return insight


def _save(insight: dict) -> None:
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    (INSIGHTS_DIR / f"{insight['insight_id']}.json").write_text(
        json.dumps(insight, ensure_ascii=False, indent=2), encoding="utf-8")


def list_insights() -> list[dict]:
    if not INSIGHTS_DIR.exists():
        return []
    out = []
    for p in sorted(INSIGHTS_DIR.glob("insight_*.json"), reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append({"insight_id": d["insight_id"], "title": d.get("title"),
                        "query": d.get("query"), "created_at": d.get("created_at"),
                        "evidence_coverage": d.get("evidence_coverage")})
        except Exception:  # noqa: BLE001
            continue
    return out


def get_insight(insight_id: str) -> dict | None:
    path = INSIGHTS_DIR / f"{insight_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_sources_map(insight: dict) -> dict:
    """Claim -> resolved evidence clips (the evidence map view)."""
    by_ref = {p["ref"]: p for p in insight.get("evidence_pool", [])}

    def resolve(claim: dict) -> dict:
        return {
            "statement": claim.get("statement"),
            "sources": [_clip(by_ref[r]) for r in claim.get("fragment_refs", []) if r in by_ref],
        }

    return {
        "insight_id": insight.get("insight_id"),
        "evidence_coverage": insight.get("evidence_coverage"),
        "patterns": [resolve(c) for c in insight.get("patterns", [])],
        "contrasts": [resolve(c) for c in insight.get("contrasts", [])],
    }


def _clip(p: dict) -> dict:
    return {
        "fragment_id": p["fragment_id"], "speaker": p.get("speaker"),
        "speaker_attribution": p.get("speaker_attribution"),
        "edition": p.get("edition"), "year": p.get("year"),
        "video_stream_url": p.get("stream_url"),
        "start": p.get("start"), "end": p.get("end"), "duration": p.get("duration"),
        "text": _truncate(p.get("text", ""), 400),
    }


def storyline_evidence_map(storyline: dict) -> dict:
    """Map an existing storyline's selected_candidates to playable evidence clips.

    Read-only over storyline JSON; resolves each candidate against the store so
    speaker/edition enrichment is included.
    """
    clips = []
    for cand in storyline.get("selected_candidates", []):
        fid = cand.get("fragment_id")
        frag = fragment_store.get(fid) or {}
        clips.append({
            "fragment_id": fid,
            "speaker": frag.get("speaker") or cand.get("speaker"),
            "speaker_attribution": frag.get("speaker_attribution"),
            "edition": frag.get("edition"),
            "year": frag.get("year"),
            "primary_theme": cand.get("primary_theme") or frag.get("primary_theme"),
            "primary_chapter": cand.get("primary_chapter") or frag.get("primary_chapter"),
            "video_stream_url": f"/api/archive/videos/{cand.get('source_video')}/stream",
            "start": cand.get("start"), "end": cand.get("end"),
            "duration": cand.get("duration"),
            "summary": cand.get("summary"),
            "resolved": bool(frag),
        })
    return {
        "user_prompt": storyline.get("user_prompt"),
        "candidate_count": len(clips),
        "resolved_count": sum(1 for c in clips if c["resolved"]),
        "evidence": clips,
    }
