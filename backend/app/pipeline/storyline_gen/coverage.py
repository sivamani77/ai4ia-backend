"""L2b coverage: can the archive answer this prompt? (docs/intent-layer-design.md §8)

A rule decides the clear cases from the retrieved fragments; the self-hosted
judge decides the rest. A judge failure never blocks a render.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from app.core.error_messages import DECLINE_MESSAGES
from app.pipeline.prompt_safety import UNTRUSTED_CONTENT_NOTICE, wrap_untrusted
from app.pipeline.storyline_gen.candidate_scoring import candidate_search_blob
from app.pipeline.storyline_gen.text_utils import normalize_name

# Intents whose prompt carries a topic the archive may not cover. Speaker,
# title and overview requests were already confirmed by the resolve stage.
CHECKED_INTENTS = frozenset({"topic", "chronological", "speaker_topic", "speaker_panel"})
JUDGE_FRAGMENTS = 10
# A topic term must appear among this many best matches. Measured: a passing
# mention of "quantum" somewhere in 90 retrieved fragments made "quantum
# computing in agriculture" look covered; its top 10 hold neither term.
TERM_WINDOW = 10
_MAX_ASPECTS = 3
_MAX_ASPECT_CHARS = 60

JUDGE_SYSTEM = """You check whether an archive of conference talks discusses what a request asks about.
You get the request and the archive fragments that match it best.

1. List the subjects the request names: 1 to 3 short noun phrases (at most 6 words).
   Leave out "AI" on its own, speakers' names, and sub-topics the request does not name.
2. For each subject, decide whether the fragments discuss it: at least one fragment
   spends a sentence or more on the subject itself. A passing mention, a shared
   word (an AI talk saying "history"), or an example used to make a point about
   something else is not discussion.

Return JSON only:
{"subjects": [{"subject": "...", "discussed": true, "fragment_ids": ["ids from the list"]}]}
""" + UNTRUSTED_CONTENT_NOTICE


@dataclass
class Coverage:
    verdict: str                       # covered | partial | not_covered
    method: str                        # not_checked | rule | judge | judge_failed
    top_similarity: float | None = None
    floor: float | None = None
    low: float | None = None
    pool_size: int = 0
    missing_terms: list[str] = field(default_factory=list)
    addressed_aspects: list[str] = field(default_factory=list)
    missing_aspects: list[str] = field(default_factory=list)
    supporting_ids: list[str] = field(default_factory=list)
    reason_code: str | None = None
    user_message: str | None = None
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _score(candidate: dict[str, Any]) -> float:
    try:
        return float(candidate.get("semantic_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def evidence_pool(intent_name: str, profile: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The fragments the verdict is about, best match first: the speaker's own
    for speaker_topic, their panels' for speaker_panel, all retrieved otherwise."""
    speaker = profile.get("requested_speaker")
    if intent_name == "speaker_topic" and speaker:
        from app.services.speaker_registry import get_registry

        registry = get_registry()
        pool = [c for c in candidates
                if (c.get("speaker_match") and not c.get("speaker_match_panel"))
                or registry.canonical(str(c.get("speaker") or "")) == speaker]
    elif intent_name == "speaker_panel":
        pool = [c for c in candidates if c.get("speaker_match_panel")]
    else:
        pool = list(candidates)
    return sorted(pool, key=_score, reverse=True)


# Endings a stemmed topic term may take: "deepfake" -> "deepfakes",
# "indigenou" -> "indigenous", never "roman" -> "romance".
_ENDINGS = r"(?:s|es|ed|ing|ings|er|ers|ion|ions|al|ally|ly|y|ies|ic|ics|ical|ity|ities|ation|ations|e|ence|ent|ents)?"


def term_pattern(term: str) -> re.Pattern | None:
    stem = normalize_name(term)
    return re.compile(rf"\b{re.escape(stem)}{_ENDINGS}\b", re.I) if stem else None


def has_term(candidate: dict[str, Any], term: str) -> bool:
    """The fragment's text or metadata holds the term (topic terms are stems)."""
    pattern = term_pattern(term)
    return bool(pattern and pattern.search(normalize_name(candidate_search_blob(candidate))))


def prompt_phrases(prompt: str, terms: list[str]) -> list[str]:
    """Topic terms the prompt puts side by side ("space law"): they must match as a phrase."""
    patterns = [p for p in (term_pattern(t) for t in terms) if p]
    words = normalize_name(prompt).split()
    is_term = [any(p.fullmatch(w) for p in patterns) for w in words]
    return [f"{a} {b}" for a, b, ta, tb in zip(words, words[1:], is_term, is_term[1:]) if ta and tb]


def missing_terms(pool: list[dict[str, Any]], terms: list[str], phrases: list[str] = ()) -> list[str]:
    """Topic terms and phrases that none of the best-matching fragments contains."""
    window = [normalize_name(candidate_search_blob(c)) for c in pool[:TERM_WINDOW]]
    missing = [t for t in terms if not any(has_term(c, t) for c in pool[:TERM_WINDOW])]
    for phrase in phrases:
        pattern = re.compile(r"\s+".join(term_pattern(w).pattern for w in phrase.split()), re.I)
        if not any(pattern.search(text) for text in window):
            missing.append(phrase)
    return missing


def rule_verdict(top1: float | None, missing: list[str], floor: float, low: float) -> str:
    """covered / not_covered when the numbers are clear, otherwise "judge"."""
    if top1 is None:
        return "judge"
    if top1 < low:
        return "not_covered"
    if top1 >= floor and not missing:
        return "covered"
    return "judge"


def term_evidence(term: str, fragments: list[dict[str, Any]], taken: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The fragment whose spoken text mentions the term most."""
    pattern = term_pattern(term)
    best, best_count = None, 0
    for c in fragments:
        count = len(pattern.findall(str(c.get("text") or ""))) if pattern else 0
        if count > best_count and all(c is not t for t in taken):
            best, best_count = c, count
    return best


def _judge_fragments(pool: list[dict[str, Any]], terms: list[str],
                     search_space: list[dict[str, Any]] | None = None) -> list[tuple[dict[str, Any], list[str]]]:
    """(fragment, terms its snippet centres on): for each topic term the fragment
    that says most about it, then the best matches overall. For "deepfakes and
    space law" the retrieved pool leant to law and space; the deepfake talks
    were only found by searching the whole archive for the term."""
    per_term: list[tuple[dict[str, Any], list[str]]] = []
    for term in terms[:JUDGE_FRAGMENTS // 2]:
        best = term_evidence(term, search_space or pool, [p for p, _ in per_term])
        if best is not None:
            per_term.append((best, [term]))
    ids = {str(p.get("fragment_id")) for p, _ in per_term}
    rest = [(c, terms) for c in pool if str(c.get("fragment_id")) not in ids]
    return rest[:JUDGE_FRAGMENTS - len(per_term)] + per_term


def _clean_aspects(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(v).strip()[:_MAX_ASPECT_CHARS] for v in values if str(v).strip()][:_MAX_ASPECTS]


def parse_judge(text: str, allowed_ids: set[str]) -> dict[str, Any] | None:
    """The verdict from the judge's per-subject answer, or None when it is unusable.
    Code, not the model, turns subjects into covered / partial / not_covered."""
    from app.pipeline.storyline_gen.storyline_llm import extract_json_object

    data = extract_json_object(text or "")
    subjects = data.get("subjects") if isinstance(data, dict) else None
    if not isinstance(subjects, list):
        return None
    subjects = [s for s in subjects if isinstance(s, dict) and str(s.get("subject") or "").strip()]
    if not subjects:
        return None
    discussed = _clean_aspects([s["subject"] for s in subjects if s.get("discussed") is True])
    missing = _clean_aspects([s["subject"] for s in subjects if s.get("discussed") is not True])
    verdict = "covered" if not missing else "not_covered" if not discussed else "partial"
    return {
        "coverage": verdict,
        "addressed_aspects": discussed,
        "missing_aspects": missing,
        # Only ids it was shown; anything else is invented.
        "supporting_ids": list(dict.fromkeys(
            str(i) for s in subjects if s.get("discussed") is True
            for i in (s.get("fragment_ids") or []) if str(i) in allowed_ids)),
    }


_SNIPPET_CHARS = 500


def snippet(candidate: dict[str, Any], terms: list[str]) -> str:
    """The fragment's text around its first topic term, so the judge sees the term."""
    text = str(candidate.get("text") or candidate.get("summary") or "")
    hits = [m.start() for t in terms if term_pattern(t)
            for m in [term_pattern(t).search(text)] if m]
    start = max(0, min(hits) - _SNIPPET_CHARS // 3) if hits else 0
    return text[start:start + _SNIPPET_CHARS]


def judge(prompt: str, fragments: list[tuple[dict[str, Any], list[str]]],
          call_llm: Callable[..., str]) -> dict[str, Any] | None:
    shown = [{"id": str(c.get("fragment_id")),
              "speaker": c.get("speaker") or c.get("speaker_attribution") or "",
              "text": snippet(c, focus)} for c, focus in fragments]
    user = ("REQUEST: " + wrap_untrusted(prompt) + "\n\nFRAGMENTS:\n"
            + wrap_untrusted(json.dumps(shown, ensure_ascii=False, indent=1)))
    try:
        text = call_llm(user, timeout=90, temperature=0.0, max_tokens=4096, system=JUDGE_SYSTEM,
                        reasoning_effort="medium")
    except Exception:
        return None
    return parse_judge(text, {s["id"] for s in shown})


def topic_phrase(profile: dict[str, Any], aspects: list[str]) -> str:
    """What the request is about, for a user-facing message: the judge's
    aspects, else the prompt's own topic words. Never the raw prompt."""
    if aspects:
        phrase = ", ".join(aspects)
    else:
        words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", profile.get("raw_prompt") or "")
        terms = [t.lower() for t in profile.get("topic_terms") or []]
        phrase = " ".join(w for w in words if any(w.lower().startswith(t) for t in terms)) or "this topic"
    return phrase[:_MAX_ASPECT_CHARS * 2]


def top_themes(pool: list[dict[str, Any]], n: int = 3) -> list[str]:
    counts = Counter(str(c.get("primary_theme") or "").strip() for c in pool[:30])
    counts.pop("", None)
    return [name for name, _ in counts.most_common(n)]


def decline(intent_name: str, profile: dict[str, Any], pool: list[dict[str, Any]],
            aspects: list[str], nearest: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
    """(reason_code, user_message, suggestions) for a request the archive cannot answer."""
    speaker = profile.get("requested_speaker")
    if speaker and intent_name in ("speaker_topic", "speaker_panel"):
        # The prompt's own topic words: the judge's aspects restate the speaker.
        themes = top_themes(pool) or ["other subjects"]
        message = DECLINE_MESSAGES["speaker_topic_not_covered"].format(
            name=speaker, topic=topic_phrase(profile, []), themes="; ".join(themes))
        return "speaker_topic_not_covered", message, [f"What did {speaker} say?"]
    themes = top_themes(nearest)
    message = DECLINE_MESSAGES["topic_not_covered"].format(
        topic=topic_phrase(profile, aspects[:1]),
        themes="; ".join(themes) if themes else "AI ethics, information access and governance")
    return "topic_not_covered", message, themes


def assess(intent_name: str, profile: dict[str, Any], candidates: list[dict[str, Any]], prompt: str,
           call_llm: Callable[..., str], floor: float, low: float,
           records: list[dict[str, Any]] | None = None) -> Coverage:
    """`records` is the whole archive; topic requests search it for each term's evidence."""
    pool = evidence_pool(intent_name, profile, candidates)
    top1 = _score(pool[0]) if pool else None
    terms = list(profile.get("topic_terms") or [])
    result = Coverage(verdict="covered", method="not_checked", top_similarity=top1, floor=floor, low=low,
                      pool_size=len(pool))
    if intent_name not in CHECKED_INTENTS:
        return result
    if not pool:
        # No scored evidence to judge: never decline on missing data.
        result.method = "no_evidence_pool"
        return result

    result.missing_terms = missing_terms(pool, terms, prompt_phrases(prompt, terms))
    verdict = rule_verdict(top1, result.missing_terms, floor, low)
    if verdict != "judge":
        result.verdict, result.method = verdict, "rule"
    else:
        # A speaker's evidence stays within their own fragments.
        search_space = records if intent_name in ("topic", "chronological") else pool
        judged = judge(prompt, _judge_fragments(pool, terms, search_space), call_llm)
        if judged is None:
            # Judge down or unusable: render, as before this check existed.
            result.method = "judge_failed"
        else:
            result.method = "judge"
            result.verdict = judged["coverage"]
            result.addressed_aspects = judged["addressed_aspects"]
            result.missing_aspects = judged["missing_aspects"]
            result.supporting_ids = judged["supporting_ids"]

    if result.verdict == "partial" and not result.missing_aspects:
        result.verdict = "covered"  # a gap nobody can name cannot be narrated
    if result.verdict == "not_covered":
        result.reason_code, result.user_message, result.suggestions = decline(
            intent_name, profile, pool, result.missing_aspects, candidates)
    return result
