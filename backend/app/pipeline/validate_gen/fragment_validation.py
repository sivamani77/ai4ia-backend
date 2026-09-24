"""Per-fragment editorial validation: LLM call, prompt building, and the score-based fallback."""
import asyncio
import re
from typing import Any

import httpx

from app.pipeline.prompt_safety import UNTRUSTED_CONTENT_NOTICE, wrap_untrusted
from app.pipeline.validate_gen.json_utils import extract_json_object
from app.pipeline.validate_gen.llm_client import STRONG_LLM_MODEL, log, post_with_retries, post_with_retries_async
from app.pipeline.validate_gen.narrative_templates import DEEPFAKE_VOICEOVER_DEFAULTS, GENERIC_VOICEOVER_DEFAULTS
from app.pipeline.validate_gen.transcript import context_window_to_prompt_text

# Weak trailing words that signal an incomplete quote
WEAK_TRAILING_WORDS = frozenset({
    "and", "or", "but", "to", "of", "with", "because", "through",
    "that", "which", "for", "as", "so", "the", "a", "an", "in",
    "on", "at", "by", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "will", "would", "could", "should",
})

# Voiceover instruction prefixes to strip
INSTRUCTION_PREFIX_PATTERNS = [
    re.compile(r"^End with\s+(?:a\s+)?(?:clear\s+)?(?:takeaway|statement|synthesis|message)[:\s]+", re.I),
    re.compile(r"^Explain\s+(?:that|how|why|what)\s+", re.I),
    re.compile(r"^Mention\s+(?:that|how|the)\s+", re.I),
    re.compile(r"^Show\s+how\s+", re.I),
    re.compile(r"^This\s+section\s+(?:should|will|must)\s+", re.I),
    re.compile(r"^The\s+(?:video|film|documentary)\s+should\s+", re.I),
    re.compile(r"^Use\s+this\s+clip\s+to\s+", re.I),
    re.compile(r"^The\s+narrative\s+should\s+", re.I),
    re.compile(r"^Start\s+with\s+", re.I),
    re.compile(r"^Begin\s+with\s+", re.I),
    re.compile(r"^Open\s+with\s+", re.I),
    re.compile(r"^Close\s+with\s+a\s+(?:clear\s+)?(?:takeaway|statement)[:\s]+", re.I),
    re.compile(r"^Conclude\s+(?:with\s+)?(?:a\s+)?(?:clear\s+)?(?:takeaway|statement)[:\s]+", re.I),
]

CANDIDATE_VALIDATION_SYSTEM = """You are an editorial director for a short documentary video.

Your job is to evaluate a single candidate fragment and decide how it should be used.

RULES:
- A speaker_clip must contain a COMPLETE thought or sentence. Do not approve clips that start or end mid-sentence unless the window can be expanded using the transcript context.
- If the fragment is incomplete but can be expanded using the surrounding transcript, provide new quote_start / quote_end that capture a complete thought.
- Use narration_broll when: the content is relevant but the speaker quote is unclear, incomplete, or low-quality audio.
- Use visual_support when: the content provides good background video but the speech itself is not needed.
- Reject clips that are: generic greetings, off-topic, repetitive of other clips, or have very low informational value.
- The final video must follow the narrative structure provided: hook → context → evidence/impact → challenges → responses → synthesis.

OUTPUT: Respond with ONLY a valid JSON object. No markdown, no explanation outside the JSON.
""" + UNTRUSTED_CONTENT_NOTICE


def _extract_speaker_from_video_title(source_video: str) -> str:
    """Extract a human-readable speaker name from a source_video ID string.

    e.g. '0124_ER7_5jzIrbI_Jaimee Stuart  - Women, Peace...' → 'Jaimee Stuart'
    """
    if not source_video:
        return ""
    # Strip leading numeric ID + video_id prefix (e.g. '0124_ER7_5jzIrbI_')
    title = re.sub(r"^[0-9]+_[A-Za-z0-9_-]+_", "", source_video.strip())
    # Split on first dash/em-dash separator
    parts = re.split(r"\s*[—–\-]\s*", title, maxsplit=1)
    name_part = parts[0].strip()
    # Clean up multiple spaces (file names can have double spaces)
    name_part = re.sub(r"\s+", " ", name_part).strip()
    # Accept if it looks like a human name: 2-4 words, each word starts uppercase
    words = [w for w in name_part.split() if w]
    if 1 < len(words) <= 5 and all(w[0].isupper() for w in words if w):
        return name_part[:60]
    # Fallback: take first word if it's a reasonable length name
    if words and words[0][0].isupper() and len(words[0]) >= 3:
        return words[0][:40]
    return ""


_SECTION_ID_PASS_SUFFIX_RE = re.compile(r"_\d+$")


def _base_section_id(section_id: str) -> str:
    """Strip the trailing "_<pass_number>" detect_narrative_sections adds when
    it repeats the base arc to cover a longer target_duration (e.g. S1_HOOK_2
    -> S1_HOOK) — the topic-default dicts below are keyed by the original,
    unsuffixed section ids only, so a repeated section must map back to its
    base id to find a default at all."""
    return _SECTION_ID_PASS_SUFFIX_RE.sub("", section_id or "")


def _clean_voiceover_text(
    text: str,
    section_id: str,
    is_deepfake_topic: bool = False,
) -> str:
    """Strip instruction phrases from voiceover text and apply topic defaults if needed."""
    base_section_id = _base_section_id(section_id)
    if not text:
        defaults = DEEPFAKE_VOICEOVER_DEFAULTS if is_deepfake_topic else GENERIC_VOICEOVER_DEFAULTS
        return defaults.get(base_section_id, "")

    cleaned = text.strip()

    # Apply instruction prefix patterns
    for pat in INSTRUCTION_PREFIX_PATTERNS:
        m = pat.match(cleaned)
        if m:
            remainder = cleaned[m.end():].strip()
            if remainder:
                # Capitalise first character of remainder
                cleaned = remainder[0].upper() + remainder[1:]
            else:
                # Nothing left — use topic default
                defaults = DEEPFAKE_VOICEOVER_DEFAULTS if is_deepfake_topic else GENERIC_VOICEOVER_DEFAULTS
                cleaned = defaults.get(base_section_id, cleaned)
            break

    # If still looks like instruction text with no factual content, use default
    instruction_verbs = ("explain", "mention", "show", "describe", "illustrate", "demonstrate")
    words = cleaned.lower().split()
    if words and words[0] in instruction_verbs:
        defaults = DEEPFAKE_VOICEOVER_DEFAULTS if is_deepfake_topic else GENERIC_VOICEOVER_DEFAULTS
        cleaned = defaults.get(base_section_id, cleaned)

    # Ensure it ends with a period
    if cleaned and not cleaned.rstrip()[-1:] in ".!?":
        cleaned = cleaned.rstrip() + "."

    return cleaned


def _quote_ends_completely(text: str) -> bool:
    """Return True if text ends at a sentence boundary."""
    t = (text or "").rstrip()
    if not t:
        return False
    if not t[-1] in ".!?\"'":
        return False
    last_word = t.rstrip(".!?\"' ").split()[-1].lower() if t.rstrip(".!?\"' ").split() else ""
    if last_word in WEAK_TRAILING_WORDS:
        return False
    return True


def _quote_starts_cleanly(text: str) -> bool:
    """Return True if text starts at a sentence boundary (not mid-sentence)."""
    t = (text or "").lstrip()
    if not t:
        return False
    return t[0].isupper()


def _measured(values: dict[str, Any], template: str) -> str:
    """Format a measurement, or say it was not measured. Zeros for a missing
    measurement read as "unusable audio" and got good clips demoted."""
    try:
        return template.format(**values)
    except (KeyError, ValueError, TypeError):
        return "not measured (no known audio problem)"


def _build_validation_prompt(
    candidate: dict[str, Any],
    context_window: list[dict[str, Any]],
    narrative_sections: list[dict[str, Any]],
    user_prompt: str,
    target_duration: int,
) -> tuple[str, str, str]:
    """Build the per-fragment validation prompt. Returns (system, user, fragment_id).

    user_prompt, the candidate's transcript text, and the surrounding
    transcript are all untrusted content (user-submitted or sourced from
    ingested video) — each is wrapped via wrap_untrusted() rather than
    concatenated raw into the prompt, so it can never be mistaken for an
    instruction by the model.
    """
    frag_id = candidate.get("fragment_id", "?")
    frag_start = float(candidate.get("start") or 0)
    frag_end = float(candidate.get("end") or frag_start + 10)
    text = candidate.get("text") or candidate.get("moment_text") or ""
    speaker = candidate.get("speaker") or candidate.get("speaker_attribution") or ""
    speech_quality = candidate.get("speech_quality") or {}
    moment_scores = candidate.get("moment_scores") or {}

    context_text = context_window_to_prompt_text(context_window, frag_start, frag_end)

    sections_summary = "\n".join(
        f"  {s['section_id']}: {s['title']} — {s['purpose'][:80]}"
        for s in narrative_sections
    )

    user = f"""USER PROMPT: {wrap_untrusted(user_prompt)}
TARGET VIDEO DURATION: {target_duration} seconds

NARRATIVE SECTIONS:
{sections_summary}

CANDIDATE FRAGMENT:
  fragment_id: {frag_id}
  speaker: {speaker}
  start: {frag_start:.2f}s
  end: {frag_end:.2f}s
  duration: {frag_end - frag_start:.2f}s
  text: {wrap_untrusted(text)}
  speech_quality: {_measured(speech_quality, "speech_ratio={speech_ratio:.2f}, is_usable={is_usable_speech}")}
  moment_scores: {_measured(moment_scores, "completeness={quote_completeness_score:.2f}, final={final_moment_score:.2f}")}

SURROUNDING TRANSCRIPT (lines marked <<<FRAGMENT>>> are the selected fragment):
{wrap_untrusted(context_text) if context_text else "  (no transcript available)"}

Decide how to use this fragment. If the fragment text starts or ends mid-sentence, check if the surrounding transcript extends it into a complete thought, and if so, provide expanded quote_start/quote_end.

Return JSON with these exact fields:
{{
  "fragment_id": "{frag_id}",
  "decision": "speaker_clip | narration_broll | visual_support | reject",
  "section_id": "S1_HOOK | S2_CONTEXT | S3_HUMAN_RISK | S4_PUBLIC_RISK | S5_RESPONSE | S6_SYNTHESIS",
  "approved": true or false,
  "reason": "one sentence explanation",
  "quote_complete": true or false,
  "quote_start": {frag_start:.2f},
  "quote_end": {frag_end:.2f},
  "quote_text": "exact text of the approved quote",
  "speaker_label": "{speaker}",
  "subtitle_text": "clean text for display",
  "visual_direction": "brief description of what should be on screen",
  "risk_flags": [],
  "confidence": 0.85
}}
"""
    return CANDIDATE_VALIDATION_SYSTEM, user, frag_id


def _parse_validation_result(raw: str, frag_id: str) -> dict[str, Any] | None:
    result = extract_json_object(raw)
    if not result:
        log(f"  [llm] No JSON from validation for {frag_id[:50]}")
        return None
    result["fragment_id"] = frag_id
    if "approved" not in result:
        result["approved"] = result.get("decision") not in ("reject",)
    return result


def validate_candidate_with_llm(
    candidate: dict[str, Any],
    context_window: list[dict[str, Any]],
    narrative_sections: list[dict[str, Any]],
    user_prompt: str,
    target_duration: int,
) -> dict[str, Any] | None:
    system_text, user_text, frag_id = _build_validation_prompt(
        candidate, context_window, narrative_sections, user_prompt, target_duration
    )
    try:
        raw = post_with_retries({
            "model": STRONG_LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.3,
            "max_completion_tokens": 4096,
            "stream": False,
        })
        return _parse_validation_result(raw, frag_id)
    except Exception as exc:
        log(f"  [llm] Validation failed for {frag_id[:50]}: {exc}")
        return None


async def validate_candidate_with_llm_async(
    candidate: dict[str, Any],
    context_window: list[dict[str, Any]],
    narrative_sections: list[dict[str, Any]],
    user_prompt: str,
    target_duration: int,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    system_text, user_text, frag_id = _build_validation_prompt(
        candidate, context_window, narrative_sections, user_prompt, target_duration
    )
    async with semaphore:
        try:
            raw = await post_with_retries_async({
                "model": STRONG_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
                "temperature": 0.3,
                "max_completion_tokens": 4096,
                "stream": False,
            }, client)
            return _parse_validation_result(raw, frag_id)
        except Exception as exc:
            log(f"  [llm] Validation failed for {frag_id[:50]}: {exc}")
            return None


def fallback_validation(candidate: dict[str, Any]) -> dict[str, Any]:
    """Score-based fallback when LLM is unavailable."""
    ms = candidate.get("moment_scores") or {}
    sq = candidate.get("speech_quality") or {}
    score = float(ms.get("final_moment_score") or 0)
    completeness = float(ms.get("quote_completeness_score") or 0)
    speech_ratio = float(sq.get("speech_ratio") or 0)
    frag_id = candidate.get("fragment_id", "?")
    text = candidate.get("text") or candidate.get("moment_text") or ""
    start = float(candidate.get("start") or 0)
    end = float(candidate.get("end") or start + 10)

    if score >= 0.45 and completeness >= 0.6 and speech_ratio >= 0.5:
        decision = "speaker_clip"
        approved = True
        reason = f"Score {score:.2f}, completeness {completeness:.2f} — suitable speaker clip (fallback)."
    elif score >= 0.3:
        decision = "narration_broll"
        approved = True
        reason = f"Score {score:.2f} — used as narration b-roll (fallback)."
    else:
        decision = "reject"
        approved = False
        reason = f"Score {score:.2f} too low (fallback)."

    return {
        "fragment_id": frag_id,
        "decision": decision,
        "section_id": None,
        "approved": approved,
        "reason": reason,
        "quote_complete": completeness >= 0.7,
        "quote_start": start,
        "quote_end": end,
        "quote_text": text,
        "speaker_label": candidate.get("speaker") or "",
        "subtitle_text": text,
        "visual_direction": "",
        "risk_flags": [],
        "confidence": score,
        "_fallback": True,
    }
