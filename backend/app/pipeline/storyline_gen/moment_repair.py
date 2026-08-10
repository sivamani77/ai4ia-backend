from typing import Any

from app.pipeline.storyline_gen.llm_client import log
from app.pipeline.storyline_gen.text_utils import SIDE_TOPIC_TERMS


PHASE4O_MOMENT_FIELDS = {
    "start", "end", "duration",
    "moment_text", "moment_selection_reason", "moment_scores", "speech_quality",
    "muted", "subtitle_kind",
}


def timeline_requires_moment_repair(export_timeline: list[dict]) -> bool:
    """Return True if any fragment-linked item is missing Phase 4O moment fields."""
    for item in export_timeline:
        fid = item.get("fragment_id")
        if not fid:
            continue
        # Null / missing type
        if not item.get("type") and not item.get("timeline_type"):
            return True
        # timeline_type present but type missing (raw LLM item)
        if item.get("timeline_type") and not item.get("type"):
            return True
        # Null start/end
        if item.get("start") is None or item.get("end") is None:
            return True
        # Any Phase 4O field missing
        for field in PHASE4O_MOMENT_FIELDS:
            if field not in item:
                return True
    return False


def _narration_broll_items_from_sections(storyline: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn storyline.sections' written narration into narration_broll timeline items.

    build_export_timeline_from_candidates only ever emits fragment-linked
    segments (one per real source clip) — it has no notion of the narrative
    text written for each section. Without this, repair_or_rebuild_timeline_
    with_moments would silently discard every word of narration the storyline
    LLM wrote (still present in storyline["sections"], just never carried into
    export_timeline), collapsing the video down to whatever handful of raw
    source clips exist — regardless of how much real narrative content was
    actually produced for the requested duration.
    """
    items: list[dict[str, Any]] = []
    for sec in (storyline.get("sections") or []):
        text = (sec.get("narration") or sec.get("viewer_narration") or "").strip()
        if not text:
            continue
        words = len(text.split())
        # ~2.5 words/sec matches the word-budget assumption used when the
        # narration was generated (see target_words = per_section_s * 2.5
        # in 07b_validate_storyline_with_llm.py) — keeps estimated duration
        # consistent with what the text was actually written to fill.
        duration = max(6.0, round(words / 2.5, 2))
        items.append({
            "type": "narration_broll",
            "segment_type": "narration_broll",
            "audio_mode": "ai_voiceover",
            "muted": True,
            "section_id": sec.get("section_id"),
            "voiceover_text": text,
            "subtitle_text": text,
            "subtitle_kind": "voiceover",
            "duration": duration,
            "start": None,
            "end": None,
            "moment_text": "",
            "moment_summary": "",
            "moment_selection_reason": "narration_from_storyline_sections",
            "section_role": sec.get("section_type") or sec.get("role"),
            "speech_quality": {},
            "moment_scores": {},
        })
    return items


def repair_or_rebuild_timeline_with_moments(
    storyline: dict[str, Any],
    selected_candidates: list[dict[str, Any]],
    prompt_profile: dict[str, Any],
    target_duration_seconds: int,
    video_mode: str | None = None,
) -> dict[str, Any]:
    """
    Ensure every fragment-linked timeline item has Phase 4O moment fields.

    If the existing export_timeline is empty or contains raw LLM items that
    lack moment fields, rebuild it from scratch using `selected_candidates`
    via the Phase 4O moment selector.

    Title/closing cards from any existing timeline are preserved.
    """
    result = dict(storyline)
    existing = list(result.get("export_timeline") or [])

    if not existing:
        needs_rebuild = True
        raw_count = 0
    else:
        needs_rebuild = timeline_requires_moment_repair(existing)
        raw_count = len(existing)

    if not needs_rebuild:
        result["timeline_repair"] = {"required": False}
        return result

    log(f"Timeline requires Phase 4O moment repair (raw_items={raw_count}). Rebuilding from selected_candidates.")

    # Preserve existing title/closing card entries
    card_items = [
        item for item in existing
        if item.get("type") in ("title_card", "closing_card")
        or item.get("timeline_type") in ("title_card", "closing_card")
    ]

    # Local import: avoids a circular import with selection.py, which imports
    # find_best_subclip_window/select_best_moment_for_section from this module.
    from app.pipeline.storyline_gen.selection import build_export_timeline_from_candidates

    rebuilt = build_export_timeline_from_candidates(
        selected_candidates=selected_candidates,
        prompt_profile=prompt_profile,
        target_duration_seconds=target_duration_seconds,
        title=result.get("title"),
        video_mode=video_mode,
    )

    # If rebuild produced title/closing cards, don't double-add from the existing timeline
    rebuilt_has_title = any(item.get("type") == "title_card" for item in rebuilt)
    rebuilt_has_closing = any(item.get("type") == "closing_card" for item in rebuilt)
    rebuilt_content = [
        item for item in rebuilt
        if item.get("type") not in ("title_card", "closing_card")
    ]

    # build_export_timeline_from_candidates only knows real source clips — it
    # never carries over the narration the storyline LLM wrote for each
    # section (still sitting in storyline["sections"], untouched by this
    # rebuild). Interleave it back in so a repaired timeline isn't reduced to
    # a bare clip reel capped at however many real fragments happened to be
    # selected — narration duration is written to fill target_duration
    # already, so this is what actually lets the video reach the requested
    # length instead of stopping at raw_fragment_count × per_segment_duration.
    narration_items = _narration_broll_items_from_sections(storyline)
    interleaved: list[dict[str, Any]] = []
    frag_idx = 0
    for nb in narration_items:
        interleaved.append(nb)
        if frag_idx < len(rebuilt_content):
            interleaved.append(rebuilt_content[frag_idx])
            frag_idx += 1
    interleaved.extend(rebuilt_content[frag_idx:])
    content = interleaved or rebuilt_content

    final: list[dict[str, Any]] = []
    if not rebuilt_has_title:
        title_cards = [item for item in card_items if item.get("type") == "title_card" or item.get("timeline_type") == "title_card"]
        final.extend(title_cards)
    else:
        final.extend(item for item in rebuilt if item.get("type") == "title_card")

    final.extend(content)

    if not rebuilt_has_closing:
        closing_cards = [item for item in card_items if item.get("type") == "closing_card" or item.get("timeline_type") == "closing_card"]
        final.extend(closing_cards)
    else:
        final.extend(item for item in rebuilt if item.get("type") == "closing_card")

    result["export_timeline"] = final
    result["timeline_repair"] = {
        "required": True,
        "reason": "llm_timeline_missing_phase4o_moment_fields",
        "raw_timeline_items": raw_count,
        "rebuilt_timeline_items": len(final),
        "narration_items_added": len(narration_items),
    }
    log(f"Timeline rebuilt: {raw_count} raw items → {len(final)} Phase 4O items "
        f"({len(narration_items)} narration_broll from storyline sections).")
    return result


def find_best_subclip_window(
    candidate: dict,
    prompt_profile: dict,
    desired_duration: float = 13.0,
) -> dict:
    """Find the best subclip window avoiding side-topic text, preferring prompt-term density."""
    start = float(candidate.get("start") or 0.0)
    end = float(candidate.get("end") or start + desired_duration)
    total_duration = end - start

    text = (candidate.get("text") or candidate.get("summary") or "").lower()
    if not text:
        return {"start": start, "end": end, "adjusted": False}

    topic_terms = set(prompt_profile.get("topic_terms") or [])
    expanded_terms = set(prompt_profile.get("expanded_terms") or [])
    all_prompt_terms = topic_terms | expanded_terms

    side_terms_lower = [t.lower() for t in SIDE_TOPIC_TERMS]
    has_side_topic = any(term in text for term in side_terms_lower)
    has_prompt_terms = any(term in text for term in all_prompt_terms)

    if not has_side_topic:
        return {"start": start, "end": end, "adjusted": False}

    words = text.split()
    total_words = len(words)
    if total_words < 10:
        return {"start": start, "end": end, "adjusted": False}

    first_40pct = " ".join(words[: int(total_words * 0.4)])
    latter_60pct = " ".join(words[int(total_words * 0.4):])

    side_in_first = any(term in first_40pct for term in side_terms_lower)
    prompt_in_latter = any(term in latter_60pct for term in all_prompt_terms)

    if side_in_first and prompt_in_latter and has_prompt_terms:
        offset = total_duration * 0.35
        new_start = round(start + offset, 2)
        new_end = round(new_start + desired_duration, 2)
        if new_end > end:
            new_end = end
        return {"start": new_start, "end": new_end, "adjusted": True, "reason": "side_topic_in_first_40pct_shifted_forward"}

    return {"start": start, "end": end, "adjusted": False}


INTRO_ADMIN_TERMS = [
    "hello everyone",
    "good morning",
    "good afternoon",
    "good evening",
    "thank you",
    "thanks for inviting me",
    "let me introduce",
    "next slide",
    "moving on",
    "today i will",
    "my name is",
    "it is an honor",
    "it is a pleasure",
    "housekeeping",
    "restroom",
    "closed captioning",
    "can you hear me",
    "sorry about",
    "welcome remarks",
    "i am delighted",
    "i am honored",
    "please welcome",
    "round of applause",
    "let me start by",
    "before i begin",
    "on behalf of",
]

SECTION_ROLE_KEYWORD_MAP: dict[str, dict[str, list[str]]] = {
    "deepfake": {
        "context": ["synthetic media", "fake video", "definition", "generated content", "digital manipulation", "what is"],
        "problem_or_tension": ["threat", "misinformation", "false", "fraud", "harm", "dangerous", "deceptive", "disinformation"],
        "human_perspective": ["ordinary", "women", "victims", "people", "families", "individuals", "targeted"],
        "expert_evidence": ["researcher", "expert", "study", "evidence", "data", "found", "shows", "analysis"],
        "solution_direction": ["regulation", "transparency", "platform", "law", "detect", "authenticate", "verify", "watermark"],
        "final_synthesis": ["trust", "democracy", "information integrity", "future", "society", "responsibility"],
    },
    "healthcare": {
        "context": ["diagnosis", "clinical", "care", "doctors", "treatment", "medical", "hospital"],
        "human_perspective": ["patients", "families", "access", "communities", "people living", "everyday"],
        "problem_or_tension": ["trust", "safety", "regulation", "risk", "overloaded", "waste", "burden"],
        "expert_evidence": ["researchers", "findings", "study", "evidence", "speakers", "data shows"],
        "solution_direction": ["governance", "ethics", "consent", "transparency", "accountability", "guidelines"],
        "final_synthesis": ["future", "measure", "human", "safer", "fairer", "dignity"],
    },
    "women": {
        "context": ["gender", "systems", "designed", "discrimination", "barriers"],
        "human_perspective": ["agency", "voice", "decision", "women", "girls", "lived experience"],
        "problem_or_tension": ["bias", "exclusion", "underrepresented", "inequality", "lack of"],
        "solution_direction": ["inclusion", "design", "participation", "representation", "leadership"],
        "final_synthesis": ["progress", "power", "real agency", "measure of", "change"],
    },
    "governance": {
        "context": ["rules", "policy", "institutions", "systems", "framework"],
        "problem_or_tension": ["accountability", "oversight", "challenge", "risk", "who decides"],
        "solution_direction": ["regulation", "rights", "transparency", "law", "participate"],
        "final_synthesis": ["responsibility", "future", "accountability", "protection"],
    },
    "crypto": {
        "context": ["blockchain", "cryptocurrency", "digital currency", "token", "web3"],
        "problem_or_tension": ["fraud", "scam", "volatile", "risk", "loss", "unregulated"],
        "solution_direction": ["consumer protection", "regulation", "transparency", "safety"],
        "final_synthesis": ["open financial", "trust", "future", "promise", "real benefit"],
    },
}


def _window_intro_penalty(window_text: str) -> float:
    """Return a penalty score (0–1) based on how much intro/admin language is in the window."""
    if not window_text:
        return 0.0
    normalized = window_text.lower()
    word_count = max(1, len(normalized.split()))
    intro_count = sum(1 for term in INTRO_ADMIN_TERMS if term in normalized)
    if intro_count == 0:
        return 0.0
    return min(1.0, intro_count * 0.35)


def _window_side_topic_penalty(window_text: str) -> float:
    """Return a penalty score based on side-topic term density."""
    if not window_text:
        return 0.0
    normalized = window_text.lower()
    count = sum(1 for term in SIDE_TOPIC_TERMS if term.lower() in normalized)
    return min(1.0, count * 0.25)


def _window_incomplete_sentence_penalty(window_text: str) -> float:
    """Penalise text that starts or ends mid-sentence."""
    if not window_text:
        return 0.0
    stripped = window_text.strip()
    penalty = 0.0
    # Starts mid-sentence: lowercase and not punctuated opener
    first_char = stripped[0] if stripped else ""
    if first_char.islower():
        penalty += 0.15
    # Ends without terminal punctuation
    last_char = stripped[-1] if stripped else ""
    if last_char not in ".!?\"'":
        penalty += 0.1
    return min(0.35, penalty)


def _window_prompt_score(window_text: str, prompt_profile: dict) -> float:
    """Score how much prompt-relevant content is in this window."""
    if not window_text:
        return 0.0
    normalized = window_text.lower()
    topic_terms = set(prompt_profile.get("topic_terms") or [])
    expanded_terms = set(prompt_profile.get("expanded_terms") or [])
    all_terms = topic_terms | expanded_terms
    hits = sum(1 for t in all_terms if t.lower() in normalized)
    return min(1.0, hits * 0.2)


def _window_section_fit_score(window_text: str, section_role: str | None, prompt_profile: dict) -> float:
    """Score how well this window fits a given section role."""
    if not window_text or not section_role:
        return 0.0
    normalized = window_text.lower()
    topic_terms = prompt_profile.get("topic_terms") or []
    primary_topic = topic_terms[0] if topic_terms else None

    role_map = SECTION_ROLE_KEYWORD_MAP.get(primary_topic or "", {})
    if not role_map:
        for topic in topic_terms:
            role_map = SECTION_ROLE_KEYWORD_MAP.get(topic, {})
            if role_map:
                break

    role_keywords = role_map.get(section_role, [])
    if not role_keywords:
        return 0.0

    hits = sum(1 for kw in role_keywords if kw.lower() in normalized)
    return min(1.0, hits * 0.3)


WEAK_MOMENT_STARTS = [
    "because",
    "and of course",
    "so if",
    "it's very interesting",
    "it is very interesting",
    "really heard today",
    "as i told you",
    "you know",
    "moving on",
    "next slide",
    "hello everyone",
    "good morning",
    "good afternoon",
    "good evening",
    "thank you",
    "welcome",
    "in conclusion",
    "and so",
    "so basically",
    "as i was saying",
    "right so",
    "okay so",
    "yeah so",
    "but anyway",
]


def is_meaningful_documentary_moment(
    moment_text: str,
    prompt_profile: dict,
    moment_scores: dict | None = None,
) -> tuple[bool, list[str]]:
    """
    Return (is_meaningful, reasons_for_rejection).

    A moment is meaningful if it starts cleanly, contains topic content,
    forms a complete thought, and scores above minimum thresholds.
    """
    reasons: list[str] = []
    if not moment_text or not moment_text.strip():
        reasons.append("empty_moment_text")
        return False, reasons

    stripped = moment_text.strip()
    normalized_lower = stripped.lower()

    # Starts with a weak phrase
    for phrase in WEAK_MOMENT_STARTS:
        if normalized_lower.startswith(phrase.lower()):
            reasons.append(f"starts_with_weak_phrase: {phrase!r}")
            break

    # Starts lowercase (mid-sentence) without being a proper noun opener
    first_char = stripped[0] if stripped else ""
    if first_char.islower():
        reasons.append("starts_lowercase_mid_sentence")

    # Contains intro/admin terms
    admin_hit = sum(1 for term in INTRO_ADMIN_TERMS if term in normalized_lower)
    if admin_hit >= 1:
        reasons.append(f"contains_admin_terms: {admin_hit}")

    # Fewer than 8 meaningful words (stop-words excluded)
    STOP_WORDS = {"a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "to", "and", "or", "but", "it", "this", "that", "i", "we", "you", "they", "he", "she", "so"}
    words = [w.strip(".,!?\"'") for w in stripped.split()]
    meaningful_words = [w for w in words if w.lower() not in STOP_WORDS and len(w) > 1]
    if len(meaningful_words) < 8:
        reasons.append(f"too_few_meaningful_words: {len(meaningful_words)}")

    # Does not contain topic/context terms
    topic_terms = set(prompt_profile.get("topic_terms") or [])
    expanded_terms = set(prompt_profile.get("expanded_terms") or [])
    all_terms = topic_terms | expanded_terms
    topic_hit = sum(1 for t in all_terms if t.lower() in normalized_lower)
    if all_terms and topic_hit == 0:
        reasons.append("no_topic_terms_found")

    # Ends mid-thought (no terminal punctuation)
    last_char = stripped[-1] if stripped else ""
    if last_char not in ".!?\"'":
        reasons.append("ends_without_terminal_punctuation")

    # Score thresholds
    if moment_scores:
        final_score = moment_scores.get("final_moment_score") or 0.0
        completeness = moment_scores.get("quote_completeness_score") or 0.0
        if float(final_score) < 0.35:
            reasons.append(f"final_moment_score_too_low: {final_score:.3f}")
        if float(completeness) < 0.55:
            reasons.append(f"quote_completeness_too_low: {completeness:.3f}")

    # Hard rejections: truly disqualifying — empty text, admin/greetings, not enough words
    # starts_lowercase is SOFT (conference transcripts often start mid-sentence in windows)
    hard_reasons = [r for r in reasons if any(r.startswith(p) for p in (
        "empty_", "contains_admin_terms", "too_few_meaningful_words",
    ))]

    # starts_with_weak_phrase is also a hard rejection when it matches admin/filler starts
    hard_weak = [r for r in reasons if r.startswith("starts_with_weak_phrase")]
    # Only treat weak_phrase start as hard if the phrase is truly admin (no topic terms present)
    if hard_weak and topic_hit > 0:
        hard_weak = []  # topic terms rescue a weak start

    is_meaningful = len(hard_reasons) == 0 and len(hard_weak) == 0
    return is_meaningful, reasons


def decide_clip_audio_role(
    moment: dict,
    candidate: dict,
    prompt_profile: dict,
) -> dict:
    """
    Decide the audio role for a selected moment window.

    Returns a dict with:
      segment_type: speaker_clip | narration_broll | archive_visual
      audio_mode:   source_audio | ai_voiceover | silent
      muted:        bool
      subtitle_kind: speaker | narration | none
      reason:       str
    """
    moment_text = moment.get("window_text") or moment.get("moment_text") or ""
    scores = {
        "final_moment_score": moment.get("final_moment_score") or 0.0,
        "quote_completeness_score": moment.get("quote_completeness_score") or 0.0,
    }
    speech = moment.get("speech_quality") or {}
    is_usable_speech = speech.get("is_usable_speech", True)
    intro_penalty = float(moment.get("intro_penalty") or 0.0)
    final_score = float(moment.get("final_moment_score") or 0.0)
    completeness = float(moment.get("quote_completeness_score") or 0.0)
    duration = float(moment.get("window_duration") or moment.get("duration") or 0.0)

    # Check meaningful moment (relaxed for Phase 4Q)
    is_meaningful, meaning_reasons = is_meaningful_documentary_moment(
        moment_text=moment_text,
        prompt_profile=prompt_profile,
        moment_scores=scores,
    )

    speaker_annotation_allowed = bool(
        candidate.get("speaker_soundbite_allowed")
        or candidate.get("speaker_clip_allowed")
    )

    # Phase 4Q: relax the speaker_allowed gate.
    # The original annotation sets speaker_clip_allowed=False when quote_strength < 0.65,
    # but conference clips at a topic-relevant event with clarity >= 0.75 and good moment
    # scores should still be eligible as speaker clips.
    # Override: allow as speaker_clip when moment score is meaningful even if the candidate
    # annotation gate didn't pre-approve it.
    clarity = float(candidate.get("clarity_score") or 0.0)
    final_video_allowed = candidate.get("final_video_allowed", True)
    speaker_allowed = (
        speaker_annotation_allowed
        or (
            final_video_allowed
            and clarity >= 0.70         # reasonable speech quality
            and final_score >= 0.20     # moment has content
            and is_meaningful           # not admin/filler
        )
    )

    # If the candidate annotation explicitly approved this as a speaker soundbite,
    # treat is_usable_speech as advisory rather than a hard gate — the heuristic
    # can be wrong (e.g. long fragments with correct window durations fail the old
    # duration check and get marked unusable even when the audio is fine).
    annotation_approved = bool(
        candidate.get("speaker_soundbite_allowed")
        or candidate.get("speaker_clip_allowed")
    )
    effective_usable_speech = is_usable_speech or annotation_approved

    # Phase 4Q speaker_clip conditions (relaxed from 4O):
    # - meaningful content (not admin/filler)
    # - usable speech quality (or annotation override)
    # - low intro penalty
    # - minimum meaningful score
    # - reasonable duration
    #
    # No upper duration bound: window length is already driven by
    # per_segment_duration (build_export_timeline_from_candidates), which
    # adapts to however many candidates survived selection — when only a
    # few candidates are available, each window has to be long to fill the
    # requested video length. There's no separate semantic/LLM step that
    # trims a window down to soundbite length, so a fixed upper bound here
    # only rejected clips for being exactly as long as they needed to be.
    is_speaker_clip = (
        is_meaningful
        and effective_usable_speech
        and intro_penalty < 0.20
        and final_score >= 0.20        # lowered from 0.45 — conference transcripts score lower
        and completeness >= 0.35       # lowered from 0.65 — partial credit for mid-window starts
        and duration >= 5.0            # floor only — guards against degenerate micro-clips
        and speaker_allowed
    )

    if is_speaker_clip:
        return {
            "segment_type": "speaker_clip",
            "audio_mode": "source_audio",
            "muted": False,
            "subtitle_kind": "speaker",
            "reason": f"meaningful_speaker_quote(score={final_score:.2f},completeness={completeness:.2f})",
        }

    # Not speaker quality — use as muted visual b-roll
    return {
        "segment_type": "archive_visual",
        "audio_mode": "silent",
        "muted": True,
        "subtitle_kind": "none",
        "reason": f"archive_visual(is_meaningful={is_meaningful},score={final_score:.2f},completeness={completeness:.2f},reasons={meaning_reasons[:2]})",
    }


def build_candidate_moment_windows(
    candidate: dict,
    prompt_profile: dict,
    window_seconds: float = 12.0,
    stride_seconds: float = 4.0,
) -> list[dict]:
    """
    Slide a window across a candidate fragment's transcript to find distinct
    moment windows. Returns a list of window dicts, each describing a time
    subrange and the text it covers.
    """
    fragment_id = candidate.get("fragment_id", "")
    source_video = candidate.get("source_video", "")
    start = float(candidate.get("start") or 0.0)
    end = float(candidate.get("end") or start + window_seconds)
    total_duration = max(0.1, end - start)

    text = (candidate.get("text") or candidate.get("summary") or "").strip()
    topic_terms = list((prompt_profile.get("topic_terms") or []) + (prompt_profile.get("expanded_terms") or []))

    windows: list[dict] = []

    if not text or total_duration < 4.0:
        # Single window covering the whole fragment
        windows.append({
            "fragment_id": fragment_id,
            "source_video": source_video,
            "window_start": round(start, 3),
            "window_end": round(end, 3),
            "window_duration": round(total_duration, 3),
            "window_text": text,
            "window_summary": text[:200] if text else "",
            "prompt_terms": [t for t in topic_terms if t.lower() in text.lower()],
            "section_role_candidates": [],
        })
        return windows

    words = text.split()
    total_words = len(words)
    seconds_per_word = total_duration / max(1, total_words)
    words_per_window = max(1, int(window_seconds / seconds_per_word))
    words_per_stride = max(1, int(stride_seconds / seconds_per_word))

    pos = 0
    while pos < total_words:
        w_end = min(pos + words_per_window, total_words)
        window_words = words[pos:w_end]
        window_text = " ".join(window_words)

        word_start_time = start + pos * seconds_per_word
        word_end_time = start + w_end * seconds_per_word
        word_end_time = min(word_end_time, end)

        win_duration = max(1.0, word_end_time - word_start_time)
        matched_terms = [t for t in topic_terms if t.lower() in window_text.lower()]

        windows.append({
            "fragment_id": fragment_id,
            "source_video": source_video,
            "window_start": round(word_start_time, 3),
            "window_end": round(word_end_time, 3),
            "window_duration": round(win_duration, 3),
            "window_text": window_text,
            "window_summary": window_text[:200],
            "prompt_terms": matched_terms,
            "section_role_candidates": [],
        })

        pos += words_per_stride
        if w_end >= total_words:
            break

    if not windows:
        windows.append({
            "fragment_id": fragment_id,
            "source_video": source_video,
            "window_start": round(start, 3),
            "window_end": round(end, 3),
            "window_duration": round(total_duration, 3),
            "window_text": text,
            "window_summary": text[:200],
            "prompt_terms": [t for t in topic_terms if t.lower() in text.lower()],
            "section_role_candidates": [],
        })

    return windows


def score_moment_window(
    window: dict,
    candidate: dict,
    prompt_profile: dict,
    section_role: str | None = None,
) -> dict:
    """
    Score a moment window for use as a speaker clip or archive visual.
    Returns a copy of `window` with scoring fields added.
    """
    scored = dict(window)
    wtext = window.get("window_text", "")

    moment_prompt_score = _window_prompt_score(wtext, prompt_profile)
    section_fit_score = _window_section_fit_score(wtext, section_role, prompt_profile)

    # Quote completeness: does it start and end like a complete thought?
    # Phase 4Q: partial credit when window starts mid-sentence but ends cleanly,
    # or when the candidate has high quote_strength (the source material is quotable).
    stripped = wtext.strip()
    starts_clean = stripped[:1].isupper() if stripped else False
    ends_clean = stripped[-1:] in ".!?\"'" if stripped else False
    candidate_quote_strength = float(candidate.get("quote_strength") or 0.5)
    if starts_clean and ends_clean:
        quote_completeness_score = 1.0
    elif ends_clean:
        # ends cleanly but starts mid-sentence: partial credit, boosted by quote_strength
        quote_completeness_score = 0.45 + 0.25 * candidate_quote_strength
    elif starts_clean:
        # starts cleanly but no terminal punctuation (window cut mid-sentence)
        quote_completeness_score = 0.35 + 0.15 * candidate_quote_strength
    else:
        # mid-sentence both ends: low but non-zero when quote_strength is high
        quote_completeness_score = 0.15 + 0.20 * candidate_quote_strength
    quote_completeness_score = round(min(1.0, quote_completeness_score), 4)

    # Speech quality: proxy from candidate metadata
    quote_strength = float(candidate.get("quote_strength") or 0.5)
    clarity_score = float(candidate.get("clarity_score") or 0.5)
    speech_quality_score = min(1.0, (quote_strength * 0.6) + (clarity_score * 0.4))

    # Documentary value: topic density and sentence quality
    word_count = len(wtext.split()) if wtext else 0
    density_score = min(1.0, len(window.get("prompt_terms", [])) * 0.2)
    length_score = min(1.0, word_count / 40.0)  # ~40 words ideal
    documentary_value_score = min(1.0, density_score * 0.6 + length_score * 0.4)

    # Penalties
    intro_penalty = _window_intro_penalty(wtext)
    silence_penalty = 0.0  # populated later by estimate_speech_presence if available
    side_topic_penalty = _window_side_topic_penalty(wtext)
    incomplete_sentence_penalty = _window_incomplete_sentence_penalty(wtext)

    final_moment_score = (
        0.30 * moment_prompt_score
        + 0.20 * section_fit_score
        + 0.20 * quote_completeness_score
        + 0.15 * speech_quality_score
        + 0.15 * documentary_value_score
        - intro_penalty
        - silence_penalty
        - side_topic_penalty
        - incomplete_sentence_penalty
    )
    final_moment_score = max(0.0, min(1.0, final_moment_score))

    # Build selection reason
    reasons = []
    if moment_prompt_score >= 0.3:
        reasons.append(f"topic-dense({moment_prompt_score:.2f})")
    if section_fit_score >= 0.3:
        reasons.append(f"role-fit({section_fit_score:.2f})")
    if quote_completeness_score >= 0.8:
        reasons.append("complete-quote")
    if intro_penalty >= 0.35:
        reasons.append("intro-language-penalty")
    if side_topic_penalty >= 0.25:
        reasons.append("side-topic-penalty")
    if not reasons:
        reasons.append(f"general-score({final_moment_score:.2f})")

    scored.update({
        "moment_prompt_score": round(moment_prompt_score, 4),
        "quote_completeness_score": round(quote_completeness_score, 4),
        "speech_quality_score": round(speech_quality_score, 4),
        "section_fit_score": round(section_fit_score, 4),
        "documentary_value_score": round(documentary_value_score, 4),
        "intro_penalty": round(intro_penalty, 4),
        "silence_penalty": round(silence_penalty, 4),
        "side_topic_penalty": round(side_topic_penalty, 4),
        "incomplete_sentence_penalty": round(incomplete_sentence_penalty, 4),
        "final_moment_score": round(final_moment_score, 4),
        "moment_selection_reason": ", ".join(reasons),
    })
    return scored


def infer_section_role_for_moment(window: dict, prompt_profile: dict) -> list[str]:
    """
    Return the list of section roles this moment window fits best.
    Used to suggest where in the documentary arc a clip belongs.
    """
    wtext = (window.get("window_text") or "").lower()
    topic_terms = prompt_profile.get("topic_terms") or []
    roles = []

    for topic in topic_terms:
        role_map = SECTION_ROLE_KEYWORD_MAP.get(topic, {})
        for role, keywords in role_map.items():
            if any(kw.lower() in wtext for kw in keywords):
                if role not in roles:
                    roles.append(role)

    # Fallback: intro/admin → not suitable for any speaking role
    if _window_intro_penalty(wtext) >= 0.35:
        return []

    if not roles:
        # Generic fallbacks
        if any(t in wtext for t in ["therefore", "conclusion", "in summary", "ultimately", "this means"]):
            roles.append("final_synthesis")
        elif any(t in wtext for t in ["however", "but", "challenge", "problem", "concern", "risk"]):
            roles.append("problem_or_tension")
        elif any(t in wtext for t in ["solution", "approach", "recommend", "propose", "should"]):
            roles.append("solution_direction")
        else:
            roles.append("expert_evidence")

    return roles


def estimate_speech_presence(candidate: dict, start: float, end: float) -> dict:
    """
    Estimate speech quality for a clip window.

    Since we don't have direct audio analysis here (that requires ffmpeg/librosa
    in the video export step), we use proxy signals from metadata:
    - quote_strength as speech density proxy
    - clarity_score as quality proxy
    - fragment text word count vs duration as speech ratio proxy

    Returns a dict compatible with the Phase 4O speech_quality field.
    """
    duration = max(0.1, end - start)
    text = (candidate.get("text") or candidate.get("summary") or "").strip()
    total_frag_start = float(candidate.get("start") or 0.0)
    total_frag_end = float(candidate.get("end") or total_frag_start + duration)
    total_frag_duration = max(0.1, total_frag_end - total_frag_start)

    # Proportional word count for this window
    total_words = len(text.split()) if text else 0
    fraction_of_frag = min(1.0, duration / total_frag_duration)
    estimated_words = int(total_words * fraction_of_frag)

    # Words-per-minute proxy: typical speech is 120-160 wpm
    words_per_minute = (estimated_words / duration) * 60 if duration > 0 else 0
    speech_ratio = min(1.0, words_per_minute / 150.0)  # 150 wpm = full speech

    quote_strength = float(candidate.get("quote_strength") or 0.5)
    clarity_score = float(candidate.get("clarity_score") or 0.5)

    # Weight: speech_ratio is proxy; also factor in metadata quality signals
    combined_ratio = min(1.0, (speech_ratio * 0.5) + (quote_strength * 0.3) + (clarity_score * 0.2))
    silence_ratio = max(0.0, 1.0 - combined_ratio)
    mean_volume = -18.0 - (1.0 - combined_ratio) * 12  # rough proxy in dBFS

    # Duration constraint is applied at the is_speaker_clip level (5.0 <= window_duration <= 20.0),
    # not here — the fragment duration can be long; we're evaluating the window we selected from it.
    is_usable = (
        combined_ratio >= 0.55
        and silence_ratio <= 0.35
    )

    return {
        "speech_ratio": round(combined_ratio, 3),
        "mean_volume": round(mean_volume, 1),
        "silence_ratio": round(silence_ratio, 3),
        "is_usable_speech": is_usable,
        "estimated_words": estimated_words,
        "duration": round(duration, 3),
    }


def select_best_moment_for_section(
    section: dict,
    candidates: list[dict],
    prompt_profile: dict,
    used_fragment_ids: set[str],
    section_role: str | None = None,
    window_seconds: float = 12.0,
) -> dict | None:
    """
    Select the best moment window for a story section from the available candidates.

    Returns a moment dict with all Phase 4O fields, or None if no usable window found.
    """
    effective_role = section_role or section.get("section_type") or section.get("role")

    best_window = None
    best_score = -999.0

    for candidate in candidates:
        fid = str(candidate.get("fragment_id") or "")
        if not fid or fid in used_fragment_ids:
            continue

        start_c = float(candidate.get("start") or 0.0)
        end_c = float(candidate.get("end") or start_c + window_seconds)
        total_dur = max(0.1, end_c - start_c)

        # Build windows for this candidate
        windows = build_candidate_moment_windows(
            candidate=candidate,
            prompt_profile=prompt_profile,
            window_seconds=min(window_seconds, total_dur),
            stride_seconds=max(2.0, window_seconds * 0.33),
        )

        for window in windows:
            scored = score_moment_window(
                window=window,
                candidate=candidate,
                prompt_profile=prompt_profile,
                section_role=effective_role,
            )
            # Assign section role candidates
            scored["section_role_candidates"] = infer_section_role_for_moment(window, prompt_profile)

            # Compute speech presence
            speech = estimate_speech_presence(
                candidate=candidate,
                start=window["window_start"],
                end=window["window_end"],
            )
            scored["speech_quality"] = speech

            # Update silence_penalty from speech estimate
            if not speech["is_usable_speech"] and speech["silence_ratio"] > 0.35:
                silence_pen = min(0.4, speech["silence_ratio"] * 0.5)
                scored["silence_penalty"] = round(silence_pen, 4)
                scored["final_moment_score"] = round(
                    max(0.0, scored["final_moment_score"] - silence_pen), 4
                )

            if scored["final_moment_score"] > best_score:
                best_score = scored["final_moment_score"]
                best_window = {
                    "candidate": candidate,
                    "window": scored,
                    "speech": speech,
                }

    if best_window is None:
        return None

    candidate = best_window["candidate"]
    window = best_window["window"]
    speech = best_window["speech"]

    # Phase 4Q: use decide_clip_audio_role for speaker vs archive decision
    # Pass speech quality into the window so decide_clip_audio_role can read it
    window["speech_quality"] = speech
    audio_role = decide_clip_audio_role(
        moment=window,
        candidate=candidate,
        prompt_profile=prompt_profile,
    )

    segment_type = audio_role["segment_type"]
    audio_mode = audio_role["audio_mode"]
    muted = audio_role["muted"]
    subtitle_kind = audio_role["subtitle_kind"]
    moment_text = window.get("window_text", "")
    subtitle_text = moment_text if segment_type == "speaker_clip" else ""

    if segment_type != "speaker_clip" and audio_role.get("reason"):
        window["moment_selection_reason"] = (
            window.get("moment_selection_reason", "") + f"; {audio_role['reason']}"
        )

    return {
        "type": segment_type,
        "segment_type": segment_type,
        "audio_mode": audio_mode,
        "fragment_id": candidate.get("fragment_id"),
        "source_video": candidate.get("source_video"),
        "start": window["window_start"],
        "end": window["window_end"],
        "duration": window["window_duration"],
        "moment_text": moment_text,
        "moment_summary": window.get("window_summary", ""),
        "moment_selection_reason": window.get("moment_selection_reason", ""),
        "section_role": effective_role,
        "muted": muted,
        "subtitle_text": subtitle_text,
        "subtitle_kind": subtitle_kind,
        "speech_quality": speech,
        "moment_scores": {
            "moment_prompt_score": window.get("moment_prompt_score"),
            "quote_completeness_score": window.get("quote_completeness_score"),
            "speech_quality_score": window.get("speech_quality_score"),
            "section_fit_score": window.get("section_fit_score"),
            "documentary_value_score": window.get("documentary_value_score"),
            "intro_penalty": window.get("intro_penalty"),
            "silence_penalty": window.get("silence_penalty"),
            "side_topic_penalty": window.get("side_topic_penalty"),
            "incomplete_sentence_penalty": window.get("incomplete_sentence_penalty"),
            "final_moment_score": window.get("final_moment_score"),
        },
        "use_as": candidate.get("use_as"),
        "relevance_role": candidate.get("relevance_role"),
        "selection_reason": candidate.get("selection_reason"),
        "speaker_clip_allowed": bool(candidate.get("speaker_clip_allowed") or candidate.get("speaker_soundbite_allowed")),
        "speaker_visual_allowed": bool(candidate.get("speaker_visual_allowed")),
        "subclip_adjusted": False,
    }
