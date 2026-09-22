import re
from typing import Any

from app.pipeline.storyline_gen.llm_client import log
from app.pipeline.storyline_gen.prompt_intent import (
    candidate_speaker_matches,
    load_known_speakers,
)
from app.pipeline.storyline_gen.text_utils import (
    SIDE_TOPIC_TERMS,
    clean_text,
    normalize_name,
    parse_list_field,
)


GENERIC_ONLY_TERMS = {"inclusion", "equity", "access", "diversity", "accessibility"}

# When a prompt's entire vocabulary is stopword-stripped (topic_terms empty),
# direct_terms/expanded_hits/theme_match/chapter_match are all structurally
# unreachable in classify_candidate_relevance_v2 -- there is no keyword-shaped
# evidence to gate on. NO_TOPIC_TERMS_SEMANTIC_FLOOR is the semantic-only bar
# used to promote a candidate to strong_related in that case instead of
# discarding it outright. Provisional value; tune against real reruns.
NO_TOPIC_TERMS_SEMANTIC_FLOOR = 0.65


CRYPTO_TERMS = {"crypto", "cryptocurrency", "blockchain", "web3", "digital currency", "token", "defi", "nft", "exchange", "wallet", "satoshi", "bitcoin", "ethereum"}
CRYPTOGRAPHY_TERMS = {"cryptography", "encryption", "decryption", "cipher", "quantum cryptography", "post quantum", "homomorphic", "zero knowledge", "cryptographic", "hashing", "hash function", "public key", "private key"}

GENERIC_EXPANSION_TERMS = {"inclusion", "equity", "access", "diversity", "transparency", "accountability", "responsibility", "knowledge", "information", "accessibility"}


def classify_candidate_relevance(
    candidate: dict[str, Any],
    prompt_terms: set[str],
    expanded_terms: set[str],
    intent: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(candidate)
    blob = candidate_search_blob(enriched)
    blob_tokens = set(re.findall(r"[a-z0-9]+", blob))
    requested_speaker = intent.get("requested_speaker")
    requested_theme = intent.get("theme")
    requested_chapter = intent.get("chapter")
    scope = intent.get("scope")

    # Detect if this is a crypto-specific prompt (not cryptography)
    is_crypto_prompt = "crypto" in prompt_terms and "cryptography" not in prompt_terms and "encryption" not in prompt_terms
    is_cryptography_prompt = any(t in prompt_terms for t in {"cryptography", "encryption", "quantum"})

    # Compute meaningful overlap: only prompt_terms (not stopwords), not expanded synonyms
    meaningful_overlap = [t for t in prompt_terms if t in blob_tokens]

    # Expanded overlap: only non-generic expansion terms count toward exactness
    expanded_overlap_raw = [t for t in expanded_terms if t in blob_tokens and t not in prompt_terms]
    expanded_overlap = [t for t in expanded_overlap_raw if t not in GENERIC_EXPANSION_TERMS]
    generic_expanded_overlap = [t for t in expanded_overlap_raw if t in GENERIC_EXPANSION_TERMS]

    semantic_score = float(enriched.get("semantic_score") or 0.0)
    quote_strength = float(enriched.get("quote_strength") or 0.0)
    clarity_score = float(enriched.get("clarity_score") or 0.0)
    usage_candidates = set(parse_list_field(enriched.get("clip_usage_candidates")))

    # Crypto/cryptography disambiguation
    if is_crypto_prompt:
        candidate_blob_has_crypto = any(t in blob_tokens for t in CRYPTO_TERMS)
        candidate_blob_has_cryptography = any(t in blob_tokens for t in CRYPTOGRAPHY_TERMS)
        if candidate_blob_has_cryptography and not candidate_blob_has_crypto:
            # This is a cryptography clip for a crypto (cryptocurrency) prompt — demote
            expanded_overlap = []
            meaningful_overlap = []
            semantic_score = min(semantic_score, 0.58)  # cap it so it can't reach exact

    # Theme/chapter match
    theme = (enriched.get("primary_theme") or "").lower()
    chapter = (enriched.get("primary_chapter") or "").lower()
    theme_chapter_text = theme + " " + chapter
    theme_tokens = set(re.findall(r"[a-z0-9]+", theme_chapter_text))
    theme_overlap = [t for t in prompt_terms if t in theme_tokens]
    expanded_theme_overlap = [t for t in expanded_terms if t in theme_tokens and t not in GENERIC_EXPANSION_TERMS]

    # Speaker constraint check
    speaker_clip_allowed = False
    broll_allowed = True
    speaker_match = requested_speaker is None
    speaker_match_reason = "no speaker requested"

    if requested_speaker is not None:
        speaker_match, speaker_match_reason = candidate_speaker_matches(enriched, requested_speaker)
        if not speaker_match:
            enriched["meaningful_prompt_overlap"] = meaningful_overlap
            enriched["requested_speaker"] = requested_speaker
            enriched["speaker_match"] = False
            enriched["speaker_match_reason"] = speaker_match_reason
            enriched["prompt_relevance_score"] = round(float(enriched.get("semantic_score") or 0.0), 6)
            enriched["relevance_role"] = "reject"
            enriched["speaker_soundbite_allowed"] = False
            enriched["speaker_visual_allowed"] = False
            enriched["speaker_clip_allowed"] = False
            enriched["broll_allowed"] = False
            enriched["final_video_allowed"] = False
            enriched["use_as"] = "reject"
            enriched["relevance_reason"] = f"Speaker constraint: {speaker_match_reason}"
            return enriched

    # Compute total evidence signals (non-generic only)
    total_meaningful = len(meaningful_overlap) + len(expanded_overlap)
    total_theme = len(theme_overlap) + len(expanded_theme_overlap)
    chapter_match = bool(requested_chapter and normalize_name(requested_chapter) == normalize_name(str(enriched.get("primary_chapter") or "")))
    theme_match = bool(requested_theme and normalize_name(requested_theme) == normalize_name(str(enriched.get("primary_theme") or "")))
    prompt_relevance_score = float(semantic_score)
    if chapter_match:
        prompt_relevance_score += 0.14
    if theme_match:
        prompt_relevance_score += 0.1
    prompt_relevance_score += min(0.16, len(meaningful_overlap) * 0.06)
    prompt_relevance_score += min(0.06, len(expanded_overlap) * 0.03)
    prompt_relevance_score += min(0.06, quote_strength * 0.06)

    # Side-topic pollution penalty
    candidate_blob_lower = blob.lower()
    side_topic_count = sum(1 for term in SIDE_TOPIC_TERMS if term.lower() in candidate_blob_lower)
    if side_topic_count >= 2:
        prompt_relevance_score -= 0.25
    elif side_topic_count == 1 and len(meaningful_overlap) == 0:
        prompt_relevance_score -= 0.15

    # Classification logic
    if requested_speaker and scope == "all_themes":
        if chapter_match or theme_match or len(meaningful_overlap) >= 1 or semantic_score >= 0.58:
            role = "exact"
            reason = "Speaker matched; selected as one of the strongest clips across available themes."
            speaker_clip_allowed = (
                "speaker_soundbite" in usage_candidates
                and quote_strength >= 0.35
                and clarity_score >= 0.75
            )
        elif semantic_score >= 0.52:
            role = "strong_related"
            reason = f"Speaker matched with limited topic evidence; semantic {semantic_score:.3f}"
            speaker_clip_allowed = (
                "speaker_soundbite" in usage_candidates
                and quote_strength >= 0.40
                and clarity_score >= 0.75
            )
        else:
            role = "weak_related"
            reason = f"Speaker matched but topic evidence is weak; semantic {semantic_score:.3f}"
            speaker_clip_allowed = False
    elif chapter_match and (theme_match or len(meaningful_overlap) >= 1 or semantic_score >= 0.56):
        role = "exact"
        reason = "Speaker and requested chapter matched."
        speaker_clip_allowed = (
            "speaker_soundbite" in usage_candidates
            and quote_strength >= 0.35
            and clarity_score >= 0.75
        )
    elif theme_match and (len(meaningful_overlap) >= 1 or semantic_score >= 0.58):
        role = "exact"
        reason = "Speaker and requested theme matched."
        speaker_clip_allowed = (
            "speaker_soundbite" in usage_candidates
            and quote_strength >= 0.35
            and clarity_score >= 0.75
        )
    elif len(meaningful_overlap) >= 2 and semantic_score >= 0.62:
        role = "exact"
        reason = f"Strong meaningful overlap ({meaningful_overlap}) + semantic {semantic_score:.3f}"
        speaker_clip_allowed = (
            "speaker_soundbite" in usage_candidates
            and quote_strength >= 0.35
            and clarity_score >= 0.75
        )
    elif len(meaningful_overlap) >= 1 and semantic_score >= 0.66:
        role = "exact"
        reason = f"Meaningful overlap ({meaningful_overlap}) + high semantic {semantic_score:.3f}"
        speaker_clip_allowed = (
            "speaker_soundbite" in usage_candidates
            and quote_strength >= 0.35
            and clarity_score >= 0.75
        )
    elif len(expanded_overlap) >= 2 and semantic_score >= 0.65:
        role = "exact"
        reason = f"Non-generic expanded overlap ({expanded_overlap}) + semantic {semantic_score:.3f}"
        speaker_clip_allowed = (
            "speaker_soundbite" in usage_candidates
            and quote_strength >= 0.35
            and clarity_score >= 0.75
        )
    elif len(generic_expanded_overlap) >= 2 and semantic_score >= 0.67:
        # Generic terms (inclusion/equity/diversity) only → strong_related at best
        role = "strong_related"
        reason = f"Generic expanded terms only ({generic_expanded_overlap}) + semantic {semantic_score:.3f}; not exact"
        speaker_clip_allowed = False
    elif semantic_score >= 0.68 and (total_theme >= 1 or total_meaningful >= 1):
        role = "strong_related"
        reason = f"High semantic {semantic_score:.3f} + theme/expanded match"
    elif (total_theme >= 1 or total_meaningful >= 1 or len(expanded_overlap) >= 1) and semantic_score >= 0.62:
        role = "strong_related"
        reason = f"Topic/theme overlap at moderate semantic {semantic_score:.3f}: meaningful={meaningful_overlap}, theme={theme_overlap}, expanded={expanded_overlap[:3]}"
    elif total_meaningful >= 1 or total_theme >= 1 or len(expanded_overlap) >= 1:
        role = "weak_related"
        reason = f"Partial overlap below semantic threshold: meaningful={meaningful_overlap}, theme={theme_overlap}, expanded={expanded_overlap[:3]}"
    elif len(generic_expanded_overlap) >= 1 and semantic_score >= 0.60:
        role = "contextual"
        reason = f"Only generic synonym match ({generic_expanded_overlap}) + semantic {semantic_score:.3f}"
    elif semantic_score >= 0.60:
        role = "contextual"
        reason = f"Broad semantic similarity {semantic_score:.3f} only, no topic overlap"
    else:
        role = "reject"
        reason = f"Low semantic {semantic_score:.3f}, no topic overlap"
        broll_allowed = False

    # speaker_soundbite_allowed: quote-quality gate (clarity, quote_strength),
    # scoped to relevant prompt-overlap roles (exact / strong_related / speaker-matched weak_related).
    speaker_soundbite_allowed = False
    if role in ("exact", "strong_related", "weak_related"):
        speaker_soundbite_allowed = (
            quote_strength >= 0.20
            and clarity_score >= 0.50
        )

    # speaker_visual_allowed: true for speaker-prompt clips that belong to the requested speaker,
    # even if not good enough for a soundbite. The clip can run muted under narration.
    speaker_visual_allowed = False
    if requested_speaker and speaker_match and role not in ("reject",):
        duration = float(enriched.get("duration") or 0.0)
        start = enriched.get("start")
        end = enriched.get("end")
        if duration > 8 and start is not None and end is not None:
            speaker_visual_allowed = True

    # Determine use_as and final_video_allowed
    if role == "reject":
        use_as = "reject"
        final_video_allowed = False
    elif role == "contextual":
        use_as = "context_only"
        final_video_allowed = False
    elif speaker_soundbite_allowed:
        use_as = "speaker_soundbite"
        final_video_allowed = True
    elif role == "weak_related":
        if speaker_visual_allowed:
            # Speaker-matched weak clip — usable as visual under narration
            use_as = "speaker_visual"
            final_video_allowed = True
        else:
            use_as = "narration_support"
            final_video_allowed = False  # will be promoted to True only for low-evidence fallback
    elif speaker_visual_allowed:
        use_as = "speaker_visual"
        final_video_allowed = True
    elif "archive_visual" in usage_candidates or "evidence_clip" in usage_candidates:
        use_as = "archive_visual"
        final_video_allowed = True
    else:
        use_as = "narration_support"
        final_video_allowed = True

    # archive_visual clips must never be speaker_soundbite_allowed
    if use_as == "archive_visual":
        speaker_soundbite_allowed = False
        speaker_clip_allowed = False

    enriched["meaningful_prompt_overlap"] = meaningful_overlap
    enriched["requested_speaker"] = requested_speaker
    enriched["speaker_match"] = speaker_match
    enriched["speaker_match_reason"] = speaker_match_reason
    enriched["prompt_relevance_score"] = round(prompt_relevance_score, 6)
    enriched["relevance_role"] = role
    enriched["speaker_soundbite_allowed"] = speaker_soundbite_allowed
    enriched["speaker_visual_allowed"] = speaker_visual_allowed
    enriched["speaker_clip_allowed"] = speaker_soundbite_allowed  # backward compat alias
    enriched["broll_allowed"] = broll_allowed
    enriched["final_video_allowed"] = final_video_allowed
    enriched["use_as"] = use_as
    enriched["relevance_reason"] = reason
    return enriched


def candidate_search_blob(record: dict[str, Any]) -> str:
    parts = [
        record.get("text", ""),
        record.get("summary", ""),
        " ".join(parse_list_field(record.get("keywords"))),
        record.get("source_video", ""),
        record.get("speaker", ""),
        record.get("primary_speaker", ""),
        record.get("primary_theme", ""),
        record.get("primary_chapter", ""),
    ]
    return clean_text(" ".join(str(part) for part in parts if part)).lower()


def match_terms_in_text(text: str, terms: list[str]) -> list[str]:
    normalized = normalize_name(text)
    matches: list[str] = []
    for term in terms:
        normalized_term = normalize_name(term)
        if normalized_term and re.search(rf"\b{re.escape(normalized_term)}\b", normalized):
            matches.append(term)
    return matches


def score_candidate_for_prompt(candidate: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(candidate)
    blob = candidate_search_blob(enriched)
    direct_terms = match_terms_in_text(blob, profile.get("topic_terms") or [])
    expanded_terms = [
        term for term in match_terms_in_text(blob, profile.get("expanded_terms") or [])
        if term not in direct_terms
    ]
    negative_terms = match_terms_in_text(blob, profile.get("negative_terms") or [])
    generic_removed = set(profile.get("generic_terms_removed") or [])
    generic_overlap = [term for term in expanded_terms if normalize_name(term) in generic_removed or normalize_name(term) in GENERIC_ONLY_TERMS]
    non_generic_expanded = [term for term in expanded_terms if term not in generic_overlap]

    semantic_score = float(enriched.get("semantic_score") or 0.0)
    quote_strength = float(enriched.get("quote_strength") or 0.0)
    clarity_score = float(enriched.get("clarity_score") or 0.0)
    theme_match = bool(
        profile.get("theme_hint")
        and normalize_name(str(enriched.get("primary_theme") or "")) == normalize_name(str(profile.get("theme_hint")))
    )
    chapter_match = bool(
        profile.get("chapter_hint")
        and normalize_name(str(enriched.get("primary_chapter") or "")) == normalize_name(str(profile.get("chapter_hint")))
    )
    requested_speaker = profile.get("requested_speaker")
    if requested_speaker:
        speaker_match, speaker_match_reason = candidate_speaker_matches(enriched, requested_speaker)
    else:
        speaker_match, speaker_match_reason = True, "no speaker requested"

    # The prompt named this fragment's video by title (see
    # lookup_requested_source_video). Keyword overlap alone cannot express
    # that: keyword_score saturates at 1.0 after two matched terms and
    # contributes only 0.25, so even a word-for-word title match landed well
    # under the 0.48 `exact` gate and was filed as weak_related — which
    # bucket_candidates drops from the video entirely.
    requested_source_video = profile.get("requested_source_video")
    title_match = bool(
        requested_source_video
        and str(enriched.get("source_video") or "") == requested_source_video
    )
    title_score = 1.0 if title_match else 0.0

    keyword_score = min(1.0, len(direct_terms) * 0.5)
    expanded_keyword_score = min(1.0, len(non_generic_expanded) * 0.35)
    theme_chapter_score = 1.0 if chapter_match else (0.8 if theme_match else 0.0)
    speaker_score = 1.0 if speaker_match and requested_speaker else 0.0
    quote_quality_score = min(1.0, (quote_strength * 0.6) + (clarity_score * 0.4))
    generic_penalty = min(0.25, len(generic_overlap) * 0.08)
    negative_term_penalty = min(0.45, len(negative_terms) * 0.18)

    # topic_terms empty -> keyword_score/expanded_keyword_score are
    # structurally always 0 (direct_terms/expanded_terms can only be
    # populated by matching against topic_terms, see extract_topic_terms and
    # infer_theme_and_chapter_hints in prompt_intent.py). Redistribute their
    # combined weight onto semantic_score instead of silently dropping it, so
    # final_prompt_score stays meaningful for sorting/capping even though the
    # role-classification branches still need NO_TOPIC_TERMS_SEMANTIC_FLOOR
    # (below) to actually reach strong_related.
    if profile.get("topic_terms"):
        semantic_weight = 0.30
    else:
        semantic_weight = 0.70
    final_prompt_score = (
        semantic_weight * semantic_score
        + 0.25 * keyword_score
        + 0.15 * expanded_keyword_score
        + 0.15 * theme_chapter_score
        + 0.10 * speaker_score
        + 0.05 * quote_quality_score
        + 0.25 * title_score
        - generic_penalty
        - negative_term_penalty
    )

    reasons = []
    if direct_terms:
        reasons.append(f"direct terms {direct_terms[:3]}")
    if non_generic_expanded:
        reasons.append(f"expanded terms {non_generic_expanded[:3]}")
    if chapter_match:
        reasons.append(f"chapter match {enriched.get('primary_chapter')}")
    elif theme_match:
        reasons.append(f"theme match {enriched.get('primary_theme')}")
    if requested_speaker and speaker_match:
        reasons.append(f"speaker match {requested_speaker}")
    if title_match:
        reasons.append("prompt names this video's title")
    if not reasons:
        reasons.append(f"semantic similarity {semantic_score:.3f}")
    if negative_terms:
        reasons.append(f"negative conflict {negative_terms[:3]}")

    enriched.update(
        {
            "keyword_score": round(keyword_score, 6),
            "expanded_keyword_score": round(expanded_keyword_score, 6),
            "theme_chapter_score": round(theme_chapter_score, 6),
            "speaker_score": round(speaker_score, 6),
            "title_score": round(title_score, 6),
            "title_match": title_match,
            "quote_quality_score": round(quote_quality_score, 6),
            "generic_penalty": round(generic_penalty, 6),
            "negative_term_penalty": round(negative_term_penalty, 6),
            "final_prompt_score": round(final_prompt_score, 6),
            "direct_evidence_terms": direct_terms,
            "expanded_evidence_terms": non_generic_expanded,
            "negative_evidence_terms": negative_terms,
            "generic_evidence_terms": generic_overlap,
            "selection_reason": "Matched " + ", ".join(reasons) + ".",
            "requested_speaker": requested_speaker,
            "speaker_match": speaker_match,
            "speaker_match_reason": speaker_match_reason,
            "chapter_match": chapter_match,
            "theme_match": theme_match,
            "meaningful_prompt_overlap": direct_terms,
            "prompt_keyword_overlap": direct_terms,
            "prompt_keyword_overlap_count": len(direct_terms),
        }
    )
    return enriched


def apply_hard_rejection_rules(candidate: dict[str, Any], profile: dict[str, Any]) -> tuple[bool, str]:
    if profile.get("requested_speaker") and not candidate.get("speaker_match"):
        return True, "Rejected: speaker prompt requires requested speaker match"

    topic_terms = set(profile.get("topic_terms") or [])
    direct_terms = candidate.get("direct_evidence_terms") or []
    expanded_terms = candidate.get("expanded_evidence_terms") or []
    negative_terms = candidate.get("negative_evidence_terms") or []
    generic_terms = candidate.get("generic_evidence_terms") or []

    if "crypto" in topic_terms:
        crypto_vocab = {normalize_name(v) for v in CRYPTO_TERMS}
        crypto_positive = any(normalize_name(term) in crypto_vocab for term in direct_terms + expanded_terms)
        if negative_terms and not crypto_positive:
            return True, "Rejected: cryptography is not cryptocurrency evidence"

    if not direct_terms and not expanded_terms and negative_terms and float(candidate.get("semantic_score") or 0.0) < 0.7:
        return True, "Rejected: negative-term conflict without direct evidence"

    return False, ""


def classify_candidate_relevance_v2(candidate: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    enriched = score_candidate_for_prompt(candidate, profile)
    reject, reject_reason = apply_hard_rejection_rules(enriched, profile)

    requested_speaker = profile.get("requested_speaker")
    scope = profile.get("scope")
    direct_terms = enriched.get("direct_evidence_terms") or []
    expanded_hits = enriched.get("expanded_evidence_terms") or []
    semantic_score = float(enriched.get("semantic_score") or 0.0)
    final_prompt_score = float(enriched.get("final_prompt_score") or 0.0)
    quote_strength = float(enriched.get("quote_strength") or 0.0)
    clarity_score = float(enriched.get("clarity_score") or 0.0)
    usage_candidates = set(parse_list_field(enriched.get("clip_usage_candidates")))
    chapter_match = bool(enriched.get("chapter_match"))
    theme_match = bool(enriched.get("theme_match"))
    generic_terms = enriched.get("generic_evidence_terms") or []

    semantic_only_match = False

    if reject:
        role = "reject"
        reason = reject_reason
    elif enriched.get("title_match"):
        # The user asked for this video by name. Its fragments are direct
        # evidence by definition, whatever their transcripts happen to say —
        # a title is about the video, not repeated inside every clip of it,
        # so the keyword/semantic gates below systematically under-rate them.
        role = "exact"
        reason = (
            f"Prompt names this video's title "
            f"({profile.get('title_match_ratio')} of prompt terms matched)."
        )
    elif requested_speaker and scope == "all_themes":
        if enriched.get("speaker_match") and (semantic_score >= 0.45 or quote_strength >= 0.55 or clarity_score >= 0.9):
            role = "exact"
            reason = f"Requested speaker match with strong archive evidence across themes. {enriched.get('selection_reason')}"
        elif enriched.get("speaker_match") and (semantic_score >= 0.30 or final_prompt_score >= 0.20):
            role = "strong_related"
            reason = f"Requested speaker match with limited evidence. {enriched.get('selection_reason')}"
        else:
            role = "weak_related"
            reason = f"Requested speaker match but weak evidence. {enriched.get('selection_reason')}"
    elif not profile.get("topic_terms") and semantic_score >= NO_TOPIC_TERMS_SEMANTIC_FLOOR:
        # No extractable topic terms after stopword filtering (e.g. every
        # prompt word is a stopword) -- direct_terms/expanded_hits/
        # theme_match/chapter_match are all structurally empty/false, so none
        # of the keyword-gated branches below can ever fire. Promote on
        # semantic similarity alone rather than discarding a genuinely strong
        # candidate; capped at strong_related, never exact, since there is no
        # keyword/theme evidence backing it.
        role = "strong_related"
        reason = f"No extractable topic terms after stopword filtering; promoted on high semantic similarity {semantic_score:.3f}."
        semantic_only_match = True
    elif direct_terms and final_prompt_score >= 0.48:
        role = "exact"
        reason = enriched.get("selection_reason")
    elif (chapter_match or theme_match) and final_prompt_score >= 0.46 and (direct_terms or expanded_hits or requested_speaker):
        role = "exact"
        reason = enriched.get("selection_reason")
    elif (theme_match or chapter_match or expanded_hits) and final_prompt_score >= 0.34:
        role = "strong_related"
        reason = enriched.get("selection_reason")
    elif direct_terms or expanded_hits:
        role = "weak_related"
        reason = enriched.get("selection_reason")
    elif generic_terms and semantic_score >= 0.60:
        role = "contextual"
        reason = f"Only generic support terms matched. {enriched.get('selection_reason')}"
    elif semantic_score >= 0.60:
        role = "contextual"
        reason = f"Broad semantic similarity only. {enriched.get('selection_reason')}"
    else:
        role = "reject"
        reason = f"Low topic evidence. {enriched.get('selection_reason')}"

    if generic_terms and not direct_terms and role == "exact" and not enriched.get("title_match"):
        role = "strong_related"
        reason = f"Generic-only evidence cannot be exact. {enriched.get('selection_reason')}"

    speaker_soundbite_allowed = False
    if role in ("exact", "strong_related", "weak_related"):
        speaker_soundbite_allowed = (
            quote_strength >= 0.20
            and clarity_score >= 0.50
        )

    speaker_visual_allowed = False
    if requested_speaker and enriched.get("speaker_match") and role != "reject":
        duration = float(enriched.get("duration") or 0.0)
        if duration > 8 and enriched.get("start") is not None and enriched.get("end") is not None:
            speaker_visual_allowed = True

    broll_allowed = role != "reject"
    if role == "reject":
        use_as = "reject"
        final_video_allowed = False
    elif role == "contextual":
        use_as = "context_only"
        final_video_allowed = False
    elif speaker_soundbite_allowed:
        use_as = "speaker_soundbite"
        final_video_allowed = True
    elif role == "weak_related":
        if speaker_visual_allowed:
            use_as = "speaker_visual"
            final_video_allowed = True
        else:
            use_as = "narration_support"
            final_video_allowed = False
    elif speaker_visual_allowed:
        use_as = "speaker_visual"
        final_video_allowed = True
    elif "archive_visual" in usage_candidates or "evidence_clip" in usage_candidates:
        use_as = "archive_visual"
        final_video_allowed = True
    else:
        use_as = "narration_support"
        final_video_allowed = True

    enriched.update(
        {
            "prompt_relevance_score": round(final_prompt_score, 6),
            "relevance_role": role,
            "speaker_soundbite_allowed": speaker_soundbite_allowed,
            "speaker_visual_allowed": speaker_visual_allowed,
            "speaker_clip_allowed": speaker_soundbite_allowed,
            "broll_allowed": broll_allowed,
            "final_video_allowed": final_video_allowed,
            "use_as": use_as,
            "relevance_reason": reason,
            "semantic_only_match": semantic_only_match,
        }
    )
    return enriched


def annotate_candidate_relevance(
    records: list[dict[str, Any]],
    prompt: str,
    intent: dict[str, Any],
    known_speakers: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not records:
        return []

    if known_speakers is None:
        known_speakers = load_known_speakers()

    requested_speaker = intent.get("requested_speaker")

    if requested_speaker:
        log(f"Speaker constraint detected: '{requested_speaker}'")
    if intent.get("topic_terms"):
        log(f"Meaningful prompt terms: {sorted(intent.get('topic_terms') or [])}")
    if intent.get("expanded_terms"):
        log(f"Expanded synonym terms: {sorted(intent.get('expanded_terms') or [])}")
    if intent.get("negative_terms"):
        log(f"Negative terms: {sorted(intent.get('negative_terms') or [])}")

    annotated = []
    for record in records:
        enriched = classify_candidate_relevance_v2(record, intent)
        annotated.append(enriched)

    return annotated


def theme_coverage(records: list[dict[str, Any]]) -> list[str]:
    return sorted({str(item.get("primary_theme")) for item in records if item.get("primary_theme")})


def chapter_coverage(records: list[dict[str, Any]]) -> list[str]:
    return sorted({str(item.get("primary_chapter")) for item in records if item.get("primary_chapter")})
