import re


INTERNAL_PIPELINE_TERMS = [
    "archive fragment",
    "archive fragments",
    "selected clips",
    "selected evidence",
    "confirmed archive clips",
    "evidence package",
    "evidence packages",
    "retrieval",
    "fallback",
    "llm",
    "json",
    "manifest",
    "qa report",
    "pipeline",
    "background task",
    "candidate bucket",
    "validation issue",
]


def detect_internal_pipeline_language(text: str) -> list[str]:
    """Detect internal pipeline/system terms in viewer-facing text."""
    issues = []
    if not text:
        return issues
    normalized = text.lower()
    for term in INTERNAL_PIPELINE_TERMS:
        if term in normalized:
            issues.append(f"internal_pipeline_language_detected: '{term}'")
    return issues


def repair_internal_pipeline_language(text: str) -> str:
    """Remove internal pipeline terms from viewer-facing text."""
    if not text:
        return text
    # Ordered list: longer / more-specific phrases first so they are replaced before
    # their sub-terms would be reached by a later pattern.
    replacements = [
        (r"\bconfirmed archive clips?\b", "recordings"),
        (r"\barchive fragments?\b", "recordings"),
        (r"\bselected clips?\b", "speakers"),
        (r"\bevidence packages?\b", "evidence"),
        (r"\bevidence package\b", "evidence"),
        (r"\bcandidate buckets?\b", "clips"),
        (r"\bvalidation issues?\b", "notes"),
        (r"\bbackground tasks?\b", ""),
        (r"\bqa report\b", ""),
        # Pipeline-specific verbs/nouns that are context-safe to remove
        (r"\bpipeline\b", ""),
        (r"\bfallback\b", ""),
        (r"\bllm\b", ""),
        (r"\bjson\b", ""),
        (r"\bmanifest\b", ""),
        (r"\bretrieval\b", "research"),
        # "candidate" used as a noun phrase for clips — only replace when clearly pipeline-context
        # (avoid replacing e.g. "candidate countries" in legitimate text)
        (r"\bcandidate clips?\b", "recordings"),
        (r"\bcandidate fragments?\b", "recordings"),
        (r"\bcandidate bucket\b", "clips"),
    ]
    result = text
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    result = re.sub(r"  +", " ", result).strip()
    return result


VIEWER_FACING_FIELDS = {
    "title", "logline", "narration", "voiceover_text", "subtitle_text",
    "closing_meaning", "central_question", "narrative_thesis",
}

VIEWER_FACING_SECTION_FIELDS = {
    "title", "narration", "viewer_narration",
}


def validate_viewer_facing_text(storyline_or_manifest: dict) -> list[str]:
    """Validate all viewer-facing text fields for internal pipeline language."""
    issues = []

    def check_text(text: str, field_path: str) -> None:
        found = detect_internal_pipeline_language(text)
        for f in found:
            issues.append(f"{f} (field: {field_path})")

    for field in VIEWER_FACING_FIELDS:
        val = storyline_or_manifest.get(field)
        if isinstance(val, str):
            check_text(val, field)

    for section in (storyline_or_manifest.get("sections") or []):
        sid = section.get("section_id", "?")
        for field in VIEWER_FACING_SECTION_FIELDS:
            val = section.get(field)
            if isinstance(val, str):
                check_text(val, f"section[{sid}].{field}")

    for item in (storyline_or_manifest.get("voiceover_script") or []):
        val = item.get("text")
        if isinstance(val, str):
            check_text(val, f"voiceover_script[{item.get('section_id', '?')}].text")

    return issues


def _repair_text_field(text: str) -> tuple[str, bool]:
    """Repair a single viewer-facing text field. Returns (repaired_text, was_changed)."""
    if not text:
        return text, False
    repaired = repair_internal_pipeline_language(text)
    return repaired, repaired != text


def sanitize_viewer_facing_storyline_text(
    storyline: dict,
    prompt_profile: dict,
) -> tuple[dict, list[str]]:
    """
    Sanitize all viewer-facing text fields in a storyline.

    Applies repair_internal_pipeline_language to every field a viewer could see.
    Returns (sanitized_storyline, list_of_repair_notes).
    Internal diagnostic fields (evidence_fragment_ids, selection_reason, etc.) are untouched.
    """
    result = dict(storyline)
    repair_notes: list[str] = []

    # Top-level viewer-facing string fields
    top_level_fields = [
        "title", "logline", "closing_meaning", "central_question", "narrative_thesis",
    ]
    for field in top_level_fields:
        val = result.get(field)
        if isinstance(val, str):
            repaired, changed = _repair_text_field(val)
            if changed:
                result[field] = repaired
                repair_notes.append(f"repaired:{field}")

    # voiceover_script list
    if isinstance(result.get("voiceover_script"), list):
        repaired_vos = []
        for item in result["voiceover_script"]:
            item = dict(item)
            for field in ("text", "subtitle_text"):
                val = item.get(field)
                if isinstance(val, str):
                    repaired, changed = _repair_text_field(val)
                    if changed:
                        item[field] = repaired
                        repair_notes.append(f"repaired:voiceover_script[{item.get('section_id','?')}].{field}")
            repaired_vos.append(item)
        result["voiceover_script"] = repaired_vos

    # sections[]
    if isinstance(result.get("sections"), list):
        repaired_sections = []
        for section in result["sections"]:
            section = dict(section)
            for field in ("title", "narration", "viewer_narration", "subtitle_text", "voiceover_text"):
                val = section.get(field)
                if isinstance(val, str):
                    repaired, changed = _repair_text_field(val)
                    if changed:
                        section[field] = repaired
                        repair_notes.append(f"repaired:section[{section.get('section_id','?')}].{field}")
            repaired_sections.append(section)
        result["sections"] = repaired_sections

    # export_timeline[]
    if isinstance(result.get("export_timeline"), list):
        repaired_timeline = []
        for item in result["export_timeline"]:
            item = dict(item)
            for field in ("title", "subtitle_text", "voiceover_text"):
                val = item.get(field)
                if isinstance(val, str):
                    repaired, changed = _repair_text_field(val)
                    if changed:
                        item[field] = repaired
                        repair_notes.append(f"repaired:export_timeline[{item.get('fragment_id','?')}].{field}")
            repaired_timeline.append(item)
        result["export_timeline"] = repaired_timeline

    # Post-repair: check if any viewer-facing issues remain
    remaining = validate_viewer_facing_text(result)
    unrepairable = [r for r in remaining if "internal_pipeline_language_detected" in r]
    if unrepairable:
        repair_notes.extend(
            f"viewer_facing_text_unrepairable: {issue}" for issue in unrepairable
        )

    return result, repair_notes
