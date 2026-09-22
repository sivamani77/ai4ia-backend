"""Narrative-structure selection for Step 07b — deepfake vs. generic arc, scaled to duration."""
from typing import Any

from app.pipeline.timeline_budget import timeline_segment_count_for_duration
from app.pipeline.validate_gen.narrative_templates import (
    DEEPFAKE_NARRATIVE_SECTIONS,
    DEEPFAKE_TOPIC_TERMS,
    GENERIC_NARRATIVE_SECTIONS,
)


def detect_narrative_sections(
    user_prompt: str,
    topic_terms: list[str],
    target_duration_seconds: int = 120,
) -> list[dict[str, Any]]:
    # Only check hardcoded deepfake terms — topic_terms came FROM the prompt and always match,
    # so including them would make every topic look like deepfake.
    prompt_lower = (user_prompt or "").lower()
    base = GENERIC_NARRATIVE_SECTIONS
    for term in DEEPFAKE_TOPIC_TERMS:
        if term in prompt_lower:
            base = DEEPFAKE_NARRATIVE_SECTIONS
            break

    # timeline_segment_count_for_duration(120) == len(base) == 6, so short/
    # default requests are unaffected. Longer requests previously stayed
    # capped at this fixed 6-section arc no matter the target — 07's own
    # storyline generation scales its section count with target_duration_seconds
    # (same helper), but 07b rebuilt its narrative structure from this
    # hardcoded, unscaled template, silently discarding everything upstream
    # wrote past section 6 and capping the whole pipeline's output duration
    # regardless of what was requested.
    desired_count = timeline_segment_count_for_duration(target_duration_seconds)
    if desired_count <= len(base):
        return base

    # Fill extra slots by cycling only the middle of the arc (evidence/
    # challenge/response), never the opening (hook, context) or closing
    # (synthesis). The previous fix repeated the *whole* arc cyclically,
    # which put a second "opening hook" and "context" section mid/late video
    # — e.g. on an 8-section video: hook, context, ..., synthesis, hook(2),
    # context(2), rendering as a second open right before the end. A longer
    # video needs more depth in the middle, not a second open or close.
    opening = base[:2]   # hook, context — exactly once, always first
    closing = base[-1:]  # synthesis — exactly once, always last
    middle = base[2:-1]  # evidence/challenge/response — cycled to fill extra slots
    extra_needed = desired_count - len(opening) - len(closing)

    cycled: list[dict[str, Any]] = []
    for i in range(extra_needed):
        src = middle[i % len(middle)]
        pass_num = i // len(middle) + 1
        sec = dict(src)
        if pass_num > 1:
            sec["section_id"] = f"{src['section_id']}_{pass_num}"
            sec["title"] = f"{src['title']} (Part {pass_num})"
        cycled.append(sec)

    return list(opening) + cycled + list(closing)
