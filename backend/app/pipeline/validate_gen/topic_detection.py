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
    # regardless of what was requested. Repeat the hook→...→synthesis arc
    # cyclically to reach the same segment count 07 already targets.
    desired_count = timeline_segment_count_for_duration(target_duration_seconds)
    if desired_count <= len(base):
        return base

    sections: list[dict[str, Any]] = []
    for i in range(desired_count):
        src = base[i % len(base)]
        pass_num = i // len(base) + 1
        sec = dict(src)
        if pass_num > 1:
            sec["section_id"] = f"{src['section_id']}_{pass_num}"
            sec["title"] = f"{src['title']} (Part {pass_num})"
        sections.append(sec)
    return sections
