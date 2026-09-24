"""Full-narrative assembly: one LLM call arranging approved candidates into a final timeline."""
import json
from typing import Any

from app.pipeline.prompt_safety import UNTRUSTED_CONTENT_NOTICE, wrap_untrusted
from app.pipeline.validate_gen.json_utils import extract_json_object
from app.pipeline.validate_gen.llm_client import STRONG_LLM_MODEL, log, post_with_retries

NARRATIVE_ASSEMBLY_SYSTEM = """You are an editorial director building a short documentary video.

You have a list of approved candidate clips. Your job is to arrange them into a coherent documentary timeline that directly addresses the user's requested topic.

RULES:
- Follow the NARRATIVE SECTIONS in order: they are this request's documentary format.
- Each section must have at least one piece of evidence (speaker clip OR narration_broll).
- The first and last sections are narration only. Speaker clips go in the middle sections,
  each after that section's narration. Never open or close the film with a clip.
- Write specific, evidence-driven voiceover narration grounded in the topic from the USER PROMPT.
- Voiceover must mention specific claims from the evidence and bridge to the next clip.
- NEVER write voiceover about a different topic than what the USER PROMPT specifies.
- Use speaker clips as SPEAKER CLIPS below says. Every narration_broll segment is required.
- Reject redundant or weak clips that duplicate other clips.
- End with a closing synthesis statement on the USER PROMPT topic.
- Every narration_broll item's "voiceover_text" is REQUIRED and must be non-empty —
  this is the only narration pass that will run; there is no separate follow-up call
  to fill gaps, so leaving voiceover_text blank or thin is not recoverable later.
- GROUNDING (violations are auto-rejected downstream): every factual claim in a
  voiceover_text must come from the quote_text/reason fields of the APPROVED
  CANDIDATE CLIPS below — never invent named people, organisations, statistics,
  percentages, or study results that are not present there. If the evidence doesn't
  support a specific claim, write thematic connective narration instead.
- CRITICAL: every "fragment_id" value you write must be copied EXACTLY,
  character-for-character, from the "fragment_id" field of one of the
  APPROVED CANDIDATE CLIPS listed below. Never invent, abbreviate,
  paraphrase, or truncate a fragment_id, and never make one up (e.g.
  "hook_001") — those are not real clips and will be discarded. If a
  timeline_items/final_timeline entry has no real approved clip behind it
  (e.g. a narration_broll bridge with no matching footage), OMIT the
  "fragment_id" field from that entry entirely rather than guessing.

OUTPUT: Respond with ONLY a valid JSON object. No markdown, no commentary outside the JSON.
""" + UNTRUSTED_CONTENT_NOTICE


def _archive_facts_block(storyline_data: dict[str, Any]) -> str:
    """Facts for a request about the archive itself; empty for every other request."""
    facts = (storyline_data.get("storyline") or {}).get("archive_facts")
    if not facts:
        return ""
    return ("\nARCHIVE FACTS (the request is about the archive itself). State figures exactly as given here "
            "(\"309 talks\", never \"more than 309\"), derive no new figures such as a number of years, "
            "never attribute them to a speaker, and expand the name AI4IA only as given here. "
            "Answer the question asked first: when the USER PROMPT asks for a figure (how many speakers, "
            "talks, themes, chapters or years), the first narration states that figure. "
            "A speaker clip you use stays a speaker_clip item: it is one of the conference's voices.\n" + facts + "\n")


def _clip_rules(clips_required: bool, speaker_target: int, cap: int) -> str:
    if clips_required:
        need = f"at least {speaker_target}, " if speaker_target else ""
        return (f"SPEAKER CLIPS: this request needs real speaker clips: use {need}at most 5, each a "
                "complete thought on the AI4IA agenda.")
    return (f"SPEAKER CLIPS: optional for this request. Use at most {cap}, only where a clip says something "
            "the narration cannot. A timeline with no speaker clips is fine.")


def _context_blocks(storyline_data: dict[str, Any]) -> str:
    """Panel roster, requested facet and archive gaps, when the job has them."""
    storyline = storyline_data.get("storyline") or {}
    blocks = ""
    if storyline.get("panel_context"):
        blocks += ("\nPANEL CONTEXT (the requested speaker appears only on these panels; clip audio may be any "
                   "panellist, so never attribute a clip's words to the requested speaker):\n"
                   + wrap_untrusted(json.dumps(storyline["panel_context"], ensure_ascii=False)) + "\n")
    if storyline.get("requested_facet"):
        blocks += ("\nREQUESTED FACET (keep every section inside it):\n"
                   + json.dumps(storyline["requested_facet"], ensure_ascii=False) + "\n")
    if storyline.get("coverage_gaps"):
        from app.pipeline.storyline_gen.storyline_llm import COVERAGE_GAP_RULE

        blocks += ("\nNOT IN THE ARCHIVE: " + wrap_untrusted(", ".join(storyline["coverage_gaps"]))
                   + "\n" + COVERAGE_GAP_RULE + "\n")
    return blocks


def assemble_narrative_with_llm(
    validated_candidates: list[dict[str, Any]],
    narrative_sections: list[dict[str, Any]],
    user_prompt: str,
    target_duration: int,
    storyline_data: dict[str, Any],
    clips_required: bool = True,
    speaker_target: int = 0,
    clip_cap: int = 2,
) -> dict[str, Any] | None:
    approved = [c for c in validated_candidates if c.get("approved")]
    if not approved:
        log("  [llm] No approved candidates for narrative assembly.")
        return None

    candidates_json = json.dumps([
        {
            "fragment_id": c.get("fragment_id"),
            "decision": c.get("decision"),
            "section_id": c.get("section_id"),
            "speaker_label": c.get("speaker_label") or c.get("speaker") or "",
            "quote_start": c.get("quote_start"),
            "quote_end": c.get("quote_end"),
            "quote_text": (c.get("quote_text") or c.get("text") or "")[:300],
            "quote_complete": c.get("quote_complete"),
            "reason": c.get("reason", ""),
            "confidence": c.get("confidence", 0),
            **({"year": c["video_year"]} if c.get("video_year") else {}),
        }
        for c in approved
    ], indent=2)

    sections_json = json.dumps([
        {"section_id": s["section_id"], "title": s["title"], "purpose": s["purpose"], "card_text": s["card_text"]}
        for s in narrative_sections
    ], indent=2)

    # Same word-budget formula generate_voiceovers_for_timeline uses, so a
    # voiceover written here doesn't get flagged as "too short" and overwritten later.
    n_sections = max(1, len(narrative_sections))
    content_budget_s = max(1, target_duration - 10)
    per_section_s = round(content_budget_s / n_sections)
    target_words = max(40, int(per_section_s * 2.5))
    min_sentences = max(3, round(target_words / 15))

    # Examples carry this format's ids; the LLM copies example ids verbatim.
    first_id = narrative_sections[0]["section_id"] if narrative_sections else "S1"
    middle_id = narrative_sections[1]["section_id"] if len(narrative_sections) > 2 else first_id

    user_text = f"""USER PROMPT: {wrap_untrusted(user_prompt)}
TARGET DURATION: {target_duration} seconds
TARGET WORDS PER NARRATION_BROLL SEGMENT: ~{target_words} words (about {min_sentences}-{min_sentences + 2} sentences) — do not undershoot this, thin narration gets replaced downstream.

NARRATIVE SECTIONS:
{sections_json}

APPROVED CANDIDATE CLIPS:
{wrap_untrusted(candidates_json)}
{_clip_rules(clips_required, speaker_target, clip_cap)}
{_archive_facts_block(storyline_data)}{_context_blocks(storyline_data)}
Build the final documentary timeline.

Return JSON with this exact structure:
{{
  "title": "short documentary title (5-7 words)",
  "logline": "one sentence summary of what this documentary reveals",
  "narrative_arc": [
    {{
      "section_id": "{first_id}",
      "purpose": "what this section achieves",
      "card_text": "text for the section card",
      "timeline_items": [
        {{
          "type": "speaker_clip | narration_broll",
          "fragment_id": "...",
          "quote_start": 0.0,
          "quote_end": 0.0,
          "quote_text": "...",
          "speaker": "...",
          "why_here": "why this clip belongs in this section"
        }}
      ],
      "voiceover_text": "specific evidence-driven narration for this section's narration_broll"
    }}
  ],
  "final_timeline": [
    {{"type": "title_card", "title": "...", "duration": 5.0}},
    {{
      "type": "narration_broll",
      "section_id": "{first_id}",
      "voiceover_text": "specific evidence-driven narration relevant to the user prompt topic...",
      "subtitle_text": "same as voiceover_text",
      "subtitle_kind": "voiceover",
      "visual_direction": "..."
    }},
    {{
      "type": "speaker_clip",
      "section_id": "{middle_id}",
      "fragment_id": "...",
      "speaker": "...",
      "render_start": 0.0,
      "render_end": 0.0,
      "render_duration": 0.0,
      "quote_text": "...",
      "subtitle_text": "...",
      "quote_complete": true
    }},
    {{"type": "closing_card", "title": "...", "duration": 5.0}}
  ],
  "rejected_candidates": [
    {{"fragment_id": "...", "reason": "why rejected from final timeline"}}
  ],
  "editorial_quality_score": 0.85,
  "warnings": []
}}
"""

    try:
        raw = post_with_retries({
            "model": STRONG_LLM_MODEL,
            "messages": [
                {"role": "system", "content": NARRATIVE_ASSEMBLY_SYSTEM},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.3,
            # This model emits its reasoning trace before the actual answer, and for
            # a full-narrative assembly (multiple sections + a full final_timeline)
            # that trace alone can exhaust a small context window, leaving the real
            # JSON answer empty/truncated. Raised so there's room for both.
            "max_completion_tokens": 12000,
            "stream": False,
        })
        result = extract_json_object(raw)
        if not result:
            log("  [llm] No JSON from narrative assembly.")
            return None
        return result
    except Exception as exc:
        log(f"  [llm] Narrative assembly failed: {exc}")
        return None
