"""Shared storyline instructions; each intent module adds its own (docs/design/intent-layer-design.md §7)."""
from __future__ import annotations

import hashlib

BASE_PROMPT = """
You are a senior documentary producer and AI research editor.

Create a production-ready storyline blueprint for a documentary-style video.
The source archive contains conference footage, speaker clips, panel discussions, and event visuals.
The final video should feel like a National Geographic / premium explainer documentary:
- narrator-led
- evidence-backed
- visually planned
- speaker soundbites used selectively
- generated or internet B-roll used where conference footage is insufficient

=== NARRATION LENGTH — CRITICAL REQUIREMENT ===
The video is {duration} seconds long with {num_sections} sections.
Each section narration MUST be at least {words_per_section} words long.
At 2.5 words/second TTS pace, {words_per_section} words = {per_section_s} seconds of voiceover per section.
NEVER write a narration shorter than {words_per_section} words. One or two sentences is NOT acceptable.
Write full, substantive paragraphs. Each narration should be a complete passage of {words_per_section}+ words.
===============================================

Important rules:
1. Do not invent speaker names unless they are visible in source_video text.
2. Use only fragment_id values from the provided candidate fragments.
3. Do not assign a permanent storyline_role to fragments.
4. Decide how each fragment is used only for this specific user prompt.
5. Separate real archive clips from generated/internet assets.
6. Keep narration cinematic but factual.
7. Return valid JSON only. No markdown. No backticks.
8. Respect evidence_gaps and must_not_claim. Never upgrade supporting/context clips into direct claims.
"""


def section(section_id: str, title: str, purpose: str, visual_type: str, keywords: list[str],
            default_voiceover: str, core: bool = False, repeat: bool = False) -> dict:
    """One section of an intent's documentary format, in the shape stage 07b uses.
    `core` sections survive trimming to a short video; `repeat` ones fill a long one."""
    return {
        "section_id": section_id,
        "title": title,
        "purpose": purpose,
        "card_text": title,
        "visual": {
            "visual_mode": "generated_asset",
            "visual_asset_type": visual_type,
            "visual_title": title,
            "visual_asset_prompt": f"Documentary card: {title}. Dark background, clean typography, subtle accents.",
            "visual_keywords": keywords,
        },
        "default_voiceover": default_voiceover,
        "core": core,
        "repeat": repeat,
    }


def agenda_themes() -> list[str]:
    """The AI4IA agenda: the archive's theme names, from config/taxonomy.json."""
    from app.core.taxonomy import THEMES

    return [theme["name"] for theme in THEMES.values()]


def agenda_rule() -> str:
    """What makes a speaker clip usable, in any intent."""
    return ("- A speaker clip must show the speaker speaking substantively about AI4IA itself or its agenda ("
            + "; ".join(agenda_themes()) + "), and about the request. Never use greetings, thanks, "
            "introductions of or hand-overs to another speaker, moderator transitions, housekeeping, "
            "logistics, audio checks, or jokes and anecdotes that make no point.")


def format_instructions(sections: list[dict], clips: str) -> str:
    """The intent's documentary format and clip rules, for the storyline LLM."""
    lines = ["DOCUMENTARY FORMAT (sections in this order; a longer video repeats the middle sections, "
             "never the first or last):"]
    lines += [f"{i}. {s['title']}: {s['purpose']}" for i, s in enumerate(sections, 1)]
    lines += ["", "SPEAKER CLIPS:",
              "- The first and last sections are narration only. Speaker clips go in the middle sections, "
              "each after that section's narration."]
    if clips == "required":
        lines.append("- This request needs real speaker clips: they are its evidence.")
    else:
        lines.append("- Speaker clips are optional unless the prompt asks for clips, quotes or what speakers said. "
                     "Otherwise use at most 2, only where a clip says something the narration cannot. "
                     "A video with no clips is fine.")
    lines.append(agenda_rule())
    return "\n".join(lines)


def system_prompt(instructions: str, **values) -> str:
    """BASE_PROMPT filled in, followed by the intent's own instructions."""
    text = BASE_PROMPT.format(**values).strip()
    if instructions.strip():
        text += "\n\n" + instructions.strip()
    return text


def template_hash(instructions: str) -> str:
    """Identifies the unfilled template, so a job records which one it used."""
    return hashlib.sha256((BASE_PROMPT + "\0" + instructions).encode()).hexdigest()[:12]
