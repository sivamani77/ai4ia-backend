import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Any

import edge_tts
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Local dev layout: backend/app/. Docker image layout (see Dockerfile): app/
# directly under PROJECT_ROOT, with no backend/ directory at all.
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.pipeline.timeline_budget import (  # noqa: E402
    build_duration_budget,
    classify_timeline_segment_type,
    normalize_candidate_time_bounds,
    segment_effective_duration,
    timeline_card_duration,
    timeline_segment_count_for_duration,
)

CINEMATIC_BACKGROUND_ASSET = PROJECT_ROOT / "config" / "cinematic_background.png"

STORYLINES_DIR = PROJECT_ROOT / "data" / "storylines"
GENERATED_ASSETS_DIR = Path(os.environ.get("AI4IA_LOCAL_ASSETS_DIR", str(PROJECT_ROOT / "data" / "generated_assets")))
FRAGMENTS_CLASSIFIED_DIR = PROJECT_ROOT / "data" / "fragments_classified"
SPEAKERS_METADATA_DIR = PROJECT_ROOT / "data" / "metadata" / "speakers"

DEFAULT_VOICE = "en-GB-RyanNeural"
DEFAULT_RATE = "-6%"
DEFAULT_VOLUME = "+0%"

CARD_WIDTH = 1920
CARD_HEIGHT = 1080

# Speaker clip render bounds
PRE_ROLL_SECONDS: float = 0.25
POST_ROLL_SECONDS: float = 0.35
CLIP_EXPAND_SECONDS: float = 3.0
QUOTE_COMPLETE_THRESHOLD: float = 0.80  # quote_completeness_score >= this → complete

BACKGROUND_COLOR = (6, 10, 18)
PANEL_COLOR = (15, 24, 38)
TEXT_COLOR = (248, 250, 252)
MUTED_TEXT_COLOR = (178, 192, 210)
ACCENT_COLOR = (84, 170, 255)
GOLD_ACCENT = (230, 184, 92)
DARK_OVERLAY = (0, 0, 0, 120)

VISIBLE_BRAND_LABEL = "AI4IA 2026"
VISIBLE_SERIES_LABEL = "Artificial Intelligence for Information Accessibility"

STRICT_NO_GENERIC_FALLBACK = True

GENERIC_CARD_TITLES = {
    "a new information age",
    "the future is already here",
    "human story",
    "people at the center",
    "a shared future",
    "designing for trust",
    "evidence from the field",
    "the way forward",
    "when access is uneven",
}

DISALLOWED_VOICEOVER_PHRASES = [
    "artificial intelligence is no longer a distant idea",
    "shaping how people learn, communicate, govern and access knowledge",
    "across the ai4ia archive",
    "the promise of ai is powerful",
    "the risks are equally real",
    "inclusive design, local data, ethical governance",
    "ai can become a bridge to knowledge",
]

INTERNAL_PIPELINE_TERMS_08 = [
    "archive fragment",
    "archive fragments",
    "selected clips",
    "confirmed archive clips",
    "evidence package",
    "retrieval",
    "fallback",
    "llm",
    "json",
    "manifest",
    "qa report",
    "pipeline",
    "background task",
    "candidate",
    "candidate bucket",
    "validation issue",
    "evidence-based context",
    "draws exclusively on confirmed",
    "these fragments",
    "the selected clips",
    "this video explores",
    "through evidence from the archive",
    "the archive evidence shows",
]

DOCUMENTARY_CHAPTER_TITLES: dict[str, dict[str, dict[str, str]]] = {
    "healthcare": {
        "opening_hook": {"eyebrow": "THE QUESTION", "headline": "AI at the Bedside", "subtitle": "How is AI entering the spaces where people seek care?"},
        "context": {"eyebrow": "CONTEXT", "headline": "Medicine in Motion", "subtitle": "Digital tools are already changing how patients and doctors connect."},
        "problem_or_tension": {"eyebrow": "THE TENSION", "headline": "Trust Before Automation", "subtitle": "Speed without safety is not progress."},
        "human_perspective": {"eyebrow": "THE HUMAN STAKES", "headline": "Care, Not Replacement", "subtitle": "Human judgment must remain at the center."},
        "expert_evidence": {"eyebrow": "WHAT SPEAKERS SAY", "headline": "Voices from the Field", "subtitle": "Practitioners and researchers on what AI can and cannot do."},
        "solution_direction": {"eyebrow": "THE PATH FORWARD", "headline": "Design for Safety", "subtitle": "Trust, transparency, and patient consent must come first."},
        "final_synthesis": {"eyebrow": "THE MEASURE", "headline": "The Human Measure of Medicine", "subtitle": "AI in healthcare will be judged not by speed, but by whether it makes care safer and fairer."},
    },
    "health": {
        "opening_hook": {"eyebrow": "THE QUESTION", "headline": "AI at the Bedside", "subtitle": "How is AI entering the spaces where people seek care?"},
        "context": {"eyebrow": "CONTEXT", "headline": "Medicine in Motion", "subtitle": "Digital tools are already changing how patients and doctors connect."},
        "problem_or_tension": {"eyebrow": "THE TENSION", "headline": "Trust Before Automation", "subtitle": "Speed without safety is not progress."},
        "human_perspective": {"eyebrow": "THE HUMAN STAKES", "headline": "Care, Not Replacement", "subtitle": "Human judgment must remain at the center."},
        "expert_evidence": {"eyebrow": "WHAT SPEAKERS SAY", "headline": "Voices from the Field", "subtitle": "Practitioners and researchers on what AI can and cannot do."},
        "solution_direction": {"eyebrow": "THE PATH FORWARD", "headline": "Design for Safety", "subtitle": "Trust, transparency, and patient consent must come first."},
        "final_synthesis": {"eyebrow": "THE MEASURE", "headline": "The Human Measure of Medicine", "subtitle": "AI in healthcare will be judged not by speed, but by whether it makes care safer and fairer."},
    },
    "women": {
        "opening_hook": {"eyebrow": "THE QUESTION", "headline": "Agency and the Algorithm", "subtitle": "Who shapes the systems that shape women's lives?"},
        "context": {"eyebrow": "CONTEXT", "headline": "The World as It Is", "subtitle": "Women are navigating AI systems they did not design."},
        "problem_or_tension": {"eyebrow": "THE TENSION", "headline": "Access Is Not Enough", "subtitle": "Having a tool is not the same as having a voice."},
        "human_perspective": {"eyebrow": "THE HUMAN STAKES", "headline": "Voice and Choice", "subtitle": "Real empowerment means real agency — in design, governance, and outcomes."},
        "expert_evidence": {"eyebrow": "WHAT SPEAKERS SAY", "headline": "From the Ground Up", "subtitle": "Women and allies on what change actually looks like."},
        "solution_direction": {"eyebrow": "THE PATH FORWARD", "headline": "Designing With, Not For", "subtitle": "Inclusion begins before the product is built."},
        "final_synthesis": {"eyebrow": "THE MEASURE", "headline": "The Measure of Progress", "subtitle": "Progress means women gain real agency, real voice, and real power."},
    },
    "empowerment": {
        "opening_hook": {"eyebrow": "THE QUESTION", "headline": "Agency and the Algorithm", "subtitle": "Who shapes the systems that shape women's lives?"},
        "context": {"eyebrow": "CONTEXT", "headline": "The World as It Is", "subtitle": "Women are navigating AI systems they did not design."},
        "problem_or_tension": {"eyebrow": "THE TENSION", "headline": "Access Is Not Enough", "subtitle": "Having a tool is not the same as having a voice."},
        "human_perspective": {"eyebrow": "THE HUMAN STAKES", "headline": "Voice and Choice", "subtitle": "Real empowerment means real agency — in design, governance, and outcomes."},
        "expert_evidence": {"eyebrow": "WHAT SPEAKERS SAY", "headline": "From the Ground Up", "subtitle": "Women and allies on what change actually looks like."},
        "solution_direction": {"eyebrow": "THE PATH FORWARD", "headline": "Designing With, Not For", "subtitle": "Inclusion begins before the product is built."},
        "final_synthesis": {"eyebrow": "THE MEASURE", "headline": "The Measure of Progress", "subtitle": "Progress means women gain real agency, real voice, and real power."},
    },
    "governance": {
        "opening_hook": {"eyebrow": "THE QUESTION", "headline": "Who Writes the Rules", "subtitle": "As AI enters public systems, accountability becomes urgent."},
        "context": {"eyebrow": "CONTEXT", "headline": "Power and Code", "subtitle": "Governance is not just technical — it is political."},
        "problem_or_tension": {"eyebrow": "THE TENSION", "headline": "Fast Systems, Slow Rules", "subtitle": "Technology moves faster than the institutions designed to oversee it."},
        "human_perspective": {"eyebrow": "THE HUMAN STAKES", "headline": "Rights Before Algorithms", "subtitle": "Decisions that affect people must be answerable to people."},
        "expert_evidence": {"eyebrow": "WHAT SPEAKERS SAY", "headline": "Voices on Accountability", "subtitle": "Experts and policymakers on what responsible AI governance requires."},
        "solution_direction": {"eyebrow": "THE PATH FORWARD", "headline": "Rules That Protect", "subtitle": "Transparency, accountability, and participation must be built in."},
        "final_synthesis": {"eyebrow": "THE MEASURE", "headline": "The Shape of Accountability", "subtitle": "AI governance will be measured by who it protects and who it answers to."},
    },
    "youth": {
        "opening_hook": {"eyebrow": "THE QUESTION", "headline": "The Next Generation", "subtitle": "A generation that grew up online is now asking harder questions."},
        "context": {"eyebrow": "CONTEXT", "headline": "Born Into the Algorithm", "subtitle": "Young people live inside systems they cannot fully see."},
        "problem_or_tension": {"eyebrow": "THE TENSION", "headline": "Shaped Without Consent", "subtitle": "Algorithmic decisions arrive before young people have a voice."},
        "human_perspective": {"eyebrow": "THE HUMAN STAKES", "headline": "Their Future, Their Voice", "subtitle": "Young people are not passive recipients of the AI era."},
        "expert_evidence": {"eyebrow": "WHAT SPEAKERS SAY", "headline": "Listening to Youth", "subtitle": "What young people and their advocates are saying about AI."},
        "solution_direction": {"eyebrow": "THE PATH FORWARD", "headline": "Participation by Design", "subtitle": "Young voices must be part of how AI is governed."},
        "final_synthesis": {"eyebrow": "THE MEASURE", "headline": "What They Inherit", "subtitle": "The next generation will decide whether to accept or change the systems they are given."},
    },
}

SECTION_HEADLINES = {
    "opening_hook": {
        "eyebrow": "OPENING",
        "headline": "The Future Is Already Here",
        "subtitle": "AI is reshaping how people learn, govern and access knowledge."
    },
    "context": {
        "eyebrow": "CONTEXT",
        "headline": "A New Information Age",
        "subtitle": "The promise of AI depends on who can understand and use it."
    },
    "problem_or_tension": {
        "eyebrow": "THE CHALLENGE",
        "headline": "When Access Is Uneven",
        "subtitle": "New technology can empower communities — or deepen old divides."
    },
    "human_perspective": {
        "eyebrow": "HUMAN PERSPECTIVE",
        "headline": "People at the Center",
        "subtitle": "Youth, women, rural communities and citizens must shape the systems that affect them."
    },
    "expert_evidence": {
        "eyebrow": "EXPERT VIEW",
        "headline": "Evidence from the Field",
        "subtitle": "Across the AI4IA archive, experts connect inclusion with trust, rights and governance."
    },
    "solution_direction": {
        "eyebrow": "THE WAY FORWARD",
        "headline": "Designing for Trust",
        "subtitle": "Responsible AI requires transparency, accountability and local participation."
    },
    "final_synthesis": {
        "eyebrow": "SYNTHESIS",
        "headline": "A Shared Future",
        "subtitle": "AI can become a bridge to knowledge when inclusion is built into its foundation."
    },
}


def log(message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {message}", flush=True)


def slugify(text: str, max_len: int = 80) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "asset"


def latest_storyline_path() -> Path:
    paths = sorted(STORYLINES_DIR.glob("storyline_*.json"))
    if not paths:
        raise FileNotFoundError(f"No storyline files found in {STORYLINES_DIR}")
    return paths[-1]


def load_storyline(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if "storyline" not in data:
        raise ValueError("Storyline file must contain top-level key: storyline")

    return data


def derive_source_video_from_fragment_id(fragment_id: str) -> str | None:
    parts = str(fragment_id).rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return None


def load_speaker_lookup() -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}

    if not SPEAKERS_METADATA_DIR.exists():
        return lookup

    for path in sorted(SPEAKERS_METADATA_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        source_video = data.get("source_video")
        video_id = data.get("video_id")

        if source_video:
            lookup[str(source_video)] = data
        if video_id:
            lookup[str(video_id)] = data

    return lookup


def load_classified_fragment_lookup() -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    speaker_lookup = load_speaker_lookup()

    if not FRAGMENTS_CLASSIFIED_DIR.exists():
        return lookup

    for path in sorted(FRAGMENTS_CLASSIFIED_DIR.glob("*.classified.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log(f"Warning: could not read classified fragments from {path}: {exc}")
            continue

        if not isinstance(payload, dict):
            continue

        fragments = payload.get("fragments", [])
        if not isinstance(fragments, list):
            continue

        container_source_video = payload.get("source_video")
        container_video_id = payload.get("video_id")
        container_video_title = payload.get("video_title") or payload.get("title")
        container_speaker = payload.get("speaker")
        container_speaker_attribution = payload.get("speaker_attribution")

        speaker_meta = (
            speaker_lookup.get(str(container_source_video))
            or speaker_lookup.get(str(container_video_id))
            or {}
        )
        fallback_speaker = speaker_meta.get("primary_speaker")
        fallback_speaker_attribution = speaker_meta.get("attribution")

        for fragment in fragments:
            if not isinstance(fragment, dict):
                continue

            fragment_id = fragment.get("fragment_id")
            if not fragment_id:
                continue

            source_video = (
                fragment.get("source_video")
                or container_source_video
                or derive_source_video_from_fragment_id(str(fragment_id))
            )
            video_id = fragment.get("video_id") or container_video_id
            video_title = fragment.get("video_title") or container_video_title
            speaker = fragment.get("speaker") or container_speaker or fallback_speaker
            speaker_attribution = (
                fragment.get("speaker_attribution")
                or container_speaker_attribution
                or fallback_speaker_attribution
            )

            enriched_fragment = dict(fragment)
            enriched_fragment["source_video"] = source_video

            if source_video and not enriched_fragment.get("source_video_file"):
                enriched_fragment["source_video_file"] = f"{source_video}.mp4"

            if video_id and not enriched_fragment.get("video_id"):
                enriched_fragment["video_id"] = video_id

            if video_title and not enriched_fragment.get("video_title"):
                enriched_fragment["video_title"] = video_title

            if speaker and not enriched_fragment.get("speaker"):
                enriched_fragment["speaker"] = speaker

            if speaker_attribution and not enriched_fragment.get("speaker_attribution"):
                enriched_fragment["speaker_attribution"] = speaker_attribution

            lookup[str(fragment_id)] = enriched_fragment

    return lookup


def enrich_clip_candidate(
    candidate: dict[str, Any],
    fragment_lookup: dict[str, dict[str, Any]],
    metadata_stats: dict[str, Any],
) -> dict[str, Any]:
    fragment_id = candidate.get("fragment_id")
    if not fragment_id:
        return dict(candidate)

    classified_fragment = fragment_lookup.get(str(fragment_id))
    if not classified_fragment:
        enriched = dict(candidate)
        derived_source_video = derive_source_video_from_fragment_id(str(fragment_id))
        if derived_source_video and not enriched.get("source_video"):
            enriched["source_video"] = derived_source_video
        if enriched.get("source_video") and not enriched.get("source_video_file"):
            enriched["source_video_file"] = f"{enriched['source_video']}.mp4"
        enriched["metadata_enrichment_warning"] = "fragment_metadata_not_found"
        metadata_stats["missing_fragment_refs"].add(str(fragment_id))
        return enriched

    enriched = dict(candidate)
    preserved_fields = dict(enriched)

    for key, value in classified_fragment.items():
        if key not in enriched or enriched[key] in (None, "", [], {}):
            enriched[key] = value

    if classified_fragment.get("text"):
        enriched["text"] = classified_fragment["text"]

    enriched["source_video"] = (
        classified_fragment.get("source_video")
        or enriched.get("source_video")
        or derive_source_video_from_fragment_id(str(fragment_id))
    )

    if enriched.get("source_video") and not enriched.get("source_video_file"):
        enriched["source_video_file"] = f"{enriched['source_video']}.mp4"

    if classified_fragment.get("video_id") and not enriched.get("video_id"):
        enriched["video_id"] = classified_fragment["video_id"]

    if classified_fragment.get("video_title") and not enriched.get("video_title"):
        enriched["video_title"] = classified_fragment["video_title"]

    if classified_fragment.get("speaker") and not enriched.get("speaker"):
        enriched["speaker"] = classified_fragment["speaker"]

    if classified_fragment.get("speaker_attribution") and not enriched.get("speaker_attribution"):
        enriched["speaker_attribution"] = classified_fragment["speaker_attribution"]

    # Preserve higher-level retrieval/selection metadata from Step 07.
    for key, value in preserved_fields.items():
        if key in {
            "use_as",
            "relevance_role",
            "final_video_allowed",
            "speaker_clip_allowed",
            "speaker_visual_allowed",
            "speaker_soundbite_allowed",
            "selection_reason",
            "retrieval_channels",
            "direct_evidence_terms",
            "expanded_evidence_terms",
            "final_prompt_score",
            "semantic_score",
            "keyword_score",
            "theme_chapter_score",
            "speaker_score",
        }:
            enriched[key] = value

    metadata_stats["enriched_fragment_refs"] += 1
    return enriched


def enrich_fragment_refs(
    obj: Any,
    fragment_lookup: dict[str, dict[str, Any]],
    metadata_stats: dict[str, Any],
) -> Any:
    if isinstance(obj, dict):
        current = dict(obj)
        if current.get("fragment_id"):
            current = enrich_clip_candidate(current, fragment_lookup, metadata_stats)
        return {
            key: enrich_fragment_refs(value, fragment_lookup, metadata_stats)
            for key, value in current.items()
        }
    if isinstance(obj, list):
        return [enrich_fragment_refs(item, fragment_lookup, metadata_stats) for item in obj]
    return obj


def sanitize_visible_text(text: str) -> str:
    """
    Prevent internal prompt/style instructions from appearing on cards.
    """

    if not text:
        return ""

    blocked_phrases = [
        "create a 3-minute",
        "create a three-minute",
        "national geographic-style",
        "national geographic style",
        "documentary explainer",
        "documentary-style",
        "prompt",
    ]

    cleaned = " ".join(text.split()).strip()

    for phrase in blocked_phrases:
        cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:|")

    return cleaned


def make_display_title(title: str) -> str:
    title = sanitize_visible_text(title)

    if not title:
        return "Empowering Youth, Ensuring Inclusion"

    # Avoid very long title cards.
    if len(title) > 85:
        parts = re.split(r"[:\-–—]", title)
        if parts and len(parts[0].strip()) >= 12:
            title = parts[0].strip()

    return title


def detect_internal_pipeline_language_08(text: str) -> list[str]:
    """Detect internal pipeline terms in viewer-facing text."""
    issues = []
    if not text:
        return issues
    normalized = text.lower()
    for term in INTERNAL_PIPELINE_TERMS_08:
        if term in normalized:
            issues.append(f"internal_pipeline_language_detected: '{term}'")
    return issues


def repair_internal_pipeline_language_08(text: str) -> str:
    """Remove internal pipeline terms from viewer-facing card text."""
    if not text:
        return text
    replacements = [
        (r"\barchive fragments?\b", "recordings"),
        (r"\bselected clips?\b", "speakers"),
        (r"\bconfirmed archive clips?\b", "recordings"),
        (r"\bevidence package\b", "evidence"),
        (r"\bcandidate buckets?\b", "clips"),
        (r"\bvalidation issues?\b", ""),
        (r"\bpipeline\b", ""),
        (r"\bfallback\b", ""),
        (r"\bllm\b", ""),
        (r"\bjson\b", ""),
        (r"\bmanifest\b", ""),
        (r"\bqa report\b", ""),
        (r"\bbackground tasks?\b", ""),
        (r"\bretrieval\b", "research"),
        (r"this video explores\b[^.]*\.", ""),
        (r"the archive evidence shows:?", ""),
        (r"draws exclusively on confirmed archive clips?\.", ""),
        (r"evidence-based context", "evidence"),
    ]
    result = text
    for pattern, repl in replacements:
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
    result = re.sub(r"  +", " ", result).strip(" .")
    return result


def infer_section_copy(
    section: dict[str, Any],
    prompt_profile: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Generate cinematic documentary section card copy. No internal pipeline terms."""
    section_type = section.get("section_type", "section")
    section_role = section.get("role", section_type)
    section_id = section.get("section_id", "")
    claim_type = section.get("claim_type", "")

    topic_terms = list((prompt_profile or {}).get("topic_terms") or [])
    requested_speaker = (prompt_profile or {}).get("requested_speaker")
    primary_topic = topic_terms[0] if topic_terms else None
    topic_label = " ".join(topic_terms[:2]).title() if topic_terms else None

    # Infer effective section type from role or section_type
    effective_type = section_role if section_role != section_type else section_type

    # Evidence gap section
    if claim_type == "evidence_gap" or section_type == "evidence_gap_disclosure":
        return {
            "eyebrow": "EVIDENCE NOTE",
            "headline": "Limited Coverage",
            "subtitle": "The archive does not have enough direct evidence for this topic.",
        }

    # Speaker prompt copy — cinematic chapter style
    if requested_speaker:
        speaker_chapter_map = {
            "hook": {"eyebrow": "THE VOICE", "headline": f"{requested_speaker}", "subtitle": "Contributions to the AI4IA conversation."},
            "opening_hook": {"eyebrow": "THE VOICE", "headline": f"{requested_speaker}", "subtitle": "Contributions to the AI4IA conversation."},
            "human_context": {"eyebrow": "THE CONTEXT", "headline": f"Questions That Matter", "subtitle": f"The themes {requested_speaker} returned to most."},
            "context": {"eyebrow": "THE CONTEXT", "headline": f"Questions That Matter", "subtitle": f"The themes {requested_speaker} returned to most."},
            "tension": {"eyebrow": "THE CHALLENGE", "headline": "The Difficult Questions", "subtitle": "Where the tensions lie."},
            "problem_or_tension": {"eyebrow": "THE CHALLENGE", "headline": "The Difficult Questions", "subtitle": "Where the tensions lie."},
            "expert_evidence": {"eyebrow": "THE EVIDENCE", "headline": "On the Record", "subtitle": f"What {requested_speaker} said."},
            "synthesis": {"eyebrow": "THE MEANING", "headline": "What It Stands For", "subtitle": f"A summation of {requested_speaker}'s contribution."},
            "final_synthesis": {"eyebrow": "THE MEANING", "headline": "What It Stands For", "subtitle": f"A summation of {requested_speaker}'s contribution."},
        }
        for key in (effective_type, section_type):
            if key in speaker_chapter_map:
                return speaker_chapter_map[key]
        readable = effective_type.replace("_", " ").title()
        return {"eyebrow": "THE STORY", "headline": f"{requested_speaker}: {readable}", "subtitle": f"From the AI4IA archive."}

    # Topic prompt copy — use DOCUMENTARY_CHAPTER_TITLES if available
    if primary_topic and primary_topic in DOCUMENTARY_CHAPTER_TITLES:
        topic_chapters = DOCUMENTARY_CHAPTER_TITLES[primary_topic]
        for key in (effective_type, section_type):
            if key in topic_chapters:
                return topic_chapters[key]
        # Map documentary arc roles to standard types
        role_to_type = {
            "hook": "opening_hook",
            "human_context": "context",
            "tension": "problem_or_tension",
            "expert_evidence": "expert_evidence",
            "synthesis": "final_synthesis",
        }
        mapped = role_to_type.get(effective_type) or role_to_type.get(section_type)
        if mapped and mapped in topic_chapters:
            return topic_chapters[mapped]

    # Generic topic copy for topics not in DOCUMENTARY_CHAPTER_TITLES
    if topic_label:
        role_map = {
            "hook": {"eyebrow": "THE QUESTION", "headline": f"AI and {topic_label}", "subtitle": f"What is at stake?"},
            "opening_hook": {"eyebrow": "THE QUESTION", "headline": f"AI and {topic_label}", "subtitle": f"What is at stake?"},
            "human_context": {"eyebrow": "THE CONTEXT", "headline": f"The Human Side", "subtitle": f"People at the center of {topic_label}."},
            "context": {"eyebrow": "THE CONTEXT", "headline": f"The Human Side", "subtitle": f"People at the center of {topic_label}."},
            "tension": {"eyebrow": "THE TENSION", "headline": f"Where It Gets Hard", "subtitle": f"The challenges shaping {topic_label}."},
            "problem_or_tension": {"eyebrow": "THE TENSION", "headline": f"Where It Gets Hard", "subtitle": f"The challenges shaping {topic_label}."},
            "expert_evidence": {"eyebrow": "WHAT SPEAKERS SAY", "headline": f"Voices on {topic_label}", "subtitle": "From the AI4IA conference."},
            "synthesis": {"eyebrow": "THE MEANING", "headline": f"What It Means for {topic_label}", "subtitle": "The story so far."},
            "final_synthesis": {"eyebrow": "THE MEANING", "headline": f"What It Means for {topic_label}", "subtitle": "The story so far."},
            "solution_direction": {"eyebrow": "THE PATH FORWARD", "headline": f"Moving Forward in {topic_label}", "subtitle": "What responsible progress requires."},
        }
        for key in (effective_type, section_type):
            if key in role_map:
                return role_map[key]
        readable = effective_type.replace("_", " ").title()
        return {"eyebrow": "THE STORY", "headline": f"{topic_label}: {readable}", "subtitle": "From the AI4IA archive."}

    # Minimal fallback — no topic info
    readable = effective_type.replace("_", " ").title()
    return {
        "eyebrow": "THE STORY",
        "headline": readable,
        "subtitle": "From the AI4IA archive.",
    }


def find_font(preferred_size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = []

    if bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        ])
    else:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        ])

    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, preferred_size)

    return ImageFont.load_default()


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_spacing: int = 12,
) -> int:
    x, y = xy
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((x, y), test, font=font)
        width = bbox[2] - bbox[0]

        if width <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_spacing

    return y


def create_cinematic_background() -> Image.Image:
    """
    Returns the dark cinematic background (gradients, soft light, subtle
    geometric depth) used as the base layer for every card and narration
    visual. The image has no per-call inputs and is identical every time, so
    it's rendered once and cached to disk (config/cinematic_background.png)
    instead of being recomputed on every one of the several calls per job.
    Delete the cached file to force a fresh render (e.g. after changing the
    gradient/vignette code below).
    """
    if CINEMATIC_BACKGROUND_ASSET.exists():
        return Image.open(CINEMATIC_BACKGROUND_ASSET).convert("RGB")

    image = _render_cinematic_background()
    CINEMATIC_BACKGROUND_ASSET.parent.mkdir(parents=True, exist_ok=True)
    image.save(CINEMATIC_BACKGROUND_ASSET)
    return image


def _render_cinematic_background() -> Image.Image:
    """Pure render of the cinematic background — see create_cinematic_background()."""

    x = np.arange(CARD_WIDTH, dtype=np.float64)
    y = np.arange(CARD_HEIGHT, dtype=np.float64)
    nx, ny = np.meshgrid(x / CARD_WIDTH, y / CARD_HEIGHT)

    # Base cool gradient (int-truncated per channel, matching the original per-pixel computation).
    r = (6 + nx * 14 + ny * 8).astype(np.int64)
    g = (10 + nx * 22 + ny * 12).astype(np.int64)
    b = (18 + nx * 48 + ny * 22).astype(np.int64)

    # Soft vignette.
    dx = nx - 0.55
    dy = ny - 0.45
    dist = np.sqrt(dx * dx + dy * dy)
    vignette = np.maximum(0.35, 1.0 - dist * 1.15)

    rgb = np.stack(
        [(r * vignette).astype(np.int64), (g * vignette).astype(np.int64), (b * vignette).astype(np.int64)],
        axis=-1,
    ).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")

    overlay = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Soft blue glow.
    draw.ellipse(
        (CARD_WIDTH - 760, -280, CARD_WIDTH + 280, 780),
        fill=(35, 120, 220, 45),
    )

    # Warm accent glow.
    draw.ellipse(
        (-280, CARD_HEIGHT - 520, 620, CARD_HEIGHT + 280),
        fill=(230, 160, 80, 26),
    )

    # Fine cinematic grid lines.
    for i in range(0, CARD_WIDTH, 160):
        draw.line((i, 0, i + 260, CARD_HEIGHT), fill=(255, 255, 255, 10), width=1)

    for j in range(160, CARD_HEIGHT, 180):
        draw.line((0, j, CARD_WIDTH, j - 80), fill=(255, 255, 255, 7), width=1)

    overlay = overlay.filter(ImageFilter.GaussianBlur(0.3))
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    return image


def add_cinematic_frame(draw: ImageDraw.ImageDraw) -> None:
    # Letterbox bars.
    draw.rectangle((0, 0, CARD_WIDTH, 84), fill=(0, 0, 0))
    draw.rectangle((0, CARD_HEIGHT - 84, CARD_WIDTH, CARD_HEIGHT), fill=(0, 0, 0))

    # Fine border.
    draw.rectangle((86, 118, CARD_WIDTH - 86, CARD_HEIGHT - 118), outline=(72, 94, 125), width=2)

    # Accent marks.
    draw.rectangle((120, 154, 370, 160), fill=ACCENT_COLOR)
    draw.rectangle((120, 168, 260, 172), fill=GOLD_ACCENT)

    draw.rectangle((CARD_WIDTH - 370, CARD_HEIGHT - 160, CARD_WIDTH - 120, CARD_HEIGHT - 154), fill=ACCENT_COLOR)
    draw.rectangle((CARD_WIDTH - 260, CARD_HEIGHT - 172, CARD_WIDTH - 120, CARD_HEIGHT - 168), fill=GOLD_ACCENT)


def create_title_card(
    output_path: Path,
    title: str,
    logline: str,
) -> None:
    image = create_cinematic_background()
    draw = ImageDraw.Draw(image)

    add_cinematic_frame(draw)

    eyebrow_font = find_font(30, bold=True)
    title_font = find_font(82, bold=True)
    subtitle_font = find_font(34, bold=False)
    footer_font = find_font(24, bold=False)

    title = make_display_title(title)

    visible_logline = sanitize_visible_text(logline)
    if not visible_logline:
        visible_logline = "Human-centered AI for knowledge, access and inclusion."

    if len(visible_logline) > 150:
        visible_logline = visible_logline[:150].rsplit(" ", 1)[0] + "."

    draw.text((130, 185), VISIBLE_BRAND_LABEL, font=eyebrow_font, fill=ACCENT_COLOR)
    draw.text((130, 225), VISIBLE_SERIES_LABEL, font=footer_font, fill=MUTED_TEXT_COLOR)

    y = 350
    y = draw_wrapped_text(
        draw=draw,
        text=title,
        xy=(130, y),
        font=title_font,
        fill=TEXT_COLOR,
        max_width=1500,
        line_spacing=18,
    )

    y += 34

    draw_wrapped_text(
        draw=draw,
        text=visible_logline,
        xy=(134, y),
        font=subtitle_font,
        fill=MUTED_TEXT_COLOR,
        max_width=1280,
        line_spacing=12,
    )

    draw.text(
        (130, CARD_HEIGHT - 148),
        "Access to Information • Responsible AI • Inclusive Futures",
        font=footer_font,
        fill=(200, 210, 222),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def create_section_card(
    output_path: Path,
    section_id: str,
    section_type: str,
    headline: str,
    subtitle: str,
    eyebrow: str,
) -> None:
    image = create_cinematic_background()
    draw = ImageDraw.Draw(image)

    add_cinematic_frame(draw)

    eyebrow_font = find_font(30, bold=True)
    section_font = find_font(24, bold=True)
    headline_font = find_font(76, bold=True)
    subtitle_font = find_font(34, bold=False)
    footer_font = find_font(22, bold=False)

    eyebrow = sanitize_visible_text(eyebrow).upper()
    headline = sanitize_visible_text(headline)
    subtitle = sanitize_visible_text(subtitle)

    if not headline:
        headline = section_type.replace("_", " ").title()

    draw.text((130, 185), eyebrow, font=eyebrow_font, fill=GOLD_ACCENT)
    draw.text((130, 228), f"{VISIBLE_BRAND_LABEL}  /  {section_id}", font=section_font, fill=MUTED_TEXT_COLOR)

    y = 370
    y = draw_wrapped_text(
        draw=draw,
        text=headline,
        xy=(130, y),
        font=headline_font,
        fill=TEXT_COLOR,
        max_width=1420,
        line_spacing=18,
    )

    y += 28

    if subtitle:
        draw_wrapped_text(
            draw=draw,
            text=subtitle,
            xy=(134, y),
            font=subtitle_font,
            fill=MUTED_TEXT_COLOR,
            max_width=1280,
            line_spacing=12,
        )

    draw.text(
        (130, CARD_HEIGHT - 148),
        "The AI4IA Archive",
        font=footer_font,
        fill=(200, 210, 222),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def create_closing_card(
    output_path: Path,
    title: str,
    closing_headline: str | None = None,
    closing_subtitle: str | None = None,
) -> None:
    image = create_cinematic_background()
    draw = ImageDraw.Draw(image)

    add_cinematic_frame(draw)

    eyebrow_font = find_font(30, bold=True)
    headline_font = find_font(74, bold=True)
    subtitle_font = find_font(34, bold=False)
    footer_font = find_font(24, bold=False)

    draw.text((130, 185), "CLOSING", font=eyebrow_font, fill=GOLD_ACCENT)

    y = 370
    y = draw_wrapped_text(
        draw=draw,
        text=closing_headline or make_display_title(title),
        xy=(130, y),
        font=headline_font,
        fill=TEXT_COLOR,
        max_width=1350,
        line_spacing=18,
    )

    y += 28

    draw_wrapped_text(
        draw=draw,
        text=closing_subtitle or "Based on the AI4IA conference archive.",
        xy=(134, y),
        font=subtitle_font,
        fill=MUTED_TEXT_COLOR,
        max_width=1300,
        line_spacing=12,
    )

    draw.text(
        (130, CARD_HEIGHT - 148),
        f"{VISIBLE_BRAND_LABEL} • {make_display_title(title)}",
        font=footer_font,
        fill=(200, 210, 222),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def create_year_stamp_card(
    year: int,
    output_path: Path,
    width: int = 1920,
    height: int = 1080,
) -> Path:
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    year_font = find_font(180, bold=True)
    label_font = find_font(28, bold=False)

    year_text = str(year)
    bbox = draw.textbbox((0, 0), year_text, font=year_font)
    year_w = bbox[2] - bbox[0]
    year_h = bbox[3] - bbox[1]
    year_x = (width - year_w) // 2
    year_y = (height - year_h) // 2 - 40
    draw.text((year_x, year_y), year_text, font=year_font, fill=(255, 255, 255))

    rule_w = width // 5
    rule_x0 = (width - rule_w) // 2
    rule_x1 = rule_x0 + rule_w
    rule_y = year_y + year_h + 28
    draw.rectangle((rule_x0, rule_y, rule_x1, rule_y + 2), fill=(255, 255, 255))

    label_text = " ".join(list("AI4IA CONFERENCE"))
    label_bbox = draw.textbbox((0, 0), label_text, font=label_font)
    label_w = label_bbox[2] - label_bbox[0]
    label_x = (width - label_w) // 2
    label_y = rule_y + 18
    draw.text((label_x, label_y), label_text, font=label_font, fill=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def create_narration_visual_asset(
    output_path: Path,
    visual_title: str,
    visual_asset_type: str,
    visual_keywords: list,
    section_id: str,
) -> None:
    """Generate a topic-specific visual asset PNG for a narration_broll item."""
    W, H = CARD_WIDTH, CARD_HEIGHT
    image = create_cinematic_background()
    draw = ImageDraw.Draw(image)

    # Letterbox bars
    draw.rectangle((0, 0, W, 70), fill=(0, 0, 0))
    draw.rectangle((0, H - 70, W, H), fill=(0, 0, 0))

    title_font = find_font(68, bold=True)
    sub_font = find_font(34, bold=False)
    kw_font = find_font(28, bold=False)
    label_font = find_font(22, bold=True)
    small_font = find_font(18, bold=False)
    footer_font = find_font(20, bold=False)

    type_label = {
        "concept_card": "OVERVIEW",
        "diagram": "EXPLAINER",
        "risk_map": "RISK ANALYSIS",
        "mindmap": "CONNECTIONS",
        "policy_card": "GOVERNANCE",
        "closing_synthesis": "SYNTHESIS",
    }.get(visual_asset_type, "NARRATION")

    draw.text((110, 86), type_label, font=label_font, fill=GOLD_ACCENT)
    draw.text((110, 110), section_id, font=small_font, fill=MUTED_TEXT_COLOR)

    # Main title
    y = draw_wrapped_text(draw, visual_title, (110, 195), title_font, TEXT_COLOR, max_width=1400, line_spacing=14)
    y += 28

    # Accent underline
    draw.rectangle((110, y, 430, y + 4), fill=ACCENT_COLOR)
    y += 32

    kws = (visual_keywords or [])

    if visual_asset_type == "concept_card":
        kw_x, kw_y = 110, y + 20
        for kw in kws[:5]:
            txt = kw.upper()
            bbox = draw.textbbox((kw_x, kw_y), txt, font=kw_font)
            w = bbox[2] - bbox[0]
            pad = 18
            draw.rectangle((kw_x - pad, kw_y - 10, kw_x + w + pad, kw_y + 38), fill=(25, 48, 80), outline=ACCENT_COLOR, width=1)
            draw.text((kw_x, kw_y), txt, font=kw_font, fill=ACCENT_COLOR)
            kw_x += w + pad * 2 + 22
            if kw_x > W - 350:
                kw_x = 110
                kw_y += 64

    elif visual_asset_type == "diagram":
        cx, cy = W // 2, y + 190
        r_c, r_n = 44, 30
        draw.ellipse((cx - r_c, cy - r_c, cx + r_c, cy + r_c), outline=ACCENT_COLOR, width=3, fill=(18, 38, 65))
        center_txt = (kws[0] if kws else "AI").upper()[:8]
        draw.text((cx - 32, cy - 12), center_txt, font=find_font(17, bold=True), fill=TEXT_COLOR)
        offsets = [(-370, -10), (0, -155), (370, -10)]
        branch_labels = (kws[1:] if len(kws) > 1 else kws) + ["concept A", "concept B", "concept C"]
        for (dx, dy_), lbl in zip(offsets, branch_labels[:3]):
            nx, ny = cx + dx, cy + dy_
            draw.line((cx, cy, nx, ny), fill=MUTED_TEXT_COLOR, width=2)
            draw.ellipse((nx - r_n, ny - r_n, nx + r_n, ny + r_n), outline=GOLD_ACCENT, width=2, fill=(22, 42, 68))
            draw.text((nx - 32, ny - 10), lbl[:10].upper(), font=find_font(16, bold=True), fill=TEXT_COLOR)

    elif visual_asset_type == "risk_map":
        cx, cy = W // 2, y + 170
        draw.ellipse((cx - 40, cy - 40, cx + 40, cy + 40), outline=(200, 80, 60), width=3, fill=(48, 12, 12))
        draw.text((cx - 26, cy - 12), "HARM", font=find_font(22, bold=True), fill=(255, 120, 100))
        ao = [(-290, -120), (-290, 120), (0, -175), (290, -120), (290, 120)]
        risk_kws = kws[:5] or ["identity", "consent", "reputation", "harm", "safety"]
        for (dx, dy_), lbl in zip(ao, risk_kws):
            nx, ny = cx + dx, cy + dy_
            draw.line((cx, cy, nx, ny), fill=(110, 55, 55), width=2)
            draw.ellipse((nx - 28, ny - 28, nx + 28, ny + 28), outline=(200, 80, 60), width=2, fill=(38, 12, 12))
            draw.text((nx - 26, ny - 10), lbl[:10].upper(), font=find_font(14, bold=True), fill=(220, 160, 150))

    elif visual_asset_type == "mindmap":
        cx, cy = W // 2, y + 150
        draw.ellipse((cx - 58, cy - 30, cx + 58, cy + 30), outline=ACCENT_COLOR, width=2, fill=(14, 32, 60))
        center_txt = (kws[0] if kws else "TOPIC").upper()[:10]
        draw.text((cx - 42, cy - 12), center_txt, font=find_font(18, bold=True), fill=TEXT_COLOR)
        branch_kws = kws[1:6] if len(kws) > 1 else kws * 4
        bxs = [cx - 460, cx - 230, cx, cx + 230, cx + 460]
        for bx, kw in zip(bxs, branch_kws[:5]):
            by = cy + 130
            draw.line((cx, cy + 30, bx, by - 20), fill=MUTED_TEXT_COLOR, width=1)
            draw.rectangle((bx - 68, by - 20, bx + 68, by + 20), outline=GOLD_ACCENT, width=1, fill=(18, 32, 52))
            draw.text((bx - 58, by - 8), kw[:12].upper(), font=find_font(14, bold=True), fill=GOLD_ACCENT)

    elif visual_asset_type == "policy_card":
        pillars = kws[:4] or ["transparency", "accountability", "detection", "literacy"]
        cols = [150, W // 2 + 50]
        rows = [y + 20, y + 148]
        for i, pillar in enumerate(pillars):
            col = cols[i % 2]
            row_y = rows[i // 2]
            draw.rectangle((col, row_y, col + 410, row_y + 108), fill=(16, 32, 58), outline=ACCENT_COLOR, width=1)
            draw.text((col + 14, row_y + 14), f"  {pillar.upper()}", font=find_font(22, bold=True), fill=ACCENT_COLOR)
            draw.text((col + 14, row_y + 58), "Policy pillar", font=small_font, fill=MUTED_TEXT_COLOR)

    elif visual_asset_type == "closing_synthesis":
        concepts = kws[:3] or ["trust", "safety", "governance"]
        pill_x, pill_y = 110, y + 30
        for concept in concepts:
            txt = concept.upper()
            bbox = draw.textbbox((pill_x, pill_y), txt, font=sub_font)
            w = bbox[2] - bbox[0]
            pad = 22
            draw.rectangle((pill_x - pad, pill_y - 12, pill_x + w + pad, pill_y + 46), fill=(18, 48, 88), outline=ACCENT_COLOR, width=2)
            draw.text((pill_x, pill_y), txt, font=sub_font, fill=TEXT_COLOR)
            pill_x += w + pad * 2 + 28
            if pill_x > W - 280:
                pill_x = 110
                pill_y += 96

    draw.text((110, H - 136), f"AI4IA 2026  ·  {visual_title[:60]}", font=footer_font, fill=(180, 195, 210))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path))


def generate_visual_assets_for_timeline(
    final_timeline: list,
    visuals_dir: Path,
) -> list:
    """Generate PIL visual assets for narration_broll items with visual_mode='generated_asset'.

    Sets visual_path, render_visual_path, asset_path on each qualifying item.
    Returns the updated list.
    """
    visuals_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for item in final_timeline:
        item = dict(item)
        t = item.get("type") or item.get("segment_type") or ""
        # A narration item without archive footage MUST have a generated visual,
        # otherwise the renderer has nothing to put on screen for it.
        needs_visual = item.get("visual_mode") == "generated_asset" or not item.get("source_video")
        if t == "narration_broll" and needs_visual:
            sid = item.get("section_id") or "S_UNKNOWN"
            vtype = item.get("visual_asset_type") or "concept_card"
            vtitle = item.get("visual_title") or sid.replace("_", " ").title()
            vkws = item.get("visual_keywords") or []
            output_path = visuals_dir / f"{sid}_visual.png"
            try:
                create_narration_visual_asset(
                    output_path=output_path,
                    visual_title=vtitle,
                    visual_asset_type=vtype,
                    visual_keywords=vkws,
                    section_id=sid,
                )
                item["visual_path"] = str(output_path)
                item["render_visual_path"] = str(output_path)
                item["asset_path"] = str(output_path)
                item["visual_mode"] = "generated_asset"
                log(f"  [visual] {sid} → {output_path.name} ({vtype})")
            except Exception as exc:
                log(f"  [visual] WARNING: failed to render visual for {sid}: {exc}")
        result.append(item)
    return result


def build_voiceover_script(storyline: dict[str, Any]) -> list[dict[str, str]]:
    sections = storyline.get("sections", [])
    voiceover_script = []

    for section in sections:
        section_id = section.get("section_id", "")
        section_type = section.get("section_type", "")
        # Phase 4O.3: use narration first, fall back to viewer_narration
        narration = (section.get("narration") or section.get("viewer_narration") or "").strip()

        if narration:
            voiceover_script.append(
                {
                    "section_id": section_id,
                    "section_type": section_type,
                    "text": narration,
                    "tone": "calm cinematic documentary narrator",
                }
            )

    # Phase 4O.3: if no sections produced narration, build a fallback from top-level fields
    if not voiceover_script:
        logline = (storyline.get("logline") or "").strip()
        closing = (storyline.get("closing_meaning") or "").strip()
        thesis = (storyline.get("narrative_thesis") or "").strip()
        central_q = (storyline.get("central_question") or "").strip()

        fallback_parts = [p for p in [logline, thesis or central_q, closing] if p]
        if fallback_parts:
            voiceover_script.append(
                {
                    "section_id": "S_FALLBACK",
                    "section_type": "narration_fallback",
                    "text": "  ".join(fallback_parts),
                    "tone": "calm cinematic documentary narrator",
                }
            )

    return voiceover_script


def voiceover_text_for_tts(voiceover_script: list[dict[str, str]]) -> str:
    parts = []

    for item in voiceover_script:
        text = item.get("text", "").strip()
        if text:
            parts.append(text)

    return "\n\n".join(parts).strip()


def get_media_duration(path: Path) -> float:
    """Probe a media file's duration in seconds via ffprobe. Returns 0.0 on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip() or 0.0)
    except (subprocess.SubprocessError, ValueError, OSError):
        return 0.0


def reconcile_narration_duration(item: dict[str, Any], audio_path: str | None) -> None:
    """Set render_duration on a narration item from its real TTS audio length.

    The segment must run long enough for the full voiceover (audio is never trimmed);
    if the fragment window is longer than the audio, the window length wins.
    """
    frag_dur = float(item.get("duration") or 0.0)
    if frag_dur <= 0.0:
        start_v, end_v = item.get("start"), item.get("end")
        try:
            frag_dur = max(0.0, float(end_v) - float(start_v))
        except (TypeError, ValueError):
            frag_dur = 0.0

    audio_dur = 0.0
    if audio_path and Path(audio_path).exists():
        audio_dur = get_media_duration(Path(audio_path))

    render_duration = max(frag_dur, audio_dur + 0.4 if audio_dur > 0 else 0.0)
    if render_duration > 0:
        item["render_duration"] = round(render_duration, 3)
        item["duration"] = item["render_duration"]
        item["tts_audio_seconds"] = round(audio_dur, 3)


async def generate_tts(
    text: str,
    output_path: Path,
    voice: str,
    rate: str,
    volume: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        volume=volume,
    )

    await communicate.save(str(output_path))


def write_voiceover_script_txt(
    output_path: Path,
    title: str,
    voiceover_script: list[dict[str, str]],
) -> None:
    lines = []
    lines.append(make_display_title(title))
    lines.append("=" * len(make_display_title(title)))
    lines.append("")

    for item in voiceover_script:
        section_id = item.get("section_id", "")
        section_type = item.get("section_type", "").replace("_", " ").title()
        text = item.get("text", "")

        lines.append(f"{section_id} - {section_type}")
        lines.append("-" * 60)
        lines.append(text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def collect_archive_fragments(storyline: dict[str, Any]) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    selected_candidates = storyline.get("selected_candidates", []) or []

    for item in selected_candidates:
        if not item.get("final_video_allowed") or not item.get("fragment_id"):
            continue
        record = dict(item)
        record["asset_role"] = "selected_candidate"
        fragments.append(record)

    if fragments:
        return dedupe_manifest_fragments(fragments)

    for section in storyline.get("sections", []):
        section_id = section.get("section_id")

        for item in section.get("archive_visual_fragments", []):
            record = dict(item)
            record["section_id"] = section_id
            record["asset_role"] = "archive_visual"
            fragments.append(record)

        for item in section.get("speaker_soundbites", []):
            record = dict(item)
            record["section_id"] = section_id
            record["asset_role"] = "speaker_soundbite"
            fragments.append(record)

    return dedupe_manifest_fragments(fragments)


def dedupe_manifest_fragments(fragments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fragment in fragments:
        fragment_id = str(fragment.get("fragment_id") or "").strip()
        if not fragment_id or fragment_id in seen:
            continue
        seen.add(fragment_id)
        deduped.append(fragment)
    return deduped


def build_export_timeline_from_candidates(
    selected_candidates: list[dict[str, Any]],
    prompt_profile: dict[str, Any],
    target_duration_seconds: int,
    title: str | None = None,
) -> list[dict[str, Any]]:
    safe_candidates = dedupe_manifest_fragments(
        [dict(item) for item in selected_candidates if item.get("final_video_allowed") is True and item.get("fragment_id")]
    )
    if not safe_candidates:
        return []

    def sort_key(candidate: dict[str, Any]) -> tuple:
        type_priority = {
            "speaker_clip": 4,
            "speaker_visual": 3,
            "archive_visual": 2,
            "narration_broll": 1,
        }.get(classify_timeline_segment_type(candidate), 0)
        role_priority = {
            "exact": 3,
            "strong_related": 2,
            "weak_related": 1,
        }.get(str(candidate.get("relevance_role") or ""), 0)
        return (
            type_priority,
            role_priority,
            float(candidate.get("final_prompt_score") or candidate.get("prompt_relevance_score") or 0.0),
            float(candidate.get("selection_score") or 0.0),
            float(candidate.get("semantic_score") or 0.0),
        )

    ranked = sorted(safe_candidates, key=sort_key, reverse=True)
    desired = min(timeline_segment_count_for_duration(target_duration_seconds), len(ranked))
    chosen = ranked[:desired]
    if not chosen:
        return []

    card_duration = timeline_card_duration(target_duration_seconds)
    content_budget = max(1, target_duration_seconds - (card_duration * 2))
    per_segment_duration = max(6.0, round(content_budget / max(1, len(chosen)), 2))

    timeline: list[dict[str, Any]] = [
        {
            "type": "title_card",
            "title": title or prompt_profile.get("raw_prompt") or "AI4IA Storyline",
            "duration": float(card_duration),
        }
    ]
    for candidate in chosen:
        fragment_id = str(candidate.get("fragment_id") or "").strip()
        if not fragment_id:
            continue
        start_value, end_value, duration_value = normalize_candidate_time_bounds(candidate)
        clip_duration = duration_value if duration_value is not None else per_segment_duration
        clip_duration = max(1.0, min(per_segment_duration, clip_duration))
        if start_value is not None:
            end_value = start_value + clip_duration if end_value is None else min(end_value, start_value + clip_duration)
            clip_duration = max(0.1, end_value - start_value)
        timeline.append(
            {
                "type": classify_timeline_segment_type(candidate),
                "fragment_id": fragment_id,
                "source_video": candidate.get("source_video"),
                "start": round(float(start_value), 3) if start_value is not None else None,
                "end": round(float(end_value), 3) if end_value is not None else None,
                "duration": round(float(clip_duration), 3),
                "use_as": candidate.get("use_as"),
                "relevance_role": candidate.get("relevance_role"),
                "selection_reason": candidate.get("selection_reason"),
                "speaker_clip_allowed": bool(candidate.get("speaker_clip_allowed") or candidate.get("speaker_soundbite_allowed")),
                "speaker_visual_allowed": bool(candidate.get("speaker_visual_allowed")),
            }
        )
    timeline.append(
        {
            "type": "closing_card",
            "title": title or prompt_profile.get("raw_prompt") or "AI4IA Storyline",
            "duration": float(card_duration),
        }
    )
    return timeline


def collect_generated_assets(storyline: dict[str, Any]) -> list[dict[str, Any]]:
    assets = []

    for section in storyline.get("sections", []):
        section_id = section.get("section_id")

        for item in section.get("generated_assets_needed", []):
            record = dict(item)
            record["section_id"] = section_id
            assets.append(record)

    return assets


def collect_internet_assets(storyline: dict[str, Any]) -> list[dict[str, Any]]:
    assets = []

    for section in storyline.get("sections", []):
        section_id = section.get("section_id")

        for item in section.get("internet_assets_needed", []):
            record = dict(item)
            record["section_id"] = section_id
            assets.append(record)

    return assets


def detect_generic_card_title(title: str) -> bool:
    return title.lower().strip() in GENERIC_CARD_TITLES


def detect_generic_voiceover(text: str) -> bool:
    if not text:
        return False
    normalized = text.lower()
    return any(phrase in normalized for phrase in DISALLOWED_VOICEOVER_PHRASES)


MOMENT_FIELDS_FROM_STORYLINE = {
    "moment_text",
    "moment_summary",
    "muted",
    "subtitle_text",
    "subtitle_kind",
    "section_role",
    "moment_selection_reason",
    "speech_quality",
    "moment_scores",
    "segment_type",
    "audio_mode",
}

INTRO_ADMIN_TERMS_08 = [
    "hello everyone",
    "good morning",
    "good afternoon",
    "thank you",
    "thanks for inviting me",
    "let me introduce",
    "next slide",
    "moving on",
    "today i will",
    "my name is",
    "it is an honor",
    "housekeeping",
    "restroom",
    "closed captioning",
    "can you hear me",
    "welcome remarks",
]


def enrich_timeline_with_moment_fields(
    export_timeline: list[dict[str, Any]],
    storyline_timeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Enrich asset manifest export_timeline with moment-level fields from the
    storyline's export_timeline. Existing moment fields are NOT overwritten.
    """
    # Build lookup by fragment_id from storyline timeline
    storyline_by_fid: dict[str, dict[str, Any]] = {}
    for item in storyline_timeline:
        fid = item.get("fragment_id")
        if fid:
            storyline_by_fid[str(fid)] = item

    enriched = []
    for item in export_timeline:
        item = dict(item)
        fid = item.get("fragment_id")
        if fid:
            source = storyline_by_fid.get(str(fid), {})
            for field in MOMENT_FIELDS_FROM_STORYLINE:
                if field not in item and field in source:
                    item[field] = source[field]

        # Phase 4Q: speaker_clip items must never receive AI voiceover
        seg_type = item.get("segment_type") or item.get("type") or ""
        if seg_type == "speaker_clip":
            item["voiceover_text"] = ""
            item["voiceover_path"] = None
            item["audio_mode"] = "source_audio"
            item["muted"] = False
            # subtitle_text must be moment_text, not narration
            if not item.get("subtitle_text"):
                item["subtitle_text"] = item.get("moment_text", "")

        enriched.append(item)
    return enriched


def _detect_intro_admin_in_moment(text: str) -> bool:
    """Return True if the moment text is dominated by intro/admin language."""
    if not text:
        return False
    normalized = text.lower()
    count = sum(1 for term in INTRO_ADMIN_TERMS_08 if term in normalized)
    return count >= 2


def validate_asset_manifest_before_save(
    manifest: dict[str, Any],
    prompt_profile: dict[str, Any],
    selected_candidates: list[dict[str, Any]],
) -> list[str]:
    """Strict pre-save validation for asset manifest."""
    if not STRICT_NO_GENERIC_FALLBACK:
        return []

    issues = []
    cards = manifest.get("cards") or {}
    for sc in (cards.get("section_cards") or []):
        headline = sc.get("headline", "")
        subtitle = sc.get("subtitle", "")
        sid = sc.get("section_id", "?")
        if detect_generic_card_title(headline):
            issues.append(f"generic_card_title_detected: '{headline}' in section {sid}")
        pipeline_issues = detect_internal_pipeline_language_08(headline)
        if pipeline_issues:
            issues.append(f"card_not_documentary_style: internal language in section {sid} headline")
        pipeline_issues_sub = detect_internal_pipeline_language_08(subtitle)
        if pipeline_issues_sub:
            issues.append(f"card_not_documentary_style: internal language in section {sid} subtitle")

    voice = manifest.get("voice") or {}
    for sv in (voice.get("section_voiceovers") or []):
        text = sv.get("text", "")
        sid = sv.get("section_id", "?")
        if detect_generic_voiceover(text):
            issues.append(f"generic_voiceover_detected in section {sid}")
        pipeline_issues_vo = detect_internal_pipeline_language_08(text)
        if pipeline_issues_vo:
            issues.append(f"internal_pipeline_language_detected in voiceover section {sid}")

    storyline = manifest.get("storyline") or {}
    export_timeline = storyline.get("export_timeline") or []
    timeline_fragment_ids = {item.get("fragment_id") for item in export_timeline if item.get("fragment_id")}
    if selected_candidates and not timeline_fragment_ids:
        issues.append("missing_evidence_links: export_timeline has no fragment_ids despite having selected_candidates")

    closing_headline = manifest.get("closing_headline") or ""
    if closing_headline:
        pipeline_close = detect_internal_pipeline_language_08(closing_headline)
        if pipeline_close:
            issues.append(f"closing_not_prompt_specific: internal language in closing headline")

    # Phase 4O: moment-level validation on export_timeline
    storyline = manifest.get("storyline") or {}
    manifest_timeline = (storyline.get("export_timeline") or [])
    for titem in manifest_timeline:
        seg_type = titem.get("type", "")
        fid = titem.get("fragment_id")
        if not fid or seg_type in ("title_card", "closing_card"):
            continue

        moment_text = titem.get("moment_text", "")
        if seg_type == "speaker_clip" and not moment_text:
            issues.append(f"selected_clip_missing_moment_text: {fid}")

        if seg_type == "speaker_clip" and moment_text and _detect_intro_admin_in_moment(moment_text):
            issues.append(f"speaker_clip_has_intro_language: {fid}")

        speech_q = titem.get("speech_quality") or {}
        if seg_type == "speaker_clip" and speech_q and not speech_q.get("is_usable_speech", True):
            issues.append(f"speaker_clip_has_low_speech_ratio: {fid}")

        if seg_type == "archive_visual" and titem.get("muted") is False:
            issues.append(f"archive_visual_not_muted: {fid}")

    return issues


def _text_ends_complete(text: str) -> bool:
    """Return True if text ends with terminal punctuation."""
    t = (text or "").rstrip()
    return bool(t) and t[-1] in ".!?"


def _text_starts_mid_sentence(text: str) -> bool:
    """Return True if text likely starts mid-sentence (first char is lowercase)."""
    t = (text or "").lstrip()
    return bool(t) and t[0].islower()


def compute_quote_render_bounds(
    item: dict[str, Any],
    cand: dict[str, Any],
    fragment_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Compute render_start, render_end, render_duration, quote_complete for a speaker clip.

    Uses the full fragment window (start → end) + padding as the baseline.
    Attempts forward expansion using the source fragment boundary if the quote is incomplete.
    Only demotes if quote_completeness < 0.20 AND incomplete_sentence_penalty > 0.30.

    Never shrinks the validated window: the render window may only pad or expand the
    selected timeframe. Content trimming is 07b validation's job alone.
    """
    # A window already set by 07b validation is a locked contract — pass it through.
    locked_start = item.get("render_start")
    locked_end = item.get("render_end")
    if locked_start is not None and locked_end is not None and float(locked_end) > float(locked_start):
        return {
            "render_start": round(float(locked_start), 3),
            "render_end": round(float(locked_end), 3),
            "render_duration": round(float(locked_end) - float(locked_start), 3),
            "quote_complete": bool(item.get("quote_complete", True)),
            "quote_boundary_method": "locked_by_validation",
            "quote_expanded": False,
            "should_demote": False,
        }

    start = float(item.get("start") or 0.0)
    end = float(item.get("end") or 0.0)
    moment_text = (item.get("moment_text") or cand.get("moment_text") or "").strip()
    moment_scores = item.get("moment_scores") or cand.get("moment_scores") or {}

    completeness = float(moment_scores.get("quote_completeness_score") or 0.0)
    penalty = float(moment_scores.get("incomplete_sentence_penalty") or 0.0)

    text_complete = _text_ends_complete(moment_text)
    text_mid_start = _text_starts_mid_sentence(moment_text)

    is_complete = text_complete or (completeness >= QUOTE_COMPLETE_THRESHOLD and penalty < 0.05)

    # Start with the full fragment window + padding
    render_start = max(0.0, start - PRE_ROLL_SECONDS)
    render_end = end + POST_ROLL_SECONDS
    method = "complete_sentence" if is_complete else "padding_only"
    expanded = False

    # Try to expand an incomplete quote using the source fragment boundary
    if not is_complete and fragment_lookup:
        fid = str(item.get("fragment_id") or "")
        cf = fragment_lookup.get(fid) or {}
        raw_source_end = cf.get("end")
        if raw_source_end is not None:
            source_end = float(raw_source_end)
            available = source_end - end
            if available > 1.5:
                expand_by = min(available - 0.1, CLIP_EXPAND_SECONDS)
                if expand_by > 0.5:
                    render_end = end + expand_by + POST_ROLL_SECONDS
                    expanded = True
                    method = "sentence_or_segment_expansion"
                else:
                    method = "at_fragment_boundary_no_expansion"
            else:
                method = "at_fragment_boundary_no_expansion"

    # The full validated window always renders — no hard-max shrink, no artificial
    # min-duration stretch into unvalidated footage.
    render_duration = render_end - render_start

    # Demote only if quote is truly unusable (very low completeness + very high penalty)
    should_demote = (
        not text_complete
        and completeness < 0.20
        and penalty > 0.30
    )

    return {
        "render_start": round(render_start, 3),
        "render_end": round(render_end, 3),
        "render_duration": round(render_duration, 3),
        "quote_complete": is_complete,
        "quote_boundary_method": method,
        "quote_expanded": expanded,
        "should_demote": should_demote,
    }


def _speaker_clip_quality_score(item: dict[str, Any]) -> float:
    """Score a timeline fragment item for speaker clip suitability (0.0–1.0)."""
    sq = item.get("speech_quality") or {}
    ms = item.get("moment_scores") or {}
    is_usable = sq.get("is_usable_speech", True)
    speech_ratio = float(sq.get("speech_ratio") or 0.0)
    final_score = float(ms.get("final_moment_score") or 0.0)
    base = speech_ratio * 0.4 + final_score * 0.6
    # If speech quality heuristic says not usable but the candidate was explicitly approved
    # as a speaker soundbite (speaker_soundbite_allowed / speaker_clip_allowed), use a floor
    # score so the annotation decision is not overridden by the heuristic alone.
    if not is_usable:
        explicitly_approved = item.get("speaker_soundbite_allowed") or item.get("speaker_clip_allowed")
        if explicitly_approved:
            return max(base, 0.30)
        return 0.0
    return base


def build_hybrid_export_timeline(
    export_timeline: list[dict[str, Any]],
    selected_candidates: list[dict[str, Any]],
    section_voiceover_paths: list[str],
    max_speaker_clips: int = 5,
    speaker_clip_score_threshold: float = 0.20,
    fragment_lookup: dict[str, dict[str, Any]] | None = None,
    vo_path_to_text: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Build a hybrid timeline mixing speaker_clip/source_audio and narration_broll/ai_voiceover.

    Algorithm:
    1. Non-card fragment items are scored for speaker clip suitability.
    2. Top-N (up to max_speaker_clips) items that pass threshold become speaker_clip.
    3. Speaker clips receive render_start/render_end/render_duration from compute_quote_render_bounds().
    4. Remaining fragment items become narration_broll, each receiving an audio_path
       from section_voiceover_paths (cycling if needed), with subtitle_text set from vo_path_to_text.
    5. All type fields (type, asset_type, kind, segment_type) are normalised.
    """
    candidate_by_fid: dict[str, dict[str, Any]] = {}
    for cand in (selected_candidates or []):
        fid = cand.get("fragment_id")
        if fid:
            candidate_by_fid[str(fid)] = cand

    # Collect eligible fragment items (non-card, has fragment_id, valid duration)
    card_types = {"title_card", "closing_card"}
    fragment_items: list[tuple[int, dict[str, Any]]] = []  # (original_index, item)
    card_items: list[tuple[int, dict[str, Any]]] = []

    for i, raw_item in enumerate(export_timeline):
        item = dict(raw_item)
        t = item.get("type") or item.get("segment_type") or ""
        if t in card_types:
            card_items.append((i, item))
        elif item.get("fragment_id"):
            fragment_items.append((i, item))
        else:
            card_items.append((i, item))  # treat non-fragment non-card as pass-through

    # Score eligible fragment items.
    # Items already typed as "speaker_clip" by 07b validation are pre-approved — don't re-score them.
    scored = []
    pre_approved_speaker_fids: set[str] = set()

    for orig_idx, item in fragment_items:
        fid = item.get("fragment_id") or ""
        cand = candidate_by_fid.get(str(fid), {})
        pre_type = item.get("type") or item.get("segment_type") or ""

        # 07b already approved this as a speaker clip — honour it, bypass re-scoring
        if pre_type == "speaker_clip" and fid:
            pre_approved_speaker_fids.add(str(fid))
            scored.append((orig_idx, item, cand, 1.0, True))
            continue

        start = item.get("start")
        end = item.get("end")
        source_video = item.get("source_video") or cand.get("source_video")
        role = item.get("relevance_role") or cand.get("relevance_role") or ""

        # Basic eligibility for speaker clip
        eligible = (
            source_video is not None
            and start is not None
            and end is not None
            and role in ("exact", "strong_related")
        )
        try:
            dur = float(item.get("duration") or 0.0) or (float(end or 0) - float(start or 0))
            if not (6.0 <= dur <= 60.0):
                eligible = False
        except (TypeError, ValueError):
            eligible = False

        score = _speaker_clip_quality_score(item) if eligible else 0.0
        scored.append((orig_idx, item, cand, score, eligible))

    # Select top-N as speaker clips (pre-filter: remove those flagged for demotion by bounds check)
    # Pre-approved items (from 07b) are always included first, then fill remaining slots from scored.
    eligible_only = [(i, item, cand, sc) for i, item, cand, sc, elig in scored if elig]
    eligible_only.sort(key=lambda x: (
        0 if str(x[1].get("fragment_id") or "") in pre_approved_speaker_fids else 1,
        -x[3]
    ))

    # Pre-compute bounds for all eligible items to check should_demote
    bounds_cache: dict[int, dict[str, Any]] = {}
    for i, item, cand, sc in eligible_only:
        b = compute_quote_render_bounds(item, cand, fragment_lookup)
        bounds_cache[i] = b

    speaker_clip_fids: set[str] = set()
    selected_for_speaker = []
    for i, item, cand, sc in eligible_only:
        fid = str(item.get("fragment_id") or "")
        is_pre_approved = fid in pre_approved_speaker_fids
        # Pre-approved items (07b validated) bypass score and demotion thresholds
        if not is_pre_approved:
            if sc < speaker_clip_score_threshold:
                continue
            if len(selected_for_speaker) >= max_speaker_clips:
                break
            b = bounds_cache.get(i, {})
            if b.get("should_demote"):
                log(f"  [hybrid] Demoting {item.get('fragment_id', '')[-40:]!r}: quote too incomplete")
                continue
        speaker_clip_fids.add(fid)
        selected_for_speaker.append(i)

    # Build voiceover cycling iterator for narration_broll items
    valid_vo_paths = [p for p in (section_voiceover_paths or []) if p and Path(p).exists()]
    vo_cycle_idx = 0

    # Diagnostics counters
    stat_complete = 0
    stat_expanded = 0
    stat_demoted = 0
    stat_zero_dur = 0

    # Rebuild timeline preserving original order
    result_by_idx: dict[int, dict[str, Any]] = {}

    for orig_idx, item, cand, score, eligible in scored:
        item = dict(item)
        fid = str(item.get("fragment_id") or "")

        if fid in speaker_clip_fids:
            # Promote to speaker_clip with computed render bounds
            bounds = bounds_cache.get(orig_idx) or compute_quote_render_bounds(item, cand, fragment_lookup)
            item["type"] = "speaker_clip"
            item["asset_type"] = "speaker_clip"
            item["kind"] = "speaker_clip"
            item["segment_type"] = "speaker_clip"
            item["audio_mode"] = "source_audio"
            item["muted"] = False
            item["voiceover_text"] = ""
            item["voiceover_path"] = None
            item["audio_path"] = None
            # Apply render bounds — keep original start/end for provenance
            item["render_start"] = bounds["render_start"]
            item["render_end"] = bounds["render_end"]
            item["render_duration"] = bounds["render_duration"]
            item["quote_complete"] = bounds["quote_complete"]
            item["quote_boundary_method"] = bounds["quote_boundary_method"]
            # Use render_duration as the operative duration
            item["duration"] = bounds["render_duration"]
            if not item.get("subtitle_text"):
                item["subtitle_text"] = item.get("moment_text") or ""
            # Carry over cleaned corpus metadata
            for field in (
                "source_transcript_version", "audio_quality_status",
                "transcript_quality_score", "word_timing_available",
                "text_hash", "text_truncated_for_embedding",
            ):
                if not item.get(field) and cand.get(field) is not None:
                    item[field] = cand[field]
            if bounds["quote_complete"]:
                stat_complete += 1
            if bounds.get("quote_expanded"):
                stat_expanded += 1
            log(
                f"  [hybrid] speaker_clip {fid[-44:]!r}: "
                f"render {bounds['render_start']:.2f}→{bounds['render_end']:.2f}s "
                f"({bounds['render_duration']:.2f}s) complete={bounds['quote_complete']} "
                f"method={bounds['quote_boundary_method']}"
            )
        else:
            # Demote to narration_broll
            item["type"] = "narration_broll"
            item["asset_type"] = "narration_broll"
            item["kind"] = "narration_broll"
            item["segment_type"] = "narration_broll"
            item["audio_mode"] = "ai_voiceover"
            item["muted"] = True
            b = bounds_cache.get(orig_idx, {})
            if b.get("should_demote"):
                stat_demoted += 1
            # Assign section voiceover path by cycling
            if valid_vo_paths:
                vo_path = valid_vo_paths[vo_cycle_idx % len(valid_vo_paths)]
                item["audio_path"] = vo_path
                item["voiceover_path"] = vo_path
                # Set subtitle/voiceover text from path→text map
                if vo_path_to_text:
                    vo_text = vo_path_to_text.get(vo_path, "")
                    item["subtitle_text"] = vo_text
                    item["voiceover_text"] = vo_text
                    item["subtitle_kind"] = "voiceover"
                vo_cycle_idx += 1
            # Reconcile duration against the real TTS audio length (never trims audio)
            reconcile_narration_duration(item, item.get("audio_path"))
            if float(item.get("duration") or 0.0) == 0.0:
                stat_zero_dur += 1

        result_by_idx[orig_idx] = item

    # Pass-through for cards and other items
    for orig_idx, item in card_items:
        item = dict(item)
        t = item.get("type") or item.get("segment_type") or ""
        if t in card_types:
            item["type"] = t
            item["asset_type"] = t
            item["kind"] = t
            item["segment_type"] = t
        result_by_idx[orig_idx] = item

    # Reconstruct in original order
    result = [result_by_idx[i] for i in range(len(export_timeline)) if i in result_by_idx]

    # Print diagnostics
    sc_count = sum(1 for x in result if (x.get("type") or x.get("segment_type")) == "speaker_clip")
    nb_count = sum(1 for x in result if (x.get("type") or x.get("segment_type")) == "narration_broll")
    log("")
    log("Hybrid timeline quote timing summary:")
    log(f"  speaker clips:           {sc_count}")
    log(f"  complete quotes:         {stat_complete}")
    log(f"  expanded quotes:         {stat_expanded}")
    log(f"  demoted incomplete:      {stat_demoted}")
    log(f"  narration_broll:         {nb_count}")
    log(f"  zero-duration items:     {stat_zero_dur}")
    log("")
    for x in result:
        st = x.get("segment_type") or x.get("type") or ""
        if st != "speaker_clip":
            continue
        fid = x.get("fragment_id", "")[-50:]
        log(
            f"  SC {fid!r}: "
            f"src={x.get('start'):.2f}→{x.get('end'):.2f} "
            f"render={x.get('render_start'):.2f}→{x.get('render_end'):.2f} "
            f"({x.get('render_duration'):.2f}s) "
            f"complete={x.get('quote_complete')} "
            f"speech_ratio={x.get('speech_quality', {}).get('speech_ratio', '?')} "
            f"score={x.get('moment_scores', {}).get('final_moment_score', '?')}"
        )

    return result


def preflight_audit_manifest(manifest: dict[str, Any], skip_tts: bool = False) -> list[str]:
    """Verify every timeline item is renderable BEFORE handing off to Step 09.

    Returns a list of failure strings; empty means the manifest is render-safe.
    Checks: card PNGs exist, every content item has a positive duration, voiceover
    audio files exist on disk, and narration items without archive footage carry a
    generated visual PNG that exists.
    """
    failures: list[str] = []
    cards = manifest.get("cards") or {}

    for card_key in ("title_card", "closing_card"):
        card_path = cards.get(card_key)
        if not card_path or not Path(card_path).exists():
            failures.append(f"{card_key}: PNG missing at {card_path!r}")

    timeline = manifest.get("timeline") or manifest.get("export_timeline") or []
    for idx, item in enumerate(timeline, start=1):
        t = item.get("segment_type") or item.get("type") or ""
        label = f"item {idx} ({t}, fragment={str(item.get('fragment_id') or '')[:40]!r})"

        if t in ("title_card", "closing_card", "year_stamp"):
            continue

        if segment_effective_duration(item) <= 0.0:
            failures.append(f"{label}: no measurable duration")

        audio_mode = item.get("audio_mode") or ""
        audio_path = item.get("audio_path") or item.get("voiceover_path")
        if audio_mode in ("ai_voiceover", "external") and not skip_tts:
            if not audio_path:
                failures.append(f"{label}: audio_mode={audio_mode} but no audio_path")
            elif not Path(audio_path).exists():
                failures.append(f"{label}: voiceover audio missing at {audio_path!r}")

        if t == "narration_broll" and not item.get("source_video"):
            vis = item.get("render_visual_path") or item.get("visual_path") or item.get("asset_path")
            if not vis:
                failures.append(f"{label}: no archive footage and no generated visual")
            elif not Path(vis).exists():
                failures.append(f"{label}: generated visual missing at {vis!r}")

        if t == "speaker_clip":
            if not item.get("source_video") and not item.get("fragment_id"):
                failures.append(f"{label}: speaker_clip without source_video/fragment_id")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate refined cinematic assets from latest AI4IA storyline.")
    parser.add_argument(
        "--storyline",
        default=None,
        help="Path to storyline JSON. Defaults to latest file in data/storylines."
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help="edge-tts voice name."
    )
    parser.add_argument(
        "--rate",
        default=DEFAULT_RATE,
        help="TTS rate, for example -6%% or +0%%."
    )
    parser.add_argument(
        "--volume",
        default=DEFAULT_VOLUME,
        help="TTS volume, for example +0%%."
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Generate cards and manifest only, skip voice-over audio."
    )

    args = parser.parse_args()

    storyline_path = Path(args.storyline) if args.storyline else latest_storyline_path()
    storyline_path = storyline_path.resolve()

    log("Starting Step 08: Generate Refined Cinematic Assets")
    log(f"Storyline: {storyline_path}")

    payload = load_storyline(storyline_path)
    story = payload["storyline"]
    story["selected_candidates"] = payload.get("selected_candidates") or []
    prompt_profile = payload.get("prompt_profile") or payload.get("prompt_intent") or {}

    # Phase 4V: if the storyline was validated by 07b, use final_timeline directly.
    # final_timeline lives at top level of the JSON (and also inside story after 07b).
    _validated_final_timeline = (
        payload.get("final_timeline")
        or story.get("final_timeline")
        or (payload.get("validated_narrative") or {}).get("final_timeline")
        or (story.get("validated_narrative") or {}).get("final_timeline")
    )
    _is_validated_storyline = bool(_validated_final_timeline) and payload.get("llm_validated", False)

    if _is_validated_storyline:
        log("Validated storyline detected — using final_timeline from 07b.")
        story["export_timeline"] = _validated_final_timeline
    elif not (story.get("export_timeline") or []) and payload.get("selected_candidates"):
        story["export_timeline"] = build_export_timeline_from_candidates(
            selected_candidates=story["selected_candidates"],
            prompt_profile=prompt_profile,
            target_duration_seconds=int(story.get("target_duration_seconds") or payload.get("target_duration_seconds") or 60),
            title=story.get("title"),
        )

    # Phase 4V: for validated storylines, override title/logline from validated_narrative.
    _validated_narrative = payload.get("validated_narrative") or story.get("validated_narrative") or {}
    if _is_validated_storyline and _validated_narrative.get("title"):
        story["title"] = _validated_narrative["title"]
    if _is_validated_storyline and _validated_narrative.get("logline"):
        story["logline"] = _validated_narrative["logline"]

    title = make_display_title(story.get("title", "AI4IA Documentary"))
    logline = sanitize_visible_text(story.get("logline", ""))
    internal_style = story.get("style", payload.get("style", ""))

    storyline_id = storyline_path.stem
    asset_root = GENERATED_ASSETS_DIR / storyline_id
    cards_dir = asset_root / "cards"
    audio_dir = asset_root / "audio"
    section_audio_dir = audio_dir / "sections"

    asset_root.mkdir(parents=True, exist_ok=True)
    cards_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    section_audio_dir.mkdir(parents=True, exist_ok=True)

    if _is_validated_storyline:
        # Phase 4V: extract voiceover script from narration_broll items in final_timeline.
        voiceover_script = []
        for idx, item in enumerate(_validated_final_timeline or []):
            t = item.get("type") or item.get("segment_type") or ""
            if t == "narration_broll":
                vo_text = item.get("voiceover_text") or item.get("subtitle_text") or ""
                sid = item.get("section_id") or f"S{idx}"
                if vo_text:
                    voiceover_script.append({
                        "section_id": sid,
                        "section_type": t,
                        "text": vo_text,
                        "_timeline_index": idx,
                    })
        log(f"Validated voiceover script: {len(voiceover_script)} narration_broll items")
    else:
        voiceover_script = build_voiceover_script(story)
    full_voiceover_text = voiceover_text_for_tts(voiceover_script)

    voiceover_script_path = asset_root / "voiceover_script.txt"
    voiceover_text_path = asset_root / "voiceover_full_text.txt"

    write_voiceover_script_txt(
        output_path=voiceover_script_path,
        title=title,
        voiceover_script=voiceover_script,
    )

    voiceover_text_path.write_text(full_voiceover_text, encoding="utf-8")

    log(f"Voice-over script written: {voiceover_script_path}")
    log(f"Voice-over full text written: {voiceover_text_path}")

    title_card_path = cards_dir / "title_card.png"
    create_title_card(
        output_path=title_card_path,
        title=title,
        logline=logline,
    )
    log(f"Title card generated: {title_card_path}")

    closing_card_path = cards_dir / "closing_card.png"
    _topic_terms_close = list(prompt_profile.get("topic_terms") or [])
    _requested_speaker_close = prompt_profile.get("requested_speaker")
    _primary_topic_close = _topic_terms_close[0] if _topic_terms_close else None
    _topic_label_close = " ".join(_topic_terms_close[:2]).title() if _topic_terms_close else None

    _cinematic_closings = {
        "healthcare": ("The Human Measure of Medicine", "AI in healthcare will be judged not by speed, but by whether it makes care safer and fairer."),
        "health": ("The Human Measure of Medicine", "AI in healthcare will be judged not by speed, but by whether it makes care safer and fairer."),
        "women": ("The Measure of Progress", "Progress means women gain real agency, real voice, and real power."),
        "empowerment": ("The Measure of Progress", "Progress means women gain real agency, real voice, and real power."),
        "governance": ("The Shape of Accountability", "AI governance will be measured by who it protects and who it answers to."),
        "youth": ("What They Inherit", "The next generation will decide whether to accept or change the systems they are given."),
        "education": ("The Promise of Learning", "The future of learning is defined by whether every student can access knowledge that is true, useful, and theirs."),
        "crypto": ("The Digital Promise", "A more open financial system will only arrive through transparency, safety, and rules that protect people."),
        "accessibility": ("The Invisible Divide", "Access to information is not a given — it must be built, maintained, and defended."),
    }

    _storyline_closing = story.get("closing_meaning", "")

    if _requested_speaker_close:
        closing_headline = f"{_requested_speaker_close}: The Contribution"
        closing_subtitle = _storyline_closing or f"A documentary on {_requested_speaker_close}'s contributions to AI4IA."
    elif _primary_topic_close and _primary_topic_close in _cinematic_closings:
        closing_headline, closing_subtitle = _cinematic_closings[_primary_topic_close]
        if _storyline_closing:
            closing_subtitle = _storyline_closing
    elif _topic_label_close:
        closing_headline = f"AI and {_topic_label_close}"
        closing_subtitle = _storyline_closing or f"A documentary on AI and {_topic_label_close}."
    else:
        closing_headline = make_display_title(title)
        closing_subtitle = _storyline_closing or "A documentary from the AI4IA conference archive."
    create_closing_card(
        output_path=closing_card_path,
        title=title,
        closing_headline=closing_headline,
        closing_subtitle=closing_subtitle,
    )
    log(f"Closing card generated: {closing_card_path}")

    section_cards = []

    for section in story.get("sections", []):
        section_id = section.get("section_id", "S")
        section_type = section.get("section_type", "section")

        copy = infer_section_copy(section, prompt_profile=prompt_profile)

        section_card_path = cards_dir / f"{section_id}_{slugify(section_type)}.png"

        create_section_card(
            output_path=section_card_path,
            section_id=section_id,
            section_type=section_type,
            headline=copy["headline"],
            subtitle=copy["subtitle"],
            eyebrow=copy["eyebrow"],
        )

        section_cards.append(
            {
                "section_id": section_id,
                "section_type": section_type,
                "path": str(section_card_path),
                "eyebrow": copy["eyebrow"],
                "headline": copy["headline"],
                "subtitle": copy["subtitle"],
            }
        )

        log(f"Section card generated: {section_card_path}")

    year_stamp_cards = []
    _seen_year_stamp_years: set[int] = set()
    for item in story.get("export_timeline") or []:
        if item.get("type") == "year_stamp":
            year_val = item.get("year")
            if year_val is not None and year_val not in _seen_year_stamp_years:
                _seen_year_stamp_years.add(year_val)
                year_stamp_path = cards_dir / f"year_stamp_{year_val}.png"
                try:
                    create_year_stamp_card(year=year_val, output_path=year_stamp_path)
                    year_stamp_cards.append({"year": year_val, "path": str(year_stamp_path)})
                    log(f"Year stamp card generated: {year_stamp_path}")
                except Exception as exc:
                    log(f"WARNING: failed to render year_stamp card for {year_val}: {exc}")

    tts_status = "skipped"
    section_voiceovers = []

    if not args.skip_tts:
        if not full_voiceover_text:
            log("No voice-over text found. Skipping TTS.")
            tts_status = "no_text"
        else:
            section_jobs = []
            for item in voiceover_script:
                section_id = item.get("section_id")
                section_type = item.get("section_type", "section")
                text = item.get("text", "").strip()
                if not section_id or not text:
                    continue
                section_audio_path = section_audio_dir / f"{section_id}_{slugify(section_type)}_voiceover.mp3"
                section_jobs.append((section_id, section_type, text, section_audio_path))

            log(f"Generating {len(section_jobs)} section voice-over(s) concurrently with edge-tts voice: {args.voice}")

            async def _tts_with_log(label: str, text: str, output_path: Path) -> None:
                t0 = time.time()
                await generate_tts(
                    text=text,
                    output_path=output_path,
                    voice=args.voice,
                    rate=args.rate,
                    volume=args.volume,
                )
                log(f"  [tts] {label} done in {time.time() - t0:.1f}s")

            async def _generate_all() -> None:
                await asyncio.gather(
                    *(
                        _tts_with_log(section_id, text, section_audio_path)
                        for section_id, _, text, section_audio_path in section_jobs
                    )
                )

            _t0 = time.time()
            asyncio.run(_generate_all())
            log(f"All TTS calls finished in {time.time() - _t0:.1f}s wall-clock.")

            for section_id, section_type, text, section_audio_path in section_jobs:
                section_voiceovers.append(
                    {
                        "section_id": section_id,
                        "section_type": section_type,
                        "text": text,
                        "audio_path": str(section_audio_path),
                        "provider": "edge-tts",
                        "voice": args.voice,
                        "rate": args.rate,
                        "volume": args.volume,
                    }
                )

            tts_status = "generated"
            log(f"Section voice-overs generated: {len(section_voiceovers)}")

    fragment_lookup = load_classified_fragment_lookup()
    metadata_stats = {
        "enriched_fragment_refs": 0,
        "missing_fragment_refs": set(),
    }
    enriched_story = enrich_fragment_refs(story, fragment_lookup, metadata_stats)
    enriched_story["selected_candidates"] = story.get("selected_candidates") or []

    if not (enriched_story.get("export_timeline") or []) and payload.get("selected_candidates"):
        enriched_story["export_timeline"] = build_export_timeline_from_candidates(
            selected_candidates=enriched_story["selected_candidates"],
            prompt_profile=payload.get("prompt_profile") or payload.get("prompt_intent") or {},
            target_duration_seconds=int(enriched_story.get("target_duration_seconds") or payload.get("target_duration_seconds") or 60),
            title=enriched_story.get("title"),
        )

    if _is_validated_storyline:
        # Phase 4V: validated storyline — final_timeline already has correct types.
        # Attach audio_path from section_voiceovers to narration_broll items (by order).
        nb_idx = 0
        final_tl = enriched_story.get("export_timeline") or []
        for item in final_tl:
            t = item.get("type") or item.get("segment_type") or ""
            if t == "narration_broll" and nb_idx < len(section_voiceovers):
                sv = section_voiceovers[nb_idx]
                item["audio_path"] = sv.get("audio_path", "")
                if not item.get("subtitle_text"):
                    item["subtitle_text"] = sv.get("text", "")
                if not item.get("voiceover_text"):
                    item["voiceover_text"] = sv.get("text", "")
                item["subtitle_kind"] = "voiceover"
                reconcile_narration_duration(item, item["audio_path"])
                nb_idx += 1
            elif t == "speaker_clip" and not item.get("render_duration"):
                # Locked validated window: render exactly what validation approved.
                s_v, e_v = item.get("render_start", item.get("start")), item.get("render_end", item.get("end"))
                try:
                    if s_v is not None and e_v is not None and float(e_v) > float(s_v):
                        item["render_duration"] = round(float(e_v) - float(s_v), 3)
                except (TypeError, ValueError):
                    pass
        enriched_story["export_timeline"] = final_tl
        log(f"Phase 4V: {nb_idx} narration_broll items received audio_path from section_voiceovers.")

        # Phase 4V+: generate topic-specific PIL visual assets for narration_broll items
        visuals_dir = asset_root / "visuals"
        final_tl = generate_visual_assets_for_timeline(final_tl, visuals_dir)
        enriched_story["export_timeline"] = final_tl
        _gen_vis_count = sum(1 for x in final_tl if x.get("render_visual_path"))
        log(f"Phase 4V+: {_gen_vis_count} narration_broll visual assets generated in {visuals_dir}")
    else:
        # Phase 4O: enrich export_timeline with moment fields from the storyline
        # (without overwriting fields already present)
        storyline_timeline = payload.get("export_timeline") or enriched_story.get("export_timeline") or []
        if enriched_story.get("export_timeline"):
            enriched_story["export_timeline"] = enrich_timeline_with_moment_fields(
                export_timeline=enriched_story["export_timeline"],
                storyline_timeline=storyline_timeline,
            )

        # Phase 4S: build hybrid timeline — select best fragments as speaker_clip/source_audio,
        # convert the rest to narration_broll/ai_voiceover using available section voiceovers.
        if enriched_story.get("export_timeline"):
            vo_paths = [sv.get("audio_path", "") for sv in section_voiceovers if sv.get("audio_path")]
            # Build audio_path → voiceover text map for narration_broll subtitle coverage
            vo_path_to_text: dict[str, str] = {
                sv["audio_path"]: sv.get("text", "")
                for sv in section_voiceovers
                if sv.get("audio_path") and sv.get("text")
            }
            enriched_story["export_timeline"] = build_hybrid_export_timeline(
                export_timeline=enriched_story["export_timeline"],
                selected_candidates=enriched_story.get("selected_candidates") or payload.get("selected_candidates") or [],
                section_voiceover_paths=vo_paths,
                fragment_lookup=fragment_lookup,
                vo_path_to_text=vo_path_to_text,
            )

    archive_fragments = collect_archive_fragments(enriched_story)
    generated_assets = collect_generated_assets(enriched_story)
    internet_assets = collect_internet_assets(enriched_story)

    manifest = {
        "created_at": datetime.utcnow().isoformat(),
        "storyline_path": str(storyline_path),
        "storyline_id": storyline_id,
        "asset_root": str(asset_root),
        "title": title,
        "logline": logline,
        "internal_style_direction": internal_style,
        "visible_brand_label": VISIBLE_BRAND_LABEL,
        "visible_series_label": VISIBLE_SERIES_LABEL,
        "target_duration_seconds": story.get("target_duration_seconds"),
        "estimated_duration_seconds": story.get("estimated_duration_seconds"),
        "storyline": enriched_story,
        "director_plan": enriched_story.get("director_plan"),
        "planning_summary": enriched_story.get("planning_summary"),
        "visual_rules": {
            "do_not_show_prompt": True,
            "do_not_show_internal_style": True,
            "do_not_show_long_narration_on_cards": True,
            "cards_use_short_cinematic_copy": True,
            "section_voiceover_only": True,
        },
        "voice": {
            "provider": "edge-tts",
            "voice": args.voice,
            "rate": args.rate,
            "volume": args.volume,
            "status": tts_status,
            "script_path": str(voiceover_script_path),
            "full_text_path": str(voiceover_text_path),
            "audio_path": None,
            "section_voiceovers": section_voiceovers,
            "important_note": "Use section_voiceovers for final export."
        },
        "cards": {
            "title_card": str(title_card_path),
            "closing_card": str(closing_card_path),
            "section_cards": section_cards,
            "year_stamp_cards": year_stamp_cards,
        },
        "archive_fragments": archive_fragments,
        "generated_assets_requested_by_storyline": generated_assets,
        "internet_assets_requested_by_storyline": internet_assets,
        "metadata_enrichment": {
            "enabled": True,
            "fragment_lookup_size": len(fragment_lookup),
            "enriched_fragment_refs": metadata_stats["enriched_fragment_refs"],
            "missing_fragment_refs": sorted(metadata_stats["missing_fragment_refs"]),
            "enriched_fields": [
                "source_video",
                "source_video_file",
                "video_id",
                "video_title",
                "speaker",
                "speaker_attribution",
                "text",
                "summary",
                "keywords",
                "primary_theme",
                "primary_chapter",
                "theme_tags",
                "chapter_tags",
                "clip_usage_candidates",
                "quote_strength",
                "clarity_score",
                "narration_support_score",
            ],
        },
        "next_step": {
            "script": "scripts/09_export_video.py",
            "purpose": "Create refined cinematic export with zoom, fades, section voice-over, speaker audio, and compact subtitles."
        }
    }

    # Visual quality gates (validated storylines only)
    if _is_validated_storyline:
        _tl_for_vqg = enriched_story.get("export_timeline") or []
        _nb_items = [x for x in _tl_for_vqg if (x.get("type") or x.get("segment_type")) == "narration_broll"]
        _gen_vis = [x for x in _nb_items if x.get("render_visual_path")]
        _muted_arch = [x for x in _nb_items if not x.get("render_visual_path") and not x.get("visual_path")]
        manifest["visual_strategy"] = (
            "speaker_clip = source video + source audio. "
            "narration_broll = generated topic visual + AI voiceover."
        )
        manifest["visual_quality_gates"] = {
            "all_narration_broll_have_generated_visual": len(_gen_vis) == len(_nb_items) and len(_nb_items) > 0,
            "generated_visual_segments": len(_gen_vis),
            "muted_archive_narration_segments": len(_muted_arch),
            "total_narration_broll": len(_nb_items),
        }

    # Write hybrid timeline to all downstream-compatible keys
    hybrid_tl = enriched_story.get("export_timeline") or []
    manifest["export_timeline"] = hybrid_tl
    manifest["timeline"] = hybrid_tl
    manifest["duration_budget"] = build_duration_budget(
        hybrid_tl,
        manifest.get("target_duration_seconds"),
        stage="08_generate_assets",
    )

    # Pre-flight audit: every timeline item must be renderable before Step 09 runs.
    # NOTE: this guards RENDERABILITY (missing media, unusable segments), not
    # duration. A below-target plan is content exhaustion — warn and ship the
    # honest best-effort duration; never a fatal preflight failure. (Integrity
    # shortfalls, i.e. rendered < planned, are still hard-failed in Step 09.)
    preflight_failures = preflight_audit_manifest(manifest, skip_tts=args.skip_tts)
    manifest["preflight_failures"] = preflight_failures

    _budget = manifest["duration_budget"]
    if _budget.get("meets_floor") is False:
        log(
            f"WARNING: planned timeline {_budget['planned_total_seconds']:.1f}s is below target "
            f"{_budget['target_seconds']:.1f}s (shortfall {_budget['shortfall_seconds']:.1f}s) — "
            f"relevant content exhausted, shipping honest best-effort duration, no filler."
        )

    # Strict asset manifest validation before saving
    manifest_validation_issues = validate_asset_manifest_before_save(
        manifest=manifest,
        prompt_profile=prompt_profile,
        selected_candidates=story.get("selected_candidates") or [],
    )
    if manifest_validation_issues:
        log("Asset manifest validation issues:")
        for issue in manifest_validation_issues:
            log(f"  - {issue}")
    existing_manifest_issues = manifest.get("validation_issues") or []
    manifest["validation_issues"] = existing_manifest_issues + manifest_validation_issues

    manifest_path = asset_root / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    latest_manifest_path = GENERATED_ASSETS_DIR / "latest_asset_manifest.json"
    latest_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Hybrid timeline diagnostics
    sc_count = sum(1 for x in hybrid_tl if (x.get("type") or x.get("segment_type")) == "speaker_clip")
    nb_count = sum(1 for x in hybrid_tl if (x.get("type") or x.get("segment_type")) == "narration_broll")
    av_count = sum(1 for x in hybrid_tl if (x.get("type") or x.get("segment_type")) == "archive_visual")
    card_tl_count = sum(1 for x in hybrid_tl if (x.get("type") or x.get("segment_type")) in ("title_card", "closing_card"))
    log("")
    log("Final hybrid timeline summary:")
    log(f"  total items:                {len(hybrid_tl)}")
    log(f"  speaker_clip/source_audio:  {sc_count}")
    log(f"  narration_broll/ai_voiceover: {nb_count}")
    log(f"  archive_visual/silent:      {av_count}")
    log(f"  cards:                      {card_tl_count}")

    log("")
    log("Refined asset generation complete")
    log(f"Asset root: {asset_root}")
    log(f"Manifest: {manifest_path}")
    log(f"Latest manifest: {latest_manifest_path}")
    log(f"Voice-over status: {tts_status}")
    log(f"Section voice-overs: {len(section_voiceovers)}")
    log(f"Archive fragments listed: {len(archive_fragments)}")
    log(f"Metadata enrichment: {metadata_stats['enriched_fragment_refs']} refs enriched, {len(metadata_stats['missing_fragment_refs'])} missing")
    log(f"Section cards generated: {len(section_cards)}")
    log("Visible card text is now sanitized. Internal style/prompt text will not appear on cards.")

    if preflight_failures:
        log("")
        log(f"PRE-FLIGHT AUDIT FAILED — {len(preflight_failures)} issue(s), manifest is NOT render-safe:")
        for failure in preflight_failures:
            log(f"  ✗ {failure}")
        sys.exit(1)


if __name__ == "__main__":
    main()
