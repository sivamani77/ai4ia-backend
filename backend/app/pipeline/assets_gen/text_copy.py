from __future__ import annotations
import re
from typing import Any

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


def detect_generic_card_title(title: str) -> bool:
    return title.lower().strip() in GENERIC_CARD_TITLES


def detect_generic_voiceover(text: str) -> bool:
    if not text:
        return False
    normalized = text.lower()
    return any(phrase in normalized for phrase in DISALLOWED_VOICEOVER_PHRASES)


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
