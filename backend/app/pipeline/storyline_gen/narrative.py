import re

from app.pipeline.timeline_budget import timeline_segment_count_for_duration
from app.pipeline.storyline_gen.text_utils import truncate_text


GENERIC_FALLBACK_PHRASES = [
    "artificial intelligence is no longer a distant idea",
    "shaping how people learn, communicate, govern and access knowledge",
    "across the ai4ia archive",
    "across the ai for ia archive",
    "the promise of ai is powerful",
    "the risks are equally real",
    "inclusive design, local data, ethical governance",
    "technology around real needs",
    "ai can become a bridge to knowledge",
    "how ai can expand access to information without deepening existing divides",
    "exclusion, weak infrastructure, language gaps, bias and loss of trust",
    "governance, rights, media literacy, accessibility, local knowledge",
    "this video explores",
    "through evidence from the ai4ia conference archive",
    "the selected clips address",
    "these archive fragments provide evidence-based context",
    "this documentary draws exclusively on confirmed archive clips",
    "the archive evidence shows:",
    "archive fragments provide",
    "draws exclusively on confirmed archive",
]

DISALLOWED_FALLBACK_TYPES = {
    "generic_ai4ia_documentary",
    "generic_ai_transformation_intro",
    "generic_governance_rights_accessibility_outro",
    "generic_archive_summary",
}

PROMPT_TERM_REQUIRED_WORDS: dict[str, set[str]] = {
    "healthcare": {"health", "healthcare", "medicine", "medical", "patient", "doctor", "clinical", "hospital"},
    "health": {"health", "healthcare", "medicine", "medical", "patient", "doctor", "clinical"},
    "women": {"women", "woman", "female", "gender", "girls", "empowerment", "agency", "representation"},
    "empowerment": {"women", "woman", "female", "gender", "empowerment", "agency", "leadership"},
    "crypto": {"crypto", "cryptocurrency", "blockchain", "web3", "digital currency"},
    "cryptocurrency": {"crypto", "cryptocurrency", "blockchain", "web3"},
}

NATIONAL_GEOGRAPHIC_STYLE_GUIDE = {
    "voice": "observational, human, cinematic, clear",
    "sentence_length": "medium",
    "avoid": [
        "technical backend terms",
        "generic AI hype",
        "claims not supported by clips",
        "bullet-like narration",
        "conference admin language",
    ],
    "prefer": [
        "human stakes",
        "place/context",
        "tension",
        "careful synthesis",
        "specific evidence from speakers",
        "clear closing insight",
    ],
}

CINEMATIC_HOOKS: dict[str, str] = {
    "healthcare": "In hospitals, clinics, and homes, artificial intelligence is beginning to reshape the way care is delivered.",
    "health": "In hospitals, clinics, and homes, artificial intelligence is beginning to reshape the way care is delivered.",
    "women": "Across the world, millions of women are navigating systems that were not designed with them in mind.",
    "empowerment": "Across the world, millions of women are navigating systems that were not designed with them in mind.",
    "education": "In classrooms, libraries, and remote communities, the question is not whether AI will change learning, but whether it will do so fairly.",
    "governance": "As artificial intelligence moves into public systems, the question of who controls it — and who answers for it — grows more urgent.",
    "youth": "A generation that has grown up online is now asking a different question: not what technology can do, but who it serves.",
    "crypto": "Digital currencies and blockchain technology have promised a new financial order. But for many, the risks have arrived before the benefits.",
    "accessibility": "For millions of people, access to information is not a given. It is a daily negotiation with systems that were built without them in mind.",
}

CINEMATIC_TENSIONS: dict[str, str] = {
    "healthcare": "But the promise of AI in healthcare depends on more than speed. Speakers warn that trust, patient safety, ethical use, and regulation must shape how these systems enter clinical life.",
    "health": "But the promise of AI in healthcare depends on more than speed. Speakers warn that trust, patient safety, ethical use, and regulation must shape how these systems enter clinical life.",
    "women": "But access to tools is not the same as access to power. For many women, technology arrives within systems that still constrain their choices and reduce their voices.",
    "empowerment": "But access to tools is not the same as access to power. For many women, technology arrives within systems that still constrain their choices and reduce their voices.",
    "governance": "The question is not only what AI can do. It is who writes the rules, who bears the risk, and who can challenge a decision made by a machine.",
    "youth": "But younger generations do not simply inherit the future. They are being shaped by algorithmic systems whose decisions they cannot see and rarely control.",
    "education": "But digital tools also replicate the inequalities they were meant to solve, reinforcing divides of language, access, and the quality of what people find.",
    "crypto": "The risks have been real. Volatility, fraud, and the absence of consumer protection have left many people worse off than before.",
    "accessibility": "Yet the technology that could expand access is often designed for those who need it least, leaving the most vulnerable populations without tools that match their realities.",
}

CINEMATIC_CLOSINGS: dict[str, str] = {
    "healthcare": "The future of AI in healthcare will not be measured only by what machines can predict, but by whether they help people receive safer, fairer, and more human care.",
    "health": "The future of AI in healthcare will not be measured only by what machines can predict, but by whether they help people receive safer, fairer, and more human care.",
    "women": "The measure of progress will not be the number of tools deployed, but whether women gain real agency, real voice, and real power in the systems that shape their lives.",
    "empowerment": "The measure of progress will not be the number of tools deployed, but whether women gain real agency, real voice, and real power in the systems that shape their lives.",
    "governance": "The future of AI governance will not be written by technology alone. It will be shaped by institutions, civil society, and people who insist that accountability must be built in from the start.",
    "youth": "The generation growing up with AI will not simply use the systems they inherit. They will decide whether to change them.",
    "education": "The future of learning is not defined by the speed of information, but by whether every learner — regardless of where they were born — can access knowledge that is true, useful, and theirs.",
    "crypto": "The promise of a more open financial system remains. But it will only arrive through transparency, safety, and rules that protect the people it was meant to serve.",
    "accessibility": "The future of information access will not be determined by what technology can do, but by whether it is built to serve the people who have been left out the longest.",
}


def build_documentary_arc(
    prompt_profile: dict,
    selected_candidates: list[dict],
    target_duration_seconds: int,
) -> dict:
    """Build a National Geographic-style documentary arc from prompt and evidence."""
    topic_terms = prompt_profile.get("topic_terms") or []
    expanded_terms = prompt_profile.get("expanded_terms") or []
    requested_speaker = prompt_profile.get("requested_speaker")

    primary_topic = topic_terms[0] if topic_terms else None
    topic_label = " ".join(topic_terms[:2]).title() if topic_terms else (requested_speaker or "Artificial Intelligence")

    hook = CINEMATIC_HOOKS.get(primary_topic, f"Across the world, conversations about artificial intelligence are raising a central question: what does it mean for technology to serve people?")
    tension = CINEMATIC_TENSIONS.get(primary_topic, "But the promise of AI is not automatic. Every benefit comes with a risk, and every advance leaves questions about who is left behind.")
    closing = CINEMATIC_CLOSINGS.get(primary_topic, f"The measure of AI will not be the speed of its systems, but whether the people it touches are better served, better heard, and more free.")

    if requested_speaker:
        central_question = f"What does {requested_speaker} contribute to our understanding of AI and its human stakes?"
        narrative_thesis = f"Through {requested_speaker}'s archive contributions, this documentary traces key questions in AI and society."
    elif primary_topic:
        central_questions = {
            "healthcare": "How can AI support doctors, patients, and families without weakening trust, safety, or human judgment?",
            "health": "How can AI support doctors, patients, and families without weakening trust, safety, or human judgment?",
            "women": "How can AI expand opportunity for women without reproducing the systems of exclusion it was meant to overcome?",
            "empowerment": "How can AI expand opportunity for women without reproducing the systems of exclusion it was meant to overcome?",
            "governance": "Who writes the rules for AI, who bears the risk, and how can accountability be built into the systems before harm arrives?",
            "youth": "What future are young people inheriting from AI systems, and do they have a say in how those systems are built?",
            "education": "Can AI help close the gap between those who have access to knowledge and those who have been shut out?",
            "crypto": "Can digital currencies deliver a more open financial system, or will the risks arrive before the benefits?",
            "accessibility": "Can AI become a genuine tool for access, or will it deepen the divides it was meant to bridge?",
        }
        central_question = central_questions.get(primary_topic, f"How is AI changing {topic_label}, and who decides whether that change is good?")
        narrative_thesis = f"This documentary draws on evidence from AI4IA speakers to examine the real promise and real risks of AI in {topic_label}."
    else:
        central_question = "How is AI changing the world, and what must change to ensure it serves everyone?"
        narrative_thesis = "This documentary draws on AI4IA conference evidence to examine AI's human stakes."

    fragment_ids_all = [c.get("fragment_id") for c in selected_candidates if c.get("fragment_id")]
    n_candidates = len(selected_candidates)
    sections_target = timeline_segment_count_for_duration(target_duration_seconds)
    chunk_size = max(1, n_candidates // max(1, sections_target - 2))

    sections = []

    hook_ids = fragment_ids_all[:2]
    sections.append({
        "role": "hook",
        "title": f"AI and {topic_label}" if not requested_speaker else f"{requested_speaker}: The Questions",
        "viewer_narration": hook,
        "evidence_fragment_ids": hook_ids,
    })

    context_candidates = selected_candidates[:chunk_size]
    context_ids = [c.get("fragment_id") for c in context_candidates if c.get("fragment_id")]
    context_texts = [truncate_text(c.get("text") or c.get("summary") or "", 200) for c in context_candidates if (c.get("text") or c.get("summary"))]

    if context_texts:
        context_narration = _synthesize_documentary_context(primary_topic, context_texts, topic_label)
    else:
        context_narration = f"Across the AI4IA conversations, {topic_label} emerges as a field where the stakes — for individuals, institutions, and communities — could not be higher."

    sections.append({
        "role": "human_context",
        "title": f"The Human Stakes of {topic_label}" if not requested_speaker else f"{requested_speaker}: Context",
        "viewer_narration": context_narration,
        "evidence_fragment_ids": context_ids,
    })

    tension_candidates = selected_candidates[chunk_size: chunk_size * 2]
    tension_ids = [c.get("fragment_id") for c in tension_candidates if c.get("fragment_id")]
    tension_texts = [truncate_text(c.get("text") or c.get("summary") or "", 200) for c in tension_candidates if (c.get("text") or c.get("summary"))]

    if tension_texts:
        tension_narration = _synthesize_documentary_tension(primary_topic, tension_texts, topic_label)
    else:
        tension_narration = tension

    sections.append({
        "role": "tension",
        "title": f"The Challenge" if not requested_speaker else f"{requested_speaker}: Tensions",
        "viewer_narration": tension_narration,
        "evidence_fragment_ids": tension_ids,
    })

    if sections_target >= 5 and n_candidates >= chunk_size * 3:
        expert_candidates = selected_candidates[chunk_size * 2: chunk_size * 3]
        expert_ids = [c.get("fragment_id") for c in expert_candidates if c.get("fragment_id")]
        expert_texts = [truncate_text(c.get("text") or c.get("summary") or "", 200) for c in expert_candidates if (c.get("text") or c.get("summary"))]
        expert_narration = _synthesize_documentary_evidence(primary_topic, expert_texts, topic_label)
        sections.append({
            "role": "expert_evidence",
            "title": "What the Speakers Say",
            "viewer_narration": expert_narration,
            "evidence_fragment_ids": expert_ids,
        })

    synthesis_ids = fragment_ids_all[-2:]
    sections.append({
        "role": "synthesis",
        "title": f"What It Means" if not requested_speaker else f"{requested_speaker}: The Meaning",
        "viewer_narration": closing,
        "evidence_fragment_ids": synthesis_ids,
    })

    # Expand narration to fill target duration.
    # Each section gets roughly (target - 10s cards) / num_sections worth of content.
    # At ~2.5 words/sec TTS pace, compute the target word count per section.
    _section_budget_s = max(8, (target_duration_seconds - 10) // max(1, len(sections)))
    _target_words = max(20, int(_section_budget_s * 2.5))

    # Build a flat lookup: fragment_id -> evidence text
    _frag_text: dict[str, str] = {}
    for c in selected_candidates:
        fid = c.get("fragment_id")
        txt = (c.get("text") or c.get("summary") or "").strip()
        if fid and txt:
            _frag_text[fid] = txt

    for sec in sections:
        narration = sec.get("viewer_narration", "")
        current_words = len(narration.split())
        if current_words >= _target_words:
            continue
        # Extend using evidence fragment texts for this section
        evidence_ids = sec.get("evidence_fragment_ids") or []
        extra_sentences = []
        for fid in evidence_ids:
            if current_words + len(" ".join(extra_sentences).split()) >= _target_words:
                break
            ftext = _frag_text.get(fid, "").strip()
            if not ftext:
                continue
            # Use the first 2 sentences of fragment text as supporting narration
            raw_sentences = [s.strip() for s in ftext.replace("\n", " ").split(".") if len(s.strip()) > 20]
            for sent in raw_sentences[:2]:
                extra_sentences.append(sent + ".")
                if current_words + len(" ".join(extra_sentences).split()) >= _target_words:
                    break
        if extra_sentences:
            sec["viewer_narration"] = narration.rstrip() + " " + " ".join(extra_sentences)

    return {
        "documentary_style": "national_geographic",
        "central_question": central_question,
        "narrative_thesis": narrative_thesis,
        "closing_meaning": closing,
        "sections": sections,
    }


def _synthesize_documentary_context(topic: str | None, texts: list[str], topic_label: str) -> str:
    """Build human-context narration from evidence texts."""
    combined = " ".join(texts[:3])
    combined_lower = combined.lower()

    if topic in ("healthcare", "health"):
        if any(w in combined_lower for w in ("digital", "online", "home", "monitor")):
            return "In everyday care, digital tools are already changing how people contact doctors, monitor health, and remain independent at home."
        if any(w in combined_lower for w in ("trust", "patient", "safety")):
            return "At the centre of these conversations is a question of trust. Patients, doctors, and families are asking whether AI systems can be relied upon when the stakes are a person's health."
        return f"Across the AI4IA conversations, {topic_label} emerges as one of the most contested frontiers of artificial intelligence — where the promise of better outcomes meets the reality of fragile systems and uneven access."

    if topic in ("women", "empowerment"):
        if any(w in combined_lower for w in ("agency", "voice", "decision")):
            return "For many women, the question is not just whether they can use AI tools, but whether they have a voice in how those tools are built and governed."
        return f"Across regions and sectors, women are engaging with AI on terms that are often not their own. The systems, the data, and the decisions often reflect the world as it has been — not as it should be."

    if topic == "governance":
        return "Governance of AI is not a technical matter alone. It involves institutions, rights, accountability, and the political will to hold powerful systems to account."

    if topic == "youth":
        return "Young people are not passive recipients of the AI era. They are users, critics, and — increasingly — the people who must live longest with the consequences of decisions made today."

    return f"Across the AI4IA conversations, {topic_label} is presented not as an abstraction, but as a challenge with concrete human consequences — for people who depend on these systems to work, and work fairly."


def _synthesize_documentary_tension(topic: str | None, texts: list[str], topic_label: str) -> str:
    """Build tension narration from evidence texts."""
    combined = " ".join(texts[:3]).lower()

    if topic in ("healthcare", "health"):
        if any(w in combined for w in ("replace", "doctor", "judgment")):
            return "The question is not whether machines will replace doctors. It is whether doctors, patients, and AI systems can work together in ways that keep human judgment at the center of care."
        if any(w in combined for w in ("waste", "overload", "verification", "staff")):
            return "Beyond diagnosis, AI also points to quieter problems inside healthcare systems — medicine waste, overloaded staff, and the need to make verification faster and safer."
        return "But the promise of AI in healthcare depends on more than speed. Speakers warn that trust, patient safety, ethical use, and regulation must shape how these systems enter clinical life."

    if topic in ("women", "empowerment"):
        return "But participation is not the same as power. Women can use digital tools and still find that the systems behind them — the data, the design, the decisions — were never made for them."

    if topic == "governance":
        return "Regulation is not the enemy of innovation. Speakers argue it is the condition for trust — and without trust, the systems that claim to help will eventually be rejected or misused."

    return f"But the arrival of AI in {topic_label} does not resolve the underlying tensions. It often sharpens them — exposing who has voice, who has access, and who bears the risks."


def _synthesize_documentary_evidence(topic: str | None, texts: list[str], topic_label: str) -> str:
    """Build expert-evidence narration from speaker evidence texts."""
    combined = " ".join(texts[:3]).lower()

    if topic in ("healthcare", "health"):
        if any(w in combined for w in ("regulation", "govern", "ethics", "ethic")):
            return "Speakers in the healthcare conversation return to a shared concern: without clear regulation, ethical frameworks, and patient consent, AI systems risk causing the very harms they were designed to prevent."
        return "Across these conversations, healthcare is not presented as a place where AI replaces people. It is presented as a field where trust, safety, and access decide whether the technology helps."

    if topic in ("women", "empowerment"):
        return "The speakers who discuss women and AI do not speak in abstractions. They describe specific moments where systems fail women — in hiring, in healthcare, in justice, in access to credit."

    if topic == "governance":
        return "Governance conversations in the AI4IA archive are marked by a shared urgency: the rules are being written now, and the people most affected by them are too often absent from the table."

    return f"Across these conversations, {topic_label} is not a distant aspiration. It is a field in motion — with real evidence, real failure, and real possibilities for change."


def _make_cinematic_title(topic_terms: list[str], topic_label: str, raw_prompt: str = "") -> str:
    """Generate a dynamic documentary title based on the user's prompt or topic."""
    if raw_prompt and len(raw_prompt.strip()) > 2:
        clean_p = raw_prompt.strip().rstrip(".!?")
        clean_p = re.sub(r"^(generate|create|build|make)\s+(a\s+)?(video|documentary)\s+(about|on|for|regarding)\s+", "", clean_p, flags=re.IGNORECASE)
        words = clean_p.split()
        if len(words) > 8:
            clean_p = " ".join(words[:8])
        return clean_p.title()
    if topic_label:
        return f"AI and {topic_label.title()}"
    return "Artificial Intelligence and Society"


def detect_generic_fallback_narration(text: str, prompt_profile: dict) -> list[str]:
    """Detect generic AI4IA fallback phrases in narration text."""
    issues = []
    if not text:
        return issues
    normalized = text.lower()
    for phrase in GENERIC_FALLBACK_PHRASES:
        if phrase in normalized:
            issues.append(f"generic_fallback_narration_detected: '{phrase[:60]}'")
    return issues


def validate_narration_prompt_alignment(
    narration_text: str,
    prompt_profile: dict,
    selected_candidates: list[dict],
) -> list[str]:
    """Check that narration includes at least one prompt-specific term."""
    issues = []
    if not narration_text:
        return issues

    normalized = narration_text.lower()
    topic_terms = set(prompt_profile.get("topic_terms") or [])
    requested_speaker = prompt_profile.get("requested_speaker")

    if requested_speaker:
        name_parts = requested_speaker.lower().split()
        if not any(part in normalized for part in name_parts if len(part) > 3):
            issues.append("narration_not_aligned_to_prompt: speaker name not mentioned in narration")
        return issues

    found_any = False
    for term in topic_terms:
        required = PROMPT_TERM_REQUIRED_WORDS.get(term, {term})
        if any(req in normalized for req in required):
            found_any = True
            break

    if not found_any:
        expanded = set(prompt_profile.get("expanded_terms") or [])
        if any(term.lower() in normalized for term in expanded):
            found_any = True

    if not found_any and topic_terms:
        issues.append(f"narration_not_aligned_to_prompt: none of {sorted(topic_terms)[:5]} found in narration")

    return issues


def build_evidence_grounded_narration(
    prompt_profile: dict,
    selected_candidates: list[dict],
    evidence_package: dict,
    target_duration_seconds: int,
) -> dict:
    """Build narration strictly from selected fragment evidence. No generic AI4IA templates."""
    topic_terms = prompt_profile.get("topic_terms") or []
    expanded_terms = prompt_profile.get("expanded_terms") or []
    requested_speaker = prompt_profile.get("requested_speaker")
    evidence_gaps = evidence_package.get("evidence_gaps") or []
    must_not_claim = evidence_package.get("must_not_claim") or []

    all_terms = list(dict.fromkeys(topic_terms + expanded_terms))
    topic_label = " ".join(topic_terms[:3]).title() if topic_terms else (requested_speaker or "AI")

    if not selected_candidates:
        gap_text = " ".join(evidence_gaps) if evidence_gaps else f"The archive does not contain enough direct evidence for this prompt."
        disclaimer = must_not_claim[0] if must_not_claim else ""
        full_text = gap_text + (" " + disclaimer if disclaimer else "")
        return {
            "generation_mode": "low_evidence_disclosure",
            "fallback_reason": "no_selected_candidates",
            "narration_text": full_text,
            "sections": [
                {
                    "section_id": "S1",
                    "title": "Evidence Gap Disclosure",
                    "narration": full_text,
                    "evidence_fragment_ids": [],
                    "source_videos": [],
                    "evidence_terms": [],
                    "claim_type": "evidence_gap",
                }
            ],
        }

    evidence_summaries = []
    for c in selected_candidates[:12]:
        text = (c.get("text") or c.get("summary") or "").strip()
        if text:
            evidence_summaries.append({
                "fragment_id": c.get("fragment_id"),
                "source_video": c.get("source_video"),
                "speaker": c.get("speaker"),
                "primary_theme": c.get("primary_theme"),
                "text": truncate_text(text, 300),
                "direct_terms": c.get("direct_evidence_terms") or [],
                "expanded_terms": c.get("expanded_evidence_terms") or [],
                "selection_reason": c.get("selection_reason", ""),
            })

    sections = []
    fragment_ids_used = [e["fragment_id"] for e in evidence_summaries if e["fragment_id"]]
    source_videos_used = list(dict.fromkeys(e["source_video"] for e in evidence_summaries if e["source_video"]))
    sections_target = timeline_segment_count_for_duration(target_duration_seconds)
    chunk_size = max(1, len(evidence_summaries) // max(1, sections_target - 2))

    doc_arc = build_documentary_arc(prompt_profile, selected_candidates, target_duration_seconds)
    arc_sections = doc_arc.get("sections") or []

    if requested_speaker:
        speaker_label = requested_speaker
        themes_covered = list(dict.fromkeys(
            c.get("primary_theme") for c in selected_candidates if c.get("primary_theme")
        ))[:4]
        hook_narration = (
            f"{speaker_label} has been a voice in the AI4IA conversation across multiple themes. "
            + (f"Their contributions touch on {', '.join(themes_covered)}." if themes_covered else "The following recordings trace the questions they returned to most often.")
        )
        sections.append({
            "section_id": "S1",
            "title": f"{speaker_label}: The Contribution",
            "narration": hook_narration,
            "evidence_fragment_ids": fragment_ids_used[:2],
            "source_videos": source_videos_used[:2],
            "evidence_terms": [],
            "claim_type": "direct_evidence",
        })
        for i, ev in enumerate(evidence_summaries[:sections_target - 2], start=2):
            theme_label = ev.get("primary_theme") or "AI and Society"
            ev_text = ev["text"][:200].rstrip(".").strip()
            narration = f"On the theme of {theme_label}: {ev_text}."
            sections.append({
                "section_id": f"S{i}",
                "title": f"{speaker_label}: {theme_label}",
                "narration": narration,
                "evidence_fragment_ids": [ev["fragment_id"]] if ev["fragment_id"] else [],
                "source_videos": [ev["source_video"]] if ev["source_video"] else [],
                "evidence_terms": ev["direct_terms"] + ev["expanded_terms"],
                "claim_type": "direct_evidence",
            })
        primary_topic = (prompt_profile.get("topic_terms") or [None])[0]
        closing = CINEMATIC_CLOSINGS.get(primary_topic or "", f"The future of AI will be shaped by the questions people like {speaker_label} are willing to ask — and by whether institutions are willing to answer them.")
        sections.append({
            "section_id": f"S{len(sections) + 1}",
            "title": "The Meaning",
            "narration": closing,
            "evidence_fragment_ids": [],
            "source_videos": [],
            "evidence_terms": [],
            "claim_type": "synthesis",
        })
    else:
        # Use documentary arc sections with cinematic narration
        for i, arc_sec in enumerate(arc_sections):
            role = arc_sec.get("role", "context")
            arc_ids = arc_sec.get("evidence_fragment_ids") or []
            arc_sources = list(dict.fromkeys(
                c.get("source_video") for c in selected_candidates
                if c.get("fragment_id") in arc_ids and c.get("source_video")
            ))
            arc_terms = list(dict.fromkeys(
                t for c in selected_candidates
                if c.get("fragment_id") in arc_ids
                for t in ((c.get("direct_evidence_terms") or []) + (c.get("expanded_evidence_terms") or []))
            ))[:6]
            if role == "synthesis":
                claim = "synthesis"
            elif not arc_ids:
                claim = "synthesis"
            else:
                claim = "direct_evidence"
            sections.append({
                "section_id": f"S{i + 1}",
                "title": arc_sec.get("title", f"{topic_label}"),
                "narration": arc_sec.get("viewer_narration", ""),
                "evidence_fragment_ids": arc_ids,
                "source_videos": arc_sources,
                "evidence_terms": arc_terms or all_terms[:4],
                "claim_type": claim,
            })

    full_narration = " ".join(s["narration"] for s in sections)
    return {
        "generation_mode": "evidence_grounded_fallback",
        "fallback_reason": "llm_unavailable_or_failed",
        "narration_text": full_narration,
        "sections": sections,
    }
