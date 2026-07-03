import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.core.config import VLLM_MODEL
from app.core.paths import STORYLINES_DIR
from app.services import fragment_store
from app.services.llm_service import generate_json

logger = logging.getLogger("ai4ia.services.concept_mindmap")


_GENERIC_CONCEPT_TERMS = {
    "ai",
    "artificial intelligence",
    "technology",
    "technologies",
    "digital",
    "access",
    "information",
    "people",
    "community",
    "communities",
    "systems",
    "system",
    "platform",
    "platforms",
    "data",
    "archive",
    "documentary",
    "video",
    "issue",
    "issues",
    "topic",
    "topics",
}

_SECTION_TYPE_LABELS = {
    "opening_hook": "Opening Context",
    "opening_context": "Opening Context",
    "exact_evidence_clip": "Direct Evidence",
    "related_theme_clip": "Related Theme",
    "related_supporting_clip": "Supporting Context",
    "related_context": "Supporting Context",
    "related_governance_clip": "Governance Response",
    "synthesis_narration": "Synthesis",
    "closing_takeaway": "Conclusion",
    "evidence_gap_narration": "Evidence Gap",
    "available_evidence_clip": "Available Evidence",
}

_TERM_SYNONYMS = {
    "youth": "youth empowerment",
    "young people": "youth empowerment",
    "women": "gender inclusion",
    "girls": "gender inclusion",
    "gender": "gender inclusion",
    "inclusion": "social inclusion",
    "marginalized": "social inclusion",
    "marginalized groups": "social inclusion",
    "trust": "public trust",
    "governance": "governance frameworks",
    "policy": "governance frameworks",
    "regulation": "governance frameworks",
}


def generate_concept_mindmap(
    query: str,
    storyline_id: str | None = None,
    mode: str = "video",
) -> dict[str, Any]:
    """Build a mindmap for a generated documentary video.

    mode="video" (default): NotebookLM-style — the map explains what the video
    itself says (its ideas, arguments, evidence, and how the narrative unfolds),
    with themes/chapters and speakers as supporting branches. Falls back to the
    legacy theme-centric map if the LLM extraction fails.

    mode="legacy": the original theme/section-centric heuristic map.
    """
    payload, source_path = _resolve_storyline(query, storyline_id)
    if mode != "legacy":
        try:
            return _generate_video_mindmap(query, payload, source_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Video-centric mindmap failed (%s); falling back to theme-centric map", exc
            )
    return _generate_theme_mindmap(query, payload, source_path)


# ---------------------------------------------------------------------------
# Video-centric (NotebookLM-style) mindmap
# ---------------------------------------------------------------------------

_VIDEO_MINDMAP_PROMPT = """You are building a NotebookLM-style mind map that explains a documentary video to someone who has not watched it.

Below is the video's complete script in playback order: the narrator's voiceover and the speakers' on-screen quotes.

Build the mind map around WHAT THE VIDEO ACTUALLY SAYS — its ideas, arguments, evidence, and message — not around production structure, chapters, or who appears in it. Never invent facts that are not in the script.

Return ONLY valid JSON with exactly this shape:
{{
  "video_overview": "2-3 sentences explaining what this video is about and what it argues",
  "key_takeaway": "the single most important message of the video, one sentence",
  "main_ideas": [
    {{
      "title": "short noun phrase, 3-6 words",
      "summary": "2-3 sentences on what the video says about this idea",
      "why_it_matters": "1-2 sentences on why the video treats this as important",
      "key_points": [
        {{
          "title": "short label, 3-8 words",
          "explanation": "1-2 sentences grounded in the script",
          "supporting_quote": "a quote copied verbatim from the script, or empty string",
          "speaker": "name of the person quoted, or empty string"
        }}
      ]
    }}
  ],
  "narrative_flow": ["3-5 short steps describing how the video's argument unfolds from opening to closing"],
  "conclusion": "what the video leaves the viewer with, 1-2 sentences"
}}

Rules:
- 4 to 7 main_ideas, ordered from most to least central, together covering everything important in the video.
- 2 to 4 key_points per idea, each grounded in the script.
- supporting_quote must be an exact substring of a quote in the script when one fits; otherwise use "".

VIDEO TITLE: {title}
LOGLINE: {logline}

VIDEO SCRIPT (playback order):
{script}
"""

_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "can", "will", "into", "about", "their",
    "them", "they", "these", "those", "when", "what", "how", "who", "why",
    "more", "most", "than", "its", "our", "your", "his", "her", "she", "him",
}


def _generate_video_mindmap(query: str, payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    story = payload.get("storyline") or payload
    segments = _build_video_script(payload, story)
    if not segments:
        raise ValueError("Storyline contains no video script content")

    all_sources = _collect_all_sources(payload)
    script_text = _render_script(segments)

    prompt = _VIDEO_MINDMAP_PROMPT.format(
        title=clean_text(story.get("title")) or "Untitled Documentary",
        logline=clean_text(story.get("logline")) or "",
        script=script_text,
    )
    extraction = generate_json(prompt, model=VLLM_MODEL, temperature=0.2)
    if not isinstance(extraction, dict) or not extraction.get("main_ideas"):
        raise ValueError("LLM returned no main_ideas for video mindmap")

    return _assemble_video_mindmap(query, payload, story, source_path, extraction, segments, all_sources)


def _build_video_script(payload: dict[str, Any], story: dict[str, Any]) -> list[dict[str, str]]:
    """Return the video's content in playback order: narration + speaker quotes."""
    segments: list[dict[str, str]] = []
    timeline = (
        payload.get("final_timeline")
        or story.get("final_timeline")
        or (story.get("validated_narrative") or {}).get("final_timeline")
        or []
    )
    for item in timeline:
        kind = clean_text(item.get("segment_type") or item.get("type"))
        if kind == "speaker_clip":
            text = clean_text(item.get("quote_text"))
            if text:
                segments.append({"kind": "quote", "speaker": clean_text(item.get("speaker")), "text": text})
        elif kind in {"narration_broll", "narration", "voiceover"}:
            text = clean_text(item.get("voiceover_text") or item.get("text"))
            if text:
                segments.append({"kind": "narration", "speaker": "", "text": text})

    if segments:
        return segments

    # Fallback: voiceover script + selected soundbites when no final timeline exists.
    for entry in story.get("voiceover_script") or []:
        text = clean_text(entry.get("text"))
        if text:
            segments.append({"kind": "narration", "speaker": "", "text": text})
    for bite in story.get("selected_speaker_soundbites") or []:
        text = clean_text(bite.get("quote_summary"))
        if text:
            frag = fragment_store.get(bite.get("fragment_id") or "") or {}
            segments.append({"kind": "quote", "speaker": clean_text(frag.get("speaker")), "text": text})
    return segments


def _render_script(segments: list[dict[str, str]], max_chars: int = 9000) -> str:
    lines = []
    total = 0
    for idx, seg in enumerate(segments, 1):
        text = seg["text"][:700]
        if seg["kind"] == "quote":
            speaker = seg["speaker"] or "Speaker"
            line = f'[{idx}] {speaker} (on camera): "{text}"'
        else:
            line = f"[{idx}] NARRATOR: {text}"
        total += len(line)
        if total > max_chars:
            break
        lines.append(line)
    return "\n".join(lines)


def _collect_all_sources(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve every evidence fragment used by the storyline into source records."""
    story = payload.get("storyline") or payload
    by_fragment = {
        item.get("fragment_id"): item
        for item in (payload.get("selected_candidates") or []) + (payload.get("supporting_candidates") or [])
        if item.get("fragment_id")
    }
    fragment_ids: list[str] = list(by_fragment.keys())
    for section in story.get("sections") or []:
        fragment_ids.extend(section.get("evidence_fragment_ids") or [])
        for item in (section.get("speaker_soundbites") or []) + (section.get("archive_visual_fragments") or []):
            if item.get("fragment_id"):
                fragment_ids.append(item["fragment_id"])
    pseudo_section = {"evidence_fragment_ids": fragment_ids}
    return _resolve_section_sources(pseudo_section, by_fragment)


def _content_tokens(text: str) -> set[str]:
    return {tok for tok in _tokenize(text) if tok not in _STOPWORDS}


def _match_sources(text: str, sources: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    """Ground a node's text against fragment sources by content-token overlap."""
    target = _content_tokens(text)
    if not target:
        return []
    scored = []
    for source in sources:
        haystack = _content_tokens(
            _join_parts(source.get("summary"), source.get("text"), " ".join(source.get("keywords") or []))
        )
        overlap = len(target & haystack)
        if overlap >= 2:
            scored.append((overlap, source))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [source for _, source in scored[:limit]]


def _find_quote_source(quote: str, speaker: str, sources: list[dict[str, Any]]) -> dict[str, Any] | None:
    quote_norm = clean_text(quote).lower()
    speaker_norm = clean_text(speaker).lower()
    for source in sources:
        if quote_norm and quote_norm[:60] in clean_text(source.get("text")).lower():
            return source
    if speaker_norm:
        for source in sources:
            if speaker_norm == clean_text(source.get("speaker")).lower():
                return source
    return None


def _assemble_video_mindmap(
    query: str,
    payload: dict[str, Any],
    story: dict[str, Any],
    source_path: Path,
    extraction: dict[str, Any],
    segments: list[dict[str, str]],
    all_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    counter = {"value": 0}

    video_title = clean_text(story.get("title")) or clean_text(query) or "Documentary Video"
    overview = clean_text(extraction.get("video_overview")) or clean_text(story.get("logline"))
    key_takeaway = clean_text(extraction.get("key_takeaway"))

    root_id = _next_id("video", counter)
    nodes.append(
        _make_node(
            node_id=root_id,
            parent_id=None,
            title=video_title,
            node_type="root_topic",
            summary=overview,
            details=_join_parts(key_takeaway, story.get("logline"), story.get("central_question")),
            evidence=[],
            sources=[],
            speakers=[],
            importance=100,
            relationships=[],
            quotes=[],
            examples=[],
            counterpoints=[],
        )
    )

    # Quotes spoken in the video, grouped per speaker, for grounding.
    script_quotes: list[dict[str, Any]] = [
        {"text": seg["text"], "speaker": seg["speaker"], "fragment_id": None, "reason": ""}
        for seg in segments
        if seg["kind"] == "quote"
    ]

    # --- Main branches: the ideas the video explains ---------------------
    idea_nodes: list[dict[str, Any]] = []
    quote_speaker_to_ideas: dict[str, set[str]] = defaultdict(set)
    main_ideas = [idea for idea in extraction.get("main_ideas") or [] if isinstance(idea, dict)]
    for idx, idea in enumerate(main_ideas[:8]):
        idea_title = clean_text(idea.get("title"))
        if not idea_title:
            continue
        idea_summary = clean_text(idea.get("summary"))
        idea_sources = _match_sources(_join_parts(idea_title, idea_summary), all_sources)
        idea_id = _next_id("idea", counter)
        idea_node = _make_node(
            node_id=idea_id,
            parent_id=root_id,
            title=idea_title,
            node_type="theme",
            summary=idea_summary,
            details=_join_parts(idea_summary, idea.get("why_it_matters")),
            evidence=_build_evidence_entries(idea_sources),
            sources=idea_sources,
            speakers=_merge_people([], [{"name": s.get("speaker")} for s in idea_sources if s.get("speaker")]),
            importance=max(60, 96 - idx * 6),
            relationships=[root_id],
            quotes=[],
            examples=[],
            counterpoints=[],
        )
        nodes.append(idea_node)
        idea_nodes.append(idea_node)
        _add_edge(edges, seen_edges, root_id, idea_id, "hierarchy")

        for point in (idea.get("key_points") or [])[:4]:
            if not isinstance(point, dict):
                continue
            point_title = clean_text(point.get("title"))
            if not point_title:
                continue
            explanation = clean_text(point.get("explanation"))
            quote_text = clean_text(point.get("supporting_quote"))
            quote_speaker = clean_text(point.get("speaker"))
            point_sources = _match_sources(_join_parts(point_title, explanation, quote_text), all_sources)
            point_quotes = []
            if quote_text:
                quote_source = _find_quote_source(quote_text, quote_speaker, all_sources)
                point_quotes.append(
                    {
                        "text": quote_text,
                        "speaker": quote_speaker or clean_text((quote_source or {}).get("speaker")),
                        "fragment_id": (quote_source or {}).get("fragment_id"),
                        "reason": "Spoken in the video",
                    }
                )
                if quote_source:
                    point_sources = _merge_sources(point_sources, [quote_source])
                if quote_speaker:
                    quote_speaker_to_ideas[quote_speaker.lower()].add(idea_id)
                idea_node["quotes"] = _merge_simple_objects(idea_node["quotes"], point_quotes, "text")

            point_id = _next_id("point", counter)
            nodes.append(
                _make_node(
                    node_id=point_id,
                    parent_id=idea_id,
                    title=point_title,
                    node_type="concept",
                    summary=explanation,
                    details=_join_parts(explanation, quote_text and f'In the video: "{quote_text}"'),
                    evidence=_build_evidence_entries(point_sources),
                    sources=point_sources,
                    speakers=_merge_people([], [{"name": quote_speaker}] if quote_speaker else []),
                    importance=max(45, idea_node["importanceScore"] - 12),
                    relationships=[idea_id],
                    quotes=point_quotes,
                    examples=[],
                    counterpoints=[],
                )
            )
            _add_edge(edges, seen_edges, idea_id, point_id, "hierarchy")

    if not idea_nodes:
        raise ValueError("LLM extraction produced no usable main ideas")

    # Narrative order links between consecutive ideas.
    for left, right in zip(idea_nodes, idea_nodes[1:]):
        _add_edge(edges, seen_edges, left["id"], right["id"], "related")
        left["relationships"] = _merge_unique(left["relationships"], [right["id"]])

    # --- Supporting branch: how the video unfolds -------------------------
    flow_steps = [clean_text(step) for step in extraction.get("narrative_flow") or [] if clean_text(step)]
    if flow_steps:
        flow_id = _next_id("flow", counter)
        nodes.append(
            _make_node(
                node_id=flow_id,
                parent_id=root_id,
                title="How the Video Unfolds",
                node_type="subtheme",
                summary=flow_steps[0],
                details=" → ".join(flow_steps),
                evidence=[],
                sources=[],
                speakers=[],
                importance=70,
                relationships=[root_id],
                quotes=[],
                examples=flow_steps,
                counterpoints=[],
            )
        )
        _add_edge(edges, seen_edges, root_id, flow_id, "hierarchy")

    # --- Supporting branch: voices heard in the video ---------------------
    speakers_in_video: dict[str, dict[str, Any]] = {}
    generic_speaker_labels = {"speaker", "panelist", "moderator", "narrator", "unknown"}
    for quote in script_quotes:
        name = clean_text(quote.get("speaker"))
        if not name or name.lower() in generic_speaker_labels:
            continue
        rec = speakers_in_video.setdefault(name.lower(), {"name": name, "quotes": []})
        rec["quotes"] = _merge_simple_objects(rec["quotes"], [quote], "text")
    if speakers_in_video:
        voices_id = _next_id("voices", counter)
        nodes.append(
            _make_node(
                node_id=voices_id,
                parent_id=root_id,
                title="Voices in the Video",
                node_type="subtheme",
                summary=f"{len(speakers_in_video)} speakers appear on camera in this video.",
                details="The people whose words carry the video's argument.",
                evidence=[],
                sources=[],
                speakers=[{"name": rec["name"]} for rec in speakers_in_video.values()],
                importance=65,
                relationships=[root_id],
                quotes=[],
                examples=[],
                counterpoints=[],
            )
        )
        _add_edge(edges, seen_edges, root_id, voices_id, "hierarchy")
        for key, rec in speakers_in_video.items():
            speaker_sources = [s for s in all_sources if clean_text(s.get("speaker")).lower() == key]
            speaker_id = _next_id("speaker", counter)
            speaker_node = _make_node(
                node_id=speaker_id,
                parent_id=voices_id,
                title=rec["name"],
                node_type="speaker",
                summary=f"{rec['name']} speaks on camera in this video.",
                details=_join_parts(*[f'"{q["text"]}"' for q in rec["quotes"][:2]]),
                evidence=_build_evidence_entries(speaker_sources),
                sources=speaker_sources,
                speakers=[{"name": rec["name"]}],
                importance=55,
                relationships=sorted(quote_speaker_to_ideas.get(key, set())),
                quotes=rec["quotes"],
                examples=[],
                counterpoints=[],
            )
            nodes.append(speaker_node)
            _add_edge(edges, seen_edges, voices_id, speaker_id, "hierarchy")
            for idea_id in quote_speaker_to_ideas.get(key, set()):
                _add_edge(edges, seen_edges, idea_id, speaker_id, "speaker")

    # --- Supporting branch: themes & chapters the evidence comes from -----
    theme_counts = Counter()
    for source in all_sources:
        for field in ("primary_theme", "primary_chapter"):
            label = clean_text(source.get(field))
            if label:
                theme_counts[label] += 1
    if theme_counts:
        themes_id = _next_id("themes", counter)
        nodes.append(
            _make_node(
                node_id=themes_id,
                parent_id=root_id,
                title="Themes & Chapters Covered",
                node_type="subtheme",
                summary="Where the video's evidence comes from in the AI4IA archive.",
                details="The archive themes and chapters the video draws its clips and quotes from.",
                evidence=[],
                sources=[],
                speakers=[],
                importance=60,
                relationships=[root_id],
                quotes=[],
                examples=[],
                counterpoints=[],
            )
        )
        _add_edge(edges, seen_edges, root_id, themes_id, "hierarchy")
        for label, count in theme_counts.most_common(6):
            theme_sources = [
                s for s in all_sources
                if label in (clean_text(s.get("primary_theme")), clean_text(s.get("primary_chapter")))
            ][:4]
            theme_id = _next_id("archivetheme", counter)
            nodes.append(
                _make_node(
                    node_id=theme_id,
                    parent_id=themes_id,
                    title=label,
                    node_type="concept",
                    summary=f"{count} evidence fragment{'s' if count != 1 else ''} in the video come from {label}.",
                    details=_join_parts(*[s.get("summary") for s in theme_sources[:2]]),
                    evidence=_build_evidence_entries(theme_sources),
                    sources=theme_sources,
                    speakers=[],
                    importance=min(75, 45 + count * 4),
                    relationships=[themes_id],
                    quotes=[],
                    examples=[],
                    counterpoints=[],
                )
            )
            _add_edge(edges, seen_edges, themes_id, theme_id, "hierarchy")

    # --- Conclusion --------------------------------------------------------
    conclusion_text = clean_text(extraction.get("conclusion")) or clean_text(story.get("closing_meaning"))
    if conclusion_text:
        conclusion_id = _next_id("conclusion", counter)
        nodes.append(
            _make_node(
                node_id=conclusion_id,
                parent_id=root_id,
                title="What the Video Leaves You With",
                node_type="conclusion",
                summary=conclusion_text,
                details=_join_parts(conclusion_text, story.get("closing_meaning"), story.get("central_question")),
                evidence=[],
                sources=[],
                speakers=[],
                importance=88,
                relationships=[root_id],
                quotes=[],
                examples=[],
                counterpoints=[],
            )
        )
        _add_edge(edges, seen_edges, root_id, conclusion_id, "hierarchy")

    _backfill_related_node_ids(nodes)

    statistics = {
        "themes": len(idea_nodes),
        "concepts": sum(1 for n in nodes if n["type"] == "concept"),
        "quotes": sum(len(n.get("quotes") or []) for n in nodes),
        "speakers": sum(1 for n in nodes if n["type"] == "speaker"),
    }

    return {
        "topic": video_title,
        "summary": overview or key_takeaway,
        "mode": "video",
        "mindmap": {
            "nodes": nodes,
            "edges": edges,
        },
        "statistics": statistics,
        "storyline": {
            "storyline_id": source_path.stem,
            "path": str(source_path),
            "title": story.get("title"),
            "logline": story.get("logline"),
            "user_prompt": payload.get("user_prompt"),
        },
    }


# ---------------------------------------------------------------------------
# Legacy theme/section-centric mindmap (fallback)
# ---------------------------------------------------------------------------

def _generate_theme_mindmap(query: str, payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    story = payload.get("storyline") or payload

    root_title = clean_text(query) or clean_text(story.get("title")) or "Documentary Topic"
    root_summary = clean_text(story.get("logline") or payload.get("user_prompt") or story.get("central_question"))
    root_details = _join_parts(
        story.get("logline"),
        story.get("central_question"),
        story.get("closing_meaning"),
    )

    selected_candidates = payload.get("selected_candidates") or []
    by_fragment = {item.get("fragment_id"): item for item in selected_candidates if item.get("fragment_id")}

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_edge_keys: set[tuple[str, str, str]] = set()
    concept_registry: dict[str, dict[str, Any]] = {}
    speaker_registry: dict[str, dict[str, Any]] = {}
    node_counter = {"value": 0}

    root_id = _next_id("topic", node_counter)
    root_node = _make_node(
        node_id=root_id,
        parent_id=None,
        title=root_title,
        node_type="root_topic",
        summary=root_summary,
        details=root_details or root_summary,
        evidence=[],
        sources=[],
        speakers=[],
        importance=100,
        relationships=[],
        quotes=[],
        examples=[],
        counterpoints=[],
    )
    nodes.append(root_node)

    sections = _extract_sections(payload)
    for section in sections:
        section_id = _next_id("theme", node_counter)
        section_sources = _resolve_section_sources(section, by_fragment)
        section_speakers = _collect_speakers(section, section_sources)
        section_quotes = _collect_quotes(section, section_sources)
        section_examples = _extract_examples(section)
        section_counterpoints = _extract_counterpoints(section)
        section_importance = _importance_from_section(section, len(section_sources), len(section_quotes))
        section_title = _section_title(section)
        section_details = _section_details(section)

        theme_node = _make_node(
            node_id=section_id,
            parent_id=root_id,
            title=section_title,
            node_type="theme",
            summary=clean_text(section.get("summary") or section.get("narration") or section.get("voiceover_text")),
            details=section_details,
            evidence=_build_evidence_entries(section_sources),
            sources=section_sources,
            speakers=section_speakers,
            importance=section_importance,
            relationships=[],
            quotes=section_quotes,
            examples=section_examples,
            counterpoints=section_counterpoints,
        )
        nodes.append(theme_node)
        _add_edge(edges, seen_edge_keys, root_id, section_id, "hierarchy")

        dominant_subtheme = _dominant_subtheme(section_sources)
        parent_for_concepts = section_id
        if dominant_subtheme:
            subtheme_id = _next_id("subtheme", node_counter)
            subtheme_node = _make_node(
                node_id=subtheme_id,
                parent_id=section_id,
                title=dominant_subtheme["title"],
                node_type="subtheme",
                summary=dominant_subtheme["summary"],
                details=dominant_subtheme["details"],
                evidence=_build_evidence_entries(section_sources),
                sources=section_sources,
                speakers=section_speakers,
                importance=max(60, section_importance - 10),
                relationships=[],
                quotes=section_quotes[:2],
                examples=section_examples[:2],
                counterpoints=section_counterpoints[:2],
            )
            nodes.append(subtheme_node)
            _add_edge(edges, seen_edge_keys, section_id, subtheme_id, "hierarchy")
            parent_for_concepts = subtheme_id

        for concept in _extract_section_concepts(section, section_sources):
            normalized = _normalize_concept(concept["title"])
            existing = concept_registry.get(normalized)
            if existing:
                existing_node = existing["node"]
                existing["sections"].add(section_id)
                existing_node["importanceScore"] = min(100, existing_node["importanceScore"] + 4)
                existing_node["relationships"] = _merge_unique(existing_node["relationships"], [section_id])
                existing_node["sources"] = _merge_sources(existing_node["sources"], concept["sources"])
                existing_node["evidence"] = _merge_simple_objects(existing_node["evidence"], concept["evidence"], "text")
                existing_node["speakers"] = _merge_people(existing_node["speakers"], concept["speakers"])
                existing_node["quotes"] = _merge_simple_objects(existing_node["quotes"], concept["quotes"], "text")
                existing_node["examples"] = _merge_simple_strings(existing_node["examples"], concept["examples"])
                existing_node["counterpoints"] = _merge_simple_strings(existing_node["counterpoints"], concept["counterpoints"])
                _add_edge(edges, seen_edge_keys, parent_for_concepts, existing_node["id"], "hierarchy")
            else:
                concept_id = _next_id("concept", node_counter)
                concept_node = _make_node(
                    node_id=concept_id,
                    parent_id=parent_for_concepts,
                    title=concept["title"],
                    node_type="concept",
                    summary=concept["summary"],
                    details=concept["details"],
                    evidence=concept["evidence"],
                    sources=concept["sources"],
                    speakers=concept["speakers"],
                    importance=concept["importance"],
                    relationships=[section_id],
                    quotes=concept["quotes"],
                    examples=concept["examples"],
                    counterpoints=concept["counterpoints"],
                )
                nodes.append(concept_node)
                concept_registry[normalized] = {"node": concept_node, "sections": {section_id}}
                _add_edge(edges, seen_edge_keys, parent_for_concepts, concept_id, "hierarchy")

        for speaker in section_speakers:
            speaker_name = clean_text(speaker.get("name"))
            if not speaker_name:
                continue
            speaker_key = speaker_name.lower()
            existing = speaker_registry.get(speaker_key)
            if existing:
                existing["sections"].add(section_id)
                existing["node"]["relationships"] = _merge_unique(existing["node"]["relationships"], [section_id])
                _add_edge(edges, seen_edge_keys, section_id, existing["node"]["id"], "speaker")
                continue

            speaker_id = _next_id("speaker", node_counter)
            speaker_node = _make_node(
                node_id=speaker_id,
                parent_id=section_id,
                title=speaker_name,
                node_type="speaker",
                summary=speaker.get("summary") or f"Speaker contributing perspective on {section_title.lower()}.",
                details=speaker.get("details") or speaker.get("summary") or "",
                evidence=[],
                sources=speaker.get("sources") or [],
                speakers=[speaker],
                importance=max(45, section_importance - 20),
                relationships=[section_id],
                quotes=speaker.get("quotes") or [],
                examples=[],
                counterpoints=[],
            )
            nodes.append(speaker_node)
            speaker_registry[speaker_key] = {"node": speaker_node, "sections": {section_id}}
            _add_edge(edges, seen_edge_keys, section_id, speaker_id, "speaker")

    closing_text = clean_text(story.get("closing_meaning"))
    if closing_text:
        conclusion_id = _next_id("conclusion", node_counter)
        conclusion_node = _make_node(
            node_id=conclusion_id,
            parent_id=root_id,
            title="Conclusion",
            node_type="conclusion",
            summary=closing_text,
            details=_join_parts(story.get("closing_meaning"), story.get("central_question"), story.get("logline")),
            evidence=[],
            sources=[],
            speakers=[],
            importance=90,
            relationships=[],
            quotes=[],
            examples=[],
            counterpoints=[],
        )
        nodes.append(conclusion_node)
        _add_edge(edges, seen_edge_keys, root_id, conclusion_id, "hierarchy")

    _link_related_concepts(concept_registry, edges, seen_edge_keys)
    _backfill_related_node_ids(nodes)

    statistics = {
        "themes": sum(1 for n in nodes if n["type"] == "theme"),
        "concepts": sum(1 for n in nodes if n["type"] == "concept"),
        "quotes": sum(len(n.get("quotes") or []) for n in nodes if n["type"] in {"theme", "concept", "speaker"}),
        "speakers": len([n for n in nodes if n["type"] == "speaker"]),
    }

    return {
        "topic": root_title,
        "summary": root_summary or root_details,
        "mode": "legacy",
        "mindmap": {
            "nodes": nodes,
            "edges": edges,
        },
        "statistics": statistics,
        "storyline": {
            "storyline_id": source_path.stem,
            "path": str(source_path),
            "title": story.get("title"),
            "logline": story.get("logline"),
            "user_prompt": payload.get("user_prompt"),
        },
    }


def _list_s3_storylines() -> list[dict]:
    """Return list of {key, last_modified, name} for storyline files in S3."""
    try:
        from app.pipeline.s3_helpers import _s3, _bucket
        s3 = _s3()
        bucket = _bucket()
        paginator = s3.get_paginator("list_objects_v2")
        results = []
        for page in paginator.paginate(Bucket=bucket, Prefix="raw/storylines/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                name = Path(key).name
                if name.startswith("storyline_") and name.endswith(".json"):
                    results.append({"key": key, "last_modified": obj["LastModified"], "name": name})
        return sorted(results, key=lambda x: x["last_modified"], reverse=True)
    except Exception:
        return []


def _read_s3_storyline(key: str) -> dict:
    from app.pipeline.s3_helpers import _s3, _bucket
    resp = _s3().get_object(Bucket=_bucket(), Key=key)
    return json.loads(resp["Body"].read().decode("utf-8"))


def _resolve_storyline_from_job(job_id: str) -> tuple[dict[str, Any], Path] | None:
    try:
        from app.services.job_store import load_job
        job = load_job(job_id)
    except FileNotFoundError:
        return None

    paths = job.get("paths") or {}
    storyline_path = paths.get("validated_storyline_path") or paths.get("storyline_path")
    if storyline_path:
        direct = Path(storyline_path)
        if direct.exists():
            return json.loads(direct.read_text(encoding="utf-8")), direct

        local_fallback = STORYLINES_DIR / direct.name
        if local_fallback.exists():
            return json.loads(local_fallback.read_text(encoding="utf-8")), local_fallback

    # Try S3 whenever the key is recorded — the file may have been generated on
    # the worker VM and uploaded there even when the API server runs in local mode.
    s3_key = paths.get("validated_storyline_s3_key") or paths.get("storyline_s3_key")
    if s3_key:
        try:
            return _read_s3_storyline(s3_key), Path(s3_key)
        except Exception:
            pass

    return None


def _resolve_storyline(query: str, storyline_id: str | None) -> tuple[dict[str, Any], Path]:
    if storyline_id:
        direct = STORYLINES_DIR / f"{storyline_id}.json"
        if direct.exists():
            return json.loads(direct.read_text(encoding="utf-8")), direct
        job_storyline = _resolve_storyline_from_job(storyline_id)
        if job_storyline:
            return job_storyline
        s3_items = _list_s3_storylines()
        for item in s3_items:
            if storyline_id in item["name"]:
                return _read_s3_storyline(item["key"]), Path(item["name"])
        raise FileNotFoundError(f"Storyline '{storyline_id}' not found")

    # Try local first, then S3
    candidates = sorted(STORYLINES_DIR.glob("storyline_*.json"), reverse=True)
    query_tokens = set(_tokenize(query))
    query_norm = clean_text(query).lower()

    # Score and pick best from local candidates
    best_path = None
    best_score = -1.0
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        story = payload.get("storyline") or payload
        fields = [
            payload.get("user_prompt", ""),
            story.get("title", ""),
            story.get("logline", ""),
            story.get("central_question", ""),
        ]
        haystack = " ".join(clean_text(f).lower() for f in fields if f)
        if not haystack:
            continue
        score = 0.0
        if query_norm and query_norm in haystack:
            score += 5.0
        overlap = query_tokens.intersection(_tokenize(haystack))
        if query_tokens:
            score += len(overlap) / max(len(query_tokens), 1)
        if path.name.endswith("_validated.json"):
            score += 0.25
        if score > best_score:
            best_score = score
            best_path = path

    if best_path and best_score >= 0.35:
        return json.loads(best_path.read_text(encoding="utf-8")), best_path

    # Fall back to S3 when no local candidates or no good match
    if use_s3():
        s3_items = _list_s3_storylines()
        if not s3_items:
            raise FileNotFoundError("No storyline files found")
        best_s3: dict | None = None
        best_score = -1.0
        for item in s3_items:
            payload = _read_s3_storyline(item["key"])
            story = payload.get("storyline") or payload
            fields = [
                payload.get("user_prompt", ""),
                story.get("title", ""),
                story.get("logline", ""),
                story.get("central_question", ""),
            ]
            haystack = " ".join(clean_text(f).lower() for f in fields if f)
            score = 0.0
            if query_norm and query_norm in haystack:
                score += 5.0
            overlap = query_tokens.intersection(_tokenize(haystack))
            if query_tokens:
                score += len(overlap) / max(len(query_tokens), 1)
            if item["name"].endswith("_validated.json"):
                score += 0.25
            if score > best_score:
                best_score = score
                best_s3 = {"payload": payload, "name": item["name"]}
        if best_s3 and best_score >= 0.0:
            return best_s3["payload"], Path(best_s3["name"])

    raise FileNotFoundError(
        "No matching storyline found for this query. Generate a storyline first or pass storyline_id."
    )


def _extract_sections(payload: dict[str, Any]) -> list[dict[str, Any]]:
    story = payload.get("storyline") or payload
    validated = story.get("validated_narrative") or {}
    arcs = validated.get("narrative_arc") or []
    sections = story.get("sections") or []

    if arcs:
        mapped_sections = {clean_text(sec.get("section_id")): sec for sec in sections}
        out = []
        for arc in arcs:
            key = clean_text(arc.get("section_id"))
            base = mapped_sections.get(key, {})
            out.append(
                {
                    "section_id": key or base.get("section_id"),
                    "card_text": arc.get("card_text"),
                    "purpose": arc.get("purpose"),
                    "voiceover_text": arc.get("voiceover_text"),
                    "timeline_items": arc.get("timeline_items") or [],
                    "section_type": base.get("section_type") or arc.get("section_type"),
                    "narration": base.get("narration") or arc.get("voiceover_text"),
                    "evidence_fragment_ids": base.get("evidence_fragment_ids") or [],
                    "speaker_soundbites": base.get("speaker_soundbites") or [],
                    "archive_visual_fragments": base.get("archive_visual_fragments") or [],
                    "evidence_terms": base.get("evidence_terms") or [],
                    "source_videos": base.get("source_videos") or [],
                    "target_duration_seconds": base.get("target_duration_seconds") or 0,
                    "planning_note": base.get("planning_note"),
                    "director_role": base.get("director_role"),
                    "summary": arc.get("purpose") or base.get("narration"),
                }
            )
        if out:
            return out

    return sections


def _resolve_section_sources(section: dict[str, Any], by_fragment: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    seen = set()

    fragment_ids = list(section.get("evidence_fragment_ids") or [])
    for item in section.get("speaker_soundbites") or []:
        if item.get("fragment_id"):
            fragment_ids.append(item["fragment_id"])
    for item in section.get("archive_visual_fragments") or []:
        if item.get("fragment_id"):
            fragment_ids.append(item["fragment_id"])
    for item in section.get("timeline_items") or []:
        if item.get("fragment_id"):
            fragment_ids.append(item["fragment_id"])

    for fragment_id in fragment_ids:
        if not fragment_id or fragment_id in seen:
            continue
        seen.add(fragment_id)
        frag = fragment_store.get(fragment_id) or {}
        candidate = by_fragment.get(fragment_id) or {}
        text = clean_text(
            frag.get("text")
            or candidate.get("text")
            or candidate.get("summary")
            or ""
        )
        sources.append(
            {
                "fragment_id": fragment_id,
                "speaker": frag.get("speaker") or candidate.get("speaker"),
                "speaker_attribution": frag.get("speaker_attribution"),
                "source_video": frag.get("source_video") or candidate.get("source_video"),
                "video_title": frag.get("video_title"),
                "primary_theme": frag.get("primary_theme") or candidate.get("primary_theme"),
                "primary_chapter": frag.get("primary_chapter") or candidate.get("primary_chapter"),
                "start": frag.get("start") or candidate.get("start"),
                "end": frag.get("end") or candidate.get("end"),
                "duration": frag.get("duration") or candidate.get("duration"),
                "text": text,
                "summary": clean_text(candidate.get("summary") or text),
                "keywords": candidate.get("keywords") or frag.get("keywords") or [],
                "stream_url": frag.get("stream_url"),
            }
        )

    return sources


def _section_title(section: dict[str, Any]) -> str:
    for field in ("card_text", "title", "heading"):
        value = clean_text(section.get(field))
        if value:
            return value
    section_type = clean_text(section.get("section_type"))
    if section_type:
        return _SECTION_TYPE_LABELS.get(section_type, section_type.replace("_", " ").title())
    purpose = clean_text(section.get("purpose"))
    if purpose:
        return purpose.split(".")[0][:70]
    return "Narrative Section"


def _section_details(section: dict[str, Any]) -> str:
    return _join_parts(
        section.get("purpose"),
        section.get("planning_note"),
        section.get("narration"),
        section.get("voiceover_text"),
    )


def _dominant_subtheme(section_sources: list[dict[str, Any]]) -> dict[str, str] | None:
    chapter_counts = Counter()
    theme_counts = Counter()
    for source in section_sources:
        chapter = clean_text(source.get("primary_chapter"))
        theme = clean_text(source.get("primary_theme"))
        if chapter:
            chapter_counts[chapter] += 1
        if theme:
            theme_counts[theme] += 1

    if chapter_counts:
        chapter, count = chapter_counts.most_common(1)[0]
        paired_theme = theme_counts.most_common(1)[0][0] if theme_counts else ""
        return {
            "title": chapter,
            "summary": f"This section mainly draws evidence from {chapter}.",
            "details": _join_parts(
                f"The documentary groups this material under {chapter}.",
                f"Most supporting fragments also point to {paired_theme}." if paired_theme else "",
            ),
        }
    return None


def _extract_section_concepts(section: dict[str, Any], section_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terms = Counter()
    term_sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    quotes = _collect_quotes(section, section_sources)
    speakers = _collect_speakers(section, section_sources)
    examples = _extract_examples(section)
    counterpoints = _extract_counterpoints(section)

    for term in section.get("evidence_terms") or []:
        label = _clean_concept_label(term)
        if label:
            terms[label] += 3

    for source in section_sources:
        for term in source.get("keywords") or []:
            label = _clean_concept_label(term)
            if label:
                terms[label] += 2
                term_sources[label].append(source)

        for field in ("primary_theme", "primary_chapter"):
            label = _clean_concept_label(source.get(field))
            if label:
                terms[label] += 1
                term_sources[label].append(source)

    chosen = [label for label, _ in terms.most_common(6)]
    concepts = []
    for label in chosen:
        related_sources = term_sources.get(label) or section_sources[:2]
        related_quotes = [q for q in quotes if _concept_matches_text(label, q.get("text", ""))]
        related_examples = [e for e in examples if _concept_matches_text(label, e)]
        related_counterpoints = [c for c in counterpoints if _concept_matches_text(label, c)]
        concepts.append(
            {
                "title": label,
                "summary": _concept_summary(label, section, related_sources),
                "details": _concept_details(label, section, related_sources),
                "evidence": _build_evidence_entries(related_sources),
                "sources": related_sources,
                "speakers": speakers,
                "importance": min(95, 45 + terms[label] * 6),
                "quotes": related_quotes[:3] or quotes[:1],
                "examples": related_examples[:3] or examples[:2],
                "counterpoints": related_counterpoints[:2],
            }
        )

    return concepts


def _collect_quotes(section: dict[str, Any], section_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quotes = []
    seen = set()

    for item in section.get("timeline_items") or []:
        if item.get("type") != "speaker_clip":
            continue
        text = clean_text(item.get("quote_text"))
        if not text or text in seen:
            continue
        seen.add(text)
        quotes.append(
            {
                "text": text,
                "speaker": clean_text(item.get("speaker")),
                "fragment_id": item.get("fragment_id"),
                "reason": clean_text(item.get("why_here")),
            }
        )

    for item in section.get("speaker_soundbites") or []:
        text = clean_text(item.get("quote_summary"))
        if not text or text in seen:
            continue
        seen.add(text)
        source = next((s for s in section_sources if s["fragment_id"] == item.get("fragment_id")), None)
        quotes.append(
            {
                "text": text,
                "speaker": clean_text((source or {}).get("speaker")),
                "fragment_id": item.get("fragment_id"),
                "reason": clean_text(item.get("reason")),
            }
        )

    return quotes


def _collect_speakers(section: dict[str, Any], section_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for source in section_sources:
        speaker = clean_text(source.get("speaker"))
        if not speaker:
            continue
        rec = grouped.setdefault(
            speaker.lower(),
            {
                "name": speaker,
                "summary": f"{speaker} contributes evidence in this section.",
                "details": "",
                "quotes": [],
                "sources": [],
            },
        )
        rec["sources"] = _merge_sources(rec["sources"], [source])

    for quote in _collect_quotes(section, section_sources):
        speaker = clean_text(quote.get("speaker"))
        if not speaker:
            continue
        rec = grouped.setdefault(
            speaker.lower(),
            {
                "name": speaker,
                "summary": f"{speaker} contributes evidence in this section.",
                "details": "",
                "quotes": [],
                "sources": [],
            },
        )
        rec["quotes"] = _merge_simple_objects(rec["quotes"], [quote], "text")
        rec["details"] = _join_parts(rec["details"], quote.get("reason"))

    return list(grouped.values())


def _build_evidence_entries(section_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = []
    for source in section_sources[:6]:
        text = clean_text(source.get("summary") or source.get("text"))
        if not text:
            continue
        evidence.append(
            {
                "text": text,
                "speaker": source.get("speaker"),
                "fragment_id": source.get("fragment_id"),
                "theme": source.get("primary_theme"),
                "chapter": source.get("primary_chapter"),
            }
        )
    return evidence


def _extract_examples(section: dict[str, Any]) -> list[str]:
    text = _join_parts(section.get("narration"), section.get("voiceover_text"))
    if not text:
        return []
    examples = []
    for sentence in _split_sentences(text):
        lowered = sentence.lower()
        if any(marker in lowered for marker in ("for example", "example", "such as", "for instance")):
            examples.append(sentence)
    return examples[:3]


def _extract_counterpoints(section: dict[str, Any]) -> list[str]:
    text = _join_parts(section.get("purpose"), section.get("narration"), section.get("voiceover_text"))
    if not text:
        return []
    counterpoints = []
    for sentence in _split_sentences(text):
        lowered = sentence.lower()
        if any(marker in lowered for marker in ("however", "but", "risk", "challenge", "tension", "inequal", "peril")):
            counterpoints.append(sentence)
    return counterpoints[:3]


def _importance_from_section(section: dict[str, Any], source_count: int, quote_count: int) -> int:
    duration = int(float(section.get("target_duration_seconds") or 0))
    score = 50 + min(duration, 60) // 3 + source_count * 5 + quote_count * 4
    if clean_text(section.get("section_type")) in {"closing_takeaway", "opening_hook"}:
        score += 8
    return max(40, min(score, 98))


def _concept_summary(label: str, section: dict[str, Any], related_sources: list[dict[str, Any]]) -> str:
    evidence_hint = related_sources[0].get("summary") if related_sources else ""
    return clean_text(
        f"{label} is one of the key ideas used to explain this part of the documentary. {evidence_hint}"
    )


def _concept_details(label: str, section: dict[str, Any], related_sources: list[dict[str, Any]]) -> str:
    source_bits = []
    for source in related_sources[:3]:
        if source.get("summary"):
            source_bits.append(source["summary"])
    return _join_parts(
        section.get("purpose"),
        f"The storyline uses {label.lower()} to structure this section.",
        " ".join(source_bits),
    )


def _make_node(
    node_id: str,
    parent_id: str | None,
    title: str,
    node_type: str,
    summary: str,
    details: str,
    evidence: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    speakers: list[dict[str, Any]],
    importance: int,
    relationships: list[str],
    quotes: list[dict[str, Any]],
    examples: list[str],
    counterpoints: list[str],
) -> dict[str, Any]:
    return {
        "id": node_id,
        "key": node_id,
        "parentId": parent_id,
        "parent": parent_id,
        "title": title,
        "text": title,
        "type": node_type,
        "summary": clean_text(summary),
        "details": clean_text(details),
        "overview": clean_text(summary),
        "whyItMatters": clean_text(details or summary),
        "evidence": evidence,
        "sources": sources,
        "speakers": speakers,
        "quotes": quotes,
        "examples": examples,
        "counterpoints": counterpoints,
        "contradictions": counterpoints,
        "deepDive": clean_text(details),
        "importanceScore": int(max(0, min(importance, 100))),
        "relationships": relationships,
    }


def _add_edge(edges: list[dict[str, Any]], seen: set[tuple[str, str, str]], source: str, target: str, relation: str) -> None:
    edge_key = (source, target, relation)
    if edge_key in seen or source == target:
        return
    seen.add(edge_key)
    edges.append(
        {
            "from": source,
            "to": target,
            "relationship": relation,
        }
    )


def _link_related_concepts(concepts: dict[str, dict[str, Any]], edges: list[dict[str, Any]], seen: set[tuple[str, str, str]]) -> None:
    items = list(concepts.values())
    for idx, item in enumerate(items):
        node = item["node"]
        for other in items[idx + 1 :]:
            other_node = other["node"]
            shared = set(item["sections"]).intersection(other["sections"])
            if shared:
                continue
            if _concepts_related(node["title"], other_node["title"]):
                node["relationships"] = _merge_unique(node["relationships"], [other_node["id"]])
                other_node["relationships"] = _merge_unique(other_node["relationships"], [node["id"]])
                _add_edge(edges, seen, node["id"], other_node["id"], "related")


def _backfill_related_node_ids(nodes: list[dict[str, Any]]) -> None:
    valid_ids = {node["id"] for node in nodes}
    for node in nodes:
        node["relationships"] = [rel for rel in _merge_unique([], node.get("relationships") or []) if rel in valid_ids]


def _concepts_related(left: str, right: str) -> bool:
    lt = set(_tokenize(left))
    rt = set(_tokenize(right))
    if not lt or not rt:
        return False
    return len(lt.intersection(rt)) >= 1


def _normalize_concept(value: str) -> str:
    value = clean_text(value).lower()
    value = _TERM_SYNONYMS.get(value, value)
    value = re.sub(r"\b(ch|chapter)\s*\d+[:\-]?\s*", "", value).strip()
    return value


def _clean_concept_label(value: Any) -> str | None:
    text = clean_text(str(value or ""))
    if not text:
        return None
    text = re.sub(r"\bCh\s*\d+[:\-]?\s*", "", text, flags=re.IGNORECASE).strip()
    normalized = _normalize_concept(text)
    if not normalized or normalized in _GENERIC_CONCEPT_TERMS:
        return None
    if len(normalized) < 4:
        return None
    return normalized.title()


def _concept_matches_text(label: str, text: str) -> bool:
    label_tokens = set(_tokenize(label))
    text_tokens = set(_tokenize(text))
    return bool(label_tokens and label_tokens.intersection(text_tokens))


def _merge_unique(existing: list[str], extra: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in list(existing) + list(extra):
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _merge_simple_strings(existing: list[str], extra: list[str]) -> list[str]:
    return _merge_unique(existing, extra)


def _merge_simple_objects(existing: list[dict[str, Any]], extra: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for item in list(existing) + list(extra):
        value = clean_text(item.get(key) if isinstance(item, dict) else "")
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(item)
    return out


def _merge_sources(existing: list[dict[str, Any]], extra: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for item in list(existing) + list(extra):
        fid = item.get("fragment_id")
        if not fid or fid in seen:
            continue
        seen.add(fid)
        out.append(item)
    return out


def _merge_people(existing: list[dict[str, Any]], extra: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for item in list(existing) + list(extra):
        name = clean_text(item.get("name") or item.get("speaker"))
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append(item)
    return out


def _join_parts(*parts: Any) -> str:
    cleaned = [clean_text(str(part)) for part in parts if clean_text(str(part or ""))]
    return " ".join(cleaned).strip()


def _split_sentences(text: str) -> list[str]:
    return [clean_text(part) for part in re.split(r"(?<=[.!?])\s+", clean_text(text)) if clean_text(part)]


def _next_id(prefix: str, counter: dict[str, int]) -> str:
    counter["value"] += 1
    return f"{prefix}_{counter['value']}"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{3,}", clean_text(text).lower())


def clean_text(text: str) -> str:
    return " ".join((text or "").split()).strip()
