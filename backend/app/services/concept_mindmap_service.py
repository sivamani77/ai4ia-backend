import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.core.paths import STORYLINES_DIR
from app.services import fragment_store


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


def generate_concept_mindmap(query: str, storyline_id: str | None = None) -> dict[str, Any]:
    payload, source_path = _resolve_storyline(query, storyline_id)
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


def _resolve_storyline(query: str, storyline_id: str | None) -> tuple[dict[str, Any], Path]:
    if storyline_id:
        direct = STORYLINES_DIR / f"{storyline_id}.json"
        if not direct.exists():
            raise FileNotFoundError(f"Storyline '{storyline_id}' not found")
        return json.loads(direct.read_text(encoding="utf-8")), direct

    candidates = sorted(STORYLINES_DIR.glob("storyline_*.json"), reverse=True)
    if not candidates:
        raise FileNotFoundError("No storyline files found")

    best_path = None
    best_score = -1.0
    query_tokens = set(_tokenize(query))
    query_norm = clean_text(query).lower()

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

    if not best_path or best_score < 0.35:
        raise FileNotFoundError(
            "No matching storyline found for this query. Generate a storyline first or pass storyline_id."
        )

    return json.loads(best_path.read_text(encoding="utf-8")), best_path


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
