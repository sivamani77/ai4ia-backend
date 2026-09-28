"""Two speakers compared (docs/design/intent-layer-design.md §7.7).

Each speaker's own clips are found, coverage-checked and selected separately with
the single-speaker rules, then balanced. A side with nothing is a stated gap; two
empty sides decline.
"""
from __future__ import annotations

import re
from typing import Any, Callable

SIDES = ("A", "B")


def side_profile(profile: dict[str, Any], speaker: str) -> dict[str, Any]:
    """The profile one side is retrieved and checked with: a single-speaker request."""
    from app.pipeline.storyline_gen.intents.context import speaker_panel_only

    side = {k: v for k, v in profile.items() if k != "compare_speakers"}
    side.update(requested_speaker=speaker, speaker_required=True, intent_type="speaker_topic_summary",
                speaker_panel_only=speaker_panel_only(speaker), evidence_requirement="speaker_source_evidence",
                strictness="high")
    return side


def select_side(
    speaker: str,
    profile: dict[str, Any],
    topic: str,
    known_speakers: list[str],
    records: list[dict[str, Any]],
    call_llm: Callable[..., str],
    top_k: int,
    target_duration: int,
    cap: int,
) -> tuple[list[dict[str, Any]], Any]:
    """(selected candidates, coverage) for one speaker, with the same steps a speaker film uses.
    The side is searched and judged as its own request ("What did A say about T?"), so the
    other speaker's name never dilutes its scores."""
    from app.core.settings import settings
    from app.pipeline.storyline_gen import coverage, intents
    from app.pipeline.storyline_gen.candidate_scoring import annotate_candidate_relevance
    from app.pipeline.storyline_gen.director import bucket_candidates
    from app.pipeline.storyline_gen.retrieval import retrieve_candidates
    from app.pipeline.storyline_gen.route import drop_excluded_speakers
    from app.pipeline.storyline_gen.selection import select_balanced_candidates, select_documentary_candidates
    from app.pipeline.storyline_gen.storyline_llm import validate_speaker_selection

    side = side_profile(profile, speaker)
    prompt = f"What did {speaker} say about {topic}?" if topic else f"What did {speaker} say?"
    side["raw_prompt"] = prompt
    retrieved = annotate_candidate_relevance(retrieve_candidates(prompt, side, top_k=top_k),
                                             prompt, side, known_speakers)
    retrieved = drop_excluded_speakers(retrieved, side)
    verdict = coverage.assess(intents.select(side, {}).NAME, side, retrieved, prompt, call_llm,
                              floor=settings.archive_relevance_floor, low=settings.coverage_judge_low,
                              records=records)
    if verdict.verdict == "not_covered":
        return [], verdict
    selected = bucket_candidates(retrieved, side)["selected_candidates"]
    if selected:
        selected = select_balanced_candidates(selected, limit=len(selected))
    selected, _ = validate_speaker_selection(selected, side)
    selected, _ = select_documentary_candidates(selected, target_duration_seconds=target_duration, cap=cap,
                                                requested_speaker=speaker)
    return selected, verdict


def combine(speakers: list[str], sides: list[list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], str | None]:
    """Both sides' candidates, each tagged with its side, and the speaker with nothing to
    show (None when both have some). Each side keeps its whole selection: 07b balances
    after validation, when it knows which clips are usable."""
    counts = [len(s) for s in sides]
    combined = []
    for tag, speaker, candidates in zip(SIDES, speakers, sides):
        for c in candidates:
            combined.append({**c, "compare_side": tag, "compare_speaker": speaker})
    gap = next((speaker for speaker, n in zip(speakers, counts) if n == 0), None) if any(counts) else None
    return combined, gap


def comparison_context(speakers: list[str], topic: str, gap: str | None) -> dict[str, Any]:
    """What the storyline and 07b are told about the two sides."""
    from app.pipeline.storyline_gen.intents.context import speaker_panel_only
    from app.services.speaker_registry import get_registry

    registry = get_registry()
    sides = []
    for tag, speaker in zip(SIDES, speakers):
        entry = {"side": tag, "speaker": speaker, "panel_only": speaker_panel_only(speaker)}
        if entry["panel_only"]:
            entry["panels"] = [v.names for _, v in registry.videos_of(speaker)]
        sides.append(entry)
    return {"sides": sides, "topic": topic, "gap": gap}


def gap_sentence(speaker: str, topic: str, archive: bool = True) -> str:
    """`archive`: the archive has nothing; otherwise it has clips, but none was usable."""
    from app.services.speaker_registry import get_registry

    # A name still in review is used for matching but never shown on screen (§6.3).
    if (get_registry().speakers.get(speaker) or {}).get("status") == "review":
        speaker = "one of the two speakers"
    if not archive:
        about = f" on {topic}" if topic else ""
        return f"None of the clips of {speaker}{about} was clear enough to include."
    if topic:
        return f"The AI4IA archive has no talk where {speaker} discusses {topic}."
    return f"The AI4IA archive has no recorded talk from {speaker} to compare."


def state_comparison_gap(timeline: list[dict[str, Any]], speaker: str, topic: str, archive: bool = True) -> bool:
    """Open the first narration by saying one side has nothing, once. True when added."""
    narrations = [i for i in timeline if (i.get("type") or i.get("segment_type")) == "narration_broll"]
    if not narrations:
        return False
    sentence = gap_sentence(speaker, topic, archive)
    first = narrations[0]
    text = first.get("voiceover_text") or ""
    if re.search(re.escape(sentence[:-1]), text):
        return False
    first["voiceover_text"] = f"{sentence} {text}".strip()
    first["subtitle_text"] = first["voiceover_text"]
    return True


SIDE_SECTIONS = {"A": "S2_FIRST", "B": "S3_SECOND"}


def place_by_side(timeline: list[dict[str, Any]], side_of: Callable[[dict[str, Any]], str | None],
                  base_of: Callable[[str], str]) -> int:
    """A speaker clip in the other speaker's section moves to its own (the contrast section may hold
    either). Items are then kept in section order. Returns the number of clips moved."""
    order: dict[str, int] = {}
    for item in timeline:
        order.setdefault(base_of(item.get("section_id") or ""), len(order))
    moved: set[int] = set()
    for item in timeline:
        if (item.get("type") or item.get("segment_type")) != "speaker_clip":
            continue
        own = SIDE_SECTIONS.get(side_of(item) or "")
        current = base_of(item.get("section_id") or "")
        if own and current in SIDE_SECTIONS.values() and current != own:
            item["section_id"] = own
            moved.add(id(item))
    if moved:
        # A moved clip goes after its new section's own items, so that section's narration still leads.
        timeline.sort(key=lambda i: (order.get(base_of(i.get("section_id") or ""), len(order)), id(i) in moved))
    return len(moved)


def balance_approved(results: list[dict[str, Any]], side_of: Callable[[dict[str, Any]], str | None]) -> list[str]:
    """07b: approved speaker clips kept equal per side; extras from the larger side are set aside.
    Returns the sides left with no approved clip."""
    by_side: dict[str, list[dict[str, Any]]] = {tag: [] for tag in SIDES}
    for r in results:
        tag = side_of(r)
        if tag in by_side and r.get("approved") and r.get("decision") == "speaker_clip":
            by_side[tag].append(r)
    counts = {tag: len(v) for tag, v in by_side.items()}
    if all(counts.values()):
        keep = min(counts.values())
        for clips in by_side.values():
            for r in sorted(clips, key=lambda r: -float(r.get("confidence") or 0))[keep:]:
                r.update(approved=False, decision="rejected", reason="set aside to keep both speakers balanced")
    return [tag for tag, n in counts.items() if n == 0]
