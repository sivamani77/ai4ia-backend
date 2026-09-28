"""How many talks or speakers discuss a topic, counted by code (docs/design/intent-layer-design.md §7.8).

A talk counts for a topic when at least two of its passages pass the coverage test:
similarity to the topic at or above `archive_relevance_floor`, and a topic or expanded
term in the passage. Speakers are counted from single-speaker talks only; panels are
counted as panels, never as a person. The LLM states these figures and never changes them.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

MIN_PASSAGES = 2
TOP_N = 5

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
         "sixteen seventeen eighteen nineteen").split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_word(n: int) -> str:
    """0-99 in words: 21 -> "twenty-one"."""
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens - 2] + (f"-{_ONES[ones]}" if ones else "")


def count_terms(terms: list[str]) -> list[str]:
    """Single-word terms as stems, so "regulating" also finds "regulation"; phrases as they are."""
    from app.pipeline.storyline_gen.text_utils import stem_term

    return list(dict.fromkeys(stem_term(t) if " " not in t else t for t in terms))


def count_question(topic: str) -> str:
    """The query the passages are scored against: a plain question, like the prompts the floor was tuned on."""
    return f"What did speakers say about {topic}?"


def count_topic(records: list[dict[str, Any]], scores: dict[str, float], terms: list[str], floor: float,
                exclude_terms: list[str] | None = None) -> dict[str, Any]:
    """Counts for the topic over `records` (the archive, or one year of it)."""
    from app.pipeline.storyline_gen.coverage import has_term
    from app.services.corpus_catalog import talk_title
    from app.services.speaker_registry import get_registry
    from app.services.video_naming import parse_source_video

    registry = get_registry()
    passages: Counter = Counter()
    video_of: dict[str, str] = {}
    for record in records:
        source_video = str(record.get("source_video") or "")
        if not source_video or scores.get(str(record.get("fragment_id")), 0.0) < floor:
            continue
        if any(has_term(record, t) for t in exclude_terms or []):
            continue
        if any(has_term(record, t) for t in terms):
            passages[source_video] += 1
            # The id in the file name is whole; a stored video_id can be cut at an underscore.
            video_of[source_video] = parse_source_video(source_video)[0] or str(record.get("video_id") or "")

    talks = {sv: n for sv, n in passages.items() if n >= MIN_PASSAGES}
    speakers: Counter = Counter()
    panels = 0
    for source_video, n in talks.items():
        video = registry.video(video_of.get(source_video))
        if video is None:
            continue
        if video.kind == "panel":
            panels += 1
        else:
            for name in video.names:
                speakers[name] += n
    ranked = speakers.most_common()
    named = [(name, n) for name, n in ranked
             if (registry.speakers.get(name) or {}).get("status", "confirmed") == "confirmed"]
    return {
        "talk_count": len(talks),
        "talks": set(talks),
        "speaker_count": len(speakers),
        "panel_count": panels,
        "top_speakers": named[:TOP_N],
        "unnamed_speakers": len(speakers) - len(named),
        # A hidden (review-status) speaker can be the real top; the answer must not crown someone else.
        "top_speaker_unnamed": bool(ranked) and bool(named) and ranked[0][0] != named[0][0],
        "top_talks": [(talk_title(sv), n) for sv, n in sorted(talks.items(), key=lambda x: -x[1])[:TOP_N]],
    }


def format_count_facts(facts: dict[str, Any], topic: str, year: int | None = None) -> str:
    """Plain-text facts block. Every figure here is one the narration may state."""
    lines = [f"- Topic: {topic}"]
    if year:
        lines.append(f"- Year: {year}")
    lines += [
        f"- Talks that discuss it: {facts['talk_count']} "
        f"(a talk counts when at least {MIN_PASSAGES} of its passages discuss it)",
        f"- Speakers with their own talk on it: {facts['speaker_count']}",
        f"- Panels that discuss it: {facts['panel_count']}",
    ]
    if facts["top_speakers"]:
        lines.append("- Speakers who discuss it most (passages): "
                     + "; ".join(f"{name} ({n})" for name, n in facts["top_speakers"]))
    if facts["unnamed_speakers"]:
        lines.append(f"- Speakers not named here: {facts['unnamed_speakers']}")
    if facts.get("top_speaker_unnamed"):
        lines.append("- The speaker who discusses it most is not named here")
    if facts["top_talks"]:
        lines.append("- Talks that discuss it most (passages): "
                     + "; ".join(f'"{title}" ({n})' for title, n in facts["top_talks"]))
    return "\n".join(lines)


def _line(facts_text: str, label: str) -> str | None:
    match = re.search(rf"^- {re.escape(label)}: (.+)$", facts_text or "", re.M)
    return match.group(1).strip() if match else None


def _ranked(text: str | None, quoted: bool) -> list[tuple[str, int]]:
    item = r'"(.*)" \((\d+)\)' if quoted else r"(.*) \((\d+)\)"
    return [(m.group(1), int(m.group(2))) for part in (text or "").split("; ")
            if (m := re.fullmatch(item, part.strip()))]


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _answer(facts_text: str, unit: str | None, aggregate: str | None) -> tuple[str, dict[str, Any]] | None:
    """(sentence, what must be stated for the narration to count as answering)."""
    topic, talks = _line(facts_text, "Topic"), _line(facts_text, "Talks that discuss it")
    if not topic or not talks:
        return None
    year = _line(facts_text, "Year")
    where = f"AI4IA {year}" if year else "the AI4IA archive"
    speakers = _ranked(_line(facts_text, "Speakers who discuss it most (passages)"), quoted=False)
    top_talks = _ranked(_line(facts_text, "Talks that discuss it most (passages)"), quoted=True)
    if aggregate == "rank" and unit == "speakers" and speakers:
        hidden_top = re.search(r"^- The speaker who discusses it most is not named here$", facts_text, re.M)
        if not hidden_top and (len(speakers) == 1 or speakers[0][1] > speakers[1][1]):
            return (f"Of the speakers in {where}, {speakers[0][0]} discusses {topic} the most.",
                    {"name": speakers[0][0]})
        names = [n for n, _ in speakers[:3]]
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        return f"Speakers who discuss {topic} most in {where} include {joined}.", {"name": names[0]}
    if aggregate == "rank" and unit != "speakers" and top_talks:
        if len(top_talks) == 1 or top_talks[0][1] > top_talks[1][1]:
            return f'The talk in {where} that discusses {topic} most is "{top_talks[0][0]}".', {"name": top_talks[0][0]}
    n_speakers = _line(facts_text, "Speakers with their own talk on it")
    if unit == "speakers" and n_speakers and int(n_speakers) > 0:
        n = int(n_speakers)
        verb = "discusses" if n == 1 else "discuss"
        return f"{_plural(n, 'speaker')} in {where} {verb} {topic} in their own talks.", {"number": n, "noun": "speaker"}
    n = int(re.match(r"\d+", talks).group(0))
    verb = "discusses" if n == 1 else "discuss"
    return f"{_plural(n, 'talk')} in {where} {verb} {topic}.", {"number": n, "noun": "talk"}


def answer_sentence(facts_text: str, unit: str | None, aggregate: str | None) -> str | None:
    """The sentence that answers the question, from the facts block."""
    answer = _answer(facts_text, unit, aggregate)
    return answer[0] if answer else None


def _states(text: str, need: dict[str, Any]) -> bool:
    if "name" in need:
        return need["name"].lower() in text.lower()
    n = need["number"]
    number = rf"(?:{n}|{number_word(n)})" if n < 100 else str(n)
    # The figure next to its noun ("twelve talks", "12 of the talks"), never a digit inside another word.
    return re.search(rf"(?<![\w-]){number}(?![\w-])(?:\W+\w+){{0,2}}?\W+{need['noun']}s?\b", text, re.I) is not None


def ensure_count_answer(timeline: list[dict[str, Any]], facts_text: str, unit: str | None,
                        aggregate: str | None) -> str | None:
    """07b: the first narration states the answer; if it does not, open with it. Returns the sentence added."""
    first = next((i for i in timeline if (i.get("type") or i.get("segment_type")) == "narration_broll"), None)
    answer = _answer(facts_text, unit, aggregate)
    if first is None or not answer:
        return None
    sentence, need = answer
    text = first.get("voiceover_text") or ""
    if _states(text, need):
        return None
    first["voiceover_text"] = f"{sentence} {text}".strip()
    first["subtitle_text"] = first["voiceover_text"]
    return sentence
