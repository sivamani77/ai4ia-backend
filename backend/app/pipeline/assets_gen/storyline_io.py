from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from app.core.paths import PROJECT_ROOT
from app.pipeline.assets_gen.utils import log

STORYLINES_DIR = PROJECT_ROOT / "data" / "storylines"
FRAGMENTS_CLASSIFIED_DIR = PROJECT_ROOT / "data" / "fragments_classified"
SPEAKERS_METADATA_DIR = PROJECT_ROOT / "data" / "metadata" / "speakers"


def _is_validated_storyline_payload(payload: dict[str, Any]) -> bool:
    story = payload.get("storyline") or {}
    final_timeline = (
        payload.get("final_timeline")
        or story.get("final_timeline")
        or (payload.get("validated_narrative") or {}).get("final_timeline")
        or (story.get("validated_narrative") or {}).get("final_timeline")
    )
    return bool(final_timeline) and payload.get("llm_validated", False)


def latest_storyline_path() -> Path:
    paths = sorted(STORYLINES_DIR.glob("storyline_*.json"), reverse=True)
    if not paths:
        raise FileNotFoundError(f"No storyline files found in {STORYLINES_DIR}")

    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if _is_validated_storyline_payload(payload):
            return path

    raise RuntimeError(
        f"Newest storyline is {paths[0].name}, but no validated storyline exists in "
        f"{STORYLINES_DIR}. Run 07b_validate_storyline_with_llm.py first, or pass "
        f"--storyline explicitly to use an unvalidated file."
    )


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
