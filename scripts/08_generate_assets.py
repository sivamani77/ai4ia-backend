import argparse
import asyncio
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Any

import edge_tts
from PIL import Image, ImageDraw, ImageFont, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STORYLINES_DIR = PROJECT_ROOT / "data" / "storylines"
GENERATED_ASSETS_DIR = PROJECT_ROOT / "data" / "generated_assets"
FRAGMENTS_CLASSIFIED_DIR = PROJECT_ROOT / "data" / "fragments_classified"
SPEAKERS_METADATA_DIR = PROJECT_ROOT / "data" / "metadata" / "speakers"

DEFAULT_VOICE = "en-GB-RyanNeural"
DEFAULT_RATE = "-6%"
DEFAULT_VOLUME = "+0%"

CARD_WIDTH = 1920
CARD_HEIGHT = 1080

BACKGROUND_COLOR = (6, 10, 18)
PANEL_COLOR = (15, 24, 38)
TEXT_COLOR = (248, 250, 252)
MUTED_TEXT_COLOR = (178, 192, 210)
ACCENT_COLOR = (84, 170, 255)
GOLD_ACCENT = (230, 184, 92)
DARK_OVERLAY = (0, 0, 0, 120)

VISIBLE_BRAND_LABEL = "AI4IA 2026"
VISIBLE_SERIES_LABEL = "Artificial Intelligence for Information Accessibility"

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
    print(message, flush=True)


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


def infer_section_copy(section: dict[str, Any]) -> dict[str, str]:
    section_type = section.get("section_type", "section")
    section_id = section.get("section_id", "")

    template = SECTION_HEADLINES.get(section_type)

    if template:
        return dict(template)

    readable = section_type.replace("_", " ").title()
    return {
        "eyebrow": section_id or "SECTION",
        "headline": readable,
        "subtitle": "A key moment in the AI4IA story."
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
    Creates a dark cinematic background with gradients, soft light and subtle geometric depth.
    No external image assets required.
    """

    image = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), BACKGROUND_COLOR)
    pixels = image.load()

    for y in range(CARD_HEIGHT):
        for x in range(CARD_WIDTH):
            nx = x / CARD_WIDTH
            ny = y / CARD_HEIGHT

            # Base cool gradient.
            r = int(6 + nx * 14 + ny * 8)
            g = int(10 + nx * 22 + ny * 12)
            b = int(18 + nx * 48 + ny * 22)

            # Soft vignette.
            dx = nx - 0.55
            dy = ny - 0.45
            dist = (dx * dx + dy * dy) ** 0.5
            vignette = max(0.35, 1.0 - dist * 1.15)

            pixels[x, y] = (
                int(r * vignette),
                int(g * vignette),
                int(b * vignette),
            )

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
        text="A Shared Future",
        xy=(130, y),
        font=headline_font,
        fill=TEXT_COLOR,
        max_width=1350,
        line_spacing=18,
    )

    y += 28

    draw_wrapped_text(
        draw=draw,
        text="AI can become a bridge to knowledge when inclusion, trust and responsibility guide its design.",
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


def build_voiceover_script(storyline: dict[str, Any]) -> list[dict[str, str]]:
    sections = storyline.get("sections", [])
    voiceover_script = []

    for section in sections:
        section_id = section.get("section_id", "")
        section_type = section.get("section_type", "")
        narration = section.get("narration", "")

        if narration:
            voiceover_script.append(
                {
                    "section_id": section_id,
                    "section_type": section_type,
                    "text": narration.strip(),
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
    fragments = []

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

    return fragments


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

    voiceover_script = build_voiceover_script(story)
    full_voiceover_text = voiceover_text_for_tts(voiceover_script)

    voiceover_script_path = asset_root / "voiceover_script.txt"
    voiceover_text_path = asset_root / "voiceover_full_text.txt"
    voiceover_audio_path = audio_dir / "voiceover_full_reference.mp3"

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
    create_closing_card(
        output_path=closing_card_path,
        title=title,
    )
    log(f"Closing card generated: {closing_card_path}")

    section_cards = []

    for section in story.get("sections", []):
        section_id = section.get("section_id", "S")
        section_type = section.get("section_type", "section")

        copy = infer_section_copy(section)

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

    tts_status = "skipped"
    section_voiceovers = []

    if not args.skip_tts:
        if not full_voiceover_text:
            log("No voice-over text found. Skipping TTS.")
            tts_status = "no_text"
        else:
            log(f"Generating reference full voice-over with edge-tts voice: {args.voice}")
            asyncio.run(
                generate_tts(
                    text=full_voiceover_text,
                    output_path=voiceover_audio_path,
                    voice=args.voice,
                    rate=args.rate,
                    volume=args.volume,
                )
            )

            for item in voiceover_script:
                section_id = item.get("section_id")
                section_type = item.get("section_type", "section")
                text = item.get("text", "").strip()

                if not section_id or not text:
                    continue

                section_audio_path = section_audio_dir / f"{section_id}_{slugify(section_type)}_voiceover.mp3"

                log(f"Generating section voice-over: {section_id}")
                asyncio.run(
                    generate_tts(
                        text=text,
                        output_path=section_audio_path,
                        voice=args.voice,
                        rate=args.rate,
                        volume=args.volume,
                    )
                )

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
            log(f"Reference full voice-over generated: {voiceover_audio_path}")
            log(f"Section voice-overs generated: {len(section_voiceovers)}")

    fragment_lookup = load_classified_fragment_lookup()
    metadata_stats = {
        "enriched_fragment_refs": 0,
        "missing_fragment_refs": set(),
    }
    enriched_story = enrich_fragment_refs(story, fragment_lookup, metadata_stats)

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
            "reference_audio_path": str(voiceover_audio_path) if tts_status == "generated" else None,
            "audio_path": None,
            "section_voiceovers": section_voiceovers,
            "important_note": "Use section_voiceovers for final export. Do not mix reference_audio_path globally over speaker clips."
        },
        "cards": {
            "title_card": str(title_card_path),
            "closing_card": str(closing_card_path),
            "section_cards": section_cards,
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


if __name__ == "__main__":
    main()
