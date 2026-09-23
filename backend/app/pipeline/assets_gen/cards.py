from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from app.core.paths import PROJECT_ROOT
from app.pipeline.assets_gen.text_copy import make_display_title, sanitize_visible_text
from app.pipeline.assets_gen.utils import log

CINEMATIC_BACKGROUND_ASSET = PROJECT_ROOT / "config" / "cinematic_background.png"

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


def _fit_node_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    base_size: int,
    bold: bool = True,
    min_size: int = 12,
) -> tuple[ImageFont.FreeTypeFont, str]:
    """Return a (font, text) pair that fits within max_width.

    Node labels used to be sized by blind character-count truncation
    (lbl[:10]) drawn at a fixed font size — neither step measures real pixel
    width, so a long or wide word still overflows its circle/box while a
    short one wastes the space. This shrinks the font first (matching the
    measure-then-size pattern concept_card's keyword pills already use via
    draw.textbbox), and only truncates — at a word boundary, with an
    ellipsis — if the label still doesn't fit at the smallest acceptable size.
    """
    text = (text or "").strip()
    if not text:
        return find_font(min_size, bold=bold), ""

    for size in range(base_size, min_size - 1, -2):
        font = find_font(size, bold=bold)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font, text

    font = find_font(min_size, bold=bold)
    words = text.split()
    while len(words) > 1:
        words = words[:-1]
        candidate = " ".join(words) + "…"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font, candidate

    # Single word still too wide even at min_size — hard character clamp,
    # shrinking one character at a time so it's always <= max_width.
    candidate = text
    while len(candidate) > 1:
        probe = candidate[:-1] + "…"
        bbox = draw.textbbox((0, 0), probe, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font, probe
        candidate = candidate[:-1]
    return font, candidate


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
        r_c, r_n = 52, 40
        draw.ellipse((cx - r_c, cy - r_c, cx + r_c, cy + r_c), outline=ACCENT_COLOR, width=3, fill=(18, 38, 65))
        center_font, center_txt = _fit_node_label(draw, (kws[0] if kws else "AI").upper(), max_width=int(r_c * 1.7), base_size=17)
        cbbox = draw.textbbox((0, 0), center_txt, font=center_font)
        draw.text((cx - (cbbox[2] - cbbox[0]) // 2, cy - 12), center_txt, font=center_font, fill=TEXT_COLOR)
        offsets = [(-370, -10), (0, -155), (370, -10)]
        branch_labels = (kws[1:] if len(kws) > 1 else kws) + ["concept A", "concept B", "concept C"]
        node_label_max_width = int(r_n * 1.8)
        for (dx, dy_), lbl in zip(offsets, branch_labels[:3]):
            nx, ny = cx + dx, cy + dy_
            draw.line((cx, cy, nx, ny), fill=MUTED_TEXT_COLOR, width=2)
            draw.ellipse((nx - r_n, ny - r_n, nx + r_n, ny + r_n), outline=GOLD_ACCENT, width=2, fill=(22, 42, 68))
            node_font, node_txt = _fit_node_label(draw, lbl.upper(), max_width=node_label_max_width, base_size=16)
            nbbox = draw.textbbox((0, 0), node_txt, font=node_font)
            draw.text((nx - (nbbox[2] - nbbox[0]) // 2, ny - 10), node_txt, font=node_font, fill=TEXT_COLOR)

    elif visual_asset_type == "risk_map":
        cx, cy = W // 2, y + 170
        draw.ellipse((cx - 40, cy - 40, cx + 40, cy + 40), outline=(200, 80, 60), width=3, fill=(48, 12, 12))
        draw.text((cx - 26, cy - 12), "HARM", font=find_font(22, bold=True), fill=(255, 120, 100))
        ao = [(-290, -120), (-290, 120), (0, -175), (290, -120), (290, 120)]
        risk_kws = kws[:5] or ["identity", "consent", "reputation", "harm", "safety"]
        risk_r = 38
        risk_max_width = int(risk_r * 1.8)
        for (dx, dy_), lbl in zip(ao, risk_kws):
            nx, ny = cx + dx, cy + dy_
            draw.line((cx, cy, nx, ny), fill=(110, 55, 55), width=2)
            draw.ellipse((nx - risk_r, ny - risk_r, nx + risk_r, ny + risk_r), outline=(200, 80, 60), width=2, fill=(38, 12, 12))
            risk_font, risk_txt = _fit_node_label(draw, lbl.upper(), max_width=risk_max_width, base_size=14)
            rbbox = draw.textbbox((0, 0), risk_txt, font=risk_font)
            draw.text((nx - (rbbox[2] - rbbox[0]) // 2, ny - 10), risk_txt, font=risk_font, fill=(220, 160, 150))

    elif visual_asset_type == "mindmap":
        cx, cy = W // 2, y + 150
        draw.ellipse((cx - 58, cy - 30, cx + 58, cy + 30), outline=ACCENT_COLOR, width=2, fill=(14, 32, 60))
        center_txt = (kws[0] if kws else "TOPIC").upper()[:10]
        draw.text((cx - 42, cy - 12), center_txt, font=find_font(18, bold=True), fill=TEXT_COLOR)
        branch_kws = kws[1:6] if len(kws) > 1 else kws * 4
        bxs = [cx - 460, cx - 230, cx, cx + 230, cx + 460]
        mindmap_box_max_width = 116
        for bx, kw in zip(bxs, branch_kws[:5]):
            by = cy + 130
            draw.line((cx, cy + 30, bx, by - 20), fill=MUTED_TEXT_COLOR, width=1)
            draw.rectangle((bx - 68, by - 20, bx + 68, by + 20), outline=GOLD_ACCENT, width=1, fill=(18, 32, 52))
            box_font, box_txt = _fit_node_label(draw, kw.upper(), max_width=mindmap_box_max_width, base_size=14)
            bxbbox = draw.textbbox((0, 0), box_txt, font=box_font)
            draw.text((bx - (bxbbox[2] - bxbbox[0]) // 2, by - 8), box_txt, font=box_font, fill=GOLD_ACCENT)

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
    """Generate a PIL visual asset for EVERY narration_broll item.

    Sets visual_path, render_visual_path, asset_path on each narration item.
    Returns the updated list.

    This used to skip items that had a source_video, on the assumption the
    renderer would fall back to archive footage for those. It no longer does:
    archive footage is reserved exclusively for speaker_clip segments, so
    narration always needs its own visual and Step 09 hard-fails without one.
    """
    visuals_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for item in final_timeline:
        item = dict(item)
        t = item.get("type") or item.get("segment_type") or ""
        if t == "narration_broll":
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
