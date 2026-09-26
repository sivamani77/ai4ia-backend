# AI4IA video inventory

`videos_inventory.csv` — the catalogue of the AI4IA video library.
**309 videos, one row each, 36 columns.**

`videos_inventory_detailed.csv` is the same 309 videos with all 108 recorded
fields, including processing timestamps, per-stage quality diagnostics and tool
configuration. Use it for troubleshooting; use the main file for everything else.

Compiled 2026-09-11.

## Columns

| Column | Meaning |
|---|---|
| `seq` | Catalogue number, 0001–0309 |
| `youtube_id`, `title`, `youtube_url` | Source identity |
| `video_year`, `upload_date` | When it was published |
| `duration_hms`, `duration_seconds` | Running time |
| `file_size_mb` | Master file size |
| `primary_speaker` | Lead speaker |
| `speakers`, `speaker_count`, `speaker_roles`, `speaker_affiliations` | Everyone identified, their role and organisation (`;`-separated, in matching order) |
| `segment_count` | Clip-ready segments extracted |
| `segment_total_seconds`, `segment_avg_seconds`, `segment_total_words` | Volume of usable content |
| `content_coverage_seconds` | Point in the video the last segment ends — compare with `duration_seconds` to see how much of the runtime is covered |
| `top_theme`, `distinct_themes`, `all_themes` | Subject matter, with the full distribution and segment counts per theme |
| `top_chapter`, `distinct_chapters`, `all_chapters` | Conference chapter placement, same format |
| `top_keywords` | Fifteen most frequent keywords |
| `clips_speaker_soundbite`, `clips_narration_support`, `clips_evidence`, `clips_context_visual`, `clips_synthesis_candidate` | Segments suited to each editorial purpose |
| `avg_quote_strength`, `avg_clarity_score` | Editorial quality, 0–1 |
| `transcript_quality_score` | Transcript reliability, 0–1 |
| `audio_quality_status` | Audio verdict: `ok`, `denoise`, `manual_review` |
| `source_video_file` | Filename stem — the key that links to every other system |

## Library at a glance

- **309 videos**, **5117 clip-ready segments**
- **306 of 309** have identified speakers; the remaining 3 are panels where no
  single speaker applies
- By year: 2025 (80), 2024 (60), 2023 (55), 2021 (12), 2026 (4)
- Total master footage 22.5 GB across the 211 titles with recorded file sizes

## Gaps

Running time, file size, year and upload date are recorded for **211 of 309**
videos. Segment, theme, chapter, speaker and quality data are complete for all
309. The gap is a reporting one, not a processing one — every video is fully
processed and searchable; the source-file details for the other 98 live on the
machine that ingested them and were never copied into the working set.

Figures are compiled from the working data stores rather than the live production
database, which may hold newer values. One field with no equivalent here is the
archive storage key for each master file.
