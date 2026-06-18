# AI4IA — Complete Pipeline & Feature Flows

How every pipeline runs, what it reads/writes, and which API endpoint triggers it.
Two distinct subsystems:

- **A. Ingestion pipeline** (scripts 01–06 + 04b) — turns raw videos into a searchable
  knowledge base. Run from the CLI (or Admin), once per batch of new videos.
- **B. Generation pipeline** (scripts 07→08→09) — turns a prompt into a finished MP4.
  Triggered through the API as background **jobs**.
- **C. Intelligence features** (Phase 1–3) — read-only queries over the knowledge base,
  served live by the API (no scripts).

Current data: **86 raw videos, 85 processed, 1,664 fragments, 71 speakers.**

---

## How the API runs scripts (the job mechanism)

The generation endpoints don't run scripts inline — they spawn them as background jobs:

```
POST /api/<generate endpoint>
  └─► job_service.create_job(type, payload)        → runtime/jobs/<job_id>.json (status:queued)
  └─► script_runner.run_script_background(job_id, command, result_resolver)
         └─ spawns a daemon thread → subprocess.Popen(["python", "scripts/NN_*.py", ...])
            • cwd = PROJECT_ROOT
            • stdout+stderr streamed live → logs/jobs/<job_id>.log
            • on exit 0  → status:completed, result_path = resolver()  (e.g. newest output file)
            • on exit ≠0 → status:failed,    error = "Script failed with exit code N"
  └─► returns { job_id, status:"running" } IMMEDIATELY (non-blocking)

Client then POLLS:
  GET /api/jobs/{job_id}  → { status, progress, error, result_path, ... }
```
**FE pattern for every 🔴 generate call:** POST → get `job_id` → poll `GET /jobs/{job_id}`
until `status` is `completed` or `failed` → fetch the result via its GET endpoint.

---

# A. Ingestion Pipeline (01–06 + 04b)

Run in order. Each step's output feeds the next. **02 & 03 skip already-done files;
04 & 05 reprocess deterministically; 06 fully rebuilds the embedding matrix.**

```
                                                        TRIGGER (CLI)
01 Download    YouTube channel ──────────────────────► python scripts/01_download_videos.py
   out: data/raw/youtube/ai4ia_channel/*.mp4 + .info.json + .jpg + .vtt
        data/metadata/youtube/download_archive.txt   (enables resume; skips downloaded IDs)
        │
02 Audio       mp4 ──ffmpeg──► data/audio/*.wav         python scripts/02_extract_audio.py
   (skips if .wav exists)
        │
03 Transcribe  wav ──faster-whisper──► data/transcripts/*.json (+ .txt)  python scripts/03_transcribe.py
   (timestamped segments; skips if .json exists)
        │
04 Fragment    transcript ──► data/fragments/*.fragments.json    python scripts/04_create_fragments.py
   (splits into ~60s chunks w/ start/end; reprocesses all)
        │
05 Classify    fragments ──rule-based──► data/fragments_classified/*.classified.json   python scripts/05_classify_fragments.py
   (theme T1–T7, chapter Ch1–Ch25, keywords, quote_strength, clip_usage)
        │
04b Speakers   classified + titles + descriptions ──LLM(8B)+title-parser──►   python scripts/04b_extract_speakers.py
   out: data/metadata/speakers/{video_id}.json  (primary_speaker + roster)   [--workers 6 --model llama3.1:8b]
        │
06 Embeddings  classified ──nomic-embed-text──► fragment_embedding_matrix.npy (N×768, L2-normalized)   python scripts/06_generate_embeddings.py
   out: + classified_fragments_embeddings.jsonl (row order == matrix rows)
   (DELETES old jsonl, RE-EMBEDS ALL fragments, saves fresh matrix)
        │
        ▼
   Searchable knowledge base ready
        │
POST /api/admin/reindex   ──► FragmentStore.reload()  (live server picks up new data, no restart)
```

**Output of A:** every clip transcribed, tagged (theme/chapter/keywords/scores),
speaker-attributed (video-level), and embedded → powers everything in B and C.

**Order note:** 04b depends only on 05's output; 06 depends on 05. Run 05 → then 04b and 06
(can run in parallel). After any ingestion run, call `/api/admin/reindex`.

---

# B. Generation Pipeline (07 → 08 → 09)  — prompt to finished MP4

Three job-tracked stages. Each consumes the previous stage's output.

### Stage 07 — Storyline (the storyboard)
```
API:  POST /api/storylines/generate
      body { prompt, duration=180, style, top_k=90, final_candidates=35, fallback_only=false }
      → job → script: python scripts/07_generate_storyline.py --prompt ... --duration ... --top-k ...

FLOW (07_generate_storyline.py):
  warm_up_ollama()                         ping + preload embed model (avoids cold-load timeout)
  embed_query(prompt)                      Ollama /api/embed → /api/embeddings fallback (300s timeout, backoff)
  matrix @ query_vector                    cosine sim vs all fragments → top_k candidates
  select_balanced_candidates()             utility boost (quote_strength/clarity/usage/duration) → final_candidates
  generate_llm_storyline() [70B]           writes documentary plan: title, logline, ordered sections
                                            (each section → fragment(s) + narration); JSON, with repair + fallback
  validate fragment_ids                    warns on empty/invalid refs
  save → data/storylines/storyline_<ts>.json   { user_prompt, selected_candidates[], storyline{sections,export_timeline} }

RESULT:  GET /api/storylines/latest   or   GET /api/storylines/{id}
EVIDENCE: GET /api/storylines/{id}/evidence-map   (each candidate → playable clip + speaker)
```

### Stage 08 — Assets (cards + voice-over)
```
API:  POST /api/assets/generate-latest        (uses newest storyline automatically)
      or POST /api/assets/generate  body { storyline_path, voice, rate, volume, skip_tts }
      → job → script: python scripts/08_generate_assets.py --storyline <path> --voice ... --rate ... --volume ...

FLOW (08_generate_assets.py):
  read storyline_<ts>.json
  Pillow  → title card, per-section cards, closing card   (branded PNGs)
  edge-tts → voice-over .mp3 for narration/bridge text ONLY
            (RULE: VO never overlaps real speaker clips — their original audio stays)
  write → data/generated_assets/storyline_<ts>/  + latest_asset_manifest.json

RESULT:  GET /api/assets/latest    (the manifest: cards + voice-over file paths)
```

### Stage 09 — Export (assemble final MP4)
```
API:  POST /api/exports/generate-latest   body { output, no_subtitles }
      or POST /api/exports/generate  body { output, manifest_path, no_subtitles }
      → job → script: python scripts/09_export_video.py --output <name> [--manifest <path>] [--no-subtitles]

FLOW (09_export_video.py):
  read asset manifest + storyline
  ffmpeg trim   each chosen fragment from its source mp4 (by start/end)
  render        card segments (image + TTS audio) for intro/section/closing
  subtitles     burn title/narration/speaker-quote text per segment (unless --no-subtitles)
  ffmpeg concat all segments in storyboard order
  write → data/exports/<name>.mp4 + <name>_export_manifest.json

RESULT:
  GET /api/exports                      list exports
  GET /api/exports/{file}/stream        play in browser
  GET /api/exports/{file}/download      download
  GET /api/exports/{file}/manifest      export manifest
```

### Full generation sequence (what the FE orchestrates)
```
POST /storylines/generate ─job─► poll /jobs/{id} ─► GET /storylines/latest
        │
POST /assets/generate-latest ─job─► poll /jobs/{id} ─► GET /assets/latest
        │
POST /exports/generate-latest ─job─► poll /jobs/{id} ─► GET /exports/{file}/stream
```
All three are 🔴 (LLM/ffmpeg, minutes). `generate-latest` variants chain on the newest
prior output so the FE doesn't pass paths around.

---

# C. Intelligence Features (Phase 1–3)  — live API, no scripts

All read the cached **FragmentStore** + **EmbeddingIndex** (loaded once on first request).
Shared spine: **retrieve → number evidence [1..N] → generate w/ forced citations →
validate (drop hallucinated refs) → resolve refs to playable clips.**

### 1. Semantic Search 🟢  `POST /api/archive/semantic-search`
```
{query, top_k, filters}
  embed_query(query) [Ollama]  → normalize → matrix @ vec (cosine, all 1,664 at once)
  argsort → top_k → attach FragmentStore metadata + similarity
  → { mode:"semantic"|"keyword_fallback", results[] }
```
The retrieval primitive every feature below reuses. No LLM generation.

### 2. Insights 🔴  `POST /api/insights/generate`
```
{query, filters, limit}
  build_evidence_pool = semantic-search(query, limit) numbered [1..N]
  generate_json [70B]  → patterns[], contrasts[], recommended_clips[]  (each claim cites [refs])
  _validate(refs)      → drop dangling refs; split supported / unsupported
  evidence_coverage = supported/total
  save → data/insights/insight_<ts>.json
RESULT:  GET /insights · GET /insights/{id}
EVIDENCE: GET /insights/{id}/sources   (claim → resolved playable clips)
```

### 3. Speaker Comparison 🔴  `POST /api/compare/speakers`
```
{speakers[], topic, limit_per_speaker}
  per speaker: FragmentStore.by_facet("speaker") + (topic? semantic rank) → take N
  merge → numbered pool
  generate_json [70B] → per_speaker[], agreements[], tensions[], synthesis
  _validate → unsupported_claims[]
```

### 4. Theme Synthesis 🔴  `POST /api/synthesis/theme`
```
{theme, limit}
  semantic-search(theme, filters={theme}) [fallback: unscoped] → numbered pool
  generate_json [70B] → title, summary, key_points[], representative_clips[]
  _validate
```

### 5. Relay of Ideas 🔴  `POST /api/relay`
```
{topic, limit}
  semantic-search(topic) → DIVERSIFY (prefer one clip per distinct speaker) → numbered pool
  generate_json [70B] → relay_steps[{ref, role: opening|development|tension|synthesis, bridge}]
  resolve each step.ref → real clip (speaker, timestamps, stream_url)
```

### Supporting reads 🟢
```
GET /api/dashboard                     pipeline + archive status (Admin)
GET /api/archive/videos[/{id}]         list / one video
GET /api/archive/videos/{id}/stream    raw MP4 (binary, seekable)
GET /api/archive/themes                NESTED theme→chapters tree w/ counts
GET /api/archive/speakers              roster + counts
GET /api/archive/search                keyword fragment search
GET /api/archive/fragments/{id}        full fragment
GET /api/archive/fragments/{id}/preview  clip player payload (stream_url + start/end)
POST /api/admin/reindex                refresh store after ingestion
```

---

# End-to-end: a video's journey

```
YouTube video
  └─(01)─► raw mp4 ─(02)─► wav ─(03)─► transcript ─(04)─► fragments
        ─(05)─► classified (theme/chapter/scores) ─(04b)─► +speaker
        ─(06)─► embedded into matrix
  ══════════ now searchable ══════════
  • appears in semantic-search / insights / compare / synthesis / relay (C)
  • a prompt retrieves its best clips ─(07)─► storyboard ─(08)─► cards+VO ─(09)─► final MP4 (B)
  • every generated claim/clip traces back to THIS video's fragment + timestamp
```

# Quick reference: trigger → script → output → result endpoint

| Action | API | Script | Output | Fetch result |
|---|---|---|---|---|
| Ingest videos | (CLI/Admin) | 01–06, 04b | data/* + matrix | `/admin/reindex` |
| Storyboard | `POST /storylines/generate` 🔴 | 07 | storyline_*.json | `/storylines/latest` |
| Assets | `POST /assets/generate-latest` 🔴 | 08 | asset manifest | `/assets/latest` |
| Export MP4 | `POST /exports/generate-latest` 🔴 | 09 | exports/*.mp4 | `/exports/{f}/stream` |
| Semantic search | `POST /archive/semantic-search` 🟢 | — | — | inline |
| Insight | `POST /insights/generate` 🔴 | — | insight_*.json | `/insights/{id}/sources` |
| Compare | `POST /compare/speakers` 🔴 | — | — | inline |
| Theme synth | `POST /synthesis/theme` 🔴 | — | — | inline |
| Relay | `POST /relay` 🔴 | — | — | inline |

🟢 instant · 🔴 slow (LLM/ffmpeg, poll the job for B; inline wait for C).
