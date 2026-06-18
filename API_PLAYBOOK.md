# AI4IA Knowledge Intelligence Platform — Frontend API Playbook

Complete integration reference for the FastAPI backend. Every request/response below
was captured from the live app against the current dataset (**86 raw videos, 85 processed,
1,664 fragments, 71 speakers**). Field names and shapes are exact.

---

## Conventions

- **Base URL:** `http://<host>:8077/api` (dev). All routes are prefixed with `/api`.
- **Content type:** all `POST` bodies are JSON; send `Content-Type: application/json`.
- **CORS:** server allows all origins (`*`) — fine for dev; lock down for prod.
- **Interactive docs:** `GET /docs` (Swagger) and `/openapi.json` (machine-readable schema —
  you can generate a typed client from this).
- **IDs with special chars:** some `fragment_id` / `source_video` values contain spaces,
  `#`, and smart quotes (`“”`). **Always URL-encode path params** (`encodeURIComponent`).
- **Latency classes:**
  - 🟢 **Instant** (file/index reads): everything under §1–§5, §8.
  - 🔴 **Slow 45–90s** (calls the 70B LLM on shared Ollama): insights generate, compare,
    synthesis, relay. **Show a loading state; set client timeout ≥ 120s.**
- **Error shape:** FastAPI returns `{"detail": "<message>"}` with an HTTP status code
  (404 not found, 422 validation/no-data, 502 LLM/upstream failure, 503 index unavailable).

---

## Endpoint index

| # | Method | Path | Latency | Purpose |
|---|--------|------|---------|---------|
| 1 | GET | `/health` | 🟢 | liveness |
| 1 | GET | `/dashboard` | 🟢 | archive + pipeline status |
| 2 | GET | `/archive/videos` | 🟢 | list/keyword-filter videos |
| 2 | GET | `/archive/videos/{video_id}` | 🟢 | one video |
| 2 | GET | `/archive/videos/{video_id}/stream` | 🟢 | MP4 stream (binary) |
| 2 | GET | `/archive/themes` | 🟢 | themes + chapters w/ counts |
| 2 | GET | `/archive/search` | 🟢 | keyword fragment search |
| 3 | GET | `/archive/speakers` | 🟢 | speaker roster + counts |
| 4 | POST | `/archive/semantic-search` | 🟢 | embedding similarity search |
| 5 | GET | `/archive/fragments/{id}` | 🟢 | full fragment record |
| 5 | GET | `/archive/fragments/{id}/preview` | 🟢 | clip player payload |
| 6 | POST | `/insights/generate` | 🔴 | grounded synthesis |
| 6 | GET | `/insights` | 🟢 | list insights |
| 6 | GET | `/insights/{id}` | 🟢 | one insight |
| 6 | GET | `/insights/{id}/sources` | 🟢 | evidence map (claim→clips) |
| 7 | GET | `/storylines` | 🟢 | list storylines |
| 7 | GET | `/storylines/{id}` | 🟢 | one storyline |
| 7 | GET | `/storylines/{id}/evidence-map` | 🟢 | storyline clip evidence |
| 7 | POST | `/storylines/generate` | 🔴 (job) | generate storyline |
| 8 | POST | `/compare/speakers` | 🔴 | cross-speaker comparison |
| 8 | POST | `/synthesis/theme` | 🔴 | theme synthesis |
| 8 | POST | `/relay` | 🔴 | Relay of Ideas thread |
| 9 | GET | `/jobs`, `/jobs/{id}` | 🟢 | job tracking |
| 9 | POST | `/admin/reindex` | 🟢 | refresh in-memory store |
| 10 | GET/POST | `/assets/*`, `/exports/*` | mixed | asset/video generation |

---

## 1. Health & Dashboard

### `GET /api/health`
```bash
curl -s http://localhost:8077/api/health
```
```json
{ "status": "ok", "service": "AI4IA Demo Video Studio Backend" }
```

### `GET /api/dashboard`
```bash
curl -s http://localhost:8077/api/dashboard
```
```jsonc
{
  "archive": {
    "raw_videos": 86, "audio_files": 85, "transcripts_json": 85,
    "transcripts_txt": 85, "fragment_files": 85,
    "classified_fragment_files": 85, "classified_fragments": 1664,
    "embeddings_ready": true,
    "embedding_matrix": "/abs/path/fragment_embedding_matrix.npy"
  },
  "latest": {
    "storyline": "storyline_20260615_153500.json",
    "asset_manifest": "/abs/path/latest_asset_manifest.json",
    "export": "ai4ia_demo_v4_refined.mp4"
  },
  "models": { "embedding": "nomic-embed-text:v1.5", "strong_llm": "llama3.1:70b",
              "fast_llm": "llama3.1:8b", "tts": "edge-tts", "export": "ffmpeg" },
  "paths": { "...": "absolute dirs" }
}
```
**FE use:** Admin dashboard cards. `embeddings_ready:false` ⇒ disable semantic features.

---

## 2. Archive — browse, video, themes, keyword search

### `GET /api/archive/videos?query=<str>&limit=<int>`
`query` optional (matches title/topic/id). `limit` 1–500, default 100.
```bash
curl -s "http://localhost:8077/api/archive/videos?limit=2"
```
Returns an **array**:
```jsonc
[
  {
    "video_id": "0002_SvVu6k02Hes_Zimbabwe_-_AI4IA_2025_1",
    "filename": "0002_SvVu6k02Hes_Zimbabwe_-_AI4IA_2025_1.mp4",
    "title": "Zimbabwe - AI4IA 2025 #1",
    "speaker_or_topic": "Zimbabwe - AI4IA 2025 1",
    "path": "/abs/path.mp4",
    "thumbnail": "/abs/path.jpg",        // may be null
    "duration": 1234.5,                  // seconds, may be null
    "webpage_url": "https://youtube.com/watch?v=...",  // may be null
    "stream_url": "/api/archive/videos/<video_id>/stream"
  }
]
```

### `GET /api/archive/videos/{video_id}`
Same object as above. `404 {"detail":"Video not found"}` if missing.

### `GET /api/archive/videos/{video_id}/stream`
Returns the **raw MP4** (`Content-Type: video/mp4`). Use directly as `<video src>`.
Supports seeking. Prefix `stream_url` with the API base.
```html
<video src="http://localhost:8077/api/archive/videos/<video_id>/stream#t=231.8" controls />
```

### `GET /api/archive/themes`
```jsonc
{
  "themes":   [ { "name": "AI Ethics, Governance & Policy", "count": 412 }, ... ],  // 7
  "chapters": [ { "name": "Ch 5: Open Data & Information Accessibility", "count": 88 }, ... ], // 25
  "total_themes": 7, "total_chapters": 25
}
```
**FE use:** theme/chapter filter chips with counts.

### `GET /api/archive/search` (keyword / substring)
Query params (all optional except you'll usually pass `q`):
`q, theme, chapter, speaker, source_video, clip_usage, min_duration, max_duration,
start_after, end_before, limit` (1–200, default 50).
```bash
curl -s "http://localhost:8077/api/archive/search?q=youth&limit=2"
```
Returns an **array of full fragment objects** (same shape as §5 `GET /fragments/{id}`),
sorted by match score then quote/clarity. Use this for fast text filtering;
use §4 for semantic relevance.

---

## 3. Speakers

### `GET /api/archive/speakers`
```jsonc
{
  "count": 71,
  "speakers": [
    { "speaker": "Cordell Green", "fragment_count": 105,
      "attribution": "video-level", "video_count": 1 },
    { "speaker": "James Sherer", "fragment_count": 28,
      "attribution": "video-level", "video_count": 2 }
  ]
}
```
**Important:** `attribution: "video-level"` means the speaker is attributed to the
whole video, not proven per-fragment (no diarization). Surface this caveat in the UI
when showing "who said this." Sorted by `fragment_count` desc.

---

## 4. Semantic search ⭐ (embedding similarity)

### `POST /api/archive/semantic-search`
```bash
curl -s -X POST http://localhost:8077/api/archive/semantic-search \
  -H "Content-Type: application/json" \
  -d '{"query":"open data and transparency","top_k":5,"filters":{"speaker":"James Sherer"}}'
```
**Payload:**
```jsonc
{ "query": "string (required)",
  "top_k": 20,                 // 1–100, default 20
  "filters": {                 // all optional, case-insensitive substring match
    "theme": "...", "chapter": "...", "speaker": "...",
    "edition": "AI4IA 2025", "year": 2025 } }
```
**Response:**
```jsonc
{
  "mode": "semantic",          // or "keyword_fallback" if Ollama was unreachable
  "count": 5,
  "results": [
    {
      "fragment_id": "0062_VOlpG0QlEbE_Thom_Vaughn_..._0007",
      "similarity": 0.7578,                 // cosine 0–1; higher = closer
      "source_video": "0062_VOlpG0QlEbE_Thom_Vaughn_...",
      "video_id": "VOlpG0QlEbE",
      "start": 474.48, "end": 539.94, "duration": 65.46,
      "text": "full fragment transcript ...",
      "summary": "...", "keywords": ["..."],
      "primary_theme": "AI Ethics, Governance & Policy",
      "primary_chapter": "Ch 7: ...",
      "theme_tags": [ {"theme_id","theme","score","matched_keywords"} ],
      "chapter_tags": [ {...} ],
      "clip_usage_candidates": ["speaker_soundbite", ...],
      "quote_strength": 0.8, "clarity_score": 0.7, "narration_support_score": 0.6,
      "speaker": "Thom Vaughn", "speaker_attribution": "video-level",
      "speaker_candidates": [ {"name","role","affiliation","confidence"} ],
      "year": 2025, "edition": "AI4IA 2025", "session_type": "talk",
      "stream_url": "/api/archive/videos/<source_video>/stream",
      "video_title": "..."
    }
  ]
}
```
**FE notes:** Check `mode` — if `"keyword_fallback"`, also returned is `reason`
(string) and results lack `similarity`. `503 {"detail":"Embedding matrix not available."}`
if embeddings were never generated.

---

## 5. Fragment lookup & preview

### `GET /api/archive/fragments/{fragment_id}`
Full fragment object (same fields as a §4 result, minus `similarity`).
`404 {"detail":"Fragment not found"}`.

### `GET /api/archive/fragments/{fragment_id}/preview` ⭐ (clip player)
```bash
curl -s "http://localhost:8077/api/archive/fragments/0002_SvVu6k02Hes_Zimbabwe_-_AI4IA_2025_1_0001/preview"
```
```jsonc
{
  "fragment_id": "0002_..._0001",
  "video_stream_url": "/api/archive/videos/0002_.../stream",
  "start": 0.0, "end": 63.2, "duration": 63.2,
  "text": "Could there be more relevant considering ...",
  "speaker": "Thom Vaughn",            // may be null
  "speaker_attribution": "video-level",
  "primary_theme": "...", "primary_chapter": "...",
  "year": 2025, "edition": "AI4IA 2025"
}
```
**FE use:** the canonical "play this clip" payload — load `video_stream_url`, seek to
`start`, stop at `end`. Encode the fragment_id in the path.

---

## 6. Insights — evidence-grounded synthesis ⭐

### `POST /api/insights/generate` 🔴 (~60–90s)
```bash
curl -s -X POST http://localhost:8077/api/insights/generate \
  -H "Content-Type: application/json" \
  -d '{"query":"How does AI4IA discuss youth empowerment and AI governance?","limit":15}'
```
**Payload:**
```jsonc
{ "query": "string (required)",
  "filters": { "theme":"...","speaker":"...","year":2025 },  // optional, same as §4
  "limit": 40 }   // evidence pool size, 5–100, default 40. SMALLER = FASTER. Use ~15.
```
**Response** (also persisted; reusable via GET below):
```jsonc
{
  "insight_id": "insight_20260615_120415",
  "title": "Empowering Youth through Responsible AI Governance",
  "query": "...", "filters": {},
  "summary": "2–4 sentence overview ...",
  "patterns":  [ { "statement": "Empowering youth is crucial ...",
                   "fragment_refs": [1,3,4,7], "support_count": 4 } ],
  "contrasts": [ { "statement": "...", "fragment_refs": [2,5], "support_count": 2 } ],
  "recommended_clips": [ { "fragment_id":"...","ref":1,"reason":"...",
                           "stream_url":"...","start":325.2,"end":393.3 } ],
  "unsupported_claims": [],         // claims the validator dropped (no valid citation)
  "evidence_coverage": 1.0,         // fraction of claims with ≥1 valid citation (trust signal)
  "evidence_pool": [                // the numbered clips claims cite into
    { "ref":1, "fragment_id":"...", "speaker":"...", "speaker_attribution":"video-level",
      "edition":"AI4IA 2025", "year":2025, "primary_theme":"...", "primary_chapter":"...",
      "start":325.2,"end":393.3,"duration":68.1,
      "stream_url":"...", "similarity":0.71, "text":"..." } ],
  "models": { "llm":"llama3.1:70b", "embedding":"nomic-embed-text:v1.5" },
  "created_at": "2026-06-15T12:04:15"
}
```
**FE rendering:** show `summary`, then `patterns`/`contrasts` as cards. For each claim,
map `fragment_refs` → matching `evidence_pool[].ref` to render clickable citation chips
(each opens the clip via `stream_url`+`start`). Badge `evidence_coverage` (e.g. "92% grounded").
`422 {"detail":"No fragments retrieved..."}` if query matches nothing; `502` on LLM failure.

> ⚠️ Insights generated **before** the Phase-3 facet change carry a legacy `region`
> field in `evidence_pool`; new ones use `edition`. Treat `region` as deprecated.

### `GET /api/insights`
```jsonc
{ "count": 3, "insights": [
  { "insight_id":"insight_20260615_120415", "title":"...", "query":"...",
    "created_at":"...", "evidence_coverage":1.0 } ] }
```

### `GET /api/insights/{insight_id}`
Full insight object (identical to generate response). `404` if not found.

### `GET /api/insights/{insight_id}/sources` ⭐ (evidence map)
```jsonc
{
  "insight_id":"insight_20260615_120415",
  "evidence_coverage":1.0,
  "patterns": [
    { "statement":"Empowering youth is crucial ...",
      "sources": [
        { "fragment_id":"...","speaker":null,"speaker_attribution":"video-level",
          "edition":null,"year":2025,"video_stream_url":"...",
          "start":325.24,"end":393.34,"duration":68.1,"text":"..." } ] } ],
  "contrasts": [ { "statement":"...","sources":[ ... ] } ]
}
```
**FE use:** the "show me the evidence" view — each claim already resolved to playable clips.
Prefer this over manually joining `fragment_refs` to `evidence_pool`.

---

## 7. Storylines

### `GET /api/storylines`
Array of summaries: `{id, filename, path, title, logline, sections, created_at, target_duration_seconds}`.

### `GET /api/storylines/{storyline_id}`
Full storyline JSON (`user_prompt`, `selected_candidates[]`, `storyline{...}`, etc.).

### `GET /api/storylines/{storyline_id}/evidence-map` ⭐
```jsonc
{
  "user_prompt": "...",
  "candidate_count": 35, "resolved_count": 35,
  "evidence": [
    { "fragment_id":"...","speaker":"Krishna Sunkara","speaker_attribution":"video-level",
      "edition":"AI4IA 2025","year":2025,"primary_theme":"...","primary_chapter":"...",
      "video_stream_url":"/api/archive/videos/<src>/stream",
      "start":718.0,"end":776.5,"duration":58.5,"summary":"...","resolved":true } ]
}
```
`resolved:false` ⇒ that candidate no longer matches a fragment in the store.

### `POST /api/storylines/generate` 🔴 (async job)
```bash
curl -s -X POST http://localhost:8077/api/storylines/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"youth and responsible AI","duration":180,"style":"documentary","top_k":90,"final_candidates":35}'
```
Returns immediately: `{ "job_id":"job_...","status":"running","message":"..." }`.
**Poll** `GET /jobs/{job_id}` until `status:"completed"`, then fetch `GET /storylines/latest`.

---

## 8. Comparison · Theme synthesis · Relay ⭐ (Phase 3) 🔴

All three call the 70B model (~45–60s). Same citation contract as insights.

### `POST /api/compare/speakers`
```bash
curl -s -X POST http://localhost:8077/api/compare/speakers \
  -H "Content-Type: application/json" \
  -d '{"speakers":["James Sherer","Stefaan G Verhulst"],"topic":"AI ethics and governance","limit_per_speaker":4}'
```
**Payload:** `{ "speakers": ["name", ...] (≥2, required), "topic": "string|null",
"limit_per_speaker": 6 (1–20) }`. Use names from §3.
**Response:**
```jsonc
{
  "topic":"AI ethics and governance",
  "speakers_requested":["James Sherer","Stefaan G Verhulst"],
  "speakers_missing":[],                  // names with no fragments
  "per_speaker":[ { "speaker":"James Sherer","position":"...","fragment_refs":[1],
                    "support_count":1 } ],
  "agreements":[ { "statement":"...","fragment_refs":[1,4],"support_count":2 } ],
  "tensions": [ { "statement":"...","fragment_refs":[2,5],"support_count":2 } ],
  "synthesis":"2–3 sentence cross-speaker takeaway",
  "unsupported_claims":[ ... ],
  "evidence_pool":[ { "ref":1,"fragment_id":"...","speaker":"...","start":..,"end":..,
                      "stream_url":"...","text":"..." } ],
  "models":{"llm":"llama3.1:70b"}, "created_at":"..."
}
```

### `POST /api/synthesis/theme`
```bash
curl -s -X POST http://localhost:8077/api/synthesis/theme \
  -H "Content-Type: application/json" \
  -d '{"theme":"AI Ethics, Governance & Policy","limit":15}'
```
**Payload:** `{ "theme":"string (required)","limit":20 (5–60) }` (use a theme name from §2).
**Response:**
```jsonc
{
  "theme":"...","title":"Building Trust in AI: A Global Imperative",
  "summary":"3–4 sentence synthesis",
  "key_points":[ { "point":"...","fragment_refs":[1,2],"support_count":2 } ],
  "representative_clips":[ { "fragment_id":"...","speaker":"...","start":..,"end":..,
                            "stream_url":"...","text":"...","reason":"..." } ],
  "unsupported_claims":[ ... ],
  "evidence_pool":[ ... ], "models":{...}, "created_at":"..."
}
```

### `POST /api/relay` ⭐ (Relay of Ideas)
```bash
curl -s -X POST http://localhost:8077/api/relay \
  -H "Content-Type: application/json" \
  -d '{"topic":"youth empowerment in AI","limit":15}'
```
**Payload:** `{ "topic":"string (required)","limit":20 (5–60) }`.
**Response:**
```jsonc
{
  "topic":"youth empowerment in AI",
  "relay_title":"Empowering Youth in AI for a Better Future",
  "relay_narrative":"short paragraph describing the through-line",
  "relay_steps":[
    { "fragment_id":"...","speaker":"Tendai Zengeni","speaker_attribution":"video-level",
      "edition":"AI4IA 2025","primary_theme":"...","video_stream_url":"...",
      "start":414.0,"end":467.0,"duration":53.0,"text":"...",
      "role_in_relay":"opening",        // opening | development | tension | synthesis
      "bridge":"1-sentence connector" } ],
  "step_count":4, "distinct_speakers":3, "edition":"AI4IA 2025",
  "models":{...}, "created_at":"..."
}
```
**FE use:** render `relay_steps` in order as a vertical timeline/story; each step plays a
real clip. `role_in_relay` drives styling (opening/tension/etc.). Known cosmetic caveat:
the `bridge` prose may occasionally name the wrong speaker — trust `speaker`, not the prose.

---

## 9. Jobs & Admin

### `GET /api/jobs` → array of job objects. `GET /api/jobs/{job_id}` → one job.
Job object: `{ job_id, type, status (queued|running|completed|failed), progress,
created_at, started_at, completed_at, payload, log_path, result_path, error }`.

### `POST /api/admin/reindex`
`{ "status":"ok","fragments":1664 }`. Call after a pipeline run so new fragments appear
without restarting the server.

---

## 10. Assets & Exports (existing pipeline wrappers)

- `GET /api/assets/latest` — latest asset manifest (cards, voice-over files).
- `POST /api/assets/generate` / `POST /api/assets/generate-latest` — 🔴 job-based asset gen.
- `GET /api/exports` — list exported MP4s.
- `POST /api/exports/generate` / `POST /api/exports/generate-latest` — 🔴 job-based render.
- `GET /api/exports/{filename}/stream` — play final MP4.
- `GET /api/exports/{filename}/download` — download MP4.
- `GET /api/exports/{filename}/manifest` — export manifest JSON.

---

## Suggested persona → endpoint mapping

| Persona | Primary endpoints |
|---|---|
| **General User** | `/archive/themes`, `/archive/semantic-search`, `/insights/generate`, `/relay`, `/storylines/generate` → `/exports/*` |
| **Expert / Researcher** | `/archive/search`, `/archive/semantic-search`, `/archive/speakers`, `/compare/speakers`, `/synthesis/theme`, `/insights/{id}/sources`, `/storylines/{id}/evidence-map`, `/archive/fragments/{id}/preview` |
| **Admin** | `/dashboard`, `/jobs`, `/admin/reindex`, `/assets/*`, `/exports/*` |

## Integration tips

1. **Generate a typed client** from `GET /openapi.json` (e.g. `openapi-typescript`) — avoids
   hand-copying these shapes.
2. **Stream URLs are relative** (`/api/archive/...`) — prepend your API base in the FE.
3. **For 🔴 endpoints:** disable the submit button + show progress; timeout ≥ 120s; allow retry.
4. **Citations pattern is uniform** across insights/compare/synthesis/relay: claims carry
   `fragment_refs` (ints) that index into the response's `evidence_pool[].ref`. Build one
   reusable "evidence chip" component.
5. **Always `encodeURIComponent`** path params (fragment/video/storyline IDs).
