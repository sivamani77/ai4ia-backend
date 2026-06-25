# Frontend API Integration Guide

## Base Setup — Two Modes

```js
// utils/api.js
import axios from 'axios'

function getSessionId() {
  let id = localStorage.getItem('ai4ia_session_id')
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem('ai4ia_session_id', id)
  }
  return id
}

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,  // http://localhost:8077/api
  headers: {
    'X-API-Key':    import.meta.env.VITE_API_KEY || '',
    'X-Session-ID': getSessionId(),
  }
})

export default api
```

**Frontend `.env`:**
```bash
# DEV MODE — both blank, no auth enforced
VITE_API_BASE_URL=http://localhost:8077/api
VITE_API_KEY=

# PROD MODE — fill both
VITE_API_BASE_URL=https://your-backend.com/api
VITE_API_KEY=your-secret-key-here
```

> The two headers go on **every** request automatically. Nothing else changes per-request.

---

## Mode Behaviour Table

| Setting | Dev Mode | Prod Mode |
|---|---|---|
| `VITE_API_KEY` | blank | filled |
| `X-API-Key` header | sent but ignored by backend | validated, `403` if wrong |
| `X-Session-ID` header | sent, stored on job | sent, stored on job |
| Job list filtering | returns your session's jobs | returns your session's jobs |
| CORS | any origin allowed | only listed origins allowed |

---

## All Endpoints — Exact curl + Response

### Health Check (no auth required)

```bash
curl http://localhost:8077/api/health
```

```json
{
  "status": "ok",
  "service": "AI4IA Demo Video Studio Backend",
  "redis_reachable": true,
  "supabase_configured": true,
  "worker_mode": "celery"
}
```

`worker_mode` is either `"celery"` (Redis up, jobs queued properly) or `"thread_fallback"` (Redis down, jobs run in process — lost on restart).

> **Frontend:** Show a warning banner when `worker_mode === "thread_fallback"`.

---

### Create a Video Job

```bash
curl -X POST http://localhost:8077/api/video-jobs \
  -H "X-API-Key: your-key" \
  -H "X-Session-ID: 550e8400-e29b-41d4-a716-446655440000" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A documentary about AI in agriculture",
    "target_duration": 150,
    "style": "documentary",
    "validation_mode": "iterative_llm",
    "visual_mode": "generated_assets",
    "subtitle_mode": "external",
    "max_validation_iterations": 3,
    "burn_subtitles": false
  }'
```

**Response `202`:**
```json
{
  "job_id": "vjob_20260625_143022_a1b2c3d4",
  "status": "queued",
  "queue_position": 2,
  "message": "Job created. Position in queue: 2"
}
```

**Error `429` — quota hit (too many active jobs):**
```json
{
  "detail": "You already have 4 active job(s). Wait for one to complete before submitting another."
}
```

**Request field reference:**

| Field | Type | Values | Default |
|---|---|---|---|
| `prompt` | string | 3–2000 chars | required |
| `target_duration` | int | 30–600 (seconds) | `150` |
| `style` | string | `"documentary"` | `"documentary"` |
| `validation_mode` | string | `"iterative_llm"` | `"iterative_llm"` |
| `visual_mode` | string | `"generated_assets"` | `"generated_assets"` |
| `subtitle_mode` | string | `"external"` | `"external"` |
| `max_validation_iterations` | int | 1–5 | `3` |
| `burn_subtitles` | bool | `true` / `false` | `false` |

---

### List Your Jobs (session-scoped)

```bash
curl "http://localhost:8077/api/video-jobs?limit=20&offset=0" \
  -H "X-API-Key: your-key" \
  -H "X-Session-ID: 550e8400-e29b-41d4-a716-446655440000"
```

**Response `200`:**
```json
{
  "jobs": [
    {
      "job_id": "vjob_20260625_143022_a1b2c3d4",
      "status": "running",
      "current_stage": "07_storyline",
      "progress_percent": 17,
      "stage_progress_percent": 45,
      "message": "Generating storyline...",
      "prompt": "A documentary about AI in agriculture",
      "created_at": "2026-06-25T14:30:22",
      "started_at": "2026-06-25T14:30:23",
      "updated_at": "2026-06-25T14:31:05",
      "completed_at": null,
      "error": null,
      "stages": [
        { "stage": "input_validation",      "label": "Input validation",                     "status": "completed", "progress_percent": 100, "summary": "" },
        { "stage": "07_storyline",           "label": "Initial storyline generation",         "status": "running",   "progress_percent": 45,  "summary": "Generating storyline..." },
        { "stage": "07b_validate_storyline", "label": "LLM editorial validation",             "status": "pending",   "progress_percent": 0,   "summary": "" },
        { "stage": "08_assets",              "label": "Voiceover and visual asset generation","status": "pending",   "progress_percent": 0,   "summary": "" },
        { "stage": "09_export",              "label": "Final video export",                   "status": "pending",   "progress_percent": 0,   "summary": "" },
        { "stage": "10_validate_export",     "label": "Export validation",                    "status": "pending",   "progress_percent": 0,   "summary": "" },
        { "stage": "11_subtitles",           "label": "Subtitle finalization",                "status": "pending",   "progress_percent": 0,   "summary": "" }
      ],
      "video_url": null,
      "result_available": false,
      "warnings": [],
      "queue_position": null
    }
  ],
  "total": 1,
  "limit": 20,
  "offset": 0
}
```

> Without `X-Session-ID` header → returns **all** jobs (no filter).
> With `X-Session-ID` → returns only that session's jobs.

**Job `status` values:**

| Status | Meaning |
|---|---|
| `queued` | Waiting in Redis queue |
| `running` | Pipeline executing |
| `completed` | Done, video ready |
| `failed` | Pipeline error |
| `cancelled` | Cancelled by user |

**Stage `status` values:** `pending` → `running` → `completed` / `failed`

**Pipeline stages in order:**

| Stage key | Label |
|---|---|
| `input_validation` | Input validation |
| `07_storyline` | Initial storyline generation |
| `07b_validate_storyline` | LLM editorial validation |
| `08_assets` | Voiceover and visual asset generation |
| `09_export` | Final video export |
| `10_validate_export` | Export validation |
| `11_subtitles` | Subtitle finalization |

---

### Get Single Job Status

```bash
curl http://localhost:8077/api/video-jobs/vjob_20260625_143022_a1b2c3d4 \
  -H "X-API-Key: your-key" \
  -H "X-Session-ID: 550e8400-e29b-41d4-a716-446655440000"
```

Same shape as one item in the list above. Use this to poll for status if not using SSE.

---

### SSE Real-time Progress Stream (recommended over polling)

```bash
curl -N "http://localhost:8077/api/video-jobs/vjob_20260625_143022_a1b2c3d4/events" \
  -H "X-API-Key: your-key" \
  -H "X-Session-ID: 550e8400-e29b-41d4-a716-446655440000"
```

**Stream output:**
```
retry: 2000

id: 1
data: {"id":1,"event_type":"created","stage":"queued","payload":{"prompt":"..."}}

id: 2
data: {"id":2,"event_type":"progress","stage":"input_validation","payload":{"stage":"input_validation","status":"running","progress":50,"global_progress":1,"message":"Validating..."}}

id: 3
data: {"id":3,"event_type":"progress","stage":"07_storyline","payload":{"stage":"07_storyline","status":"running","progress":30,"global_progress":19,"message":"Generating storyline..."}}

event: done
data: {"status":"completed"}
```

**Frontend JS (using `@microsoft/fetch-event-source` for custom headers):**

```js
import { fetchEventSource } from '@microsoft/fetch-event-source'

fetchEventSource(`/api/video-jobs/${jobId}/events`, {
  headers: {
    'X-API-Key':    import.meta.env.VITE_API_KEY,
    'X-Session-ID': getSessionId(),
  },
  onmessage(event) {
    if (event.event === 'done') {
      const { status } = JSON.parse(event.data)
      // status = 'completed' | 'failed' | 'cancelled'
      // fetch /result now
      return
    }
    const data = JSON.parse(event.data)
    // data.event_type: 'created' | 'progress' | 'failed' | 'cancelled' | 'completed'
    // data.payload.global_progress: 0–100
    // data.payload.stage: which stage is running
    // data.payload.message: human-readable status message
    updateProgressBar(data.payload.global_progress)
    updateStageLabel(data.payload.stage, data.payload.message)
  },
  onerror(err) {
    console.error('SSE error', err)
  }
})
```

> `retry: 2000` means the browser auto-reconnects after 2 seconds if the connection drops.
> Pass `?last_event_id=3` to resume from where you left off after a reconnect.

---

### Cancel a Job

```bash
curl -X DELETE http://localhost:8077/api/video-jobs/vjob_20260625_143022_a1b2c3d4 \
  -H "X-API-Key: your-key" \
  -H "X-Session-ID: 550e8400-e29b-41d4-a716-446655440000"
```

**Response `202`:**
```json
{ "job_id": "vjob_20260625_143022_a1b2c3d4", "status": "cancelled" }
```

**Error `409` — job already in terminal state:**
```json
{ "detail": { "detail": "Job is not cancellable", "status": "completed" } }
```

> Disable the cancel button once job status is `completed`, `failed`, or `cancelled`.

---

### Get Final Result (after job completed)

```bash
curl http://localhost:8077/api/video-jobs/vjob_20260625_143022_a1b2c3d4/result \
  -H "X-API-Key: your-key" \
  -H "X-Session-ID: 550e8400-e29b-41d4-a716-446655440000"
```

**Response:**
```json
{
  "job_id": "vjob_20260625_143022_a1b2c3d4",
  "status": "completed",
  "title": "AI in Agriculture",
  "video_url": "/api/video-jobs/vjob_.../video/stream",
  "video_stream_url": "/api/video-jobs/vjob_.../video/stream",
  "video_download_url": "/api/video-jobs/vjob_.../video/download",
  "subtitles": {
    "srt": "/api/video-jobs/vjob_.../subtitles/srt",
    "vtt": "/api/video-jobs/vjob_.../subtitles/vtt"
  },
  "manifest_url": "/api/video-jobs/vjob_.../manifest",
  "assets_url": "/api/video-jobs/vjob_.../assets",
  "summary": {
    "speaker_clips": 8,
    "ai_voiceovers": 5,
    "generated_visuals": 5,
    "subtitle_segments": 42
  },
  "warnings": [],
  "error": null
}
```

Use `video_stream_url` in a `<video>` tag. Use `video_download_url` for a download button. Use `subtitles.vtt` for subtitle track.

```html
<video controls>
  <source src="/api/video-jobs/vjob_.../video/stream" type="video/mp4" />
  <track kind="subtitles" src="/api/video-jobs/vjob_.../subtitles/vtt" srclang="en" default />
</video>
```

---

### Semantic Search

```bash
curl -X POST http://localhost:8077/api/archive/semantic-search \
  -H "X-API-Key: your-key" \
  -H "X-Session-ID: 550e8400-e29b-41d4-a716-446655440000" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "AI in agriculture",
    "top_k": 10,
    "filters": { "theme": "AI in Agriculture" }
  }'
```

**Response:**
```json
{
  "mode": "semantic",
  "count": 10,
  "results": [
    {
      "fragment_id": "0045_abc123_..._0012",
      "text": "...",
      "similarity": 0.8923,
      "theme": "AI in Agriculture",
      "chapter": "Precision Farming",
      "start": 142.3,
      "end": 198.7,
      "duration": 56.4,
      "video_id": "abc123"
    }
  ]
}
```

`mode` is `"semantic"` (ChromaDB used) or `"keyword_fallback"` (Ollama unreachable, fell back to text search).

---

## Error Codes — Complete Reference

| Code | When | Frontend action |
|---|---|---|
| `400` | Malformed `X-Session-ID` (not a UUID4) | Clear `localStorage` session ID and regenerate |
| `400` | Path outside allowed directory | Do not use user-supplied file paths |
| `403` | Wrong or missing `X-API-Key` | Show auth error, check env config |
| `404` | Job ID not found | Show "job not found" message |
| `409` | Cancelling a completed / failed / cancelled job | Disable cancel button once job is terminal |
| `422` | Request body validation failed | Show field-level errors from `detail` |
| `429` | Quota: too many active jobs per session | Show "please wait for a job to finish" |
| `500` | Server error | Show generic error, log `detail` for debugging |
| `502` | Upstream failure (Ollama / Supabase down) | Show "service temporarily unavailable" |
| `503` | Embedding index not available | Semantic search degraded, show keyword fallback notice |

---

## Recommended Job Lifecycle Flow

```
User submits prompt
        │
        ▼
POST /api/video-jobs
        │
        ├── 429 → show quota message, stop
        └── 202 → store job_id, show queue_position
                │
                ▼
        Open SSE /api/video-jobs/{job_id}/events
                │
                ├── event_type=progress   → update progress bar + stage label
                ├── event_type=failed     → show error, close stream
                ├── event_type=cancelled  → show cancelled state, close stream
                └── event: done (status=completed)
                        │
                        ▼
                GET /api/video-jobs/{job_id}/result
                        │
                        ▼
                Show <video> with video_stream_url
                Show download button with video_download_url
                Show subtitle toggle with subtitles.vtt
```

---

## Static File URLs — Removed

These direct URLs **no longer work:**
```
❌  /exports/filename.mp4
❌  /generated-assets/filename.png
```

Use the API endpoints instead:
```
✅  GET /api/video-jobs/{job_id}/video/stream     — video file
✅  GET /api/video-jobs/{job_id}/video/download   — video download
✅  GET /api/video-jobs/{job_id}/assets           — asset manifest
✅  GET /api/video-jobs/{job_id}/subtitles/vtt    — VTT subtitles
✅  GET /api/video-jobs/{job_id}/subtitles/srt    — SRT subtitles
✅  GET /api/video-jobs/{job_id}/result           — final result + all URLs
```

---

## YouTube Channel Ingestion API

Ingests new videos from the YouTube channel through a 7-stage pipeline:
`discover → download → extract_audio → transcribe → create_fragments → classify → embed`

**Key behaviours:**
- Returns immediately with a `job_id` — all stages run in background
- Only one sync job can run at a time (409 if already active)
- Per-video progress tracked independently in `ingest_status` table
- SSE stream closes automatically when job reaches terminal state

**Typical timings (per video):**

| Stage | Time | Notes |
|---|---|---|
| discover | 5–15s | yt-dlp flat playlist scan |
| download | 2–10 min/video | depends on video length and bandwidth |
| extract_audio | 30–90s/video | ffmpeg WAV extraction |
| transcribe | 3–15 min/video | Whisper model size dependent |
| create_fragments | 10–30s/video | text splitting |
| classify | 1–5 min/video | LLM classification via Ollama |
| embed | 2–10 min total | batch embedding + ChromaDB import |

**Total for a new video: ~15–35 minutes end-to-end**

---

### POST /api/ingest/sync — Trigger full channel sync

```bash
curl -s -X POST https://corbin-submucronated-repletely.ngrok-free.dev/api/ingest/sync \
  -H "Content-Type: application/json" \
  -d '{
    "channel_url": "https://www.youtube.com/@ai4iaconference861/videos",
    "video_limit": 2
  }'
```

> `channel_url` — optional, defaults to the value in settings  
> `video_limit` — optional, limits how many NEW videos to download per sync. Omit for full sync.

**Response `202` — job queued immediately (~50ms):**
```json
{
  "job_id": "ijob_20260625_143022_a1b2c3d4",
  "status": "queued",
  "message": "Channel sync job queued"
}
```

**Response `409` — another sync already running:**
```json
{
  "detail": {
    "message": "A sync job is already active",
    "active_job_id": "ijob_20260625_143022_a1b2c3d4"
  }
}
```

> **Frontend:** Store `job_id` and immediately open the SSE stream. Disable the sync button and show `active_job_id` if 409 received.

---

### GET /api/ingest/jobs — List all sync jobs

```bash
curl -s "https://corbin-submucronated-repletely.ngrok-free.dev/api/ingest/jobs?limit=20"
```

**Response `200` (~100ms):**
```json
{
  "count": 2,
  "jobs": [
    {
      "job_id": "ijob_20260625_143022_a1b2c3d4",
      "status": "completed",
      "progress": 100,
      "message": "Channel sync completed",
      "channel_url": "https://www.youtube.com/@ai4iaconference861/videos",
      "video_limit": 2,
      "current_stage": "completed",
      "created_at": "2026-06-25T14:30:22Z",
      "started_at": "2026-06-25T14:30:23Z",
      "completed_at": "2026-06-25T15:02:11Z",
      "summary": {
        "new_videos_found": 2,
        "downloaded": 2,
        "audio_extracted": 2,
        "transcribed": 2,
        "fragmented": 2,
        "classified": 2,
        "embedded": 2
      },
      "error": null
    }
  ]
}
```

**Job `status` values:**

| Status | Meaning |
|---|---|
| `queued` | Waiting for worker |
| `running` | Pipeline executing |
| `completed` | All stages done |
| `failed` | A stage failed — check `error` |
| `cancelled` | Manually cancelled |

---

### GET /api/ingest/jobs/{job_id} — Single job status

```bash
curl -s https://corbin-submucronated-repletely.ngrok-free.dev/api/ingest/jobs/ijob_20260625_143022_a1b2c3d4
```

**Response `200` while running (~100ms):**
```json
{
  "job_id": "ijob_20260625_143022_a1b2c3d4",
  "status": "running",
  "progress": 42,
  "message": "Downloading 2 new videos",
  "channel_url": "https://www.youtube.com/@ai4iaconference861/videos",
  "video_limit": 2,
  "current_stage": "download",
  "created_at": "2026-06-25T14:30:22Z",
  "started_at": "2026-06-25T14:30:23Z",
  "completed_at": null,
  "summary": null,
  "error": null
}
```

**Response `200` on failure:**
```json
{
  "job_id": "ijob_20260625_143022_a1b2c3d4",
  "status": "failed",
  "progress": 35,
  "message": null,
  "current_stage": "transcribe",
  "error": {
    "stage": "transcribe",
    "message": "03_transcribe.py failed (rc=1): CUDA out of memory",
    "details": "Traceback (most recent call last): ..."
  },
  "summary": null
}
```

**`progress` field:** 0–100 global progress computed from stage weights:

| Stage | Weight |
|---|---|
| discover | 5% |
| download | 30% |
| extract_audio | 10% |
| transcribe | 25% |
| create_fragments | 12% |
| classify | 10% |
| embed | 8% |

---

### GET /api/ingest/jobs/{job_id}/events — SSE stream

```bash
curl -N "https://corbin-submucronated-repletely.ngrok-free.dev/api/ingest/jobs/ijob_20260625_143022_a1b2c3d4/events"

# Resume after reconnect — pass last received event id
curl -N "https://corbin-submucronated-repletely.ngrok-free.dev/api/ingest/jobs/ijob_20260625_143022_a1b2c3d4/events?after=17"
```

**Stream output — events in order:**
```
id: 1
event: created
data: {"channel_url": "https://www.youtube.com/@ai4iaconference861/videos", "video_limit": 2}

id: 2
event: progress
data: {"stage": "discover", "stage_status": "running", "progress": 5, "message": "Fetching channel playlist"}

id: 3
event: channel_scanned
data: {"total_on_channel": 142}

id: 4
event: discover_completed
data: {"new_videos": [{"video_id": "abc123", "title": "AI in Farming", "index": 141}, {"video_id": "def456", "title": "Precision Irrigation", "index": 142}], "skipped_count": 140, "total_on_channel": 142}

id: 5
event: progress
data: {"stage": "download", "stage_status": "running", "progress": 12, "message": "Downloading 2 new videos"}

id: 6
event: video_downloaded
data: {"video_id": "abc123", "title": "AI in Farming"}

id: 7
event: video_downloaded
data: {"video_id": "def456", "title": "Precision Irrigation"}

id: 8
event: progress
data: {"stage": "extract_audio", "stage_status": "running", "progress": 47, "message": "Extracting audio from 2 videos"}

id: 9
event: progress
data: {"stage": "transcribe", "stage_status": "running", "progress": 57, "message": "Transcribing 2 audio files"}

id: 10
event: progress
data: {"stage": "create_fragments", "stage_status": "running", "progress": 82, "message": "Creating fragments for 2 transcripts"}

id: 11
event: fragments_created
data: {"video_id": "abc123", "fragment_count": 38}

id: 12
event: fragments_created
data: {"video_id": "def456", "fragment_count": 24}

id: 13
event: progress
data: {"stage": "classify", "stage_status": "running", "progress": 92, "message": "Classifying fragments for 2 videos"}

id: 14
event: progress
data: {"stage": "embed", "stage_status": "running", "progress": 96, "message": "Generating embeddings for 2 videos"}

id: 15
event: completed
data: {"new_videos_found": 2, "downloaded": 2, "audio_extracted": 2, "transcribed": 2, "fragmented": 2, "classified": 2, "embedded": 2, "message": "Channel sync completed"}
```

> Stream **closes automatically** after `completed`, `failed`, or `cancelled` event.  
> Poll interval: 1.5 seconds between DB checks.  
> Timeout: stream closes with `event: timeout` after ~5 min of silence.

**Event types reference:**

| Event type | When | Key data fields |
|---|---|---|
| `created` | Job created | `channel_url`, `video_limit` |
| `progress` | Stage state changes | `stage`, `stage_status`, `progress` (0–100 global), `message` |
| `channel_scanned` | After yt-dlp playlist fetch | `total_on_channel` |
| `discover_completed` | New videos identified | `new_videos[]`, `skipped_count`, `total_on_channel` |
| `video_downloaded` | Each video download done | `video_id`, `title` |
| `video_download_failed` | A video failed to download | `video_id` |
| `fragments_created` | Fragment file written | `video_id`, `fragment_count` |
| `failed` | Job failed | `stage`, `message` |
| `cancelled` | Job cancelled | `message` |
| `completed` | All stages done | full summary counts |
| `timeout` | SSE idle timeout (~5 min) | — reconnect with `?after=<last_id>` |

**Frontend JS:**
```js
import { fetchEventSource } from '@microsoft/fetch-event-source'

const TERMINAL = new Set(['completed', 'failed', 'cancelled'])

function streamIngestJob(jobId, { onProgress, onVideoFound, onDone, onError }) {
  let lastEventId = 0

  const connect = () => fetchEventSource(`/api/ingest/jobs/${jobId}/events?after=${lastEventId}`, {
    onmessage(ev) {
      if (ev.id) lastEventId = Number(ev.id)
      const data = JSON.parse(ev.data || '{}')

      if (ev.event === 'progress') {
        onProgress({ stage: data.stage, progress: data.progress, message: data.message })
      } else if (ev.event === 'discover_completed') {
        onVideoFound(data.new_videos)                    // array of {video_id, title}
      } else if (ev.event === 'timeout') {
        connect()                                         // reconnect with cursor
      } else if (TERMINAL.has(ev.event)) {
        onDone({ status: ev.event, summary: data })
      }
    },
    onerror(err) { onError(err) }
  })

  connect()
}
```

---

### GET /api/ingest/videos — Video inventory

```bash
# All videos on disk with their ingest status
curl -s "https://corbin-submucronated-repletely.ngrok-free.dev/api/ingest/videos"

# Filter by status
curl -s "https://corbin-submucronated-repletely.ngrok-free.dev/api/ingest/videos?status=pending"
curl -s "https://corbin-submucronated-repletely.ngrok-free.dev/api/ingest/videos?status=failed"
curl -s "https://corbin-submucronated-repletely.ngrok-free.dev/api/ingest/videos?status=completed"
```

**`status` filter values:** `pending` | `partial` | `running` | `completed` | `failed`

**Response `200` (~200–500ms depending on library size):**
```json
{
  "count": 3,
  "total_on_disk": 142,
  "videos": [
    {
      "video_id": "abc123",
      "title": "AI in Farming",
      "duration": 3642,
      "webpage_url": "https://youtube.com/watch?v=abc123",
      "overall_status": "completed",
      "stages": {
        "discover":         { "status": "completed", "error": null, "metadata": {}, "updated_at": "2026-06-25T14:31:00Z" },
        "download":         { "status": "completed", "error": null, "metadata": { "title": "AI in Farming" }, "updated_at": "2026-06-25T14:35:22Z" },
        "extract_audio":    { "status": "completed", "error": null, "metadata": {}, "updated_at": "2026-06-25T14:36:50Z" },
        "transcribe":       { "status": "completed", "error": null, "metadata": { "transcript_file": "0141_abc123_AI_in_Farming.json" }, "updated_at": "2026-06-25T14:52:10Z" },
        "create_fragments": { "status": "completed", "error": null, "metadata": { "fragment_count": 38 }, "updated_at": "2026-06-25T14:52:45Z" },
        "classify":         { "status": "completed", "error": null, "metadata": {}, "updated_at": "2026-06-25T15:01:33Z" },
        "embed":            { "status": "completed", "error": null, "metadata": {}, "updated_at": "2026-06-25T15:02:11Z" }
      }
    },
    {
      "video_id": "def456",
      "title": "Precision Irrigation",
      "duration": 2811,
      "webpage_url": "https://youtube.com/watch?v=def456",
      "overall_status": "failed",
      "stages": {
        "download":         { "status": "completed", "error": null, "metadata": {}, "updated_at": "2026-06-25T14:38:01Z" },
        "extract_audio":    { "status": "completed", "error": null, "metadata": {}, "updated_at": "2026-06-25T14:39:20Z" },
        "transcribe":       { "status": "failed", "error": "CUDA out of memory", "metadata": {}, "updated_at": "2026-06-25T14:55:00Z" },
        "create_fragments": { "status": "pending", "error": null, "metadata": {}, "updated_at": null },
        "classify":         { "status": "pending", "error": null, "metadata": {}, "updated_at": null },
        "embed":            { "status": "pending", "error": null, "metadata": {}, "updated_at": null }
      }
    }
  ]
}
```

**`overall_status` derivation:**

| Value | Condition |
|---|---|
| `completed` | `embed` stage is completed |
| `failed` | Any stage is failed |
| `running` | Any stage is running |
| `partial` | At least one stage completed, not all |
| `pending` | No stages started (video on disk but never ingested) |

> **Frontend:** Use `overall_status` for the status badge colour. Use `stages` for a per-stage progress breakdown tooltip/drawer.

---

### GET /api/ingest/videos/{video_id} — Single video status

```bash
curl -s "https://corbin-submucronated-repletely.ngrok-free.dev/api/ingest/videos/abc123"
```

**Response `200` (~100ms):**
```json
{
  "video_id": "abc123",
  "title": "AI in Farming",
  "overall_status": "completed",
  "stages": {
    "discover":         { "status": "completed", "error": null, "metadata": {}, "updated_at": "2026-06-25T14:31:00Z", "job_id": "ijob_20260625_143022_a1b2c3d4" },
    "download":         { "status": "completed", "error": null, "metadata": { "title": "AI in Farming" }, "updated_at": "2026-06-25T14:35:22Z", "job_id": "ijob_20260625_143022_a1b2c3d4" },
    "extract_audio":    { "status": "completed", "error": null, "metadata": {}, "updated_at": "2026-06-25T14:36:50Z", "job_id": "ijob_20260625_143022_a1b2c3d4" },
    "transcribe":       { "status": "completed", "error": null, "metadata": { "transcript_file": "0141_abc123_AI_in_Farming.json" }, "updated_at": "2026-06-25T14:52:10Z", "job_id": "ijob_20260625_143022_a1b2c3d4" },
    "create_fragments": { "status": "completed", "error": null, "metadata": { "fragment_count": 38 }, "updated_at": "2026-06-25T14:52:45Z", "job_id": "ijob_20260625_143022_a1b2c3d4" },
    "classify":         { "status": "completed", "error": null, "metadata": {}, "updated_at": "2026-06-25T15:01:33Z", "job_id": "ijob_20260625_143022_a1b2c3d4" },
    "embed":            { "status": "completed", "error": null, "metadata": {}, "updated_at": "2026-06-25T15:02:11Z", "job_id": "ijob_20260625_143022_a1b2c3d4" }
  }
}
```

**Response `404` — not on disk and no ingest record:**
```json
{ "detail": "Video not found: abc123" }
```

---

### POST /api/ingest/videos/{video_id}/sync — Retry a single video

```bash
curl -s -X POST "https://corbin-submucronated-repletely.ngrok-free.dev/api/ingest/videos/def456/sync"
```

**Response `202` (~50ms):**
```json
{
  "job_id": "ijob_20260625_160055_ff112233",
  "video_id": "def456",
  "status": "queued",
  "already_downloaded": true,
  "message": "Video re-ingest job queued"
}
```

> `already_downloaded: true` → skips download, re-runs from `extract_audio`  
> `already_downloaded: false` → runs full pipeline from `discover`

> **Frontend:** Show a "Retry" button on any video where `overall_status === "failed"`. Use this endpoint on click.

---

## Ingest Lifecycle — Recommended Frontend Flow

```
Admin clicks "Sync Channel"
        │
        ▼
POST /api/ingest/sync
        │
        ├── 409 → show banner: "Sync already running — job_id: {active_job_id}"
        │          link to that job's progress view
        └── 202 → store job_id, open SSE stream
                │
                ├── event: progress        → update stage progress bar
                ├── event: discover_completed → show "X new videos found" badge
                ├── event: video_downloaded  → append video to live list
                ├── event: fragments_created → show fragment count per video
                ├── event: timeout         → reconnect with ?after={lastEventId}
                ├── event: failed          → show error, offer retry button
                └── event: completed       → show summary panel
                          │
                          ▼
                GET /api/ingest/videos    → refresh inventory grid
                          │
                          ▼
                Videos with overall_status="completed"
                are now searchable via POST /api/archive/semantic-search
```
