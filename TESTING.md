# AI4IA Backend — Testing Guide

Covers every endpoint built in Phases 1–3 plus the pipeline scripts.
All example IDs below are real values from the current dataset (85 videos, 1,664 fragments).

---

## 0. Start the server

```bash
cd /home/ash/ai4ia-demo-pipeline/backend
source ../.venv/bin/activate
uvicorn app.main:app --reload --port 8077
```

Leave it running in one terminal. Run all `curl` commands below in a **second** terminal.

Set a base var to save typing:
```bash
BASE=http://127.0.0.1:8077/api
```

**Tip:** pipe any response through `python3 -m json.tool` (or `jq`) to pretty-print:
```bash
curl -s "$BASE/health" | python3 -m json.tool
```

**Best UI option:** open **http://127.0.0.1:8077/docs** — FastAPI's interactive Swagger page lists every endpoint with a "Try it out" button. Easiest way to test the POST endpoints.

---

## 1. Health & Dashboard

```bash
curl -s "$BASE/health"
curl -s "$BASE/dashboard"
```
Expect: health `{"status":"ok"}`; dashboard shows archive/pipeline counts.

---

## 2. Archive — browse & keyword search (Phase 0/1)

```bash
# list videos (should show ~85)
curl -s "$BASE/archive/videos?limit=5"

# keyword search a video
curl -s "$BASE/archive/videos?query=Zimbabwe&limit=5"

# themes & chapters
curl -s "$BASE/archive/themes"

# keyword fragment search (substring)
curl -s "$BASE/archive/search?q=youth&limit=3"
curl -s "$BASE/archive/search?speaker=James%20Sherer&limit=3"
curl -s "$BASE/archive/search?theme=AI%20Ethics&limit=3"
```

---

## 3. Speakers (Phase 1)

```bash
curl -s "$BASE/archive/speakers"
```
Expect: ~71 distinct speakers with fragment/video counts and `attribution: "video-level"`.

---

## 4. Semantic search (Phase 1) — the embedding-powered one

```bash
curl -s -X POST "$BASE/archive/semantic-search" \
  -H "Content-Type: application/json" \
  -d '{"query":"youth empowerment and responsible governance","top_k":5}'
```
Expect: `"mode":"semantic"`, results sorted by `similarity` (~0.6–0.7 at top).
Compare against keyword search (#2) — semantic should surface relevant clips that don't contain the literal words.

With a filter:
```bash
curl -s -X POST "$BASE/archive/semantic-search" \
  -H "Content-Type: application/json" \
  -d '{"query":"AI ethics","top_k":5,"filters":{"speaker":"James Sherer"}}'
```

---

## 5. Fragment lookup & preview (Phase 1)

```bash
FID="0002_SvVu6k02Hes_Zimbabwe_-_AI4IA_2025_1_0001"

# full fragment record
curl -s "$BASE/archive/fragments/$FID"

# preview = stream URL + timestamps (for a video player)
curl -s "$BASE/archive/fragments/$FID/preview"
```
Expect preview: `{video_stream_url, start, end, duration, speaker, edition, ...}`.

**Play the clip:** open the `video_stream_url` in a browser, e.g.
`http://127.0.0.1:8077/api/archive/videos/0002_SvVu6k02Hes_Zimbabwe_-_AI4IA_2025_1/stream`
then seek to the `start` time.

---

## 6. Insights — evidence-grounded synthesis (Phase 2)

```bash
# generate (uses 70B; takes ~60-90s — be patient)
curl -s -X POST "$BASE/insights/generate" \
  -H "Content-Type: application/json" \
  -d '{"query":"How does AI4IA discuss youth empowerment and AI governance?","limit":12}'
```
Expect: `title`, `summary`, `patterns[]` and `contrasts[]` each with `fragment_refs`,
`evidence_coverage` (1.0 = every claim cited), `unsupported_claims` (should be empty),
and an `evidence_pool` of numbered clips.

```bash
# list saved insights
curl -s "$BASE/insights"

# fetch one (use an id from the list, or the existing one below)
curl -s "$BASE/insights/insight_20260615_120415"

# EVIDENCE MAP: every claim -> its source clips (timestamps + stream URLs)
curl -s "$BASE/insights/insight_20260615_120415/sources"
```
The `/sources` response is the **trust feature** — each claim links to playable clips.

---

## 7. Storyline evidence map (Phase 2)

```bash
SID="storyline_20260615_153500"
curl -s "$BASE/storylines/$SID/evidence-map"
```
Expect: `candidate_count` / `resolved_count` (should match, e.g. 35/35) and an
`evidence[]` list with speaker, timestamps, and stream URL per clip. Read-only.

---

## 8. Speaker comparison (Phase 3)  — ~50–60s

```bash
curl -s -X POST "$BASE/compare/speakers" \
  -H "Content-Type: application/json" \
  -d '{"speakers":["James Sherer","Stefaan G Verhulst"],"topic":"AI ethics and governance","limit_per_speaker":4}'
```
Expect: `per_speaker[]` (each with a position + refs), `agreements[]`, `tensions[]`,
`synthesis`, and `speakers_missing` (should be `[]` if both names exist).
Pick names from the `/archive/speakers` list (#3).

---

## 9. Theme synthesis (Phase 3)  — ~50s

```bash
curl -s -X POST "$BASE/synthesis/theme" \
  -H "Content-Type: application/json" \
  -d '{"theme":"AI Ethics, Governance & Policy","limit":12}'
```
Expect: `title`, `summary`, `key_points[]` (cited), `representative_clips[]`.

---

## 10. Relay of Ideas (Phase 3)  — ~45s

```bash
curl -s -X POST "$BASE/relay" \
  -H "Content-Type: application/json" \
  -d '{"topic":"youth empowerment in AI","limit":12}'
```
Expect: `relay_title`, `relay_narrative`, and `relay_steps[]` — an ordered thread
where each step is a DIFFERENT speaker's clip tagged
`opening | development | tension | synthesis`, with timestamps. `distinct_speakers` ≥ 2.

---

## 11. Admin reindex (after any pipeline run)

```bash
curl -s -X POST "$BASE/admin/reindex"
```
Expect: `{"status":"ok","fragments":1664}`. Refreshes the in-memory store without
restarting the server.

---

## 12. Assets / Exports (existing pipeline wrappers)

```bash
curl -s "$BASE/assets/latest"
curl -s "$BASE/exports"
# stream/download an existing export:
#   http://127.0.0.1:8077/api/exports/ai4ia_demo_v4_refined.mp4/stream
```

---

## 13. Jobs

```bash
curl -s "$BASE/jobs"
curl -s "$BASE/jobs/<job_id>"
```

---

## Quick smoke test (all the fast ones at once)

```bash
BASE=http://127.0.0.1:8077/api
for path in health dashboard "archive/videos?limit=2" archive/themes archive/speakers; do
  echo "=== GET /$path ==="
  curl -s "$BASE/$path" | head -c 300; echo; echo
done

echo "=== POST semantic-search ==="
curl -s -X POST "$BASE/archive/semantic-search" -H "Content-Type: application/json" \
  -d '{"query":"open data and transparency","top_k":3}' | head -c 500; echo
```

---

## Notes / expected quirks

- **LLM endpoints (insights, compare, synthesis, relay) are slow** (~45–90s) because they
  call the 70B model on the shared Ollama server. This is expected; not a bug.
- If an LLM call times out, retry — the server has retry logic but the upstream can be busy.
- `evidence_coverage: 1.0` and empty `unsupported_claims` means every generated claim
  was backed by a real fragment. Lower coverage on vague queries is the validator working.
- Fragment IDs with spaces/quotes (e.g. the Chile Event video) must be URL-encoded; the
  examples above use a clean ID to avoid that.
```
