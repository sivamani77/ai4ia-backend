# Ingest pipeline: disk space and batch processing

**Status:** proposal, not implemented
**Date:** 2026-09-20

## Why this exists

Two things forced this design:

1. **S3 mode never worked end to end.** Each ingest stage uploads its outputs and deletes the
   local copies (`delete_after=True` at [ingest_job_runner.py:621, 675, 737, 790](../../backend/app/services/ingest_job_runner.py#L621)),
   but the runner has only upload and list helpers — nothing ever downloads inputs back, and the
   stage scripts glob local directories exclusively. So `extract_audio` deletes the WAVs, then
   `03_transcribe.py` globs an empty folder and finds nothing to do.
2. **Fixing (1) naively makes disk pressure worse.** Rehydrating each stage's inputs without
   releasing them afterwards means one run accumulates mp4s + WAVs + transcripts + fragments and
   never gives the space back — and it never gets it back on its own, because the volume is
   persistent rather than ephemeral.

A rehydration change is sitting uncommitted pending this design.

## Deployment constraints

Current deployment, as configured:

| | |
|---|---|
| Storage | **30 GiB persistent volume** (= 32.2 GB) |
| API container | 1 CPU, 1536 Mi |
| Worker container | 4 CPU, 8 Gi |

Do **not** size against [examples/local-dev/k8s/11-worker.yaml](../../examples/local-dev/k8s/11-worker.yaml),
which shows 60Gi ephemeral / 20Gi requested. Per [CLAUDE.md](../../CLAUDE.md) the manifests under
`examples/local-dev/` "deploy nothing and describe a superseded topology". An earlier draft of
this document sized against those figures and was wrong by 2×.

Two properties of the real setup drive everything below:

- **Persistent, not ephemeral.** Nothing wipes the volume between runs. Orphaned files from a
  failed or evicted job stay forever until something deletes them deliberately.
- **Shared.** The API serves media from these paths, so worker scratch and API-served content
  compete for the same 30 GiB.

## The corpus, measured

From [docs/reference/inventory/videos_inventory_detailed.csv](../reference/inventory/videos_inventory_detailed.csv) —
309 videos catalogued, 211 with local size data:

| Metric | Value |
|---|---|
| Video bytes (211 measured) | **21.0 GB** |
| Projected for all 309 at the same mean | **~31 GB** |
| Mean video | 102 MB |
| Median video | 46 MB |
| p90 video | 207 MB |
| **Max single video** | **1302 MB** |
| Total duration | 61.8 hours |
| WAV corpus @ 16kHz mono PCM | **7.0 GB** |
| Mean WAV | 34 MB |
| **Max single WAV** | **444 MB** (a 231-minute video) |

WAV size is exact, not estimated: [02_extract_audio.py](../../scripts/02_extract_audio.py) encodes
`pcm_s16le -ar 16000 -ac 1`, which is 32,000 bytes/sec = **115.2 MB per hour**, always.

The small-file working set is negligible by comparison — fragments 24 MB, classified 15 MB,
metadata 6.3 MB, **~45 MB total**.

### The corpus does not fit

Against a 32.2 GB volume:

| | Size | Share of volume |
|---|---|---|
| Video corpus (309 videos) | ~31 GB | **96%** |
| WAV corpus (scaled to 309) | ~10 GB | 32% |
| Both resident together | ~41 GB | **128% — impossible** |

The video corpus alone consumes essentially the whole volume, and video plus audio exceeds it
outright. **There is no configuration in which a full local ingest completes.** S3 offload with
per-stage cleanup is therefore a correctness requirement, not a performance optimization — which
also means the currently-broken S3 mode is the only viable mode, and the pipeline has no working
path today at full corpus size.

## The finding that shapes the design

**Video sizes are heavily skewed: median 46 MB, max 1302 MB — a 28× spread.**

This makes *batch-by-count* the wrong primitive. "Process 5 videos at a time" means anywhere
between 230 MB and 6.5 GB of input depending on which five you draw. Sizing a fixed count safely
requires assuming worst case for every video, which wastes most of the budget most of the time.

**Batch by bytes instead.** And the sizes are already free: `list_objects_v2` returns `Size` on
every object, and the rehydration helper already iterates that response. No extra API calls, no
guessing from the inventory CSV.

## Design

### 1. Per-stage input cleanup — the must-have

After a stage uploads its outputs, delete the inputs it pulled down. **S3 mode only.**

This asymmetry is the whole point and must not be gotten wrong: in S3 mode the bucket is the
source of truth and local files are pure scratch; in local mode those files *are* the source of
truth and deleting them destroys data. Every cleanup call sits behind the existing
`_ingest_use_s3()` guard, exactly as the rehydration does.

This mirrors what the video pipeline already does with `tmp_dir` / `cleanup_tmp_dir` in
[pipeline/s3_helpers.py](../../backend/app/pipeline/s3_helpers.py).

### 2. Byte-budgeted chunking for the media stages

Applies to **`extract_audio`** and **`transcribe`** only — the two stages handling large files.

The stage scripts glob their entire input directory, so **what is on disk defines the batch**.
That means chunking needs no script changes, no new CLI flags, and no modification to the scripts
at all:

```
for chunk in chunks_under_budget(video_ids, INGEST_BATCH_BYTES):
    rehydrate chunk's inputs
    run stage script          # sees only this chunk on disk
    verify + record per-video state
    upload outputs (delete local)
    delete chunk's inputs
```

A single video larger than the budget forms its own chunk — never skipped, never stalled.

`create_fragments` is excluded: its inputs are transcripts (text) and `.info.json`, both tiny.

### 3. Leave `classify` and `embed` whole-corpus

Both rebuild whole-corpus outputs and must see every file:

- `05_classify_fragments.py` rewrites `classified_fragments_catalog.jsonl` with `open("w")` —
  truncate — from whatever is in its input directory. A partial view silently deletes every other
  video from the catalog.
- `06_generate_embeddings.py` rebuilds the embedding JSONL and matrix from the full directory.

At ~45 MB this costs nothing. **These two must never be chunked** without first making their
writers merge rather than truncate.

### 4. Never clean `data/metadata/`

`data/metadata/youtube/download_archive.txt` is yt-dlp's dedup record and
[`_stage_discover`](../../backend/app/services/ingest_job_runner.py#L411) calls it "most
authoritative — survives local deletes". Deleting it makes the next sync re-download the entire
channel — ~31 GB onto a 32.2 GB volume. Cleanup must be scoped to the media directories by
explicit allow-list, never by sweeping `data/`.

### 5. Orphan sweep on startup

Because the volume is persistent, a job killed mid-chunk leaves its inputs behind permanently —
and repeated failures accumulate until the volume fills. The runner already has
`reap_orphaned_ingest_jobs()` running at startup to fail stale job rows; the same hook should
delete media files left in the working directories, which is safe precisely because in S3 mode
those directories hold nothing authoritative.

### 6. Guard rails

- **Free-space check before each chunk.** Fail the job with a clear error rather than filling the
  shared volume — which would take the API's media serving down with it, not just the worker.
- **`INGEST_BATCH_BYTES` as a typed setting** in [core/settings.py](../../backend/app/core/settings.py)
  with a conservative default, per the project convention of never reading `os.environ` at a call
  site.
- **Log reclaimed bytes per stage** so the space behaviour is observable in job events rather than
  reconstructed after the volume fills.

## Peak disk

Peak is bounded by the largest single chunk, not by corpus size — so it stays flat as the channel
grows, which is the property that matters:

| Budget | Peak during extract_audio (inputs + WAVs) | Share of the 32.2 GB volume | Videos per chunk (at median 46 MB) |
|---|---|---|---|
| 0.5 GB | ~0.8 GB | 2.3% | ~11 |
| **1 GB** | **~1.5 GB** | **4.7%** | ~22 |
| 2 GB | ~3.0 GB | 9.3% | ~43 |
| 4 GB | ~6.0 GB | 18.6% | ~87 |

WAV output averages about a third of video bytes across the corpus (21 GB → 7 GB), but the ratio
inverts for long, heavily-compressed talks, so the table assumes WAVs at 50% of the budget.

**Recommended default: 1 GB**, giving ~1.5 GB peak — under 5% of a volume that is also serving
API media and any video-pipeline output. 4 GB is defensible on a quiet volume but leaves less room
for a concurrent render; the cost of a smaller budget is only more chunks, and the per-chunk
overhead is one S3 listing.

Note the budget must be sized for *concurrent* jobs, not one: the worker's concurrency setting
multiplies peak disk. Worth confirming what `WORKER_CONCURRENCY` is in the deployed environment
before fixing the default.

## Code touch points

| File | Change |
|---|---|
| [`services/ingest_job_runner.py`](../../backend/app/services/ingest_job_runner.py) | `_s3_cleanup_dir` alongside `_s3_download_dir`; chunk loop in `_stage_extract_audio` and `_stage_transcribe`; free-space check; orphan media sweep in `reap_orphaned_ingest_jobs` |
| [`core/settings.py`](../../backend/app/core/settings.py) | `ingest_batch_bytes` typed field with default |
| `tests/unit/test_ingest_s3_relay.py` | extend: cleanup runs in S3 mode, **never** in local mode; `data/metadata` never swept; chunking respects the budget; an oversized video still forms its own chunk |

The existing `GET /api/admin/storage/space` ([api/admin.py:186](../../backend/app/api/admin.py#L186))
reports S3 usage per prefix. Nothing measures the *volume*, which is the thing that actually fills
— and because it is shared, filling it degrades API media serving too, not just ingest. A local
free-space figure on that endpoint would be cheap and is worth doing alongside this.

## Out of scope

- **Making 05's catalog and 06's matrix merge instead of truncate.** That would unlock chunking
  those stages too, but it is a correctness change to whole-corpus outputs and deserves its own
  review.
- **The `data/subtitles` and embedding `.npy`/`.json` upload gaps** — stage 03's subtitles and two
  of the three embedding artifacts are never uploaded to S3, yet
  [stage_07_storyline.py:20-24](../../backend/app/pipeline/stage_07_storyline.py#L20) requires all
  three. Separate bug.
- **Ingest job resume.** Tracked separately; depends on this landing first.

## Verification

1. Unit tests as above, with the local-mode no-cleanup assertions treated as the critical ones —
   a regression there deletes source data.
2. Dry-run the chunker against the real inventory CSV: assert every video lands in exactly one
   chunk and no chunk exceeds the budget except a single oversized video.
3. The real proof is an S3-mode ingest run with `df` on the volume watched throughout. Nothing on
   the current dev box can validate this — it has no local corpus and the configured S3 endpoint
   is a cluster-internal hostname that does not resolve here.

## Open questions

- **What is `WORKER_CONCURRENCY` in the deployed environment?** The repo's `.env` sets 6 while
  `celery_app.conf` sets `worker_concurrency=1`. Concurrent jobs multiply peak disk, so the batch
  default depends on the answer. (4 CPU / 8 Gi also makes 6 concurrent Whisper transcriptions look
  optimistic on memory, independent of disk.)
- **What else occupies the volume today?** Sizing assumes ingest has most of the 32.2 GB. If API
  media, exports or generated assets hold a material share, the budget should come down.
- **Is the volume ReadWriteMany across API and worker, or does each get its own?** Determines
  whether a worker filling it takes API media serving down with it.
