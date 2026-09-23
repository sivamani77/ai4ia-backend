# Speaker cleanup plan

Status: Phases 0–6 done and rehearsed locally (2026-09-23). Next: Phase 7, UAT-only rollout (step 1). Owner: Aswin Prabhu. Last updated: 2026-09-23.

## Why

Speaker names in the archive are unreliable:

- The cross-check by YouTube video ID (`docs/AI4IA_speaker_video_xcheck.csv`) found only 140 of the 265 register videos carry the correct name.
- About 105 videos are the right person with a misspelt name, 2 name the wrong person, and 18 multi-speaker panels are credited to one person or to nobody.
- 42 videos have a cut-short `video_id`: `extract_video_id()` splits on `_`, and some YouTube IDs contain `_`.
- The in-code alias map `app/services/speaker_aliases.ALIASES` points several names the wrong way (e.g. `Cordel Green → Cordell Green`). `SPEAKER_ALIAS_MAP` in `prompt_intent.py` disagrees with it.
- Alias seeding uses `ON CONFLICT DO NOTHING`, so the wrong alias rows cannot be corrected by redeploying.

The goal:

- one list of speakers with their aliases, checked into the repo;
- every store corrected once, in a controlled way;
- ingest fixed so the bad names do not return.

## Where speaker data lives

Every one of these has to end up consistent, or a rebuild reintroduces the old names.

| Store | Field | Rebuilt from |
|---|---|---|
| ChromaDB `fragments` | `speaker`, `speaker_attribution`, `video_id` metadata | `raw/embeddings` in S3 |
| Postgres `fragments` | `speaker`, `speaker_attribution`, `video_id` | ChromaDB (`migrate_fragments_to_postgres.py`) |
| Postgres `speakers`, `speaker_aliases` | canonical names, aliases | `ALIASES` seed at startup |
| Postgres `videos` | `video_id` | `sync_videos_registry.py` |
| S3 `raw/metadata/speakers/*.json` | per-video speaker list | `04b_extract_speakers.py` |
| S3 `raw/fragments_classified/*.json` | per-fragment speaker | `05_classify_fragments.py` |
| Code | `ALIASES`, `SPEAKER_ALIAS_MAP` | hand-edited |

Speaker names are not part of the embedding text (`build_embedding_text`), so the fix is a metadata update only. No new embeddings are needed.

## Decisions already made

| Video | Decision |
|---|---|
| `YxzoN31lAvc` | Canonical name **Nancy Nkwanga**; `Nancy Kwangwa` (register) is an alias |
| `QP_5ovm6wn8` India Event 2025 | **Samridhi Arora**, single speaker (was `Kodel Green`) |
| `xmkczLmiOf0` Europe "Driving European AI" | Panel, taken from the video description. Moderators Dariia Opryshko and Andrea Millwood Hargrave; speakers Stefaan G. Verhulst, Natali Helberger, Krisztina Rozgonyi, Stefan Baack |
| All Europe panels | `Dario Prishko`, `Dario Pryshko`, `Daria Pryshko` are aliases of **Dariia Opryshko** |
| `zJLv8sreIiQ` | **A Singh** (`Mr A Singh` is an alias), status `review`. Not Ashita Singh |
| `jpfE9ULuv9M` | **David Daou**, status `review` (not in the register) |
| `iHUEZ0WiErM` China Satellite | Drop `Dr. Zhu Zhai Zhang`. Panel from the register: Lin Haibo, Shahbaz Khan |
| `_c244n-7nd0` | Panel from the title: **Hibaq Warsame** and **M Ekotto** (Ekotto status `review`) |
| `zy45aKgJUiY` Botswana Satellite | Drop `Kelly`. Panel from the register's list |

Spelling rules:

1. The register wins by default.
2. We keep our accented forms (Škopac, Kerševan, María José, …).
3. We keep our fix of the register's typos: Philbeck, Svitlana Mazepa, Shepherd.
4. Where a video description lists the speakers, it wins over the register. The register gives both 2025 Europe panel videos the same merged list.

Panels get `speaker = ""`, `speaker_attribution = "Multiple speakers"` and the full list in `speaker_roster`. Naming the speaker on each panel clip needs diarization and is out of scope.

## Who reviews what, and how

The cluster's Postgres and ChromaDB cannot be reached from a dev machine, but the S3 bucket (`ai4ia-data`) can: the keys are in the local `.env.example`. So each cluster run writes its report to S3, and it is read locally. No new API is needed for this cleanup.

| # | What | Reviewer | How | Scope |
|---|---|---|---|---|
| 1 | Speaker registry | Aswin, plus whoever owns the register if a name is disputed | PR diff of `config/speaker_registry.json`, plus `docs/speaker_registry_videos.csv` and `docs/speaker_registry_speakers.csv` (open in a spreadsheet) | Now |
| 2 | Local dry-run and apply | Aswin | Report files under `backups/speaker_registry_v1/` | Now |
| 3 | Cluster dry run (deploy 1) | Aswin | `s3://ai4ia-data/reports/data_repairs/speaker_registry_v1/dry_run.json`, plus pod logs | Now |
| 4 | Cluster apply result (deploy 2) | Aswin | `…/apply.json`, plus `GET /api/archive/speakers` | Now |
| 5 | New unknown names from future ingests (`status='review'`) | Admin user | Admin review queue (see below) | **Later** |

`scripts/fetch_repair_report.py --mode dry_run|apply|rollback` downloads a report and prints the summary; `--full` lists every changed video. It only reads S3. uat and release share one database, so there is one report per mode, not one per environment; the report names the pod that ran it. Older runs are kept under `history/`.

Each report contains:

- the status: `dry_run`, `applied`, `failed` or `rolled_back`;
- the attempt count;
- counts per change type;
- videos found in the cluster but not in the registry;
- names left in `review`;
- the check results;
- the per-video changes (old speaker → new speaker, fragment count).

Until item 5 exists, names in `review` show up in these reports. An engineer resolves them by adding them to the registry in a PR.

### Admin endpoints (later scope)

- `GET /api/admin/data-repairs/{name}`: shows the same report through the API, for anyone without S3 access. Read-only; it goes on the existing admin router (`_admin_dep` in `main.py`).

Speaker review queue:

- `GET /api/admin/speakers?status=review`: unconfirmed names, with the videos and fragment counts for each.
- `POST /api/admin/speakers/{name}/approve`: marks the name confirmed.
- `POST /api/admin/speakers/{name}/merge` with body `{"into": "<canonical>"}`: adds the name as an alias, then updates matching fragments in Postgres and ChromaDB.
- `POST /api/admin/speakers/{name}/rename` with body `{"to": "<name>"}`.
- Admin changes to the database must be exported back into `config/speaker_registry.json`. Otherwise the next seed from the registry would overwrite them. Decide which of the two is authoritative before building this.

## Phase 0: Safe local setup

1. Create `.env.localtest`:
   - Postgres at `localhost:15432`;
   - ChromaDB over HTTP at `localhost:8001`, the same client code the cluster uses;
   - `AI4IA_STORAGE_BACKEND=local`;
   - no S3 keys.
   The normal `.env` points at cluster hosts and S3, so it must never be used for this.
2. The repair CLI refuses to write to a non-localhost Postgres or ChromaDB, or to S3, unless `--target=cluster` is passed. Local runs only read S3.
3. Pull a read-only copy of S3 into `data/`, using the keys from `.env.example` for this step only:
   - `raw/fragments_classified/`: 309 files in S3, only 211 locally;
   - `raw/metadata/speakers/`: 211 files.
   The rehearsal then rewrites these local copies, which exercises the same code the cluster uses on S3.
4. Take a snapshot before any change:
   - a tar of the ChromaDB Docker volume;
   - `data/metadata/speakers/` and `data/fragments_classified/`;
   - a checksum manifest.

## Phase 1: Load local Postgres

Local Postgres currently has 0 fragments.

1. Run `ensure_pipeline_schema()`.
2. Run `scripts/migrate_fragments_to_postgres.py`. Expect 5,112 fragments.
3. Run `scripts/sync_videos_registry.py`. Expect 309 videos.
4. Record the baseline: 278 distinct speaker values and 42 cut-short `video_id`s.

## Phase 2: Build the registry

`scripts/build_speaker_registry.py` writes `config/speaker_registry.json` and two review files: `docs/speaker_registry_videos.csv` (one row per video) and `docs/speaker_registry_speakers.csv` (one row per speaker). `--check` exits 1 if the registry is out of date.

Inputs:

- `docs/AI4IA Speaker Register_updated.html` (the latest register, with affiliations);
- `docs/AI4IA_speaker_cleanup.csv`;
- current corpus speakers from `data/chroma/chroma.sqlite3`, read-only (the video-ID cross-check is recomputed, not read from `docs/AI4IA_speaker_video_xcheck.csv`);
- video descriptions from `.info.json`, read by code for the "Moderators: … / speakers include …" pattern;
- the decisions above.

```json
{
  "version": "speaker_registry_v1",
  "speakers": {
    "Cordel Green": {"aliases": ["Cordell Green", "Kodel Green"], "status": "confirmed"}
  },
  "videos": {
    "GWMcNLuXnRA": {"kind": "single", "speakers": ["Chuck Howell"]},
    "xmkczLmiOf0": {"kind": "panel", "speakers": ["Dariia Opryshko", "..."]}
  }
}
```

The build fails, rather than guessing, when:

- an alias maps to two people;
- two people normalize to the same key;
- a video lists a name that is not a registry speaker.

Review: item 1 in the review table.

## Phase 3: Code changes

1. **Schema (`db/schema.py`):**
   - `speakers`: add `register_name`, `status`, `updated_at`;
   - `speaker_aliases`: add `normalized_alias`;
   - new `video_speakers(video_id, canonical_name, role, ordinal)`;
   - new `data_repairs(name, status, attempts, summary, updated_at)`;
   - new `fragment_speaker_backup(fragment_id, store, video_id, speaker, speaker_attribution)`;
   - `videos`: add `description`.
2. **Name resolution (`services/speaker_aliases.py`):**
   - `canonicalize()` reads the registry: exact match, then normalized (case, accents, titles, punctuation), then initials (`C Howell` → the one registry person with that surname and initial).
   - Delete the hardcoded `ALIASES`. Build `SPEAKER_ALIAS_MAP` from the registry.
   - Alias seeding upserts, so wrong existing rows get corrected.
3. **ChromaDB HTTP client (`services/embedding_index._HttpCollection`):** add paged `get` and `update`.
4. **`services/speaker_repair.py`:**
   - backup, plan, apply, verify, rollback;
   - a CLI: `python -m app.services.speaker_repair --dry-run | --apply | --rollback`;
   - uploads each run's report to `reports/data_repairs/<REPAIR_NAME>/<mode>.json` (local runs: `backups/<REPAIR_NAME>/reports/`).
5. **`scripts/fetch_repair_report.py`:** reads a report from S3 and prints its summary.
6. **Ingest fixes:**
   - `04_create_fragments.py` and `04_regenerate_fragments_from_clean_transcripts.py` use `video_naming.parse_source_video` instead of `split("_")`.
   - `04b_extract_speakers.py` resolves every name through the registry, reads description speaker lists with code, and saves speaker lists by real `video_id`.
   - An unknown name is stored with `status='review'`. It never becomes a new confirmed person silently.
7. **Unit tests:**
   - normalization and initials matching;
   - the planner on fixtures: single speaker, panel, placeholder name, cut-short ID, unknown video, already-correct (no change);
   - the registry build failing on conflicts.

## Phase 4: Full local rehearsal

| Step | Pass condition |
|---|---|
| Dry run | Report reviewed; every change explained |
| Apply | Checks pass: Postgres and ChromaDB agree on every fragment; 0 cut-short IDs; every speaker is in the registry; every panel has a roster |
| Apply again | 0 changes |
| Rollback | Data matches the Phase 0 snapshot byte for byte; then apply again |
| Kill mid-apply, restart | Completes; the backup still holds the original values |
| Start the API with 4 workers, switch on | Exactly one worker runs the repair; `/health` responds during the run |
| App checks | `/api/archive/speakers` has no duplicates; a video job for a "Cordel Green" prompt shows correct names on screen |
| Test suite | No worse than the known baseline (78 failed / 833 passed) |
| Rebuild | Re-running the ChromaDB → Postgres migration and re-classifying a sample does not bring old names back |

### Rehearsal results (2026-09-23)

Evidence: `backups/speaker_registry_v1/phase4/` (logs and state exports).

- Apply: 5,112 fragments in each store updated; all five checks passed.
- Apply again: 0 changes; state identical.
- Rollback: Postgres, Chroma and all 520 files identical to the pre-apply state.
- API with 4 workers and the switch on: one worker applied, three logged `locked_by_other_pod`; `/api/health` answered during the run.
- Killed with `kill -9` after Postgres and 400 of 5,112 Chroma updates: the next run finished the remaining 4,712, all checks passed, and all 10,224 backup rows still held the original values.
- Rebuild: re-running the Chroma → Postgres migration and re-classifying original pre-repair files both produced the corrected names.
- Bug found and fixed: `video_speakers` was rebuilt with a delete and insert in one statement, which fails when rows already exist (any apply after a rollback). Now two repeatable statements.

Gaps found in the rehearsal, fixed before Phase 5:

- 55 people appear only on panels (panel fragments have no single speaker), so they were missing from `/api/archive/speakers` and from speaker prompts. 6 of them (Dariia Opryshko, Geoffrey Rockwell, Sebastian Waisbrot, Ashita Singh, Aaron Cameron, Hibaq Warsame) had been findable before, through wrong whole-panel credits.
  - `/api/archive/speakers` now lists all 307 people and adds `panel_video_count` / `panel_fragment_count` (from `video_speakers`). Panel-only people have `fragment_count: 0`. Frontends that assume every entry has solo fragments should check this.
  - Speaker prompts use panel clips as a fallback: only when the person has fewer solo clips than `speaker_panel_fallback_min_clips` (default 6), ranked after solo clips. Panel clips carry no on-screen name.
- Prompts with initials ("C Howell", "C. Howell") now resolve through the registry when exactly one speaker matches. Lowercase initials are not matched, to avoid false hits.
- Found while checking: a prompt for "Ashita Singh" resolved to "A Singh", because speaker confirmation used plain substrings. It now prefers exact, then whole-word matches.

## Phase 5: Clean local backup

Saved to `backups/speaker_registry_v1/`, untracked and local only:

- `pg_dump` of the local database;
- a tar of the ChromaDB volume;
- `data/metadata/speakers/` and `data/fragments_classified/`;
- the dry-run and apply reports;
- a checksum manifest.

This is the known-good copy.

## Phase 6: Remove old data

Built into `services/speaker_repair.py` as two more modes, controlled like the repair:

```python
RUN_SPEAKER_CLEANUP = False    # on for deploy 4
SPEAKER_CLEANUP_APPLY = False  # False = preview only; True = delete
MIN_DAYS_BEFORE_CLEANUP = 7    # the real cleanup waits this long after the repair was applied
```

Guards:

- It runs only when the repair's marker says `applied`, which is written only when all repair checks passed.
- The real cleanup is refused for 7 days after the repair was applied (`too_soon_after_apply`). A preview can run any time after the apply.
- Its own marker (`speaker_registry_v1_cleanup`) makes it run once.
- Once it has run, rollback is refused (`backup_removed_by_cleanup`), even with `--force`.

What it does:

- deletes `videos` rows stored under a cut-short ID, when the row with the correct ID exists. Rows without a correct twin are only reported;
- renames roster files named by a cut-short ID to `<video_id>.json`. These are the only roster for their video, so they are renamed, not deleted. If the correct name already holds different content, both are left and reported;
- deletes the `fragment_speaker_backup` rows and the pre-repair file copies (`classified/`, `roster/`, `created_rosters.json`). The files-applied marker is kept, so the repair's file step stays done;
- re-runs the repair checks, plus: no backup rows left, no copies left, every roster named by its video ID.

Not needed: wrong alias rows are corrected by the registry seed and the repair; stale storylines stop being used when the cache version changes (deploy 2).

CLI: `--cleanup-preview`, `--cleanup` (`--force` skips the 7-day wait; for rehearsals only).

### Local rehearsal (2026-09-23)

Evidence: `backups/speaker_registry_v1/phase6/`.

- Preview: 0 `videos` rows to fix, 10,224 backup rows, 27 misnamed rosters, 521 file copies.
- `--cleanup` without `--force`: skipped, `too_soon_after_apply`.
- `--cleanup --force`: all eight checks passed. Postgres, Chroma and classified files unchanged; 309 rosters with the same contents, 27 renamed.
- Afterwards: rollback refused, cleanup and apply both `already_done`.
- Local restore points remain: `00_pre_snapshot/` (before the repair) and `05_clean_backup/` (after the repair, with the backup rows, so rollback is still possible by restoring it first).

## Phase 7: Deploy to the cluster

Hardcoded in `services/speaker_repair.py`:

```python
REPAIR_NAME = "speaker_registry_v1"  # marker key; bump to run again
RUN_SPEAKER_REPAIR = False           # True for deploys 1 and 2
SPEAKER_REPAIR_APPLY = False         # False = dry run and report only
RUN_SPEAKER_ROLLBACK = False         # emergency only; impossible after the cleanup
RUN_SPEAKER_CLEANUP = False          # Phase 6, deploy 4
SPEAKER_CLEANUP_APPLY = False        # False = preview only
```

How it runs:

- **From startup:** it is called from the startup code (`lifespan`) after `ensure_pipeline_schema`, in a background thread, so startup is not blocked.
- **One copy only:** it takes a Postgres advisory lock, so only one worker or pod runs it.
- **Once only:** it does nothing once the marker records the result for the current switch settings.

### Rollout: UAT only (decided 2026-09-23)

Only the `uat` branch gets this work for now; `release` is merged later, with other changes. A push to `uat` deploys the test service (`ai4ia-backend-test`), which **shares Postgres, ChromaDB, Redis and S3 with release**. So every step below that writes data also changes what release reads, while release still runs the old code.

| Step | Push to `uat` with | Effect on the shared data | Check |
|---|---|---|---|
| 1 | all switches off | New tables and columns are added; the registry's speakers and aliases are loaded. No fragment changes. | UAT app works: `/api/archive/speakers`, a speaker prompt, an ingest of one test video if possible |
| 2 | repair on, apply off | Dry run only; writes the report and the `dry_run` marker | `fetch_repair_report.py --mode dry_run`: counts match the local rehearsal (5,112 fragments, 0 unknown videos), or explain the difference |
| 3 | apply on, plus the storyline-cache version change | The real repair, once, in one UAT pod | Logs: one `RESULT status=applied` and five `CHECK ... PASS`; `fetch_repair_report.py --mode apply` |
| 4 | repair off | Nothing runs | Wait at least 7 days |
| 5a | cleanup on, cleanup apply off | Cleanup preview; nothing deleted | `fetch_repair_report.py --mode cleanup_preview` |
| 5b | cleanup apply on | Phase 6 cleanup; rollback no longer possible | `fetch_repair_report.py --mode cleanup`; logs show eight `CHECK ... PASS` |
| 5c | cleanup off | Nothing runs | — |

Before step 3:

- pause ingest in **both** UAT and release: `ENABLE_AUTO_SYNC` off, and no ingest jobs queued or running;
- wait until no video jobs are running in either environment;
- in the same commit, change the `storyline_cache_corpus_version` default in `core/settings.py` (and `STORYLINE_CACHE_CORPUS_VERSION` in UAT's deployment config if it is set there).

What release sees from step 3 until it gets this code:

- Corrected names everywhere, and panel fragments with no single speaker. That is the intended data.
- Its old code has no panel fallback, so the 55 panel-only people cannot be the subject of a speaker prompt in release, and its speakers list shows 252 people instead of 307.
- Release keeps its own cache key until the merge brings the new default, so it may keep serving cached storylines with old names until then.
- **Release must not ingest until it has this code.** Its old ingest would write old spellings (the old alias map points several names the wrong way). Keep release ingest paused, or accept a drift check afterwards.
- Old release code re-adds a few old alias rows to `speaker_aliases` on restart. Nothing reads that table yet, and UAT's next start corrects it.

When release is merged later:

- Merge with every switch off. The markers in the shared database say `applied` (and `cleaned` after step 5b), so nothing runs again.
- If release ingested anything in between, bump `REPAIR_NAME` to `speaker_registry_v2` and run a dry run from either environment. Any drift shows as changes in the report; apply it the same way.
- Make sure release's storyline-cache version matches UAT's.

Rollback: set `RUN_SPEAKER_ROLLBACK = True` and deploy. It restores every fragment from `fragment_speaker_backup`. This is possible only until Phase 6 drops that table.

## What the pod logs show

Pod logs are the only view into the cluster, so every run logs clear lines with one prefix, `[speaker_repair]`:

```
[speaker_repair] START name=speaker_registry_v1 mode=dry_run pod=<hostname>
[speaker_repair] STEP backup rows=5112
[speaker_repair] STEP plan changes=... single=... panel=... unknown_videos=...
[speaker_repair] STEP postgres updated=...   (apply only)
[speaker_repair] STEP chroma updated=...     (apply only)
[speaker_repair] STEP s3 rewritten=...       (apply only)
[speaker_repair] CHECK <name> PASS|FAIL <detail>
[speaker_repair] RESULT status=dry_run|applied|failed|skipped reason=... report=s3://ai4ia-data/...
```

- Every run ends with exactly one `RESULT` line. A failure logs at ERROR level with the traceback, then `RESULT status=failed`.
- Only one API pod does the work, because of the lock. The other pods log `RESULT status=skipped reason=locked_by_other_pod` or `reason=already_done`. Check the logs of every API pod, or grep all of them for `RESULT`.
- The repair runs in a background thread, so the pod shows as healthy even if the repair fails. A healthy pod is not proof of success; only a `RESULT` line is.
- Pod logs are lost when a pod restarts or is replaced, so the same summary also goes to the S3 report, which can be read from a dev machine.
- The worker pod never runs the repair; it runs only from the API's startup code.

## Edge cases

- **A pod dies mid-run.**
  - The advisory lock is released with the connection.
  - The backup only inserts and never overwrites, so the first original values survive.
  - No marker is written, so the next start completes the run.
- **It keeps failing.** After 3 failed attempts the status becomes `failed`, and the repair stops retrying to avoid a crash loop.
- **The cluster has fragments that local does not.** Targets come from `video_id` and the registry, not from the old values, so new fragments are still handled. Videos missing from the registry fall back to `canonicalize()` and are listed in the report.
- **Postgres updates but ChromaDB fails.** The check compares both stores, so the marker is not written and the next start finishes the job.
- **ChromaDB rejects `None`.** Empty values are written as `""`.
- **Video IDs starting with `_` or `-`.** They are parsed by fixed length, never split.
- **Renaming a canonical name (foreign key).** Insert the new `speakers` row, move its aliases, then delete the old row.
- **Old completed jobs keep old names in their stored metadata.** They are left as history.
- **Cleanup makes rollback impossible.** Hence the waiting period and the separate deploy.
- **S3 versioning is unknown.** Our keys get AccessDenied on `GetBucketVersioning`. So the repair copies every S3 object to `backups/speaker_registry_v1/` before overwriting it, and never relies on versioning.
- **uat and release (prod) share Postgres, ChromaDB, Redis and S3.** One run fixes both environments, and whichever deploys first runs it. The lock and the done marker live in the shared Postgres, so it runs once. Consequences:
  - Both environments must be on the new code **before** deploy 2. Old code re-seeds the old `ALIASES` on every restart, and its `04b` writes old spellings, which would undo the repair.
  - Schema changes must be additive only (new tables; new columns nullable or with defaults), so the old code keeps working during the rollout.
  - Rollback and cleanup affect both environments at once. Schedule them for both.
  - If dev also uses `ai4ia-data`, the first environment to apply writes an `s3://ai4ia-data/backups/speaker_registry_v1/_FILES_APPLIED` marker. Later environments skip the S3 step and fix only their own Postgres and ChromaDB.
  - Redis (the storyline cache) is shared too, so both environments must set the same new `STORYLINE_CACHE_CORPUS_VERSION`.

## Out of scope

- Naming the speaker on each panel clip. This needs diarization or detecting self-introductions.
- The admin endpoints: the repair report and the speaker review queue (review table item 5).
- The 84 register people with no video in the corpus.
