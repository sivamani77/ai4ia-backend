# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions: [SemVer](https://semver.org/).
Each PR adds a line under `Unreleased`. At release time, rename `Unreleased` to the version and date.

## [Unreleased]

## [1.0.0] - 2026-09-26

First tagged release. It covers everything merged since the last production deploy
(`461b5fc`). The frontend must ship F-1 and F-5 from
`docs/security/frontend-changes-for-security-fixes.md` before this deploys.

### Added
- Intent layer: prompts are classified and resolved before the storyline, with a format and clip policy per intent (overview, panel speaker, theme and chapter).
- Archive coverage check: prompts about speakers, chapters or years the archive lacks are declined before any LLM call.
- Speaker registry, and a switch-controlled speaker repair (applied from uat on 2026-09-23).
- Job recovery: leases, heartbeats and a sweeper fail jobs whose worker is lost, and re-send lost queued jobs.
- Per-environment video queue (`WORKER_QUEUE`).
- Per-user and platform video job quotas.
- Feedback, notifications and tour-seen endpoints, replacing direct Supabase calls from the frontend.
- Read-only admin data explorer.
- Record of each request and every LLM exchange (`prompt_resolutions`, `llm_call_payloads`), with an expiry sweep.
- Ingest resume for failed videos, and ingest that works on S3.
- Delivery process, issue and PR templates (`docs/github-templates/`) and per-stage delivery guides (`docs/runbooks/delivery/`).

### Changed
- Auth runs on Postgres instead of Supabase. Existing users are copied once on startup; the `supabase` dependency is removed.
- Access tokens are short-lived and revocable; clients renew them through `/api/auth/refresh`.
- Storylines are retrieved from all 309 videos.
- Speaker prompts use the speaker's own clips (#10).
- Docs are organised into `docs/{design,security,runbooks,reference,process}`.

### Fixed
- Final export: narration no longer freezes as a still image, video no longer runs ahead of its audio, and speaker lip sync no longer drifts.
- Panel clips are credited to the right speaker, and panel and overview videos keep real clips.
- Stage 07 no longer crashes on an empty video year from Chroma.
- A stale job snapshot can no longer un-cancel a running job.
- Ingest reports its real status and fails a stage when every video in it fails.
- Null `stream_url` in AI Insights, Theme Synthesis/Relay and mind map sources.
- S3 video links are signed for the bucket's regional host; a Range past the end returns 416.

### Security
- Closed anonymous and cross-user data access; `user_id` always comes from the token.
- Rate limits on OTP and reset codes, backed by Redis.
- Removed committed load-test tokens and a Supabase service key.
- Auth data older than 30 days is purged daily.
