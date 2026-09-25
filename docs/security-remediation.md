# Security remediation

Tracks the fixes for the findings in the AI4IA API & Code Validation report
(`MOB-AI4IA-VAL-2026-09`, run on 2026-09-24 against `uat @ b6f39e1`). Each fix
has an entry below. Update the entry when a fix lands or is deployed.

Frontend work that goes with these fixes is in
[frontend-changes-for-security-fixes.md](frontend-changes-for-security-fixes.md).

| | |
|---|---|
| Working branch | `develop` (cut from `uat` @ `58aa7e8`, 2026-09-25) |
| Promotion | `develop` → `uat` (check on the test service) → `release` |
| Scope | 2 critical and 12 high findings first; medium and low afterwards |
| Owner | Backend team |

## Contents

1. [Status](#1-status)
2. [Plan](#2-plan)
3. [Fix records](#3-fix-records)
4. [Open decisions](#4-open-decisions)
5. [How each fix is verified](#5-how-each-fix-is-verified)
6. [Deployment notes](#6-deployment-notes)

---

## 1. Status

Status values: **Open**, **In progress**, **Fixed on develop**, **On uat**, **On release**.

| ID | Sev | Finding | PR | Status |
|---|---|---|---|---|
| A-01 | Critical | Archive and dashboard served to anonymous callers | 1 | Fixed on develop |
| C-01 | Critical | Insights have no owner; every user reads every user's insights | 1 | Fixed on develop |
| C-02 | High | Concept mindmap reads any user's storyline | 1 | Fixed on develop |
| C-03 | Medium | Path traversal through `storyline_id` | 1 | Fixed on develop |
| C-14 | Medium | Ownership check admits jobs with no owner | 1 | Fixed on develop |
| C-15 | Medium | Insight ids are timestamps | 1 | Fixed on develop |
| C-05 | High | Password-reset codes can be guessed without limit | 2 | Fixed on develop |
| C-06 | High | Signing and API keys fail open when unset | 2 | Fixed on develop (deploy gate) |
| A-03 | High | 401 responses omit `WWW-Authenticate` | 2 | Fixed on develop |
| C-13 | Medium | Client IP is the proxy's IP | 2 | Fixed on develop (verify on uat) |
| C-16 | Medium | OTP attempts and refresh rotation are not atomic | 2 | Fixed on develop |
| C-26 | Low | Verifying a reset code consumes it before the reset | 2 | Fixed on develop |
| A-14 | Medium | Production CORS list trusts `http://localhost:8080` | 2 | Repo fixed; cluster variable pending |
| C-04 | High | Access tokens live 30 days and cannot be revoked | 3 | Fixed on develop; lifetime change waits for frontend F-5 |
| C-08 | High | Signed user tokens committed to the repo | 3 | Fixed on develop |
| NEW-01 | High | Supabase service-role key committed in a k8s example | 3 | Removed on develop; **rotation needed in Supabase** |
| Q-01 | High | Per-user job quota can be exceeded by concurrent requests | 4 | Fixed on develop |
| Q-03 | Medium | A failed dispatch leaves a queued job that uses quota forever | 4 | Fixed on develop |
| Q-04 | Medium | Quota evaluation scans the whole jobs table on every request | 4 | Fixed on develop |
| Q-06 | Low | Quota limits are hard-coded | 4 | Fixed on develop (limits are settings) |
| C-10 | High | Unpinned dependencies, stale lockfile, no scanning | 5 | Open |
| C-11 | High | CI has no quality gates; branch conditions wrong | 5 | Open |
| A-02 | High | Secured operations document no 401/403 | later | Open |
| A-05 | High | Errors untyped; no RFC 9457 problem details | later | Open |
| A-06 | High | No idempotency on unsafe operations | later | Open |

Mediums and lows pulled into the PR that touches the same code: C-13, C-16,
C-26 and A-14 went in PR 2; Q-03 and Q-04 go in PR 4.

---

## 2. Plan

The PRs land in order, and each PR is reviewed before the next one starts.
Every fixed finding gets a regression test that fails on the old code. The
report's generator (`validation/run_validation.sh`) is not in this repo, so
these tests are what keeps the findings from coming back.

| PR | Theme | Findings | Frontend dependency |
|---|---|---|---|
| 1 | Close anonymous and cross-user data access | A-01, C-01, C-02, C-03, C-14, C-15 | Archive search must send the bearer token **before** this deploys |
| 2 | Credential paths | C-05, C-26, C-16, C-06, C-13, A-03, A-14 | Handle 429 on forgot-password |
| 3 | Token lifetime and revocation | C-04, C-08 | Global refresh-on-401 **must ship first** |
| 4 | Job quota as a guarantee | Q-01, Q-03, Q-04 | None |
| 5 | Build and CI gates | C-10, C-11 | None |
| later | API contract | A-02, A-05, A-06 | Error-shape handling |

### PR 2: Credential paths (done)

See the fix records in section 3. It has two deploy gates, listed in section 6.

### PR 3: Token lifetime and revocation (done)

See the fix records in section 3. The code is on `develop`. Merge to uat
only after frontend F-5 is live, because every existing session is asked to
refresh at deploy.

### PR 4: Job quota (done)

See the fix records in section 3. It doesn't depend on the frontend.

### PR 5: Build and CI (planned)

- **C-10:** one hashed lockfile, installed with `--require-hashes`, and `pip-audit` on every PR.
- **C-11:** a `pull_request` workflow running pytest, pip-audit and a secret scan. Pin the reusable `gaiangroup/access` workflows to commit SHAs. Fix the unauthorised-branch guard (it checks `release` twice) and `prod_job` (it keys on a branch that is never pushed).

---

## 3. Fix records

Each record uses the same fields: finding, problem, root cause, fix, files,
tests, deployment impact, frontend impact.

### A-01: Archive and dashboard served to anonymous callers

| | |
|---|---|
| Severity | Critical: OWASP API2:2023, CWE-306 |
| Status | Fixed on develop |
| Verified before the fix | Local API: `GET /api/archive/speakers` with no credentials returned 200 and 43,552 bytes; a bogus `X-API-Key` returned 200 |

**Problem.** The archive and dashboard routers were protected only by the
shared API key. `require_api_key` lets every request through when `API_KEY`
is empty, and it is empty on uat and release. So speaker lists, fragment text,
semantic search and source-video streaming were public. The dashboard also
gave away internal paths, the S3 bucket URI, model names, and the file name
of the most recent job's export. That file name carries the other user's
prompt, for example `…_ai_law_and_human_rights.mp4`.

**Root cause.** The routers never required a signed-in user
(`main.py`, `include_router(..., dependencies=_auth_dep)`).

**Fix.**
- The archive router now requires a signed-in user: new `_user_dep` = API key plus `get_current_user`.
- `GET /api/dashboard` is the only public endpoint, because the landing page calls it before login. It now returns only `{"archive": {"raw_videos", "classified_fragments", "speakers", "themes"}}`, cached for 5 minutes per process, so anonymous traffic no longer triggers S3 listings on every call.
- A count that fails comes back as `null` (the frontend falls back to its defaults), and that result is cached for only 15 seconds, so a brief S3 or database outage doesn't leave zeros on the landing page.
- The full dashboard moved to the admin-only `GET /api/admin/dashboard`.
- `speakers` and `themes` are new. The landing page hard-coded 71 and 7. `speakers` counts speakers with footage in the archive (the same list `/api/archive/speakers` returns, about 307 on current data), so the landing page matches the Speakers page.

**Files.** `backend/app/main.py`, `backend/app/api/dashboard.py`,
`backend/app/services/dashboard_service.py`.

**Tests.**
- `tests/api/test_dashboard_public.py`: anonymous payload shape, caching, and admin-only access to the full view.
- `tests/api/test_catalog_routes.py::test_archive_requires_a_signed_in_user`.

**Deployment impact.** None on the database.

**Frontend impact.** F-1, F-2, F-3. F-1 must ship first, or archive search
breaks for signed-in users.

### C-01: Insights have no owner

| | |
|---|---|
| Severity | Critical: CWE-639, OWASP API1:2023 (BOLA) |
| Status | Fixed on develop |

**Problem.** The `insights` table had no owner column. `GET /api/insights`
listed every user's insights, and `GET /api/insights/{id}` with `/sources` and
`/provenance` fetched by id alone. Provenance includes the decision record,
which carries the original prompt.

**Root cause.** `InsightRepository.create()` dropped the `user_id` the service
already passed in, and nothing filtered reads by owner.

**Fix.**
- `insights.user_id` column and a `(user_id, created_at)` index, added in `db/schema.py`.
- A one-time backfill fills `user_id` from `agent_decision_records.insight_id`. It only touches rows where `user_id` is null, and it is skipped if that table does not exist. It assigns an owner only when exactly one user has a decision record for the id: old ids only go down to the second, so two users can share one, and those rows stay ownerless rather than being given to the wrong person.
- The owner is saved on create.
- List, get, sources and provenance filter by the caller's id. Admins see every insight.
- Someone else's insight returns 404, not 403, so its existence is not disclosed.
- Insights that stay ownerless after the backfill are visible to admins only.
- `hard_delete_user` now deletes the user's insights too.

**Files.** `backend/app/db/schema.py`,
`backend/app/db/repositories/insight_repository.py`,
`backend/app/services/insight_service.py`, `backend/app/api/insights.py`,
`backend/app/services/admin_user_service.py`.

**Tests.**
- `tests/api/test_insights_ownership.py`: owner-only list, 404 on another user's insight and on ownerless ones, admin sees all, and create records the owner.
- `tests/integration/test_insight_owner_backfill_postgres.py`: the backfill against real Postgres. Unambiguous owner assigned; colliding id left null; existing owner kept; idempotent; skipped when there are no decision records.
- `tests/unit/test_job_ownership.py::test_hard_delete_user_removes_their_insights`.

**Deployment impact.** The column and index are additive, and the backfill is idempotent.

**Rollout window.** Release code on the shared Postgres ignores the new column, and its inserts leave it null. So insights generated through release while uat runs the new code have no owner: they are admin-only and `hard_delete_user` does not remove them. The backfill picks them up at the next uat or release restart. The frontend never lists insights, so no user sees a difference. The window closes when release is upgraded.

**Frontend impact.** None. The frontend calls only
`POST /api/insights/generate`.

### C-15: Insight ids are timestamps

| | |
|---|---|
| Severity | Medium: CWE-340 |
| Status | Fixed on develop |

**Problem and fix.** `insight_YYYYMMDD_HHMMSS` collided when two insights were
generated in the same second, and the ids could be enumerated. Ids are now
`insight_<uuid4 hex>`. Existing ids are unchanged.

**Files.** `backend/app/services/insight_service.py`.

**Frontend impact.** None. Nothing parses the id.

### C-02 and C-03: Concept mindmap reads any user's storyline; path traversal

| | |
|---|---|
| Severity | High (C-02: CWE-639, CWE-200); Medium (C-03: CWE-22) |
| Status | Fixed on develop |

**Problem.**
- `_resolve_storyline` accepted any `storyline_id`: a storyline file name, another user's job id, or a substring of an S3 key.
- With no id, it ranked every user's storylines against the query.
- The response returned the source storyline's `user_prompt` and the absolute server path.
- The id was joined into a file path without validation, so `../..` could reach other JSON files on the host.

**Fix.**
- `storyline_id` must match `^[A-Za-z0-9_-]+$`. Anything else is a 404.
- Non-admins resolve storylines only through jobs they own. The id must be one of their job ids (what the frontend sends) or the file stem of one of their jobs' storylines.
- With no id, only the caller's own storylines are ranked, and at most the 20 newest of them, because each one can be an S3 read.
- Admins keep the old resolver.
- The absolute `path` is no longer returned. `user_prompt` is still returned, because it is now always the caller's own prompt and the documentary PDF shows it.
- A 500 no longer echoes the exception text.
- The duplicated scoring code was merged into `_score_storyline`.

**Files.** `backend/app/services/concept_mindmap_service.py`,
`backend/app/api/concept_mindmap.py`.

**Tests.** `tests/unit/test_concept_mindmap_service.py::TestOwnedStoryline`:
own job id and stem resolve; another user's job id, stem or substring does
not; ranking with no id uses only the caller's own storylines; path-shaped
ids are rejected.

**Frontend impact.** F-4 (no change is needed, but the behaviour is described
there).

### C-14: Ownership check admits jobs with no owner

| | |
|---|---|
| Severity | Medium: CWE-285, OWASP API1:2023 |
| Status | Fixed on develop |

**Problem and fix.** `_assert_owns` skipped the check when a job had no
`user_id`, so any signed-in user could open ownerless jobs and all their
media. It now treats a missing owner as "someone else's" and returns 404.
Admins still bypass the check. Users never saw ownerless jobs in their job
list, so no normal flow depends on the old behaviour.

**Files.** `backend/app/api/video_jobs.py`, `backend/app/api/video_media.py`.

**Tests.** `tests/unit/test_job_ownership.py`.

### C-05: Password-reset codes can be guessed without limit

| | |
|---|---|
| Severity | High: CWE-307, OWASP API2/API4:2023 |
| Status | Fixed on develop |
| Verified before the fix | Local API: 8 forgot-password calls all returned 200; 8 verify-otp calls all returned 400; no 429 |

**Problem.** forgot-password, verify-otp and reset-password had no limits.
Each forgot-password call issued a new code with a fresh budget of five
guesses, so an attacker could loop "new code, five guesses" forever against
a 10⁶ code space. The endpoint could also be used to flood any inbox.

**Fix.** Three tiers of limits on requests (forgot-password, resend-otp) and
on guesses (verify-otp, reset-password):

| Tier | Requests / hour | Guesses / hour | Why |
|---|---|---|---|
| e-mail + client IP | 5 | 10 | Tight, but an attacker elsewhere can't use it to lock the owner out |
| e-mail only | 20 | 30 | Caps an attacker who rotates IPs |
| client IP only | 30 | 60 | Caps one source trying many addresses |

- **Daily limits:** codes are capped at 10 per e-mail+IP and 30 per e-mail each day. Guesses are capped at 20 per e-mail+IP and 100 per e-mail and purpose each day, however many codes or IPs are used. 100 is the NIST SP 800-63B ceiling for failed attempts per account.
- **Before the account lookup:** the limits apply first, so known and unknown addresses get the same answers.
- **Silent on reset:** the daily cap looks like success on the reset paths, so it can't reveal an account either. Elsewhere it is a 429.
- **Retry-After:** every 429 carries it, read from the Redis window's TTL.
- **Proxy fallback:** if the client IP can't be resolved, or resolves to a proxy address, the IP tiers are skipped rather than shared by everyone.
- **Settings:** all limits are in `settings.py` (`OTP_*`).

**Known trade-off.** Any per-account limit can be used against the account's
owner. The tiers make that expensive:
- **Codes:** using up the daily code cap takes at least three IPs. Every one of those codes is e-mailed to the owner, so the owner can use the latest one while it is valid (10 minutes).
- **Guesses:** using up the daily guess cap takes 100 wrong guesses from at least five IPs.

The worst case is that the owner waits for the daily window to reset. This
is accepted: a tighter per-account limit makes lockout easier, and a looser
one makes guessing easier.

**Files.** `backend/app/services/auth_service.py`, `backend/app/api/auth.py`,
`backend/app/core/rate_limit.py` (`retry_after`), `backend/app/core/settings.py`.

**Frontend impact.** F-6.

### C-16: OTP attempts and refresh rotation are not atomic

| | |
|---|---|
| Severity | Medium: CWE-362, CWE-367 |
| Status | Fixed on develop |

**Problem.** The attempt counter was read and then written, and "is the code
or token unused?" was checked separately from marking it used. Parallel
requests could exceed the attempt cap, redeem one code twice, or rotate one
refresh token twice.

**Fix.**
- **OTP attempts:** `OtpRepository.claim_attempt` counts the guess in one conditional `UPDATE … WHERE used_at IS NULL AND attempts < max RETURNING`, before the code is checked.
- **OTP use:** `mark_used` only succeeds for the first caller.
- **Refresh tokens:** `RefreshTokenRepository.mark_used` claims the token the same way.
- **Refresh grace window:** a refresh token presented again within 30 seconds of its rotation, by the same user agent, gets a retryable 401 (`refresh_superseded`) and the session survives. This is the two-tabs case. Any other replay still revokes the whole session family.

**Files.** `backend/app/db/repositories/otp_repository.py`,
`backend/app/db/repositories/refresh_token_repository.py`,
`backend/app/services/auth_service.py`.

**Frontend impact.** F-5: retry once on `refresh_superseded`.

### C-26: Verifying a reset code consumes it before the reset

| | |
|---|---|
| Severity | Low: CWE-841 |
| Status | Fixed on develop |

**Fix.** `/verify-otp` with `purpose=password_reset` now checks the code
without using it up, and `/reset-password` redeems it. A check followed by a
reset counts as two attempts against the code's five. The frontend doesn't
call verify-otp for resets, so nothing changes for users.

### C-06: Signing key fails open when unset

| | |
|---|---|
| Severity | High: CWE-1188, CWE-636 |
| Status | Fixed on develop. **Deploy gate:** confirm the key length first |

**What was actually true.** The report said an empty key would let anyone
sign admin tokens. With the pinned PyJWT 2.13, an empty key is refused
outright, so auth failed closed, with a 500 rather than a 401. The real risks
were:
- **A future library change:** that protection came from the library, not from our code.
- **A short key:** it can be brute-forced offline from any issued token, including the 100 in `k6/loadtest_users.json`.

**Fix.**
- `security._signing_key()` refuses an empty key itself.
- Any JWT library error on a presented token is a 401, never a 500.
- The API refuses to start when a key is set but shorter than 32 bytes (`JWT_MIN_KEY_BYTES`).
- An empty key only logs a warning, so local pipeline work without sign-in still runs.

`API_KEY` stays optional on purpose. The frontend would have to ship it in
the browser bundle, so it protects nothing. Signed-in-user checks (PR 1) are
the real control.

### C-13: Client IP is the proxy's IP

| | |
|---|---|
| Severity | Medium: CWE-348 |
| Status | Fixed on develop. **Verify on uat after deploy** |

**Fix.** `core/client_ip.py` reads `X-Forwarded-For` from the right and skips
addresses in `TRUSTED_PROXY_CIDRS` (loopback, private ranges, `100.64/10`).
The first address that isn't a proxy is the client. Entries a client adds on
the left are never reached, so they can't spoof it. It is used for the login
limiter key, session and audit IPs, and the new per-IP limits.

**Verify on uat.** After the deploy, check that new `auth_audit` and
`user_sessions` rows show public client addresses, not one private address.
If they all show the same private address, the ingress is not forwarding
the client address. The per-IP limits then switch themselves off (see
C-05), and the ingress config needs fixing.

### A-03: 401 responses omit the WWW-Authenticate challenge

| | |
|---|---|
| Severity | High: RFC 9110 §11.6.1, RFC 6750 §3 |
| Status | Fixed on develop |

**Fix.**
- `get_current_user` sends `Bearer realm="ai4ia"`, adding `error="invalid_token"` for bad or expired tokens.
- One exception handler adds the default challenge to every other 401.

### A-14: Production CORS list trusts localhost

| | |
|---|---|
| Severity | Medium: CWE-942 |
| Status | Removed from the repo `.env`. **Action for DevOps:** remove `http://localhost:8080` from the `ALLOWED_ORIGINS` cluster variable on uat and release |

The deployed value is a separate cluster variable (185 bytes, which differs
from the repo copy), so a code change can't fix it.

### C-04: Access tokens live 30 days and cannot be revoked

| | |
|---|---|
| Severity | High: CWE-613, OWASP API2:2023 |
| Status | Fixed on develop. The lifetime change is a cluster setting, made after frontend F-5 ships |

**Problem.**
- Access tokens lived 30 days (`ACCESS_TOKEN_TTL_SECONDS=2592000` on the cluster) and carried no version.
- `get_current_user` trusted the token's persona claim without reading the account.
- Demotion, deactivation, password reset and logout revoked only refresh tokens, so an access token kept working until it expired.

**Fix.**
- **Versioned tokens:** `app_users.token_version` is a new column, and every access token carries the version it was minted with (`ver`).
- **Checked on every request:** `get_current_user` compares `ver` with the account, and also rejects deleted or deactivated accounts. The account row is cached per process for 15 seconds, so revocation reaches every replica within 15 seconds without a database read on every request.
- **Newer tokens are re-read:** a token whose version is newer than the cached one (minted after a change another replica made) causes a fresh read instead of a rejection. A load that started before a revocation never caches the stale result.
- **Persona from the account:** persona now comes from the account, not the token, so a demoted admin loses admin rights immediately.
- **Retryable rejection:** a stale or version-less token gets `401 token_expired`, the same code as expiry, so the frontend refreshes.
- **Database outage:** returns 503, not 401, so users aren't signed out.
- **What bumps the version** (ending every access token the user holds, on every device):
  - password reset or change
  - persona upgrade (the response carries the new token)
  - admin persona change
  - admin password reset
  - deactivation
- **Refresh:** now also refuses deactivated accounts.
- **Lifetime:** the code default is 15 minutes. Startup logs a warning when `ACCESS_TOKEN_TTL_SECONDS` is over an hour.

**Not covered.** Logging out one session doesn't end that session's access
token; it expires within the token lifetime (15 minutes once the cluster
setting changes). "Log out everywhere" events do end all tokens.

**Files.** `backend/app/core/auth.py`, `backend/app/core/security.py`,
`backend/app/db/schema.py`, `backend/app/db/repositories/user_repository.py`,
`backend/app/services/auth_service.py`, `backend/app/services/admin_user_service.py`,
`backend/app/core/settings.py`.

**Tests.** `tests/unit/test_token_revocation.py`.

**Rollout.**
1. Frontend F-5 (global refresh on 401) goes live.
2. Deploy this backend. Every token minted before it lacks `ver`, so each open session makes one silent refresh.
3. Set `ACCESS_TOKEN_TTL_SECONDS=900` on uat, then release, or remove the variable to use the default.

If the backend deploys before F-5, each signed-in user is sent back to the
login page once. Nothing else breaks.

### C-08: Signed user tokens committed to the repository

| | |
|---|---|
| Severity | High: CWE-798, CWE-540 |
| Status | Fixed on develop |

**Fix.**
- `k6/loadtest_users.json` (100 signed tokens) is no longer tracked, and `k6/loadtest_users*.json` is git-ignored. `k6/k6.js` reads the file named by `USERS_FILE`.
- The tokens remain in git history, but they have no `ver` claim, so C-04 makes every one of them invalid. The signing key doesn't need rotating, provided it is at least 32 bytes and random (PR 2 gate).

### NEW-01: Supabase service-role key committed in a k8s example

| | |
|---|---|
| Severity | High: CWE-798. **Not in the report**; found while fixing C-08 |
| Status | Removed on develop. **Action: rotate the key in Supabase, or shut the project down** |

**Problem.** `examples/local-dev/k8s/02-secret.yaml.example`, tracked since
2026-07-02, contained:
- a real service-role key for Supabase project `rzcaziwizcektrlrdujr`, valid until 2036;
- a 10-character `API_KEY` value.

A service-role key bypasses row-level security and gives full access to that
project, which was the backend's pipeline and auth database before the move
to Postgres.

**Fix.** Both values are replaced with placeholders. The key is still in git
history, so the real fix is outside the repo: rotate the service-role key in
the Supabase dashboard, or delete the project if nothing uses it any more.
The frontend uses a different project (`yiwycwaziewmyenfruyg`), so this
doesn't affect it.

A scan of all tracked files for other key patterns (AWS, private keys, API
tokens, JWTs) found nothing else. The only hit was the fake key used in
`tests/unit/test_presigned_url.py`.

### Q-01, Q-03, Q-04, Q-06: Video job quota

| | |
|---|---|
| Severity | High (Q-01: CWE-367, OWASP API4:2023); Medium (Q-03, Q-04); Low (Q-06) |
| Status | Fixed on develop |

**Problem.**
- **Q-01:** the quota (1 running and 3 queued per user, 50 active on the platform) was a read followed by an insert, with nothing in between, so parallel submissions all passed the check.
- **Q-04:** each submission loaded every job on the platform twice (`select * from jobs`, with full JSON metadata), and the job list loaded it again, then filtered and paged in Python.
- **Q-03:** if the job queue was unreachable, the job row stayed `queued` forever and counted against its owner. Three failed submissions locked the user out.

**Fix.**
- **Atomic quota (Q-01):** `JobRepository.create_job_within_quota` runs the count and the insert in one transaction, under a Postgres advisory lock shared by every replica. The count is one `count(*) FILTER` over queued and running jobs, which uses the existing `status` index.
- **SQL paging (Q-04):** the job list pages in SQL (`LIMIT/OFFSET` plus a window count), and the queue position is a SQL count.
- **Dispatch rollback (Q-03):** if dispatch fails, or the job's bookkeeping fails after the insert, the job goes through the normal failure path (`completed_at`, a "failed" event, a user message, and the processing record closed) before the error is returned, so it no longer counts.
- **Lock timeout:** a submission waits at most 5 seconds for the quota lock, then gets 503 "busy", so a burst can't drain the connection pool.
- **Limits in settings (Q-06):** `VIDEO_JOB_MAX_RUNNING_PER_USER`, `VIDEO_JOB_MAX_QUEUED_PER_USER` and `VIDEO_JOB_MAX_ACTIVE`.
- **New helper:** `db/postgres.py` gained `pg_transaction()` for multi-statement transactions.

**Tests.**
- `tests/integration/test_job_quota_postgres.py`: 8 parallel submissions against real Postgres; exactly 3 are accepted.
- `tests/api/test_video_jobs_and_media.py`: each limit, dispatch rollback, paging.

**Before release.** The job list now filters on the `user_id` column. Rows whose owner sits only in metadata would drop out of their owner's list. There are none locally; confirm on uat:
`select count(*) from jobs where coalesce(user_id,'') = '' and coalesce(metadata->>'user_id','') <> ''` should return 0.

**Not included.** Q-02 (`Retry-After` on these 429/503s), Q-05 (daily model budgets) and Q-07 (the fixed-window limiter) are still open.

---

## 4. Open decisions

| # | Decision | Needed for | Owner |
|---|---|---|---|
| 1 | The landing page's speaker count changes from the hard-coded 71 to the real count (about 307). Confirm this is the definition wanted: speakers with footage in the archive. | PR 1 frontend | Product |
| 2 | When to set `ACCESS_TOKEN_TTL_SECONDS=900` on the cluster. It follows the frontend refresh work (F-5); the code side is done. | PR 3 | Backend + frontend |
| 3 | `JWT_SIGNING_KEY` on uat is 64 characters, which clears the PR 2 gate (confirmed 2026-09-25). Release still needs the same check. Most of the key was pasted into a chat, so rotate it with the PR 3 deploy (decision 4). | PR 2 gate (release) | DevOps |
| 4 | Rotate `JWT_SIGNING_KEY` when PR 3 deploys (`openssl rand -base64 48`, written straight into the cluster Secret). Only access tokens depend on it; refresh tokens are random values checked in the database. Once frontend F-5 is live, rotation costs each user one silent refresh, and PR 3 already causes one at that moment. | PR 3 deploy | DevOps |
| 5 | Rotate the Supabase service-role key for project `rzcaziwizcektrlrdujr`, or shut the project down (NEW-01). | Now, independent of deploys | Whoever owns the Supabase account |

**Note on decision 3: how tokens are signed today.**
- `JWT_SIGNING_KEY` is not in the uat/release variable list, but it must be set somewhere.
- The image pins `pyjwt==2.13.0`, which raises `InvalidKeyError: HMAC key must not be empty` when signing with an empty key. So an empty key would make every login fail with a 500. Logins work on uat, and the report measured a 30-day token issued there.
- `DATABASE_URL` and the object-storage keys are also missing from that list, so there is a second source of environment variables, most likely a Kubernetes Secret or the deploy repo.
- **Confirmed on uat, 2026-09-24 17:04 pod start** (`ai4ia-backend-test`): the only startup warning is `API_KEY is not set`. The `JWT_SIGNING_KEY is not set` warning is absent, so the key is set.
- Same log, relevant to other findings:
  - `API_KEY is not set` confirms A-01's root cause on uat.
  - The Celery worker runs as root (`uid=0`), as C-12 describes.
  - The worker consumes `ingest_jobs` and `video_jobs_uat`, so uat uses its own video queue on the shared Redis.

---

## 5. How each fix is verified

1. Regression tests for the finding (listed in each record) pass, and fail when the fix is reverted.
2. The full suite passes: `.venv/bin/python -m pytest`.
3. A supervising review of the PR's diff looks for bypasses, edge cases and regressions. Its findings are fixed and re-checked before the PR moves on.
   - PR 4 review, 2026-09-25: no blockers. Fixed: a failed dispatch skipped the normal failure bookkeeping; a bookkeeping error after the insert left a job holding a queued slot; the lock wait had no timeout. Pre-release check added for legacy owner rows.
   - PR 3 review, 2026-09-25: no blockers. Fixed: a token newer than another replica's cache was rejected for up to 15 seconds; a load racing a revocation could cache stale state; the real account-row mapping had no test. Also fixed a test that failed at random: it tampered with the signature's last base64 character, which can carry only padding bits.
   - PR 2 review, 2026-09-25: one deploy gate, four should-fix items, three notes. Fixed: resend-otp bypassed the new limits; per-e-mail limits let an attacker lock the owner out, so limits are now tiered by e-mail+IP; per-IP limits could turn global if IP resolution collapsed, so they are skipped for proxy addresses; the refresh grace window now requires the same user agent. Documented rather than changed: the two-attempt cost of check-then-reset, and existing enumeration signals (C-17).
   - PR 1 review, 2026-09-25: six findings. Fixed: the backfill could give an insight to the wrong user when ids collided; a failed dashboard count was cached as 0 for 5 minutes; the no-id mindmap path was unbounded; three test gaps. Checked and needed no change: archive media playback, because the frontend fetches it with the bearer token.
4. Local probes run against an API started with local Postgres, Redis and storage only, never against the shared uat/release stores.
5. After the uat deploy, the anonymous probes from the report are re-run:

```bash
H=https://uat-api.ai4ia.io
curl -s -o /dev/null -w '%{http_code}\n' $H/api/archive/speakers   # expect 401
curl -s $H/api/dashboard                                          # expect only archive counts
curl -s -o /dev/null -w '%{http_code}\n' $H/api/admin/dashboard   # expect 401
```

---

## 6. Deployment notes

- **Shared stores.** uat and release share Postgres, ChromaDB, Redis and S3. Schema changes stay additive, so release keeps working while uat runs newer code.
- **PR 1 order.** Deploy the frontend change F-1 (archive search sends the token) first, then the backend. Deploying the backend first breaks archive search until the frontend catches up.
- **AdminSettings.** It needs F-3 (switch to `/api/admin/dashboard`). Until then the model names and storage counts on that page are empty. Nothing else breaks.
- **PR 2 deploy gates:**
  1. Before pushing to uat, confirm that `JWT_SIGNING_KEY` on uat, and later on release, is at least 32 bytes. A shorter key stops every replica from starting. If it is shorter, rotate it first; this signs all users out.
  2. After the uat deploy, check the client IPs in `auth_audit` (see C-13).
- **PR 2 cluster change:** remove `http://localhost:8080` from `ALLOWED_ORIGINS` on uat and release (A-14).
- **PR 3 order:** frontend F-5 goes live, then this backend, then `ACCESS_TOKEN_TTL_SECONDS=900` on the cluster. At deploy, every open session refreshes once.
- **PR 3 schema:** `app_users.token_version` is additive. Release's old code ignores it and keeps working on the shared Postgres until it is upgraded.
