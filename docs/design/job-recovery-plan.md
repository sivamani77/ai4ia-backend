# Video job recovery: design plan

**Scope:** detect jobs whose worker is lost and recover them, and stop
long jobs from being run twice. This is items 1 and 2 of the recovery review.
Resuming from the last completed stage (item 3) and worker shutdown on deploys
(item 4) are out of scope; section 10 notes how this design prepares for them.

**Status:** built on `develop` (2026-09-26); see section 15 for what was built and how it was verified. It was revised after an adversarial review; the review's findings and where each is addressed are in section 14. **Owner:** backend.

### Decisions (2026-09-25)

These override anything below that says otherwise.

| # | Decision | Effect on the design |
|---|---|---|
| 1 | ~~No automatic retry.~~ **Retry on worker loss only** (issue #19, 2026-09-26, reverses the original decision). Up to 3 retries, 4 runs in total. Stage errors, cancellation and the runtime cap are never retried. | R1 requeues while retries are left, otherwise fails. A redelivered message takes over a dead holder's job while retries are left. `JOB_MAX_WORKER_LOSS_RETRIES` = 3 (0 restores the no-retry behaviour). See §16. |
| 2 | **Environment marker is the queue name.** uat uses `video_jobs_uat`, release uses `video_jobs`. | No `AI4IA_ENV`. `worker_queue` is the marker. Startup warns if a sweeper runs on the default `video_jobs` while `AI4IA_STORAGE_BACKEND=local` (a developer machine pointed at shared stores). |
| 3 | **Job cap of 35 minutes** (`JOB_MAX_RUNTIME_SECONDS=2100`). Real jobs finish well inside 30 minutes, and users won't watch a loading screen longer. Queue wait is capped separately at 30 minutes (`JOB_QUEUED_MAX_AGE_SECONDS=1800`). | Celery hard `time_limit` = 40 min. That is below the broker's default 1-hour visibility timeout, so a long job is **never** handed out twice. Item 2 (visibility timeout) needs no Redis change on either environment; it's set explicitly to 3600 s and validated at startup to stay above the time limit. |
| 4 | **Worker shutdown grace** is a separate change in the deploy repo. | Out of scope here. |

---

## 1. What happens today

Verified in code on `develop` @ `3bb91c4`:

| Event | Current behaviour | Effect on users |
|---|---|---|
| Worker child process killed (OOM) | Celery re-queues at once (`task_acks_late` + `task_reject_on_worker_lost`); `_claim_delivery` allows 2 deliveries | Job restarts from the beginning; a second death fails it as `poison_pill` |
| Whole worker pod dies (node loss, crash, rollout past the grace period) | Message sits in Redis `unacked` until the visibility timeout. It isn't set, so it is kombu's default of **3600 s** | Job shows `running` with no progress for up to **1 hour**, then restarts from the beginning. It holds the user's single "running" slot all that time (PR 4 quota). |
| Job legitimately runs > 1 h | kombu restores the still-unacked message after 3600 s and another worker runs it | **The same job renders twice at once**, writing the same S3 keys. Worst case per job ≈ 2.5 h (600 warmup + 600 + 1800 + 1200 + 4200 export for a 600 s video + 120 + 600), so this is reachable. |
| Stage error (script failure, stage timeout, LLM/S3 error) | Job marked `failed`, no retry (`max_retries=0`) | User resubmits. **Correct; unchanged by this plan.** |
| API pod starts with Redis unreachable | `_reap_orphaned_jobs` fails **every** queued/running job, including ones workers are rendering (finding C-23) | Live jobs are wrongly failed |
| Redis restarts without persistence | Queued messages are lost; jobs stay `queued` forever | Jobs never run and hold quota slots |
| Worker child dies while a script runs | The script subprocess is in its own process group (`setsid`) and keeps running | An orphan ffmpeg/whisper keeps using CPU and can write to the job's temp directory while the retry runs |

**Shared-infrastructure facts that shape the design:**
- **uat and release share Redis** (and Postgres). kombu keeps one `unacked` set per Redis database, and *any* consumer restores entries older than *its own* visibility timeout, whatever the queue (`kombu/transport/redis.py`: `unacked_key = 'unacked'`; `maybe_restore_messages` says they "are not local to channel"). So the **shortest visibility timeout among all workers on that Redis database wins**. Raising it on uat alone changes nothing while release workers run the default.
- **uat and release share the `jobs` table.** Any sweeper must only touch jobs its own environment dispatched, or uat would act on release's jobs and the other way round.
- **Old code keeps running during the rollout.** Release runs old code until promoted, so every new column and rule must be ignored safely by old code, and must not misread jobs created by old code.

---

## 2. Goals and non-goals

**Goals**
1. **Timely detection.** A job whose worker is gone is detected within ~5 minutes, not 60.
2. **Recovery:** such a job is automatically retried once, then failed with a clear message.
3. **Single run:** a job never runs on two workers at once, whatever the broker does.
4. **Lost queue messages:** jobs whose queue message was lost are re-dispatched.
5. **C-23 closed:** Redis reachability at API startup no longer decides anything about jobs.
6. **Per-environment:** everything is per environment on shared stores, and safe while old code runs.

**Non-goals**
- Resuming mid-pipeline (section 10).
- Retrying pipeline errors. A stage that fails for a real reason stays failed.
- Ingest jobs. The same pattern applies later; they have their own store and a thread fallback.

---

## 3. Design overview

Three mechanisms, each covering a different gap:

```
            dispatch (API)                       worker
 create ──► status=queued ──Celery msg──► CLAIM (lease) ──► run stages ──► finish (fenced)
            dispatch_id=D1                 heartbeat thread every 30s
            dispatched_at=now()            lease lost? → abort
                     ▲                                   │
                     │ re-dispatch (new D2)              │ no heartbeat for 5 min
                     └──────────── SWEEPER (API, one per env, every 60s) ◄──┘
```

1. **Lease (single run).** A worker may run a job only after winning an atomic *claim*:
   - it sets itself as `lease_owner`, and only if the lease is free or its holder's heartbeat is stale;
   - the message's `dispatch_id` must match the job's current one.

   Duplicate deliveries of any kind (redelivery, restore, sweeper re-dispatch) lose the claim and are dropped.
2. **Heartbeat (liveness).** A thread in the task process stamps `heartbeat_at` every 30 s, fenced on the lease. If the write shows the lease was lost (the sweeper took over, or the job was cancelled), the run aborts.
3. **Sweeper (recovery).** A background loop in the API, one leader per environment through a Postgres advisory lock, finds:
   - running jobs with a stale heartbeat;
   - queued jobs whose message is gone;
   - jobs past an absolute deadline.

   It re-dispatches them or fails them, with conditional updates so it can't race a live worker.

Plus configuration:
- **Visibility timeout** above the longest job, so the broker never restores a live job.
- **Celery hard time limit** as a backstop for a hung process.
- **Script process groups recorded on the job,** so a retry on the same pod kills anything left over.

---

## 4. Data model

New columns on `jobs`, all nullable or defaulted. Added in `db/schema.py` behind `to_regclass('jobs')`, because the table is created by `db/init/001_schema.sql`. Old code ignores them: `upsert_job` names its columns explicitly, so it never overwrites these.

| Column | Type | Meaning |
|---|---|---|
| `dispatch_id` | text | Id of the current valid Celery message. A new id on every (re)dispatch; messages carrying an older id are dropped. |
| `dispatched_at` | timestamptz | When the current message was sent (queue staleness is measured from this, not `created_at`). |
| `worker_queue` | text | Queue the job was dispatched to. It is the **environment marker** (decision 2): the sweeper only touches jobs with its own `WORKER_QUEUE`. |
| `lease_owner` | text | `hostname:pid:delivery_tag` of the worker running it; null when not running. |
| `script_pgid` / `script_host` | integer / text | Process group and pod of the script currently running, so a retry on the same pod can kill a leftover group (5.4). |
| `heartbeat_at` | timestamptz | Last heartbeat, from the **database clock** (`now()`). |
| `attempts` | integer not null default 0 | Claims so far. Replaces `metadata.delivery_attempts`, which is read as a fallback during rollout. |

Index: `create index if not exists idx_jobs_running_heartbeat on jobs (heartbeat_at) where status = 'running'`, plus a partial index for `status = 'queued'` on `dispatched_at`.

**Who may change `status`.** For rows created by the new code (non-null `dispatch_id`), only these write `status`:
- the claim;
- the sweeper;
- cancellation (`update_job_partial`);
- the fenced completion and failure writes.

`upsert_job`, the full-snapshot write that `update_stage`, `run_script`'s log flush and others use, keeps `jobs.status` unchanged on those rows (`status = case when jobs.dispatch_id is not null then jobs.status else … end`). A lagging old attempt can therefore never write `running` back over a job the sweeper has requeued. `load_job` already reads status from the column, so readers see the authoritative value. Rows from old code keep today's behaviour.

The status values are unchanged (the check constraint allows `queued|running|completed|failed|cancelled`), so no constraint migration is needed. "Interrupted" and "requeued" are expressed as events and error types, not new statuses.

---

## 5. Process-level recovery (inside the worker)

### 5.1 Claim

Replaces `_claim_delivery` in `worker/tasks.py`. One statement, database clock:

```sql
update jobs
   set status = 'running', lease_owner = %(me)s, heartbeat_at = now(),
       attempts = attempts + 1, started_at = coalesce(started_at, now())
 where id = %(job_id)s
   and status in ('queued', 'running')
   and (dispatch_id = %(dispatch_id)s or (%(dispatch_id)s is null and dispatch_id is null))
   and (lease_owner is null or heartbeat_at < now() - make_interval(secs => %(stale)s))
returning attempts
```

**Outcomes:**
- **Row returned, `attempts ≤ max`:** run.
- **Row returned, `attempts > max`:** fail the job as `worker_lost_repeatedly` (today's `poison_pill`), ack, stop.
- **No row:** the job is terminal (done or cancelled), superseded (a newer `dispatch_id`), or actively leased by a live worker. Ack and drop, and log which case by re-reading the row.

This one statement makes every duplicate harmless:
- Celery redelivery after an OOM: the lease holder's heartbeat is stale, so the retry takes over.
- kombu restoring a live long job: the holder is fresh, so the copy is dropped.
- A stale message after a sweeper re-dispatch: `dispatch_id` doesn't match, so it's dropped.

**Legacy messages** (queued before deploy, no `dispatch_id` argument) are claimed only while the job has no `dispatch_id`. The task signature becomes `run_video_job_task(self, job_id, request, dispatch_id=None)`, which is backward compatible.

`run_video_job` stops setting `status="running"` itself (the claim does) but keeps its terminal-state guard as a second check.

### 5.2 Heartbeat

- **Where it runs:** a daemon thread started by the task after a successful claim, stopped in `finally`. It lives in the prefork child, so it dies with the process. A dead process means no heartbeat, which is exactly the signal wanted.
- **What it writes,** every `JOB_HEARTBEAT_INTERVAL_SECONDS` (30):
  ```sql
  update jobs set heartbeat_at = now()
   where id = %s and lease_owner = %s and status = 'running'
  returning 1
  ```
- **No row returned:** the thread re-reads the row and sets `ctx.stop_reason` to the cause:
  - `cancelled` (status is cancelled);
  - `superseded` (another lease or `dispatch_id`);
  - `swept` (requeued or failed by the sweeper).
- **Database error:** log it and keep trying. The thread tracks the time of its last *successful* beat. If that's older than `0.8 × JOB_STALE_AFTER_SECONDS`, it sets `stop_reason = "heartbeat_lost"`: the worker assumes the sweeper can no longer see it and stops itself before a second run starts. This covers a worker cut off from the database while the API still reaches it, where the whole-database outage grace (6.4) doesn't help.
- **Stopping cleanly at the end:** the runner stops the heartbeat thread *before* the fenced completion write and before `_fail_job`. A finished job never triggers `stop_reason`, and the post-completion housekeeping (S3 upload, decision record, media registry) always runs.
- **Why a thread, not beats at stage boundaries:** the model warm-up can block for up to 600 s, and some scripts are silent for minutes. A thread covers every phase without touching each stage.

### 5.3 Abort on lost lease

- **During a subprocess:** `run_script`'s loop (`pipeline/base.py`) checks `ctx.stop_reason` on every pump tick (~2 s, no database read), as well as its existing status poll. When it's set, the loop kills the process group and routes by cause:
  - **`cancelled`:** raises the existing `JobCancelledException`, so the cancel event and processing record are written as today.
  - **`superseded` / `swept` / `heartbeat_lost`:** raises `JobSupersededException`. The runner ends quietly and records nothing, because the job belongs to another attempt, or will be taken over.
- **Between stages:** the runner checks `ctx.stop_reason` before each stage and before the post-completion housekeeping.
- **Terminal writes are fenced:** completion and `_fail_job` use conditional updates (`… where id = %s and lease_owner = %s`), so a superseded run can never mark the job completed or failed.

**Remaining overlap:** a superseded run can keep going for up to one poll interval (~10 s) or until the in-process step ends. During that window, progress writes can interleave with the new attempt's, and a late S3 upload can overwrite a same-named object. This is accepted: the new attempt rewrites everything it produces, and terminal state is fenced.

### 5.4 Orphan subprocesses

The scripts start their own children (ffmpeg, whisper), and `run_script` already starts each script in its own process group (`setsid`).

- **The problem:** `PR_SET_PDEATHSIG` isn't enough, because grandchildren don't inherit it.
- **The design:** `run_script` records the group id and pod name on the job (`script_pgid`, `script_host`) when it starts a script, and clears them when the script ends.
- **Same pod:** after a successful claim, if `script_host` is this pod and the group still exists, the claim first runs `os.killpg(script_pgid, SIGKILL)`. That removes any leftover script and its ffmpeg or whisper children before the retry touches the temp directory.
- **Different pod:** the old pod is gone (or partitioned, and it stops itself through `heartbeat_lost`), and its processes went with it.
- **Stale temp files:** a retry in the same pod reuses `/tmp/ai4ia_pipeline/{stage}_{job_id}`. Every stage must clear its temp directory at start, not only at the end. Check all stage modules (08 and 09 especially) and add the clear where it's missing.

### 5.5 Hard time limit

- Set `time_limit = JOB_MAX_RUNTIME_SECONDS + 600` on `run_video_job_task`. Celery kills a hung child, and with `task_acks_on_failure_or_timeout` (default true) the message is acked.
- This catches a main thread deadlocked while its heartbeat thread keeps beating, which the heartbeat alone can't see.
- The sweeper's deadline rule (6.2 R3) records the user-facing failure first.

---

## 6. System-level recovery (the sweeper)

### 6.1 Where it runs, and why

It runs as a background task started in the API `lifespan`, in every uvicorn worker of every replica. Each pass takes a dedicated connection and tries the **session-level** `pg_try_advisory_lock(hashtext('job_sweeper:' || worker_queue))`, holding it for the whole pass. Only the winner acts, and it keeps leadership across the pass's separate rule transactions; it unlocks and returns the connection at the end. Celery messages are sent only after the transaction that updated the row has committed, never with a transaction open. That gives one leader per environment, with no extra process.

**Why the API, not Celery beat or the worker:**
- The API is always up, even when every worker is dead. That's exactly when users need jobs failed instead of spinning forever.
- It doesn't depend on a beat process being deployed.
- It already has everything needed to dispatch.

Interval: `JOB_SWEEP_INTERVAL_SECONDS` (60) plus 0–10 s of random jitter.

### 6.2 Rules

Each rule is one conditional `UPDATE … RETURNING`, scoped to `worker_queue = settings.worker_queue`, using the database clock. It acts only when the row still matches what was evaluated, so a worker heartbeating at the same instant wins.

| # | Condition | Action |
|---|---|---|
| R1 | `status='running'` and `heartbeat_at < now() - JOB_STALE_AFTER_SECONDS` (300) | **Worker lost.** If `attempts < JOB_MAX_ATTEMPTS` (2) and `JOB_RETRY_ON_WORKER_LOSS`: set `status='queued'`, a new `dispatch_id`, `dispatched_at=now()`, clear `lease_owner`; add event `interrupted` ("Restarting after a server interruption"); then send the Celery message. Otherwise fail the job as `worker_lost` (user message: "The server processing your video stopped. Please try again."). |
| R2 | `status='queued'` and `dispatched_at < now() - 120 s` and the env's queue is **empty** (message count 0, see 6.3) | **Message lost** (Redis flush or restart). Re-dispatch with a new `dispatch_id`. This doesn't use up an attempt, because the job never ran. |
| R3 | `status='running'` and `started_at < now() - JOB_MAX_RUNTIME_SECONDS` | **Deadline.** Fail the job as `deadline_exceeded`. Clearing the lease makes the worker abort at its next heartbeat. |
| R4 | `status='queued'` and `dispatched_at < now() - JOB_QUEUED_MAX_AGE_SECONDS` (6 h) | **Expired in queue.** Fail the job as `queue_expired`. |
| R5 | `env is null` (created before this deploy, or by old code on the other environment) and `status in ('queued','running')` and the last update (`metadata->>'updated_at'`) is older than `JOB_QUEUED_MAX_AGE_SECONDS` (6 h, well past the ~2.5 h worst case) | **Legacy row, certainly dead:** fail it as `expired`. This is safe from either environment, because no live job goes 6 h without an update, and the update is conditional, so two sweepers acting on the same row is harmless. It stops pre-deploy rows holding quota forever once the startup reaper is gone. Younger legacy rows are left alone. |

**Order within a sweep:** R3, R1, R2, R4. Each job is handled once per sweep.

**Sending a re-dispatch:** the row is updated *before* the message is sent, so the old message is fenced off first. If the send fails (Redis down), the job is left `queued` with a fresh `dispatch_id` and no message; R2 re-sends once Redis is back.

### 6.3 Why "queue empty" for R2

Checking whether a specific message is still in the list means decoding kombu's encoded bodies, which is brittle. Instead the sweeper asks for the queue's message count with kombu's public `channel.queue_declare(queue, passive=True).message_count`. That sums the priority sub-lists and applies any key prefix, which a raw `LLEN` would miss. A count of 0, together with a 120 s grace (time for a worker to pop and claim), proves no message is waiting for *any* job in this environment. So a queued job past the grace has certainly lost its message.

If the queue isn't empty, R2 waits; R4 is the backstop.

This deliberately avoids re-dispatching a job that is still legitimately waiting. A new message goes to the *back* of the queue and would starve it behind the old, now-invalid copy.

### 6.4 Outage grace (false-positive protection)

If the database was unreachable, workers couldn't write heartbeats. When it comes back, every running job looks stale, and R1 would restart live jobs.

The sweeper remembers when it last completed a pass successfully. If that was longer ago than `JOB_STALE_AFTER_SECONDS`, the first pass after recovery only records state and takes no R1 action. Workers get one full stale window to beat again.

The same grace applies after the API itself starts (no pass completed yet in this process). That also covers "API was down for an hour".

### 6.5 Startup reaper (C-23)

For video jobs, `main._reap_orphaned_jobs` is **removed**. The video path has no thread fallback any more, so the "local-thread orphan" case it existed for can't happen, and the sweeper replaces it with rules that don't depend on Redis reachability. The ingest reaper stays until ingest gets the same treatment.

---

## 7. Configuration

### 7.1 Broker visibility timeout (item 2)

Set in `celery_app.conf.broker_transport_options = {"visibility_timeout": CELERY_VISIBILITY_TIMEOUT_SECONDS}`, default **14 400 s (4 h)**. That's above the worst case (~2.5 h), with margin.

**It only takes effect when every consumer on the same Redis database has it** (section 1). There are two options:

| Option | Pros | Cons |
|---|---|---|
| **A. Separate Redis database per environment** (uat `…:6379/1`, release stays `/0`) | Isolates the broker, the `unacked` set and the rate-limit keys (which uat and release also share today). Takes effect on the uat deploy alone. | Config change on the cluster. Messages queued in `/0` at switch time are left behind, but R2 re-dispatches them. |
| **B. Same database, both environments on the new value** | No Redis change | Not effective until release is promoted. Until then, release's 1 h default still restores uat's long jobs, and only the lease protects them. |

**Recommendation: A.** With the lease in place, B is still *safe* (a restored live job fails its claim and is dropped). It is just noisier.

**Trade-off:** a longer visibility timeout means a dead worker's message waits longer before the *broker* restores it. That no longer matters: the sweeper detects the loss in ~5 minutes through heartbeats and re-dispatches. When the old message is eventually restored, it fails the `dispatch_id` check and is dropped.

### 7.2 Settings (all in `core/settings.py`)

| Setting | Default | Notes |
|---|---|---|
| `JOB_HEARTBEAT_INTERVAL_SECONDS` | 30 | |
| `JOB_STALE_AFTER_SECONDS` | 300 | Must be ≥ 5 × the heartbeat interval; validated at startup |
| `JOB_MAX_RUNTIME_SECONDS` | 2100 (35 min) | Decision 3. Must be < the visibility timeout minus the time-limit margin; validated at startup |
| `JOB_QUEUED_MAX_AGE_SECONDS` | 1800 (30 min) | Decision 3 |
| `JOB_SWEEP_INTERVAL_SECONDS` | 60 | |
| `CELERY_VISIBILITY_TIMEOUT_SECONDS` | 3600 | Must be > `JOB_MAX_RUNTIME_SECONDS` + 600; validated at startup |

**Explicit Celery setting:** `worker_cancel_long_running_tasks_on_connection_loss = False`. On a Redis blip, a running render keeps going, and any redelivered copy is fenced by the lease rather than killing real work.

---

## 8. Edge cases

| # | Scenario | What happens |
|---|---|---|
| 1 | Child OOM-killed mid-stage | Celery re-queues at once, but the dead holder's heartbeat is at most 30 s old, so that immediate redelivery **loses the claim and is dropped**. Recovery comes from R1 about 5 min later: a new `dispatch_id`, the claim wins, `attempts` = 2, and the same-pod claim kills the leftover script group (5.4). A second OOM fails the job as `worker_lost_repeatedly`. |
| 2 | Whole pod killed | Heartbeats stop. R1 re-dispatches after ~5 min. The old message is restored hours later, fails the `dispatch_id` check, and is dropped. |
| 3 | Job legitimately runs 2 h | Heartbeats stay fresh. No restore, because the visibility timeout is 4 h. Even if a copy is restored (option B), its claim fails. |
| 4 | Main thread hung, heartbeat thread alive | R3 fails the job at 3 h. The Celery hard time limit kills the child at 3 h 10 min. |
| 5 | Database outage of 10 min | Heartbeats fail and the sweeper pauses. After recovery, the outage grace skips R1 for one window while workers beat again. No live job is restarted. |
| 6 | Redis restart, queue lost | Queued jobs fall under R2 (queue empty, more than 120 s old) and are re-dispatched. Running jobs continue; their acks fail silently, which is harmless. |
| 7 | Redis down during a re-dispatch | The row already carries the new `dispatch_id`, but the send fails. R2 re-sends when Redis is back. |
| 8 | Cancel while a job waits for re-dispatch | The status becomes `cancelled`. All rules skip terminal jobs, and any message fails the claim. |
| 9 | Cancel while running | Unchanged: the `run_script` poll sees `cancelled`. The lease check is additional, not a replacement. |
| 10 | Two API replicas sweep at once | The advisory lock allows one leader. Every action is a conditional update anyway, so double processing is impossible. |
| 11 | uat sweeper sees release's jobs | `worker_queue` differs (or is null for old code), so they're ignored. |
| 12 | Release (old code) runs alongside uat (new code) | Old code ignores the new columns. The old startup reaper in release still touches all jobs when Redis is down: the same risk as today, gone once release is promoted. |
| 13 | Messages in flight when the new code deploys | Legacy messages (no `dispatch_id`) are claimed while the job has no `dispatch_id`. Legacy rows have no `worker_queue`, so the sweeper leaves them alone. |
| 14 | Worker loses the lease but keeps running briefly | It aborts at the next poll or heartbeat (≤ 30 s), and terminal writes are fenced. Output writes may overlap for that window (5.3). |
| 15 | Clock skew between pods | All timing uses the database clock. |
| 16 | Heartbeat write slow (database under load) | Stale is 10 missed beats (300 s), so there's a wide margin. |
| 17 | Every worker down for hours | R1 fails running jobs after the retry budget, and R4 fails queued ones at 6 h. Users get a clear failure instead of a spinner. |
| 18 | Quota interaction (PR 4) | A requeued job counts as queued and a failed one frees its slot. The stale job's "running" slot is freed within ~5 min, instead of 1 h today. |
| 19 | Storyline cache on retry | If 07/07b finished and passed quality checks before the loss, the retry skips them (this already happens). Otherwise everything reruns. |
| 20 | User watching the job | They see event `interrupted`, the status goes back to `queued`, and progress restarts at 0. Frontend note: progress can go backwards after an `interrupted` event, so show the event's message. |
| 21 | Retried job produces a different storyline | Possible, since the LLM output isn't deterministic. That's acceptable; the job id is unchanged. |
| 22 | Processing and provenance records | The claim leaves the processing record open. `_fail_job` (fenced) closes it on final failure. The decision record is written only on completion. |
| 23 | The sweeper's own failures | Each rule is a separate transaction. One failed rule doesn't block the others, and it is retried next pass. |
| 24 | Worker warm shutdown (a rollout within the grace period) | Celery puts running messages straight back in the queue (`restore_at_shutdown`). If the child is still finishing, the copy loses the claim. If the child was killed, the copy loses the claim until the heartbeat goes stale, and R1 recovers the job. Either way there's no double run. |
| 25 | Worker cut off from the database, API not | The worker's heartbeats fail. At 0.8 × stale it stops itself (`heartbeat_lost`) and kills its script group. At 1.0 × stale R1 re-dispatches. There's no overlap. |
| 26 | Old attempt still writing snapshots after R1 requeued the job | `upsert_job` leaves `status` alone on new-era rows, so the job stays `queued` and R1 doesn't fire twice (4, "Who may change `status`"). |

---

## 9. Observability

**Metrics:**
- `job_heartbeat_age_seconds` (gauge, max over running jobs, per env)
- `jobs_recovered_total{rule="R1|R2"}`
- `jobs_failed_by_sweeper_total{reason}`
- `job_claims_total{outcome="run|superseded|leased|terminal|exhausted"}`
- `job_sweeper_runs_total{outcome}`

**Events:** `interrupted`, `requeued`, `superseded`, each carrying the rule and attempt number.

**Logs:** the claim outcome with the lease owner, and one line per sweeper decision.

**Alerts (for ops):**
- heartbeat age > 10 min;
- any `worker_lost_repeatedly`;
- sweeper leader absent for > 5 min.

---

## 10. How this prepares for resume (item 3)

- **The lease** guarantees one runner. That's the precondition for resuming from saved stage outputs without two runs fighting over them.
- **`attempts` and the `interrupted` event** tell the resume logic it's a retry.

Resume would add a step after a successful claim: rebuild `PipelineContext` from `job.paths` and `stage_status`, then skip stages whose outputs exist. None of that is in this plan.

---

## 11. Build plan

Order within the PR; each step lands with its tests:

1. **Schema:** new columns and indexes; job-store and repository methods (`claim`, `heartbeat`, `fenced_complete`, `fenced_fail`, sweeper queries).
2. **Dispatch:** `create_video_job` and dispatch set `env`, `dispatch_id`, `dispatched_at` and `worker_queue`, and pass `dispatch_id` to the task.
3. **Worker:** claim replaces `_claim_delivery`; heartbeat thread with cause-carrying `stop_reason` and self-fencing; heartbeat stops before completion; `stop_reason` checked in `run_script` and at stage boundaries; `upsert_job` status ownership; fenced terminal writes; script process-group recording and same-pod kill on claim; temp-directory clear at stage start; hard time limit.
4. **Sweeper:** background task in the API lifespan, rules R1–R4, outage grace; remove the video startup reaper.
5. **Configuration:** visibility timeout, settings with startup validation, `worker_cancel_long_running_tasks_on_connection_loss=False`.
6. **Docs:** update the README, CLAUDE.md (cancellation and redelivery sections) and `security-remediation.md` (C-23).

### Tests

- **Unit:**
  - claim outcomes;
  - heartbeat fencing sets `stop_reason`;
  - `run_script` aborts on `stop_reason`;
  - each sweeper rule, including "no action when the row changed";
  - the outage grace;
  - settings validation.
- **Postgres integration** (scratch schema, like `test_job_quota_postgres.py):
  - concurrent claims for one job: exactly one wins;
  - stale takeover;
  - superseded `dispatch_id`;
  - fenced completion after takeover;
  - sweeper leader election across two connections.
- **Chaos run** (local infra only, never uat/release):
  - `kill -9` the worker child mid-09 → recovery by R1 with attempts=2, and no leftover ffmpeg (`pgrep ffmpeg` is empty before the retry renders);
  - block the worker's database port only (iptables on the worker container) → the worker stops itself before R1 fires; no overlapping renders;
  - `kill -9` the whole worker → R1 within 6 min;
  - `FLUSHDB` Redis with jobs queued → R2 re-dispatch;
  - stop Postgres for 10 min mid-render → no restart after recovery;
  - start a second worker with a 60 s visibility timeout during a long job → the restored copy is dropped.

  Each run's timings get recorded in the doc.

---

## 12. Rollout

1. **Deploy to uat** with option A (uat on Redis database `/1`). Messages queued in `/0` at switch time are re-dispatched by R2 within ~2 minutes.
2. **Watch for a week:**
   - `job_claims_total` should show no unexpected `superseded` or `leased` outcomes;
   - heartbeat age should stay under 60 s;
   - the chaos checklist should be repeated on uat, with an agreed maintenance window.
3. **Promote to release.** Release keeps Redis `/0`, now with the long visibility timeout. Both reapers are gone.

**Rollback:** revert the code. The new columns are inert for old code, and the old startup reaper returns.

---

## 13. Decisions needed

| # | Question | Recommendation |
|---|---|---|
| ~~1~~ | *Decided: no retry (see Decisions).* After a worker loss, retry once automatically or fail at once? A retry reruns the LLM stages (cost, and a possibly different storyline). | Retry once (`JOB_RETRY_ON_WORKER_LOSS=true`) |
| 2 | Move uat to its own Redis database (option A)? | Yes |
| 3 | Is a 3 h deadline acceptable for the longest video (600 s target on a CPU host)? | Yes; it's configurable |
| ~~5~~ | *Decided: queue name is the marker; no new variable.* | |
| 4 | Longer `terminationGracePeriodSeconds` for the worker in the deploy repo (item 4), so rollouts don't kill renders at all? | Yes, separately, e.g. 20 min plus Celery warm shutdown |

---

## 14. Review record

Adversarial review of this plan against the code and Celery 5.6.3 / kombu 5.6.2, 2026-09-25. Every finding is addressed above:

| # | Finding | Severity | Addressed in |
|---|---|---|---|
| 1 | The queue name isn't a reliable environment marker (both default to `video_jobs`) | Blocker | §4 `env` column, §7.2 `AI4IA_ENV` |
| 2 | A heartbeat after completion would set `stop_reason` and skip the housekeeping | Blocker | §5.2 "Stopping cleanly at the end" |
| 3 | PDEATHSIG isn't inherited by grandchildren (ffmpeg, whisper) | Blocker | §5.4 process group recorded and killed on a same-pod claim |
| 4 | Snapshot writes could reset `status` after R1 requeued the job, so R1 fires twice | Gap | §4 "Who may change `status`" |
| 5 | A cancel noticed by the heartbeat would skip cancel bookkeeping | Gap | §5.2 / §5.3 cause-carrying `stop_reason` |
| 6 | A worker cut off from the database would run alongside the retry | Gap | §5.2 self-fencing at 0.8 × stale, edge case 25 |
| 7 | A per-transaction advisory lock lets leadership change partway through a sweep | Gap | §6.1 session-level lock held for the pass |
| 8 | Pre-deploy rows would never be cleaned up once the startup reaper is removed | Gap | §6.2 R5 |
| 9 | `LLEN` misses priority sub-queues and key prefixes | Nit | §6.3 `queue_declare(passive=True)` |
| 10 | Edge case 1 was wrong; warm-shutdown restore was missing; PDEATHSIG in `preexec_fn` was unsafe with threads | Nit | §8 rows 1 and 24; PDEATHSIG dropped |

Confirmed sound by the review:
- the claim predicate's null handling;
- a stale takeover racing R1 is safe;
- the shared-`unacked` and shortest-timeout analysis;
- the Celery `time_limit` and ack-on-timeout behaviour;
- `worker_cancel_long_running_tasks_on_connection_loss` defaults to false;
- R2's grace holds with prefetch multiplier 1.

---

## 15. Built (2026-09-26)

**Where the build differs from the plan text above,** following the decisions and three reviews:

- **No retry** (superseded by §16). The claim only runs a never-claimed, queued job with a matching `dispatch_id` (`attempts = 0`). A later delivery for a job whose holder is dead fails it as `worker_lost`. R1 always fails.
- **The worker enforces the 35-minute cap itself.** The stop signal `deadline` is raised in `run_script` and between stages, and the job is failed through `_fail_job` with a user message. The sweeper's deadline rule is a backstop at cap + stale window, for a silent worker.
- **Celery limits:**
  - `soft_time_limit` at cap + 4 min: the runner maps it to `deadline_exceeded`, and `run_script` kills its script on any exit.
  - Hard `time_limit` at cap + 5 min, below the 1 h visibility timeout, which stays at the default and is set explicitly.
- **Snapshot saves** on a leased row never change `status` (except to `cancelled`). Once the row has finished, they don't change `error` or `completed_at` either.
- **Leftover scripts:** the script's pid and host are kept in the job's metadata (`active_pid`, `active_host`). A delivery on the same host, or the task's own `finally`, kills the script's process group once its leader is confirmed to name the job. PR_SET_PDEATHSIG and clearing temp directories at stage start were dropped: with no retry, a job id never runs twice. Temp-directory clearing is restored in §16.
- **Legacy rows:** `worker_queue is null` rows are failed as `expired` after 6 h. Queued-too-long is measured from `created_at`.

**Code:**
- `services/job_lease.py` (claim, heartbeat, stop signal, leftover kill);
- `services/job_sweeper.py`;
- `db/repositories/job_repository.py` (claim, lease_state, heartbeat, finish, fail_if_lease, fail_where, lost_queue_candidates, redispatch, and the `upsert_job` rules);
- `db/postgres.py` (`pg_session`);
- `db/schema.py` (jobs columns and indexes);
- `worker/tasks.py`, `worker/celery_app.py`, `pipeline/base.py`, `services/video_job_runner.py`, `main.py` (sweeper task; video startup reaper removed, which closes C-23).

**Tests:**
- `tests/unit/test_job_lease.py`, `test_job_sweeper.py`, `test_job_recovery_runner.py`, `test_worker_tasks.py`.
- `tests/integration/test_job_lease_postgres.py`, against real Postgres:
  - 8 parallel deliveries, one claim;
  - fencing;
  - the status and error rules;
  - sweeper scoping and leadership.

**Chaos runs.** A real Celery worker (prefork, acks_late) ran against local Postgres and an isolated Redis database, with the pipeline replaced by one long `run_script` call. Heartbeat interval was 2 s and the stale window 12 s.

| # | Scenario | Result |
|---|---|---|
| 0 | Normal run | Completed, attempts 1, lease cleared |
| 1 | `kill -9` the worker child mid-script (OOM) | Celery redelivered at once; the claim saw the holder's pid was gone, killed the leftover script, and failed the job as `worker_lost` in under 1 s; no rerun |
| 2 | `kill -9` the whole worker (pod loss) | The sweeper failed the job as `worker_lost` 12 s later (= stale window) |
| 3 | `FLUSHDB` with a job queued, no worker | Untouched inside the 120 s grace; then re-sent with a new `dispatch_id`; ran once when a worker started |
| 4 | Running job's message restored early (the shared-Redis case) | The restored copy was delivered and dropped as `leased`; the job completed once |
| 5 | Cancel mid-script, the way the API does | The heartbeat saw the cancel; the script stopped in 3 s; the job ended `cancelled` |

---

## 16. Retry on worker loss (issue #19, 2026-09-26)

Reverses decision 1. A job is rerun from the start only when its worker is lost.

**When a job is retried.** Only on the dead-holder path:
- **Claim takeover.** A redelivered message with the job's current `dispatch_id` finds the holder dead (its pid is gone on this host, or its heartbeat is stale). It kills the leftover script group, then takes the lease in one statement fenced on the dead owner, the `dispatch_id` and the budget.
- **Sweeper R1.** A running job with a stale heartbeat is requeued: `status = queued`, a new `dispatch_id`, lease cleared, fenced on the owner and a still-stale heartbeat. The message is sent after the row commits; if the send fails, R2 re-sends it.

Stage errors, cancellation, the worker's own runtime cap, the Celery time limits and the sweeper's deadline rule all fail the job as before. The deadline rule runs before R1 in each pass.

**Budget and guard.** `JOB_MAX_WORKER_LOSS_RETRIES` (default 3). A loss is retried while `attempts <= retries`. The job fails as `worker_lost_repeatedly` when the budget is spent, or when its worker is lost a second time in the same stage (read from the job's `interrupted` events), since that stage likely kills the worker. With 0, the job fails as `worker_lost`, as before.

**What a retry resets.**
- `started_at` is set by every claim, so the runtime cap is per run, like the worker's own cap and the Celery limits. Snapshot saves no longer change `started_at` on leased rows.
- Progress, `current_stage` and `stage_status` go back to the start, so progress does not jump.
- Queued-too-long counts from the requeue (`metadata.requeued_at`) for a retried job, and from `created_at` for a first run. A lost-message re-send does not reset it.
- Each stage clears its job-scoped directories when it starts: `/tmp/ai4ia_pipeline/{07,07b,08,10}_{job_id}`, `data/jobs/{job_id}/generated_assets` (local mode) and `exports/hls/{job_id}`. The `07_` directory is now also removed when a job ends.
- A retried claim on the same host kills the lost run's leftover script group.
- Stage 09 (S3 mode) deletes the job's own asset folder in the local cache before downloading, so a storyline-cache hit never renders with the lost run's audio. Assets shared across jobs are kept.

**What the user sees.** An `interrupted` event with `attempt`, `max_attempts` and the message "Restarting after a server interruption". The status goes `running` → `queued` → `running` (sweeper) or stays `running` (takeover). Processing records stay open until the job ends.

**Keeping failures out of the retry path.**
- If a run ends while it still holds the lease (for example, its failure write raised), the task fails the job as `pipeline_error`, so the sweeper never sees it as a lost worker.
- The worker stops itself once its last good heartbeat is older than 0.8 × the stale window, even when the heartbeat write hangs instead of failing. So the old run stops before R1 requeues the job.
- A superseded run leaves the temp directories and the recorded script alone, since they may belong to the new run.
- A queued job with more attempts than the setting allows (the setting was lowered, e.g. set to 0 for rollback) is failed on delivery, not left to expire.

**Accepted.**
- A worker cut off from the database for more than about 4 minutes stops itself and is retried. It counts as a lost worker.
- Snapshot saves are not fenced on the lease owner. A run that is still alive after losing its lease can overwrite progress fields until it stops (edge case 14). The self-stop above makes this window close before R1 fires.
- Processing-record flags from a lost run stay on the record.

**Chaos runs.** Same setup as §15: a real Celery worker, local Postgres, an isolated Redis database, heartbeat 2 s and stale window 12 s, with the pipeline replaced by one long `run_script` call.

| # | Scenario | Result |
|---|---|---|
| 0 | Normal run | Completed, attempts 1 |
| 1 | `kill -9` the worker child mid-script (OOM) | The redelivery took the job over within 3 s and killed the leftover script. It completed on run 2 with one `interrupted` event. |
| 2 | `kill -9` the whole worker (pod loss) | The sweeper requeued the job after 12 s (the stale window). The next worker's claim killed the leftover script, and the job completed on run 2. |
| 3 | 4 losses, each in a new stage | Failed as `worker_lost_repeatedly` after 4 runs, with 3 `interrupted` events |
| 4 | 2 losses in the same stage | Failed as `worker_lost_repeatedly` after 2 runs |
| 5 | Stage error | Failed as `pipeline_error` at once. Still attempts 1 after the stale window. |

The interruption's stage is kept in the event payload, because `job_events` has no stage column.

**Metrics.** `jobs_retried_total{path="claim|sweeper"}`; `job_claims_total` gains `retry` and `worker_lost_repeatedly`; `jobs_failed_by_sweeper_total` gains `worker_lost_repeatedly`.

**Cost.** A retry reruns the LLM stages unless the storyline cache hits, and may produce a different storyline.

**Rollback.** Set `JOB_MAX_WORKER_LOSS_RETRIES=0` and restart, or redeploy the previous image. Rows are unchanged, so old code reads them.
