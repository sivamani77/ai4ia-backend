# 4. Verify on uat (process step 6)

Last verified: 2026-09-26.

Record the results in the promotion PR, or in the release PR's checklist.

## Steps

1. **Deploy finished.** The uat deploy workflow for the merge commit is green in GitHub Actions.
2. **Startup log is clean.** The first lines of the API log show no
   `validate_deployment_env` warnings (Redis, `API_KEY`, `JWT_SIGNING_KEY`, `MAIL_SMTP_*`,
   the job timing checks). The app boots even when they are missing, so read the log.
3. **Worker is on the right queue.** Redis is shared, so each environment needs its own
   `WORKER_QUEUE`, set on both the API and the worker (`--queues`). The default and the tracked
   `.env` both say `video_jobs`, so uat must override it in the deploy config (for example
   `video_jobs_uat`). If the queues match, one environment runs the other's jobs.
4. **Smoke test the main flow**: sign in → create a video job from a prompt → it completes →
   the video plays → delete a test job. Note the job id in the PR.
5. **Frontend** works against uat for anything the release changed in the API.

## Re-verify

```bash
grep -n "worker_queue" backend/app/core/settings.py
grep -rn "WORKER_QUEUE" .env
```

Pod logs and S3 are the only uat views available from a dev machine. The database is not
directly reachable.
