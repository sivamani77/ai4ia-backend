# 3. Promote to uat (process step 5)

Last verified: 2026-09-26.

Merging to `uat` deploys the test service (`ai4ia-backend-test`). Because uat and release
share the data stores, **this is the step where schema and data changes reach production**.
Review them here, not at release time.

## Steps

1. Check that nothing is on `uat` or `release` that `dev` lacks:
   ```bash
   git fetch origin
   git log --oneline origin/dev..origin/uat
   git log --oneline origin/dev..origin/release   # e.g. hotfixes or direct edits
   ```
   If either is non-empty, merge it into `dev` first through a PR.
2. Open the PR `dev → uat`, titled `Promote to uat: <YYYY-MM-DD>`. List the included issues:
   ```bash
   git log --no-merges --format='- %s' origin/uat..origin/dev
   ```
3. G3: lead approves. If anything is `data-change`, the PR needs an explicit
   "data change OK".

## What to look for in the diff

```bash
git diff origin/uat origin/dev -- backend/app/db/schema.py      # schema
git diff origin/uat origin/dev -- backend/app/core/settings.py  # config
git diff origin/uat origin/dev --stat -- backend/app/api        # API surface
git diff origin/uat origin/dev -- requirements.txt docker/      # image
```

- A schema change must only add things (`add column if not exists`, new tables). Anything that
  drops, renames or rewrites data needs a written rollback plan in the PR.
- Startup code that writes data (seeding, repairs) runs on the shared database.
