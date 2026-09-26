# 2. Build and PR to dev (process steps 3–4)

Last verified: 2026-09-26.

## Steps

```bash
git fetch origin
git switch -c feature/<issue#>-<slug> origin/dev
# ...work...
.venv/bin/python -m pytest                  # full suite, paste the summary into the PR
git push -u origin feature/<issue#>-<slug>
```

1. Add a line under `## [Unreleased]` in `CHANGELOG.md`, ending with `(#<issue>)`.
2. Update any `docs/` file the change affects. If you move a doc, grep for its old path in
   code comments and tests.
3. Open the PR to `dev` with the template. `Closes #<issue>`. G2: 1 approval.

## Checks specific to this repo

- **Branch.** Run `git branch --show-current` before every commit.
- **New config** goes in `Settings` (`backend/app/core/settings.py`) as a typed field with a
  default. List it in the PR under "New or changed env vars".
- **New table or column** goes in `backend/app/db/schema.py` first, written so it can run
  more than once (`if not exists`). Tick "Schema or data change" in the PR.
- **Pipeline scripts** in `scripts/` must not import `app.*`. Stage modules call them through
  `run_script` in `backend/app/pipeline/base.py`.
- **Long-running work** must go through `run_script`, or it cannot be cancelled or stopped.
- **Tests.** Known failures and flakes are listed in the project `CLAUDE.md`, "Test suite
  state". Treat any other red test as yours.

## Re-verify

```bash
.venv/bin/python -m pytest --co -q | tail -1   # current test count
```
