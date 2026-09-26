# Delivery stage guides

One short guide per stage of the [delivery process](../../process/README.md), with the
repo-specific commands and checks for this backend. They exist so that each stage can be
run without re-reading the whole codebase.

**Do not trust them blindly.** Each guide records when it was last checked. Before you act
on a fact here, run the "Re-verify" command next to it. If the code disagrees, the code
wins: fix the guide in the same PR.

| Stage | Process steps | Guide |
|---|---|---|
| Raise and approve | 1–2 | [1-raise-and-approve.md](1-raise-and-approve.md) |
| Build and PR to dev | 3–4 | [2-build-and-pr-to-dev.md](2-build-and-pr-to-dev.md) |
| Promote to uat | 5 | [3-promote-to-uat.md](3-promote-to-uat.md) |
| Verify on uat | 6 | [4-verify-on-uat.md](4-verify-on-uat.md) |
| Release PR | 7 | [5-release.md](5-release.md) |
| Publish | 8 | [6-publish.md](6-publish.md) |
| Hotfix | 6 (hotfix) | [hotfix.md](hotfix.md) |

## Repo facts every stage relies on

Last verified: 2026-09-26.

| Fact | Re-verify |
|---|---|
| Pushes to `dev`, `uat` and `release` deploy. No tests or lint run in CI. | `cat .github/workflows/main.yaml` |
| uat and release share Postgres, ChromaDB, Redis and S3. A uat deploy already changes what production reads. | Compare the uat and release deploy configs (external deploy repo) |
| Schema is applied on API startup from `backend/app/db/schema.py`, not from `db/migrations/`. | `grep -n ensure_pipeline_schema backend/app/main.py` |
| Config is `Settings` in `backend/app/core/settings.py`. Missing production vars only log warnings at startup. | `grep -n "def validate_deployment_env" -A40 backend/app/core/settings.py` |
| The worker command and deploy env live in the external deploy repo, not here. | — |
| `.env` is tracked and holds non-secret config only. Secrets come from the deploy Secret. | `git show HEAD:.env \| cut -d= -f1` |
| Test command: `.venv/bin/python -m pytest` (about 5 minutes). | `grep -n -A5 "tool.pytest" pyproject.toml` |
