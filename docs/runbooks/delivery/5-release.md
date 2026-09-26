# 5. Release PR (process step 7)

Last verified: 2026-09-26.

## Steps

1. **Pick the version** (SemVer). Breaking API change = MAJOR, new feature = MINOR, fixes
   only = PATCH.
2. **Cut the CHANGELOG** on a feature branch from `dev`: rename `## [Unreleased]` to
   `## [X.Y.Z] - <YYYY-MM-DD>`, then add an empty `## [Unreleased]` above it. Take it through
   dev and uat like any change, so the release PR carries it.
3. **Check that `release` has nothing `uat` lacks.** A commit that exists only on `release`
   still ships, but uat never tested it:
   ```bash
   git fetch origin && git log --oneline origin/uat..origin/release
   ```
4. **Check that the merge is clean:**
   ```bash
   git switch --detach origin/uat && git merge --no-commit --no-ff origin/release
   git diff --cached --stat; git merge --abort; git switch -
   ```
5. **Open the PR `uat → release`**, titled `Release vX.Y.Z`. The body is the CHANGELOG
   section plus the release checklist from the process doc, filled in. Extract the section with:
   ```bash
   sed -n '/^## \[X.Y.Z\]/,/^## \[/p' CHANGELOG.md | sed '$d'
   ```
6. G4: release owner and product owner approve.

## Filling in the checklist

| Item | How |
|---|---|
| Issues linked and approved | Milestone `vX.Y.Z` lists them all, each with `approved` |
| Full suite green | `.venv/bin/python -m pytest` on the uat head; paste the summary |
| New env vars | `git diff origin/release origin/uat -- backend/app/core/settings.py`. Fields with safe defaults need nothing. Anything required must be in the deploy config |
| Schema/data changes | `git diff origin/release origin/uat -- backend/app/db/schema.py`. These are already live, because uat shares the database. The rollback plan is "redeploy the previous image"; confirm the old code tolerates the new schema |
| uat healthy | See [4-verify-on-uat.md](4-verify-on-uat.md) |
| Frontend impact | `git diff origin/release origin/uat --stat -- backend/app/api` plus `docs/security/frontend-changes-for-security-fixes.md` |
