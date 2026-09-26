# Hotfix (production is broken)

Last verified: 2026-09-26.

## Steps

1. Open a bug issue labelled `hotfix`. The lead approves in the issue (G1 and G4 together).
2. Branch from production and fix:
   ```bash
   git fetch origin && git switch -c hotfix/<issue#>-<slug> origin/release
   .venv/bin/python -m pytest
   ```
3. PR into `release` with 1 approval. Add a `## [X.Y.Z+1]` CHANGELOG section in the same PR.
4. After the merge, publish as in [6-publish.md](6-publish.md).
5. PR `release → dev`, then promote as usual, so the fix is not lost and uat matches.

## Watch out

- uat and release share the data stores, so testing a data fix on uat first means applying
  it to production. Use a dry-run switch (see the speaker repair in
  `backend/app/services/speaker_repair.py`) for anything that writes data.
- Without step 5, the next `uat → release` PR will look like it removes the fix.
