# 6. Publish (process step 8)

Last verified: 2026-09-26.

Run these after the `Release vX.Y.Z` PR is merged.

## Steps

1. **Production healthy.** The release deploy workflow is green. The API startup log has no
   config warnings. One smoke video job completes.
2. **Tag the merge commit** (`gh` must be authenticated for step 3):
   ```bash
   git fetch origin && git switch release && git pull --ff-only
   git log -1 --oneline            # must be the "Merge pull request ... uat" commit
   git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z
   ```
3. **GitHub Release** from the tag:
   ```bash
   gh release create vX.Y.Z --title "vX.Y.Z" \
     --notes-file <(sed -n '/^## \[X.Y.Z\]/,/^## \[/p' CHANGELOG.md | sed '$d')
   ```
4. **Confluence**: add `Releases/vX.Y.Z` with the date, a summary for non-engineers, a link
   to the GitHub Release, and known issues.
5. **Close** the linked issues and the `vX.Y.Z` milestone.
6. **Back-merge** if `release` gained anything during the PR (conflict fixes): PR
   `release → dev`.
