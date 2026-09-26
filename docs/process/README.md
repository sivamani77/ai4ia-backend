# Delivery Framework

How we take a change from idea to production. It applies to this repo and to every
application we build. To reuse it, copy `docs/process/`, `docs/github-templates/` and
`CHANGELOG.md` into the new repo. GitHub only reads templates from `.github/`, so copy
the contents of `docs/github-templates/` into `.github/` to turn them on.

## 1. One home for each kind of information

Each kind of information lives in one place. Other places link to it and never copy it.

| What | Where | Owner |
|---|---|---|
| Work items: bugs, features, change requests | **GitHub Issues** | Whoever raises it |
| Code review and approval | **GitHub Pull Requests** | Author + reviewer |
| Engineering docs: design, runbooks, decisions | **Repo `docs/`** (versioned with the code) | Author of the change |
| What shipped, and when | **`CHANGELOG.md` + GitHub Releases** | Release owner |
| Product/stakeholder view: overview, release summaries, how-to | **Confluence** (links to GitHub) | Release owner |
| Test execution and sign-off for stakeholders | **Spreadsheet**, only when asked for (one per release) | QA / release owner |

Rule: if it isn't in an Issue, it isn't planned work. If it isn't in a PR, it doesn't ship.

## 2. Branches and environments

```
feature/<issue#>-<slug>  ──PR──▶  dev      ──PR──▶  uat  ──PR──▶  release
                                  (integration)    (testing)      (production)
hotfix/<issue#>-<slug>   ──PR──▶  release, then merge release back into dev
```

- `dev` is the integration branch. All feature work branches from it and merges back into it.
- `uat` is where changes are tested before release. `release` is production.
- Protect `dev`, `uat` and `release` in GitHub settings: no direct pushes, a PR
  and at least 1 approval required.

## 3. The lifecycle and its approval gates

| Step | What happens | Gate (who approves) |
|---|---|---|
| **1. Raise** | Open an Issue from a template (Bug / Feature / Change request). | — |
| **2. Approve scope** | Change requests and anything that changes behaviour users see get the `needs-approval` label. | **G1**: product owner/lead comments "Approved" and swaps the label for `approved`. No work starts before that. |
| **3. Build** | Branch `feature/<issue#>-<slug>` from `dev`. Commit, and update the `docs/` that the change affects. | — |
| **4. PR to dev** | Fill in the PR template. Link the issue (`Closes #123`). Run the full test suite and paste the result. | **G2**: 1 reviewer approves. |
| **5. Promote to uat** | PR `dev → uat`, titled `Promote to uat: <date>`, listing the included issues. | **G3**: lead approves. Schema or data changes need an explicit "data change OK" in the PR. |
| **6. Verify on uat** | Run the release checklist (below). Record results in the PR, or in the test spreadsheet if stakeholders need it. | QA / author confirms in the PR. |
| **7. Release** | PR `uat → release`, titled `Release vX.Y.Z`, with the CHANGELOG section as its body. | **G4**: release owner + product owner approve. |
| **8. Publish** | Tag, create the GitHub Release, publish the Confluence page, close the issues. | — |

## 4. Versioning

Use Semantic Versioning `vMAJOR.MINOR.PATCH`, tagged on `release`:

- **PATCH**: bug fixes only.
- **MINOR**: new features, backwards compatible.
- **MAJOR**: breaking API or contract changes (the frontend must change too).

The first tag is `v1.0.0`: whatever is in production when the framework is adopted.

## 5. Release checklist

Copy this into the `Release vX.Y.Z` PR.

```
Before the uat → release PR
- [ ] Every included issue is linked and approved (G1)
- [ ] Full test suite green
- [ ] CHANGELOG.md has a section for vX.Y.Z
- [ ] New env vars are listed and set in the deploy config
- [ ] Schema/data changes: migration reviewed, rollback plan written
- [ ] uat healthy: no errors or config warnings in the logs
- [ ] Smoke test on uat: run the main user flow end-to-end
- [ ] Frontend impact noted (and the frontend team told)

After merge to release
- [ ] Production deploy healthy (logs, one smoke job)
- [ ] git tag vX.Y.Z on the release merge commit, then push the tag
- [ ] GitHub Release created from the tag (notes = CHANGELOG section)
- [ ] Confluence release page published (summary + link to the GitHub Release)
- [ ] Linked issues closed, with the milestone set to vX.Y.Z
```

Commands:

```bash
git checkout release && git pull
git tag -a v1.1.0 -m "v1.1.0" && git push origin v1.1.0
gh release create v1.1.0 --title "v1.1.0" --notes-file <(sed -n '/^## \[1.1.0\]/,/^## \[/p' CHANGELOG.md | sed '$d')
```

## 6. Hotfix (production is broken)

1. Open a Bug issue labelled `hotfix`. The lead approves in the issue (G1 and G4 combined).
2. Branch `hotfix/<issue#>-<slug>` from `release`, then PR into `release` with 1 approval.
3. Tag a PATCH version and publish it as a normal release (step 8).
4. Merge `release` back into `dev` (and into `uat`) so the fix isn't lost.

## 7. Labels

Create these labels once per repo:

`bug` · `feature` · `change-request` · `hotfix` · `needs-approval` · `approved` ·
`data-change` (schema, data repair or migration) · `docs`

Group issues into a milestone named after the target version (`v1.1.0`).

## 8. Confluence page structure

Use one space per application:

```
<App name>
├── Overview            (what it is, architecture diagram, links to the repo README)
├── Environments        (URLs, who owns what; no secrets)
├── Releases
│   └── vX.Y.Z          (date, summary for non-engineers, link to GitHub Release, known issues)
└── How-to / Runbooks   (links to repo docs/ files, not copies)
```

## 9. Next step: automate the gates

These are recommended and not done yet:

- Branch protection on `dev`/`uat`/`release` (a GitHub settings change, which needs admin).
- A GitHub Actions job that runs the tests on every PR, so G2 doesn't depend on a pasted result.
