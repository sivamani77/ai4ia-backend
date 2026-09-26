# 1. Raise and approve (process steps 1–2)

Last verified: 2026-09-26.

## Steps

1. Open an issue using a template from `docs/github-templates/ISSUE_TEMPLATE/` (bug, feature,
   change request). Paste its text into the new issue; `.github/` holds only the workflow.
2. Add labels: the type (`bug` / `feature` / `change-request`) and `data-change` if it touches
   schema, a data repair or seeding.
3. Anything users will notice, or any change request, gets `needs-approval`. Work waits for
   G1: the lead comments "Approved" and swaps the label for `approved`.
4. Set the milestone to the target version (for example `v1.1.0`).

## Checks specific to this repo

- **Does it touch shared data?** Anything that writes to Postgres, ChromaDB or S3 on startup
  also changes production once it reaches uat. Mark it `data-change` now, not at PR time.
- **Does it change the API?** Routes are in `backend/app/api/`. A changed request or response
  shape needs the frontend team named in the issue.

## Re-verify

```bash
ls docs/github-templates/ISSUE_TEMPLATE/
```
