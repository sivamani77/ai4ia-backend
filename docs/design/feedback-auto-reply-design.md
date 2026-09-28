# Feedback auto-reply

Issue #30. Gives a user an almost-instant reply the moment they submit feedback, instead of
waiting on an admin. It's a first-touch layer on the existing flow (`app/api/feedback.py`,
`app/services/feedback_service.py`), not a redesign of it — an admin still sees everything and
still resolves it themselves.

## What changed, and what didn't

- **Fires instantly on submission.** Not gated on admin approval, not labeled as automated.
- **Single-shot.** Same one-reply shape as before (`resolution_message`), just automatic — no new
  threading or conversation model.
- **Never resolves the feedback.** Status stays `open`. An admin still explicitly resolves it,
  same as before this feature existed. A bad or unhelpful auto-reply is always still visible in
  the open queue, never silently closed out.
- **Separate columns.** `auto_reply`, `auto_reply_category`, `auto_reply_sent_at` are new columns
  on `feedbacks`, distinct from `resolution_message` — a later human resolve never overwrites
  what the user already saw.
- **Fail-open.** Any LLM timeout, exception or invalid output falls back to a canned
  acknowledgment (`feedback_auto_reply.CANNED_FALLBACK`). Feedback submission itself never fails
  or blocks because of the LLM call — the same contract `notification_service.notify_admins`
  already has.

## The LLM's job, and what it's not trusted with

Modeled directly on `app/services/intent_router.py`'s shape: one system-prompt template, a
single short-timeout attempt (`generate_json(..., attempts=1, timeout=20, record_failure=False)`),
structural validation of the output, never raises.

The LLM returns exactly two things:
- `category`: one of `bug` / `question` / `feature` / `other`.
- `reply`: the actual reply text (validated: non-empty string, capped at 1500 characters).

It is explicitly **not** trusted to:
- Decide anything about resolution timing, refunds, or a specific promised outcome. The system
  prompt tells it not to, and — because a free-text reply can't be structurally validated the way
  `intent_router` validates a name against a registry — the actual guarantee comes from code, not
  the model: `_CATEGORY_SUFFIX` in `feedback_auto_reply.py` appends one fixed sentence per
  category (e.g. bug → "our team typically looks into reports like this within a few business
  days"). The LLM never writes that sentence itself.
- Invent platform facts. The system prompt gives it what AI4IA is and does, and instructs it not
  to state anything beyond that.

This is a weaker safety boundary than `intent_router`'s (which hard-validates every slot against
the speaker registry and corpus catalog before trusting it) — a free-text reply can't be
validated that way. The mitigations here are the system prompt's explicit constraints, the fixed
category sentence taking timing language out of the LLM's hands entirely, and the human still
being in the loop (status never auto-resolves, so nothing bad ships silently).

## Context given to the LLM

- **Static platform knowledge**: inlined in `_SYSTEM_PROMPT` — what AI4IA is (a searchable,
  rights-cleared conference-talk archive that generates short documentary videos on request) and
  how it's used. There's no existing FAQ/help-content doc to source this from, so it's
  hand-written and lives in this file's `_SYSTEM_PROMPT` constant.
- **The submitting user's own recent job history**: last 5 jobs via
  `JobRepository.list_by_user(user_id)` (already exists, reused as-is), reduced to
  `{status, error.user_message if failed, created_at}` per job — enough for "why did my video
  fail" to get a specific answer, without dumping the full row. A lookup failure here is
  best-effort (logged, empty context used), never breaks the reply.

## Prompt-injection handling

Feedback text is untrusted user input that partially reaches another user (the same user)
unreviewed. It's wrapped with `app.pipeline.prompt_safety.wrap_untrusted`, and the system prompt
carries `UNTRUSTED_CONTENT_NOTICE` (the shared boilerplate every untrusted-content prompt in this
codebase uses) plus its own explicit line: text that looks like an instruction to the model is
ordinary feedback content, not a command.

## Rate limiting

Feedback submission was previously unlimited — harmless when it was free. It now triggers a paid
LLM call every time, so `feedback_submissions_per_user_per_hour` (default 10) caps it, checked via
`FeedbackRepository.count_recent_by_user` before creating the record. Over the limit returns 429.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `FEEDBACK_AUTO_REPLY_ENABLED` | `true` | Off skips the LLM call entirely, straight to the canned acknowledgment. |
| `FEEDBACK_AUTO_REPLY_MODEL` | `""` | Empty uses `prompt_guardrail_model`. |
| `FEEDBACK_AUTO_REPLY_TIMEOUT_SECONDS` | `20` | One attempt; past this, falls back. |
| `FEEDBACK_SUBMISSIONS_PER_USER_PER_HOUR` | `10` | Cost/abuse guard, see above. |

## Frontend

`AdminSettings.jsx`'s existing feedbacks table gets a read-only category badge and the auto-reply
text, shown alongside (not replacing) the existing "Resolution & Action" column. No new page, no
user-facing UI change — the auto-reply reaches the user through the existing notification
system (`notify_user`), the same path today's resolution message already uses.
