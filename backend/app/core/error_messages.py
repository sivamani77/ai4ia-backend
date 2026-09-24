"""Maps internal failures to short, user-facing messages.

Technical detail (exception text, tracebacks, ffmpeg/ffprobe output) stays in
logs and the job's `details` field for support/debugging — this module is the
one place that decides what a user actually sees. An exception can carry its
own `.user_message` (see ExportIntegrityError, GuardrailRejection) for a more
specific message; otherwise callers fall back to the table below by error_type.
"""

_BY_ERROR_TYPE: dict[str, str] = {
    "model_unavailable": (
        "Our AI models are still starting up. Please try again in a few minutes."
    ),
    "guardrail_rejected": (
        "This request couldn't be processed as written. Please rephrase your topic or question."
    ),
    "export_integrity": (
        "We hit a problem while assembling your video and stopped before producing "
        "a broken file. Please try generating it again."
    ),
    "subprocess_failure": (
        "Video rendering failed unexpectedly. Please try again — if this keeps "
        "happening, contact support."
    ),
    "validation_error": (
        "Some of the details provided weren't valid. Please check your input and try again."
    ),
}

# L0 prompt rules (app/services/prompt_rules.py), keyed by reason code.
PROMPT_RULE_MESSAGES: dict[str, str] = {
    "prompt_empty": "Please enter a topic or question about the AI4IA archive.",
    "prompt_too_long": "Your prompt is too long. Please keep it under {max_chars} characters.",
    "prompt_greeting": (
        "Hi! Ask about a topic, speaker or theme from the AI4IA archive, for example "
        "\"What did speakers say about deepfakes?\""
    ),
    "prompt_gibberish": "We couldn't read that as a topic or question. Please rephrase it.",
}

DEFAULT_MESSAGE = (
    "Something went wrong while processing your request. Please try again, and "
    "contact support if the problem continues."
)


class ExportSubprocessError(RuntimeError):
    """The 09_export_video.py subprocess exited non-zero or produced no output.

    `message` carries the raw ffmpeg/ffprobe stderr tail for logs; a client
    should never see that — use `user_message` (see friendly_message below).
    """

    user_message = (
        "Video rendering failed unexpectedly. Please try again — if this keeps "
        "happening, contact support."
    )


def friendly_message(error_type: str | None, exc: Exception | None = None) -> str:
    """User-facing message for a failure. Prefers exc.user_message (set by
    error classes that know a better answer than the generic table), then the
    error_type table, then a generic fallback — never the raw exception text."""
    if exc is not None:
        specific = getattr(exc, "user_message", None)
        if specific:
            return specific
    return _BY_ERROR_TYPE.get(error_type or "", DEFAULT_MESSAGE)
