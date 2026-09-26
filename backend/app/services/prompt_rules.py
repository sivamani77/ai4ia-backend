"""L0 prompt rules: cheap checks that need no LLM (docs/design/intent-layer-design.md §4).

They reject only prompts with no usable subject. Anything doubtful passes on
to the LLM guardrail, which judges subject matter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from app.core.error_messages import PROMPT_RULE_MESSAGES
from app.core.settings import settings

_TOKEN = re.compile(r"[A-Za-z0-9']+")
_VOWELS = set("aeiouy")

_GREETING_WORDS = frozenset("""
    hi hello hey hiya heya yo howdy greetings good morning afternoon evening night
    how are you doing what's whats up sup thanks thank ok okay test testing there
    all everyone world today tonight friend friends
""".split())

# Flagged for review only, never blocked: "prompt injection" is a real AI topic.
_INJECTION = re.compile(
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above)\s+(instructions|prompts|rules)"
    r"|disregard\s+(all\s+|the\s+)?(previous|prior|above)"
    r"|system\s+prompt|you\s+are\s+now|developer\s+mode",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RuleResult:
    reason_code: str | None = None
    message: str | None = None
    flags: dict = field(default_factory=dict)

    @property
    def rejected(self) -> bool:
        return self.reason_code is not None


@lru_cache(maxsize=1)
def _known_words() -> frozenset[str]:
    """Words the letter heuristic might misjudge: prompt stopwords and speaker names."""
    from app.pipeline.storyline_gen.prompt_intent import STOPWORDS
    from app.services.speaker_registry import get_registry

    words = set(STOPWORDS)
    registry = get_registry()
    for canonical, entry in registry.speakers.items():
        for name in (canonical, *(entry.get("aliases") or [])):
            words.update(t.lower() for t in _TOKEN.findall(name))
    return frozenset(words)


def _longest_consonant_run(word: str) -> int:
    longest = run = 0
    for ch in word:
        run = 0 if ch in _VOWELS else run + 1
        longest = max(longest, run)
    return longest


def is_wordlike(token: str) -> bool:
    """True for anything that could be a word, name, acronym or code like AI4IA."""
    lower = token.lower().strip("'")
    if not lower or lower.isdigit():
        return False
    if lower in _known_words():
        return True
    bare = re.sub(r"'?s$", "", token)
    if bare.isupper() and 2 <= len(bare) <= 6:  # acronyms: UNESCO, LLMs, GPT's
        return True
    if any(c.isdigit() for c in lower):  # AI4IA, GPT4, 2024s
        return len(lower) <= 10 and any(c.isalpha() for c in lower)
    letters = [c for c in lower if c.isalpha()]
    vowels = sum(c in _VOWELS for c in letters)
    if not vowels or len(letters) > 25 or _longest_consonant_run(letters) > 5:
        return False
    # Long keyboard mashes can still contain a vowel or two.
    return len(letters) < 8 or vowels / len(letters) >= 0.1


def check_prompt(prompt: str | None) -> RuleResult:
    """First failing rule wins; flags are returned either way."""
    text = (prompt or "").strip()
    flags = {"injection_pattern": True} if _INJECTION.search(text) else {}
    tokens = _TOKEN.findall(text)

    def reject(code: str, **fmt) -> RuleResult:
        return RuleResult(code, PROMPT_RULE_MESSAGES[code].format(**fmt), flags)

    if not tokens:
        return reject("prompt_empty")
    if len(text) > settings.prompt_max_chars:
        return reject("prompt_too_long", max_chars=settings.prompt_max_chars)
    if all(t.lower() in _GREETING_WORDS for t in tokens):
        return reject("prompt_greeting")
    if not any(is_wordlike(t) for t in tokens):
        return reject("prompt_gibberish")
    return RuleResult(flags=flags)
