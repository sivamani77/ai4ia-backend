"""Canonical speaker-name aliases.

scripts/04b_extract_speakers.py identifies each video's speaker(s)
independently via an LLM, with no cross-video pass — so the same real
person can come out spelled differently in different videos, often
because the source is a noisy ASR transcript of a spoken
self-introduction (e.g. "Dario Prishko" / "Daria Pryshko" / "Dario
Pryshko" were all the same speaker).

This is the single source of truth for known duplicates, checked both
at extraction time (04b_extract_speakers.py, so a newly-ingested video
referencing an already-known speaker is canonicalized immediately) and
during the Postgres backfill (scripts/migrate_fragments_to_postgres.py,
so already-migrated rows get corrected on the next re-run).

Only add a name here once you're confident it's the same person as the
canonical spelling — a single-letter typo, initials vs. full name, or a
title prefix, always with an exact surname match. A surname mismatch or
an unusual first-name difference (e.g. "M Oliver" vs "M Oliveira",
"Zabel Ferrari" vs "Isabela Ferrari") could be two different real
people — leave those separate rather than guessing.
"""
from __future__ import annotations

ALIASES: dict[str, str] = {
    "Kodel Green": "Cordell Green",
    "Cordel Green": "Cordell Green",
    "Daria Pryshko": "Dario Prishko",
    "Dario Pryshko": "Dario Prishko",
    "Christena Williams": "Christina Williams",
    "Denise Eldemire-Shearer": "Prof. Denise Eldemire-Shearer",
    "James Sheer": "James Sherer",
    "F Roumate": "Fatima Roumate",
    "Tyler Janes": "Tyler James",
    "Ntsibane Ntatlapa": "Ntsibane Nntllapa",
    "Dr Tabani Moyo": "Tabani Moyo",
    "Siva Prasad Rambhatla": "S P Rambhatla",
    "Felipe Chivas Ortiz": "Felipe Chibas Ortiz",
}


def canonicalize(name: str | None) -> str | None:
    """Return the canonical spelling for a known alias, or `name` unchanged
    if it's not a known duplicate (including if it's already canonical)."""
    if not name:
        return name
    return ALIASES.get(name, name)
