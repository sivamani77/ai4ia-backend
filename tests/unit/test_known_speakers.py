"""Speaker detection reads the speaker registry, not local metadata files."""
from __future__ import annotations

import pytest

from app.pipeline.storyline_gen.prompt_intent import detect_requested_speaker, load_known_speakers
from app.services.speaker_registry import get_registry

REGISTRY = get_registry()
MULTI_WORD_NAMES = [
    (name, canonical)
    for canonical, entry in REGISTRY.speakers.items()
    for name in (canonical, *(entry.get("aliases") or []))
    if len(name.split()) >= 2
]


def test_known_speakers_are_the_registry_names():
    assert load_known_speakers() == list(REGISTRY.speakers)


@pytest.mark.parametrize("name,canonical", MULTI_WORD_NAMES)
def test_every_registry_name_resolves_to_its_canonical_name(name, canonical):
    prompt = f"What did {name} say about AI regulation?"
    assert detect_requested_speaker(prompt, load_known_speakers()) == canonical


def test_result_does_not_depend_on_the_known_list():
    # Deployed stage 07 used to get an empty list, since the image has no data/.
    for name, _ in MULTI_WORD_NAMES:
        prompt = f"What did {name} say about AI regulation?"
        assert detect_requested_speaker(prompt, load_known_speakers()) == detect_requested_speaker(prompt, [])
