"""Detecting prompts that ask about the archive itself.

Every subject noun in "how many speakers are there, what are the major themes"
is a STOPWORD, so topic_terms comes back as pure question scaffolding and the
prompt was scored as a topical request demanding evidence no clip can supply.
The detector keys on what survives stopword removal: an archive noun plus no
remaining subject means the question is about the archive.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.pipeline.storyline_gen.prompt_intent import (  # noqa: E402
    detect_archive_overview,
    detect_prompt_intent,
    extract_topic_terms,
)

CORPUS = [{"source_video": "0001_abcdefghijk_Some Talk", "fragment_id": "f1"}]

OVERVIEW = [
    "tell me what is ai4ia about, how many speakers are there and what is the "
    "conferences about, what are the major themes, chapters and important and "
    "most discussed topics?",
    "give me an overview of the AI4IA conference",
    "summarize the whole archive",
    "what are the main themes covered across all videos",
    "how many speakers are in the archive?",
    "what is AI4IA",
    "what's in the archive",
    "how many videos are there",
    "what chapters does the conference cover",
    "most discussed topics in the archive",
    "introduce the AI4IA conference",
    "give me a summary of all the talks",
]

TOPICAL = [
    "what did the speakers say about access to information?",
    "AI governance and transparency",
    "women empowerment in artificial intelligence",
    "which talks discuss AI governance in Africa?",
    "what themes relate to healthcare",
    "deepfakes and copyright law",
    "AI in education",
    "make a video about the conference themes around youth",
]


def overview(prompt: str, **kwargs) -> bool:
    return detect_archive_overview(prompt, extract_topic_terms(prompt), **kwargs)


class TestDetection:
    @pytest.mark.parametrize("prompt", OVERVIEW)
    def test_archive_questions_are_detected(self, prompt):
        assert overview(prompt)

    @pytest.mark.parametrize("prompt", TOPICAL)
    def test_topical_prompts_are_left_alone(self, prompt):
        """Mentioning speakers or talks is not enough when a subject remains."""
        assert not overview(prompt)

    def test_archive_is_subtracted_as_a_meta_noun(self):
        # "archive" is not a STOPWORD, so it reaches topic_terms and read as a
        # subject until the meta nouns were subtracted too.
        assert "archive" in extract_topic_terms("summarize the whole archive")
        assert overview("summarize the whole archive")

    def test_needs_an_archive_noun(self):
        """Pure scaffolding with no archive noun is not an overview request."""
        assert not overview("what are the most important and major ones")

    def test_speaker_request_takes_precedence(self):
        assert not overview("what is AI4IA", requested_speaker="Cordel Green")

    def test_title_request_takes_precedence(self):
        assert not overview("what is AI4IA", requested_source_video="0001_x_Talk")


class TestIntentProfile:
    def test_overview_sets_intent_scope_and_evidence(self):
        profile = detect_prompt_intent("what is AI4IA", [], CORPUS)
        assert profile["intent_type"] == "archive_overview"
        assert profile["scope"] == "archive_overview"
        assert profile["evidence_requirement"] == "corpus_facts"

    def test_topical_prompt_profile_is_unchanged(self):
        profile = detect_prompt_intent("AI governance and transparency", [], CORPUS)
        assert profile["intent_type"] != "archive_overview"
        assert profile["evidence_requirement"] == "direct_topic_evidence"
