"""Speaker prompts: the speaker's own fragments are the evidence (A), and with a
topic, their closest fragments count even without the literal word (B)."""
from __future__ import annotations

from app.pipeline.storyline_gen.candidate_scoring import annotate_candidate_relevance

SPEAKER = "Cordel Green"


def frag(fid, sem, text="policy and oversight of platforms", speaker=SPEAKER, **extra):
    return {"fragment_id": fid, "semantic_score": sem, "text": text, "summary": text, "speaker": speaker,
            "primary_theme": "AI Ethics, Governance & Policy", "quote_strength": 0.6, "clarity_score": 0.9,
            "duration": 30.0, "start": 0.0, "end": 30.0, **extra}


def profile(topic_terms=(), panel_only=False):
    return {"requested_speaker": SPEAKER, "topic_terms": list(topic_terms), "scope": "general",
            "speaker_panel_only": panel_only}


def roles(frags, prof):
    return {c["fragment_id"]: (c["relevance_role"], c["final_video_allowed"])
            for c in annotate_candidate_relevance(frags, "p", prof)}


class TestSpeakerOnly:  # A
    def test_own_fragments_are_selectable_at_low_similarity(self):
        # Measured: "What did Cordell Green say?" scores his own talk 0.58-0.64; all were rejected before.
        got = roles([frag("a", 0.58), frag("b", 0.61)], profile())
        assert got == {"a": ("exact", True), "b": ("exact", True)}

    def test_another_speaker_is_still_rejected(self):
        assert roles([frag("x", 0.9, speaker="Andrea Millwood Hargrave")], profile())["x"][0] == "reject"

    def test_panel_fragment_is_not_his_own_voice(self):
        got = roles([frag("p", 0.6, speaker="", speaker_roster=SPEAKER, speaker_match_panel=True)], profile())
        assert got["p"][0] != "exact"

    def test_panel_only_speaker_keeps_panel_fragments(self):
        got = roles([frag("p", 0.6, speaker="", speaker_roster=SPEAKER, speaker_match_panel=True)],
                    profile(panel_only=True))
        assert got["p"] == ("exact", True)


class TestSpeakerTopic:  # B
    def test_closest_third_is_promoted_without_the_literal_word(self):
        frags = [frag(f"f{i}", s) for i, s in enumerate([0.69, 0.66, 0.63, 0.60, 0.58, 0.52])]
        got = roles(frags, profile(["regulation"]))
        assert [f for f, (role, _) in got.items() if role == "strong_related"] == ["f0", "f1"]
        assert got["f0"][1] is True
        assert got["f5"][0] in ("reject", "contextual")          # far off topic stays out

    def test_floor_applies_even_at_the_top(self):
        got = roles([frag("a", 0.54), frag("b", 0.50), frag("c", 0.49)], profile(["regulation"]))
        assert all(role != "strong_related" for role, _ in got.values())

    def test_literal_matches_are_never_demoted(self):
        text = "regulation of media platforms"   # both topic words: exact before ranking
        got = roles([frag("a", 0.70, text=text), frag("b", 0.60), frag("c", 0.59)], profile(["regulation", "media"]))
        assert got["a"] == ("exact", True)

    def test_other_speakers_and_panels_are_never_promoted(self):
        frags = [frag("own", 0.60), frag("other", 0.80, speaker="Ben Green"),
                 frag("panel", 0.80, speaker="", speaker_roster=SPEAKER, speaker_match_panel=True)]
        got = roles(frags, profile(["regulation"]))
        assert got["other"][0] == "reject" and got["panel"][0] != "strong_related"
        assert got["own"][0] == "strong_related"

    def test_topic_prompt_without_a_speaker_is_unchanged(self):
        prof = {"requested_speaker": None, "topic_terms": ["regulation"], "scope": "specific_topic"}
        got = annotate_candidate_relevance([frag("a", 0.58)], "p", prof)
        assert got[0]["relevance_role"] in ("reject", "contextual")
