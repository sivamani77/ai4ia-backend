"""Facts stated in archive-overview narration.

These end up spoken in a permanent video, so the rules are: every figure is
computed live from the corpus, the speaker count is never overstated, and no
real person's name is spoken unless its spelling is unambiguous.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.pipeline.storyline_gen.corpus_facts import (  # noqa: E402
    build_corpus_facts,
    format_facts_for_prompt,
)


def _distinct_names(n):
    """Names dissimilar enough that none cluster as spelling variants."""
    import hashlib
    out = []
    for i in range(n):
        h = hashlib.sha1(str(i).encode()).hexdigest()
        out.append(f"{h[:7]} {h[7:14]}")
    return out


def frag(video, speaker="", theme="T1", chapter="C1", year=""):
    return {"source_video": video, "speaker": speaker, "primary_theme": theme,
            "primary_chapter": chapter, "video_year": year}


class TestCounts:
    def test_counts_distinct_videos_themes_and_chapters(self):
        facts = build_corpus_facts([
            frag("v1", theme="A", chapter="X"), frag("v1", theme="A", chapter="Y"),
            frag("v2", theme="B", chapter="X"),
        ])
        assert facts["video_count"] == 2
        assert facts["fragment_count"] == 3
        assert facts["theme_count"] == 2
        assert facts["chapter_count"] == 2

    def test_themes_are_ranked_by_how_much_they_are_discussed(self):
        facts = build_corpus_facts([frag("v1", theme="Rare")] + [frag("v2", theme="Common")] * 3)
        assert [t["name"] for t in facts["themes"]] == ["Common", "Rare"]


class TestSpeakerCount:
    def test_extractor_placeholders_are_not_speakers(self):
        facts = build_corpus_facts([frag("v1", "I"), frag("v2", "null"), frag("v3", "Ana Ruiz")])
        assert facts["speaker_count"] == 1

    def test_honorifics_do_not_split_one_person_in_two(self):
        facts = build_corpus_facts([frag("v1", "Dr Tabani Moyo"), frag("v2", "Tabani Moyo")])
        assert facts["speaker_count"] == 1

    def test_known_aliases_resolve_to_one_person(self):
        facts = build_corpus_facts([frag("v1", "Cordell Green"), frag("v2", "Cordel Green")])
        assert facts["speaker_count"] == 1

    def test_near_miss_spellings_lower_the_bound_but_are_not_merged(self):
        """Merging them could join two people, so they only bound the count."""
        facts = build_corpus_facts([frag("v1", "Marta Kowalska"), frag("v2", "Marta Kowalski")])
        assert facts["speaker_count"] == 2
        assert facts["speaker_count_lower_bound"] == 1

    def test_registry_aliases_are_merged(self):
        """Spellings the speaker registry confirms as one person count once."""
        facts = build_corpus_facts([frag("v1", "Dario Pryshko"), frag("v2", "Dario Prishko")])
        assert facts["speaker_count"] == 1

    def test_stated_count_never_exceeds_the_lower_bound(self):
        facts = build_corpus_facts([frag(f"v{i}", n) for i, n in enumerate(_distinct_names(263))])
        assert facts["speaker_count_lower_bound"] == 263
        stated = int(format_facts_for_prompt(facts).split("Speakers: over ")[1].split()[0])
        assert stated == 260

    def test_over_merging_only_ever_lowers_the_stated_count(self):
        """Near-identical names chain together and collapse the bound. That
        errs conservative — the stated figure drops, it is never inflated."""
        facts = build_corpus_facts([frag(f"v{i}", f"Person Number{i}") for i in range(263)])
        assert facts["speaker_count_lower_bound"] < facts["speaker_count"]
        stated = int(format_facts_for_prompt(facts).split("Speakers: over ")[1].split()[0])
        assert stated <= facts["speaker_count"]


class TestNamedSpeakers:
    def test_alias_resolved_name_uses_the_canonical_spelling(self):
        facts = build_corpus_facts([frag("v1", "Cordell Green")] * 3)
        assert facts["top_speakers"][0]["name"] == "Cordel Green"

    def test_ambiguous_spellings_are_withheld(self):
        facts = build_corpus_facts(
            [frag("v1", "Marta Kowalska")] * 5 + [frag("v2", "Marta Kowalski")] * 4
            + [frag("v3", "Ana Ruiz")]
        )
        assert [s["name"] for s in facts["top_speakers"]] == ["Ana Ruiz"]

    def test_registry_aliases_are_named_canonically(self):
        facts = build_corpus_facts(
            [frag("v1", "Dario Pryshko")] * 5 + [frag("v2", "Dario Prishko")] * 4
            + [frag("v3", "Ana Ruiz")]
        )
        assert [s["name"] for s in facts["top_speakers"]] == ["Dariia Opryshko", "Ana Ruiz"]

    def test_first_name_only_is_withheld(self):
        facts = build_corpus_facts([frag("v1", "Kelly")] * 5 + [frag("v2", "Ana Ruiz")])
        assert [s["name"] for s in facts["top_speakers"]] == ["Ana Ruiz"]


class TestYears:
    def test_chroma_blank_year_is_omitted_not_guessed(self):
        facts = build_corpus_facts([frag("v1", year="")])
        assert "year_first" not in facts
        assert "Years covered" not in format_facts_for_prompt(facts)

    def test_years_span_the_videos(self):
        facts = build_corpus_facts([frag("v1", year=2021), frag("v2", year="2025")])
        assert (facts["year_first"], facts["year_last"]) == (2021, 2025)
