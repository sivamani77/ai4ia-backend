"""Speaker registry, name helpers and the registry-built alias map."""
from __future__ import annotations

from collections import defaultdict

import pytest

from app.services.speaker_names import description_roster, is_placeholder, norm_key, title_names
from app.services.speaker_registry import MULTIPLE_SPEAKERS, SpeakerRegistry, get_registry

FIXTURE = {
    "version": "test",
    "speakers": {
        "Cordel Green": {"aliases": ["Cordell Green", "Kodel Green"], "status": "confirmed"},
        "Chuck Howell": {"aliases": [], "status": "confirmed"},
        "Ben Green": {"aliases": [], "status": "confirmed"},
        "María José Labrador Blanes": {"aliases": [], "status": "confirmed"},
        "Samridhi Arora": {"aliases": [], "status": "confirmed"},
        "Ashita Singh": {"aliases": [], "status": "confirmed"},
    },
    "videos": {
        "GWMcNLuXnRA": {"kind": "single", "speakers": [{"name": "Chuck Howell", "role": "Speaker"}]},
        "AL8la33n0dw": {"kind": "panel", "speakers": [{"name": "Ashita Singh", "role": "Speaker"},
                                                      {"name": "Samridhi Arora", "role": "Speaker"}]},
        "Fuxk-tT6ngg": {"kind": "panel", "speakers": []},
        "TUieQkV_9Xc": {"kind": "single", "speakers": [{"name": "Samridhi Arora", "role": "Speaker"}]},
    },
}


@pytest.fixture
def registry() -> SpeakerRegistry:
    return SpeakerRegistry(FIXTURE)


class TestNames:
    def test_norm_key_drops_accents_titles_and_punctuation(self):
        assert norm_key("Prof. Dr. María-José  Labrador") == "maria jose labrador"
        assert norm_key("Dr. Karen Nelson") == norm_key("karen nelson")

    def test_placeholders(self):
        assert is_placeholder("I") and is_placeholder("null") and is_placeholder(None)
        assert not is_placeholder("Nisha")

    def test_description_roster_moderators_and_list(self):
        text = (
            "The panel focuses on AI.\n\n"
            "Moderators: Dr. Dariia Opryshko and Andrea Millwood Hargrave.\n\n"
            "The speakers on the panel include:\n"
            "Dr. Stefaan Verhulst (Co-Founder of The GovLab)\n"
            "Stefan Baack, PhD (independent researcher)\n"
        )
        assert description_roster(text) == [
            ("Dariia Opryshko", "Moderator"), ("Andrea Millwood Hargrave", "Moderator"),
            ("Stefaan Verhulst", "Speaker"), ("Stefan Baack", "Speaker"),
        ]

    def test_description_roster_pipe_lines_and_bare_header(self):
        text = (
            "Presenters:\n"
            "Dr. Andy Knight, Arts | Professor\n"
            "Miguel “Steppa” Williams | Digicel Foundation\n\n"
            "Moderator\n"
            "Dr. Geoffrey Rockwell, Professor\n\n"
            "Partners:\n"
            "Broadcasting Commission of Jamaica (BCJ) | UNESCO\n"
        )
        assert description_roster(text) == [
            ("Andy Knight", "Presenter"), ("Miguel Williams", "Presenter"), ("Geoffrey Rockwell", "Moderator"),
        ]

    def test_description_without_roles_gives_nothing(self):
        assert description_roster("Sebastian Waisbrot set the conceptual groundwork.") == []

    @pytest.mark.parametrize("title,expected", [
        ("(6K) C Howell - The need for early STS considerations", ["C Howell"]),
        ("H Warsame & M Ekotto - Curated Realities", ["H Warsame", "M Ekotto"]),
        ("M Schneider   Information Ethics and Critique", ["M Schneider"]),
        ("Keynote: Adv Tlakula", ["Adv Tlakula"]),
        ("Europe Satellite Event - Panel 1", []),
        ("Canada Regional Event", []),
    ])
    def test_title_names(self, title, expected):
        assert title_names(title) == expected


class TestLookup:
    def test_canonical_exact_alias_and_honorific(self, registry):
        assert registry.canonical("Cordel Green") == "Cordel Green"
        assert registry.canonical("Kodel Green") == "Cordel Green"
        assert registry.canonical("Prof. Cordell Green") == "Cordel Green"
        assert registry.canonical("maria jose labrador blanes") == "María José Labrador Blanes"

    def test_initials_only_when_unique(self, registry):
        assert registry.canonical("C Howell") == "Chuck Howell"
        assert registry.canonical("B Green") == "Ben Green"
        assert registry.canonical("G Green") is None  # no match
        assert registry.canonical("X Unknown") is None

    def test_canonicalize_keeps_unknown_and_drops_placeholders(self, registry):
        assert registry.canonicalize("Someone New") == "Someone New"
        assert registry.canonicalize("I") is None
        assert registry.canonicalize("null") is None

    def test_missing_file_passes_names_through(self, tmp_path):
        empty = SpeakerRegistry.load(tmp_path / "missing.json")
        assert empty.canonicalize("Kodel Green") == "Kodel Green"


class TestSpeakerFields:
    def test_single_speaker_video_overrides_garbled_name(self, registry):
        fields = registry.speaker_fields("0074_GWMcNLuXnRA_(6K) C Howell - STS", "Chuck Out", "video-level")
        assert fields == {"video_id": "GWMcNLuXnRA", "speaker": "Chuck Howell",
                          "speaker_attribution": "video-level", "speaker_roster": "Chuck Howell"}

    def test_panel_clears_speaker_and_keeps_roster(self, registry):
        fields = registry.speaker_fields("0200_AL8la33n0dw_India Satellite Event", "Ashita Singh", "video-level")
        assert fields["speaker"] == ""
        assert fields["speaker_attribution"] == MULTIPLE_SPEAKERS
        assert fields["speaker_roster"] == "Ashita Singh; Samridhi Arora"

    def test_panel_with_unknown_names(self, registry):
        fields = registry.speaker_fields("0300_Fuxk-tT6ngg_UNESCO Caribbean AI Policy Roadmap", "null", "")
        assert (fields["speaker"], fields["speaker_attribution"], fields["speaker_roster"]) == (
            "", MULTIPLE_SPEAKERS, "")

    def test_cut_short_video_id_is_repaired(self, registry):
        fields = registry.speaker_fields("0252_TUieQkV_9Xc_(7I) G Radwan", "S ARORA", "video-level",
                                         video_id="TUieQkV")
        assert fields["video_id"] == "TUieQkV_9Xc"
        assert fields["speaker"] == "Samridhi Arora"

    def test_unknown_video_canonicalizes(self, registry):
        fields = registry.speaker_fields("0999_AAAAAAAAAAA_New Talk", "Kodel Green", "video-level")
        assert fields["speaker"] == "Cordel Green"
        assert fields["speaker_roster"] == "Cordel Green"

    def test_unknown_video_placeholder_is_blanked(self, registry):
        fields = registry.speaker_fields("0999_AAAAAAAAAAA_New Talk", "I", "video-level")
        assert fields["speaker"] == "" and fields["speaker_roster"] == ""

    def test_unknown_multi_speaker_video_keeps_attribution(self, registry):
        fields = registry.speaker_fields("0999_AAAAAAAAAAA_New Panel", None, MULTIPLE_SPEAKERS)
        assert fields["speaker_attribution"] == MULTIPLE_SPEAKERS and fields["speaker"] == ""

    def test_is_idempotent(self, registry):
        first = registry.speaker_fields("0074_GWMcNLuXnRA_x", "Chuck Out", "video-level")
        second = registry.speaker_fields("0074_GWMcNLuXnRA_x", first["speaker"], first["speaker_attribution"],
                                         first["video_id"])
        assert first == second


class TestShippedRegistry:
    """Invariants of config/speaker_registry.json itself."""

    def test_loads(self):
        registry = get_registry()
        assert registry.version == "speaker_registry_v1"
        assert len(registry.speakers) > 300

    def test_every_roster_name_is_a_speaker(self):
        registry = get_registry()
        missing = [(v, n) for v in registry.video_ids() for n in registry.video(v).names
                   if n not in registry.speakers]
        assert missing == []

    def test_no_key_belongs_to_two_people(self):
        owners = defaultdict(set)
        for canonical, entry in get_registry().speakers.items():
            owners[norm_key(canonical)].add(canonical)
            for alias in entry["aliases"]:
                owners[norm_key(alias)].add(canonical)
        assert {k: v for k, v in owners.items() if len(v) > 1} == {}

    def test_decisions(self):
        registry = get_registry()
        assert registry.video("QP_5ovm6wn8").names == ["Samridhi Arora"]
        assert registry.canonical("Kodel Green") == "Cordel Green"
        assert registry.canonical("Dario Pryshko") == "Dariia Opryshko"
        assert registry.canonical("Nancy Kwangwa") == "Nancy Nkwanga"
        assert registry.video("Fuxk-tT6ngg").kind == "panel"
        assert registry.canonical("Kelly") is None


class TestPromptAliasMap:
    def test_built_from_registry(self):
        from app.pipeline.storyline_gen.prompt_intent import SPEAKER_ALIAS_MAP

        assert SPEAKER_ALIAS_MAP["kodel green"] == "Cordel Green"
        assert SPEAKER_ALIAS_MAP["codrel green"] == "Cordel Green"
        assert all(len(key.split()) >= 2 for key in SPEAKER_ALIAS_MAP)
