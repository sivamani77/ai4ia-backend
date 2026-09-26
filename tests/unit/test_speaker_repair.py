"""Pure parts of the speaker repair: plan, skip rules, checks, file rewrite and rollback."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from app.services import speaker_repair as repair
from app.services.speaker_registry import MULTIPLE_SPEAKERS, SpeakerRegistry

REGISTRY = SpeakerRegistry({
    "version": "test",
    "speakers": {
        "Chuck Howell": {"aliases": ["Chuck Out"]},
        "Cordel Green": {"aliases": ["Cordell Green"]},
        "Ashita Singh": {"aliases": []},
        "Samridhi Arora": {"aliases": []},
    },
    "videos": {
        "GWMcNLuXnRA": {"kind": "single", "speakers": [{"name": "Chuck Howell", "role": "Speaker"}]},
        "AL8la33n0dw": {"kind": "panel", "speakers": [{"name": "Ashita Singh", "role": "Speaker"},
                                                      {"name": "Samridhi Arora", "role": "Speaker"}]},
        "TUieQkV_9Xc": {"kind": "single", "speakers": [{"name": "Samridhi Arora", "role": "Speaker"}]},
    },
})

SV_SINGLE = "0074_GWMcNLuXnRA_(6K) C Howell - STS"
SV_PANEL = "0200_AL8la33n0dw_India Satellite Event"
SV_CUT = "0252_TUieQkV_9Xc_(7I) Talk"
SV_OTHER = "0999_AAAAAAAAAAA_New Talk"


def frag(fid, source_video, speaker, attribution="video-level", video_id=None, roster=None):
    return repair.Fragment(fid, source_video, {"video_id": video_id, "speaker": speaker,
                                               "speaker_attribution": attribution, "speaker_roster": roster})


def stores():
    return {
        "a": frag("a", SV_SINGLE, "Chuck Out", video_id="GWMcNLuXnRA"),
        "b": frag("b", SV_PANEL, "Ashita Singh", video_id="AL8la33n0dw"),
        "c": frag("c", SV_CUT, "S ARORA", video_id="TUieQkV"),
        "d": frag("d", SV_OTHER, "Cordell Green", video_id="AAAAAAAAAAA"),
    }


def applied(fragments, changes):
    out = {}
    for fid, f in fragments.items():
        raw = dict(f.raw)
        raw.update(changes.get(fid, {}))
        out[fid] = repair.Fragment(fid, f.source_video, raw)
    return out


class TestPlan:
    def test_plan_changes_and_report(self):
        pg, chroma = stores(), stores()
        plan = repair._plan(pg, chroma, REGISTRY)
        assert set(plan["chroma_changes"]) == {"a", "b", "c", "d"}
        assert plan["chroma_changes"]["b"]["speaker_attribution"] == MULTIPLE_SPEAKERS
        assert plan["chroma_changes"]["c"]["video_id"] == "TUieQkV_9Xc"
        assert plan["report"]["videos_not_in_registry"] == ["AAAAAAAAAAA"]
        assert plan["report"]["speakers_not_in_registry"] == {}
        assert plan["report"]["chroma"]["changed"] == 4

    def test_second_run_changes_nothing(self):
        chroma = stores()
        plan = repair._plan(chroma, chroma, REGISTRY)
        fixed = applied(chroma, plan["chroma_changes"])
        again = repair._plan(fixed, fixed, REGISTRY)
        assert again["chroma_changes"] == {} and again["pg_changes"] == {}

    def test_checks_pass_after_apply_and_fail_before(self):
        chroma = stores()
        fixed = applied(chroma, repair._plan(chroma, chroma, REGISTRY)["chroma_changes"])
        counts = {"postgres": 4, "chroma": 4}
        assert all(c["ok"] for c in repair._checks(fixed, fixed, REGISTRY, counts))
        failing = {c["name"] for c in repair._checks(chroma, fixed, REGISTRY, counts) if not c["ok"]}
        assert {"postgres_matches_registry", "stores_agree", "no_cut_short_video_ids"} <= failing


class TestSkipReason:
    @pytest.mark.parametrize("mode,marker,expected", [
        ("dry_run", None, None),
        ("apply", None, None),
        ("rollback", None, "nothing_to_roll_back"),
        ("apply", {"status": "applied"}, "already_done"),
        ("dry_run", {"status": "applied"}, "already_done"),
        ("dry_run", {"status": "dry_run", "registry_version": "fp"}, "already_done"),
        ("dry_run", {"status": "dry_run", "registry_version": "old"}, None),
        ("apply", {"status": "dry_run", "registry_version": "fp"}, None),
        ("apply", {"status": "failed", "attempts": 2}, None),
        ("apply", {"status": "failed", "attempts": 3}, "too_many_failures"),
        ("apply", {"status": "rolled_back"}, None),
        ("rollback", {"status": "applied"}, None),
        ("rollback", {"status": "failed"}, None),
        ("rollback", {"status": "rolled_back"}, "already_done"),
        ("rollback", {"status": "dry_run"}, "nothing_to_roll_back"),
    ])
    def test_matrix(self, mode, marker, expected):
        assert repair._skip_reason(mode, marker, "fp", force=False) == expected

    def test_force_ignores_marker(self):
        assert repair._skip_reason("apply", {"status": "applied"}, "fp", force=True) is None


class TestFiles:
    @pytest.fixture
    def files(self, tmp_path, monkeypatch):
        dirs = {"roster": tmp_path / "speakers", "classified": tmp_path / "classified"}
        for d in dirs.values():
            d.mkdir()
        monkeypatch.setattr(repair.s3_helpers, "use_s3", lambda: False)
        monkeypatch.setattr(repair, "_LOCAL_DIRS", dirs)
        monkeypatch.setattr(repair, "_LOCAL_ROOT", tmp_path / "backup")
        classified = {"source_video": SV_CUT, "video_id": "TUieQkV",
                      "fragments": [{"fragment_id": "c", "source_video": SV_CUT, "speaker": "S ARORA",
                                     "speaker_attribution": "video-level", "video_id": "TUieQkV"}]}
        (dirs["classified"] / f"{SV_CUT}.classified.json").write_text(json.dumps(classified))
        roster = {"video_id": "AL8la33n0dw", "source_video": SV_PANEL, "primary_speaker": "Kelly",
                  "speakers": [{"name": "Kelly", "role": "Speaker"}]}
        (dirs["roster"] / "AL8la33n0dw.json").write_text(json.dumps(roster))
        return dirs

    def test_rewrite_backup_and_rollback(self, files):
        originals = {p: p.read_bytes() for d in files.values() for p in d.iterdir()}
        source_videos = {"AL8la33n0dw": SV_PANEL, "GWMcNLuXnRA": SV_SINGLE, "TUieQkV_9Xc": SV_CUT}

        counts = repair._apply_files(repair._Files(), REGISTRY, source_videos)
        assert counts == {"classified_rewritten": 1, "rosters_rewritten": 1, "rosters_created": 2}

        payload = json.loads((files["classified"] / f"{SV_CUT}.classified.json").read_text())
        assert payload["video_id"] == "TUieQkV_9Xc"
        assert payload["fragments"][0]["speaker"] == "Samridhi Arora"
        roster = json.loads((files["roster"] / "AL8la33n0dw.json").read_text())
        assert roster["primary_speaker"] is None and roster["attribution"] == MULTIPLE_SPEAKERS
        assert [s["name"] for s in roster["speakers"]] == ["Ashita Singh", "Samridhi Arora"]
        assert (files["roster"] / "GWMcNLuXnRA.json").exists()

        # A second run is skipped by the files marker.
        assert repair._apply_files(repair._Files(), REGISTRY, source_videos) == {"skipped": 1}

        restored = repair._rollback_files(repair._Files())
        assert restored["rosters_deleted"] == 2
        assert {p: p.read_bytes() for d in files.values() for p in d.iterdir()} == originals


class TestCleanup:
    NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)
    FRESH = {"status": "applied", "updated_at": NOW - timedelta(days=2)}
    OLD = {"status": "applied", "updated_at": NOW - timedelta(days=8)}

    @pytest.mark.parametrize("mode,repair_marker,cleanup_marker,force,expected", [
        ("cleanup", None, None, False, "repair_not_applied"),
        ("cleanup", {"status": "dry_run"}, None, True, "repair_not_applied"),
        ("cleanup_preview", FRESH, None, False, None),
        ("cleanup_preview", FRESH, {"status": "cleanup_preview"}, False, "already_done"),
        ("cleanup", FRESH, None, False, "too_soon_after_apply"),
        ("cleanup", FRESH, None, True, None),
        ("cleanup", OLD, None, False, None),
        ("cleanup", OLD, {"status": "cleaned"}, True, "already_done"),
    ])
    def test_skip_rules(self, mode, repair_marker, cleanup_marker, force, expected):
        assert repair._cleanup_skip_reason(mode, repair_marker, cleanup_marker, force, self.NOW) == expected

    def test_files_rename_and_backup_removal(self, tmp_path, monkeypatch):
        dirs = {"roster": tmp_path / "speakers", "classified": tmp_path / "classified"}
        for d in dirs.values():
            d.mkdir()
        monkeypatch.setattr(repair.s3_helpers, "use_s3", lambda: False)
        monkeypatch.setattr(repair, "_LOCAL_DIRS", dirs)
        monkeypatch.setattr(repair, "_LOCAL_ROOT", tmp_path / "backup")
        misnamed = json.dumps({"video_id": "TUieQkV", "source_video": SV_CUT})
        (dirs["roster"] / "TUieQkV.json").write_text(misnamed)
        clash = json.dumps({"video_id": "GWMcN", "source_video": SV_SINGLE, "note": "old"})
        (dirs["roster"] / "GWMcN.json").write_text(clash)
        (dirs["roster"] / "GWMcNLuXnRA.json").write_text(json.dumps({"video_id": "GWMcNLuXnRA", "note": "new"}))
        files = repair._Files()
        files.write_backup("classified/a.classified.json", b"old")
        files.write_backup("roster/TUieQkV.json", b"old")
        files.write_backup("created_rosters.json", b"[]")
        files.write_backup("_FILES_APPLIED", b"x")

        preview = repair._cleanup_files(files, apply=False)
        assert preview == {"rosters_renamed": 1, "roster_name_conflicts": ["GWMcN.json"], "backup_copies_deleted": 3}
        assert (dirs["roster"] / "TUieQkV.json").exists()  # preview changes nothing

        repair._cleanup_files(files, apply=True)
        assert (dirs["roster"] / "TUieQkV_9Xc.json").read_text() == misnamed
        assert not (dirs["roster"] / "TUieQkV.json").exists()
        assert (dirs["roster"] / "GWMcN.json").exists()  # conflicts are left for a human
        assert files.backup_names("classified") == [] and files.backup_names("roster") == []
        assert files.read_backup("_FILES_APPLIED") == b"x"
