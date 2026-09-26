"""Stage 07 must retrieve from every video, not one transcript provenance.

Confirmed incident this guards against: stage_07_storyline.py built its
subprocess command without --corpus-version, so argparse fell back to
07_generate_storyline.py's own "original_v1" default and every production
video job retrieved from 220 of the 309 videos in the collection.

source_transcript_version is per-video provenance, not a corpus generation:
find_transcript() in scripts/04_regenerate_fragments_from_clean_transcripts.py
picks ONE transcript per video (aligned > cleaned > original) and stamps
whichever it used. Each video therefore carries exactly one label and the
label sets are disjoint. Measured on the deployed collection: 220 videos
original_v1, a different 89 cleaned_v2, zero in both, 309 total. Filtering to
either label makes the other set unreachable rather than deprioritized.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.core.settings import settings  # noqa: E402
from app.pipeline.storyline_gen import retrieval  # noqa: E402


def version_filter_for(corpus_version: str) -> set | None:
    """Mirror of the allowed-version branch in activate_corpus, which is not
    separately callable — None means 'admit every label'."""
    if corpus_version == "mixed_cleaned_v2":
        return None
    if corpus_version in {"original_v1", "cleaned_v2"}:
        return {corpus_version}
    return None


class TestCorpusVersionSetting:
    def test_default_admits_every_provenance_label(self):
        assert version_filter_for(settings.storyline_corpus_version) is None

    def test_a_single_label_default_would_hide_videos(self):
        # Guards the reasoning, not just the value: if someone sets this to a
        # concrete label, retrieval silently loses every video stamped with
        # the other one.
        assert version_filter_for("original_v1") == {"original_v1"}
        assert version_filter_for("cleaned_v2") == {"cleaned_v2"}

    def test_aligned_label_is_admitted_by_the_default(self):
        # find_transcript can also emit "cleaned_v2_aligned", which neither
        # single-label filter admits — so those videos would be dropped by
        # BOTH of them.
        allowed = version_filter_for(settings.storyline_corpus_version)
        assert allowed is None or "cleaned_v2_aligned" in allowed


class TestStageSevenPassesTheFlag:
    def test_command_names_the_corpus_version_explicitly(self, monkeypatch):
        """The original defect was an absent flag, so assert on the command."""
        captured: dict = {}

        from app.pipeline import stage_07_storyline

        def fake_run_script(job_id, stage, cmd, **kwargs):
            captured["cmd"] = cmd
            raise RuntimeError("stop after command construction")

        monkeypatch.setattr(stage_07_storyline, "run_script", fake_run_script)
        monkeypatch.setattr(stage_07_storyline, "update_stage", lambda *a, **k: None)
        monkeypatch.setattr(stage_07_storyline, "append_event", lambda *a, **k: None)
        # The stage fetches embeddings and the duration sidecar before it
        # builds the command; neither is what this test is about.
        monkeypatch.setattr(stage_07_storyline, "use_s3", lambda: False)
        monkeypatch.setattr(stage_07_storyline, "_prepare_embeddings", lambda job_id: {})
        monkeypatch.setattr(stage_07_storyline, "_prepare_source_duration_index", lambda job_id: {})

        try:
            stage_07_storyline.run(
                "job-1",
                {"prompt": "anything", "target_duration": 180},
                context=None,
            )
        except Exception:
            pass

        cmd = captured.get("cmd")
        assert cmd is not None, "stage 07 never reached run_script"
        assert "--corpus-version" in cmd, (
            "stage 07 must pass --corpus-version; omitting it silently inherits "
            "the script's default and narrows retrieval to one provenance label"
        )
        assert cmd[cmd.index("--corpus-version") + 1] == settings.storyline_corpus_version


class TestScriptDefaultMatchesTheSetting:
    def test_script_and_stage_cannot_drift(self):
        """A manual script run must see the same corpus a pipeline run does.

        Checked against the source rather than by importing the script, which
        would pull in the whole storyline stack for one argparse default.
        """
        source = (PROJECT_ROOT / "scripts" / "07_generate_storyline.py").read_text()
        assert "default=settings.storyline_corpus_version" in source, (
            "the script's --corpus-version default must come from the setting, "
            "not a hardcoded provenance label"
        )


class TestCacheKeyTracksTheCorpus:
    def test_changing_the_corpus_invalidates_cached_storylines(self, monkeypatch):
        """Entries cached under the narrower corpus must not be served."""
        from app.services import storyline_cache

        request = {"prompt": "p", "target_duration": 180, "style": "documentary"}

        monkeypatch.setattr(settings, "storyline_corpus_version", "original_v1")
        narrow = storyline_cache._params_key(request)

        monkeypatch.setattr(settings, "storyline_corpus_version", "mixed_cleaned_v2")
        wide = storyline_cache._params_key(request)

        assert narrow != wide
