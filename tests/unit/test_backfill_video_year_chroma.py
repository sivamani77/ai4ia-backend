"""The video_year backfill must repair metadata without destroying it.

Confirmed incident this guards against: every fragment in the collection
carried video_year="" and video_upload_date="" (5112 of 5112), because these
rows were produced by 04_regenerate_fragments_from_clean_transcripts.py,
which never stamped the field. Two features were silently inert as a result —
choose_video_mode's chronological_relay branch needs >= 2 distinct non-null
years and could never fire, and the per-year cap in
select_balanced_candidates bucketed every candidate under one key.

The risk in the repair itself is Chroma's update(), which REPLACES a
document's metadata rather than merging it: writing only the two changed
keys would drop source_video, speaker, primary_theme and everything else
retrieval reads.
"""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location(
    "backfill_video_year_chroma",
    PROJECT_ROOT / "scripts" / "backfill_video_year_chroma.py",
)
backfill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill)


# A stem whose YouTube id contains an underscore — 42 of the 309 videos look
# like this, and split("_") truncates the id so they go unmatched.
UNDERSCORE_STEM = "0005_qifoTD_3vNc_Latin America Event"
PLAIN_STEM = "0001_yqqiJmWKZ9I_Chile Event"

UPLOAD_DATES = {"qifoTD_3vNc": "20250921", "yqqiJmWKZ9I": "20260112"}


def record(fragment_id: str, source_video: str, **extra) -> tuple[str, dict]:
    metadata = {
        "source_video": source_video,
        "speaker": "Ana Ruiz",
        "primary_theme": "AI Ethics, Governance & Policy",
        "primary_chapter": "Ch 3: AI Policy, Regulation & Legislation",
        "text": "a transcript line",
        "video_year": "",
        "video_upload_date": "",
    }
    metadata.update(extra)
    return fragment_id, metadata


class TestYearOf:
    def test_reads_the_year_from_a_yyyymmdd_upload_date(self):
        assert backfill.year_of("20250921") == 2025

    def test_missing_or_short_dates_are_none_not_a_crash(self):
        assert backfill.year_of("") is None
        assert backfill.year_of("202") is None


class TestPlanUpdates:
    def test_fills_in_year_and_upload_date(self):
        ids, metadatas, unmatched, already, years = backfill.plan_updates(
            [record("f1", PLAIN_STEM)], UPLOAD_DATES
        )
        assert ids == ["f1"]
        assert metadatas[0]["video_year"] == 2026
        assert metadatas[0]["video_upload_date"] == "20260112"
        assert unmatched == set()
        assert already == 0
        assert years == {2026: 1}

    def test_preserves_every_other_metadata_field(self):
        """Chroma's update replaces rather than merges — the planned write
        must carry the whole existing dict, not just the changed keys."""
        _, original = record("f1", PLAIN_STEM)
        _, metadatas, _, _, _ = backfill.plan_updates([("f1", original)], UPLOAD_DATES)
        written = metadatas[0]
        for key, value in original.items():
            if key in ("video_year", "video_upload_date"):
                continue
            assert written[key] == value, f"{key} was dropped by the backfill"
        assert set(written) == set(original)

    def test_matches_videos_whose_youtube_id_contains_an_underscore(self):
        ids, metadatas, unmatched, _, _ = backfill.plan_updates(
            [record("f1", UNDERSCORE_STEM)], UPLOAD_DATES
        )
        assert unmatched == set(), "underscore ids must not be truncated by split('_')"
        assert ids == ["f1"]
        assert metadatas[0]["video_year"] == 2025

    def test_is_idempotent(self):
        """A second run must plan no writes, so the script can be re-run."""
        first = [record("f1", PLAIN_STEM)]
        _, metadatas, _, _, _ = backfill.plan_updates(first, UPLOAD_DATES)
        second = [("f1", metadatas[0])]
        ids, _, _, already, _ = backfill.plan_updates(second, UPLOAD_DATES)
        assert ids == []
        assert already == 1

    def test_video_with_no_upload_date_is_reported_not_written(self):
        ids, _, unmatched, _, _ = backfill.plan_updates(
            [record("f1", "0999_UnknownVid1_Some Talk")], UPLOAD_DATES
        )
        assert ids == []
        assert unmatched == {"0999_UnknownVid1_Some Talk"}

    def test_unknown_year_stays_an_empty_string_not_none(self):
        """Chroma metadata cannot hold None; "" is the shape readers expect."""
        ids, metadatas, _, _, _ = backfill.plan_updates(
            [record("f1", PLAIN_STEM)], {"yqqiJmWKZ9I": "bad"}
        )
        assert metadatas[0]["video_year"] == ""
        assert ids == ["f1"]
