import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASSIFIED_DIR = PROJECT_ROOT / "data" / "fragments_classified"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "youtube" / "ai4ia_channel"
EMBEDDINGS_JSONL = PROJECT_ROOT / "data" / "embeddings" / "classified_fragments_embeddings.jsonl"


def lookup_video_year(source_video: str) -> tuple[int | None, str | None]:
    info_path = RAW_DIR / f"{source_video}.info.json"
    if not info_path.exists():
        return None, None
    try:
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        upload_date = info.get("upload_date")
        if upload_date and len(upload_date) >= 4:
            return int(upload_date[:4]), upload_date
    except Exception:
        pass
    return None, None


def backfill_classified_files() -> dict[str, tuple[int | None, str | None]]:
    fragment_year_map: dict[str, tuple[int | None, str | None]] = {}

    total_updated = 0
    total_null = 0

    for classified_path in sorted(CLASSIFIED_DIR.glob("*.classified.json")):
        with classified_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        changed = False
        for fragment in payload.get("fragments", []):
            source_video = fragment.get("source_video", "")
            if "video_year" not in fragment or "video_upload_date" not in fragment:
                video_year, video_upload_date = lookup_video_year(source_video)
                fragment["video_year"] = video_year
                fragment["video_upload_date"] = video_upload_date
                changed = True
            else:
                video_year = fragment["video_year"]
                video_upload_date = fragment["video_upload_date"]

            fragment_id = fragment.get("fragment_id", "")
            fragment_year_map[fragment_id] = (video_year, video_upload_date)

            if video_year is None:
                total_null += 1
            else:
                total_updated += 1

        if changed:
            classified_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Updated: {classified_path.name}")

    print(f"\nClassified files: {total_updated} fragments with year, {total_null} with null year")
    return fragment_year_map


def backfill_embeddings_jsonl(fragment_year_map: dict[str, tuple[int | None, str | None]]) -> None:
    if not EMBEDDINGS_JSONL.exists():
        print("No embeddings JSONL found — skipping.")
        return

    lines = EMBEDDINGS_JSONL.read_text(encoding="utf-8").splitlines()
    updated_lines = []
    count_updated = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        fragment_id = record.get("fragment_id", "")
        if fragment_id in fragment_year_map:
            video_year, video_upload_date = fragment_year_map[fragment_id]
            record["video_year"] = video_year
            record["video_upload_date"] = video_upload_date
            count_updated += 1
        updated_lines.append(json.dumps(record, ensure_ascii=False))

    EMBEDDINGS_JSONL.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    print(f"Embeddings JSONL: {count_updated}/{len(updated_lines)} rows updated with video_year")


def main() -> None:
    print("Backfilling video_year into classified fragments and embeddings...\n")
    fragment_year_map = backfill_classified_files()
    backfill_embeddings_jsonl(fragment_year_map)
    print("\nDone.")


if __name__ == "__main__":
    main()
