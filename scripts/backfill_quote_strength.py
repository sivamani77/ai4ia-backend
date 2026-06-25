"""
Backfill script: re-score quote_strength and clip_usage_candidates on all existing
classified fragments using the improved estimate_quote_strength() from 05_classify_fragments.py.

Does NOT re-run classification, embedding, or any LLM step.
Updates in-place: data/fragments_classified/*.classified.json
Also updates: data/embeddings/classified_fragments_embeddings.jsonl (quote_strength field only)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("classify_fragments", ROOT / "scripts" / "05_classify_fragments.py")
_cf = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cf)
estimate_quote_strength = _cf.estimate_quote_strength
determine_clip_usage = _cf.determine_clip_usage
estimate_narration_support = _cf.estimate_narration_support

CLASSIFIED_DIR = ROOT / "data" / "fragments_classified"
EMBEDDINGS_JSONL = ROOT / "data" / "embeddings" / "classified_fragments_embeddings.jsonl"


def backfill_classified():
    files = sorted(CLASSIFIED_DIR.glob("*.classified.json"))
    total, updated, soundbite_before, soundbite_after = 0, 0, 0, 0

    for fpath in files:
        data = json.loads(fpath.read_text(encoding="utf-8"))
        frags = data.get("fragments", [])
        changed = False

        for frag in frags:
            text = frag.get("text", "")
            duration = float(frag.get("duration") or 0)
            word_count = int(frag.get("word_count") or 0)
            clarity = float(frag.get("clarity_score") or 0)

            old_qs = frag.get("quote_strength", 0)
            old_usage = frag.get("clip_usage_candidates") or []

            if "speaker_soundbite" in old_usage:
                soundbite_before += 1

            chapter_tags = frag.get("chapter_tags") or []
            narration_score = estimate_narration_support(text, chapter_tags)
            new_qs = estimate_quote_strength(text)
            new_usage = determine_clip_usage(
                text=text,
                quote_strength=new_qs,
                clarity_score=clarity,
                narration_support_score=narration_score,
            )

            if "speaker_soundbite" in new_usage:
                soundbite_after += 1

            if new_qs != old_qs or set(new_usage) != set(old_usage):
                frag["quote_strength"] = new_qs
                frag["clip_usage_candidates"] = new_usage
                changed = True
                updated += 1

            total += 1

        if changed:
            fpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return total, updated, soundbite_before, soundbite_after


def backfill_embeddings_jsonl():
    if not EMBEDDINGS_JSONL.exists():
        print("  embeddings JSONL not found, skipping")
        return 0

    # Build lookup: fragment_id -> new quote_strength
    qs_lookup = {}
    usage_lookup = {}
    for fpath in CLASSIFIED_DIR.glob("*.classified.json"):
        data = json.loads(fpath.read_text(encoding="utf-8"))
        for frag in data.get("fragments", []):
            fid = frag.get("fragment_id")
            if fid:
                qs_lookup[fid] = frag.get("quote_strength")
                usage_lookup[fid] = frag.get("clip_usage_candidates")

    rows = EMBEDDINGS_JSONL.read_text(encoding="utf-8").splitlines()
    updated = 0
    out_lines = []
    for line in rows:
        if not line.strip():
            out_lines.append(line)
            continue
        row = json.loads(line)
        fid = row.get("fragment_id")
        if fid and fid in qs_lookup:
            if row.get("quote_strength") != qs_lookup[fid]:
                row["quote_strength"] = qs_lookup[fid]
                row["clip_usage_candidates"] = usage_lookup.get(fid, row.get("clip_usage_candidates"))
                updated += 1
        out_lines.append(json.dumps(row, ensure_ascii=False))

    EMBEDDINGS_JSONL.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return updated


if __name__ == "__main__":
    print("Backfilling quote_strength and clip_usage_candidates...")
    total, updated, sb_before, sb_after = backfill_classified()
    print(f"  Classified fragments: {total} total, {updated} updated")
    print(f"  speaker_soundbite eligible: {sb_before} → {sb_after} (+{sb_after - sb_before})")

    print("Updating embeddings JSONL...")
    emb_updated = backfill_embeddings_jsonl()
    print(f"  Embeddings JSONL rows updated: {emb_updated}")

    print("Done.")
