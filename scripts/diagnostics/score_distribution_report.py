#!/usr/bin/env python3
"""Phase 0 diagnostic — score distributions across the classified fragment corpus.

Read-only: computes histograms and gate pass-rates from data/fragments_classified/
without touching any pipeline output. Answers, before any scoring-formula change:
how many fragments actually clear the thresholds the storyline pipeline gates on.

Usage:
    python scripts/diagnostics/score_distribution_report.py
"""
import json
import statistics
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIED_DIR = PROJECT_ROOT / "data" / "fragments_classified"
OUT_PATH = PROJECT_ROOT / "data" / "reports" / "score_distribution_report.json"

# Thresholds actually used downstream (scripts/07_generate_storyline.py):
# speaker_soundbite_allowed / speaker_clip_allowed require clarity_score >= 0.75 or 0.80
# depending on relevance role; quote_strength >= 0.50/0.55.
CLARITY_GATES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
QUOTE_GATES = [0.40, 0.45, 0.50, 0.55, 0.60]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def histogram(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "count": len(values),
        "min": round(min(values), 4),
        "p10": round(percentile(values, 0.10), 4),
        "p25": round(percentile(values, 0.25), 4),
        "median": round(percentile(values, 0.50), 4),
        "mean": round(statistics.mean(values), 4),
        "p75": round(percentile(values, 0.75), 4),
        "p90": round(percentile(values, 0.90), 4),
        "max": round(max(values), 4),
    }


def gate_pass_rates(values: list[float], gates: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {}
    return {
        f">={g}": round(sum(1 for v in values if v >= g) / n, 4)
        for g in gates
    }


def main() -> None:
    files = sorted(CLASSIFIED_DIR.glob("*.classified.json"))
    print(f"Fragment files: {len(files)}")

    quote_strength = []
    clarity_score = []
    narration_support_score = []
    classification_confidence = []

    soundbite_tagged_clarity = []  # clarity_score for fragments already tagged speaker_soundbite
    soundbite_tagged_quote = []

    speakers_per_video = Counter()
    fragments_per_speaker = Counter()
    total_fragments = 0
    fragments_with_speaker = 0

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        video_speakers = set()

        for frag in payload.get("fragments", []):
            total_fragments += 1

            qs = frag.get("quote_strength")
            cs = frag.get("clarity_score")
            ns = frag.get("narration_support_score")
            cc = frag.get("classification_confidence")

            if qs is not None:
                quote_strength.append(float(qs))
            if cs is not None:
                clarity_score.append(float(cs))
            if ns is not None:
                narration_support_score.append(float(ns))
            if cc is not None:
                classification_confidence.append(float(cc))

            usage = frag.get("clip_usage_candidates") or []
            if "speaker_soundbite" in usage:
                if cs is not None:
                    soundbite_tagged_clarity.append(float(cs))
                if qs is not None:
                    soundbite_tagged_quote.append(float(qs))

            speaker = frag.get("speaker")
            if speaker:
                fragments_with_speaker += 1
                fragments_per_speaker[speaker] += 1
                video_speakers.add(speaker)

        speakers_per_video[len(video_speakers)] += 1

    # Downstream gate simulation: fraction of speaker_soundbite-tagged fragments
    # that would actually pass speaker_soundbite_allowed's clarity_score >= 0.75/0.80
    # (scripts/07_generate_storyline.py:1118-1122), independent of role/quote_strength.
    n_soundbite = len(soundbite_tagged_clarity)
    clears_075 = sum(1 for c in soundbite_tagged_clarity if c >= 0.75)
    clears_080 = sum(1 for c in soundbite_tagged_clarity if c >= 0.80)

    report = {
        "total_fragment_files": len(files),
        "total_fragments": total_fragments,
        "fragments_with_speaker": fragments_with_speaker,
        "distributions": {
            "quote_strength": histogram(quote_strength),
            "clarity_score": histogram(clarity_score),
            "narration_support_score": histogram(narration_support_score),
            "classification_confidence": histogram(classification_confidence),
        },
        "gate_pass_rates": {
            "clarity_score": gate_pass_rates(clarity_score, CLARITY_GATES),
            "quote_strength": gate_pass_rates(quote_strength, QUOTE_GATES),
        },
        "speaker_soundbite_tagged_fragments": {
            "count": n_soundbite,
            "clarity_score_distribution": histogram(soundbite_tagged_clarity),
            "quote_strength_distribution": histogram(soundbite_tagged_quote),
            "pass_clarity_gate_0.75": clears_075,
            "pass_clarity_gate_0.75_pct": round(clears_075 / n_soundbite, 4) if n_soundbite else 0.0,
            "pass_clarity_gate_0.80": clears_080,
            "pass_clarity_gate_0.80_pct": round(clears_080 / n_soundbite, 4) if n_soundbite else 0.0,
        },
        "speakers_per_video_histogram": dict(sorted(speakers_per_video.items())),
        "top_20_speakers_by_fragment_count": fragments_per_speaker.most_common(20),
        "distinct_speakers": len(fragments_per_speaker),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nTotal fragments: {total_fragments}")
    print(f"Fragments with speaker: {fragments_with_speaker}")

    print("\n--- clarity_score distribution ---")
    print(json.dumps(report["distributions"]["clarity_score"], indent=2))

    print("\n--- clarity_score gate pass rates (all fragments) ---")
    print(json.dumps(report["gate_pass_rates"]["clarity_score"], indent=2))

    print("\n--- speaker_soundbite-tagged fragments: clarity gate reality check ---")
    print(f"Tagged speaker_soundbite: {n_soundbite}")
    print(f"  of those, clarity_score >= 0.75: {clears_075} ({report['speaker_soundbite_tagged_fragments']['pass_clarity_gate_0.75_pct']:.1%})")
    print(f"  of those, clarity_score >= 0.80: {clears_080} ({report['speaker_soundbite_tagged_fragments']['pass_clarity_gate_0.80_pct']:.1%})")

    print("\n--- quote_strength distribution ---")
    print(json.dumps(report["distributions"]["quote_strength"], indent=2))

    print("\n--- speakers per video (0 = no speaker attributed) ---")
    print(json.dumps(report["speakers_per_video_histogram"], indent=2))

    print(f"\nDistinct speakers: {report['distinct_speakers']}")
    print(f"Report written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
