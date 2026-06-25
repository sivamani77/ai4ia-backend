"""
Phase 4R Orchestrator: Audio-Aware Reingestion Pipeline

Runs the full reprocessing pipeline in order:
    1. Audit audio quality           (02b_audit_audio_quality.py)
    2. Clean flagged audio           (02c_clean_audio.py)
    3. Retranscribe cleaned audio    (03_retranscribe_cleaned.py)
    4. Compare old vs new            (03b_compare_transcripts.py)
    5. Regenerate fragments          (04_regenerate_fragments_from_clean_transcripts.py)
    6. Regenerate embeddings         (06_generate_embeddings.py)

Each step is a subprocess call so errors in one step don't kill the whole run.

Usage:
    # Full run (all steps):
    .venv/bin/python scripts/reprocess_audio_transcripts.py --audit --clean-audio --transcribe --align --regenerate-fragments --regenerate-embeddings

    # Audit only:
    .venv/bin/python scripts/reprocess_audio_transcripts.py --audit

    # Audit + clean + transcribe for flagged only:
    .venv/bin/python scripts/reprocess_audio_transcripts.py --audit --clean-audio --transcribe --only-flagged

    # Single video end-to-end:
    .venv/bin/python scripts/reprocess_audio_transcripts.py --audit --clean-audio --transcribe --regenerate-fragments --regenerate-embeddings --video-id 0026

    # Dry run:
    .venv/bin/python scripts/reprocess_audio_transcripts.py --audit --clean-audio --dry-run

    # Limit processing:
    .venv/bin/python scripts/reprocess_audio_transcripts.py --audit --clean-audio --transcribe --limit 10
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PYTHON = sys.executable


def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_step(
    label: str,
    script: str,
    extra_args: list[str],
    dry_run: bool,
) -> bool:
    cmd = [PYTHON, str(SCRIPTS_DIR / script)] + extra_args
    log(f"{'DRY-RUN' if dry_run else 'RUNNING'}: {label}")
    log(f"  CMD: {' '.join(str(c) for c in cmd)}")

    if dry_run:
        return True

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        log(f"STEP FAILED: {label} (exit code {result.returncode})")
        return False

    log(f"STEP DONE: {label}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4R: Audio-aware reingestion orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Step toggles
    parser.add_argument("--audit", action="store_true", help="Run audio quality audit (02b)")
    parser.add_argument("--clean-audio", action="store_true", help="Run audio cleaning (02c)")
    parser.add_argument("--transcribe", action="store_true", help="Run retranscription (03_retranscribe)")
    parser.add_argument("--align", action="store_true",
                        help="Enable WhisperX alignment during retranscription")
    parser.add_argument("--compare", action="store_true",
                        help="Run transcript comparison report (03b)")
    parser.add_argument("--regenerate-fragments", action="store_true",
                        help="Run fragment regeneration (04_regenerate)")
    parser.add_argument("--regenerate-embeddings", action="store_true",
                        help="Run embedding regeneration (06) from cleaned fragments")

    # Filters
    parser.add_argument("--only-flagged", action="store_true",
                        help="Only process audio flagged in quality report")
    parser.add_argument("--video-id", help="Restrict all steps to this video ID substring")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of files per step")
    parser.add_argument("--force", action="store_true",
                        help="Re-run steps even if output exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")

    # Audio cleaning profile
    parser.add_argument("--profile", default="voice_norm_denoise",
                        choices=["voice_norm_denoise", "loudnorm_only"],
                        help="FFmpeg filter profile for audio cleaning")

    # Whisper model
    parser.add_argument("--model", default="medium",
                        help="Faster-Whisper model size (default: medium)")

    args = parser.parse_args()

    any_step = (
        args.audit or args.clean_audio or args.transcribe or
        args.compare or args.regenerate_fragments or args.regenerate_embeddings
    )
    if not any_step:
        parser.print_help()
        print("\nNo steps selected. Use --audit, --clean-audio, --transcribe, etc.")
        return

    log("=" * 60)
    log("Phase 4R: Audio-Aware Reingestion Pipeline")
    log(f"  Only flagged : {args.only_flagged}")
    log(f"  Video ID     : {args.video_id or '(all)'}")
    log(f"  Limit        : {args.limit or '(none)'}")
    log(f"  Force        : {args.force}")
    log(f"  Dry run      : {args.dry_run}")
    log("=" * 60)

    # Build shared filter args
    filter_args: list[str] = []
    if args.only_flagged:
        filter_args.append("--only-flagged")
    if args.video_id:
        filter_args.extend(["--video-id", args.video_id])
    if args.limit:
        filter_args.extend(["--limit", str(args.limit)])
    if args.force:
        filter_args.append("--force")

    results: list[tuple[str, bool]] = []

    # Step 1: Audit
    if args.audit:
        ok = run_step(
            label="02b: Audio Quality Audit",
            script="02b_audit_audio_quality.py",
            extra_args=(["--video-id", args.video_id] if args.video_id else [])
                       + (["--limit", str(args.limit)] if args.limit else []),
            dry_run=args.dry_run,
        )
        results.append(("02b: Audit", ok))

    # Step 2: Clean audio
    if args.clean_audio:
        clean_args = list(filter_args) + ["--profile", args.profile]
        if args.dry_run:
            clean_args.append("--dry-run")
        ok = run_step(
            label="02c: Clean Audio",
            script="02c_clean_audio.py",
            extra_args=clean_args,
            dry_run=False,  # dry_run is passed through clean_args
        )
        results.append(("02c: Clean Audio", ok))

    # Step 3: Retranscribe
    if args.transcribe:
        trans_args = list(filter_args) + ["--model", args.model]
        if not args.align:
            trans_args.append("--no-align")
        ok = run_step(
            label="03_retranscribe: Retranscribe Cleaned Audio",
            script="03_retranscribe_cleaned.py",
            extra_args=trans_args,
            dry_run=args.dry_run,
        )
        results.append(("03: Retranscribe", ok))

    # Step 4: Compare
    if args.compare:
        cmp_args = []
        if args.video_id:
            cmp_args.extend(["--video-id", args.video_id])
        if args.limit:
            cmp_args.extend(["--limit", str(args.limit)])
        ok = run_step(
            label="03b: Compare Transcripts",
            script="03b_compare_transcripts.py",
            extra_args=cmp_args,
            dry_run=args.dry_run,
        )
        results.append(("03b: Compare", ok))

    # Step 5: Regenerate fragments
    if args.regenerate_fragments:
        frag_args = []
        if args.video_id:
            frag_args.extend(["--video-id", args.video_id])
        if args.limit:
            frag_args.extend(["--limit", str(args.limit)])
        if args.force:
            frag_args.append("--force")
        ok = run_step(
            label="04_regenerate: Fragments from Cleaned Transcripts",
            script="04_regenerate_fragments_from_clean_transcripts.py",
            extra_args=frag_args,
            dry_run=args.dry_run,
        )
        results.append(("04: Fragments", ok))

    # Step 6: Regenerate embeddings
    if args.regenerate_embeddings:
        cleaned_fragments_dir = PROJECT_ROOT / "data" / "fragments_cleaned"
        cleaned_embeddings_dir = PROJECT_ROOT / "data" / "embeddings_cleaned"
        emb_args = [
            "--fragments", str(cleaned_fragments_dir),
            "--output", str(cleaned_embeddings_dir),
        ]
        ok = run_step(
            label="06: Generate Embeddings from Cleaned Fragments",
            script="06_generate_embeddings.py",
            extra_args=emb_args,
            dry_run=args.dry_run,
        )
        results.append(("06: Embeddings", ok))

    # Final summary
    log("")
    log("=" * 60)
    log("Pipeline Summary")
    log("=" * 60)
    all_ok = True
    for label, ok in results:
        status = "OK" if ok else "FAILED"
        log(f"  {status:6s}  {label}")
        if not ok:
            all_ok = False

    log("")
    if all_ok:
        log("All steps completed successfully.")
    else:
        log("Some steps failed. Review output above.")

    if not args.dry_run:
        report_dir = PROJECT_ROOT / "data" / "reports"
        log("")
        log("Reports available:")
        for report in sorted(report_dir.glob("*.csv")):
            log(f"  {report}")
        for report in sorted(report_dir.glob("*.json")):
            log(f"  {report}")


if __name__ == "__main__":
    main()
