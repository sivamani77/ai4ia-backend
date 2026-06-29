"""
End-to-end pipeline timing test.

Runs stages 07 → 07b → 08 → 09 directly as subprocesses and prints
a per-stage timing table at the end.

Usage:
    python3 scripts/test_pipeline_timing.py \
        --prompt "AI ethics and governance in modern society" \
        --duration 180

Options:
    --prompt      Topic prompt (default: "AI ethics and governance")
    --duration    Target video duration in seconds (default: 180)
    --style       Documentary style (default: documentary)
    --from-stage  Skip to this stage using existing outputs: 07b | 08 | 09
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
STORYLINES_DIR = DATA_DIR / "storylines"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
ASSETS_DIR = DATA_DIR / "generated_assets"
EXPORTS_DIR = DATA_DIR / "exports"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

timings: list[tuple[str, float, bool]] = []


def header(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}{'='*65}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'='*65}{RESET}\n", flush=True)


def run_stage(name: str, cmd: list[str], env: dict | None = None) -> tuple[float, bool]:
    header(f"Stage {name}")
    print(f"CMD: {' '.join(cmd)}\n", flush=True)
    t0 = time.time()
    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=merged_env)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = f"{GREEN}✓ OK{RESET}" if ok else f"{RED}✗ FAILED (rc={result.returncode}){RESET}"
    print(f"\n  [{name}] {status} — {elapsed:.1f}s", flush=True)
    timings.append((name, elapsed, ok))
    return elapsed, ok


def newest_file(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def print_summary() -> None:
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  PIPELINE TIMING SUMMARY{RESET}")
    print(f"{BOLD}{'='*65}{RESET}")
    total = 0.0
    for name, elapsed, ok in timings:
        mins, secs = divmod(int(elapsed), 60)
        status = f"{GREEN}OK{RESET}  " if ok else f"{RED}FAIL{RESET}"
        bar_len = min(int(elapsed / 10), 40)
        bar = "█" * bar_len
        print(f"  {status}  {name:<30}  {mins:2d}m {secs:02d}s  {YELLOW}{bar}{RESET}")
        total += elapsed
    print(f"{BOLD}{'='*65}{RESET}")
    total_mins, total_secs = divmod(int(total), 60)
    print(f"  {'TOTAL':<36}  {BOLD}{total_mins:2d}m {total_secs:02d}s{RESET}")
    print(f"{BOLD}{'='*65}{RESET}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="AI ethics and governance in modern society")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--style", default="documentary")
    parser.add_argument(
        "--from-stage",
        choices=["07b", "08", "09"],
        default=None,
        help="Skip to this stage using the most recent existing outputs",
    )
    args = parser.parse_args()

    py = sys.executable
    storyline_path: Path | None = None
    validated_path: Path | None = None
    manifest_path: Path | None = None

    # ── Resolve existing outputs when skipping stages ──────────────────
    if args.from_stage in ("07b", "08", "09"):
        storyline_path = newest_file(STORYLINES_DIR, "storyline_*.json")
        if not storyline_path:
            print(f"{RED}ERROR: no storyline found in {STORYLINES_DIR} to skip to {args.from_stage}{RESET}")
            sys.exit(1)
        print(f"Using storyline: {storyline_path.name}")

    if args.from_stage in ("08", "09"):
        validated_path = newest_file(STORYLINES_DIR, "*_validated.json")
        if not validated_path:
            print(f"{RED}ERROR: no validated storyline found — cannot skip to {args.from_stage}{RESET}")
            sys.exit(1)
        print(f"Using validated: {validated_path.name}")

    if args.from_stage == "09":
        manifest_path = ASSETS_DIR / "latest_asset_manifest.json"
        if not manifest_path.exists():
            print(f"{RED}ERROR: no asset manifest at {manifest_path} — cannot skip to 09{RESET}")
            sys.exit(1)
        print(f"Using manifest:  {manifest_path.name}")

    print(f"\n{BOLD}Prompt  :{RESET} {args.prompt}")
    print(f"{BOLD}Duration:{RESET} {args.duration}s")
    print(f"{BOLD}Style   :{RESET} {args.style}")
    if args.from_stage:
        print(f"{BOLD}Starting from stage {args.from_stage}{RESET}")
    print()

    pipeline_start = time.time()

    # ── Stage 07: Storyline generation ────────────────────────────────
    if args.from_stage is None:
        t0_before = time.time()
        _, ok = run_stage("07_storyline", [
            py, str(SCRIPTS_DIR / "07_generate_storyline.py"),
            "--prompt", args.prompt,
            "--duration", str(args.duration),
            "--style", args.style,
        ])
        if not ok:
            print_summary()
            sys.exit(1)
        # Find the newly created storyline
        storyline_path = newest_file(STORYLINES_DIR, "storyline_*.json")
        if not storyline_path:
            print(f"{RED}ERROR: storyline file not found after stage 07{RESET}")
            print_summary()
            sys.exit(1)
        print(f"  Storyline: {storyline_path}")

    # ── Stage 07b: Validate storyline ─────────────────────────────────
    if args.from_stage not in ("08", "09"):
        stem = storyline_path.stem
        validated_path = STORYLINES_DIR / f"{stem}_validated.json"

        _, ok = run_stage("07b_validate_storyline", [
            py, str(SCRIPTS_DIR / "07b_validate_storyline_with_llm.py"),
            "--storyline", str(storyline_path),
            "--transcripts-dir", str(TRANSCRIPTS_DIR),
            "--fallback-transcripts-dir", str(TRANSCRIPTS_DIR),
            "--output", str(validated_path),
            "--max-iterations", "3",
        ])
        if not ok:
            print_summary()
            sys.exit(1)
        print(f"  Validated: {validated_path}")

    # ── Stage 08: Generate assets (TTS + visuals) ──────────────────────
    if args.from_stage != "09":
        storyline_for_08 = validated_path if (validated_path and validated_path.exists()) else storyline_path

        _, ok = run_stage("08_assets", [
            py, str(SCRIPTS_DIR / "08_generate_assets.py"),
            "--storyline", str(storyline_for_08),
        ])
        if not ok:
            print_summary()
            sys.exit(1)
        manifest_path = ASSETS_DIR / "latest_asset_manifest.json"
        if not manifest_path.exists():
            print(f"{RED}ERROR: asset manifest not found after stage 08{RESET}")
            print_summary()
            sys.exit(1)
        print(f"  Manifest: {manifest_path}")

    # ── Stage 09: Export video ─────────────────────────────────────────
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_video = EXPORTS_DIR / f"timing_test_{ts}.mp4"
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    _, ok = run_stage("09_export_video", [
        py, str(SCRIPTS_DIR / "09_export_video.py"),
        "--manifest", str(manifest_path),
        "--output", str(output_video),
    ])
    if not ok:
        print_summary()
        sys.exit(1)

    if output_video.exists():
        size_mb = output_video.stat().st_size / 1_000_000
        print(f"  Output: {output_video}  ({size_mb:.1f} MB)")

    # ── Summary ───────────────────────────────────────────────────────
    print_summary()

    total_elapsed = time.time() - pipeline_start
    total_mins, total_secs = divmod(int(total_elapsed), 60)
    print(f"Wall-clock total: {BOLD}{total_mins}m {total_secs:02d}s{RESET}")


if __name__ == "__main__":
    main()
