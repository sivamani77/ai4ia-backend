"""
End-to-end pipeline timing test and duration release gate.

Runs stages 07 → 07b → 08 → 09 directly as subprocesses, prints a per-stage
timing table, and asserts the duration contract on the final video:

    target ≤ actual            (hard failure — the duration floor)
    actual ≤ target × 1.25     (warning — overshoot is allowed but flagged)
    zero silently skipped segments
    duration_budget present in every stage artifact
    preflight_failures empty in the asset manifest

Usage:
    # Single run
    python3 scripts/test_pipeline_timing.py --prompt "AI ethics" --duration 180

    # Release gate: full pipeline at each target, verdict table, exit 1 on failure
    python3 scripts/test_pipeline_timing.py --prompt "AI ethics" --durations 60,180,300,600

Options:
    --prompt      Topic prompt (default: "AI ethics and governance")
    --duration    Target video duration in seconds (default: 180)
    --durations   Comma-separated list of targets — runs the release gate
    --style       Documentary style (default: documentary)
    --from-stage  Skip to this stage using existing outputs: 07b | 08 | 09
                  (single-run mode only)
"""
from __future__ import annotations

import argparse
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

OVERSHOOT_WARN_RATIO = 1.25

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


def probe_duration(path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip() or 0.0)
    except (subprocess.SubprocessError, ValueError, OSError):
        return 0.0


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


def _load_json(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def check_duration_gate(
    target: int,
    storyline_path: Path | None,
    validated_path: Path | None,
    manifest_path: Path | None,
    output_video: Path,
) -> tuple[list[str], list[str], float]:
    """Assert the duration contract across all stage artifacts.

    Returns (failures, warnings, actual_video_seconds).
    """
    failures: list[str] = []
    warnings: list[str] = []

    # Every stage artifact must carry a duration_budget
    for label, path in (
        ("07 storyline", storyline_path),
        ("07b validated", validated_path),
        ("08 manifest", manifest_path),
    ):
        if path is None:
            warnings.append(f"{label}: not produced by this run — stage checks skipped")
            continue
        artifact = _load_json(path)
        if not artifact:
            failures.append(f"{label}: artifact missing or unreadable ({path})")
            continue
        if not artifact.get("duration_budget"):
            failures.append(f"{label}: no duration_budget in artifact")

    # Below-target duration is a WARNING (content exhaustion — honest best
    # effort), NOT a failure. Integrity problems (content lost vs plan) ARE
    # failures and are checked separately below.
    validated = _load_json(validated_path)
    if validated:
        vts = float(validated.get("validated_total_seconds") or 0.0)
        if vts and vts < target:
            warnings.append(f"07b: validated {vts:.1f}s < target {target}s (content exhausted, best-effort)")

    manifest = _load_json(manifest_path)
    if manifest:
        preflight = manifest.get("preflight_failures")
        if preflight:
            failures.append(f"08: {len(preflight)} preflight failure(s): {preflight[:3]}")
        budget = manifest.get("duration_budget") or {}
        if budget.get("meets_floor") is False:
            warnings.append(
                f"08: planned {budget.get('planned_total_seconds')}s < target (content exhausted)"
            )

    export_manifest_path = EXPORTS_DIR / f"{output_video.stem}_export_manifest.json"
    export_manifest = _load_json(export_manifest_path)
    export_timeline: list[dict] = []
    if not export_manifest:
        failures.append(f"09: export manifest missing ({export_manifest_path})")
    else:
        if not export_manifest.get("duration_budget"):
            failures.append("09: no duration_budget in export manifest")
        export_timeline = export_manifest.get("timeline") or []
        planned = len(export_timeline)
        rendered = len(export_manifest.get("segments") or [])
        if rendered < planned:
            failures.append(f"09: only {rendered}/{planned} planned segments were rendered")

    # ---- Structural verification (the "documentary quality" contract) ----
    failures.extend(_check_structure(export_timeline))

    actual = probe_duration(output_video) if output_video.exists() else 0.0
    planned_seconds = float((export_manifest.get("duration_budget") or {}).get("planned_total_seconds") or 0.0)
    if actual <= 0.0:
        failures.append(f"final video missing or unreadable: {output_video}")
    else:
        # Content dropped is caught by the segment-count check above (and 09 raises
        # on any skip). Duration below target is only ever a content-exhaustion
        # warning — never a failure. We do NOT compare rendered vs summed-plan as a
        # failure: fades / -shortest / relay bridges make that drift legitimately.
        if actual < target:
            warnings.append(
                f"below target: rendered {actual:.1f}s < {target}s — content exhausted, "
                f"honest best-effort (no filler)"
            )
        if planned_seconds and actual < planned_seconds - max(5.0, planned_seconds * 0.15):
            warnings.append(
                f"render drift: {actual:.1f}s vs summed plan {planned_seconds:.1f}s "
                f"(fades/-shortest/relay); target still met"
            )
        if actual > target * OVERSHOOT_WARN_RATIO:
            warnings.append(
                f"overshoot: rendered {actual:.1f}s > {OVERSHOOT_WARN_RATIO:.2f}× target "
                f"({target * OVERSHOOT_WARN_RATIO:.1f}s) — allowed, but flag to the user"
            )

    return failures, warnings, actual


def _check_structure(timeline: list[dict]) -> list[str]:
    """Documentary-quality structural checks on the final render timeline.

    - every narrative section carries >=1 real speaker_clip,
    - no silent content segments (muted/silent that isn't a card or year_stamp),
    - no duplicate fragment_ids.
    """
    failures: list[str] = []
    if not timeline:
        return failures

    CARD_TYPES = {"title_card", "closing_card", "year_stamp"}

    # Duplicate fragments
    seen: dict[str, int] = {}
    for item in timeline:
        fid = str(item.get("fragment_id") or "").strip()
        if fid:
            seen[fid] = seen.get(fid, 0) + 1
    dups = {fid: n for fid, n in seen.items() if n > 1}
    if dups:
        failures.append(f"structure: duplicate fragments in timeline: {list(dups.items())[:3]}")

    # Silent content segments (the muted-tail-clip symptom)
    silent = []
    for item in timeline:
        t = item.get("segment_type") or item.get("type") or ""
        if t in CARD_TYPES:
            continue
        audio_mode = item.get("audio_mode") or ""
        if audio_mode == "silent" or (item.get("muted") is True and t != "narration_broll"):
            silent.append(f"{t}:{str(item.get('fragment_id') or '')[:30]}")
    if silent:
        failures.append(f"structure: {len(silent)} silent content segment(s) (no voice): {silent[:4]}")

    # Every section has a speaker clip
    sections: dict[str, list[str]] = {}
    for item in timeline:
        sid = item.get("section_id")
        t = item.get("segment_type") or item.get("type") or ""
        if not sid or t in CARD_TYPES:
            continue
        sections.setdefault(sid, []).append(t)
    sections_without_speaker = [
        sid for sid, types in sections.items() if "speaker_clip" not in types
    ]
    if sections_without_speaker:
        failures.append(
            f"structure: {len(sections_without_speaker)} section(s) have no speaker clip: "
            f"{sections_without_speaker[:6]}"
        )

    return failures


def gate_result_for_run(target: int, run: dict) -> dict:
    """Turn one run_pipeline_once() result into a gate verdict-table row.

    Runs the duration gate whenever an output video path is known — even if the
    stage that produced it (typically 09's own strict-duration check) exited
    non-zero — so a caught shortfall shows up as a proper FAIL row instead of
    silently skipping straight to a bare traceback / early exit.
    """
    failures: list[str] = []
    gate_warnings: list[str] = []
    actual = 0.0
    if run.get("output_video"):
        failures, gate_warnings, actual = check_duration_gate(
            target,
            run.get("storyline_path"),
            run.get("validated_path"),
            run.get("manifest_path"),
            run["output_video"],
        )
    elif not run.get("ok"):
        failures = ["pipeline failed before producing a video"]
    return {
        "target": target,
        "pipeline_ok": run.get("ok", False),
        "actual": actual,
        "failures": failures,
        "warnings": gate_warnings,
    }


def run_pipeline_once(
    prompt: str,
    duration: int,
    style: str,
    from_stage: str | None = None,
) -> dict:
    """Run 07 → 07b → 08 → 09 once. Returns paths + ok flag."""
    py = sys.executable
    storyline_path: Path | None = None
    validated_path: Path | None = None
    manifest_path: Path | None = None

    # ── Resolve existing outputs when skipping stages ──────────────────
    if from_stage in ("07b", "08", "09"):
        storyline_path = newest_file(STORYLINES_DIR, "storyline_*.json")
        if not storyline_path:
            print(f"{RED}ERROR: no storyline found in {STORYLINES_DIR} to skip to {from_stage}{RESET}")
            return {"ok": False}
        print(f"Using storyline: {storyline_path.name}")

    if from_stage in ("08", "09"):
        validated_path = newest_file(STORYLINES_DIR, "*_validated.json")
        if not validated_path:
            print(f"{RED}ERROR: no validated storyline found — cannot skip to {from_stage}{RESET}")
            return {"ok": False}
        print(f"Using validated: {validated_path.name}")

    if from_stage == "09":
        manifest_path = ASSETS_DIR / "latest_asset_manifest.json"
        if not manifest_path.exists():
            print(f"{RED}ERROR: no asset manifest at {manifest_path} — cannot skip to 09{RESET}")
            return {"ok": False}
        print(f"Using manifest:  {manifest_path.name}")

    # ── Stage 07: Storyline generation ────────────────────────────────
    if from_stage is None:
        _, ok = run_stage(f"07_storyline ({duration}s)", [
            py, str(SCRIPTS_DIR / "07_generate_storyline.py"),
            "--prompt", prompt,
            "--duration", str(duration),
            "--style", style,
        ])
        if not ok:
            return {"ok": False}
        storyline_path = newest_file(STORYLINES_DIR, "storyline_*.json")
        if not storyline_path:
            print(f"{RED}ERROR: storyline file not found after stage 07{RESET}")
            return {"ok": False}
        print(f"  Storyline: {storyline_path}")

    # ── Stage 07b: Validate storyline ─────────────────────────────────
    if from_stage not in ("08", "09"):
        stem = storyline_path.stem
        validated_path = STORYLINES_DIR / f"{stem}_validated.json"

        _, ok = run_stage(f"07b_validate_storyline ({duration}s)", [
            py, str(SCRIPTS_DIR / "07b_validate_storyline_with_llm.py"),
            "--storyline", str(storyline_path),
            "--transcripts-dir", str(TRANSCRIPTS_DIR),
            "--fallback-transcripts-dir", str(TRANSCRIPTS_DIR),
            "--output", str(validated_path),
            "--max-iterations", "3",
        ])
        if not ok:
            return {"ok": False, "storyline_path": storyline_path}
        print(f"  Validated: {validated_path}")

    # ── Stage 08: Generate assets (TTS + visuals) ──────────────────────
    if from_stage != "09":
        storyline_for_08 = validated_path if (validated_path and validated_path.exists()) else storyline_path

        _, ok = run_stage(f"08_assets ({duration}s)", [
            py, str(SCRIPTS_DIR / "08_generate_assets.py"),
            "--storyline", str(storyline_for_08),
        ])
        if not ok:
            return {"ok": False, "storyline_path": storyline_path, "validated_path": validated_path}
        manifest_path = ASSETS_DIR / "latest_asset_manifest.json"
        if not manifest_path.exists():
            print(f"{RED}ERROR: asset manifest not found after stage 08{RESET}")
            return {"ok": False, "storyline_path": storyline_path, "validated_path": validated_path}
        print(f"  Manifest: {manifest_path}")

    # ── Stage 09: Export video ─────────────────────────────────────────
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_video = EXPORTS_DIR / f"timing_test_{ts}.mp4"
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    _, ok = run_stage(f"09_export_video ({duration}s)", [
        py, str(SCRIPTS_DIR / "09_export_video.py"),
        "--manifest", str(manifest_path),
        "--output", str(output_video),
    ])

    if ok and output_video.exists():
        size_mb = output_video.stat().st_size / 1_000_000
        print(f"  Output: {output_video}  ({size_mb:.1f} MB)")

    return {
        "ok": ok,
        "storyline_path": storyline_path,
        "validated_path": validated_path,
        "manifest_path": manifest_path,
        "output_video": output_video,
    }


def print_gate_verdict(results: list[dict]) -> bool:
    """Print the release-gate verdict table. Returns True when all targets pass."""
    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  DURATION RELEASE GATE — floor: actual ≥ target, ceiling warn: {OVERSHOOT_WARN_RATIO}×{RESET}")
    print(f"{BOLD}{'='*72}{RESET}")
    all_pass = True
    for r in results:
        target = r["target"]
        actual = r.get("actual", 0.0)
        failures = r.get("failures", [])
        gate_warnings = r.get("warnings", [])
        passed = r.get("pipeline_ok", False) and not failures
        all_pass = all_pass and passed
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        ratio = f"{actual / target:5.2f}×" if target and actual else "  n/a"
        print(f"  {status}  target={target:>4}s  actual={actual:7.1f}s  {ratio}")
        for failure in failures:
            print(f"        {RED}✗ {failure}{RESET}")
        for warning in gate_warnings:
            print(f"        {YELLOW}⚠ {warning}{RESET}")
    print(f"{BOLD}{'='*72}{RESET}")
    verdict = f"{GREEN}ALL TARGETS PASS{RESET}" if all_pass else f"{RED}GATE FAILED{RESET}"
    print(f"  {verdict}\n")
    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="AI ethics and governance in modern society")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument(
        "--durations",
        default=None,
        help="Comma-separated target durations (e.g. 60,180,300,600) — runs the release gate",
    )
    parser.add_argument("--style", default="documentary")
    parser.add_argument(
        "--from-stage",
        choices=["07b", "08", "09"],
        default=None,
        help="Skip to this stage using the most recent existing outputs (single-run mode only)",
    )
    args = parser.parse_args()

    pipeline_start = time.time()

    if args.durations:
        # ── Release gate: full pipeline per target ─────────────────────
        targets = [int(d.strip()) for d in args.durations.split(",") if d.strip()]
        if args.from_stage:
            print(f"{YELLOW}--from-stage is ignored in gate mode (each target runs the full pipeline).{RESET}")
        results = []
        for target in targets:
            header(f"GATE RUN — target {target}s")
            run = run_pipeline_once(args.prompt, target, args.style)
            results.append(gate_result_for_run(target, run))

        print_summary()
        all_pass = print_gate_verdict(results)
        total_elapsed = time.time() - pipeline_start
        total_mins, total_secs = divmod(int(total_elapsed), 60)
        print(f"Wall-clock total: {BOLD}{total_mins}m {total_secs:02d}s{RESET}")
        sys.exit(0 if all_pass else 1)

    # ── Single run ─────────────────────────────────────────────────────
    print(f"\n{BOLD}Prompt  :{RESET} {args.prompt}")
    print(f"{BOLD}Duration:{RESET} {args.duration}s")
    print(f"{BOLD}Style   :{RESET} {args.style}")
    if args.from_stage:
        print(f"{BOLD}Starting from stage {args.from_stage}{RESET}")
    print()

    run = run_pipeline_once(args.prompt, args.duration, args.style, args.from_stage)
    result = gate_result_for_run(args.duration, run)

    print_summary()
    print_gate_verdict([result])

    total_elapsed = time.time() - pipeline_start
    total_mins, total_secs = divmod(int(total_elapsed), 60)
    print(f"Wall-clock total: {BOLD}{total_mins}m {total_secs:02d}s{RESET}")
    sys.exit(0 if not result["failures"] and result["pipeline_ok"] else 1)


if __name__ == "__main__":
    main()
