#!/usr/bin/env python3
"""Crash-safe transcription driver for low RAM (15GB) + low VRAM (GTX 1650 4GB).

Two failure modes are handled:
  1. GPU OOM on a normal file  -> retry that single file on CPU (isolated subprocess).
  2. SYSTEM RAM OOM on multi-hour panel videos (60-231 min) -> the killer.
     faster-whisper buffers all segments in RAM; long files exceed 15GB and the
     OS OOM-killer reboots the machine. Fix: split files over CHUNK_THRESHOLD_S
     into ~CHUNK_S segments with ffmpeg, transcribe each (CPU, bounded memory),
     and stitch with time offsets into one transcript JSON.

Each completed .wav is deleted immediately to keep disk flat.

Usage:  python scripts/run_transcribe_safe.py
"""
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "data" / "audio"
TRANS = ROOT / "data" / "transcripts"
CHUNKS = AUDIO / "_chunks"
SCRIPT = ROOT / "scripts" / "03_transcribe.py"
PY = sys.executable

CHUNK_THRESHOLD_S = 2700   # files longer than 45 min get chunked
CHUNK_S = 1200             # 20-min chunks keep RAM bounded


def transcript_exists(stem: str) -> bool:
    p = TRANS / f"{stem}.json"
    return p.exists() and p.stat().st_size > 0


def duration_s(wav: Path) -> float:
    try:
        w = wave.open(str(wav), "rb")
        d = w.getnframes() / w.getframerate()
        w.close()
        return d
    except Exception:  # noqa: BLE001
        return 0.0


def run_one(wav_name: str, device: str, compute: str, audio_dir: Path = AUDIO) -> bool:
    env = {**os.environ, "WHISPER_ONLY_FILE": wav_name,
           "WHISPER_DEVICE": device, "WHISPER_COMPUTE": compute}
    # Point 03 at a custom audio dir when transcribing chunks.
    if audio_dir != AUDIO:
        env["WHISPER_AUDIO_DIR"] = str(audio_dir)
    print(f"  -> {device}: {wav_name}", flush=True)
    try:
        subprocess.run([PY, str(SCRIPT)], cwd=str(ROOT), env=env, timeout=3600, check=False)
    except subprocess.TimeoutExpired:
        print(f"     timeout on {device}", flush=True)
    return (TRANS / f"{Path(wav_name).stem}.json").exists()


def transcribe_chunked(wav: Path) -> bool:
    """Split a long file into chunks, transcribe each on CPU, stitch with offsets."""
    CHUNKS.mkdir(exist_ok=True)
    stem = wav.stem
    total = duration_s(wav)
    n = int(total // CHUNK_S) + 1
    print(f"  chunking {stem} ({total/60:.0f} min) into {n} x {CHUNK_S//60}min", flush=True)

    all_segments = []
    for ci in range(n):
        start = ci * CHUNK_S
        cname = f"{stem}__chunk{ci:02d}.wav"
        cpath = CHUNKS / cname
        # extract chunk (re-encode to same 16k mono)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav),
                        "-ss", str(start), "-t", str(CHUNK_S),
                        "-ar", "16000", "-ac", "1", str(cpath)], check=False)
        if not cpath.exists() or cpath.stat().st_size < 1000:
            cpath.unlink(missing_ok=True)
            break
        # transcribe the chunk on GPU (20-min chunk fits in 4GB VRAM); CPU fallback.
        if not run_one(cname, "cuda", "int8_float16", audio_dir=CHUNKS) \
           and not run_one(cname, "cpu", "int8", audio_dir=CHUNKS):
            print(f"     chunk {ci} failed on GPU+CPU", flush=True)
            cpath.unlink(missing_ok=True)
            return False
        cjson = TRANS / f"{Path(cname).stem}.json"
        data = json.loads(cjson.read_text(encoding="utf-8"))
        for seg in data.get("segments", []):
            seg["start"] += start
            seg["end"] += start
            all_segments.append(seg)
        cjson.unlink(missing_ok=True)          # remove chunk transcript
        (TRANS / f"{Path(cname).stem}.txt").unlink(missing_ok=True)
        cpath.unlink(missing_ok=True)          # remove chunk wav

    if not all_segments:
        return False
    payload = {
        "audio_file": wav.name, "model": "base", "device": "cpu-chunked",
        "language": "en", "duration_seconds": total,
        "segment_count": len(all_segments), "segments": all_segments,
        "full_text": "\n".join(s["text"] for s in all_segments).strip(),
    }
    (TRANS / f"{stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    return True


def main() -> int:
    wavs = sorted(AUDIO.glob("*.wav"))
    todo = [w for w in wavs if not transcript_exists(w.stem)]
    print(f"To transcribe: {len(todo)} (of {len(wavs)} wav on disk)", flush=True)

    ok = cpu_fallback = chunked = failed = 0
    for i, wav in enumerate(todo, 1):
        dur = duration_s(wav)
        print(f"[{i}/{len(todo)}] {wav.name} ({dur/60:.0f} min)", flush=True)

        if dur > CHUNK_THRESHOLD_S:
            # Long file -> chunk on CPU (only safe path; whole-file OOMs the machine).
            if transcribe_chunked(wav):
                ok += 1; chunked += 1
            else:
                failed += 1; print(f"     FAILED (chunked): {wav.name}", flush=True)
        elif run_one(wav.name, "cuda", "int8_float16"):
            ok += 1
        elif run_one(wav.name, "cpu", "int8"):
            ok += 1; cpu_fallback += 1; print("     recovered on CPU", flush=True)
        else:
            failed += 1; print(f"     FAILED on GPU+CPU: {wav.name}", flush=True)

        if transcript_exists(wav.stem):
            wav.unlink(missing_ok=True)

    print(f"\nDone. ok={ok} (cpu_fallback={cpu_fallback}, chunked={chunked}) failed={failed}",
          flush=True)
    print(f"Transcripts now: {len(list(TRANS.glob('*.json')))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
