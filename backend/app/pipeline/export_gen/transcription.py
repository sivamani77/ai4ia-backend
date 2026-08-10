from __future__ import annotations
from pathlib import Path
from typing import Any

DEFAULT_SUBTITLE_CHUNK_WORDS = 10

_WHISPER_MODEL_NAME = "small"
_whisper_model_singleton = None


def _get_whisper_model():
    """Lazily load the same faster-whisper 'small' model pre-downloaded into
    the Docker image (see Dockerfile) — reused here for real per-word
    timestamps on speaker_clip captions instead of an even-pace estimate."""
    global _whisper_model_singleton
    if _whisper_model_singleton is None:
        from faster_whisper import WhisperModel
        _whisper_model_singleton = WhisperModel(_WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
    return _whisper_model_singleton


def _transcribe_words(media_path: Path) -> list[dict[str, Any]]:
    """Transcribe a rendered segment file and return real per-word timestamps
    (segment-local time, i.e. relative to that clip file's own start)."""
    model = _get_whisper_model()
    segments_iter, _info = model.transcribe(
        str(media_path),
        task="transcribe",
        vad_filter=True,
        beam_size=5,
        word_timestamps=True,
    )
    words: list[dict[str, Any]] = []
    for segment in segments_iter:
        for w in (segment.words or []):
            text = (w.word or "").strip()
            if not text:
                continue
            words.append({"word": text, "start": float(w.start), "end": float(w.end)})
    return words


def _real_timed_entries_from_words(
    words: list[dict[str, Any]], max_words: int = DEFAULT_SUBTITLE_CHUNK_WORDS
) -> list[dict[str, Any]]:
    """Group transcribed words into caption chunks using each word's own real
    start/end (not an evenly-divided estimate)."""
    entries: list[dict[str, Any]] = []
    for i in range(0, len(words), max_words):
        group = words[i:i + max_words]
        if not group:
            continue
        entries.append(
            {
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "text": " ".join(w["word"] for w in group),
            }
        )
    return entries
