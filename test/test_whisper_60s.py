from pathlib import Path
from faster_whisper import WhisperModel

AUDIO_PATH = Path("temp/test_60s.wav")

print("Loading model...", flush=True)

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8",
)

print("Model loaded.", flush=True)
print(f"Transcribing: {AUDIO_PATH}", flush=True)

segments, info = model.transcribe(
    str(AUDIO_PATH),
    beam_size=1,
    vad_filter=True,
)

print(f"Detected language: {info.language}", flush=True)
print(f"Duration: {info.duration}", flush=True)

all_text = []

for segment in segments:
    line = f"[{segment.start:.2f} -> {segment.end:.2f}] {segment.text.strip()}"
    print(line, flush=True)
    all_text.append(line)

output_path = Path("temp/test_60s_transcript.txt")
output_path.write_text("\n".join(all_text), encoding="utf-8")

print(f"Saved transcript to: {output_path}", flush=True)
