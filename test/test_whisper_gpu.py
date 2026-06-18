from faster_whisper import WhisperModel

print("Loading model on CUDA...", flush=True)

model = WhisperModel(
    "tiny",
    device="cuda",
    compute_type="int8_float16",
)

print("GPU model loaded successfully.", flush=True)
