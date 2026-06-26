FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY requirements.txt ./
COPY backend/app/ ./app/
COPY scripts/ ./scripts/
COPY config/ ./config/

RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Whisper small model into the image so workers never download at runtime.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

# Override the __file__-based project_root detection in settings.py
ENV PROJECT_ROOT=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
