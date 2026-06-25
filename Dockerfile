FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY requirements.txt ./
COPY backend/app/ ./app/
COPY scripts/ ./scripts/

RUN pip install --no-cache-dir -r requirements.txt

ENV PROJECT_ROOT=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
