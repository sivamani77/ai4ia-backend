# syntax=docker/dockerfile:1.7

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PROJECT_ROOT=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first (improves layer caching)
COPY requirements.txt pyproject.toml ./

# Install Python dependencies using BuildKit cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy application code
COPY backend/app ./app
COPY scripts ./scripts
COPY config ./config

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]