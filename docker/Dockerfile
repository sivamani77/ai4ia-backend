# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.10-slim

# ---------- Stage 1: build Python deps into a venv ----------
FROM python:${PYTHON_VERSION} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# build-essential only exists in this stage — never ships in the final image
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ---------- Stage 2: slim runtime image ----------
FROM python:${PYTHON_VERSION} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/app \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# nodejs: yt-dlp needs a JS runtime >=20 to solve YouTube's JS challenge in
# scripts/01_download_videos.py; without it, downloads silently fail as
# "not available" for every video.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        libgomp1 \
        libpq5 \
        ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get purge -y curl \
    && apt-get autoremove -y

# bring in only the installed packages, not the build toolchain
COPY --from=builder /opt/venv /opt/venv

COPY backend/app ./app
COPY scripts ./scripts
COPY config ./config

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
