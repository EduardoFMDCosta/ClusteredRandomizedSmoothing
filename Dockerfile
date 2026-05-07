# syntax=docker/dockerfile:1.7
#
# Reproducible environment for "Clustered Randomized Smoothing for Stochastic
# Prediction Functions" (NeurIPS submission 34554).
#
# Build:    docker build -t clustered-rs .
# Run:      docker run --rm -it -v "$PWD/experiments/data:/app/experiments/data" clustered-rs
# (mount the experiments/data folder so generated figures and analysis files
# land back on your host.)

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.9.6

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

FROM python:${PYTHON_VERSION}-slim-bookworm

COPY --from=uv-bin /uv /uvx /usr/local/bin/

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgl1 \
        libglib2.0-0 \
        libxrender1 \
        libxext6 \
        libsm6 \
        libxcb1 \
        libxkbcommon0 \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    PYVISTA_OFF_SCREEN=true \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra plots

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra plots

CMD ["bash"]
