#
# Reproducible environment for "Clustered Randomized Smoothing for Stochastic
# Prediction Functions" (NeurIPS submission 34554).
#
# Build:    docker build -t clustered-rs .
# Run:      docker run --rm -it -v "$PWD/experiments/data:/app/experiments/data" \
#                  -v "$PWD/experiments/uav_3d:/app/experiments/uav_3d" clustered-rs
# (mount the experiments/data folder so generated figures and analysis files
# land back on your host.)

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.9.6

FROM python:${PYTHON_VERSION}-slim-bookworm

ARG UV_VERSION
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
        libxcursor1 \
        libxft2 \
        libosmesa6 \
        libegl1 \
        libgl1-mesa-dri \
        xvfb \
        cm-super \
        dvipng \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "uv==${UV_VERSION}"

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/experiments \
    PYVISTA_OFF_SCREEN=true \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-install-project --extra plots

COPY . /app
RUN uv sync --frozen --extra plots

CMD ["bash"]
