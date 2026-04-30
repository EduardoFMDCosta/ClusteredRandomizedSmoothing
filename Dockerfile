FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# Minimal system packages for scientific Python wheels and plotting backends.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    grep -Ev '^(torch|torch_geometric|torch_scatter)==' /tmp/requirements.txt > /tmp/requirements-nontorch.txt && \
    python -m pip install -r /tmp/requirements-nontorch.txt && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.1.0 && \
    python -m pip install torch_geometric==2.4.0 && \
    python -m pip install torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

COPY . /workspace

RUN python -m pip install -e .

CMD ["bash"]
