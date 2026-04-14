# ── Stage 1: build flash_attn wheel ──────────────────────────────────────────
# Building flash_attn from source requires CUDA headers and can take 20-40 min.
# We isolate this cost in a separate builder stage so the final image stays lean.
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3-pip \
        build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1

RUN pip install --upgrade pip setuptools wheel

# Install PyTorch before flash_attn — the build introspects torch's CUDA config.
RUN pip install \
        torch==2.8.0 \
        torchvision \
        --index-url https://download.pytorch.org/whl/cu126

# Build flash_attn wheel. MAX_JOBS limits parallelism to avoid OOM during build.
ENV MAX_JOBS=4
RUN pip wheel --no-build-isolation --wheel-dir /wheels flash_attn==2.8.3


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Hugging Face cache — mount a volume here to persist model weights across runs.
    HF_HOME=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1

WORKDIR /app

RUN pip install --upgrade pip

# PyTorch (runtime wheels — no compiler needed here)
RUN pip install \
        torch==2.8.0 \
        torchvision \
        --index-url https://download.pytorch.org/whl/cu126

# Install pre-built flash_attn wheel from the builder stage
COPY --from=builder /wheels /tmp/wheels
RUN pip install /tmp/wheels/flash_attn*.whl && rm -rf /tmp/wheels

# Install remaining dependencies.
# transformers is pinned to the exact commit required by the model.
RUN pip install \
        "accelerate" \
        "Pillow" \
        "numpy" \
        "tqdm" \
        "timm==1.0.21" \
        "git+https://github.com/huggingface/transformers@9293856c419762ebf98fbe2bd9440f9ce7069f1a"

# Copy repo files
COPY inference.py .
COPY figures/  figures/

# Optional: pass a Hugging Face token at build time for gated models.
# Usage: docker build --build-arg HF_TOKEN=hf_...
ARG HF_TOKEN=""
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

# Default entrypoint runs the bundled inference demo.
# Override CMD or mount your own script to customise behaviour.
CMD ["python", "inference.py"]
