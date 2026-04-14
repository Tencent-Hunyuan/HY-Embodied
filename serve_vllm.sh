#!/usr/bin/env bash
# serve_vllm.sh — start an OpenAI-compatible HTTP server for HY-Embodied.
#
# Requirements:
#   pip install "vllm>=0.7"
#   pip install git+https://github.com/huggingface/transformers@9293856c419762ebf98fbe2bd9440f9ce7069f1a
#
# Usage:
#   bash serve_vllm.sh                     # default settings
#   MODEL_PATH=/path/to/local bash serve_vllm.sh
#   PORT=8080 bash serve_vllm.sh
#
# Query the running server (example):
#   curl http://localhost:${PORT:-8000}/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#       "model": "tencent/HY-Embodied-0.5",
#       "messages": [
#         {"role": "user", "content": [
#           {"type": "image_url", "image_url": {"url": "https://..."}},
#           {"type": "text", "text": "Describe the image."}
#         ]}
#       ]
#     }'

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-tencent/HY-Embodied-0.5}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

exec vllm serve "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --trust-remote-code \
    --model-impl transformers \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --served-model-name "tencent/HY-Embodied-0.5"
