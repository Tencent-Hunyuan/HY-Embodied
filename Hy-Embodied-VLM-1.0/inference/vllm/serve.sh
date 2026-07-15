#!/bin/bash
# serve.sh — start a vLLM OpenAI-compatible server for Hy-Embodied-VLM-1.0.
#
# Usage:
#   MODEL_PATH=./Hy-Embodied-VLM-1.0 bash serve.sh
#
# Environment variables (all optional):
#   MODEL_PATH        Path to a downloaded checkpoint dir, or a HuggingFace
#                     hub id (default: tencent/Hy-Embodied-VLM-1.0).
#   SERVED_NAME       Model name exposed by the OpenAI-compat server
#                     (default: hy_a3b — pick anything you like).
#   PORT              HTTP port (default: 8080).
#   HOST              Bind address (default: 0.0.0.0).
#   TP                Tensor parallel size (default: 4). Set to number of
#                     GPUs you want to use per replica.
#   GPU_MEM_UTIL      Fraction of GPU memory to reserve (default: 0.85).
#   MAX_MODEL_LEN     Maximum context length (default: 32768). The model was
#                     trained up to 32k tokens.
#   CHAT_TEMPLATE     Path to chat_template.jinja. Defaults to the one
#                     shipped inside the weights repo.
#   EXTRA_ARGS        Any additional flags to pass to `vllm serve`.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-tencent/Hy-Embodied-VLM-1.0}"
SERVED_NAME="${SERVED_NAME:-hy_a3b}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
TP="${TP:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${MODEL_PATH}/chat_template.jinja}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "═════════════════════════════════════════════════════════════════"
echo " Hy-Embodied-VLM-1.0 · vLLM serve"
echo "   MODEL_PATH:     $MODEL_PATH"
echo "   SERVED_NAME:    $SERVED_NAME"
echo "   PORT/HOST:      $PORT / $HOST"
echo "   TP:             $TP"
echo "   GPU_MEM_UTIL:   $GPU_MEM_UTIL"
echo "   MAX_MODEL_LEN:  $MAX_MODEL_LEN"
echo "   CHAT_TEMPLATE:  $CHAT_TEMPLATE"
echo "═════════════════════════════════════════════════════════════════"

# `--reasoning-parser hunyuan_v3` and `--tool-call-parser hy_v3` are provided
# by the `hy-embodied-vllm-plugin` package — make sure you `pip install -e .`
# from this directory (or `pip install hy-embodied-vllm-plugin`) before running.
vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --trust-remote-code \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --limit-mm-per-prompt '{"image":128,"video":0}' \
    --chat-template "$CHAT_TEMPLATE" \
    --reasoning-parser hunyuan_v3 \
    --tool-call-parser hy_v3 \
    --enforce-eager \
    $EXTRA_ARGS
