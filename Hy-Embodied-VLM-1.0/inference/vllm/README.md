# Hy-Embodied-VLM-1.0 · vLLM plugin & inference

High-throughput serving path. Ships a vLLM plugin (`hy-embodied-vllm-plugin`)
that registers Hy-Embodied-VLM-1.0's model class and its reasoning / tool
parsers into `vllm` — no fork of the `vllm` library is required.

## Install

```bash
# 1) Install vLLM at the version we validated. `--torch-backend auto`
#    matches the CUDA build to your driver, and pulls a compatible torch,
#    torchvision, and transformers along with vllm — one shot, no fork.
uv pip install vllm==0.14.1 --torch-backend auto

# 2) Install this plugin (from source, editable). It auto-loads at
#    `vllm serve` startup via the `vllm.general_plugins` entry-point.
uv pip install -e .
```

If you don't have `uv` yet, install it once with
`curl -LsSf https://astral.sh/uv/install.sh | sh`. Plain `pip` also works,
but you'll need to install `torch==2.9.1 torchvision==0.24.1` from
`https://download.pytorch.org/whl/cu<XYZ>` first so the CUDA wheel matches
your driver — see the root README for details.

Verify the plugin is registered:

```bash
python -c "
from vllm import ModelRegistry
assert 'HYV3VLForConditionalGeneration' in ModelRegistry.get_supported_archs()
print('OK: HYV3VLForConditionalGeneration registered')
"
```

## Start the server

```bash
# Uses tencent/Hy-Embodied-VLM-1.0 from HuggingFace by default; will download
# on first run. Set MODEL_PATH to a local directory to skip the download.
MODEL_PATH=./Hy-Embodied-VLM-1.0 bash serve.sh
```

Common environment overrides:

| Var | Default | Notes |
|---|---|---|
| `MODEL_PATH` | `tencent/Hy-Embodied-VLM-1.0` | HuggingFace id or local dir |
| `SERVED_NAME` | `hy_a3b` | Name exposed via `/v1/models` |
| `PORT` | `8080` | HTTP port |
| `TP` | `4` | Tensor parallel size (GPUs per replica) |
| `GPU_MEM_UTIL` | `0.85` | Fraction of GPU RAM to reserve |
| `MAX_MODEL_LEN` | `32768` | Max context length |
| `CHAT_TEMPLATE` | `$MODEL_PATH/chat_template.jinja` | Shipped inside the weights repo |
| `EXTRA_ARGS` | (empty) | Extra `vllm serve` flags |

Readiness check (first load takes ~1–5 min depending on I/O):

```bash
curl -sf http://127.0.0.1:8080/v1/models
```

## Call the server

Python example (openai SDK):

```bash
python example_client.py --port 8080
```

Raw `curl` (skip thinking, direct answer):

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hy_a3b",
    "messages": [{"role": "user", "content": "How to open a fridge?"}],
    "max_tokens": 512,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

## Reasoning mode via `chat_template_kwargs`

Hy-Embodied-VLM-1.0 is a hybrid reasoning model. Pass `enable_thinking`
(a boolean) inside `chat_template_kwargs` to switch modes per request:

| `enable_thinking` | Prompt suffix | Behavior |
|---|---|---|
| `true` (default) | `<think>` | Model emits chain-of-thought, then answer |
| `false` | `<think></think>` | Model answers directly, lower latency |

We deliberately avoid `reasoning_effort` as the kwarg name: vLLM prior to
v0.22 has a top-level `request.reasoning_effort` field that silently
clobbers `chat_template_kwargs["reasoning_effort"]` (fixed by
[vllm-project/vllm#43401](https://github.com/vllm-project/vllm/pull/43401)).
`enable_thinking` (Qwen3 convention) sidesteps that clobber and works
uniformly across all vLLM versions.

## What the plugin does

The plugin's `hy_embodied_vllm:register` function is invoked by vLLM at
startup via the `vllm.general_plugins` entry-point (see `pyproject.toml`).
It calls:

1. `ModelRegistry.register_model("HYV3VLForConditionalGeneration", ...)` —
   lazily binds the model class from `hy_embodied_vllm.hunyuan_v3_vision`.
2. `ModelRegistry.register_model("HYV3ForCausalLM", ...)` —
   the LLM base class.
3. `ToolParserManager.register_module("hy_v3", HYV3ToolParser)` —
   parses the model's XML-style tool-call output.
4. `ReasoningParserManager.register_module("hunyuan_v3", HYV3ReasoningParser)` —
   splits `<think>...</think>` reasoning from the final answer.

Once upstream `vllm` merges Hy-Embodied-VLM-1.0 natively, this plugin can
be uninstalled without any user-side code changes.

## Stop the server

```bash
pkill -9 -f "vllm serve"
```
