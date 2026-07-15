# Hy-Embodied-VLM-1.0 · HuggingFace transformers inference

Reference implementation for running the model with native `transformers`
(no vLLM). Best suited for single-instance, low-latency inference or for
integrating into existing HuggingFace pipelines.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
# Update MODEL_PATH in infer_hf.py or export via env var
python infer_hf.py
```

`MODEL_PATH` defaults to `tencent/Hy-Embodied-VLM-1.0` (downloads on first
run). Set it to a local directory to use a pre-downloaded checkpoint.

## Why `trust_remote_code=True`

The model's `modeling_*.py` files are shipped inside the HuggingFace weights
repo. Passing `trust_remote_code=True` lets `AutoModelForImageTextToText` /
`AutoProcessor` dynamically load them from the repo. No fork of the
`transformers` library is required.

Once upstream `transformers` merges Hy-Embodied-VLM-1.0 support natively,
this flag can be removed and the model will be loaded from the standard
`transformers.models.hy_v3_vl` module — no code changes needed on the user
side.

## Reasoning mode

Hy-Embodied-VLM-1.0 is a hybrid reasoning model. Toggle between chain-of-thought
and direct-answer modes via `enable_thinking` (passed through
`apply_chat_template`):

| `enable_thinking` | Behavior |
|---|---|
| `True` (default) | Model emits `<think>...</think>` block, then final answer. Recommended for complex/spatial reasoning tasks. |
| `False` | Model emits `<think></think>` (empty) and answers directly. Fastest, lower cost. |

We deliberately use `enable_thinking` rather than `reasoning_effort` because
vLLM prior to v0.22 has a top-level `request.reasoning_effort` field that
silently clobbers `chat_template_kwargs["reasoning_effort"]` (fixed by
[vllm-project/vllm#43401](https://github.com/vllm-project/vllm/pull/43401)).
`enable_thinking` (Qwen3 convention) sidesteps that clobber and works
uniformly across all vLLM versions.

## For high-throughput serving

Use vLLM instead — see [`../vllm/`](../vllm/).
