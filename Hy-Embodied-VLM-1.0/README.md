# Hy-Embodied-VLM-1.0

This directory contains all assets for the **Hy-Embodied-VLM-1.0** release:
inference code, the vLLM plugin package, and the license. High-level
documentation, model card, and quick-start instructions live in the
[repository root README](../README.md).

## Layout

```
Hy-Embodied-VLM-1.0/
├── inference/
│   ├── transformers/       ← HuggingFace transformers inference (single-instance)
│   │   ├── infer_hf.py     ← demo script (trust_remote_code path)
│   │   ├── requirements.txt
│   │   └── README.md
│   └── vllm/               ← vLLM plugin + serving (high-throughput)
│       ├── hy_embodied_vllm/  ← plugin package (register HYV3VL model + parsers)
│       ├── pyproject.toml
│       ├── serve.sh        ← one-shot server launcher
│       ├── example_client.py  ← OpenAI-SDK client demo
│       ├── README.md
│       └── requirements.txt
├── figures/                ← teaser images referenced from root README
├── LICENSE                 ← Apache-2.0
└── requirements.txt        ← top-level pins (Python / PyTorch)
```

## Quick pointers

- **Inference (HF transformers)** → [`inference/transformers/`](inference/transformers/)
- **Inference (vLLM serving)** → [`inference/vllm/`](inference/vllm/)
- **Full documentation** → [root README](../README.md)

## Two-line installation

```bash
# vLLM plugin (in-tree install; no fork of vllm required)
uv pip install vllm==0.14.1 --torch-backend auto && uv pip install -e inference/vllm/

# HF transformers path (uses trust_remote_code shipped with the weights repo)
pip install -r inference/transformers/requirements.txt
```

Full context, model card, benchmark results, and reasoning-mode usage are
covered in the [root README](../README.md).
