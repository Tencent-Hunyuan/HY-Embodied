"""
HY-Embodied vLLM Inference
==========================

Two execution paths are supported:

  1. **Transformers backend** (default, available with vLLM >= 0.7)
     vLLM's scheduler handles continuous batching and memory management;
     the model forward pass is delegated to the pinned transformers fork
     via ``model_impl="transformers"``.  This gives you multi-request
     batching and an OpenAI-compatible serving API without requiring a
     native vLLM model backend.

  2. **Native vLLM backend** (future work — see NOTE below)
     A native integration requires porting the MoT variable-length flash
     attention to vLLM's paged-attention kernel and registering the model
     with vLLM's ModelRegistry.  Set ``VLLM_NATIVE=1`` to opt in once
     the plugin is available.

Requirements
------------
    pip install "vllm>=0.7"
    # The pinned transformers fork must also be installed:
    pip install git+https://github.com/huggingface/transformers@9293856c419762ebf98fbe2bd9440f9ce7069f1a

NOTE — native vLLM backend
--------------------------
``HunYuanVLMoTForConditionalGeneration`` uses ``flash_attn_varlen_func``
directly inside its Mixture-of-Transformers attention layers.  vLLM's
paged KV-cache mechanism cannot wrap that call transparently, so a
native backend requires:

  1. Replacing ``HunYuanVLMoTAttention`` with vLLM's ``Attention``
     module (backed by PagedAttention / FlashAttention backends).
  2. Implementing ``SupportsMultiModal`` and registering the class with
     ``vllm.model_executor.models.ModelRegistry``.
  3. Writing a vLLM ``MultiModalPlugin`` for image / video pre-processing.

Until then, the transformers backend provides identical output quality
with slightly higher per-token latency.
"""

import os
from pathlib import Path
from typing import Optional

from PIL import Image
from vllm import LLM, SamplingParams
from vllm.inputs import TextPrompt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "tencent/HY-Embodied-0.5")

# Set VLLM_NATIVE=1 once a native vLLM backend is registered.
USE_NATIVE_BACKEND = os.environ.get("VLLM_NATIVE", "0") == "1"

SAMPLING_PARAMS = SamplingParams(
    temperature=0.8,
    max_tokens=1024,
)

THINKING_MODE = False


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

def load_llm() -> LLM:
    """Initialise the vLLM engine."""
    common_kwargs = dict(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        # Limit GPU memory fraction so the vision encoder leaves headroom.
        gpu_memory_utilization=0.90,
        # Disable prefix caching for image-heavy workloads where prompts
        # rarely share a common prefix.
        enable_prefix_caching=False,
    )

    if USE_NATIVE_BACKEND:
        # Native vLLM backend: model must be registered in ModelRegistry.
        return LLM(**common_kwargs)
    else:
        # Transformers backend: vLLM delegates forward() to transformers.
        return LLM(
            **common_kwargs,
            model_impl="transformers",
        )


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _build_prompt(processor, text: str, image_path: Optional[str] = None) -> str:
    """Apply the model's chat template and return the formatted prompt string."""
    content = []
    if image_path is not None:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": text})

    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=THINKING_MODE,
    )


def _make_vllm_input(
    processor,
    text: str,
    image_path: Optional[str] = None,
) -> dict:
    """Build a vLLM input dict, attaching image data when provided."""
    prompt_str = _build_prompt(processor, text, image_path)

    if image_path is None:
        return {"prompt": prompt_str}

    image = Image.open(image_path).convert("RGB")
    return {
        "prompt": prompt_str,
        "multi_modal_data": {"image": image},
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def single_inference(llm: LLM, processor, image_path: str, text_prompt: str) -> str:
    """Run inference on a single image + text prompt."""
    vllm_input = _make_vllm_input(processor, text_prompt, image_path)
    outputs = llm.generate([vllm_input], SAMPLING_PARAMS)
    return outputs[0].outputs[0].text


def batch_inference(llm: LLM, processor, requests: list[dict]) -> list[str]:
    """Run inference on a batch of requests.

    Each request is a dict with keys:
      - ``text``  (str, required)
      - ``image_path`` (str, optional)

    Returns one decoded string per request.
    """
    vllm_inputs = [
        _make_vllm_input(processor, req["text"], req.get("image_path"))
        for req in requests
    ]
    outputs = llm.generate(vllm_inputs, SAMPLING_PARAMS)
    return [out.outputs[0].text for out in outputs]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    from transformers import AutoProcessor

    print(f"Loading processor from {MODEL_PATH} ...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print(f"Initialising vLLM engine (backend: {'native' if USE_NATIVE_BACKEND else 'transformers'}) ...")
    llm = load_llm()

    # ── Single inference ────────────────────────────────────────────────────
    print("\n=== Single Inference ===")
    example_image = "./figures/example.jpg"
    try:
        result = single_inference(
            llm, processor,
            image_path=example_image,
            text_prompt="Describe the image in detail.",
        )
        print("Result:", result)
    except Exception as exc:
        print(f"Single inference failed: {exc}")
        print(f"Make sure {example_image!r} exists or set a valid path.")

    # ── Batch inference ─────────────────────────────────────────────────────
    print("\n=== Batch Inference ===")
    batch_requests = [
        {"text": "Describe the image in detail.", "image_path": example_image},
        {"text": "How do you open a fridge?"},
    ]
    try:
        results = batch_inference(llm, processor, batch_requests)
        for i, text in enumerate(results):
            print(f"\n--- Sample {i} ---")
            print(text)
    except Exception as exc:
        print(f"Batch inference failed: {exc}")


if __name__ == "__main__":
    main()
