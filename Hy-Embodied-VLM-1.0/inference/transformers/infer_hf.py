# -*- coding: utf-8 -*-
"""
Hy-Embodied-VLM-1.0 — HuggingFace transformers inference demo.

This example uses `trust_remote_code=True` because the model ships its own
`modeling_hy_v3_vl.py` / `modeling_hunyuan_vl.py` (and processors) inside the
HuggingFace weights repo. Passing `trust_remote_code=True` tells `transformers`
to load those files rather than relying on any built-in model type — no fork
of the `transformers` library is required.

Reasoning-mode toggle:
    Hy-Embodied-VLM-1.0 is a hybrid reasoning model. Switch between the
    chain-of-thought "think" mode and the direct "no-think" mode by passing
    `enable_thinking` to `apply_chat_template`:
        enable_thinking=True  (default)  -> emits <think> (model reasons then answers)
        enable_thinking=False            -> emits <think></think> (model answers directly)

    We deliberately avoid the `reasoning_effort` name here: vLLM prior to
    v0.22 has a top-level `request.reasoning_effort` field that silently
    clobbers `chat_template_kwargs["reasoning_effort"]` (fixed by
    vllm-project/vllm#43401). `enable_thinking` (Qwen3 convention) sidesteps
    that clobber and works uniformly across all vLLM versions.

Usage:
    pip install -U "transformers>=4.57" accelerate torch pillow
    python infer_hf.py
"""

import os

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# Set MODEL_PATH to a local checkpoint path or the HuggingFace hub id.
MODEL_PATH = "tencent/Hy-Embodied-VLM-1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default reasoning mode. True = chain-of-thought (recommended for reasoning
# tasks), False = direct answer (lower latency).
ENABLE_THINKING = True

TEMPERATURE = 0.8
MAX_NEW_TOKENS = 1024


def load_model_and_processor():
    """Load model and processor. `trust_remote_code=True` picks up the
    modeling/processor .py files shipped in the weights repo."""
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # The weights repo already ships chat_template.jinja; the processor picks
    # it up automatically. This block is kept for the local-path case where
    # `MODEL_PATH` points at a directory that already has the file.
    chat_template_path = os.path.join(MODEL_PATH, "chat_template.jinja")
    if os.path.exists(chat_template_path):
        processor.chat_template = open(chat_template_path).read()

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(DEVICE).eval()
    return model, processor


def single_inference(model, processor, image_path, text_prompt,
                     enable_thinking=ENABLE_THINKING):
    """Run single inference with an image + text prompt."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            temperature=TEMPERATURE,
            do_sample=TEMPERATURE > 0,
        )

    output_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0]


def batch_inference(model, processor, messages_batch,
                    enable_thinking=ENABLE_THINKING):
    """Run batch inference over a list of message lists (one per sample)."""
    all_inputs = []
    for msgs in messages_batch:
        inp = processor.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        )
        all_inputs.append(inp)

    batch = processor.pad(all_inputs, padding=True, padding_side="left").to(model.device)

    with torch.no_grad():
        batch_generated_ids = model.generate(
            **batch,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            temperature=TEMPERATURE,
            do_sample=TEMPERATURE > 0,
        )

    padded_input_len = batch["input_ids"].shape[1]
    results = []
    for i, _ in enumerate(messages_batch):
        out_ids = batch_generated_ids[i][padded_input_len:]
        results.append(processor.decode(out_ids, skip_special_tokens=True))
    return results


def main():
    print(f"Loading Hy-Embodied-VLM-1.0 from {MODEL_PATH}")
    model, processor = load_model_and_processor()
    print(f"Model loaded on {DEVICE}")

    print("\n=== Single Inference (enable_thinking=True) ===")
    try:
        result = single_inference(
            model, processor,
            image_path="./figures/example.jpg",
            text_prompt="Describe the image in detail.",
            enable_thinking=True,
        )
        print("Result:", result)
    except Exception as e:
        print(f"Single inference failed: {e}")
        print("Note: make sure './figures/example.jpg' exists or provide a valid image path.")

    print("\n=== Single Inference (enable_thinking=False, direct answer) ===")
    try:
        result = single_inference(
            model, processor,
            image_path="./figures/example.jpg",
            text_prompt="What is the main object in the picture?",
            enable_thinking=False,
        )
        print("Result:", result)
    except Exception as e:
        print(f"Inference failed: {e}")

    print("\n=== Batch Inference (mixed modes) ===")
    messages_batch = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "./figures/example.jpg"},
                    {"type": "text", "text": "Describe the image in detail."},
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "How to open a fridge?"},
                ],
            }
        ],
    ]
    try:
        batch_results = batch_inference(model, processor, messages_batch)
        for i, result in enumerate(batch_results):
            print(f"\n--- Sample {i} ---\n{result}")
    except Exception as e:
        print(f"Batch inference failed: {e}")


if __name__ == "__main__":
    main()
