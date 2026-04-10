import os
import inspect
import torch

from mps_compat import enable_hunyuan_mps_support, get_default_device, get_default_dtype

# Configuration
MODEL_PATH = "tencent/HY-Embodied-0.5"
DEVICE = get_default_device()
MODEL_DTYPE = get_default_dtype(DEVICE)
THINKING_MODE = False
TEMPERATURE = 0.8
MAX_NEW_TOKENS = 1024

if DEVICE == "mps":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    enable_hunyuan_mps_support()

from transformers import AutoModelForImageTextToText, AutoProcessor


def _apply_chat_template(processor, messages, **kwargs):
    """Only pass chat template kwargs supported by the installed processor."""
    params = inspect.signature(processor.apply_chat_template).parameters
    if "enable_thinking" not in params:
        kwargs.pop("enable_thinking", None)
    return processor.apply_chat_template(messages, **kwargs)

def load_model_and_processor():
    """Load model and processor with proper configuration."""
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # Load chat template if available
    chat_template_path = os.path.join(MODEL_PATH, "chat_template.jinja")
    if os.path.exists(chat_template_path):
        processor.chat_template = open(chat_template_path).read()

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=MODEL_DTYPE
    )
    model.to(DEVICE).eval()

    return model, processor

def single_inference(model, processor, image_path, text_prompt):
    """Run single inference with image and text prompt."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    inputs = _apply_chat_template(
        processor,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=THINKING_MODE,
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

def batch_inference(model, processor, messages_batch):
    """Run batch inference with multiple prompts."""
    # Process each message independently
    all_inputs = []
    for msgs in messages_batch:
        inp = _apply_chat_template(
            processor,
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=THINKING_MODE,
        )
        all_inputs.append(inp)

    # Left-pad and batch
    batch = processor.pad(all_inputs, padding=True, padding_side="left").to(model.device)

    with torch.no_grad():
        batch_generated_ids = model.generate(
            **batch,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            temperature=TEMPERATURE,
            do_sample=TEMPERATURE > 0,
        )

    # Decode: strip the padded input portion
    padded_input_len = batch["input_ids"].shape[1]
    results = []
    for i, msgs in enumerate(messages_batch):
        out_ids = batch_generated_ids[i][padded_input_len:]
        results.append(processor.decode(out_ids, skip_special_tokens=True))

    return results

def main():
    """Main function demonstrating both single and batch inference."""
    print("Loading HY-Embodied model...")
    model, processor = load_model_and_processor()
    print(f"Model loaded successfully on {DEVICE}")

    # Single inference example
    print("\n=== Single Inference Example ===")
    try:
        result = single_inference(
            model, processor,
            image_path="./figures/example.jpg",
            text_prompt="Describe the image in detail."
        )
        print("Result:", result)
    except Exception as e:
        print(f"Single inference failed: {e}")
        print("Note: Make sure './figures/example.jpg' exists or provide a valid image path")

    # Batch inference example
    print("\n=== Batch Inference Example ===")
    messages_batch = [
        # Sample A: image + text
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "./figures/example.jpg"},
                    {"type": "text", "text": "Describe the image in detail."},
                ],
            }
        ],
        # Sample B: text only
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
            print(f"\n--- Sample {i} ---")
            print(result)
    except Exception as e:
        print(f"Batch inference failed: {e}")

if __name__ == "__main__":
    main()
