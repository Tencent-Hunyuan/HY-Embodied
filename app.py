"""
HY-Embodied Gradio Demo
=======================
Interactive chat interface for HY-Embodied-0.5 with image upload support
and streaming output.

Usage:
    python app.py                        # local, http://127.0.0.1:7860
    python app.py --share                # public Gradio link
    MODEL_PATH=/local/dir python app.py  # local weights

Requirements:
    pip install "gradio>=4.44"
"""

import os
import threading
from pathlib import Path

import torch
import gradio as gr
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "tencent/HY-Embodied-0.5")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

TITLE = "HY-Embodied-0.5"
DESCRIPTION = (
    "**HY-Embodied-0.5** is a multimodal foundation model for embodied intelligence "
    "by Tencent Robotics X × HY Vision Team. Upload an image and ask questions about "
    "spatial reasoning, object interactions, and robot planning."
)

# ---------------------------------------------------------------------------
# Model — loaded once on first request
# ---------------------------------------------------------------------------

_model = None
_processor = None
_lock = threading.Lock()


def _get_model():
    global _model, _processor
    if _model is None:
        with _lock:
            if _model is None:
                print(f"Loading {MODEL_PATH} on {DEVICE} …")
                _processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
                # Load chat template if bundled separately (local checkpoints)
                chat_template_path = Path(MODEL_PATH) / "chat_template.jinja"
                if chat_template_path.exists():
                    _processor.chat_template = chat_template_path.read_text()
                _model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=DTYPE,
                    trust_remote_code=True,
                )
                _model.to(DEVICE).eval()
                print("Model loaded.")
    return _model, _processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _history_to_messages(history: list[dict]) -> list[dict]:
    """Convert Gradio message history to the transformers chat template format."""
    messages = []
    for turn in history:
        role = turn["role"]
        content = turn["content"]
        if isinstance(content, str):
            messages.append({"role": role, "content": [{"type": "text", "text": content}]})
        elif isinstance(content, list):
            # Already in multi-part format
            messages.append({"role": role, "content": content})
        else:
            # gr.Image / file path from a previous multimodal turn
            messages.append({"role": role, "content": [{"type": "image", "image": content}]})
    return messages


def respond(
    message: dict,
    history: list[dict],
    temperature: float,
    max_new_tokens: int,
    thinking_mode: bool,
):
    """Generator that yields streamed response tokens."""
    model, processor = _get_model()

    # Build content for the current turn
    content = []
    for file_path in message.get("files", []):
        content.append({"type": "image", "image": file_path})
    text = message.get("text", "").strip()
    if text:
        content.append({"type": "text", "text": text})

    if not content:
        yield ""
        return

    # Assemble full conversation
    messages = _history_to_messages(history)
    messages.append({"role": "user", "content": content})

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=thinking_mode,
    ).to(model.device)

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=60,
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else 1.0,
        do_sample=temperature > 0,
        use_cache=True,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
    thread.start()

    partial = ""
    for token in streamer:
        partial += token
        yield partial


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="Chat",
                    type="messages",
                    height=560,
                    show_copy_button=True,
                    avatar_images=(None, "figures/teaser.png") if Path("figures/teaser.png").exists() else None,
                )
                input_box = gr.MultimodalTextbox(
                    placeholder="Upload an image and ask a question, or type a text-only prompt…",
                    file_types=["image"],
                    file_count="single",
                    submit_btn=True,
                    label="Message",
                )

            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### Settings")
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.8,
                    step=0.05,
                    label="Temperature",
                    info="0 = greedy (deterministic)",
                )
                max_new_tokens = gr.Slider(
                    minimum=64,
                    maximum=2048,
                    value=1024,
                    step=64,
                    label="Max new tokens",
                )
                thinking_mode = gr.Checkbox(
                    value=False,
                    label="Thinking mode",
                    info="Enable step-by-step chain-of-thought reasoning",
                )
                clear_btn = gr.Button("Clear chat", variant="secondary")

                gr.Markdown("---")
                gr.Markdown(
                    "**Model**: [tencent/HY-Embodied-0.5](https://huggingface.co/tencent/HY-Embodied-0.5)  \n"
                    "**Code**: [Tencent-Hunyuan/HY-Embodied](https://github.com/Tencent-Hunyuan/HY-Embodied)"
                )

        # Example prompts
        gr.Examples(
            examples=[
                [{"text": "How would you open this fridge? Describe the steps.", "files": []}],
                [{"text": "What objects are visible and how are they spatially arranged?", "files": []}],
                [{"text": "Which object should a robot grasp first to clear the table?", "files": []}],
            ],
            inputs=[input_box],
            label="Text-only examples (upload an image to ground them)",
        )

        # Wire up streaming chat
        chat_interface = gr.ChatInterface(
            fn=respond,
            chatbot=chatbot,
            textbox=input_box,
            additional_inputs=[temperature, max_new_tokens, thinking_mode],
            type="messages",
        )

        clear_btn.click(fn=lambda: [], outputs=[chatbot])

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
