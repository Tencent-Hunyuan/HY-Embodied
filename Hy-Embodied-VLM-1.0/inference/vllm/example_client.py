#!/usr/bin/env python3
"""
example_client.py — OpenAI-SDK example client for a Hy-Embodied-VLM-1.0
vLLM server started with `serve.sh`.

Demonstrates:
  1. text-only chat completion
  2. image + text chat completion (multimodal)
  3. enable_thinking toggle (think vs direct answer)
  4. streaming responses

Install:
    pip install "openai>=1.30" pillow

Usage:
    python example_client.py --port 8080
"""

import argparse
import base64
from pathlib import Path

from openai import OpenAI


def encode_image(path):
    """Return a data-URL base64 payload for the given image file."""
    b = Path(path).read_bytes()
    mime = "image/jpeg" if path.lower().endswith(('.jpg', '.jpeg')) else "image/png"
    return f"data:{mime};base64,{base64.b64encode(b).decode()}"


def demo_text_only(client, model_name, enable_thinking):
    print(f"\n─── text-only, enable_thinking={enable_thinking} ───")
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "How do you open a fridge?"}],
        max_tokens=512,
        temperature=0.7,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    msg = resp.choices[0].message
    if getattr(msg, "reasoning_content", None):
        print(f"[thinking] {msg.reasoning_content[:300]}...")
    print(f"[answer]   {msg.content}")


def demo_image(client, model_name, image_path, enable_thinking):
    print(f"\n─── image + text, enable_thinking={enable_thinking} ───")
    if not Path(image_path).exists():
        print(f"[skip] {image_path} not found")
        return
    data_url = encode_image(image_path)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "Describe the image in detail."},
            ],
        }],
        max_tokens=1024,
        temperature=0.7,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    msg = resp.choices[0].message
    if getattr(msg, "reasoning_content", None):
        print(f"[thinking] {msg.reasoning_content[:300]}...")
    print(f"[answer]   {msg.content}")


def demo_streaming(client, model_name, enable_thinking):
    print(f"\n─── streaming, enable_thinking={enable_thinking} ───")
    stream = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Explain how a robot arm reaches for a cup."}],
        max_tokens=512,
        temperature=0.7,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model", default="hy_a3b", help="Served model name (default: hy_a3b)")
    ap.add_argument("--image", default="./figures/example.jpg", help="Optional image path")
    args = ap.parse_args()

    client = OpenAI(base_url=f"http://{args.host}:{args.port}/v1", api_key="EMPTY")

    demo_text_only(client, args.model, enable_thinking=True)
    demo_text_only(client, args.model, enable_thinking=False)
    demo_image(client, args.model, args.image, enable_thinking=True)
    demo_streaming(client, args.model, enable_thinking=True)


if __name__ == "__main__":
    main()
