<div align="center">
<h1>HY-Embodied</h1>
<p><b>A Family of Embodied Foundation Models for Real-World Agents</b></p>
<p><i>Tencent Robotics X × HY Vision Team</i></p>

<a href="hy_embodied_tech_report.pdf"><img src="https://img.shields.io/badge/Paper-Report-red?logo=report" alt="Tech Report"></a>
<a href="https://huggingface.co/tencent/HY-Embodied-0.5/tree/main"><img src="https://img.shields.io/badge/Models-HuggingFace-yellow?logo=huggingface" alt="Models"></a>

</div>

https://github.com/user-attachments/assets/a5c6b872-2cb0-4f52-8321-894fee7da27e

## 🔥 Updates

  * **`[2026-04-09]`** 🚀 We have released **HY-Embodied-0.5**, featuring the open-sourced `HY-Embodied-0.5 MoT-2B` weights on [Hugging Face](https://huggingface.co/tencent/HY-Embodied-0.5/tree/main) along with the official inference code\!

## 📖 Abstract

We introduce **HY-Embodied-0.5**, a suite of foundation models tailored specifically for real-world embodied intelligence. To bridge the gap between general Vision-Language Models (VLMs) and the strict demands of physical agents, our models are engineered to excel in spatial-temporal visual perception and complex embodied reasoning (prediction, interaction, and planning).

The suite features an innovative **Mixture-of-Transformers (MoT)** architecture utilizing latent tokens for modality-specific computing, significantly enhancing fine-grained perception. It includes two primary variants: a highly efficient **2B model** for edge deployment and a powerful **32B model** for complex reasoning. Through a self-evolving post-training paradigm and large-to-small on-policy distillation, our compact MoT-2B outperforms state-of-the-art models of similar size across 16 benchmarks, while the 32B variant achieves frontier-level performance comparable to Gemini 3.0 Pro. Ultimately, HY-Embodied serves as a robust "brain" for Vision-Language-Action (VLA) pipelines, delivering compelling results in real-world physical robot control.

<div align="center">
<img src="figures/teaser.png" alt="HY-Embodied Teaser" width="85%">
</div>

## ⭐️ Key Features

  * 🧠 **Evolved MoT Architecture:** Designed for maximum efficiency without sacrificing visual acuity. The MoT-2B variant contains 4B total parameters but requires **only 2.2B activated parameters** during inference. By emphasizing modality-specific computing in the vision pathway, it achieves the high inference speed of a dense 2B model while delivering superior, fine-grained perceptual representations.
  * 🔗 **High-Quality Mixed Chain Reasoning:** We introduce an advanced iterative, self-evolving post-training pipeline. By employing on-policy distillation, we successfully transfer the sophisticated step-by-step reasoning, planning, and high-quality "thinking" capabilities from our powerful 32B model directly to the compact 2B variant.
  * 🌍 **Large-Scale Embodied Pre-training:** Grounded in a massive, specially curated dataset comprising **\>100 million** embodied and spatial-specific data points. Trained on a corpus exceeding **200 billion tokens**, the model develops a deep, native understanding of 3D spaces, physical object interactions, and agent dynamics.
  * 🦾 **Stronger VLA Application:** Beyond standard academic benchmarks, HY-Embodied is engineered to be the core cognitive engine for physical robots. It seamlessly integrates into Vision-Language-Action (VLA) frameworks, acting as a highly robust and capable brain to drive high success rates in complex, real-world robotic control tasks.

<div align="center">
<img src="figures/arch.png" alt="HY-Embodied Architecture" width="85%">
</div>

## 📅 Plannings

- [x] Transformers Inference
- [ ] vLLM Inference
- [ ] Online Gradio Demo

## 🛠️ Dependencies and Installation

### Prerequisites

- 🖥️ **Operating System**: Linux (recommended), Apple Silicon macOS (experimental inference only)
- 🐍 **Python**: 3.12+ (recommended and tested)
- ⚡ **CUDA**: 12.6
- 🔥 **PyTorch**: 2.8.0
- 🎮 **GPU**: NVIDIA GPU with CUDA support
- 🍎 **Apple Silicon**: Experimental PyTorch MPS inference is also supported through the provided runtime compatibility shim in `mps_compat.py`

### Installation

1. **Install the specific Transformers version required for this model:**
```bash
pip install git+https://github.com/huggingface/transformers@9293856c419762ebf98fbe2bd9440f9ce7069f1a
```

> **Note**: We will merge the improvements into the Transformers main branch later.

2. **Install other dependencies:**
```bash
pip install -r requirements.txt
```

> **Apple Silicon note**: on macOS, `flash_attn` is skipped automatically by `requirements.txt`, and the demo script enables an experimental MPS fallback path.

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/Tencent-Hunyuan/HY-Embodied
cd HY-Embodied/
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run inference:**
```bash
python inference.py
```

The example script demonstrates both single generation and batch generation capabilities.

### Model Download

The code automatically downloads the model `tencent/HY-Embodied-0.5` from Hugging Face Hub. Ensure you have sufficient disk space (8 GB) for the model weights.

### Hardware Requirements

- **GPU**: Recommended for optimal performance (NVIDIA GPU with at least 16GB VRAM)
- **CPU**: Supported but slower
- **Memory**: At least 16GB RAM recommended
- **Storage**: 20GB+ free space for model and dependencies

## 🚀 Quick Start with Transformers

### Basic Inference Example

```python
import os
import torch
from mps_compat import enable_hunyuan_mps_support, get_default_device, get_default_dtype

# Load model & processor
MODEL_PATH = "tencent/HY-Embodied-0.5"
DEVICE = get_default_device()
DTYPE = get_default_dtype(DEVICE)
THINKING_MODE = False
TEMPERATURE = 0.8

if DEVICE == "mps":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    enable_hunyuan_mps_support()

from transformers import AutoModelForImageTextToText, AutoProcessor

processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Load chat template if available
chat_template_path = os.path.join(MODEL_PATH, "chat_template.jinja")
if os.path.exists(chat_template_path):
    processor.chat_template = open(chat_template_path).read()

model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, torch_dtype=DTYPE)
model.to(DEVICE).eval()

# Prepare input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "./figures/example.jpg"},
            {"type": "text", "text": "Describe the image in detail."},
        ],
    }
]

# Process and generate
inputs = processor.apply_chat_template(
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
        max_new_tokens=32768,
        use_cache=True,
        temperature=TEMPERATURE,
        do_sample=TEMPERATURE > 0,
    )

output_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
print(processor.batch_decode(output_ids, skip_special_tokens=True)[0])
```

### Batch Inference

```python
import os
import torch
from mps_compat import enable_hunyuan_mps_support, get_default_device, get_default_dtype

# Load model & processor
MODEL_PATH = "tencent/HY-Embodied-0.5"
DEVICE = get_default_device()
DTYPE = get_default_dtype(DEVICE)
THINKING_MODE = False
TEMPERATURE = 0.8

if DEVICE == "mps":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    enable_hunyuan_mps_support()

from transformers import AutoModelForImageTextToText, AutoProcessor

processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Load chat template if available
chat_template_path = os.path.join(MODEL_PATH, "chat_template.jinja")
if os.path.exists(chat_template_path):
    processor.chat_template = open(chat_template_path).read()

model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, torch_dtype=DTYPE)
model.to(DEVICE).eval()

# Batch Inference (multiple prompts at once)
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

# Process each message independently
all_inputs = []
for msgs in messages_batch:
    inp = processor.apply_chat_template(
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
        max_new_tokens=32768,
        use_cache=True,
        temperature=TEMPERATURE,
        do_sample=TEMPERATURE > 0,
    )

# Decode: strip the padded input portion
padded_input_len = batch["input_ids"].shape[1]
for i, msgs in enumerate(messages_batch):
    out_ids = batch_generated_ids[i][padded_input_len:]
    print(f"\n--- Sample {i} ---")
    print(processor.decode(out_ids, skip_special_tokens=True))
```

## 📊 Evaluation

### Visual Perception

> **Note**: We evaluated HY-Embodied-0.5 MoT-2B across 22 embodied-relevant benchmarks against models of similar size. For detailed performance metrics and methodology, please refer to our technical report.
> **Note**: We observed that small models from the Qwen3.5 series produce repetitive thinking patterns in some benchmarks, which leads to lower overall results. Therefore, we compare against Qwen3-VL models in our evaluations.

| Benchmark | HY-Embodied 0.5 MoT-2B | Qwen3-VL 2B | Qwen3-VL 4B | RoboBrain 2.5 4B | MiMo-Embodied 7B |
|-----------|------------------------|-------------|-------------|------------------|------------------|
| CV-Bench  | **89.2** | 80.0 | 85.7 | 86.9 | 88.8 |
| DA-2K     | **92.3** | 69.5 | 76.5 | 79.4 | 72.2 |

### Embodied Understanding

| Benchmark | HY-Embodied 0.5 MoT-2B | Qwen3-VL 2B | Qwen3-VL 4B | RoboBrain 2.5 4B | MiMo-Embodied 7B |
|-----------|------------------------|-------------|-------------|------------------|------------------|
| ERQA | **54.5** | 41.8 | 47.3 | 43.3 | 46.8 |
| EmbSpatial-Bench | **82.8** | 75.9 | 80.7 | 73.8 | 76.2 |
| RoboBench-MCQ | **49.2** | 36.9 | 45.8 | 44.4 | 43.6 |
| RoboBench-Planning | 54.2 | 36.2 | 36.4 | 39.2 | **58.7** |
| RoboSpatial-Home | 55.7 | 45.3 | **63.2** | 62.3 | 61.8 |
| ShareRobot-Aff. | **26.8** | 19.8 | 25.5 | 25.5 | 9.0 |
| ShareRobot-Traj. | 73.3 | 41.6 | 62.2 | **81.4** | 50.6 |
| Ego-Plan2 | 45.5 | 35.5 | 38.8 | **52.6** | 39.9 |

### Spatial Understanding

| Benchmark | HY-Embodied 0.5 MoT-2B | Qwen3-VL 2B | Qwen3-VL 4B | RoboBrain 2.5 4B | MiMo-Embodied 7B |
|-----------|------------------------|-------------|-------------|------------------|------------------|
| 3DSRBench | **57.0** | 39.9 | 43.9 | 44.8 | 42.0 |
| All-Angles Bench | **55.1** | 42.3 | 46.7 | 43.8 | 49.0 |
| MindCube | **66.3** | 28.4 | 31.0 | 26.9 | 36.2 |
| MMSI-Bench | **33.2** | 23.6 | 25.1 | 20.5 | 31.9 |
| RefSpatial-Bench | 45.8 | 28.9 | 45.3 | **56.0** | 48.0 |
| SAT | 76.7 | 45.3 | 56.7 | 51.3 | **78.7** |
| SIBench-mini | **58.2** | 42.0 | 50.9 | 47.3 | 53.1 |
| SITE-Bench-Image | **62.7** | 52.3 | 61.0 | 57.9 | 49.9 |
| SITE-Bench-Video | **63.5** | 52.2 | 58.0 | 54.8 | 58.9 |
| ViewSpatial | **53.1** | 37.2 | 41.6 | 36.6 | 36.1 |
| VSIBench | **60.5** | 48.0 | 55.2 | 41.7 | 48.5 |
| Where2Place | **68.0** | 45.0 | 59.0 | 65.0 | 63.6 |

*Note: Results for HY-Embodied-0.5 MoT-2B are reported in thinking mode, while for all other models, we report the better performance between non-thinking and thinking modes.*

## 📚 Citation

*Citation information will be available soon. Please check back for updates.*

## 🙏 Acknowledgements

We thank the Hugging Face community for their support and the open-source contributions that made this implementation possible.
