"""
HY-Embodied Fine-tuning
=======================
LoRA supervised fine-tuning for HY-Embodied-0.5 (MoT-2B).

The MoT architecture has dual projection paths per decoder layer:
  - Text path  : q/k/v/o_proj, mlp  (gate/up/down_proj)
  - Vision path: q/k/v/o_proj_v, mlp_v (gate/up/down_proj)

LoRA is applied to both paths.  The vision encoder (ViT) is frozen by
default; pass --train_projector to also fine-tune the vision-to-text
projector (model.visual.merger).

Dataset format (JSONL, one JSON object per line)
-------------------------------------------------
Text-only turn:
  {"messages": [
      {"role": "user",      "content": "How do you open a fridge?"},
      {"role": "assistant", "content": "First, grasp the handle …"}
  ]}

Image + text turn:
  {"messages": [
      {"role": "user", "content": [
          {"type": "image", "image": "path/to/image.jpg"},
          {"type": "text",  "text": "Describe the scene."}
      ]},
      {"role": "assistant", "content": "The image shows …"}
  ]}

Multi-turn conversations are fully supported.

Usage
-----
  # Single GPU
  python finetune.py \\
      --model_path tencent/HY-Embodied-0.5 \\
      --data_path data/train.jsonl \\
      --output_dir checkpoints/my-run \\
      --lora_r 64 --lora_alpha 128

  # Multi-GPU with accelerate
  accelerate launch --num_processes 4 finetune.py \\
      --model_path tencent/HY-Embodied-0.5 \\
      --data_path data/train.jsonl \\
      --output_dir checkpoints/my-run

Requirements
------------
  pip install peft datasets
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_path: str = field(
        default="tencent/HY-Embodied-0.5",
        metadata={"help": "HF repo ID or local path to model and processor."},
    )
    trust_remote_code: bool = field(default=True)
    # LoRA
    lora_r: int = field(default=64, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha (scaling = alpha / r)."})
    lora_dropout: float = field(default=0.05)
    # Frozen components
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Freeze the ViT vision encoder (recommended)."},
    )
    train_projector: bool = field(
        default=False,
        metadata={"help": "Fine-tune the vision-to-text MLP projector (model.visual.merger)."},
    )


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to training JSONL file."})
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to optional evaluation JSONL file."},
    )
    max_seq_len: int = field(
        default=2048,
        metadata={"help": "Maximum token sequence length; longer examples are skipped."},
    )
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "Pass enable_thinking=True to apply_chat_template."},
    )


# ---------------------------------------------------------------------------
# LoRA target modules
# ---------------------------------------------------------------------------

# Text attention path
_ATTN_TEXT = ["q_proj", "k_proj", "v_proj", "o_proj"]
# Vision attention path (MoT-specific, _v suffix)
_ATTN_VISION = ["q_proj_v", "k_proj_v", "v_proj_v", "o_proj_v"]
# MLP gates — PEFT matches by suffix, so this covers both mlp.* and mlp_v.*
_MLP = ["gate_proj", "up_proj", "down_proj"]

LORA_TARGET_MODULES = _ATTN_TEXT + _ATTN_VISION + _MLP


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """
    Loads a JSONL file of multi-turn conversations and tokenises them.

    Labels are -100 everywhere except the assistant turns so that loss is
    only computed on the model's responses.
    """

    def __init__(
        self,
        data_path: str,
        processor: AutoProcessor,
        max_seq_len: int,
        enable_thinking: bool = False,
    ):
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.enable_thinking = enable_thinking
        self.samples = self._load(data_path)
        logger.info("Loaded %d samples from %s", len(self.samples), data_path)

    @staticmethod
    def _load(path: str) -> list[dict]:
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _tokenise_messages(self, messages: list[dict]) -> dict:
        """Tokenise a full conversation and return a BatchFeature."""
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.enable_thinking,
        )

    def _build_labels(self, messages: list[dict], input_ids: torch.Tensor) -> torch.Tensor:
        """
        Build a label tensor from -100 (masked) with assistant tokens unmasked.

        Strategy: for each assistant turn, tokenise the conversation up to and
        including that turn, then tokenise up to (but not including) that turn.
        The difference in length gives the assistant token span.
        """
        labels = torch.full_like(input_ids, fill_value=-100)
        seq_len = input_ids.shape[1]

        # Walk through turns, find assistant spans
        prefix_messages: list[dict] = []
        for turn in messages:
            if turn["role"] != "assistant":
                prefix_messages.append(turn)
                continue

            # Tokenise everything up to (but not including) this response
            prefix_ids = self._tokenise_messages(
                prefix_messages + [{"role": "assistant", "content": ""}]
                if self.processor.tokenizer.bos_token  # keep BOS consistent
                else prefix_messages
            )["input_ids"]
            prefix_len = prefix_ids.shape[1]

            # Tokenise everything up to and including this response
            full_ids = self._tokenise_messages(prefix_messages + [turn])["input_ids"]
            turn_end = full_ids.shape[1]

            # Unmask the assistant response tokens (clamped to actual seq_len)
            start = min(prefix_len, seq_len)
            end = min(turn_end, seq_len)
            if end > start:
                labels[0, start:end] = input_ids[0, start:end]

            prefix_messages.append(turn)

        return labels

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        sample = self.samples[idx]
        messages = sample["messages"]

        try:
            encoded = self._tokenise_messages(messages)
        except Exception as e:
            logger.warning("Skipping sample %d due to processing error: %s", idx, e)
            return None

        input_ids = encoded["input_ids"]
        if input_ids.shape[1] > self.max_seq_len:
            logger.debug("Skipping sample %d: length %d > max_seq_len %d", idx, input_ids.shape[1], self.max_seq_len)
            return None

        labels = self._build_labels(messages, input_ids)

        item: dict[str, Any] = {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }
        # Include vision tensors when present
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if key in encoded and encoded[key] is not None:
                item[key] = encoded[key].squeeze(0) if encoded[key].dim() > 1 else encoded[key]

        return item


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class SFTDataCollator:
    """
    Left-pads token sequences in a batch to the same length.
    Vision tensors are concatenated along the batch/patch dimension.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict | None]) -> dict[str, torch.Tensor]:
        # Drop None (skipped) samples
        batch = [b for b in batch if b is not None]
        if not batch:
            raise ValueError("Empty batch after filtering — check your dataset.")

        max_len = max(b["input_ids"].shape[0] for b in batch)
        input_ids, attention_masks, labels_list = [], [], []

        for item in batch:
            seq_len = item["input_ids"].shape[0]
            pad_len = max_len - seq_len
            input_ids.append(torch.cat([
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                item["input_ids"],
            ]))
            attention_masks.append(torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                item["attention_mask"],
            ]))
            labels_list.append(torch.cat([
                torch.full((pad_len,), -100, dtype=torch.long),
                item["labels"],
            ]))

        out: dict[str, torch.Tensor] = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels_list),
        }

        # Concatenate vision tensors from all samples that have them
        for key in ("pixel_values", "pixel_values_videos"):
            tensors = [b[key] for b in batch if key in b]
            if tensors:
                out[key] = torch.cat(tensors, dim=0)

        for key in ("image_grid_thw", "video_grid_thw"):
            tensors = [b[key] for b in batch if key in b]
            if tensors:
                out[key] = torch.cat(tensors, dim=0)

        return out


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_model_for_training(args: ModelArguments) -> tuple:
    logger.info("Loading processor from %s", args.model_path)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )

    chat_template_path = Path(args.model_path) / "chat_template.jinja"
    if chat_template_path.exists():
        processor.chat_template = chat_template_path.read_text()

    logger.info("Loading model from %s", args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    )

    # Freeze vision encoder
    if args.freeze_vision_tower:
        logger.info("Freezing vision tower (model.visual.vision_tower)")
        for param in model.model.visual.vision_tower.parameters():
            param.requires_grad = False

    # Freeze projector unless explicitly requested
    if not args.train_projector:
        logger.info("Freezing vision projector (model.visual.merger)")
        for param in model.model.visual.merger.parameters():
            param.requires_grad = False

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        # Ensure we cover the inner language model path
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, processor = load_model_for_training(model_args)

    train_dataset = SFTDataset(
        data_path=data_args.data_path,
        processor=processor,
        max_seq_len=data_args.max_seq_len,
        enable_thinking=data_args.enable_thinking,
    )
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = SFTDataset(
            data_path=data_args.eval_data_path,
            processor=processor,
            max_seq_len=data_args.max_seq_len,
            enable_thinking=data_args.enable_thinking,
        )

    collator = SFTDataCollator(pad_token_id=processor.tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    logger.info("Starting training …")
    trainer.train()

    logger.info("Saving adapter to %s", training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
