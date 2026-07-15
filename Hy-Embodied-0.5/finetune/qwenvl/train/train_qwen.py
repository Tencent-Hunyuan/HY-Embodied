# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging

# Dynamically resolve project root (3 levels up from this file: train_qwen.py -> train -> qwenvl -> root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pathlib
import torch
import transformers
import sys
from pathlib import Path

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # Truncated File Read
Image.MAX_IMAGE_PIXELS = None # DecompressionBombWarning
ImageFile.MAX_IMAGE_PIXELS = None

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from transformers import AutoTokenizer, AutoImageProcessor, AutoVideoProcessor, Trainer
from qwenvl.model.model_wrapper import HunYuanVLMoTWithAuxLoss
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


mot_patterns = [
    '.mlp_v.',
    '.post_attention_layernorm_v.',
    '.input_layernorm_v.',
    '.o_proj_v.',
    '.k_proj_v.',
    '.q_proj_v.',
    '.v_proj_v.',
]

def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.model.visual.named_parameters():
            p.requires_grad = False
    
    if model_args.tune_mm_mlp:
        for n, p in model.model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.language_model.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.model.language_model.named_parameters():
            p.requires_grad = False
    
    if model_args.tune_mot_vision:
        for n, p in model.model.language_model.named_parameters():
            if any(pattern in n for pattern in mot_patterns):
                p.requires_grad = True
                rank0_print(f"[MoT] Fine-tuning {n}")

    # =========================================================================
    # Patch HYViT2_400MAnyRes to match OLD SigLIPViTAnysizeWrapper behavior
    # The official model always uses torch.no_grad() for ViT and keeps it in eval.
    # OLD code has VIT_WITH_GRAD, UPDATE_VIT_EVERY_N, and gradient_checkpointing.
    # =========================================================================
    VIT_WITH_GRAD = os.environ.get('VIT_WITH_GRAD')
    UPDATE_VIT_EVERY_N = int(os.environ['UPDATE_VIT_EVERY_N']) if 'UPDATE_VIT_EVERY_N' in os.environ else None

    import types
    visual = model.model.visual

    if VIT_WITH_GRAD and model_args.tune_mm_vision:
        # 1. Enable gradient checkpointing for ViT (saves memory during ViT training)
        if hasattr(visual.vision_tower, 'set_grad_checkpointing'):
            visual.vision_tower.set_grad_checkpointing(True)
            rank0_print("[VIT] Enabled gradient checkpointing for vision_tower")
        elif hasattr(visual.vision_tower, 'grad_checkpointing'):
            visual.vision_tower.grad_checkpointing = True
            rank0_print("[VIT] Enabled gradient checkpointing for vision_tower")

        # 2. Put vision_tower in train mode
        visual.vision_tower.train()

        # 3. Override train() to keep vision_tower in train mode
        def _visual_train(self, mode=True):
            self.training = mode
            if self.is_loaded:
                self.vision_tower.train(mode)
            return self
        visual.train = types.MethodType(_visual_train, visual)

        # 4. Add vit_forward_count and override forward() with UPDATE_VIT_EVERY_N logic
        visual.vit_forward_count = 0

        def _patched_visual_forward(self, images, cal_attn_pool=False):
            """Patched forward matching OLD SigLIPViTAnysizeWrapper behavior."""
            if UPDATE_VIT_EVERY_N is not None:
                if self.vit_forward_count % UPDATE_VIT_EVERY_N == 0:
                    # This step: compute ViT with grad
                    for p in self.vision_tower.parameters():
                        p.requires_grad = True
                    image_features, img_size, cls_token = self._forward_func(images, cal_attn_pool=cal_attn_pool)
                else:
                    # Other steps: no grad for ViT
                    if self.vit_forward_count % UPDATE_VIT_EVERY_N == 1:
                        for p in self.vision_tower.parameters():
                            p.requires_grad = False
                    with torch.no_grad():
                        image_features, img_size, cls_token = self._forward_func(images, cal_attn_pool=cal_attn_pool)
                self.vit_forward_count += 1
            else:
                # No periodic update — always compute with grad
                image_features, img_size, cls_token = self._forward_func(images, cal_attn_pool=cal_attn_pool)

            # Merger (always with grad)
            if isinstance(images, list):
                image_features = [self.merger(x, s).squeeze(0) for x, s in zip(image_features, img_size)]
            else:
                image_features = self.merger(image_features, img_size)
                C = image_features.shape[-1]
                image_features = [image_features.reshape(-1, C)]

            return image_features

        visual.forward = types.MethodType(_patched_visual_forward, visual)
        rank0_print(f"[VIT] Patched visual.forward() with VIT_WITH_GRAD=1, UPDATE_VIT_EVERY_N={UPDATE_VIT_EVERY_N}")

    # # 5. Freeze unused ViT classification head to avoid NaN from unused parameters
    # if hasattr(visual, 'vision_tower'):
    #     vit = visual.vision_tower
    #     if hasattr(vit, 'head') and isinstance(vit.head, torch.nn.Linear):
    #         for p in vit.head.parameters():
    #             p.requires_grad = False
    #     if hasattr(vit, 'fc_norm') and isinstance(vit.fc_norm, torch.nn.LayerNorm):
    #         for p in vit.fc_norm.parameters():
    #             p.requires_grad = False
    #     rank0_print("[VIT] Froze unused vision_tower.head and fc_norm")


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    model = HunYuanVLMoTWithAuxLoss.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    data_args.model_type = "hunyuanvl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')

    from hunyuan_vl_mot.processing_hunyuan_vl_mot import HunYuanVLMoTProcessor, HunYuanVLMoTProcessorKwargs
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    image_processor = AutoImageProcessor.from_pretrained(model_args.model_name_or_path)
    video_processor = AutoVideoProcessor.from_pretrained(model_args.model_name_or_path)
    # Use training chat_template (without latent tokens - processor will insert them)
    chat_template_path = os.path.join(PROJECT_ROOT, 'chat_template_train.jinja')
    with open(chat_template_path, 'r') as f:
        chat_template_content = f.read()
    processor = HunYuanVLMoTProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=chat_template_content,
    )
    rank0_print(f"[Processor] Loaded training chat_template from {chat_template_path}")

    # Monkey-patch processor.__call__ to add VISUAL_LATENT token insertion (always on)
    # The official model relies on a latent token being inserted right before every
    # vision_end_token. This is a required behavior of the fine-tuning pipeline,
    # so it is unconditionally enabled (no env var gating).
    if True: # default to turn on VISUAL_LATENT
        vision_end_token = processor.vision_end_token
        # latent_token = "<｜hy_place▁holder▁no▁672｜>"
        latent_token = getattr(processor.tokenizer, 'latent_token', '<｜hy_place▁holder▁no▁672｜>')

        import types

        def _patched_call_with_latent(self, images=None, text=None, videos=None, **kwargs):
            """Patched __call__ that inserts VISUAL_LATENT tokens after image/video expansion."""
            from transformers.feature_extraction_utils import BatchFeature

            output_kwargs = self._merge_kwargs(
                HunYuanVLMoTProcessorKwargs,
                tokenizer_init_kwargs=self.tokenizer.init_kwargs,
                **kwargs,
            )
            if images is not None:
                image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
                image_grid_thw = image_inputs["image_grid_thw"]
            else:
                image_inputs = {}
                image_grid_thw = None

            if videos is not None:
                videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
                video_grid_thw = videos_inputs["video_grid_thw"]
                if "return_metadata" not in kwargs:
                    video_metadata = videos_inputs.pop("video_metadata")
                else:
                    video_metadata = videos_inputs["video_metadata"]
            else:
                videos_inputs = {}
                video_grid_thw = None

            if not isinstance(text, list):
                text = [text]
            text = text.copy()

            # Image token expansion
            if image_grid_thw is not None:
                index = 0
                for i in range(len(text)):
                    while self.image_token in text[i]:
                        row_tokens = (
                            "<|placeholder|>" * (image_grid_thw[index][2] // self.image_processor.merge_size)
                            + self.image_newline_token
                        )
                        image_prompt = row_tokens * (
                            image_grid_thw[index][0] * image_grid_thw[index][1] // self.image_processor.merge_size
                        )
                        text[i] = text[i].replace(self.image_token, image_prompt, 1)
                        index += 1
                    text[i] = text[i].replace("<|placeholder|>", self.image_token)

            # Video token expansion
            if video_grid_thw is not None:
                index = 0
                for i in range(len(text)):
                    while self.video_token in text[i]:
                        metadata = video_metadata[index]
                        if metadata.fps is None:
                            metadata.fps = 24

                        row_tokens = (
                            "<|placeholder|>" * (video_grid_thw[index][2] // self.video_processor.merge_size)
                            + self.image_newline_token
                        )
                        video_prompt = row_tokens * (video_grid_thw[index][1] // self.video_processor.merge_size)

                        video_placeholder = ""
                        for frame_idx in range(video_grid_thw[index][0]):
                            video_placeholder += (
                                self.vision_start_token + video_prompt + self.vision_end_token
                            )

                        if f"{self.vision_start_token}{self.video_token}{self.vision_end_token}" in text[i]:
                            text[i] = text[i].replace(
                                f"{self.vision_start_token}{self.video_token}{self.vision_end_token}",
                                video_placeholder, 1,
                            )
                        else:
                            text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                        index += 1
                    text[i] = text[i].replace("<|placeholder|>", self.video_token)

            # VISUAL_LATENT: insert latent token before every vision_end_token
            for i in range(len(text)):
                text[i] = text[i].replace(vision_end_token, latent_token + vision_end_token)

            return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
            return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
            text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
            self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

            if return_mm_token_type_ids:
                import numpy as np
                array_ids = np.array(text_inputs["input_ids"])
                mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
                mm_token_type_ids[array_ids == self.image_token_id] = 1
                text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

            return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

        processor.__class__.__call__ = _patched_call_with_latent
        rank0_print(f"[Processor] Patched __call__ with VISUAL_LATENT insertion (latent_token={latent_token})")

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class() 
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        fix_mistral_regex=True
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        print(f'### Analyze the Model with Trainable Params ... ')
        param_count = 0        
        for name, p in model.named_parameters():
            if p.requires_grad:
                param_count += p.numel()
                rank0_print(f'Trainable Param: {name}')
        rank0_print(f"model's all trainable paramter is {param_count/ 1e6} M ")

    if torch.distributed.get_rank() == 0:
        print(f'### Start to build dataset ... ')
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    
    if torch.distributed.get_rank() == 0:
        print(f'### Finish to build dataset ... ')
    
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )
    
    if training_args.save_after_load:
        # Save the freshly-loaded model (no aux head init; aux losses are pretraining-only)
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)
        exit()

    if torch.distributed.get_rank() == 0:
        print(f'### train args: {training_args}')
        print(f"is_deepspeed_enabled = {trainer.is_deepspeed_enabled}")
        print(f"args.deepspeed = {trainer.args.deepspeed}")

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
