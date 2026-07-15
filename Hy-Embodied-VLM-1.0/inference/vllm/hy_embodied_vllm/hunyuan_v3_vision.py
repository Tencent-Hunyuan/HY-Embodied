# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# coding=utf-8
# Copyright 2025 The HunYuan team.
# Copyright 2025 The vLLM team.
# Copyright 2025 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only HunYuan-VL model compatible with HuggingFace weights."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import PromptReplacement, PromptUpdate, PromptUpdateDetails
from vllm.sequence import IntermediateTensors

from .hunyuan_vision import (
    HunYuanVisionTransformer,
    HunYuanVLDummyInputsBuilder,
    HunYuanVLImageEmbeddingInputs,
    HunYuanVLImageInputs,
    HunYuanVLImagePixelInputs,
    HunYuanVLMultiModalProcessor,
    HunYuanVLProcessingInfo,
    HunYuanVLVideoEmbeddingInputs,
    HunYuanVLVideoInputs,
    HunYuanVLVideoPixelInputs,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
    SupportsXDRoPE,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

logger = init_logger(__name__)


class HYV3VLProcessingInfo(HunYuanVLProcessingInfo):
    def get_hf_config(self):
        # Pass no type argument so vLLM accepts any PretrainedConfig subclass
        # — needed so the plugin works with the trust_remote_code-loaded
        # config (a `transformers_modules.<hash>.HYV3VLConfig` instance
        # which has a different class identity than any plugin-local class).
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object):
        # Likewise no processor-class argument — vLLM defaults to
        # `ProcessorMixin`, which trust_remote_code's HYV3VLProcessor
        # correctly subclasses.
        return self.ctx.get_hf_processor(
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )


class HYV3VLMultiModalProcessor(HunYuanVLMultiModalProcessor):
    # Runtime info is injected by MULTIMODAL_REGISTRY via
    # @register_processor(..., info=HYV3VLProcessingInfo, ...).
    # Keep behavior from HunYuanVLMultiModalProcessor, but make the binding
    # explicit for readability and type checking.
    info: HYV3VLProcessingInfo

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        hf_config = self.info.get_hf_config()
        cat_extra_token = bool(hf_config.vision_config.cat_extra_token)

        image_start_token_id = hf_processor.image_start_token_id
        image_end_token_id = hf_processor.image_end_token_id
        placeholder = {
            "image": hf_processor.image_token_id,
            "video": hf_processor.video_token_id,
        }

        merge_size = image_processor.merge_size
        spatial_patch_size = image_processor.spatial_patch_size
        temporal_patch_size = image_processor.temporal_patch_size

        def _get_video_timestamps(
            item_idx: int,
            num_merged_frames: int,
        ) -> list[float]:
            second_per_grid_ts = hf_processor_mm_kwargs.get("second_per_grid_ts")
            if isinstance(second_per_grid_ts, (list, tuple)) and item_idx < len(
                second_per_grid_ts
            ):
                second_per_grid_t = float(second_per_grid_ts[item_idx])
                return [i * second_per_grid_t for i in range(num_merged_frames)]

            video_item = mm_items["video"][item_idx]
            metadata = (
                video_item[1]
                if isinstance(video_item, tuple) and len(video_item) > 1
                else {}
            )
            fps = metadata.get("fps")
            frame_indices = metadata.get("frames_indices")

            if fps and frame_indices:
                if not isinstance(frame_indices, list):
                    frame_indices = list(frame_indices)
                if frame_indices:
                    if len(frame_indices) % temporal_patch_size != 0:
                        frame_indices = frame_indices + [frame_indices[-1]] * (
                            temporal_patch_size
                            - len(frame_indices) % temporal_patch_size
                        )
                    timestamps = []
                    for i in range(0, len(frame_indices), temporal_patch_size):
                        group = frame_indices[i : i + temporal_patch_size]
                        timestamps.append(sum(group) / len(group) / float(fps))
                    if len(timestamps) >= num_merged_frames:
                        return timestamps[:num_merged_frames]

            return [float(i) for i in range(num_merged_frames)]

        def get_replacement_hyv3vl(item_idx: int, modality: str):
            out_item = out_mm_kwargs[modality][item_idx]
            grid_thw = out_item[f"{modality}_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            grid_t, grid_h, grid_w = grid_thw
            if modality == "image":
                num_tokens = (int(grid_h) // merge_size) * (
                    int(grid_w) // merge_size + 1
                )
                if cat_extra_token:
                    num_tokens += 2
                return [placeholder[modality]] * num_tokens

            merge_video_size = merge_size * spatial_patch_size
            num_tokens = (int(grid_h) // merge_video_size) * (
                int(grid_w) // merge_video_size + 1
            )
            if cat_extra_token:
                num_tokens += 2

            num_merged_frames = (
                int(grid_t) + temporal_patch_size - 1
            ) // temporal_patch_size
            timestamps = _get_video_timestamps(item_idx, num_merged_frames)
            frames_idx_token = [
                tokenizer.encode(f"<{curr_time:.1f} seconds>", add_special_tokens=False)
                for curr_time in timestamps
            ]

            placeholder_tokens: list[int] = []
            for frame_tokens in frames_idx_token:
                placeholder_tokens.extend(frame_tokens)
                placeholder_tokens.extend(
                    [image_start_token_id]
                    + [placeholder[modality]] * num_tokens
                    + [image_end_token_id]
                )
            return PromptUpdateDetails.select_token_id(
                placeholder_tokens, placeholder[modality]
            )

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=lambda item_idx, m=modality: get_replacement_hyv3vl(
                    item_idx, m
                ),
            )
            for modality in ("image", "video")
        ]


class HYV3VLDummyInputsBuilder(HunYuanVLDummyInputsBuilder):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    HYV3VLMultiModalProcessor,
    info=HYV3VLProcessingInfo,
    dummy_inputs=HYV3VLDummyInputsBuilder,
)
class HYV3VLForConditionalGeneration(
    nn.Module,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
    SupportsXDRoPE,
):
    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "vit.vit.": "visual.",
            "vit.": "visual.",
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    supports_encoder_tp_data = True

    def get_xdrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> torch.Tensor:
        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {"image_grid_thw"},
        )
        image_grid_thw = [item.tolist() for item in kwargs.get("image_grid_thw", [])]
        kwargs_video = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {"video_grid_thw"},
        )
        video_grid_thw = [
            item.tolist() for item in kwargs_video.get("video_grid_thw", [])
        ]
        hf_config = self.config
        image_start_token_id = hf_config.image_start_token_id
        image_end_token_id = hf_config.image_end_token_id  # noqa: F841
        video_token_id = hf_config.video_token_id  # noqa: F841
        video_start_token_id = hf_config.video_start_token_id
        video_end_token_id = hf_config.video_end_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        cat_extra_token = bool(hf_config.vision_config.cat_extra_token)
        temporal_patch_size = hf_config.vision_config.temporal_patch_size
        spatial_patch_size = hf_config.vision_config.spatial_patch_size

        positions = torch.arange(len(input_tokens))
        positions_w = torch.arange(len(input_tokens))
        positions_h = torch.arange(len(input_tokens))
        positions_t = torch.arange(len(input_tokens))

        xd_num = len(hf_config.rope_scaling["xdrope_section"])
        input_tokens_tensor = torch.tensor(input_tokens)
        image_start_token_pos_indices = torch.where(
            input_tokens_tensor == image_start_token_id
        )[0]
        image_index = -1
        video_index = -1
        mm_index = 0
        inside_video = False
        token_scan_pos = 0
        while mm_index < len(image_start_token_pos_indices):
            start_index = int(image_start_token_pos_indices[mm_index])
            while token_scan_pos < start_index:
                curr_token = int(input_tokens_tensor[token_scan_pos])
                if curr_token == video_start_token_id:
                    inside_video = True
                elif curr_token == video_end_token_id:
                    inside_video = False
                token_scan_pos += 1

            is_video = inside_video
            if (
                not is_video
                and image_index + 1 >= len(image_grid_thw)
                and video_index + 1 < len(video_grid_thw)
            ):
                is_video = True

            if is_video:
                video_index += 1
                if video_index >= len(video_grid_thw):
                    raise ValueError(
                        "Mismatch between video placeholders and video_grid_thw: "
                        f"video_index={video_index}, "
                        f"total_videos={len(video_grid_thw)}."
                    )
                t, h, w = video_grid_thw[video_index]
                llm_grid_t = (t + temporal_patch_size - 1) // temporal_patch_size
                llm_grid_h = h // spatial_merge_size // spatial_patch_size
                llm_grid_w = w // spatial_merge_size // spatial_patch_size
                image_tokens_num = (llm_grid_w + 1) * llm_grid_h
                frame_count = min(
                    llm_grid_t, len(image_start_token_pos_indices) - mm_index
                )
                for offset in range(frame_count):
                    start_index = image_start_token_pos_indices[mm_index + offset] + 1
                    if cat_extra_token:
                        start_index += 1
                    positions_w[start_index : start_index + image_tokens_num] = (
                        torch.tensor(
                            list(range(llm_grid_w + 1)) * llm_grid_h,
                            dtype=positions.dtype,
                        )
                    )
                    positions_h[start_index : start_index + image_tokens_num] = (
                        torch.arange(llm_grid_h)
                        .unsqueeze(-1)
                        .expand(-1, llm_grid_w + 1)
                        .reshape(-1)
                    )
                    positions_t[start_index : start_index + image_tokens_num] = (
                        mm_index + offset
                    )
                mm_index += frame_count
            else:
                image_index += 1
                if image_index >= len(image_grid_thw):
                    raise ValueError(
                        "Mismatch between image placeholders and image_grid_thw: "
                        f"image_index={image_index}, "
                        f"total_images={len(image_grid_thw)}."
                    )
                t, h, w = image_grid_thw[image_index]
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t,
                    h // spatial_merge_size,
                    w // spatial_merge_size,
                )

                image_tokens_num = (llm_grid_w + 1) * llm_grid_h
                start_index = image_start_token_pos_indices[mm_index] + 1
                if cat_extra_token:
                    start_index += 1
                positions_w[start_index : start_index + image_tokens_num] = (
                    torch.tensor(
                        list(range(llm_grid_w + 1)) * llm_grid_h, dtype=positions.dtype
                    )
                )
                positions_h[start_index : start_index + image_tokens_num] = (
                    torch.arange(llm_grid_h)
                    .unsqueeze(-1)
                    .expand(-1, llm_grid_w + 1)
                    .reshape(-1)
                )
                positions_t[start_index : start_index + image_tokens_num] = mm_index
                mm_index += 1

        if xd_num == 4:
            llm_positions = torch.stack(
                [positions, positions_w, positions_h, positions_t]
            )
        elif xd_num == 3:
            llm_positions = torch.stack([positions_w, positions_h, positions_t])

        return llm_positions

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {"image_grid_thw", "video_grid_thw", "second_per_grid_ts"},
        )
        image_grid_thw = [item.tolist() for item in kwargs.get("image_grid_thw", [])]
        video_grid_thw = [item.tolist() for item in kwargs.get("video_grid_thw", [])]
        second_per_grid_ts = kwargs.get("second_per_grid_ts", [])  # noqa: F841

        hf_config = self.config
        image_token_id = hf_config.image_token_id  # noqa: F841
        video_token_id = hf_config.video_token_id  # noqa: F841
        image_start_token_id = hf_config.image_start_token_id
        image_end_token_id = hf_config.image_end_token_id
        video_start_token_id = hf_config.video_start_token_id
        video_end_token_id = hf_config.video_end_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        cat_extra_token = bool(hf_config.vision_config.cat_extra_token)
        tokens_per_second = getattr(  # noqa: F841
            hf_config.vision_config, "tokens_per_second", 1.0
        )
        temporal_patch_size = hf_config.vision_config.temporal_patch_size
        spatial_patch_size = hf_config.vision_config.spatial_patch_size

        input_tokens_tensor = torch.tensor(input_tokens)
        image_start_indices = torch.argwhere(
            input_tokens_tensor == image_start_token_id
        ).squeeze(1)

        t_index = torch.arange(len(input_tokens_tensor))
        h_index = torch.arange(len(input_tokens_tensor))
        w_index = torch.arange(len(input_tokens_tensor))
        start_pos = 0
        start_index = 0
        image_index = -1
        video_index = -1
        mm_index = 0
        inside_video = False
        token_scan_pos = 0

        while mm_index < len(image_start_indices):
            pos = image_start_indices[mm_index] + 1
            if cat_extra_token:
                pos += 1

            # For frames_idx_token
            start_marker_pos = int(image_start_indices[mm_index])
            while token_scan_pos < start_marker_pos:
                curr_token = int(input_tokens_tensor[token_scan_pos])
                if curr_token == video_start_token_id:
                    inside_video = True
                elif curr_token == video_end_token_id:
                    inside_video = False
                token_scan_pos += 1

            is_video = inside_video
            if (
                not is_video
                and image_index + 1 >= len(image_grid_thw)
                and video_index + 1 < len(video_grid_thw)
            ):
                is_video = True

            if start_pos < pos:
                text_index = torch.arange(pos - start_pos) + start_index
                t_index[start_pos:pos] = text_index
                h_index[start_pos:pos] = text_index
                w_index[start_pos:pos] = text_index
                start_index += pos - start_pos

            if is_video:
                image_end_indices = torch.argwhere(
                    input_tokens_tensor == image_end_token_id
                ).squeeze(1)
                video_index += 1
                if video_index >= len(video_grid_thw):
                    raise ValueError(
                        "Mismatch between video placeholders and video_grid_thw: "
                        f"video_index={video_index}, "
                        f"total_videos={len(video_grid_thw)}."
                    )
                t, h, w = video_grid_thw[video_index]
                llm_grid_t = (t + temporal_patch_size - 1) // temporal_patch_size
                llm_grid_h = h // spatial_merge_size // spatial_patch_size
                llm_grid_w = w // spatial_merge_size // spatial_patch_size
                token_num = (llm_grid_w + 1) * llm_grid_h
                frame_count = min(llm_grid_t, len(image_start_indices) - mm_index)

                # For each frame of the video
                for frame_idx in range(frame_count):
                    frame_pos = image_start_indices[mm_index + frame_idx] + 1
                    if cat_extra_token:
                        frame_pos += 1
                    curr_pos = frame_pos + token_num

                    w_index[frame_pos:curr_pos].copy_(
                        torch.arange(0, llm_grid_w + 1)
                        .reshape(1, -1)
                        .expand(llm_grid_h, -1)
                        .reshape(-1)
                        + start_index
                    )
                    h_index[frame_pos:curr_pos].copy_(
                        torch.arange(0, llm_grid_h)
                        .reshape(-1, 1)
                        .expand(-1, llm_grid_w + 1)
                        .reshape(-1)
                        + start_index
                    )
                    t_index[frame_pos:curr_pos] = start_index

                    frame_max = int(
                        torch.stack(
                            [
                                t_index[frame_pos:curr_pos].max(),
                                h_index[frame_pos:curr_pos].max(),
                                w_index[frame_pos:curr_pos].max(),
                            ]
                        )
                        .max()
                        .item()
                    )
                    start_index = frame_max + 1

                    # Update temporal token position ids
                    if frame_idx + 1 < frame_count:
                        curr_end_pos = int(image_end_indices[mm_index + frame_idx])
                        next_start_pos = (
                            int(image_start_indices[mm_index + frame_idx + 1]) + 1
                        )
                        if curr_end_pos < next_start_pos:
                            bridge_index = (
                                torch.arange(
                                    next_start_pos - curr_end_pos,
                                    dtype=t_index.dtype,
                                    device=t_index.device,
                                )
                                + start_index
                            )
                            t_index[curr_end_pos:next_start_pos] = bridge_index
                            h_index[curr_end_pos:next_start_pos] = bridge_index
                            w_index[curr_end_pos:next_start_pos] = bridge_index
                            start_index += next_start_pos - curr_end_pos

                last_end_pos = int(image_end_indices[mm_index + frame_count - 1])
                start_pos = last_end_pos
                mm_index += frame_count
            else:
                image_index += 1
                if image_index >= len(image_grid_thw):
                    raise ValueError(
                        "Mismatch between image placeholders and image_grid_thw: "
                        f"image_index={image_index}, "
                        f"total_images={len(image_grid_thw)}."
                    )
                t, h, w = image_grid_thw[image_index]
                _, llm_grid_h, llm_grid_w = (
                    t,
                    h // spatial_merge_size,
                    w // spatial_merge_size,
                )

                token_num = (llm_grid_w + 1) * llm_grid_h
                w_index[pos : pos + token_num].copy_(
                    torch.arange(0, llm_grid_w + 1)
                    .reshape(1, -1)
                    .expand(llm_grid_h, -1)
                    .reshape(-1)
                    + start_index
                )
                h_index[pos : pos + token_num].copy_(
                    torch.arange(0, llm_grid_h)
                    .reshape(-1, 1)
                    .expand(-1, llm_grid_w + 1)
                    .reshape(-1)
                    + start_index
                )
                t_index[pos : pos + token_num] = start_index

                start_pos = pos + token_num
                start_index += max(llm_grid_h, llm_grid_w + 1)
                mm_index += 1

        if start_pos < len(input_tokens_tensor):
            text_index = (
                torch.arange(len(input_tokens_tensor) - start_pos) + start_index
            )
            t_index[start_pos:] = text_index
            h_index[start_pos:] = text_index
            w_index[start_pos:] = text_index

        llm_positions = torch.stack([t_index, h_index, w_index])
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()

        return llm_positions, mrope_position_delta

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<｜hy_image▁start｜><｜hy_image▁pad｜><｜hy_image▁end｜>"
        if modality.startswith("video"):
            return "<｜hy_video▁start｜><｜hy_video▁pad｜><｜hy_video▁end｜>"

        raise ValueError("Only image/video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        if multimodal_config.get_limit_per_prompt(
            "image"
        ) or multimodal_config.get_limit_per_prompt("video"):
            attn_backend_override = (
                multimodal_config.mm_encoder_attn_backend
                if multimodal_config is not None
                else None
            )
            vit_quant_config = self._maybe_ignore_quant_config(self.quant_config)
            self.visual = HunYuanVisionTransformer(
                config.vision_config,
                quant_config=vit_quant_config,
                prefix=maybe_prefix(prefix, "visual"),
                multimodal_config=multimodal_config,
                attn_backend_override=attn_backend_override,
            )
        else:
            self.visual = None

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model.model"),
            architectures=[
                "HYV3ForCausalLM",
            ],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _maybe_ignore_quant_config(
        self, quant_config: QuantizationConfig | None = None
    ):
        # GPTQ configs do not have a list of ignored modules, however AutoGPTQ
        # seems to avoid vision encoder sections for some models.
        if isinstance(
            quant_config,
            (GPTQConfig, GPTQMarlinConfig, Fp8Config, CompressedTensorsConfig),
        ):
            return None

        return quant_config

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> HunYuanVLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        # TODO: refine
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)
        if len(pixel_values.shape) == 3:
            last_dim = pixel_values.shape[-1]
            pixel_values = pixel_values.reshape(-1, last_dim)
            image_grid_thw = image_grid_thw.reshape(-1, 3)

        if pixel_values is not None:
            return HunYuanVLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return HunYuanVLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> HunYuanVLVideoInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values"
            )
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw"
            )

            return HunYuanVLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

        if video_embeds is not None:
            video_embeds = self._validate_and_reshape_mm_tensor(
                video_embeds, "video embeds"
            )
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw"
            )

            if not isinstance(video_embeds, torch.Tensor):
                raise ValueError(
                    "Incorrect type of video embeddings. "
                    f"Got type: {type(video_embeds)}"
                )
            return HunYuanVLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
            )

    def _validate_and_reshape_mm_tensor(
        self, mm_input: object, name: str
    ) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(
                    f"{name} should be 2D or batched 3D tensor. "
                    f"Got ndim: {mm_input.ndim} "
                    f"(shape={mm_input.shape})"
                )
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _process_image_input(
        self, image_input: HunYuanVLImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"]

            # TODO: use_data_parallel (split image_embeds in visual)
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)

        return image_embeds

    def _process_video_input(
        self, video_input: HunYuanVLVideoInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"]
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw_list)

        return video_embeds

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(
                    **kwargs
                )
        return mm_input_by_modality

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += tuple(video_embeddings)
        return multimodal_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model.model",
            connector="visual.perceive",
            tower_model="visual",
        )
