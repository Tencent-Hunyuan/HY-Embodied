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

from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature

from vllm.config import MultiModalConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.vision import get_vit_attn_backend
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    ImageItem,
    ModalityData,
    MultiModalDataDict,
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    VideoItem,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ImageSize,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from transformers.video_utils import VideoMetadata

from .hunyuan_vl_config import (
    HunYuanVLConfig,
    HunYuanVLVisionConfig,
)
from .hunyuan_vl_processor import HunYuanVLProcessor
from .hunyuan_vl_image_processor import smart_resize
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.v1.attention.backends.registry import AttentionBackendEnum

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

_MAX_FRAMES_PER_VIDEO = 16

# === Vision Inputs === #


class HunYuanVLImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - np: Number of patches
        - ni: Number of images
        - cps: Number of channels * patch_size * patch_size
    """

    type: Literal["pixel_values"]

    pixel_values: Annotated[
        torch.Tensor,
        TensorShape("np", "cps"),
    ]

    image_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("ni", 3),
    ]


class HunYuanVLImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - nf: Number of image features
        - hs: Hidden size
        - ni: Number of images
    """

    type: Literal["image_embeds"]

    image_embeds: Annotated[
        torch.Tensor,
        TensorShape("nf", "hs"),
    ]

    image_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("ni", 3),
    ]


HunYuanVLImageInputs: TypeAlias = (
    HunYuanVLImagePixelInputs | HunYuanVLImageEmbeddingInputs
)


class HunYuanVLVideoPixelInputs(TensorSchema):
    """
    Dimensions:
        - np: Number of patches
        - nv: Number of videos
        - ctps: Number of channels * temporal_patch_size * patch_size *
          patch_size
    """

    type: Literal["pixel_values_videos"]

    pixel_values_videos: Annotated[
        torch.Tensor,
        TensorShape("np", "ctps"),
    ]

    video_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("nv", 3),
    ]


class HunYuanVLVideoEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - nf: Number of video features
        - hs: Hidden size
        - nv: Number of videos
    """

    type: Literal["video_embeds"]

    video_embeds: Annotated[
        torch.Tensor,
        TensorShape("nf", "hs"),
    ]

    video_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("nv", 3),
    ]


HunYuanVLVideoInputs: TypeAlias = (
    HunYuanVLVideoPixelInputs | HunYuanVLVideoEmbeddingInputs
)

# === Vision Encoder === #


class HunYuanVisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = True,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.dense_h_to_4h = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.dense_h_to_4h",
            disable_tp=use_data_parallel,
        )
        self.dense_4h_to_h = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.dense_4h_to_h",
            disable_tp=use_data_parallel,
        )
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        x_up, _ = self.dense_h_to_4h(x)
        x_down, _ = self.dense_4h_to_h(self.act_fn(x_up))
        return x_down


class HunYuanVisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        # Per attention head and per partition values.
        self.tp_size = (
            1
            if use_data_parallel
            else parallel_state.get_tensor_model_parallel_world_size()
        )
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size
        )

        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )

        self.o_proj = RowParallelLinear(
            input_size=projection_size,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            disable_tp=use_data_parallel,
        )

        self.scale = self.hidden_size_per_attention_head**-0.5
        self.attn = MMEncoderAttention(
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            self.scale,
            prefix=f"{prefix}.attn",
            multimodal_config=multimodal_config,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,  # Only used for Flash Attention
    ) -> torch.Tensor:
        qkv, _ = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        out = self.attn(q, k, v, cu_seqlens, max_seqlen)
        output, _ = self.o_proj(out)
        return output


class HunYuanVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        norm_layer: Callable[[int], nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.input_layernorm = norm_layer(dim)
        self.post_attention_layernorm = norm_layer(dim)
        self.self_attn = HunYuanVisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.self_attn",
            use_data_parallel=use_data_parallel,
        )
        self.mlp = HunYuanVisionMLP(
            dim,
            mlp_hidden_dim,
            act_fn=act_fn,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,  # Only used for Flash Attention
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.input_layernorm(x),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class HunYuanVisionPatchEmbed(nn.Module):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.spatial_merge_size = config.spatial_merge_size
        self.interpolate_mode = config.interpolate_mode

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.max_num_patches = (config.max_image_size // self.patch_size) ** 2

        self.num_positions = self.max_num_patches + 1
        self.position_edge = int(self.num_positions**0.5)
        # first token is cls token, skip it
        self.position_embedding = nn.Embedding(
            self.num_positions, self.embed_dim, dtype=torch.float32
        )

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: list[list[int]]
    ) -> torch.Tensor:
        num_patches = pixel_values.size(0)
        pixel_values = pixel_values.reshape(
            num_patches, self.num_channels, self.patch_size, self.patch_size
        )

        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.squeeze(-1).squeeze(-1).unsqueeze(0)

        patch_pos_shape = (1, self.position_edge, self.position_edge, self.embed_dim)
        pos_embedding = (
            self.position_embedding.weight[1:, :]
            .reshape(patch_pos_shape)
            .permute(0, 3, 1, 2)
        )
        assert pos_embedding.dtype == torch.float32

        patch_pos_embed_list = []
        for grid in grid_thw:
            t0, h0, w0 = grid
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            h0, w0 = h0 + 0.1, w0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                pos_embedding,
                scale_factor=(h0 / self.position_edge, w0 / self.position_edge),
                mode=self.interpolate_mode,
                align_corners=False,
            )

            patch_pos_embed = (
                patch_pos_embed.reshape(self.embed_dim, -1)
                .transpose(0, 1)
                .unsqueeze(0)
                .to(patch_embeds.dtype)
            )
            patch_pos_embed = patch_pos_embed.repeat(1, t0, 1)
            patch_pos_embed_list.append(patch_pos_embed)

        patch_pos_embed = torch.cat(patch_pos_embed_list, dim=1)
        embeddings = patch_embeds + patch_pos_embed

        return embeddings


class HunYuanVisionPatchMerger(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_merge_size=2,
        spatial_patch_size=1,
        temporal_patch_size=1,
        rms_norm_eps=1e-5,
        cat_extra_token=True,
        perceive_pre_norm=True,
        perceive_post_norm=True,
        prefix="",
        projector_type=None,
        perceive_intermediate_size=None,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.cat_extra_token = cat_extra_token
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.projector_type = projector_type
        self.perceive_pre_norm = False if projector_type else perceive_pre_norm
        self.perceive_post_norm = False if projector_type else perceive_post_norm
        self.extra_mlp = True

        hidden_size = (
            in_channels * (spatial_merge_size**2)
            if perceive_intermediate_size is None
            else perceive_intermediate_size
        )
        embed_std = out_channels**-0.5

        if projector_type == "rmsnorm_mlp_adapter":
            self.norm = RMSNorm(in_channels, eps=rms_norm_eps)
            self.proj = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    hidden_size,
                    kernel_size=spatial_merge_size,
                    stride=spatial_merge_size,
                ),
                nn.GELU(),
                nn.Conv2d(hidden_size, out_channels, kernel_size=1),
            )
            self.image_newline = nn.Parameter(torch.randn(out_channels) * embed_std)
            self.extra_mlp = False
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels * spatial_merge_size,
                    kernel_size=spatial_merge_size,
                    stride=spatial_merge_size,
                ),
                nn.GELU(),
                nn.Conv2d(in_channels * spatial_merge_size, hidden_size, kernel_size=1),
            )
            self.mlp = nn.Linear(hidden_size, out_channels)
            self.image_newline = nn.Parameter(torch.randn(hidden_size) * embed_std)
            # Not used, compatible with old checkpoint
            self.image_sep = nn.Parameter(torch.randn(out_channels) * embed_std)

        if self.cat_extra_token:
            self.image_begin = nn.Parameter(torch.randn(out_channels) * embed_std)
            self.image_end = nn.Parameter(torch.randn(out_channels) * embed_std)

        self.before_rms = (
            RMSNorm(in_channels, eps=rms_norm_eps) if self.perceive_pre_norm else None
        )
        self.after_rms = (
            RMSNorm(out_channels, eps=rms_norm_eps) if self.perceive_post_norm else None
        )

    def forward(self, x, size=(16, 16)):
        if self.projector_type:
            x = self.norm(x)

        if self.perceive_pre_norm:
            x = self.before_rms(x)

        h, w = size
        dtype = x.dtype
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, h, w)

        x = self.proj(x)  # b,c,h,w
        b, c, h, w = x.shape
        if b > 1:
            # (b, c, h, w) -> (c, b, h, w)
            x = x.permute(1, 0, 2, 3)
            x = F.avg_pool3d(
                x,
                kernel_size=(self.temporal_patch_size, 1, 1),
                stride=(self.temporal_patch_size, 1, 1),
            )
            # (c, b, h, w) -> (b, c, h, w)
            x = x.permute(1, 0, 2, 3)
            x = F.avg_pool2d(
                x, kernel_size=self.spatial_patch_size, stride=self.spatial_patch_size
            )

        b, c, h, w = x.shape
        x = torch.cat(
            [x, self.image_newline.reshape(1, c, 1, 1).expand(b, c, h, 1).to(dtype)],
            dim=-1,
        )
        x = x.reshape(b, c, -1).permute(0, 2, 1)
        if self.extra_mlp:
            x = self.mlp(x)

        if self.cat_extra_token:
            begin = (
                self.image_begin.reshape(1, 1, -1).expand(b, 1, x.shape[-1]).to(dtype)
            )
            end = self.image_end.reshape(1, 1, -1).expand(b, 1, x.shape[-1]).to(dtype)
            x = torch.cat([begin, x, end], dim=1)

        b, c, d = x.shape
        x = x.reshape(1, b * c, d)

        if self.perceive_post_norm:
            return self.after_rms(x)
        return x


class HunYuanVisionTransformer(nn.Module):
    def __init__(
        self,
        vision_config: HunYuanVLVisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        multimodal_config: MultiModalConfig | None = None,
        attn_backend_override: AttentionBackendEnum | None = None,
    ) -> None:
        super().__init__()

        num_hidden_layers = vision_config.num_hidden_layers
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_attention_heads
        head_dim = self.hidden_size // self.num_heads
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_patch_size = vision_config.spatial_patch_size
        self.temporal_patch_size = vision_config.temporal_patch_size

        # Get device-specific vision attention backend.
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
            attn_backend_override=attn_backend_override,
        )

        from vllm.compilation.backends import set_model_tag

        with set_model_tag("HunYuanVisionPatchEmbed"):
            self.embeddings = HunYuanVisionPatchEmbed(vision_config)

        norm_layer = partial(nn.LayerNorm, eps=vision_config.rms_norm_eps)

        with set_model_tag("HunYuanVisionBlock"):
            self.layers = nn.ModuleList(
                [
                    HunYuanVisionBlock(
                        dim=vision_config.hidden_size,
                        num_heads=vision_config.num_attention_heads,
                        mlp_hidden_dim=vision_config.intermediate_size,
                        act_fn=get_act_fn(vision_config.hidden_act),
                        norm_layer=norm_layer,
                        quant_config=quant_config,
                        multimodal_config=multimodal_config,
                        prefix=f"{prefix}.layers.{layer_idx}",
                        use_data_parallel=use_data_parallel,
                    )
                    for layer_idx in range(num_hidden_layers)
                ]
            )

        with set_model_tag("HunYuanVisionPatchMerger"):
            self.perceive = HunYuanVisionPatchMerger(
                vision_config.hidden_size,
                vision_config.out_hidden_size,
                spatial_merge_size=vision_config.spatial_merge_size,
                spatial_patch_size=vision_config.spatial_patch_size,
                temporal_patch_size=vision_config.temporal_patch_size,
                rms_norm_eps=vision_config.rms_norm_eps,
                cat_extra_token=bool(vision_config.cat_extra_token),
                perceive_pre_norm=vision_config.perceive_pre_norm,
                perceive_post_norm=vision_config.perceive_post_norm,
                prefix=f"{prefix}.perceive",
                projector_type=getattr(vision_config, "projector_type", None),
                perceive_intermediate_size=getattr(
                    vision_config, "perceive_intermediate_size", None
                ),
            )

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.embeddings.patch_embedding.weight.device

    def compute_attn_mask_seqlen(
        self,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_seqlen = torch.zeros([], device=cu_seqlens.device)
        if self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
        }:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        # Attention backend 'XFORMERS' has been removed
        # elif self.attn_backend == AttentionBackendEnum.XFORMERS:
        #     seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        # return max_seqlen, seqlens
        return max_seqlen

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        # patchify
        seq_len = x.size(0)
        cu_seqlens: list = [0]

        hidden_states = x.to(device=self.device, dtype=self.dtype)
        # embeddings = patch_embeds + patch_pos_embed
        hidden_states = self.embeddings(hidden_states, grid_thw)

        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            for _ in range(t):
                cu_seqlens.append(h * w)

        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        cu_seqlens = torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32)

        max_seqlen_full = self.compute_attn_mask_seqlen(cu_seqlens)

        cu_seqlens = cu_seqlens.to(device=self.device, non_blocking=True)

        hidden_states = hidden_states.reshape(seq_len, -1)
        hidden_states = hidden_states.unsqueeze(0)

        for layer_num, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen_full,
            )

        # adapter
        split_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        split_items = hidden_states.split(split_lengths, dim=1)
        embeds_list = []
        split_index = 0
        for grid in grid_thw:
            split_item = list(split_items[split_index : split_index + grid[0]])
            if (grid[0] > 1) and (grid[0] % self.temporal_patch_size != 0):
                pad_size = self.temporal_patch_size - grid[0] % self.temporal_patch_size
                pad_item = [split_items[split_index + grid[0] - 1]] * pad_size
                split_item += pad_item
            split_item = torch.cat(split_item)
            split_index += grid[0]
            embeds_list.append(
                self.perceive(split_item.contiguous(), size=grid[1:]).squeeze(0)
            )

        return embeds_list

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv", ".q_proj", "q"),
            (".qkv", ".k_proj", "k"),
            (".qkv", ".v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


def _hunyuan_vl_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
    image_grid_sizes = image_grid_thw.prod(-1)
    video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
    video_grid_sizes = video_grid_thw.prod(-1)
    return dict(
        pixel_values=MultiModalFieldConfig.flat_from_sizes("image", image_grid_sizes),
        image_embeds=MultiModalFieldConfig.flat_from_sizes("image", image_grid_sizes),
        image_grid_thw=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
            "video", video_grid_sizes
        ),
        video_embeds=MultiModalFieldConfig.flat_from_sizes("video", video_grid_sizes),
        video_grid_thw=MultiModalFieldConfig.batched("video"),
    )


class HunYuanVLMultiModalDataParser(MultiModalDataParser):
    def __init__(self, **kwargs):
        super().__init__(video_needs_metadata=True, **kwargs)

    def _parse_image_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[ImageItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="image",
                required_fields={"image_embeds", "image_grid_thw"},
                fields_factory=_hunyuan_vl_field_config,
            )

        return super()._parse_image_data(data)

    def _parse_video_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[VideoItem],
    ):
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="video",
                required_fields={"video_embeds", "video_grid_thw"},
                fields_factory=_hunyuan_vl_field_config,
            )

        return super()._parse_video_data(data)


class HunYuanVLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        # Pass no type argument so vLLM accepts any PretrainedConfig subclass
        # — needed for trust_remote_code-loaded configs whose class identity
        # lives under `transformers_modules.<hash>` rather than this plugin.
        return self.ctx.get_hf_config()

    def get_hf_processor(
        self,
        **kwargs: object,
    ):
        return self.ctx.get_hf_processor(
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def get_image_processor(
        self,
        **kwargs: object,
    ):
        return self.get_hf_processor(**kwargs).image_processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        max_image_tokens = self.get_max_image_tokens()
        max_video_tokens = self.get_max_video_tokens(seq_len, mm_counts)
        return {"image": max_image_tokens, "video": max_video_tokens}

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 1,
        do_resize: bool = True,
        image_processor: HunYuanVLProcessor | None,
    ) -> tuple[ImageSize, int]:
        if image_processor is None:
            image_processor = self.get_image_processor()

        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        spatial_merge_size = vision_config.spatial_merge_size
        cat_extra_token = bool(hf_config.vision_config.cat_extra_token)

        if do_resize:
            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * spatial_merge_size,
                min_pixels=image_processor.min_pixels,
                max_pixels=image_processor.max_pixels,
            )
            preprocessed_size = ImageSize(width=resized_width, height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width, height=image_height)

        grid_t = num_frames
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_vision_tokens = (
            grid_t * grid_h // spatial_merge_size * (grid_w // spatial_merge_size + 1)
        )
        if cat_extra_token:
            num_vision_tokens += 2

        return preprocessed_size, num_vision_tokens

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor: HunYuanVLProcessor | None,
    ) -> int:
        _, num_image_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            image_processor=image_processor,
        )
        return num_image_tokens

    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
        image_processor: HunYuanVLProcessor | None,
    ) -> int:
        _, num_video_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=num_frames,
            image_processor=image_processor,
        )
        return num_video_tokens

    def get_image_size_with_most_features(self) -> ImageSize:
        image_processor = self.get_image_processor()
        max_pixels = image_processor.max_pixels
        max_edge = int(max_pixels**0.5)
        max_image_size, _ = self._get_vision_info(
            image_width=max_edge // 4,
            image_height=max_edge * 4,
            image_processor=None,
        )
        return max_image_size

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            image_processor=None,
        )

    def _get_max_video_frames(self, max_tokens: int) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        num_frames = 0
        while True:
            next_num_frames = num_frames + 1
            next_max_tokens = self.get_num_video_tokens(
                image_width=target_width,
                image_height=target_height,
                num_frames=next_num_frames,
                image_processor=None,
            )
            if next_max_tokens > max_tokens:
                break
            num_frames = next_num_frames

        return num_frames

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        max_images = mm_counts.get("image", 0)
        max_videos = mm_counts.get("video", 0)

        max_image_tokens = self.get_max_image_tokens() * max_images
        max_total_frames = self._get_max_video_frames(seq_len - max_image_tokens)
        max_frames_per_video = min(
            max_total_frames // max(max_videos, 1), _MAX_FRAMES_PER_VIDEO
        )

        return max(max_frames_per_video, 1)

    def get_max_video_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_video_tokens(
            image_width=target_width,
            image_height=target_height,
            num_frames=self.get_num_frames_with_most_features(seq_len, mm_counts),
            image_processor=None,
        )


class HunYuanVLDummyInputsBuilder(BaseDummyInputsBuilder[HunYuanVLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        hf_processor = self.info.get_hf_processor()
        image_token: str = hf_processor.image_token
        image_start_token: str = hf_processor.image_start_token
        image_end_token: str = hf_processor.image_end_token
        video_token: str = hf_processor.video_token
        video_start_token: str = hf_processor.video_start_token
        video_end_token: str = hf_processor.video_end_token
        if num_images > 0:
            image_str = image_start_token + image_token * num_images + image_end_token
        else:
            image_str = ""
        if num_videos > 0:
            video_str = video_start_token + video_token * num_videos + video_end_token
        else:
            video_str = ""

        return image_str + video_str

    def _get_dummy_videos(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        num_videos: int,
    ) -> list[VideoItem]:
        if num_videos == 0:
            return []

        video = np.full((num_frames, width, height, 3), 255, dtype=np.uint8)
        video_items: list[VideoItem] = []
        for _ in range(num_videos):
            video_metadata = {
                "fps": 2.0,
                "duration": num_frames / 2.0,
                "total_num_frames": num_frames,
                "frames_indices": list(range(num_frames)),
                "video_backend": "opencv",
                "do_sample_frames": False,
            }
            video_items.append((video.copy(), video_metadata))
        return video_items

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        target_width, target_height = self.info.get_image_size_with_most_features()

        target_num_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts
        )

        return {
            "image": self._get_dummy_images(
                width=target_width, height=target_height, num_images=num_images
            ),
            "video": self._get_dummy_videos(
                width=target_width,
                height=target_width,
                num_frames=target_num_frames,
                num_videos=num_videos,
            ),
        }


class HunYuanVLMultiModalProcessor(BaseMultiModalProcessor[HunYuanVLProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return HunYuanVLMultiModalDataParser()

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        processor = self.info.get_hf_processor(**mm_kwargs)

        if videos := mm_data.pop("videos", []):
            video_grid_thw_lst = []
            pixel_values_videos_lst = []

            for item in videos:
                video_array, metadata = item

                video_mm_kwargs = dict(**mm_kwargs)
                if "do_sample_frames" not in video_mm_kwargs:
                    video_mm_kwargs["do_sample_frames"] = metadata.get(
                        "do_sample_frames", False
                    )

                metadata = VideoMetadata(
                    **{k: metadata[k] for k in metadata if k != "do_sample_frames"}
                )

                video_mm_data = dict()
                video_mm_data["videos"] = [[video_array]]
                video_mm_data["video_metadata"] = [[metadata]]

                video_outputs = super()._call_hf_processor(
                    prompt=processor.video_start_token
                    + processor.video_token
                    + processor.video_end_token,
                    mm_data=video_mm_data,
                    mm_kwargs=video_mm_kwargs,
                    tok_kwargs=tok_kwargs,
                )
                input_ids = video_outputs.pop("input_ids")
                video_placeholder = processor.tokenizer.batch_decode(input_ids)[0]
                prompt = prompt.replace(
                    processor.video_start_token
                    + processor.video_token
                    + processor.video_end_token,
                    video_placeholder,
                    1,
                )

                video_grid_thw_lst.append(video_outputs["video_grid_thw"])
                pixel_values_videos_lst.append(
                    video_outputs["pixel_values_videos"]
                )
            video_outputs = dict(
                pixel_values_videos=torch.cat(pixel_values_videos_lst),
                video_grid_thw=torch.cat(video_grid_thw_lst),
            )
        else:
            video_outputs = dict()

        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        combined_outputs = dict(
            processed_outputs,
            **video_outputs,
        )
        return BatchFeature(combined_outputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
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

        def get_replacement_hunyuan_vl(item_idx: int, modality: str):
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
            if modality == "video":
                merge_video_size = merge_size * spatial_patch_size
                num_tokens = (int(grid_h) // merge_video_size) * (
                    int(grid_w) // merge_video_size + 1
                )
                if cat_extra_token:
                    num_tokens += 2
                img_tokens = (
                    [image_start_token_id]
                    + [placeholder[modality]] * num_tokens
                    + [image_end_token_id]
                )
                return img_tokens * (
                    (grid_t + temporal_patch_size - 1) // temporal_patch_size
                )

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=partial(get_replacement_hunyuan_vl, modality=modality),
            )
            for modality in ("image", "video")
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _hunyuan_vl_field_config(hf_inputs)


@MULTIMODAL_REGISTRY.register_processor(
    HunYuanVLMultiModalProcessor,
    info=HunYuanVLProcessingInfo,
    dummy_inputs=HunYuanVLDummyInputsBuilder,
)
class HunYuanVLForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
    SupportsXDRoPE,
    SupportsMRoPE,
):
    multimodal_cpu_fields = {"image_grid_thw", "video_grid_thw"}
    # Fused module mapping used by compressed-tensors to expand fused layer
    # names (e.g. qkv_proj) to their component shards when checking the
    # ignore list / target scheme map.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "vit.vit.": "visual.",
            "vit.": "visual.",
            "model.": "language_model.model.",
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
        video_end_token_id = hf_config.video_end_token_id  # noqa: F841
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
        while mm_index < len(image_start_token_pos_indices):
            start_index = image_start_token_pos_indices[mm_index]
            prev_token = input_tokens_tensor[start_index - 1]
            if prev_token == video_start_token_id:
                video_index += 1
                t, h, w = video_grid_thw[video_index]
                llm_grid_t = (t + temporal_patch_size - 1) // temporal_patch_size
                llm_grid_h = h // spatial_merge_size // spatial_patch_size
                llm_grid_w = w // spatial_merge_size // spatial_patch_size
                image_tokens_num = (llm_grid_w + 1) * llm_grid_h
                for offset in range(llm_grid_t):
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
                mm_index += llm_grid_t
            else:
                image_index += 1
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
        # TODO: support mrope for video modality
        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {"image_grid_thw", "video_grid_thw", "second_per_grid_ts"},
        )
        image_grid_thw = [item.tolist() for item in kwargs.get("image_grid_thw", [])]
        video_grid_thw = [item.tolist() for item in kwargs.get("video_grid_thw", [])]  # noqa: F841
        second_per_grid_ts = kwargs.get("second_per_grid_ts", [])  # noqa: F841

        hf_config = self.config
        image_token_id = hf_config.image_token_id  # noqa: F841
        video_token_id = hf_config.video_token_id  # noqa: F841
        image_start_token_id = hf_config.image_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        cat_extra_token = bool(hf_config.vision_config.cat_extra_token)
        tokens_per_second = getattr(  # noqa: F841
            hf_config.vision_config, "tokens_per_second", 1.0
        )

        input_tokens_tensor = torch.tensor(input_tokens)
        image_start_indices = torch.argwhere(
            input_tokens_tensor == image_start_token_id
        ).squeeze(1)

        t_index = torch.arange(len(input_tokens_tensor))
        h_index = torch.arange(len(input_tokens_tensor))
        w_index = torch.arange(len(input_tokens_tensor))
        start_pos = 0
        start_index = 0
        for image_index in range(len(image_start_indices)):
            # +1 : first image_token, +2: for xdrope positions
            pos = image_start_indices[image_index] + 1
            if cat_extra_token:
                pos += 1

            if start_pos < pos:
                text_index = torch.arange(pos - start_pos) + start_index
                t_index[start_pos:pos] = text_index
                h_index[start_pos:pos] = text_index
                w_index[start_pos:pos] = text_index
                start_index += pos - start_pos

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
        config: HunYuanVLConfig = vllm_config.model_config.hf_config
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
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=[
                "HunYuanDenseV1ForCausalLM",
                "HunYuanMoEV1ForCausalLM",
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
