# coding=utf-8
# Copyright (C) 2025 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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

"""Modular definition for HunYuanVL-MoT (Mixture of Transformers) multimodal model.

Key design decisions:
- Language decoder (HunYuanVLMoTAttention, HunYuanVLMoTDecoderLayer) has dual
  text/vision projection paths controlled by `modality_mask`.
- Flash Attention 2 via `flash_attn_varlen_func` is **required** (hard dependency)
  because the MoT mechanism calls it twice: causal=True for text, causal=False
  for vision segments.
- Vision encoder wraps HYViT2_400MAnyRes defined in `modeling_hunyuan_vl_mot.py`.
- Weight keys use `_v` suffix for vision modules
  (e.g. `q_proj_v`, `mlp_v`, `input_layernorm_v`).
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs

from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
    HunYuanDenseV1RMSNorm as HunYuanVLMoTRMSNorm,          # weight: layernorm.weight
    HunYuanDenseV1MLP as HunYuanVLMoTMLP,                   # weight: mlp.{gate,up,down}_proj.weight
    HunYuanDenseV1RotaryEmbedding as HunYuanVLMoTRotaryEmbedding,
)

from .configuration_hunyuan_vl_mot import HunYuanVLMoTConfig

logger = logging.getLogger(__name__)

HY_VL_MOT_IMAGE_TOKEN_ID = 120687
HY_VL_MOT_VIDEO_TOKEN_ID = 120688
HY_VL_MOT_LATENT_TOKEN_ID = 120690

# ============================================================================
# Flash Attention — required for MoT varlen mechanism
# ============================================================================
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False


def _check_flash_attn():
    if not _FLASH_ATTN_AVAILABLE:
        raise ImportError(
            "flash-attn is required for HunYuanVL-MoT. The Mixture of Transformers mechanism uses "
            "variable-length flash attention with per-modality causal masking.\n"
            "Install it with:  pip install flash-attn>=2.0"
        )


# ============================================================================
# Output dataclass
# ============================================================================

@dataclass
@auto_docstring(custom_intro="Base class for HunYuanVLMoT outputs.")
class HunYuanVLMoTModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*):
        Pre-computed hidden-states for fast sequential decoding.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


# ============================================================================
# MoT (Mixture of Transformers) helpers — new to this model
# ============================================================================

def _mask_apply(hidden_states: torch.Tensor, mask: torch.Tensor, text_funcs, vision_funcs, out_dims=None):
    """Route tokens to modality-specific functions.
    hidden_states: (B, S, D), mask: (B, S) bool — True = vision token.
    """
    B, S, D = hidden_states.size()
    flat = hidden_states.reshape(B * S, D)
    mask_flat = mask.reshape(B * S).bool()

    if out_dims is None:
        out_flat = [torch.empty_like(flat) for _ in text_funcs]
    else:
        out_flat = [torch.empty(B * S, od, device=flat.device, dtype=flat.dtype) for od in out_dims]

    placeholder = hidden_states[0:1, 0:1, :]
    zero_feature = 0

    text_idx = ~mask_flat
    if text_idx.any():
        hs_t = flat[text_idx]
        for i, fn in enumerate(text_funcs):
            out_flat[i][text_idx] = fn(hs_t)
    else:
        for fn in text_funcs:
            zero_feature = zero_feature + fn(placeholder).mean() * 0

    vis_idx = mask_flat
    if vis_idx.any():
        hs_v = flat[vis_idx]
        for i, fn in enumerate(vision_funcs):
            out_flat[i][vis_idx] = fn(hs_v)
    else:
        for fn in vision_funcs:
            zero_feature = zero_feature + fn(placeholder).mean() * 0

    result = [o.view(B, S, -1) for o in out_flat]
    result[0] = result[0] + zero_feature
    return result


def _flash_attention_forward_mot(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """Varlen flash attention with per-modality causal masks.
    Text tokens: causal=True. Vision tokens: causal=False.
    """
    _check_flash_attn()

    if kwargs.get("output_attentions", False):
        logger.warning_once("`flash_attention_2` does not support `output_attentions=True`.")

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(m for m in module.modules() if isinstance(m, nn.Linear)).weight.dtype
        query, key, value = query.to(target_dtype), key.to(target_dtype), value.to(target_dtype)

    query = query.squeeze(0)
    key = key.squeeze(0)
    value = value.squeeze(0)

    cu_seqlens_q = torch.tensor([0, query.shape[0]], dtype=torch.int32, device=query.device)
    cu_seqlens_k = torch.tensor([0, key.shape[0]], dtype=torch.int32, device=query.device)
    v_seqlens = attention_mask["v_seqlens"]

    with torch.no_grad():
        max_seqlen_q = max(cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(cu_seqlens_q.size(0) - 1)).item()
        max_seqlen_k = max(cu_seqlens_k[i + 1] - cu_seqlens_k[i] for i in range(cu_seqlens_k.size(0) - 1)).item()

    # Text path: causal attention
    attn_output = flash_attn_varlen_func(
        query, key, value,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        causal=True,
    )

    # Vision path: non-causal attention over visual segments
    if not (v_seqlens == 0).all():
        fake_visual = len(v_seqlens) == 0
        if fake_visual:
            v_seqlens = [(0, 2)]

        visual_query, visual_key, visual_value = [], [], []
        visual_mask = torch.zeros(query.shape[0], dtype=torch.bool, device=query.device)
        cu_v = [0]
        max_v_len = 0
        for s, e in v_seqlens:
            visual_query.append(query[s:e])
            visual_key.append(key[s:e])
            visual_value.append(value[s:e])
            visual_mask[s:e] = True
            cu_v.append(cu_v[-1] + (e - s))
            max_v_len = max(max_v_len, e - s)

        vq = torch.cat(visual_query, dim=0)
        vk = torch.cat(visual_key, dim=0)
        vv = torch.cat(visual_value, dim=0)
        cu_v_seqlens = torch.tensor(cu_v, device=query.device, dtype=torch.int32)

        visual_attn_out = flash_attn_varlen_func(
            vq, vk, vv,
            cu_seqlens_q=cu_v_seqlens, cu_seqlens_k=cu_v_seqlens,
            max_seqlen_q=max_v_len, max_seqlen_k=max_v_len,
            causal=False,
        )
        if fake_visual:
            attn_output = attn_output + visual_attn_out.mean() * 0
        else:
            attn_output = attn_output.clone()
            attn_output[visual_mask] = visual_attn_out

    return attn_output.unsqueeze(0), None


def _modality_mask_to_segments(mask: torch.Tensor) -> torch.Tensor:
    """Convert a boolean modality mask to (start, end) visual segment pairs."""
    if mask.size(1) == 1:
        return torch.tensor([[0, 0]], device=mask.device)
    if mask.dim() == 2:
        if mask.size(0) != 1:
            raise ValueError("Batch size > 1 not supported")
        mask = mask[0]
    mask = mask.to(torch.int64)
    slen = mask.numel()
    is_zero = (mask == 0).to(torch.int64)
    padded = torch.cat([
        torch.zeros(1, device=mask.device, dtype=torch.int64),
        is_zero,
        torch.zeros(1, device=mask.device, dtype=torch.int64),
    ])
    diff = padded[1:] - padded[:-1]
    zero_starts = (diff == 1).nonzero(as_tuple=True)[0]
    zero_ends = (diff == -1).nonzero(as_tuple=True)[0] - 1
    separators = [(s.item(), e.item()) for s, e in zip(zero_starts, zero_ends) if e - s + 1 >= 2]
    segments = []
    seg_start = 0
    for s, e in separators:
        seg_end = s - 1
        if seg_end >= seg_start:
            segments.append([seg_start, seg_end])
        seg_start = e + 1
    if seg_start < slen:
        segments.append([seg_start, slen - 1])
    for i in range(len(segments)):
        segments[i][1] = segments[i][1] + 2
    return (torch.tensor(segments, device=mask.device)
            if segments else torch.zeros((0, 2), dtype=torch.long, device=mask.device))


# ============================================================================
# MoT Attention — dual text/vision projection paths
# ============================================================================

class HunYuanVLMoTAttention(nn.Module):
    """Multi-headed attention with per-modality text/vision projection paths.
    
    Weight keys:
    - Text path: q_proj, k_proj, v_proj, o_proj, query_layernorm, key_layernorm
    - Vision path: q_proj_v, k_proj_v, v_proj_v, o_proj_v
    """

    def __init__(self, config: HunYuanVLMoTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Text projections
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.query_layernorm = HunYuanVLMoTRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = HunYuanVLMoTRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Vision projections (_v suffix matches ckpt keys)
        self.q_proj_v = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj_v = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj_v = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj_v = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def _mask_apply(self, hidden_states, modality_mask, text_funcs, vision_funcs, out_dims=None):
        if modality_mask is None:
            return [text_funcs[0](hidden_states)]
        return _mask_apply(hidden_states, modality_mask, text_funcs, vision_funcs, out_dims)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, key_states, value_states = self._mask_apply(
            hidden_states, modality_mask,
            [self.q_proj, self.k_proj, self.v_proj],
            [self.q_proj_v, self.k_proj_v, self.v_proj_v],
            out_dims=[
                self.config.num_attention_heads * self.head_dim,
                self.config.num_key_value_heads * self.head_dim,
                self.config.num_key_value_heads * self.head_dim,
            ],
        )

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = _flash_attention_forward_mot(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self._mask_apply(attn_output, modality_mask, [self.o_proj], [self.o_proj_v])[0]
        return attn_output, attn_weights


# ============================================================================
# Decoder Layer — dual norm/MLP paths
# ============================================================================

class HunYuanVLMoTDecoderLayer(GradientCheckpointingLayer):
    """Transformer decoder layer with per-modality norm and MLP paths.
    
    Weight keys:
    - Text path: mlp, input_layernorm, post_attention_layernorm
    - Vision path: mlp_v, input_layernorm_v, post_attention_layernorm_v
    """

    def __init__(self, config: HunYuanVLMoTConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HunYuanVLMoTAttention(config=config, layer_idx=layer_idx)
        # Text paths
        self.mlp = HunYuanVLMoTMLP(config)
        self.input_layernorm = HunYuanVLMoTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HunYuanVLMoTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Vision paths
        self.mlp_v = HunYuanVLMoTMLP(config)
        self.input_layernorm_v = HunYuanVLMoTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_v = HunYuanVLMoTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        modality_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = _mask_apply(
            hidden_states, modality_mask,
            [self.input_layernorm], [self.input_layernorm_v],
        )[0]
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            modality_mask=modality_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = _mask_apply(
            hidden_states, modality_mask,
            [lambda x: self.mlp(self.post_attention_layernorm(x))],
            [lambda x: self.mlp_v(self.post_attention_layernorm_v(x))],
        )[0]
        hidden_states = residual + hidden_states
        return hidden_states


# ============================================================================
# Top-level model classes (see modeling_hunyuan_vl_mot.py for full implementations)
# ============================================================================

class HunYuanVLMoTPreTrainedModel(PreTrainedModel):
    config_class = HunYuanVLMoTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HunYuanVLMoTDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = False
    _can_compile_fullgraph = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class HunYuanVLMoTModel(HunYuanVLMoTPreTrainedModel):
    """HunYuanVL-MoT: HYViT2 vision encoder + MoT language decoder."""
    ...


class HunYuanVLMoTForConditionalGeneration(HunYuanVLMoTPreTrainedModel, GenerationMixin):
    """HunYuanVL-MoT with LM head for multimodal conditional generation."""
    ...
