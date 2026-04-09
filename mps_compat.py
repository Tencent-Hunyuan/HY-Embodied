"""Experimental Apple Silicon / PyTorch MPS compatibility helpers."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Iterable

import torch
import torch.nn.functional as F


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_default_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def _repeat_kv_heads(x: torch.Tensor, target_heads: int) -> torch.Tensor:
    if x.shape[1] == target_heads:
        return x
    if target_heads % x.shape[1] != 0:
        raise ValueError(f"Cannot expand {x.shape[1]} KV heads to {target_heads} query heads")
    repeats = target_heads // x.shape[1]
    return x.repeat_interleave(repeats, dim=1)


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    k = _repeat_kv_heads(k, q.shape[1])
    v = _repeat_kv_heads(v, q.shape[1])
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal, scale=softmax_scale)


def _flash_attn_func_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    **_: object,
) -> torch.Tensor:
    out = _sdpa(q, k, v, softmax_scale=softmax_scale, causal=causal)
    return out.transpose(1, 2).contiguous().to(q.dtype)


def _iter_varlen_slices(cu_seqlens: torch.Tensor) -> Iterable[tuple[int, int]]:
    for idx in range(cu_seqlens.numel() - 1):
        start = int(cu_seqlens[idx].item())
        end = int(cu_seqlens[idx + 1].item())
        yield start, end


def _flash_attn_varlen_func_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    **_: object,
) -> torch.Tensor:
    outputs = []
    for (q_start, q_end), (k_start, k_end) in zip(_iter_varlen_slices(cu_seqlens_q), _iter_varlen_slices(cu_seqlens_k)):
        q_chunk = q[q_start:q_end].unsqueeze(0)
        k_chunk = k[k_start:k_end].unsqueeze(0)
        v_chunk = v[k_start:k_end].unsqueeze(0)
        chunk_out = _sdpa(q_chunk, k_chunk, v_chunk, softmax_scale=softmax_scale, causal=causal)
        outputs.append(chunk_out.squeeze(0).transpose(0, 1).contiguous().to(q.dtype))
    if not outputs:
        return q.new_zeros((0, q.shape[1], q.shape[2]))
    return torch.cat(outputs, dim=0)


def _install_flash_attn_fallback() -> None:
    if "flash_attn" in sys.modules:
        return
    module = types.ModuleType("flash_attn")
    module.flash_attn_func = _flash_attn_func_fallback
    module.flash_attn_varlen_func = _flash_attn_varlen_func_fallback
    sys.modules["flash_attn"] = module


def enable_hunyuan_mps_support() -> None:
    """Install MPS-safe fallbacks before importing the Hunyuan model."""
    if getattr(enable_hunyuan_mps_support, "_enabled", False):
        return

    _install_flash_attn_fallback()
    modeling = importlib.import_module("transformers.models.hunyuan_vl_mot.modeling_hunyuan_vl_mot")

    def sample_positional_embedding(self, grid):
        pos_embed_shape = int(self.pos_embed.shape[1] ** 0.5)
        pe_2d = self.pos_embed[0].T.contiguous().view(1, -1, pos_embed_shape, pos_embed_shape)
        n, _ = grid.shape
        grid = grid.view(1, n, 1, 2)
        pos_embedding = F.grid_sample(
            pe_2d.float(), grid.float(), mode="bilinear", align_corners=False, padding_mode="border"
        )
        return pos_embedding.view(1, -1, n).to(dtype=self.pos_embed.dtype, device=self.pos_embed.device).transpose(1, 2)

    modeling._HYViT2VisionTransformer.sample_positional_embedding = sample_positional_embedding

    projector_cls = modeling._HYViT2MLPProjector
    original_forward = projector_cls.forward
    original_forward_list = projector_cls._forward_list

    def _cast_list(self, xs):
        target_dtype = self.proj1.weight.dtype
        return [x.to(target_dtype) for x in xs]

    def patched_forward_list(self, x, size):
        return original_forward_list(self, _cast_list(self, x), size)

    def patched_forward(self, x, size=(16, 16), x2=None, size2=(16, 16)):
        target_dtype = self.proj1.weight.dtype
        if isinstance(x, list):
            x = _cast_list(self, x)
            if x2 is not None:
                x2 = _cast_list(self, x2)
        else:
            x = x.to(target_dtype)
            if x2 is not None and not isinstance(x2, list):
                x2 = x2.to(target_dtype)
        return original_forward(self, x, size=size, x2=x2, size2=size2)

    projector_cls._forward_list = patched_forward_list
    projector_cls.forward = patched_forward
    enable_hunyuan_mps_support._enabled = True
