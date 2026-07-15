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
"""Inference-only HYV3 model compatible with HuggingFace weights."""

import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def _is_moe(config: PretrainedConfig, layer_id: int = -1) -> bool:
    num_experts = getattr(config, "num_experts", None)
    # moe layer only if layer_id >= first_k_dense_replace, else mlp
    if layer_id >= 0 and layer_id < getattr(config, "first_k_dense_replace", 0):
        return False
    # For MTP model
    if layer_id >= getattr(config, "num_hidden_layers", 0):
        # MTP layer is moe layer by default
        mtp_type = getattr(config, "mtp_type", "moe")
        if mtp_type == "moe":
            return True
        elif mtp_type == "mlp":
            return False
    if isinstance(num_experts, int):
        return num_experts > 1
    if isinstance(num_experts, list) and num_experts:
        # Ensure all elements are integers before calling max.
        if all(isinstance(e, int) for e in num_experts):
            return max(num_experts) > 1
        else:
            return False
    return False


class HYV3Attention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        if hasattr(config, "head_dim") and config.head_dim:
            self.head_dim = config.head_dim
        elif hasattr(config, "attention_head_dim"):
            self.head_dim = config.attention_head_dim
        else:
            self.head_dim = self.hidden_size // self.total_num_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.qk_norm = getattr(config, "qk_norm", False)

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            quant_config=quant_config,
            bias=None,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=True,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Add qk-norm
        if self.qk_norm:
            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim).contiguous())
            k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim).contiguous())
            # Stock vllm mrope requires 2D q/k; flatten head dim before rotary_emb
            # and restore shape after so downstream self.attn sees the original layout.
            q_shape, k_shape = q.shape, k.shape
            q = q.reshape(q_shape[0], -1)
            k = k.reshape(k_shape[0], -1)
            q, k = self.rotary_emb(positions, q, k)
            q = q.reshape(q_shape)
            k = k.reshape(k_shape)
        else:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)

        return output


class HYV3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            reduce_results=reduce_results,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class HYV3SparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = -1,
        prefix: str = "",
        enable_eplb: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )
        self.batch_invariant = vllm_is_batch_invariant()

        # Get layer_id topk if config.num_experts_per_tok is a list
        if isinstance(config.num_experts_per_tok, list):
            assert layer_id >= 0
            assert len(config.num_experts_per_tok) > layer_id
            # Avoid mtp error
            if layer_id >= config.num_hidden_layers:
                top_k = config.num_experts_per_tok[0]
            else:
                top_k = config.num_experts_per_tok[layer_id]
        else:
            top_k = config.num_experts_per_tok

        # If it is moe, expert_hidden_dim is preferred
        intermediate_size = config.intermediate_size
        if config.expert_hidden_dim is not None:
            if isinstance(config.expert_hidden_dim, int):
                intermediate_size = config.expert_hidden_dim
            else:
                assert layer_id >= 0
                # Avoid mtp error
                if layer_id >= config.num_hidden_layers:
                    intermediate_size = config.expert_hidden_dim[0]
                else:
                    intermediate_size = config.expert_hidden_dim[layer_id]

        router_scaling_factor = getattr(config, "router_scaling_factor", 1.0)
        use_sigmoid = getattr(config, "moe_router_use_sigmoid", False)
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        if self.batch_invariant:
            self.gate = ReplicatedLinear(
                config.hidden_size,
                config.num_experts,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.gate",
            )
        else:
            self.gate = ReplicatedLinear(
                config.hidden_size,
                config.num_experts,
                bias=False,
                quant_config=None,
                params_dtype=torch.float32,
                prefix=f"{prefix}.gate",
            )

        # Get layer_id num_shared_experts if config.num_shared_experts is a list.
        if isinstance(config.num_shared_experts, list):
            assert layer_id >= 0
            assert len(config.num_shared_experts) > layer_id
            # Avoid mtp error
            if layer_id >= config.num_hidden_layers:
                num_shared_experts = config.num_shared_experts[0]
            else:
                num_shared_experts = config.num_shared_experts[layer_id]
        else:
            num_shared_experts = config.num_shared_experts

        self.shared_mlp = HYV3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size * num_shared_experts,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_mlp",
        )

        if not use_sigmoid:
            self.experts = SharedFusedMoE(
                shared_experts=self.shared_mlp,
                num_experts=self.n_routed_experts,
                top_k=top_k,
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                reduce_results=False,
                renormalize=config.route_norm,
                quant_config=quant_config,
                prefix=f"{prefix}.experts",
                enable_eplb=self.enable_eplb,
                num_redundant_experts=self.n_redundant_experts,
                routing_method_type=RoutingMethodType.Renormalize,
                routed_scaling_factor=router_scaling_factor,
            )
        else:
            self.expert_bias = nn.Parameter(torch.empty(config.num_experts))
            self.experts = SharedFusedMoE(
                shared_experts=self.shared_mlp,
                num_experts=self.n_routed_experts,
                top_k=top_k,
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                reduce_results=False,
                renormalize=config.route_norm,
                quant_config=quant_config,
                prefix=f"{prefix}.experts",
                scoring_func="sigmoid",
                enable_eplb=self.enable_eplb,
                use_grouped_topk=True,
                num_expert_group=1,
                topk_group=1,
                num_redundant_experts=self.n_redundant_experts,
                routing_method_type=None,
                routed_scaling_factor=router_scaling_factor,
                e_score_correction_bias=self.expert_bias,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        # Gate precision: default FP32.
        if self.batch_invariant:
            router_logits, _ = self.gate(hidden_states)
        else:
            router_logits, _ = self.gate(hidden_states.to(dtype=torch.float32))

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        if self.shared_mlp is not None:
            final_hidden_states = final_hidden_states[0] + final_hidden_states[1]

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(orig_shape)


class HYV3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_id: int = -1,
        enable_eplb: bool = False,
    ) -> None:
        super().__init__()
        assert layer_id >= 0
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        if isinstance(config.intermediate_size, int):
            self.intermediate_size = config.intermediate_size
        else:
            # Avoid mtp error
            if layer_id >= config.num_hidden_layers:
                self.intermediate_size = config.intermediate_size[0]
            else:
                self.intermediate_size = config.intermediate_size[layer_id]
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        self.self_attn = HYV3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )

        if _is_moe(config, layer_id=layer_id):
            self.mlp = HYV3SparseMoeBlock(
                config=config,
                quant_config=quant_config,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
                enable_eplb=enable_eplb,
            )
        else:
            self.mlp = HYV3MLP(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                bias=getattr(config, "mlp_bias", False),
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        idx: int = -1,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (xd, seq_len) if xdrope is enabled for hunyuan-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class HYV3Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        parallel_config = vllm_config.parallel_config
        enable_eplb = vllm_config.parallel_config.enable_eplb
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: HYV3DecoderLayer(
                config=config,
                layer_id=int(prefix.split(".")[-1]),
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                enable_eplb=enable_eplb,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _split_qkv_weight(self, qkv: torch.Tensor):
        num_attention_heads = self.config.num_attention_heads
        num_kv_heads = getattr(
            self.config, "num_key_value_heads", self.config.num_attention_heads
        )
        num_key_value_groups = num_attention_heads // num_kv_heads
        hidden_size = self.config.hidden_size

        if hasattr(self.config, "head_dim"):
            attention_head_dim = self.config.head_dim
        elif hasattr(self.config, "attention_head_dim"):
            attention_head_dim = self.config.attention_head_dim
        else:
            attention_head_dim = self.config.hidden_size // num_attention_heads

        if qkv.shape[-1] == hidden_size:
            qkv = qkv.reshape(
                num_kv_heads, num_key_value_groups + 2, attention_head_dim, hidden_size
            )
            q, k, v = torch.split(qkv, (num_key_value_groups, 1, 1), dim=1)
            q = q.reshape(-1, hidden_size)
            k = k.reshape(-1, hidden_size)
            v = v.reshape(-1, hidden_size)
        else:
            if self.quant_config.weight_block_size is not None:
                block_n, block_k = (
                    self.quant_config.weight_block_size[0],
                    self.quant_config.weight_block_size[1],
                )
                assert attention_head_dim % block_n == 0
                assert hidden_size % block_k == 0
                assert qkv.shape[-1] == hidden_size // block_k
                qkv = qkv.reshape(
                    num_kv_heads,
                    num_key_value_groups + 2,
                    attention_head_dim // block_n,
                    hidden_size // block_k,
                )
                q, k, v = torch.split(qkv, (num_key_value_groups, 1, 1), dim=1)
                q = q.reshape(-1, hidden_size // block_k)
                k = k.reshape(-1, hidden_size // block_k)
                v = v.reshape(-1, hidden_size // block_k)
        return torch.concat((q, k, v))

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        num_attention_heads = self.config.num_attention_heads
        num_kv_heads = getattr(
            self.config, "num_key_value_heads", self.config.num_attention_heads
        )
        split_params_mapping = [
            (".gate_up_proj", ".gate_and_up_proj", 2, [(1, 1), (0, 1)], None),
            (
                ".qkv_proj",
                ".qkv_proj",
                num_attention_heads + num_kv_heads * 2,
                [("q", num_attention_heads), ("k", num_kv_heads), ("v", num_kv_heads)],
                self._split_qkv_weight,
            ),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            # Deal with combined gate and up projection weights for MoE
            if ".mlp.experts." in name and ".gate_and_up_proj." in name:
                up, gate = loaded_weight.split(loaded_weight.shape[0] // 2, dim=0)
                weight_list = [
                    (name.replace("gate_and_up_proj", "gate_proj"), gate),
                    (name.replace("gate_and_up_proj", "up_proj"), up),
                ]
                loaded_params.update(self.load_weights(weight_list))
                continue
            if "rope_cache" in name:
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if "gate_proj_bias" in name:
                name = name.replace("gate_proj_bias", "gate_proj.bias")
            if "up_proj_bias" in name:
                name = name.replace("up_proj_bias", "up_proj.bias")
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                continue

            is_found = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                is_found = True
                break
            if is_found:
                continue

            for (
                param_name,
                weight_name,
                den,
                split_param,
                func,
            ) in split_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                assert loaded_weight.shape[0] % den == 0
                units = loaded_weight.shape[0] // den

                param = params_dict[name]
                weight_loader = param.weight_loader
                offset = 0
                for shard_id, num in split_param:
                    new_offset = offset + num * units
                    if func:
                        weight_loader(
                            param, func(loaded_weight)[offset:new_offset], shard_id
                        )
                    else:
                        weight_loader(param, loaded_weight[offset:new_offset], shard_id)
                    offset = new_offset

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    # this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    if name.endswith(".query_layernorm.weight"):
                        name = name.replace(".query_layernorm.weight", ".q_norm.weight")

                    if name.endswith(".key_layernorm.weight"):
                        name = name.replace(".key_layernorm.weight", ".k_norm.weight")

                    if name.endswith(".gate.wg.weight"):
                        name = name.replace(".gate.wg.weight", ".router.gate.weight")

                    if "mlp.router.gate." in name:
                        name = name.replace("router.gate.", "gate.")

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(positions, hidden_states, residual, idx=idx)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states = hidden_states + residual
        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        return hidden_states


class HYV3ForCausalLM(nn.Module, SupportsPP, SupportsLoRA, MixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.model = HYV3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.expert_weights = []
        self.num_expert_groups = 1
        self.moe_layers = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, HYV3DecoderLayer)
            if isinstance(layer.mlp, HYV3SparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            self.num_moe_layers = 0
            raise RuntimeError("No HYV3 MoE layer found in model.layers.")
            return

        self.num_moe_layers = len(self.moe_layers)
        self.num_logical_experts = getattr(example_layer, "n_logical_experts", None)
        self.num_physical_experts = getattr(example_layer, "n_physical_experts", None)
        self.num_local_physical_experts = getattr(
            example_layer, "n_local_physical_experts", None
        )
        self.num_routed_experts = getattr(example_layer, "n_routed_experts", None)
        self.num_redundant_experts = getattr(example_layer, "n_redundant_experts", None)

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, HYV3SparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
