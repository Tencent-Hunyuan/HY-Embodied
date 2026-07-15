"""
Custom Trainer-side patches for fine-tuning HunYuanVLMoT.

This module is import-side-effecting: importing it monkey-patches
``transformers.Trainer`` (compute_loss / sampler / dataloader / scheduler /
optimizer) and the official ``hunyuan_vl_mot.modeling_hunyuan_vl_mot``
attention/text-model forwards, so that the released model can be trained with
**data packing** (multiple samples concatenated into a single B=1 sequence).

Public surface used by ``train_qwen.py``:
    * ``replace_qwen2_vl_attention_class()`` — applies the data-packing
      attention/text-model patches. Legacy name kept for callsite stability.
"""

import math
import os
import time
from functools import partial
from typing import List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Sampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler

from flash_attn.flash_attn_interface import flash_attn_varlen_func

from transformers import Trainer
from transformers.trainer import has_length
from transformers.utils import logging

from hunyuan_vl_mot.modeling_hunyuan_vl_mot import _modality_mask_to_segments

logger = logging.get_logger(__name__)

# Set with `export SAMPLE_INDEPENDENTLY=1` to make every DDP rank draw its own
# random training samples (used when datasets are pre-sharded per rank).
SAMPLE_INDEPENDENTLY = 'SAMPLE_INDEPENDENTLY' in os.environ
if SAMPLE_INDEPENDENTLY:
    print("SAMPLE_INDEPENDENTLY is set")


# ============================================================================
# Patched functions for official HunYuanVLMoT model to support data packing
# ============================================================================

def _patched_flash_attention_forward_mot(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """
    Replacement for the official ``_flash_attention_forward_mot`` that supports
    **data packing** (multiple samples concatenated into a single sequence with
    B=1) instead of standard right/left-padded batches.

    Why this patch is needed
    ------------------------
    The official ``_flash_attention_forward_mot`` in
    ``hunyuan_vl_mot.modeling_hunyuan_vl_mot`` is written for the
    inference / standard-batch-training path, where the attention_mask is::

        attention_mask = {
            "v_seqlens":    [...],
            "padding_mask": tensor of shape (B, S),  # 1=valid, 0=padding
        }

    Each sample is a separate row of a (B, S) tensor and shorter samples are
    padded up to the longest sequence in the batch. Padding-mask semantics are
    used to drop pad tokens before the varlen call.

    For training we instead use **data packing**: many samples of very different
    lengths are concatenated into one long sequence of shape (1, total_len), and
    the per-sample boundaries are encoded as a 1D ``cu_seqlens`` tensor
    (e.g. ``[0, 312, 547, 1024]`` = 3 samples of length 312/235/477). This
    eliminates padding waste entirely — important when sample lengths vary by
    10x+ — but it changes the meaning of ``attention_mask`` from a 2D
    padding-mask to a 1D cumulative-length tensor.

    The two formats are not interchangeable: feeding the packed ``cu_seqlens``
    tensor through the official forward would either error on shape, or worse,
    silently treat the whole packed sequence as one giant causal block (samples
    leaking into each other across boundaries → corrupted training).

    Therefore, when data packing is enabled we monkey-patch this function
    (together with ``_HunYuanVLMoTTextModel.forward``, which builds the
    attention_mask dict). This patched version expects::

        attention_mask = {
            "v_seqlens":  [(s, e), ...],   # vision spans within the packed seq
            "cu_seqlens": tensor [0, l1, l1+l2, ...],
        }

    and calls ``flash_attn_varlen_func`` directly with ``cu_seqlens`` so each
    sample stays its own causal block. Vision spans are then re-attended
    bidirectionally (causal=False) and written back over the causal output —
    same MoT behavior as the official model, just adapted to B=1 packing.

    Behavior parity with the official model:
        * causal attention per sample boundary (via cu_seqlens)
        * bidirectional attention within each vision span
        * ``fake_visual`` path keeps vision params in the autograd graph when a
          packed batch happens to contain no vision tokens (avoids DDP
          "unused parameters" errors).
    """
    if kwargs.get("output_attentions", False):
        pass  # flash attention doesn't support output_attentions

    # Transpose from (B, heads, S, D) -> (B, S, heads, D)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    # Squeeze batch dim for varlen attention (data packing uses B=1)
    query = query.squeeze(0)
    key = key.squeeze(0)
    value = value.squeeze(0)

    cu_seqlens = attention_mask['cu_seqlens']
    v_seqlens = attention_mask['v_seqlens']

    with torch.no_grad():
        max_seqlen = max(
            [
                cu_seqlens[idx + 1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]
        ).item()

    # Causal attention with per-sample boundaries
    attn_output = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )

    # Visual bidirectional attention override
    if len(v_seqlens) > 0:
        fake_visual = False
    else:
        fake_visual = True
        v_seqlens = [(0, 2)]

    visual_query = []
    visual_key = []
    visual_value = []
    visual_mask = torch.zeros(query.shape[0], dtype=torch.bool, device=query.device)
    cu_v_seqlen = [0]
    max_v_seqlen = 0
    for s, e in v_seqlens:
        visual_query.append(query[s:e])
        visual_key.append(key[s:e])
        visual_value.append(value[s:e])
        visual_mask[s:e] = True
        cu_v_seqlen.append(cu_v_seqlen[-1] + (e - s))
        if e - s > max_v_seqlen:
            max_v_seqlen = e - s

    visual_query = torch.cat(visual_query, dim=0)
    visual_key = torch.cat(visual_key, dim=0)
    visual_value = torch.cat(visual_value, dim=0)
    cu_v_seqlens = torch.tensor(cu_v_seqlen, device=query.device, dtype=torch.int32)
    visual_attn_output = flash_attn_varlen_func(
        visual_query,
        visual_key,
        visual_value,
        cu_seqlens_q=cu_v_seqlens,
        cu_seqlens_k=cu_v_seqlens,
        max_seqlen_q=max_v_seqlen,
        max_seqlen_k=max_v_seqlen,
        causal=False
    )

    if fake_visual:
        attn_output = attn_output + visual_attn_output.mean() * 0
    else:
        attn_output = attn_output.clone()
        attn_output[visual_mask] = visual_attn_output

    attn_output = attn_output.unsqueeze(0)
    return attn_output, None


def _patched_text_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    cache_position=None,
    use_cache=None,
    modality_mask=None,
    **kwargs,
):
    """
    Replacement for the official ``_HunYuanVLMoTTextModel.forward`` that wires
    up the **data-packing** attention_mask format expected by
    ``_patched_flash_attention_forward_mot``.

    Why this patch is needed
    ------------------------
    The official ``_HunYuanVLMoTTextModel.forward`` constructs the per-layer
    ``causal_mask`` as::

        causal_mask = {
            "v_seqlens":    visual_segs,
            "padding_mask": attention_mask,   # the (B, S) 2D padding mask
        }

    That works for the standard batched / padded inputs the released model is
    built for, but it's the wrong shape for our data-packing data collator,
    which produces ``attention_mask`` as a **1D cu_seqlens tensor**
    (e.g. ``[0, 312, 547, 1024]``).

    What this patched forward does differently
    ------------------------------------------
    * Builds ``causal_mask = {"cu_seqlens": attention_mask, "v_seqlens": ...}``
      — passes the collator's 1D tensor straight through to the kernel.
    * Drops the official's left-padding-aware ``position_ids`` reconstruction
      (``cumsum`` over the padding mask), because in packing mode there is no
      padding and ``cache_position`` already gives the correct positions.
    * Drops the decode-time mask truncation, because training never decodes.

    Everything else (embedding lookup, KV cache wiring, RoPE, decoder-layer
    loop, final RMSNorm, return type) is identical to the official forward, so
    behavior is unchanged outside the attention_mask plumbing.
    """
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)

    # For data packing, position_ids should be passed from outside or computed from cache_position
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    text_position_ids = position_ids

    if modality_mask is None:
        modality_mask = torch.zeros(inputs_embeds.shape[:-1], dtype=torch.bool, device=inputs_embeds.device)

    # Compute visual segments from modality mask
    visual_segs = _modality_mask_to_segments(modality_mask)

    # Create attention mask dict with cu_seqlens for data packing
    # attention_mask here is the cu_seqlens tensor from the data collator
    causal_mask = {
        'cu_seqlens': attention_mask,  # cumulative sequence lengths
        'v_seqlens': visual_segs,      # list of (start, end) tuples for vision tokens
    }

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, text_position_ids)

    for decoder_layer in self.layers:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            modality_mask=modality_mask,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


def replace_qwen2_vl_attention_class():
    """Patch the official HunYuanVLMoT attention/text-model forward for data packing.

    Despite the legacy name (kept for callsite stability), this only patches
    HunYuanVLMoT — Qwen2/Qwen2.5/Qwen3-VL training paths are not supported here.
    """
    import hunyuan_vl_mot.modeling_hunyuan_vl_mot as hvlm
    hvlm._flash_attention_forward_mot = _patched_flash_attention_forward_mot
    hvlm._HunYuanVLMoTTextModel.forward = _patched_text_model_forward


# ============================================================================
# Optimizer: per-group learning rates (vision_tower / mm_projector / mot)
# ============================================================================

def create_optimizer(self):
    """Custom optimizer builder that supports per-group learning rates.

    Parameters are split into four categories by name:
        * ``merger`` (multimodal projector)         → optional ``mm_projector_lr``
        * ``visual`` (ViT vision tower)             → optional ``vision_tower_lr``
        * ``*_v.`` MoT layers (vision-side decoder) → optional ``vision_mot_lr``
        * everything else                           → default ``args.learning_rate``

    Each category is further split into decay / no-decay groups (biases and
    norms get no decay). If a per-group lr arg is None / 0, that category falls
    through to the default group.
    """
    if self.optimizer is not None:
        return self.optimizer

    opt_model = self.model
    decay_parameters = self.get_decay_parameter_names(opt_model)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    # ---- 1. Categorize parameters by name ---------------------------------
    MOT_PATTERNS = (
        '.mlp_v.',
        '.post_attention_layernorm_v.',
        '.input_layernorm_v.',
        '.o_proj_v.', '.k_proj_v.', '.q_proj_v.', '.v_proj_v.',
    )

    projector_parameters = []
    vision_tower_parameters = []
    vision_mot_parameters = []
    others_parameters = []
    for name, _ in opt_model.named_parameters():
        if "merger" in name:
            projector_parameters.append(name)
        elif "visual" in name:
            vision_tower_parameters.append(name)
        elif any(pat in name for pat in MOT_PATTERNS):
            vision_mot_parameters.append(name)
        else:
            others_parameters.append(name)

    use_mm_projector_lr = getattr(self.args, 'mm_projector_lr', None) not in [None, 0]
    use_vision_tower_lr = getattr(self.args, 'vision_tower_lr', None) not in [None, 0]
    use_vision_mot_lr   = getattr(self.args, 'vision_mot_lr',   None) not in [None, 0]

    def get_param_category(name):
        if use_mm_projector_lr and name in projector_parameters:
            return 'projector'
        if use_vision_tower_lr and name in vision_tower_parameters:
            return 'vision_tower'
        if use_vision_mot_lr and name in vision_mot_parameters:
            return 'vision_mot'
        return 'default'

    # ---- 2. Build (decay, no_decay) × (default, projector, vt, mot) groups
    param_groups_config = {
        'default_decay':       {'names': [], 'weight_decay': self.args.weight_decay, 'lr': None},
        'default_no_decay':    {'names': [], 'weight_decay': 0.0,                    'lr': None},
        'projector_decay':     {'names': [], 'weight_decay': self.args.weight_decay, 'lr': self.args.mm_projector_lr},
        'projector_no_decay':  {'names': [], 'weight_decay': 0.0,                    'lr': self.args.mm_projector_lr},
        'vision_tower_decay':  {'names': [], 'weight_decay': self.args.weight_decay, 'lr': self.args.vision_tower_lr},
        'vision_tower_no_decay': {'names': [], 'weight_decay': 0.0,                  'lr': self.args.vision_tower_lr},
        'vision_mot_decay':    {'names': [], 'weight_decay': self.args.weight_decay, 'lr': self.args.vision_mot_lr},
        'vision_mot_no_decay': {'names': [], 'weight_decay': 0.0,                    'lr': self.args.vision_mot_lr},
    }
    for name, param in opt_model.named_parameters():
        if not param.requires_grad:
            continue
        category = get_param_category(name)
        needs_decay = name in decay_parameters
        suffix = '_decay' if needs_decay else '_no_decay'
        prefix = 'default' if category == 'default' else category
        param_groups_config[prefix + suffix]['names'].append((name, param))

    optimizer_grouped_parameters = []
    for group_config in param_groups_config.values():
        if not group_config['names']:
            continue
        group = {
            'params': [p for _, p in group_config['names']],
            'weight_decay': group_config['weight_decay'],
        }
        if group_config['lr'] is not None:
            group['lr'] = group_config['lr']
        optimizer_grouped_parameters.append(group)

    if not optimizer_grouped_parameters:
        raise ValueError(
            "create_optimizer: no trainable parameters found. "
            "Check tune_mm_* flags or LoRA config."
        )

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
    self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return self.optimizer



# ============================================================================
# Sampler: groups same-modality / similar-length samples into the same batch
# ============================================================================
# Only the ``group_by_modality_length=True`` code path is used in this
# fine-tuning code (see ``patched_get_train_sampler``); the variants that
# existed in the original repo (variable-length / auto-modality / hf-default)
# have been removed as dead code.

def split_to_even_chunks(indices, lengths, num_chunks):
    """Split ``indices`` into ``num_chunks`` chunks of roughly equal total length."""
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")
    return chunks


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """Permute indices so that items inside each megabatch have similar length.

    The element with the maximum length is placed at the start of each
    megabatch so an OOM (if any) happens early rather than mid-training.
    """
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i: i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(mb, key=lambda i: lengths[i], reverse=True) for mb in megabatches]
    megabatches = [split_to_even_chunks(mb, lengths, world_size) for mb in megabatches]
    return [i for mb in megabatches for batch in mb for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """Group items by modality (positive length = multimodal, negative = text-only)
    AND by similar length within each modality bucket.
    """
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)

    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i: i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i: i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    additional_batch = mm_megabatches[-1] + lang_megabatches[-1]
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatches = [megabatches[i] for i in torch.randperm(len(megabatches), generator=generator)]
    if additional_batch:
        megabatches.append(sorted(additional_batch))

    return [i for mb in megabatches for i in mb]


class LengthGroupedSampler(Sampler):
    """Sampler that groups indices of similar length and similar modality together."""

    def __init__(self, batch_size: int, world_size: int,
                 lengths: Optional[List[int]] = None, generator=None):
        if lengths is None:
            raise ValueError("Lengths must be provided.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_modality_length_grouped_indices(
            self.lengths, self.batch_size, self.world_size, generator=self.generator,
        )
        return iter(indices)


_ORIG_SAMPLER = Trainer._get_train_sampler

def patched_get_train_sampler(self, train_dataset) -> Optional[torch.utils.data.Sampler]:
    """Return ``LengthGroupedSampler`` when ``--group_by_modality_length`` is set,
    otherwise fall back to the default Trainer sampler."""
    if train_dataset is None:
        train_dataset = self.train_dataset
    if train_dataset is None or not has_length(train_dataset):
        return None

    if self.args.group_by_modality_length:
        print("Using LengthGroupedSampler with modality grouping for training.")
        return LengthGroupedSampler(
            batch_size=self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            lengths=train_dataset.modality_lengths,
        )
    return _ORIG_SAMPLER(self)
# ============================================================================
# compute_loss + extra training-progress logging (llm_loss, ETA, step_time)
# ============================================================================

def gather_log_metrics_sum(metrics):
    """Average loss-like metrics across DDP ranks (skipping zero-valued ranks)."""
    if not torch.distributed.is_initialized():
        return metrics

    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, metrics)
    averaged = {}
    for k in metrics.keys():
        values = [o[k] for o in gathered if o is not None]
        if 'loss' in k:
            non_zero = sum(1 for v in values if v != 0.0)
            averaged[k] = sum(values) / max(non_zero, 1)
        else:
            averaged[k] = values[0] if values else 0.0
    return averaged


_ORIG_COMPUTE_LOSS = Trainer.compute_loss
Trainer.step_time_meters = []


def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    """Wrap Trainer.compute_loss to log llm_loss / step_time / ETA every step.

    Note: only the standard LM loss is computed; auxiliary losses (CLIP / ViQ /
    DA3) are pretraining-only and not part of this fine-tuning code.
    """
    begin_time = time.time()
    loss, outputs = _ORIG_COMPUTE_LOSS(self, model, inputs, return_outputs=True)
    now = time.time()

    # Rolling window of the last 100 step timestamps for a smooth step_time
    if len(self.step_time_meters) < 100:
        self.step_time_meters.append(now)
    else:
        self.step_time_meters = self.step_time_meters[1:] + [now]

    if len(self.step_time_meters) > 1:
        diffs = [self.step_time_meters[i] - self.step_time_meters[i - 1]
                 for i in range(1, len(self.step_time_meters))]
        step_time = sum(diffs) / len(diffs)
    else:
        # Estimate first-step time as 3x forward time (forward + backward + opt)
        step_time = (now - begin_time) * 3

    eta_time = (step_time
                * (self.state.max_steps - self.state.global_step)
                / 3600
                * self.args.gradient_accumulation_steps)

    logs = {'llm_loss': loss.item() if loss is not None else 0.0}
    logs = gather_log_metrics_sum(logs)
    logs.update({
        'eta_time': eta_time,
        'step_time': round(step_time, 4),
        'now_step': self.state.global_step,
        'task_max_steps': self.state.max_steps,
    })
    self.log(logs)
    return (loss, outputs) if return_outputs else loss



# ============================================================================
# DataLoader: rank-independent random sampling for pre-sharded datasets
# ============================================================================

def get_train_dataloader(self):
    """Build a training DataLoader where each DDP rank uses its own RNG seed.

    Use this only when the dataset has already been pre-sharded per rank
    (otherwise different ranks will see overlapping samples). Activated by
    ``export SAMPLE_INDEPENDENTLY=1``.
    """
    print('Sampling data independently on each rank. '
          'Use this setting only if the dataset has already been pre-sharded per rank.')
    train_dataset = self.train_dataset
    dataloader_params = {
        "batch_size": self._train_batch_size,
        "collate_fn": self.data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
        "persistent_workers": self.args.dataloader_persistent_workers,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        # Per-rank seed → each rank gets a different shuffle order.
        generator = torch.Generator()
        seed = int(torch.empty((), dtype=torch.int64).random_().item()) + torch.distributed.get_rank()
        generator.manual_seed(seed)
        dataloader_params["sampler"] = RandomSampler(train_dataset, generator=generator)
        dataloader_params["drop_last"] = self.args.dataloader_drop_last

    return DataLoader(train_dataset, **dataloader_params)


# ============================================================================
# Cosine LR scheduler with configurable min-lr floor
# ============================================================================

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int,
    num_cycles: float, min_lr_ratio: float,
):
    assert 0.0 <= min_lr_ratio <= 1.0, "min_lr_ratio should be in [0.0, 1.0]"
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_value = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    cosine_value = (1 - min_lr_ratio) * cosine_value + min_lr_ratio
    return max(min_lr_ratio, cosine_value)


def get_cosine_schedule_with_warmup_and_min_lr(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
    num_cycles: float = 0.5, last_epoch: int = -1, min_lr_ratio: float = 0.0,
):
    """Cosine schedule with linear warmup that decays to ``min_lr_ratio * lr``
    instead of all the way to 0."""
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    """Replacement for ``Trainer.create_scheduler`` that uses the cosine + min-lr schedule."""
    self.step_time_meters = []
    self.avg_loss_meters = []
    if self.lr_scheduler is None:
        self.lr_scheduler = get_cosine_schedule_with_warmup_and_min_lr(
            optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
            min_lr_ratio=self.args.min_lr_ratio,
        )
        self._created_lr_scheduler = True
    return self.lr_scheduler


# ============================================================================
# Apply Trainer monkey-patches at import time
# ============================================================================
Trainer.create_optimizer = create_optimizer
Trainer.create_scheduler = create_scheduler
Trainer._get_train_sampler = patched_get_train_sampler
Trainer.compute_loss = compute_loss
if SAMPLE_INDEPENDENTLY:
    Trainer.get_train_dataloader = get_train_dataloader
