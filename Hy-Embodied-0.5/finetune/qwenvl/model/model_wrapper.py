"""
Model wrapper for official HunYuanVLMoTForConditionalGeneration.

This module wraps the official open-source model for fine-tuning.

Note:
    This fine-tuning code only supports the standard language-modeling (LM) loss.
    The auxiliary loss heads (CLIP, ViQ, DA3) are only used during the model's
    pretraining stage and have been intentionally removed from this wrapper.
    If you need them, refer to the pretraining codebase.
"""

from typing import Optional, Union

import torch

from transformers.cache_utils import Cache

from hunyuan_vl_mot.modeling_hunyuan_vl_mot import HunYuanVLMoTForConditionalGeneration


class HunYuanVLMoTWithAuxLoss(HunYuanVLMoTForConditionalGeneration):
    """
    Thin subclass of the official ``HunYuanVLMoTForConditionalGeneration``.

    Kept as a separate class so the training entry point has a single, stable
    symbol to import. The class name is preserved for backward compatibility
    with existing scripts/configs, but no auxiliary losses are computed here.

    Note:
        This fine-tuning code only supports the standard LM loss. The auxiliary
        loss heads (CLIP, ViQ, DA3) are only used during the model's pretraining
        stage; they are intentionally not constructed or invoked here.
    """

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        """Delegate fully to the official model. Only LM loss is returned."""
        return self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            labels=labels,
            **kwargs,
        )
