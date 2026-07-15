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

"""
Processor class for HunYuanVL-MoT model.
Combines image/video processing and text tokenization.
"""

import torch
from typing import List, Union

import numpy as np

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.video_utils import VideoInput

logger = logging.get_logger(__name__)


class HunYuanVLMoTProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }


class HunYuanVLMoTProcessor(ProcessorMixin):
    r"""
    Constructs a HunYuanVL-MoT processor which wraps image/video processors and a tokenizer
    into a single processor.

    Args:
        image_processor (`AutoImageProcessor`, *optional*):
            The image processor is a required input.
        tokenizer (`PreTrainedTokenizerFast`, *optional*):
            The tokenizer is a required input.
        video_processor (`AutoVideoProcessor`, *optional*):
            The video processor is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
        self.image_token = tokenizer.image_token
        self.video_token = tokenizer.video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.vision_start_token = tokenizer.vision_start_token
        self.vision_end_token = tokenizer.vision_end_token
        self.vision_start_token_id = (
            tokenizer.vision_start_token_id
            if getattr(tokenizer, "vision_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_start_token)
        )
        self.vision_end_token_id = (
            tokenizer.vision_end_token_id
            if getattr(tokenizer, "vision_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_end_token)
        )
        self.image_newline_token = tokenizer.image_newline_token

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[HunYuanVLMoTProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s)/video(s).

        Args:
            images: The image or batch of images to be prepared.
            text: The sequence or batch of sequences to be encoded.
            videos: The video or batch of videos to be prepared.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with input_ids, attention_mask, pixel_values, etc.
        """
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
            # If user has not requested video metadata, pop it
            if "return_metadata" not in kwargs:
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place

        # --- Image token replacement ---
        # Replace each image_token with a grid of image tokens + newline tokens
        # Layout: (grid_w // merge_size) image_tokens + 1 newline token, repeated (grid_h * grid_t // merge_size) times
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size ** 2
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

        # --- Video token replacement ---
        # Each frame is wrapped in vision_start_token...vision_end_token
        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size ** 2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    metadata = video_metadata[index]
                    if metadata.fps is None:
                        logger.warning_once(
                            "HunYuanVL-MoT requires frame timestamps to construct prompts, but the `fps` of "
                            "the input video could not be inferred. Defaulting to `fps=24`."
                        )
                        metadata.fps = 24

                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length

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

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.
        """
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = HunYuanVLMoTProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size ** 2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        if video_sizes is not None:
            videos_kwargs = HunYuanVLMoTProcessorKwargs._defaults.get("videos_kwargs", {})
            videos_kwargs.update(kwargs)
            merge_size = videos_kwargs.get("merge_size", None) or self.video_processor.merge_size
            num_video_patches = [
                self.video_processor.get_number_of_video_patches(*video_size, videos_kwargs)
                for video_size in video_sizes
            ]
            num_video_tokens = [(num_patches // merge_size ** 2) for num_patches in num_video_patches]
            vision_data["num_video_tokens"] = num_video_tokens

        return MultiModalData(**vision_data)

    def pad(
        self,
        inputs_list: List[BatchFeature],
        padding: bool = True,
        padding_side: str = "left",
        return_tensors: str = "pt",
    ) -> BatchFeature:
        """
        Pad a list of single-sample BatchFeature dicts and combine into a batch.

        Supports left-padding (default, for generation) and right-padding.
        Pads input_ids with pad_token_id and attention_mask with 0.
        Vision tensors (pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw)
        are concatenated along dim 0 when present in ALL inputs; skipped otherwise.

        Args:
            inputs_list: List of BatchFeature / dict from apply_chat_template, each with batch dim 1.
            padding: Whether to pad (must be True).
            padding_side: "left" (default) or "right".
            return_tensors: Tensor type for output (default "pt").

        Returns:
            BatchFeature with all inputs batched and padded.
        """
        if not padding:
            raise ValueError("padding=False is not supported; use padding=True.")

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        seq_lens = [inp["input_ids"].shape[1] for inp in inputs_list]
        max_len = max(seq_lens)

        batched = {}

        # input_ids: pad with pad_token_id
        padded_ids = []
        for inp, slen in zip(inputs_list, seq_lens):
            pad_len = max_len - slen
            ids = inp["input_ids"]
            if pad_len > 0:
                pad = torch.full((1, pad_len), pad_token_id, dtype=ids.dtype)
                if padding_side == "left":
                    ids = torch.cat([pad, ids], dim=1)
                else:
                    ids = torch.cat([ids, pad], dim=1)
            padded_ids.append(ids)
        batched["input_ids"] = torch.cat(padded_ids, dim=0)

        # attention_mask: pad with 0
        padded_masks = []
        for inp, slen in zip(inputs_list, seq_lens):
            pad_len = max_len - slen
            mask = inp["attention_mask"]
            if pad_len > 0:
                pad = torch.zeros((1, pad_len), dtype=mask.dtype)
                if padding_side == "left":
                    mask = torch.cat([pad, mask], dim=1)
                else:
                    mask = torch.cat([mask, pad], dim=1)
            padded_masks.append(mask)
        batched["attention_mask"] = torch.cat(padded_masks, dim=0)

        # Vision tensors: concatenate along dim 0 when present in ALL inputs
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            tensors = [inp[key] for inp in inputs_list if key in inp and inp[key] is not None]
            if len(tensors) > 0:
                batched[key] = torch.cat(tensors, dim=0)

        return BatchFeature(data=batched, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        """
        Post-process the output of the model to decode the text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["HunYuanVLMoTProcessor"]
