# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://github.com/ManaEstras/transformers/blob/v4.57.1.hyvl/src/transformers/models/hunyuan_vl/processing_hunyuan_vl.py


import numpy as np
import torch
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import (
    ImagesKwargs,
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    VideosKwargs,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput


class HunYuanVLVideosProcessorKwargs(VideosKwargs, total=False):  # type: ignore[call-arg]
    fps: list[float] | float


class HunYuanVLImagesKwargs(ImagesKwargs):
    min_pixels: int | None
    max_pixels: int | None
    patch_size: int | None
    temporal_patch_size: int | None
    merge_size: int | None


class HunYuanVLProcessorKwargs(ProcessingKwargs, total=False):  # type: ignore[call-arg]
    images_kwargs: HunYuanVLImagesKwargs
    videos_kwargs: HunYuanVLVideosProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
    }


class HunYuanVLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer", "video_processor"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "AutoTokenizer"  # ("AutoTokenizer", None)

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        self.tokenizer = tokenizer

        # Check if required tokenizer attributes exist
        required_attrs = [
            "image_token",
            "image_start_token",
            "image_end_token",
            "vocab_size",
            "pad_token",
        ]
        for attr in required_attrs:
            if not hasattr(tokenizer, attr):
                raise ValueError(
                    f"Tokenizer missing required attribute '{attr}'. "
                    "Please add the corresponding field mapping to "
                    "`extra_special_tokens` in tokenizer_config.json"
                )

        self.image_token = tokenizer.image_token
        self.image_token_id = self.tokenizer.encode(self.tokenizer.image_token)[0]
        self.image_start_token = tokenizer.image_start_token
        self.image_start_token_id = self.tokenizer.encode(
            self.tokenizer.image_start_token
        )[0]
        self.image_end_token = tokenizer.image_end_token
        self.image_end_token_id = self.tokenizer.encode(self.tokenizer.image_end_token)[
            0
        ]
        self.video_token = tokenizer.video_token
        self.video_token_id = self.tokenizer.encode(self.tokenizer.video_token)[0]
        self.video_start_token = (
            "<｜hy_place▁holder▁no▁104｜>"
            if not hasattr(tokenizer, "video_start_token")
            else tokenizer.video_start_token
        )
        self.video_start_token_id = (
            tokenizer.video_start_token_id
            if getattr(tokenizer, "video_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_start_token)
        )
        self.video_end_token = (
            "<｜hy_place▁holder▁no▁105｜>"
            if not hasattr(tokenizer, "video_end_token")
            else tokenizer.video_end_token
        )
        self.video_end_token_id = (
            tokenizer.video_end_token_id
            if getattr(tokenizer, "video_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_end_token)
        )
        self.placeholder_token = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.vocab_size - 1
        )
        self.pad_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]

        super().__init__(
            image_processor, tokenizer, video_processor, chat_template=chat_template
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        **kwargs,
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            HunYuanVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        image_inputs = videos_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(
                images=images, **output_kwargs["images_kwargs"]
            )
            image_grid_thw = image_inputs["image_grid_thw"]

        if videos is not None:
            videos_inputs = self.video_processor(
                videos=videos, **output_kwargs["videos_kwargs"]
            )
            video_grid_thw = videos_inputs["video_grid_thw"]

        # below lines change text in-place
        if not isinstance(text, list):
            text = [text]
        text = text.copy()

        if images is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    grid_h, grid_w = image_grid_thw[index][-2:]
                    patch_h = grid_h // self.image_processor.merge_size
                    patch_w = grid_w // self.image_processor.merge_size
                    num_image_tokens = patch_h * (patch_w + 1) + 2
                    # text[i] = text[i].replace(
                    #     self.image_token,
                    #     self.image_start_token
                    #     + self.placeholder_token * num_image_tokens
                    #     + self.image_end_token,
                    #     1,
                    # )
                    text[i] = text[i].replace(
                        self.image_token, self.placeholder_token * num_image_tokens, 1
                    )
                    index += 1
                text[i] = text[i].replace(self.placeholder_token, self.image_token)
                # text[i] = self.tokenizer.bos_token + text[i]

        if videos is not None:
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    grid_t, grid_h, grid_w = video_grid_thw[index].tolist()
                    grid_t = (
                        grid_t + self.image_processor.temporal_patch_size - 1
                    ) // self.image_processor.temporal_patch_size
                    patch_h = grid_h // self.image_processor.merge_size
                    patch_w = grid_w // self.image_processor.merge_size
                    patch_h = patch_h // self.image_processor.spatial_patch_size
                    patch_w = patch_w // self.image_processor.spatial_patch_size
                    num_image_tokens = patch_h * (patch_w + 1) + 2
                    replace_str = ""
                    for _ in range(grid_t):
                        replace_str += (
                            self.image_start_token
                            + self.placeholder_token * num_image_tokens
                            + self.image_end_token
                        )
                    text[i] = text[i].replace(self.video_token, replace_str, 1)
                    index += 1
                text[i] = text[i].replace(self.placeholder_token, self.video_token)

        text_inputs = self.tokenizer(text, add_special_tokens=False, **kwargs)
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        input_ids = text_inputs["input_ids"]
        position_ids = torch.arange(len(input_ids[0]))
        position_ids_w = torch.arange(len(input_ids[0]))
        position_ids_h = torch.arange(len(input_ids[0]))
        position_ids_t = torch.arange(len(input_ids[0]))

        if images is not None or videos is not None:
            image_start_token_pos_indices = torch.where(
                input_ids[0] == self.image_start_token_id
            )[0]
            image_count = 0
            video_index = -1
            image_index = -1
            video_mode = False  # noqa: F841
            while image_count < len(image_start_token_pos_indices):
                start_pos = image_start_token_pos_indices[image_count]
                prev_token = input_ids[0][start_pos - 1]
                if prev_token == self.video_start_token_id:
                    video_index += 1
                    grid_t, grid_h, grid_w = video_grid_thw[video_index].tolist()
                    grid_t = (
                        grid_t + self.image_processor.temporal_patch_size - 1
                    ) // self.image_processor.temporal_patch_size
                    patch_h = grid_h // self.image_processor.merge_size
                    patch_w = grid_w // self.image_processor.merge_size
                    patch_h = patch_h // self.image_processor.spatial_patch_size
                    patch_w = patch_w // self.image_processor.spatial_patch_size
                    replace_num = patch_h * (patch_w + 1)
                    for image_count_new in range(image_count, image_count + grid_t):
                        start_pos = image_start_token_pos_indices[image_count_new] + 2
                        position_ids_w[start_pos : start_pos + replace_num] = (
                            torch.tensor(
                                list(range(patch_w + 1)) * patch_h, dtype=torch.int64
                            )
                        )
                        patch_h_list = []
                        for h in range(patch_h):
                            patch_h_list += [h] * (patch_w + 1)
                        position_ids_h[start_pos : start_pos + replace_num] = (
                            torch.tensor(patch_h_list, dtype=torch.int64)
                        )
                        position_ids_t[start_pos : start_pos + replace_num] = (
                            image_count_new
                        )
                    image_count += grid_t
                else:
                    image_index += 1
                    grid_h, grid_w = image_grid_thw[image_index][-2:]
                    patch_h = grid_h // self.image_processor.merge_size
                    patch_w = grid_w // self.image_processor.merge_size
                    start_pos = start_pos + 2
                    replace_num = (patch_w + 1) * patch_h
                    position_ids_w[start_pos : start_pos + replace_num] = torch.tensor(
                        list(range(patch_w + 1)) * patch_h, dtype=torch.int64
                    )
                    patch_h_list = []
                    for h in range(patch_h):
                        patch_h_list += [h] * (patch_w + 1)
                    position_ids_h[start_pos : start_pos + replace_num] = torch.tensor(
                        patch_h_list, dtype=torch.int64
                    )
                    position_ids_t[start_pos : start_pos + replace_num] = image_count
                    image_count += 1

        position_ids = torch.stack(
            [position_ids, position_ids_w, position_ids_h, position_ids_t]
        ).unsqueeze(0)
        text_inputs["position_ids"] = position_ids

        attention_mask = input_ids.ne(self.pad_id)
        text_inputs["attention_mask"] = attention_mask
        text_inputs["imgs_pos"] = [self.get_imgs_pos(input_ids)]

        return_tensors = kwargs.pop("return_tensors", None)
        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs},
            tensor_type=return_tensors,
        )

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs
        with the given sizes.
        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.
            video_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (num_frames, height, width) per
                each video.
        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of
            tokens per each of the provided input modalities, along with other
            useful data.
        """

        vision_data = {}
        if image_sizes is not None:
            merge_size = kwargs.get("merge_size") or self.image_processor.merge_size

            num_image_patches_size = [
                self.image_processor.get_number_of_image_patches(*image_size, kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [
                (patch_hw[0] // merge_size * (patch_hw[1] // merge_size + 1) + 2)
                for patch_hw in num_image_patches_size
            ]
            num_image_patches = [
                (patch_hw[0] * patch_hw[1]) for patch_hw in num_image_patches_size
            ]
            vision_data.update(
                {
                    "num_image_tokens": num_image_tokens,
                    "num_image_patches": num_image_patches,
                }
            )
            print(f"vision_data: {vision_data}")

        return MultiModalData(**vision_data)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(
        self,
        generated_outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        assert 0

    def apply_chat_template(self, *args, **kwargs):
        token_ids = self.tokenizer.apply_chat_template(*args, **kwargs)
        return token_ids

    def get_imgs_pos(self, doc_ids):
        doc_ids = np.array(doc_ids, dtype=np.int64)
        img_begin_index = np.where(doc_ids == self.image_start_token_id)[0]
        img_end_index = np.where(doc_ids == self.image_end_token_id)[0]
        imgs_pos = np.concatenate(
            (
                np.reshape(img_begin_index + 1, (-1, 1)),
                np.reshape(img_end_index, (-1, 1)),
            ),
            axis=-1,
        ).tolist()
        return imgs_pos

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


def split_image_into_patch_blocks(
    pixel_values: torch.Tensor,  # shape: [batch_size, 3, H, W]
    patch_size: int = 16,  # e.g. 16
    # e.g. 4 --> 表示每个 patch_size 切成 4x4 小区域，即 patch_size // 4
    adaptor_patch_div: int = 4,
) -> torch.Tensor:
    """
    Split the input image tensor (supporting batch) into large patches of size
    `patch_size`, and then further divide each large patch into smaller regions
    of size (patch_size // adaptor_patch_div) x (patch_size // adaptor_patch_div).
    Each small region is extracted as a tensor of shape [3, patch_size, patch_size].
    The final output contains all such small region tensors.

    Args:
        pixel_values: Input image tensor of shape [batch_size, 3, H, W].
        patch_size: Size of the large patch, e.g., 16.
        adaptor_patch_div: Each large patch is divided into
                          (patch_size // adaptor_patch_div) x
                          (patch_size // adaptor_patch_div) smaller regions.

    Returns:
        patches: A tensor of shape [N, 3, patch_size, patch_size],
                 where N = batch_size * (H // patch_size) * (W // patch_size)
                 * (patch_size // adaptor_patch_div)^2.
                 Each element in the batch corresponds to one small image region.
    """
    batch_size, channels, height, width = pixel_values.shape
    assert channels == 3, "Pixel values must have 3 channels in dim=1"
    assert height % patch_size == 0 and width % patch_size == 0, (
        "H and W must be divisible by patch_size"
    )

    patch_height_num = height // patch_size
    patch_width_num = width // patch_size
    small_regions_per_patch = (patch_size // adaptor_patch_div) ** 2  # noqa: F841

    # Reshape to [B, 3, ph, ps, pw, ps]
    img = pixel_values.reshape(
        batch_size, 3, patch_height_num, patch_size, patch_width_num, patch_size
    )

    # Further split each psxps patch into (ps//aps)x(ps//aps) small regions
    img = img.reshape(
        batch_size,
        3,
        patch_height_num,
        patch_size // adaptor_patch_div,  # ps // aps
        adaptor_patch_div,
        patch_width_num,
        patch_size // adaptor_patch_div,  # ps // aps
        adaptor_patch_div,
    )

    # Permute to group the small regions: [B, ph, pw, ps//aps, ps//aps, 3, aps, aps]
    img = img.permute(0, 2, 5, 3, 6, 1, 4, 7)

    # Reshape into [B * ph * pw * (ps//aps)^2, 3, patch_size, patch_size]
    patches = img.reshape(-1, 3, patch_size, patch_size)

    return patches


AutoProcessor.register("HunYuanVLProcessor", HunYuanVLProcessor)
