import base64
import json
import os
import random
import logging
import re
import time
import itertools
from dataclasses import dataclass
from typing import Dict, List, Any
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import glob

import transformers
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

if 'TRAIN_QWEN' in os.environ:
    print("TRAIN_QWEN is set")
    TRAIN_QWEN = True
else:
    TRAIN_QWEN = False


if 'SAMPLE_INDEPENDENTLY' in os.environ:
    print("SAMPLE_INDEPENDENTLY is set")
    SAMPLE_INDEPENDENTLY = True
else:
    SAMPLE_INDEPENDENTLY = False

if 'SHUFFLE_DATA' in os.environ:
    print("SHUFFLE_DATA is set")
    SHUFFLE_DATA = True
else:
    SHUFFLE_DATA = False

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 120687
VIDEO_TOKEN_INDEX = 120688
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files):
    if isinstance(files, dict):
        return files
    else:
        abs_path = f"{(base / files).resolve()}"
        abs_path_new = abs_path.replace('/apdcephfs_gy5_303464260/share_303464260', '/apdcephfs_gy5/share_303464260')
        if os.path.exists(abs_path_new):
            return abs_path_new
        else:
            return abs_path

def _make_abs_paths_video(base: Path, files):
    if isinstance(files, dict):
        return files
    else:
        abs_path = f"{(base / files).resolve()}"
        abs_path_new = abs_path.replace('/apdcephfs_gy5/share_303588738/peterrao', '/apdcephfs_hldy/share_303576955/peterrao')
        if os.path.exists(abs_path_new):
            return abs_path_new
        else:
            return abs_path
        
def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


def read_image_patch(patch_info):
        
    if 'img_path' in patch_info.keys():
        image_file_name = patch_info['img_path']
        if '/apdcephfs_nj3/share_300377003/peterrao/' in image_file_name:
            image_file_name = image_file_name.replace('/apdcephfs_nj3/share_300377003/peterrao/', '/apdcephfs_jn/share_302244400/peterrao/nj3/')
        return image_file_name
    
    if 'image_path' in patch_info.keys():
        image_file_name = patch_info['image_path']
        if '/apdcephfs_nj3/share_300377003/peterrao/' in image_file_name:
            image_file_name = image_file_name.replace('/apdcephfs_nj3/share_300377003/peterrao/', '/apdcephfs_jn/share_302244400/peterrao/nj3/')
        return image_file_name
    
    if 'image_encoing' in patch_info.keys():
        patch_info['image_encoding'] = patch_info['image_encoing']

    image_file_name = patch_info['patch']
    if '/apdcephfs_nj3/share_300377003/peterrao/' in image_file_name:
        image_file_name = image_file_name.replace('/apdcephfs_nj3/share_300377003/peterrao/', '/apdcephfs_jn/share_302244400/peterrao/nj3/')
    if '/apdcephfs_jn3/share_303660324/peterrao/sa1b_data' in image_file_name:
        image_file_name = image_file_name.replace('/apdcephfs_jn3/share_303660324/peterrao/sa1b_data', '/apdcephfs_jn3/share_303660324/peterrao/data/sa1b/sa1b_data')
    
    start_bytes = int(patch_info['start_num'])
    file_size = int(patch_info['size'])

    with open(image_file_name, 'rb') as f:
        f.seek(start_bytes)
        img_bytes = f.read(file_size)
        img_b64 = base64.b64encode(img_bytes).decode()
    return f"data:image/jpeg;base64,{img_b64}"


def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    processor_kwargs = {}

    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str) or isinstance(images, dict):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str) or isinstance(videos, dict):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = [{"type": "image", "image": _make_abs_paths(base_path, img)} for img in images]
    video_pool = [{"type": "video", "video": _make_abs_paths_video(base_path, vid)} for vid in videos]

    image_pool = [{"type": "image", "image": read_image_patch(img['image'])} if isinstance(img['image'], dict) else img for img in image_pool]

    messages = []
    for turn in item["conversations"]:
        role = "user" if (turn["from"] == "human" or turn["from"] == "user") else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages, processor_kwargs


def preprocess_qwen_visual(
    sources,
    processor,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages, processor_kwargs = _build_messages(source, base_path)

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=True, **processor_kwargs
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    if TRAIN_QWEN:
        input_ids_flat = input_ids[0].tolist()
        L = len(input_ids_flat)
        pos = 0
        while pos < L:
            if input_ids_flat[pos] == 77091:
                ans_start = pos + 2
                ans_end = ans_start
                while ans_end < L and input_ids_flat[ans_end] != 151645:
                    ans_end += 1
                if ans_end < L:
                    labels[0, ans_start : ans_end + 2] = input_ids[
                        0, ans_start : ans_end + 2
                    ]
                    pos = ans_end
            pos += 1

    else:
        input_ids_flat = input_ids[0].tolist()
        L = len(input_ids_flat)
        pos = 0
        while pos < L:
            if input_ids_flat[pos] == 120007:
                ans_start = pos + 1
                ans_end = ans_start
                while ans_end < L and input_ids_flat[ans_end] != 120020:
                    ans_end += 1
                if ans_end < L:
                    labels[0, ans_start : ans_end + 2] = input_ids[
                        0, ans_start : ans_end + 2
                    ]
                    pos = ans_end
            pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor, data_args):
        super(LazySupervisedDataset, self).__init__()

        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        elif data_args.model_type == "hunyuanvl":
            self.get_rope_index = None
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")
        
        self.safe_id = 0

        dataset = data_args.dataset_use.split(",")
        list_data_dict_all = []

        for meta_dataset in dataset:
            if meta_dataset.endswith(".json"):
                rank0_print(f"Loading datasets: {meta_dataset}")
                list_data_dict = json.load(open(meta_dataset, "r"))
            elif meta_dataset.endswith(".jsonl"):
                rank0_print(f"Loading datasets: {meta_dataset}")
                list_data_dict = read_jsonl(meta_dataset)
            elif 'splits' in meta_dataset:
                list_data_dict = []
                rank_idx = torch.distributed.get_rank()
                file_list = glob.glob(os.path.join(meta_dataset, "rank*"))
                file_list.sort()
                file_path = [os.path.join(f, f"pack_{str(rank_idx).zfill(3)}.jsonl") for f in file_list]
                for fp in file_path:
                    list_data_dict += read_jsonl(fp)
                print(f"Loaded {len(list_data_dict)} samples for rank {rank_idx} from {meta_dataset}")
            else:
                raise ValueError(
                    f"Unsupported dataset spec: {meta_dataset!r}. "
                    f"Pass a .json / .jsonl file path or a 'splits' directory; "
                    f"the legacy dataset-name registry has been removed."
                )
            list_data_dict_all += list_data_dict

        print(f"Total training samples: {len(list_data_dict_all)}")

        if SHUFFLE_DATA:
            # set seed
            random.seed(42)
            random.shuffle(list_data_dict_all)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict_all

        if data_args.data_packing or SAMPLE_INDEPENDENTLY:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            if 'image' in sample:
                if type(sample['image']) == list:
                    img_tokens = 128 * len(sample['image'])
                else:
                    img_tokens = 128
            elif 'video' in sample:
                img_tokens = 128 * 16  # assume 4 video tokens
            else:
                img_tokens = 0

            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )

            if 'image' in sample:
                if type(sample['image']) == list:
                    cur_len += 128 * len(sample['image'])
                else:
                    cur_len += 128
            elif 'video' in sample:
                cur_len += 128 * 16  # assume 4 video tokens

            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 2
        num_final_retries = 5

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                data_start_time = time.time()
                sample = self.item_fn(sources)
                self.safe_id = i
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e, f"use time {time.time() - data_start_time} seconds")
                if time.time() - data_start_time > 60:
                    print(f"Warning: data sample {i} took {time.time() - data_start_time} seconds to process. \n Details is {sources}")
                    print("try to use the data at index 0")

                    sources = self.list_data_dict[0]
                    if isinstance(sources, dict):
                        sources = [sources]

                    sample = self.item_fn(sources)
                    return sample
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_final_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                self.safe_id = next_index
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            fix_id = self.safe_id
            print(f"Use fix id as {fix_id} for a valid forward")
            sources = self.list_data_dict[fix_id]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        data_dict = preprocess_qwen_visual(
            sources,
            self.processor,
        )

        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        if TRAIN_QWEN:
            position_ids, _ = self.get_rope_index(
                self.merge_size,
                data_dict["input_ids"],
                image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
                video_grid_thw=(
                    torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
                ),
                second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
            )
            data_dict["position_ids"] = position_ids

        data_dict["attention_mask"] = [seq_len]

        text = self.processor.tokenizer.decode(
            data_dict["input_ids"][0], skip_special_tokens=False
        )

        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)

        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:

        if isinstance(sources, dict):
            if isinstance(source, dict):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            return self._get_item(sources)

        if isinstance(sources, list):
            data_list = []
            new_data_dict = {}
            for source in sources:
                if isinstance(source, dict):
                    source = [source]
                assert (
                    len(source) == 1
                ), f"Don't know why it is wrapped to a list.\n {source}"  # FIXME
                data_list.append(self._get_item(source))

            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            labels = torch.cat([d["labels"] for d in data_list], dim=1)
            if TRAIN_QWEN:
                position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]

            if TRAIN_QWEN:
                new_data_dict = {
                    "input_ids": input_ids,
                    "labels": labels,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask if attention_mask else None,
                }

            else:
                new_data_dict = {
                    "input_ids": input_ids,
                    "labels": labels,
                    # "position_ids": position_ids,
                    "attention_mask": attention_mask if attention_mask else None,
                }

            if any("pixel_values" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values": torch.cat(
                            [
                                d["pixel_values"]
                                for d in data_list
                                if "pixel_values" in d
                            ],
                            dim=0,
                        ),
                        "image_grid_thw": torch.cat(
                            [
                                d["image_grid_thw"]
                                for d in data_list
                                if "image_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )


            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values_videos": torch.cat(
                            [
                                d["pixel_values_videos"]
                                for d in data_list
                                if "pixel_values_videos" in d
                            ],
                            dim=0,
                        ),
                        "video_grid_thw": torch.cat(
                            [
                                d["video_grid_thw"]
                                for d in data_list
                                if "video_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )
            if new_data_dict.get('input_ids') is not None:
                if new_data_dict['input_ids'].shape[1] > 40 * 1000:
                    raise ValueError(
                            f"find too large input ids with shape {new_data_dict['input_ids'].shape}. Drop it"
                        )
            # print("input_ids", new_data_dict['input_ids'].shape)
            if new_data_dict.get('pixel_values') is not None:
                if new_data_dict['pixel_values'].shape[0] > 145 * 1000:
                    # print("pixel_values", new_data_dict['pixel_values'].shape)
                    raise ValueError(
                            f"find too large image patch with shape {new_data_dict['pixel_values'].shape}. input ids shape is {new_data_dict['input_ids'].shape}. Drop it"
                        )
            if new_data_dict.get('pixel_values_videos') is not None:
                if new_data_dict['pixel_values_videos'].shape[0] > 145 * 1000:
                    raise ValueError(
                            f"find too large video patch with shape {new_data_dict['pixel_values_videos'].shape}. input ids shape is {new_data_dict['input_ids'].shape}. Drop it"
                        )
            return new_data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        if TRAIN_QWEN:
            input_ids, labels, position_ids = tuple(
                [instance[key] for instance in instances]
                for key in ("input_ids", "labels", "position_ids")
            )
        else:
            input_ids, labels = tuple(
                [instance[key] for instance in instances]
                for key in ("input_ids", "labels")
            )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        if TRAIN_QWEN:
            position_ids = pad_and_cat(position_ids)
            position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        if TRAIN_QWEN:
            batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if TRAIN_QWEN:
            input_ids, labels, position_ids, attention_mask = tuple(
                [instance[key] for instance in instances]
                for key in ("input_ids", "labels", "position_ids", "attention_mask")
            )
        else:
            input_ids, labels, attention_mask = tuple(
                [instance[key] for instance in instances]
                for key in ("input_ids", "labels", "attention_mask")
            )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        
        if TRAIN_QWEN:
            position_ids = torch.cat(position_ids, dim=2)

        if input_ids.size(1) > self.tokenizer.model_max_length:
            print(f'Warning: input_ids length {input_ids.size(1)} exceeds model max length {self.tokenizer.model_max_length}. Truncating.')
            input_ids = input_ids[:, : self.tokenizer.model_max_length]
            labels = labels[:, : self.tokenizer.model_max_length]
            if TRAIN_QWEN:
                position_ids = position_ids[:, :, : self.tokenizer.model_max_length]


        if TRAIN_QWEN:
            batch = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=cumsum_seq_lens,
                position_ids=position_ids,
            )
        else:
            batch = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=cumsum_seq_lens,
            )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)
    if data_args.data_flatten or data_args.data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

