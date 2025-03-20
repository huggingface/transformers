# coding=utf-8
# Copyright 2025 The Vita Team and The HuggingFace Inc. team. All rights reserved.
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

from typing import List, Union, Optional, Dict
from functools import partial
import os
import re

import numpy as np
import torch
import decord
from PIL import Image

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, VideoInput, VideoMetadata
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs, AllKwargsForChatTemplate, ImagesKwargs
from ...tokenization_utils_base import PreTokenizedInput, TextInput


IMG_TAG_TOKEN = "<image>"
VID_TAG_TOKEN = "<video>"
AUD_TAG_TOKEN = "<audio>"

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'

VID_CONTEXT_TOKEN = '<VID_CONTEXT>'
VID_START_TOKEN = '<vid>'
VID_END_TOKEN = '</vid>'

PATCH_CONTEXT_TOKEN = '<PATCH_CONTEXT>'
PATCH_START_TOKEN = '<patch>'
PATCH_END_TOKEN = '</patch>'


class ImageProcessor:
    def __init__(
        self,
        process_type,
        image_size=448,
        normalize_type="imagenet",
        min_patch_grid=1,
        max_patch_grid=6,
    ):
        self.process_type = process_type
        self.image_size = image_size
        self.IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
        self.IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
        self.IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
        self.OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
        self.OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

        if normalize_type == "imagenet":
            MEAN, STD = self.IMAGENET_DEFAULT_MEAN, self.IMAGENET_DEFAULT_STD
        elif normalize_type == "clip":
            MEAN, STD = self.OPENAI_CLIP_MEAN, self.OPENAI_CLIP_STD
        elif normalize_type == "siglip":
            MEAN, STD = self.IMAGENET_STANDARD_MEAN, self.IMAGENET_STANDARD_STD
        else:
            raise NotImplementedError
        self.mean = MEAN
        self.std = STD

        self.patch_size = image_size
        self.min_patch_grid = min_patch_grid
        self.max_patch_grid = max_patch_grid

        if self.process_type == "anyres":
            self.grid_pinpoints = [
                (i, j)
                for i in range(min_patch_grid, max_patch_grid + 1)
                for j in range(min_patch_grid, max_patch_grid + 1)
            ]
            self.possible_resolutions = [
                [dim * self.patch_size for dim in pair] for pair in self.grid_pinpoints
            ]

        if self.process_type == "dynamic":
            max_num = self.max_patch_grid
            min_num = self.min_patch_grid
            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if i * j <= max_num and i * j >= min_num
            )
            self.target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
            self.possible_resolutions = [
                [dim * self.patch_size for dim in pair] for pair in self.target_ratios
            ]

    def get_frame_paths(self, frame_root, num_frames=8):
        os.makedirs(frame_root, exist_ok=True)

        self.frame_tmpl = "frame-{}-of-{}.jpg"
        return [
            os.path.join(frame_root, self.frame_tmpl.format(i, num_frames))
            for i in range(1, num_frames + 1)
        ]

    def save_video_frames(self, vid_path, max_fps=1, num_frames=8):

        vid = decord.VideoReader(vid_path, num_threads=1)

        step_size = len(vid) / (num_frames + 1)
        # step_size = max(1, step_size)
        fps = vid.get_avg_fps()
        step_size = max(fps / max_fps, step_size)

        # indices = [int(i * step_size) for i in range(1, num_frames + 1)]
        indices = [int(i * step_size) for i in range(0, num_frames)]
        indices = [i for i in indices if i < len(vid)]

        num_frames = len(indices)

        frame_paths = self.get_frame_paths(vid_path + ".saved_frames", num_frames)
        flag = np.all([os.path.exists(p) for p in frame_paths])
        if flag:
            return frame_paths

        images = [vid[i].asnumpy() for i in indices]
        images = [Image.fromarray(arr) for arr in images]

        for im, pth in zip(images, frame_paths):
            im.save(pth)
        return frame_paths

    def get_video_frames(self, vid_path, max_fps=1, num_frames=8):
        vid = decord.VideoReader(vid_path, num_threads=1)

        step_size = len(vid) / (num_frames + 1)
        # step_size = max(1, step_size)
        fps = vid.get_avg_fps()
        step_size = max(fps / max_fps, step_size)

        # indices = [int(i * step_size) for i in range(1, num_frames + 1)]
        indices = [int(i * step_size) for i in range(0, num_frames)]
        indices = [i for i in indices if i < len(vid)]
        
        images = [vid[i].asnumpy() for i in indices]
        images = [Image.fromarray(arr) for arr in images]
        return images

    def process_video(self, video_file_or_dir, max_num_frame=8, max_fps=1):
        if os.path.isdir(video_file_or_dir):
            all_filepath = []
            for root, dirs, files in os.walk(video_file_or_dir):
                for filename in files:
                    if (
                        filename.endswith("png")
                        or filename.endswith("jpeg")
                        or filename.endswith("jpg")
                    ):
                        filepath = os.path.join(root, filename)
                        all_filepath.append(filepath)

            if len(all_filepath) == 0:
                return None

            all_filepath.sort()
            total_frame = len(all_filepath)
            if "ShareGPTVideo" in video_file_or_dir:
                fps = 2
            else:
                fps = 1
            target_frame = int(min(total_frame / fps * max_fps, max_num_frame))
            index = [int(1.0 * total_frame / target_frame) * x for x in range(target_frame)]
            selected_filepath = [all_filepath[x] for x in index]
            img_or_path_list = selected_filepath
        elif os.path.isfile(video_file_or_dir):
            img_or_path_list = self.get_video_frames(
                video_file_or_dir, num_frames=max_num_frame, max_fps=max_fps
            )
        else:
            raise NotImplementedError
        return self.process_images(img_or_path_list), img_or_path_list

    def process_images(self, img_or_path_list):
        if isinstance(img_or_path_list[0], str):
            images = [Image.open(x).convert("RGB") for x in img_or_path_list]
        elif isinstance(img_or_path_list[0], Image.Image):
            images = [x.convert("RGB") for x in img_or_path_list]
        else:
            images = img_or_path_list

        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        image_tensor = torch.ones([len(images), 3, self.image_size, self.image_size])

        for i, image in enumerate(images):
            image = expand2square(image, tuple(int(x * 255) for x in self.mean))

            image = image.resize(
                (self.image_size, self.image_size), resample=Image.Resampling.BICUBIC
            )

            image = np.array(image, dtype=np.float32)
            image = image * 1.0 / 255.0

            mean = np.array(self.mean, dtype=image.dtype)
            std = np.array(self.std, dtype=image.dtype)
            image = (image - mean) / std

            image = torch.tensor(image, dtype=torch.float32)
            image = image.permute(2, 0, 1)

            image_tensor[i] = image

        return image_tensor

    def process_images_with_subpatch(self, img_or_path):
        if self.process_type == "anyres":
            return self.process_anyres(img_or_path)
        if self.process_type == "dynamic":
            return self.process_dynamic(img_or_path)

        if isinstance(img_or_path, str):
            image = Image.open(img_or_path).convert("RGB")
        elif isinstance(img_or_path, Image.Image):
            image = img_or_path.convert("RGB")
        else:
            image = img_or_path

        return self.process_images([images])

    def process_anyres(self, img_or_path):
        if isinstance(img_or_path, str):
            image = Image.open(img_or_path).convert("RGB")
        elif isinstance(img_or_path, Image.Image):
            image = img_or_path.convert("RGB")
        else:
            image = img_or_path

        best_resolution = select_best_resolution(image.size, self.possible_resolutions)
        image_padded = resize_and_pad_image(image, best_resolution)
        patches = divide_to_patches(image_padded, self.patch_size)

        if best_resolution == (self.patch_size, self.patch_size):
            image_patches = [image]
        else:
            image_patches = [image] + patches

        image_patches = self.process_images(image_patches)
        return image_patches, best_resolution

    def process_dynamic(self, img_or_path):
        if isinstance(img_or_path, str):
            image = Image.open(img_or_path).convert("RGB")
        elif isinstance(img_or_path, Image.Image):
            image = img_or_path.convert("RGB")
        else:
            image = img_or_path

        image_patches, best_resolution = dynamic_preprocess(
            image,
            min_num=self.min_patch_grid,
            max_num=self.max_patch_grid,
            image_size=self.patch_size,
            use_thumbnail=True,
        )
        image_patches = self.process_images(image_patches)
        return image_patches, best_resolution


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale
        )

        # Calculate effective and wasted resolutions
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        # processed_images.append(thumbnail_img)
        processed_images = [
            thumbnail_img,
        ] + processed_images
    return processed_images, (target_width, target_height)

image_processor = ImageProcessor(
                        process_type="dynamic",
                        image_size=448,
                        normalize_type="imagenet",
                        min_patch_grid=1,
                        max_patch_grid=12,
                    )

def get_external_inputs(tokens, tokenizer, image_list=None, video_list=None):
    image_token_length = 256
    max_num_frame = 4096
    max_fps = 1

    IMG_CONTEXT_ID = tokenizer(IMG_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    IMG_START_ID = tokenizer(IMG_START_TOKEN, add_special_tokens=False).input_ids
    IMG_END_ID = tokenizer(IMG_END_TOKEN, add_special_tokens=False).input_ids

    VID_CONTEXT_ID = tokenizer(VID_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    VID_START_ID = tokenizer(VID_START_TOKEN, add_special_tokens=False).input_ids
    VID_END_ID = tokenizer(VID_END_TOKEN, add_special_tokens=False).input_ids

    PATCH_CONTEXT_ID = tokenizer(PATCH_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    PATCH_START_ID = tokenizer(PATCH_START_TOKEN, add_special_tokens=False).input_ids
    PATCH_END_ID = tokenizer(PATCH_END_TOKEN, add_special_tokens=False).input_ids

    IMG_TAG_ID = tokenizer(IMG_TAG_TOKEN, add_special_tokens=False).input_ids
    VID_TAG_ID = tokenizer(VID_TAG_TOKEN, add_special_tokens=False).input_ids

    assert len(IMG_CONTEXT_ID) == 1
    assert len(IMG_START_ID) == 1
    assert len(IMG_END_ID) == 1

    assert len(VID_CONTEXT_ID) == 1
    assert len(VID_START_ID) == 1
    assert len(VID_END_ID) == 1

    assert len(PATCH_CONTEXT_ID) == 1
    assert len(PATCH_START_ID) == 1
    assert len(PATCH_END_ID) == 1

    IMG_CONTEXT_ID = IMG_CONTEXT_ID[0]
    IMG_START_ID = IMG_START_ID[0]
    IMG_END_ID = IMG_END_ID[0]

    VID_CONTEXT_ID = VID_CONTEXT_ID[0]
    VID_START_ID = VID_START_ID[0]
    VID_END_ID = VID_END_ID[0]

    PATCH_CONTEXT_ID = PATCH_CONTEXT_ID[0]
    PATCH_START_ID = PATCH_START_ID[0]
    PATCH_END_ID = PATCH_END_ID[0]

    IMG_TAG_ID = IMG_TAG_ID[0]
    VID_TAG_ID = VID_TAG_ID[0]

    nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids
    image_indices = []
    images = []

    # ----------------------------------------------------------------
    # image
    for batch_idx, input_ids in enumerate(tokens):
        img_positions = [i for i, x in enumerate(input_ids) if x == IMG_TAG_ID]
        if len(img_positions) == 0:
            continue

        new_input_ids = []
        st = 0
        for img_idx, img_pos in enumerate(img_positions):
            if image_list is not None:
                image_patches, (best_width, best_height) = image_processor.process_images_with_subpatch(image_list[img_idx])
            images.append(image_patches)

            new_input_ids += input_ids[st:img_pos]
            new_input_ids += [IMG_START_ID]
            image_indice_b = torch.zeros(
                1, image_token_length, dtype=torch.int64
            )
            image_indice_s = (
                torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                .unsqueeze(0)
                .repeat(1, 1)
            )
            image_indice_b_s = torch.stack(
                [image_indice_b, image_indice_s], dim=0
            )
            image_indices.append(image_indice_b_s)

            new_input_ids += [IMG_CONTEXT_ID] * image_token_length

            new_input_ids += [IMG_END_ID]

            if len(image_patches) > 1:
                for i in range(0, best_height, image_processor.patch_size):
                    new_input_ids += nl_tokens
                    for j in range(0, best_width, image_processor.patch_size):
                        new_input_ids += [PATCH_START_ID]

                        image_indice_b = torch.zeros(
                            1, image_token_length, dtype=torch.int64
                        )  # This will change in collate_fn
                        image_indice_s = (
                            torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                            .unsqueeze(0)
                            .repeat(1, 1)
                        )
                        image_indice_b_s = torch.stack(
                            [image_indice_b, image_indice_s], dim=0
                        )  # 2, num_image, image_length
                        image_indices.append(image_indice_b_s)

                        new_input_ids += [PATCH_CONTEXT_ID] * image_token_length
                        new_input_ids += [PATCH_END_ID]
            st = img_pos + 1
        new_input_ids += input_ids[st:]
        input_ids = new_input_ids
        tokens[batch_idx] = input_ids
    # ----------------------------------------------------------------
    # video
    for batch_idx, input_ids in enumerate(tokens):
        # vid_positions = [i for i, x in enumerate(input_ids) if x == VID_CONTEXT_ID]
        vid_positions = [i for i, x in enumerate(input_ids) if x == VID_TAG_ID]
        if len(vid_positions) == 0:
            continue
        if video_list is not None:
            assert len(vid_positions) == len(video_list), f"{vid_positions} {video_list} {VID_CONTEXT_TOKEN} {VID_CONTEXT_ID} {tokens}"

        new_input_ids = []
        st = 0
        for vid_idx, vid_pos in enumerate(vid_positions):
            if video_list is not None:
                video_frames, _ = image_processor.process_video(video_list[vid_idx], max_num_frame, max_fps)
            images.append(video_frames)

            new_input_ids += input_ids[st:vid_pos]

            for _ in video_frames:
                new_input_ids += [VID_START_ID]

                image_indice_b = torch.zeros(
                    1, image_token_length, dtype=torch.int64
                )  # This will change in collate_fn
                image_indice_s = (
                    torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                    .unsqueeze(0)
                    .repeat(1, 1)
                )
                image_indice_b_s = torch.stack(
                    [image_indice_b, image_indice_s], dim=0
                )  # 2, num_image, image_length
                image_indices.append(image_indice_b_s)

                new_input_ids += [VID_CONTEXT_ID] * image_token_length

                new_input_ids += [VID_END_ID]

            st = vid_pos + 1

        new_input_ids += input_ids[st:]

        input_ids = new_input_ids
        tokens[batch_idx] = input_ids

    images = torch.cat(images, dim=0)
    image_indices = torch.cat(image_indices, dim=1)

    image_indices = image_indices.contiguous().to(torch.cuda.current_device())
    images = torch.tensor(images, dtype=torch.bfloat16).contiguous().to(torch.cuda.current_device())
    tokens = torch.tensor(tokens, dtype=torch.long, device='cuda')
    return tokens, images, image_indices


class Long_vitaVideosProcessorKwargs(VideosKwargs, total=False):
    fps: Union[List[float], float]


class Long_vitaProcessorKwargs(ProcessingKwargs, total=False):
    videos_kwargs: Long_vitaVideosProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "videos_kwargs": {"fps": 2.0},
    }

class Long_vitaImagesKwargs(ImagesKwargs, total=False):
    crop_to_patches: Optional[bool]
    min_patches: Optional[int]
    max_patches: Optional[int]


class Long_vitProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Long_vitaImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding_side": "left",
        },
        "images_kwargs": {
            "crop_to_patches": True,
        },
        "videos_kwargs": {
            "crop_to_patches": False,
        },
    }


class Long_vitaProcessor(ProcessorMixin):
    r"""
    Constructs a InternVL processor which wraps a [`AutoImageProcessor`] and
    [`PretrainedTokenizerFast`] tokenizer into a single processor that inherits both the image processor and
    tokenizer functionalities. See the [`~InternVLProcessor.__call__`] and [`~InternVLProcessor.decode`] for more information.
    Args:
        image_processor ([`AutoImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`, `PreTrainedTokenizerFast`], *optional*):
            The tokenizer is a required input.
        image_seq_length (`int`, *optional*, defaults to 256):
            The number of image token to use per image patch. it should be set so that:
            image_seq_length = (config.image_size // config.patch_size) ** 2 * (config.scale_factor**2)
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "image_seq_length"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self, image_processor=None, tokenizer=None, image_seq_length: int = 256, chat_template=None, **kwargs
    ):
        self.image_seq_length = image_seq_length
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        audio=None,
        videos: Optional[VideoInput] = None,
        **kwargs: Unpack[Long_vitProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizerFast.__call__`] to encode the text if `text`
        is not `None`, otherwise encode default OCR queries which depends on the `format`, `box`, `color`, `multi_page` and
        `crop_to_patches` arguments. To prepare the vision inputs, this method forwards the `images` and `kwrags` arguments to
        GotOcr2ImageProcessor's [`~GotOcr2ImageProcessor.__call__`] if `images` is not `None`.
        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if text is None:
            raise ValueError("You have to specify text.")

        output_kwargs = self._merge_kwargs(
            Long_vitProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if not isinstance(text, (list, tuple)):
            text = [text]
        if sum([prompt.count("<video>\n") for prompt in text]) > 0 and not videos:
            videos = []
        for idx, prompt in enumerate(text):
            if prompt.count("<video_path>") > 0:
                pattern = r'<video_path>(.*?)<video_path>'
                videos += re.findall(pattern, prompt)
                prompt = re.sub(pattern, '', prompt)
            while prompt.count("<image>\n") > 1 or prompt.count("<video>\n") > 1:
                if prompt.count("<image>\n") > 1:
                    prompt = prompt.replace("<image>\n", "<image>", 1)
                if prompt.count("<video>\n") > 1:
                    prompt = prompt.replace("<video>\n", "<video>", 1)
            text[idx] = prompt

        inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        inputs, images, image_indices = get_external_inputs(inputs['input_ids'].tolist(), tokenizer=self.tokenizer, image_list=images, video_list=videos)
        image_videos_inputs = {'images': images,
                               'image_indices': image_indices}
        inputs = {
            'input_ids': inputs,
            'attention_mask': torch.ones_like(inputs)
        }
        return BatchFeature(data={**inputs, **image_videos_inputs})

    def sample_indices_fn(
        self, metadata: VideoMetadata, num_frames: int = None, initial_shift: Union[bool, float, int] = True
    ):
        """
        The function to generate indices of frames to sample from a video.
        Args:
            metadata (`VideoMetadata`):
                `VideoMetadata` object containing metadat about the video, such as "total_num_frames" or "fps".
            num_frames (`int`, *optional*):
                Number of frames to sample uniformly. If None, all frames are sampled.
            initial_shift (`bool`, `float` or `int`, defaults to `0`):
                The initial shift to apply when sampling frames. If `True`, the shift is set so that frames are sampled from the middle of the video.
        Returns:
            `np.ndarray`: Array of frame indices to sample.
        """
        if initial_shift is True:
            initial_shift = metadata.total_num_frames / num_frames / 2
        if num_frames is not None:
            indices = np.arange(
                initial_shift, metadata.total_num_frames, metadata.total_num_frames / num_frames
            ).astype(int)
        else:
            indices = np.arange(initial_shift, metadata.total_num_frames).astype(int)

        return indices

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(tokenizer_input_names) + list(image_processor_input_names)

    # Add model-specific video sampling method when applying the template
    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        chat_template: Optional[str] = None,
        num_frames: int = 8,
        initial_shift: Union[bool, float, int] = True,
        video_load_backend="pyav",
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ):
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.
        The input is expected to be in the following format, where each message content is a list consisting of text and
        optionally image or video inputs. One can also provide an image, video, URL or local path which will be used to form
        `pixel_values` when `return_dict=True`. If not provided, one will get only the formatted text, optionally tokenized text.
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "Please describe this image in detail."},
                ],
            },
        ]
        Args:
            conversation (`Union[List[Dict, [str, str]], List[List[Dict[str, str]]]]`):
                The conversation to format.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
                chat template is used.
            num_frames (`int`, *optional*, defaults to 8):
                Number of frames to sample from a video when using the default `sample_indices_fn`.
            initial_shift (`bool`, `float` or `int`, defaults to `0`):
                The initial shift to apply when sampling frames using the default `sample_indices_fn`.
                If `True`, the shift is set so that frames are sampled from the middle of the video.
        """
        sample_indices_fn = kwargs.pop(
            "sample_indices_fn", partial(self.sample_indices_fn, num_frames=num_frames, initial_shift=initial_shift)
        )

        return super().apply_chat_template(
            conversation,
            chat_template,
            video_load_backend=video_load_backend,
            num_frames=num_frames,
            sample_indices_fn=sample_indices_fn,
            **kwargs,
        )


__all__ = ["Long_vitaProcessor"]
