# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
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

"""Tokenization classes and Image processor class, Processor class for Ernie_45T_VL."""

import base64
import copy
import datetime
import hashlib
import io
import math
import os
import random
import threading
import uuid
from collections import defaultdict
from pathlib import Path
from shutil import copyfile
from tempfile import NamedTemporaryFile as ntf
from typing import Any, Dict, List, Optional, Tuple, Union

import decord
import numpy as np
import requests
import sentencepiece as spm
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL.ExifTags import TAGS

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_valid_image,
    make_flat_list_of_images,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy
from transformers.utils import TensorType, logging
from transformers.video_utils import VideoInput


logger = logging.get_logger(__name__)


class Ernie4_5_VLTokenizer(PreTrainedTokenizer):
    """
    Ernie4_5_VLTokenizer
    """

    vocab_files_names = {
        "vocab_file": "tokenizer.model",
    }
    # Model input names expected by the tokenizer
    model_input_names = ["input_ids", "position_ids", "attention_mask", "labels"]
    # Padding side (where to add padding tokens)
    padding_side = "right"

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        cls_token="<cls>",
        eos_token="</s>",
        mask_token="<mask:0>",
        pad_token="<pad>",
        sep_token="<sep>",
        unk_token="<unk>",
        additional_special_tokens=None,
        **kwargs,
    ):
        """
        Initialize the Ernie4_5_VLTokenizer

        Args:
            vocab_file (str): Path to the tokenizer vocabulary model.
            bos_token (str, optional): The beginning of sequence token. Defaults to `"<s>"`.
            cls_token (str, optional): The classifier token. Defaults to `"<cls>"`.
            eos_token (str, optional): The end of sequence token. Defaults to `"</s>"`.
            mask_token (str, optional): The masking token. Defaults to `"<mask:0>"`.
            pad_token (str, optional): The padding token. Defaults to `"<pad>"`.
            sep_token (str, optional): The separation token. Defaults to `"<sep>"`.
            unk_token (str, optional): The unknown tokens symbol. Defaults to `"<unk>"`.
            additional_special_tokens (List[str], optional): Additional special tokens to use.
                Defaults to `["<mask:1>", "<mask:7>"]`.
            **kwargs (dict): Additional keyword arguments passed along to the superclass.
        """

        # Store vocabulary file path
        self.vocab_file = vocab_file
        # Initialize SentencePiece processor
        self.sp_model = spm.SentencePieceProcessor()
        # Load the vocabulary model
        self.sp_model.Load(vocab_file)

        # Set default additional special tokens if none provided
        if additional_special_tokens is None:
            additional_special_tokens = ["<mask:1>", "<mask:7>"]
        super().__init__(
            bos_token=bos_token,
            cls_token=cls_token,
            eos_token=eos_token,
            mask_token=mask_token,
            pad_token=pad_token,
            sep_token=sep_token,
            unk_token=unk_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    @property
    def space_token(self):
        """Return the space token"""
        return "<mask:1>"

    @property
    def space_token_id(self):
        """Return the ID of the space token"""
        return self.sp_model.piece_to_id("<mask:1>")

    @property
    def gend_token(self):
        """Return the gender token"""
        return "<mask:7>"

    @property
    def gend_token_id(self):
        """Return the ID of the gender token"""
        return self.sp_model.piece_to_id("<mask:7>")

    @property
    def im_start_id(self):
        """Return the ID of the image start token"""
        return self.sp_model.piece_to_id("<|im_start|>")

    @property
    def im_end_id(self):
        """Return the ID of the image end token"""
        return self.sp_model.piece_to_id("<|im_end|>")

    @property
    def vocab_size(self):
        """Return the size of the vocabulary"""
        return self.sp_model.vocab_size()

    def get_vocab(self):
        """Return the vocabulary as a dictionary mapping tokens to IDs"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Tokenize the input text into pieces"""
        return self.sp_model.encode_as_pieces(text)

    def _convert_token_to_id(self, token):
        """Convert a token to its corresponding ID"""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, id):
        """Convert an ID to its corresponding token"""
        return self.sp_model.id_to_piece(id)

    def convert_tokens_to_string(self, tokens):
        """Convert a sequence of tokens back to a string"""
        current_sub_tokens = []
        out_string = ""

        for token in tokens:
            # Handle special tokens differently
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)

        # Add any remaining sub-tokens
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def prepare_for_model(self, *args, **kwargs):
        """Prepare the tokenized inputs for the model"""
        # Remove add_special_tokens if present (not supported)
        if "add_special_tokens" in kwargs:
            kwargs.pop("add_special_tokens")
        return super().prepare_for_model(*args, **kwargs)

    def save_vocabulary(
        self, save_directory, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`): The directory to save the vocabulary to
            filename_prefix (`str`, optional): Prefix to add to the filename

        Returns:
            `Tuple(str)`: Paths to the saved files
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # Construct output vocabulary file path
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + self.vocab_files_names["vocab_file"],
        )

        # Copy or create vocabulary file
        if os.path.abspath(self.vocab_file) != os.path.abspath(
            out_vocab_file
        ) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def _decode(self, *args, **kwargs):
        """Decode token_id back to text"""
        # Remove some parameters that aren't used
        kwargs.pop("clean_up_tokenization_spaces", None)
        kwargs.pop("spaces_between_special_tokens", None)

        # Call parent decode method with specific parameters
        return super()._decode(
            *args,
            **kwargs,
            clean_up_tokenization_spaces=False,
            spaces_between_special_tokens=False,
        )

    def _pad(
        self,
        encoded_inputs: Dict,
        max_length: Optional[int] = None,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        **kwargs
    ) -> dict:
        """Pad the encoded inputs to the specified length"""
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names
        if return_attention_mask:
            required_input = encoded_inputs[self.model_input_names[0]]
            if padding_strategy == PaddingStrategy.LONGEST:
                max_length = len(required_input)

            # Adjust max_length if needed for multiple of padding
            if (
                max_length is not None
                and pad_to_multiple_of is not None
                and (max_length % pad_to_multiple_of != 0)
            ):
                max_length = (
                    (max_length // pad_to_multiple_of) + 1
                ) * pad_to_multiple_of

            # Check if padding is needed
            needs_to_be_padded = (
                padding_strategy != PaddingStrategy.DO_NOT_PAD
                and len(required_input) != max_length
            )

            # Handle attention mask if present
            if (
                "attention_mask" in encoded_inputs
                and encoded_inputs["attention_mask"] is not None
            ):
                attention_mask = encoded_inputs.pop("attention_mask")
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = attention_mask.numpy()
                elif isinstance(attention_mask, list):
                    attention_mask = np.array(attention_mask)
                elif not isinstance(attention_mask, np.ndarray):
                    raise ValueError(
                        f"Unexpected type {type(attention_mask)} of attention_mask, "
                    )
            else:
                # Create default attention mask if none provided
                attention_mask = np.tril(
                    np.ones((len(required_input), len(required_input)), dtype=np.int64)
                )
                attention_mask = np.expand_dims(attention_mask, axis=0)

            # Perform padding if needed
            if needs_to_be_padded:
                difference = max_length - len(required_input)
                if self.padding_side == "right":
                    if attention_mask.ndim == 1:
                        pad_width = [(0, difference)]
                    else:
                        pad_width = [(0, 0), (0, difference), (0, difference)]
                elif self.padding_side == "left":
                    if attention_mask.ndim == 1:
                        pad_width = [(difference, 0)]
                    else:
                        pad_width = [(0, 0), (difference, 0), (difference, 0)]
                else:
                    raise ValueError(
                        "Invalid padding strategy:" + str(self.padding_side)
                    )

                attention_mask = np.pad(
                    attention_mask,
                    pad_width=pad_width,
                    mode="constant",
                    constant_values=0,
                )

        # Call parent padding method
        encoded_inputs = super()._pad(
            encoded_inputs,
            max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=False,
        )

        # Add attention mask back if needed
        if return_attention_mask:
            encoded_inputs["attention_mask"] = attention_mask.tolist()

        return encoded_inputs


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 6177 * 28 * 28
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def is_scaled_image(image: np.ndarray) -> bool:
    """
    Checks to see whether the pixel values have already been rescaled to [0, 1].
    """
    if image.dtype == np.uint8:
        return False

    # It's possible the image has pixel values in [0, 255] but is of floating type
    return np.min(image) >= 0 and np.max(image) <= 1


def make_batched_images(images) -> List[List[ImageInput]]:
    """
    Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    if (
        isinstance(images, (list, tuple))
        and isinstance(images[0], (list, tuple))
        and is_valid_image(images[0][0])
    ):
        return [img for img_list in images for img in img_list]

    elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images

    elif is_valid_image(images):
        return [images]

    raise ValueError(f"Could not make batched images from {images}")


# Copied from transformers.models.llava_next_video.image_processing_llava_next_video.make_batched_videos
def make_batched_videos(videos) -> List[VideoInput]:
    """dummy"""
    if (
        isinstance(videos, (list, tuple))
        and isinstance(videos[0], (list, tuple))
        and is_valid_image(videos[0][0])
    ):
        return videos

    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        if isinstance(videos[0], Image.Image):
            return [videos]
        elif len(videos[0].shape) == 4:
            return [list(video) for video in videos]

    elif is_valid_image(videos) and len(videos.shape) == 4:
        return [list(videos)]

    raise ValueError(f"Could not make batched video from {videos}")


class Ernie4_5_VLImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Ernie 4.5 VL image processor that dynamically resizes images based on the original images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        size (`dict[str, int]`, *optional*, defaults to `{"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 6177}`):
            Size of the image after resizing. `shortest_edge` and `longest_edge` keys must be present.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `list[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `list[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel
            in the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_pixels (`int`, *optional*, defaults to `56 * 56`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `28 * 28 * 6177`):
            The max pixels of the image to resize the image.
        patch_size (`int`, *optional*, defaults to 14):
            The spatial patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
    """

    model_input_names = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: Optional[int] = 56 * 56,
        max_pixels: Optional[int] = 6177 * 28 * 28,  # TODO: diff value
        patch_size: int = 14,
        merge_size: int = 2,
        # TODO: no `temporal_patch_size`
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        else:
            size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
        # backward compatibility: override size with min_pixels and max_pixels if they are provided
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
        self.min_pixels = size["shortest_edge"]
        self.max_pixels = size["longest_edge"]
        self.size = size

        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD

        self.patch_size = patch_size
        self.merge_size = merge_size
        self.do_convert_rgb = do_convert_rgb

    def get_smarted_resize(self, height, width, min_pixels=None, max_pixels=None):
        actual_min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        actual_max_pixels = max_pixels if max_pixels is not None else self.max_pixels
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.patch_size * self.merge_size,
            min_pixels=actual_min_pixels,
            max_pixels=actual_max_pixels,
        )
        return (resized_height, resized_width), (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )

    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        predetermined_grid_thw=None,
    ):
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.
            vision_info (`list[Dict]`, *optional*):
                Optional list of dictionaries containing additional information about vision inputs.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. `shortest_edge` and `longest_edge` keys must be present.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the `PILImageResampling` enums.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Scale factor to use if rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                The spatial patch size of the vision encoder.
            temporal_patch_size (`int`, *optional*, defaults to `self.temporal_patch_size`):
                The temporal patch size of the vision encoder.
            merge_size (`int`, *optional*, defaults to `self.merge_size`):
                The merge size of the vision encoder to llm encoder.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.   - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []

        if predetermined_grid_thw is not None:
            assert len(predetermined_grid_thw) == len(
                images
            ), f"len(predetermined_grid_thw) {len(predetermined_grid_thw)} == len(images) {len(images)}"

        for img_idx, image in enumerate(images):
            if do_resize:
                if predetermined_grid_thw is not None:
                    (resized_height, resized_width) = predetermined_grid_thw[img_idx]
                    resized_height *= self.patch_size
                    resized_width *= self.patch_size
                else:
                    resized_height, resized_width = smart_resize(
                        height,
                        width,
                        factor=self.patch_size * self.merge_size,
                        min_pixels=self.min_pixels,
                        max_pixels=self.max_pixels,
                    )

                image = resize(
                    image, size=(resized_height, resized_width), resample=resample, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose([0, 3, 1, 2])

        channel = patches.shape[1]
        grid_t = patches.shape[0]
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = patches.reshape(
            [
                grid_t,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            ]
        )
        # [grid_t, grid_h/merge, grid_w/merge, merge, merge, channel, patch, patch]
        patches = patches.transpose([0, 2, 5, 3, 6, 1, 4, 7])
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * patch_size * patch_size
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        predetermined_grid_thw=None,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            videos (`VideoInput`):
                Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
                passing in videos with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            min_pixels (`int`, *optional*, defaults to `self.min_pixels`):
                The min pixels of the image to resize the image.
            max_pixels (`int`, *optional*, defaults to `self.max_pixels`):
                The max pixels of the image to resize the image.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                The spatial patch size of the vision encoder.
            temporal_patch_size (`int`, *optional*, defaults to `self.temporal_patch_size`):
                The temporal patch size of the vision encoder.
            merge_size (`int`, *optional*, defaults to `self.merge_size`):
                The merge size of the vision encoder to llm encoder.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        """
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        if size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
            min_pixels = size["shortest_edge"]
        elif min_pixels is not None and max_pixels is not None:
            # backward compatibility: override size with min_pixels and max_pixels if they are provided
            size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        else:
            size = {**self.size}

        do_resize = do_resize if do_resize is not None else self.do_resize

        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = None
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if images is not None:
            images = self.fetch_images(images)
            images = make_flat_list_of_images(images)

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        data = {}
        if images is not None:
            pixel_values, vision_grid_thws = [], []
            for img_idx, image in enumerate(images):
                if predetermined_grid_thw is not None:
                    predetermined_grid_thw_one = [predetermined_grid_thw[img_idx]]
                else:
                    predetermined_grid_thw_one = None

                patches, image_grid_thw = self._preprocess(
                    image,
                    do_resize=do_resize,
                    size=size,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=merge_size,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                    predetermined_grid_thw=predetermined_grid_thw_one,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data.update(
                {"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws}
            )

        if videos is not None:
            videos = make_batched_videos(videos)
            pixel_values, vision_grid_thws = [], []
            for images in videos:
                patches, video_grid_thw = self._preprocess(
                    images,
                    do_resize=do_resize,
                    size=size,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=merge_size,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                    predetermined_grid_thw=predetermined_grid_thw,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(video_grid_thw)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)

            data.update(
                {
                    "pixel_values_videos": pixel_values,
                    "video_grid_thw": vision_grid_thws,
                }
            )

        return BatchFeature(data=data, tensor_type=return_tensors)


RAW_VIDEO_DIR = "./download_tmp/raw_video/"
RAW_IMAGE_DIR = "./download_tmp/raw_images/"
EXTRACTED_FRAME_DIR = "./download_tmp/extracted_frames/"
TMP_DIR = "./download_tmp/upload_tmp/"

FONT_PATH = os.path.join(Path(__file__).parent.absolute(), "Roboto-Regular.ttf")
if not os.path.exists(FONT_PATH):
    ttf = requests.get("https://paddlenlp.bj.bcebos.com/vision-language-models/materials/Roboto-Regular.ttf")
    open(FONT_PATH, "wb").write(ttf.content)


def is_gif(data: bytes) -> bool:
    """
    check if a bytes is a gif based on the magic head
    """
    return data[:6] in (b"GIF87a", b"GIF89a")


class VideoReaderWrapper(decord.VideoReader):
    """
    Solving memory leak bug

    https://github.com/dmlc/decord/issues/208
    """

    def __init__(self, video_path, *args, **kwargs):
        with ntf(delete=True, suffix=".gif") as gif_file:
            gif_input = None
            self.original_file = None
            if isinstance(video_path, str):
                self.original_file = video_path
                if video_path.lower().endswith(".gif"):
                    gif_input = video_path
            elif isinstance(video_path, bytes):
                if is_gif(video_path):
                    gif_file.write(video_path)
                    gif_input = gif_file.name
            elif isinstance(video_path, io.BytesIO):
                video_path.seek(0)
                tmp_bytes = video_path.read()
                video_path.seek(0)
                if is_gif(tmp_bytes):
                    gif_file.write(tmp_bytes)
                    gif_input = gif_file.name

            if gif_input is not None:
                try:
                    # moviepy 1.0
                    import moviepy.editor as mp
                except:
                    # moviepy 2.0
                    import moviepy as mp
                clip = mp.VideoFileClip(gif_input)
                mp4_file = ntf(delete=False, suffix=".mp4")
                clip.write_videofile(mp4_file.name, logger=None)
                clip.close()
                video_path = mp4_file.name
                self.original_file = video_path

            super().__init__(video_path, *args, **kwargs)
            self.seek(0)

    def __getitem__(self, key):
        frames = super().__getitem__(key)
        self.seek(0)
        return frames

    def __del__(self):
        if self.original_file and os.path.exists(self.original_file):
            os.remove(self.original_file)


def get_filename(url=None):
    """
    Get Filename
    """
    if url is None:
        return str(uuid.uuid4()).replace("-", "")
    t = datetime.datetime.now()
    if not isinstance(url, bytes):
        url = url.encode("utf-8")

    md5_hash = hashlib.md5(url).hexdigest()
    pid = os.getpid()
    tid = threading.get_ident()

    # Remove the suffix to prevent save-jpg from reporting errors
    image_filname = f"{t.year}-{t.month:02d}-{t.day:02d}-{pid}-{tid}-{md5_hash}"
    return image_filname


def file_download(url, download_dir, save_to_disk=False, retry=0, retry_interval=3):
    """
    Description: Download url, if url is PIL, return directly
    Args:
        url(str, PIL): http/local path/io.Bytes, note that io.Bytes is the image byte stream
        download_path: when save_to_disk=True, return the saved address
        save_to_disk: whether to save in the local path
    """

    if isinstance(url, Image.Image):
        return url
    elif isinstance(url, VideoReaderWrapper):
        return url
    elif url.startswith("http"):
        response = requests.get(url)
        bytes_data = response.content
    elif os.path.isfile(url):
        if save_to_disk:
            return url
        bytes_data = open(url, "rb").read()
    else:
        bytes_data = base64.b64decode(url)
    if not save_to_disk:
        return bytes_data

    download_path = os.path.join(download_dir, get_filename(url))
    Path(download_path).parent.mkdir(parents=True, exist_ok=True)
    with open(download_path, "wb") as f:
        f.write(bytes_data)
    return download_path


def get_downloadable(
    url, download_dir=RAW_VIDEO_DIR, save_to_disk=False, retry=0, retry_interval=3
):
    """download video and store it in the disk

    return downloaded **path** if save_to_disk is set to true
    return downloaded **bytes** if save_to_disk is set to false
    """

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    downloaded_path = file_download(
        url,
        download_dir,
        save_to_disk=save_to_disk,
        retry=retry,
        retry_interval=retry_interval,
    )
    return downloaded_path


def get_downloadable_image(
    download_path, need_exif_info, retry_max_time=0, retry_interval=3
):
    """
    Get downloadable with exif info and image processing
    """

    def get_image_exif(image):
        exif_data = image._getexif()
        exif_info = {}
        if exif_data is not None:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                exif_info[tag_name] = value.strip()
        return exif_info

    def has_transparent_background(img):
        """has_transparent_background"""
        if img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        ):
            # Check for any pixel with alpha channel less than 255 (fully opaque)
            alpha = img.convert("RGBA").split()[-1]
            if alpha.getextrema()[0] < 255:
                return True
        return False

    def add_white_background(img):
        """
        Add a white background to a transparent background image
        """
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        # Create an image with a white background and the same size as the original image
        img_white_background = Image.new("RGBA", img.size, (255, 255, 255))

        # Paste the original image onto a white background
        img_white_background.paste(img, (0, 0), img)

        return img_white_background

    def change_I16_to_L(img):
        """
        Convert image from I;16 mode to L mode
        """
        # Since the point function in I mode only supports addition, subtraction, and multiplication,
        # the following * (1 / 256) cannot be changed to division.
        return img.point(lambda i: i * (1 / 256)).convert("L")

    image = get_downloadable(
        download_path,
        save_to_disk=False,
        retry=retry_max_time,
        retry_interval=retry_interval,
    )
    if isinstance(image, Image.Image):
        pil_image = image
    else:
        pil_image = Image.open(io.BytesIO(image))
    if need_exif_info:
        try:
            exif_info = get_image_exif(pil_image)
        except Exception as why:
            exif_info = {}
    else:
        exif_info = {}

    try:
        if pil_image.mode == "I;16":
            pil_image = change_I16_to_L(pil_image)
        if has_transparent_background(pil_image):
            pil_image = add_white_background(pil_image)
    except Exception as e:
        pass

    return pil_image.convert("RGB"), exif_info


def read_video_decord(video_path, save_to_disk):
    """get reader and meta by decord"""
    video_path = get_downloadable(video_path, save_to_disk=save_to_disk)
    if isinstance(video_path, VideoReaderWrapper):
        video_reader = video_path
    else:
        if isinstance(video_path, bytes):
            video_path = io.BytesIO(video_path)
        video_reader = VideoReaderWrapper(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    video_meta = {"fps": fps, "duration": duration, "num_of_frame": vlen}

    return video_reader, video_meta, video_path


def get_frame_indices(
    vlen,
    target_frames=-1,
    target_fps=-1,
    frames_sample="middle",
    fix_start=None,
    input_fps=-1,
):
    """get_frame_indices"""
    assert frames_sample in ["rand", "middle", "leading"]
    if target_frames > 0:
        assert target_fps <= 0, "target_fps must be negative if target_frames is given."
        if target_frames > vlen:
            acc_samples = vlen
            logger.info(
                f"target_frames={target_frames} is larger than video length {vlen}, "
                f"will sample {acc_samples} frames."
            )
        else:
            acc_samples = target_frames
            logger.debug(
                f"sampling at target_frames={target_frames}, frames_sample={frames_sample}"
            )

        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if frames_sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except Exception as e:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif frames_sample == "leading":
            frame_indices = [x[0] for x in ranges]
        elif frames_sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

    elif target_fps > 0:
        assert (
            target_frames <= 0
        ), "target_frames must be negative if target_fps is given."
        assert input_fps > 0, "input_fps must be provided if target_fps is given."
        logger.info(f"sampling at fps={target_fps}, frames_sample={frames_sample}")
        duration = float(vlen) / input_fps
        delta = (
            1 / target_fps
        )  # gap between frames, this is also the clip length each frame represents
        if frames_sample == "middle":
            frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        elif frames_sample == "leading":
            frame_seconds = np.arange(0, duration, delta)
        if frames_sample == "rand":
            frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
            rand_offset = np.random.rand(*(frame_seconds.shape)) - 0.5
            frame_seconds += rand_offset * delta
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]

    else:
        raise ValueError(
            "Must provide either positive target_fps or positive target_frames."
        )

    return frame_indices


def read_frames_decord(
    video_path,
    video_reader,
    video_meta,
    target_frames=-1,
    target_fps=-1,
    frames_sample="middle",
    fix_start=None,
    save_to_disk=False,
    cache_dir=EXTRACTED_FRAME_DIR,
    frame_indices=None,
    tol=10,
):
    """get frames by decord"""

    if frame_indices is None:
        frame_indices = get_frame_indices(
            video_meta["num_of_frame"],
            target_frames=target_frames,
            target_fps=target_fps,
            frames_sample=frames_sample,
            fix_start=fix_start,
            input_fps=video_meta["fps"],
        )

    frames = []
    for frame_indice_index in range(0, len(frame_indices)):
        frame_indice = frame_indices[frame_indice_index]
        try:
            frames.append(video_reader[frame_indice].asnumpy())  # (T, H, W, C)
        except Exception as e:
            logger.debug(f"encounter error when get frame: {frame_indice}, error: {e}")
            previous_counter = 1
            later_counter = 1
            previous_after_flag = True
            if frame_indice == 0 or frame_indice == len(video_reader) - 1:
                cur_tol = tol * 2
            else:
                cur_tol = tol
            while previous_counter < cur_tol or later_counter < cur_tol:
                if previous_after_flag:
                    if frame_indice - previous_counter < 0:
                        previous_counter += 1
                        previous_after_flag = not previous_after_flag
                        continue
                    try:
                        frames.append(
                            video_reader[frame_indice - previous_counter].asnumpy()
                        )
                        logger.info(
                            f"replace {frame_indice}-th frame with {frame_indice-previous_counter}-th frame"
                        )
                        frame_indices[frame_indice_index] = (
                            frame_indice - previous_counter
                        )
                        break
                    except Exception as e:
                        previous_counter += 1
                else:
                    if frame_indice + later_counter >= len(video_reader):
                        later_counter += 1
                        previous_after_flag = not previous_after_flag
                        continue
                    try:
                        frames.append(
                            video_reader[frame_indice + later_counter].asnumpy()
                        )
                        logger.info(
                            f"replace {frame_indice}-th frame with {frame_indice+later_counter}-th frame"
                        )
                        frame_indices[frame_indice_index] = frame_indice + later_counter
                        break
                    except Exception as e:
                        later_counter += 1
                previous_after_flag = not previous_after_flag

    frames = np.stack(frames, axis=0)
    assert len(frames) == len(
        frame_indices
    ), f"len(frames): {len(frames)} != len(frame_indices): {len(frame_indices)}"

    ret = []

    url_sha1 = get_filename()
    for idx, frame in enumerate(frames):
        tmp = Image.fromarray(frame, "RGB")
        if save_to_disk:
            save_path = os.path.join(cache_dir, f"{url_sha1}", f"{idx}.png")
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            tmp.save(save_path)
            tmp = save_path
        ret.append(tmp)

    time_stamps = [
        frame_idx * video_meta["duration"] / video_meta["num_of_frame"]
        for frame_idx in frame_indices
    ]

    return ret, frame_indices, time_stamps


def render_single_image_with_timestamp(
    image: Image, number: str, rate: float, font_path: str = FONT_PATH
):
    """
    Function: Renders a timestamp to the image of pil.image
    The timestamp size is the rate of min(width, height)
    The font color is black, the outline is white, and the outline size is 10% of the font
    Returns an Image object
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font_size = int(min(width, height) * rate)
    outline_size = int(font_size * 0.1)
    font = ImageFont.truetype(font_path, font_size)
    x = 0
    y = 0

    # Draw a black timestamp with a white border
    draw.text(
        (x, y),
        number,
        font=font,
        fill=(0, 0, 0),
        stroke_width=outline_size,
        stroke_fill=(255, 255, 255),
    )

    return image


def timestamp_converting(time_stamp_in_seconds):
    """
    convert timestamp format from seconds to hr:min:sec
    """
    # get hours
    hours = 0
    while time_stamp_in_seconds >= 3600:
        hours += 1
        time_stamp_in_seconds -= 3600
    # get minutes
    mins = 0
    while time_stamp_in_seconds >= 60:
        mins += 1
        time_stamp_in_seconds -= 60
    time_hours = f"{int(hours):02d}"
    time_mins = f"{int(mins):02d}"
    time_secs = f"{time_stamp_in_seconds:05.02f}"
    fi_time_stamp = time_hours + ":" + time_mins + ":" + time_secs

    return fi_time_stamp


def render_frame_timestamp(frame, timestamp, font_rate=0.1):
    """
    Function, given a frame, render the index in order
    Logic: render the index to the upper left corner of the image
    frame: frame, PIL.Image object
    timestamp: timestamp, in seconds
    font_rate: the ratio of font size to min(wi, hei)
    """
    time_stamp = "time: " + timestamp_converting(timestamp)
    new_frame = render_single_image_with_timestamp(frame, time_stamp, font_rate)

    return new_frame


IDS_TYPE_FLAG = {"text": 0, "image": 1, "video": 2, "audio": 3}


class Ernie4_5_VLProcessor(ProcessorMixin):
    """
    Processes multimodal chat messages into model-ready inputs,
    handling text, images, and videos with 3D positional embeddings.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "spatial_conv_size",
        "temporal_conv_size",
        "image_min_pixels",
        "image_max_pixels",
        "video_min_pixels",
        "video_max_pixels",
        "video_target_frames",
        "video_frames_sample",
        "video_max_frames",
        "video_min_frames",
        "video_fps",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    CLS_TOKEN = "<|begin_of_sentence|>"
    SEP_TOKEN = "<|end_of_sentence|>"
    IMG_START = "<|IMAGE_START|>"
    IMG_END = "<|IMAGE_END|>"
    VID_START = "<|VIDEO_START|>"
    VID_END = "<|VIDEO_END|>"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        spatial_conv_size: int = 2,
        temporal_conv_size: int = 2,
        image_min_pixels: int = 4 * 28 * 28,
        image_max_pixels: int = 6177 * 28 * 28,
        video_min_pixels: int = 299 * 28 * 28,
        video_max_pixels: int = 1196 * 28 * 28,
        video_target_frames: int = -1,
        video_frames_sample: str = "leading",
        video_max_frames: int = 180,
        video_min_frames: int = 16,
        video_fps: int = 2,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.tokenizer.ignored_index = -100

        # Convolution sizes for patch aggregation
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size

        # Pixel constraints
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels

        # Video sampling parameters
        self.target_frames = video_target_frames
        self.frames_sample = video_frames_sample
        self.max_frames = video_max_frames
        self.min_frames = video_min_frames
        self.fps = video_fps

        # Special tokens and IDs
        self.cls_token = self.CLS_TOKEN
        self.sep_token = self.SEP_TOKEN
        self.image_start = self.IMG_START
        self.image_end = self.IMG_END
        self.video_start = self.VID_START
        self.video_end = self.VID_END
        self.image_patch_id = self.tokenizer.convert_tokens_to_ids(
            "<|IMAGE_PLACEHOLDER|>"
        )

        self.token_type_mapping = self._build_token_type_mapping()
        self.is_training = True
        self.role_prefixes = {"system": "", "user": "User: ", "bot": "Assistant: "}

    def _build_token_type_mapping(self) -> Dict[Any, int]:
        mapping = defaultdict(lambda: IDS_TYPE_FLAG["text"])
        for token in (self.IMG_START, self.IMG_END, self.VID_START, self.VID_END):
            mapping[token] = IDS_TYPE_FLAG["image"]
        mapping[self.image_patch_id] = IDS_TYPE_FLAG["image"]
        return mapping

    def train(self) -> None:
        """Enable training mode (produces labels)."""
        self.is_training = True

    def eval(self) -> None:
        """Enable evaluation mode (doesn't produce labels)."""
        self.is_training = False

    def _download_image(
        self,
        item: Dict,
    ):
        """Download image from url and resize it to the specified size."""
        url_info = item.get("image_url", {})
        url = url_info.get("url")
        w = url_info.get("image_width", None)
        h = url_info.get("image_height", None)
        data = get_downloadable(url, download_dir=RAW_IMAGE_DIR, save_to_disk=False)

        img = Image.open(io.BytesIO(data) if isinstance(data, bytes) else data)
        if w and h:
            img = img.resize((w, h))
        return img

    def _download_video(self, item: Dict):
        """Download video from url and resize it to the specified size."""
        url_info = item.get("video_url", {})
        url = url_info.get("url")

        frames = self._load_and_process_video(url, item)

        pixel_stack = np.stack([np.array(f.convert("RGB")) for f in frames], axis=0)
        return pixel_stack

    def process_vision_info(self, messages: List[Dict[str, Any]]):
        """Preprocess messages into lists of text, images, and videos."""
        images = []
        videos = []

        for msg in messages:
            content_items = msg.get("content")
            if not isinstance(content_items, list):
                content_items = [content_items]

            for item in content_items:
                if item.get("type") == "image_url":
                    img = self._download_image(item)
                    images.append(img)
                elif item.get("type") == "video_url":
                    pixel_stack = self._download_video(item)
                    videos.append(pixel_stack)
                    
        return images, videos

    def __call__(
        self,
        text: Union[str, List[str]],
        images: List[Image.Image] = None,
        videos: List[List[Image.Image]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Convert chat messages into model inputs.
        Returns a dict with input_ids, token_type_ids, position_ids, images, grid_thw, image_type_ids, labels.
        """
        outputs = {
            "input_ids": [],
            "token_type_ids": [],
            "position_ids": [],
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "cur_position": 0,
            "pic_cnt": 0,
            "video_cnt": 0,
        }
        if images is None:
            images = []
        if videos is None:
            videos = []
        if not isinstance(text, list):
            text = [text]
            
        texts = text[0]

        new_video_seg = True
        for text_with_image in texts.split(self.VID_START + "<|video@placeholder|>" + self.VID_END):
            new_text_seg = True
            if not new_video_seg:
                self._add_video(videos[outputs["video_cnt"]], outputs)
            for text in text_with_image.split(self.IMG_START + "<|image@placeholder|>" + self.IMG_END):
                if not new_text_seg:
                    self._add_image(images[outputs["pic_cnt"]], outputs)
                self._add_text(text, outputs)
                new_text_seg = False
            new_video_seg = False

        for key in ["cur_position", "pic_cnt", "video_cnt"]:
            outputs.pop(key, None)

        outputs = self._pack_outputs(outputs)
        for key in outputs.keys():
            if isinstance(outputs[key], np.ndarray):
                if key in ["images", "grid_thw"]:
                    outputs[key] = torch.tensor(np.array(outputs[key]))
                else:
                    outputs[key] = torch.tensor(np.array([outputs[key]]))

        return BatchFeature(data=outputs)

    def _add_special_token(self, token: Union[str, int], outputs: Dict) -> None:
        """add special token to outputs"""
        token_id = (
            token
            if isinstance(token, int)
            else self.tokenizer.convert_tokens_to_ids(token)
        )
        outputs["input_ids"].append(token_id)
        outputs["token_type_ids"].append(self.token_type_mapping[token])
        pos = outputs["cur_position"]
        outputs["position_ids"].append([pos] * 3)
        outputs["cur_position"] += 1
    
    def _add_text(self, text: str, outputs: Dict) -> None:
        """add text to outputs"""
        tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        outputs["input_ids"].extend(tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * len(tokens))

        start = outputs["cur_position"]
        for i in range(len(tokens)):
            outputs["position_ids"].append([start + i] * 3)
        outputs["cur_position"] += len(tokens)

    def _add_image(self, img: Image.Image, outputs: Dict) -> None:
        """add image to outputs"""
        outputs["pic_cnt"] += 1
        self._add_special_token(self.IMG_START, outputs)

        patches_h, patches_w = self.image_processor.get_smarted_resize(
            img.height,
            img.width,
            min_pixels=self.image_min_pixels,
            max_pixels=self.image_max_pixels,
        )[1]
        num_tokens = (patches_h * patches_w) // (self.spatial_conv_size**2)

        outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)

        pos_ids = self._compute_3d_positions(
            1, patches_h, patches_w, outputs["cur_position"]
        )
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

        # Preprocess pixels
        ret = self.image_processor.preprocess(
            images=[img],
            do_normalize=True,
            do_rescale=True,
            do_convert_rgb=True,
            #predetermined_grid_thw=np.array([[patches_h, patches_w]]),
            input_data_format=ChannelDimension.LAST,
        )

        outputs["images"].append(ret["pixel_values"])
        outputs["grid_thw"].append(ret["image_grid_thw"])
        outputs["image_type_ids"].append(0)

        self._add_special_token(self.IMG_END, outputs)

    def _add_video(
        self, pixel_stack: np.ndarray, outputs: Dict
    ) -> None:
        outputs["video_cnt"] += 1
        self._add_special_token(self.VID_START, outputs)

        patches_h, patches_w = self.image_processor.get_smarted_resize(
            pixel_stack.shape[1],
            pixel_stack.shape[2],
            min_pixels=self.video_min_pixels,
            max_pixels=self.video_max_pixels,
        )[1]
        num_frames = pixel_stack.shape[0]
        num_tokens = (num_frames * patches_h * patches_w) // (
            self.spatial_conv_size**2 * self.temporal_conv_size
        )

        ret = self.image_processor.preprocess(
            images=None,
            videos=pixel_stack,
            do_normalize=True,
            do_rescale=True,
            predetermined_grid_thw=np.array([[patches_h, patches_w]] * num_frames),
            do_convert_rgb=True,
            input_data_format=ChannelDimension.LAST,
        )

        outputs["images"].append(ret["pixel_values_videos"])
        outputs["grid_thw"].append(ret["video_grid_thw"])
        outputs["image_type_ids"].extend([1] * num_frames)

        outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)

        pos_ids = self._compute_3d_positions(
            num_frames, patches_h, patches_w, outputs["cur_position"]
        )
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

        self._add_special_token(self.VID_END, outputs)

    def _load_and_process_video(self, url: str, item: Dict) -> List[Image.Image]:
        reader, meta, path = read_video_decord(url, save_to_disk=False)

        video_frame_args = dict()
        video_frame_args["fps"] = item.get("fps", self.fps)
        video_frame_args["min_frames"] = item.get("min_frames", self.min_frames)
        video_frame_args["max_frames"] = item.get("max_frames", self.max_frames)
        video_frame_args["target_frames"] = item.get(
            "target_frames", self.target_frames
        )
        video_frame_args["frames_sample"] = item.get(
            "frames_sample", self.frames_sample
        )

        video_frame_args = self._set_video_frame_args(video_frame_args, meta)

        frames_data, _, timestamps = read_frames_decord(
            path,
            reader,
            meta,
            target_frames=video_frame_args["target_frames"],
            target_fps=video_frame_args["fps"],
            frames_sample=video_frame_args["frames_sample"],
            save_to_disk=False,
        )

        frames: List[Image.Image] = []
        for img_array, ts in zip(frames_data, timestamps):
            frames.append(render_frame_timestamp(img_array, ts))
        # Ensure even number of frames for temporal conv
        if len(frames) % 2 != 0:
            frames.append(copy.deepcopy(frames[-1]))
        return frames

    def _set_video_frame_args(self, video_frame_args, video_meta):
        """
        Set the final frame extraction parameters based on known parameters and priorities
        """
        # Priority: video_target_frames > (video_min_frames, video_max_frames) > video_fps
        if video_frame_args["target_frames"] > 0:
            if video_frame_args["fps"] >= 0:
                raise ValueError("fps must be negative if target_frames is given")
            if (
                video_frame_args["min_frames"] > 0
                and video_frame_args["target_frames"] < video_frame_args["min_frames"]
            ):
                raise ValueError("target_frames must be larger than min_frames")
            if (
                video_frame_args["max_frames"] > 0
                and video_frame_args["target_frames"] > video_frame_args["max_frames"]
            ):
                raise ValueError("target_frames must be smaller than max_frames")
        else:
            if video_frame_args["fps"] < 0:
                raise ValueError(
                    "Must provide either positive target_fps or positive target_frames."
                )
            # First calculate the number of frames extracted under video_fps
            frames_to_extract = int(video_meta["duration"] * video_frame_args["fps"])
            # Determine whether it is within the target range. If not, take target_frames as the upper or lower bound
            if (
                video_frame_args["min_frames"] > 0
                and video_frame_args["max_frames"] > 0
                and video_frame_args["min_frames"] > video_frame_args["max_frames"]
            ):
                raise ValueError("min_frames must be smaller than max_frames")
            if (
                video_frame_args["min_frames"] > 0
                and frames_to_extract < video_frame_args["min_frames"]
            ):
                video_frame_args["target_frames"] = video_frame_args["min_frames"]
                video_frame_args["fps"] = -1
            if (
                video_frame_args["max_frames"] > 0
                and frames_to_extract > video_frame_args["max_frames"]
            ):
                video_frame_args["target_frames"] = video_frame_args["max_frames"]
                video_frame_args["fps"] = -1

        return video_frame_args

    def _compute_3d_positions(
        self, t: int, h: int, w: int, start_idx: int
    ) -> List[List[int]]:
        # Downsample time if needed
        t_eff = t // self.temporal_conv_size if t != 1 else 1
        gh, gw = h // self.spatial_conv_size, w // self.spatial_conv_size
        time_idx = np.repeat(np.arange(t_eff), gh * gw)
        h_idx = np.tile(np.repeat(np.arange(gh), gw), t_eff)
        w_idx = np.tile(np.arange(gw), t_eff * gh)

        coords = list(zip(time_idx, h_idx, w_idx))
        return [
            [start_idx + ti, start_idx + hi, start_idx + wi] for ti, hi, wi in coords
        ]

    def _pack_outputs(self, outs: Dict) -> Dict[str, Any]:
        # Stack or nullify image-related fields
        if not outs["images"]:
            outs["images"] = None
            outs["grid_thw"] = None
            outs["image_type_ids"] = None
        else:
            outs["images"] = np.vstack(outs["images"])
            outs["grid_thw"] = np.vstack(outs["grid_thw"])
            outs["image_type_ids"] = np.array(outs["image_type_ids"])

        # Convert lists to arrays
        outs["input_ids"] = np.array(outs["input_ids"], dtype=np.int64)
        outs["token_type_ids"] = np.array(outs["token_type_ids"], dtype=np.int64)
        outs["position_ids"] = np.array(outs["position_ids"], dtype=np.int64)
        return outs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Ernie4_5_VLTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Ernie4_5_VLTokenizer's [`~PreTrainedTokenizer.decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """get model input names"""
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(tokenizer_input_names) + list(image_processor_input_names)


__all__ = ["Ernie4_5_VLTokenizer", "Ernie4_5_VLImageProcessor", "Ernie4_5_VLProcessor"]
