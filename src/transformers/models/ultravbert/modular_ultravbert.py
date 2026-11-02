# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict
from torchvision.transforms.v2 import functional as tvF

from ...activations import ACT2FN
from ...configuration_utils import SubConfigSpec
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring
from ...utils.generic import can_return_tuple
from ..auto import AutoConfig
from ..idefics3.image_processing_idefics3 import (
    MAX_IMAGE_SIZE,
    Idefics3ImageProcessor,
    Idefics3ImageProcessorKwargs,
    _resize_output_size_rescale_to_max_len,
    _resize_output_size_scale_below_upper_bound,
    get_device_from_images,
    get_max_height_width,
    get_num_channels,
)
from ..modernvbert.configuration_modernvbert import ModernVBertConfig
from ..modernvbert.modeling_modernvbert import (
    ModernVBertBaseModelOutput,
    ModernVBertConnector,
    ModernVBertForMaskedLM,
    ModernVBertForSequenceClassification,
    ModernVBertForTokenClassification,
    ModernVBertMaskedLMOutput,
    ModernVBertModel,
    ModernVBertPreTrainedModel,
)
from ..ultrabert.modeling_ultrabert import UltrabertRMSNorm


@auto_docstring(checkpoint="athrael-soju/UltraVBERT")
@strict
class UltraVBertConfig(ModernVBertConfig):
    r"""
    pixel_shuffle_factor (`int`, *optional*, defaults to 4):
        Side of the square of vision patches that the connector merges into one image token.
    initializer_cutoff_factor (`float`, *optional*, defaults to 2.0):
        The cutoff factor for the truncated_normal_initializer for initializing all weight matrices.
    classifier_pooling (`Literal["cls", "mean"]`, *optional*, defaults to `"cls"`):
        The pooling strategy to use for classification tasks.
    """

    model_type = "ultravbert"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="ultrabert"),
        "vision_config": SubConfigSpec(
            config_class=AutoConfig, model_type="siglip_vision_model", init_kwargs={"vision_use_head": False}
        ),
    }

    image_token_id: int = 49191
    classifier_bias = AttributeError()


def get_tile_grid(height: int, width: int, longest_edge: int, tile_size: int) -> tuple[int, int]:
    height, width = _resize_output_size_rescale_to_max_len(height, width, max_len=longest_edge)
    height, width = _resize_output_size_scale_below_upper_bound(height, width, max_len=MAX_IMAGE_SIZE)
    aspect_ratio = width / height
    if width >= height:
        cols = math.ceil(width / tile_size)
        rows = math.ceil(int(cols * tile_size / aspect_ratio) / tile_size)
    else:
        rows = math.ceil(height / tile_size)
        cols = math.ceil(int(rows * tile_size * aspect_ratio) / tile_size)
    return rows, cols


def get_capped_longest_edge(height: int, width: int, longest_edge: int, tile_size: int, max_image_tiles: int) -> int:
    while longest_edge > tile_size:
        rows, cols = get_tile_grid(height, width, longest_edge, tile_size)
        if rows * cols <= max_image_tiles:
            break
        longest_edge -= tile_size
    return longest_edge


class UltraVBertImageProcessorKwargs(Idefics3ImageProcessorKwargs, total=False):
    r"""
    do_image_splitting (`bool`, *optional*, defaults to `True`):
        Whether to split the image into sub-images concatenated with the original image. They are split into patches
        such that each patch has a size of `max_image_size["height"]` x `max_image_size["width"]`.
    max_image_size (`Dict`, *optional*, defaults to `{"longest_edge": 512}`):
        Maximum resolution of the patches of images accepted by the model. This is a dictionary containing the key "longest_edge".
    return_row_col_info (`bool`, *optional*, defaults to `False`):
        Whether to return the row and column information of the images.
    max_image_tiles (`int`, *optional*, defaults to 27):
        Largest number of tiles an image splits into, the global image excluded. An image that would split into more
        is resized to a longest edge lowered in steps of `max_image_size["longest_edge"]` until it fits.
    """

    max_image_tiles: int


class UltraVBertImageProcessor(Idefics3ImageProcessor):
    size = {"longest_edge": 3072}
    max_image_size = {"longest_edge": 512}
    max_image_tiles = 27
    valid_kwargs = UltraVBertImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[UltraVBertImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[UltraVBertImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[list["torch.Tensor"]],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        do_image_splitting: bool | None,
        max_image_size: dict[str, int] | None,
        max_image_tiles: int,
        return_row_col_info: bool | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(
            images, is_nested=True, disable_grouping=disable_grouping
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                resize_size = size
                # Idefics3 resizes to `size`, while this lowers the longest edge until the page fits `max_image_tiles`.
                if do_image_splitting:
                    longest_edge = get_capped_longest_edge(
                        *stacked_images.shape[-2:], size.longest_edge, max_image_size["longest_edge"], max_image_tiles
                    )
                    resize_size = SizeDict(longest_edge=longest_edge)
                stacked_images = self.resize(stacked_images, resize_size, resample=resample)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index, is_nested=True)

        grouped_images, grouped_images_index = group_images_by_shape(
            resized_images, is_nested=True, disable_grouping=disable_grouping
        )
        split_images_grouped = {}
        if do_image_splitting:
            rows_grouped = {}
            cols_grouped = {}
            for shape, stacked_images in grouped_images.items():
                stacked_images = self.resize_for_vision_encoder(
                    stacked_images, max_image_size["longest_edge"], resample=resample
                )
                stacked_images, rows, cols = self.split_images(
                    stacked_images, max_image_size=max_image_size, resample=resample
                )
                split_images_grouped[shape] = stacked_images
                rows_grouped[shape] = rows
                cols_grouped[shape] = cols
            processed_images = reorder_images(split_images_grouped, grouped_images_index, is_nested=True)
            rows = reorder_images(rows_grouped, grouped_images_index, is_nested=True)
            cols = reorder_images(cols_grouped, grouped_images_index, is_nested=True)
            for i, group_images in enumerate(processed_images):
                processed_images[i] = [image for sublist in group_images for image in sublist]
        else:
            for shape, stacked_images in grouped_images.items():
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=max_image_size["longest_edge"], width=max_image_size["longest_edge"]),
                    resample=resample,
                )
                split_images_grouped[shape] = stacked_images
            processed_images = reorder_images(split_images_grouped, grouped_images_index, is_nested=True)
            rows = [[0] * len(images) for images in processed_images]
            cols = [[0] * len(images) for images in processed_images]

        grouped_images, grouped_images_index = group_images_by_shape(
            processed_images, is_nested=True, disable_grouping=disable_grouping
        )
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index, is_nested=True)
        if do_pad:
            max_num_images = max(len(images_) for images_ in processed_images)
            max_height, max_width = get_max_height_width(processed_images)
            num_channels = get_num_channels(processed_images)
            device = get_device_from_images(processed_images)

            processed_images_padded = torch.zeros(
                len(processed_images),
                max_num_images,
                *(num_channels, max_height, max_width),
                device=device,
            )
            pixel_attention_masks = torch.zeros(
                len(processed_images),
                max_num_images,
                *(max_height, max_width),
                dtype=torch.int64,
                device=device,
            )
            for i, images in enumerate(processed_images):
                for j, image in enumerate(images):
                    processed_images_padded[i, j], pixel_attention_masks[i, j] = self.pad(
                        image, (max_height, max_width)
                    )
            data = {"pixel_values": processed_images_padded, "pixel_attention_mask": pixel_attention_masks}
        elif return_tensors == "pt":
            data = {"pixel_values": torch.stack([torch.stack(images) for images in processed_images])}
        else:
            data = {"pixel_values": processed_images}
        encoding = BatchFeature(data=data, tensor_type=return_tensors)

        if return_row_col_info:
            encoding["rows"] = rows
            encoding["cols"] = cols

        return encoding

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs: dict | None = None
    ) -> tuple[int, int, int]:
        """
        Returns the number of image patches for an image of the given size, with the `max_image_tiles` cap applied.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*):
                Any kwargs to override defaults of the image processor.
        Returns:
            `tuple[int, int, int]`: Number of patches per image, and the number of rows and columns they form.
        """
        images_kwargs = images_kwargs or {}
        if not images_kwargs.get("do_image_splitting", self.do_image_splitting):
            return 0, 0, 0
        size = images_kwargs.get("size", self.size)
        tile_size = images_kwargs.get("max_image_size", self.max_image_size)["longest_edge"]
        max_image_tiles = images_kwargs.get("max_image_tiles", self.max_image_tiles)
        longest_edge = get_capped_longest_edge(height, width, size["longest_edge"], tile_size, max_image_tiles)
        num_rows, num_cols = get_tile_grid(height, width, longest_edge, tile_size)
        if num_rows * num_cols == 1:
            return 0, 0, 0
        return num_rows * num_cols + 1, num_rows, num_cols


@auto_docstring
@dataclass
class UltraVBertBaseModelOutput(ModernVBertBaseModelOutput):
    pass


@auto_docstring
@dataclass
class UltraVBertMaskedLMOutput(ModernVBertMaskedLMOutput):
    pass


class UltraVBertRMSNorm(UltrabertRMSNorm):
    pass


class UltraVBertConnector(ModernVBertConnector):
    def __init__(self, config):
        super().__init__(config)
        # UltraBERT applies no norm to its input embeddings, so this norm sets the image tokens' scale.
        self.norm = UltraVBertRMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.pixel_shuffle_factor)
        return self.norm(self.modality_projection(image_hidden_states))


@auto_docstring
class UltraVBertPreTrainedModel(ModernVBertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"vision_model\.head\."]


@auto_docstring
class UltraVBertModel(ModernVBertModel):
    @can_return_tuple
    @auto_docstring(checkpoint="athrael-soju/UltraVBERT")
    def forward(self, **super_kwargs):
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_images, image_size, image_size)`, *optional*):
            Mask over the real pixels of each image tile.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        """
        return super().forward(**super_kwargs)


class UltraVBertPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.prediction_head_size, bias=False)
        self.linear2 = nn.Linear(config.prediction_head_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN["gelu_pytorch_tanh"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act_fn(self.linear1(hidden_states))
        return self.act_fn(self.linear2(hidden_states))


@auto_docstring
class UltraVBertForMaskedLM(ModernVBertForMaskedLM):
    _tied_weights_keys = {"lm_head.weight": "model.text_model.embed_tokens.weight"}

    def __init__(self, config: UltraVBertConfig):
        super().__init__(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, self.vocab_size, bias=False)

    @can_return_tuple
    @auto_docstring(checkpoint="athrael-soju/UltraVBERT")
    def forward(self, **super_kwargs):
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_images, image_size, image_size)`, *optional*):
            Mask over the real pixels of each image tile.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        """
        return super().forward(**super_kwargs)


class UltraVBertForSequenceClassification(ModernVBertForSequenceClassification):
    def __init__(self, config: UltraVBertConfig):
        super().__init__(config)
        self.head = nn.Identity()

    @can_return_tuple
    @auto_docstring(checkpoint="athrael-soju/UltraVBERT")
    def forward(self, **super_kwargs):
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_images, image_size, image_size)`, *optional*):
            Mask over the real pixels of each image tile.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        """
        return super().forward(**super_kwargs)


class UltraVBertForTokenClassification(ModernVBertForTokenClassification):
    def __init__(self, config: UltraVBertConfig):
        super().__init__(config)
        self.head = nn.Identity()

    @can_return_tuple
    @auto_docstring(checkpoint="athrael-soju/UltraVBERT")
    def forward(self, **super_kwargs):
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_images, image_size, image_size)`, *optional*):
            Mask over the real pixels of each image tile.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        """
        return super().forward(**super_kwargs)


__all__ = [
    "UltraVBertConfig",
    "UltraVBertImageProcessor",
    "UltraVBertPreTrainedModel",
    "UltraVBertModel",
    "UltraVBertForMaskedLM",
    "UltraVBertForSequenceClassification",
    "UltraVBertForTokenClassification",
]
