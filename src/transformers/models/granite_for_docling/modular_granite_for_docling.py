# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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
"""PyTorch GraniteForDocling model."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...image_processing_utils import BatchFeature
from ...image_transforms import to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    infer_channel_dimension_format,
    make_nested_list_of_images,
)
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AddedToken
from ...utils import TensorType, TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.import_utils import requires
from ...utils.output_capturing import capture_outputs
from ..got_ocr2.image_processing_got_ocr2 import GotOcr2ImageProcessor, GotOcr2ImageProcessorKwargs
from ..got_ocr2.image_processing_pil_got_ocr2 import GotOcr2ImageProcessorPil
from ..granite.configuration_granite import GraniteConfig
from ..granite.modeling_granite import (
    GraniteAttention,
    GraniteDecoderLayer,
    GraniteModel,
    GranitePreTrainedModel,
    GraniteRMSNorm,
    GraniteRotaryEmbedding,
)
from ..granitemoeshared.modeling_granitemoeshared import GraniteMoeSharedMLP
from ..idefics3.configuration_idefics3 import Idefics3Config, Idefics3VisionConfig
from ..idefics3.modeling_idefics3 import (
    Idefics3BaseModelOutputWithPast,
    Idefics3CausalLMOutputWithPast,
    Idefics3Connector,
    Idefics3ForConditionalGeneration,
    Idefics3Model,
    Idefics3PreTrainedModel,
    Idefics3VisionAttention,
    Idefics3VisionEmbeddings,
)
from ..idefics3.processing_idefics3 import Idefics3Processor


logger = logging.get_logger(__name__)

# The tokenizer has tile position markers from `<row_1_col_1>` to `<row_16_col_16>`.
MAX_TILES_PER_SIDE = 16


@auto_docstring(checkpoint="docling-project/granite-for-docling-500m")
@strict
class GraniteForDoclingVisionConfig(Idefics3VisionConfig):
    r"""
    Configuration of the SigLIP-style vision encoder that embeds every 512x512 tile of a page. The outputs of the
    layers listed in `GraniteForDoclingConfig.deepstack_visual_indexes` are tapped as well.

    Example:

    ```python
    >>> from transformers import GraniteForDoclingVisionConfig
    >>> from transformers.models.granite_for_docling.modeling_granite_for_docling import GraniteForDoclingVisionTransformer

    >>> # Initializing a GraniteForDoclingVisionConfig with docling-project/granite-for-docling-500m style configuration
    >>> configuration = GraniteForDoclingVisionConfig()

    >>> # Initializing a GraniteForDoclingVisionTransformer (with random weights) from the docling-project/granite-for-docling-500m style configuration
    >>> model = GraniteForDoclingVisionTransformer(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "granite_for_docling_vision"

    hidden_size: int = 768
    num_attention_heads: int = 12
    image_size: int | list[int] | tuple[int, int] = 512
    patch_size: int | list[int] | tuple[int, int] = 16


@auto_docstring(checkpoint="docling-project/granite-for-docling-500m")
@strict
class GraniteForDoclingTextConfig(GraniteConfig):
    r"""
    shared_intermediate_size (`int`, *optional*, defaults to 2048):
        Dimension of the gated MLP of each decoder layer.
    """

    model_type = "granite_for_docling_text"
    base_config_key = "text_config"
    base_model_tp_plan = None
    mlp_bias = AttributeError()

    vocab_size: int = 133800
    hidden_size: int = 1024
    intermediate_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int | None = 4
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-5
    pad_token_id: int | None = 100256
    bos_token_id: int | None = 100257
    eos_token_id: int | list[int] | None = 100257
    tie_word_embeddings: bool = True
    embedding_multiplier: float | int = 12.0
    logits_scaling: float | int = 4.0
    residual_multiplier: float | int = 0.263
    attention_multiplier: float | int = 0.015625
    shared_intermediate_size: int = 2048


@auto_docstring(checkpoint="docling-project/granite-for-docling-500m")
@strict
class GraniteForDoclingConfig(Idefics3Config):
    r"""
    scale_factor (`int`, *optional*, defaults to 4):
        Pixel shuffle factor of the connector. Each tile of `(image_size // patch_size) ** 2` vision tokens is
        reduced by `scale_factor ** 2`.
    deepstack_visual_indexes (`list[int]`, *optional*, defaults to `[3, 7, 10]`):
        Indices of the vision encoder layers whose output is projected and added to the image token positions after
        the corresponding `deepstack_attn_layers`. Index 0 is the output of the first layer.
    deepstack_attn_layers (`list[int]`, *optional*, defaults to `[0, 1, 2]`):
        Text decoder layers after which the corresponding DeepStack features are added.
    num_mtp_layers (`int`, *optional*, defaults to 0):
        Number of multi-token prediction heads on top of the decoder. From the hidden state at position `t`, the
        language modeling head predicts token `t + 1` and head `i` predicts token `t + i + 2`, so head 0 predicts the
        token after the one `lm_head` predicts. They add an auxiliary loss when `labels` are given. Serving engines
        such as vLLM can use the same heads for speculative decoding; `generate` does not.
    mtp_num_attention_heads (`int`, *optional*):
        Number of attention heads in each multi-token prediction head. Defaults to `text_config.num_attention_heads`.
    mtp_intermediate_size (`int`, *optional*):
        Feed-forward size of each multi-token prediction head. Defaults to `text_config.intermediate_size`.
    mtp_loss_weight (`float`, *optional*, defaults to 0.3):
        Weight of the multi-token prediction loss, averaged over the heads, relative to the language modeling loss.
    use_fine_route (`bool`, *optional*, defaults to `True`):
        Whether to build the fine connector path, which shuffles pixels by half of `scale_factor` and yields four
        times as many image tokens per tile. Checkpoints trained with the coarse path only set it to `False`.
    density_router_hidden_size (`int`, *optional*):
        Hidden size of the density router, which predicts from the vision encoder features whether a page needs the
        fine connector path. No router is built when `None`.
    density_router_threshold (`float`, *optional*, defaults to 0.4):
        Probability above which the density router selects the fine connector path.
    density_router_loss_weight (`float`, *optional*, defaults to 0.1):
        Weight of the density router loss when `router_labels` are given.
    """

    model_type = "granite_for_docling"
    sub_configs = {"text_config": GraniteForDoclingTextConfig, "vision_config": GraniteForDoclingVisionConfig}

    image_token_id: int = 100337
    tie_word_embeddings: bool = True
    scale_factor: int = 4
    pad_token_id: int | None = 100256
    deepstack_visual_indexes: list[int] | None = None
    deepstack_attn_layers: list[int] | None = None
    num_mtp_layers: int = 0
    mtp_num_attention_heads: int | None = None
    mtp_intermediate_size: int | None = None
    mtp_loss_weight: float = 0.3
    use_fine_route: bool = True
    density_router_hidden_size: int | None = None
    density_router_threshold: float = 0.4
    density_router_loss_weight: float = 0.1

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = GraniteForDoclingVisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = GraniteForDoclingVisionConfig(**self.vision_config)

        if self.text_config is None:
            self.text_config = GraniteForDoclingTextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = GraniteForDoclingTextConfig(**self.text_config)

        if self.deepstack_visual_indexes is None:
            self.deepstack_visual_indexes = [3, 7, 10]
        if self.deepstack_attn_layers is None:
            self.deepstack_attn_layers = [0, 1, 2]
        if len(self.deepstack_visual_indexes) != len(self.deepstack_attn_layers):
            raise ValueError("`deepstack_visual_indexes` and `deepstack_attn_layers` must have the same length.")
        if self.density_router_hidden_size is not None and not self.use_fine_route:
            raise ValueError("The density router selects the fine connector path, it needs `use_fine_route=True`.")
        PreTrainedConfig.__post_init__(**kwargs)


class GraniteForDoclingImageProcessorKwargs(GotOcr2ImageProcessorKwargs):
    r"""
    crop_to_patches (`bool`, *optional*, defaults to `True`):
        Whether to split the page into 512x512 tiles laid out on the grid that best matches its aspect ratio (at
        most 16 tiles per side). Can be overridden by the `crop_to_patches` parameter in the `preprocess` method.
    min_patches (`int`, *optional*, defaults to 1):
        The minimum number of tiles per page. Only has an effect if `crop_to_patches` is set to `True`. Can be
        overridden by the `min_patches` parameter in the `preprocess` method.
    max_patches (`int`, *optional*, defaults to 32):
        The maximum number of tiles per page. Only has an effect if `crop_to_patches` is set to `True`. Can be
        overridden by the `max_patches` parameter in the `preprocess` method.
    fine_route (`bool`, *optional*, defaults to `False`):
        Whether to route every tile through the fine connector path, which yields four times as many image tokens
        per tile.
    """

    fine_route: bool


@lru_cache(maxsize=10)
def get_all_supported_aspect_ratios(min_image_tiles: int, max_image_tiles: int) -> list[tuple[int, int]]:
    """
    Computes all `(num_columns, num_rows)` tile grids holding between `min_image_tiles` and `max_image_tiles` tiles,
    with at most `MAX_TILES_PER_SIDE` tiles per side.
    """
    max_tiles_per_side = min(max_image_tiles, MAX_TILES_PER_SIDE)
    aspect_ratios = [
        (width, height)
        for width in range(1, max_tiles_per_side + 1)
        for height in range(1, max_tiles_per_side + 1)
        if min_image_tiles <= width * height <= max_image_tiles
    ]
    return sorted(aspect_ratios, key=lambda x: x[0] * x[1])


@lru_cache(maxsize=100)
def get_optimal_tiled_canvas(
    original_image_size: tuple[int, int],
    target_tile_size: tuple[int, int],
    min_image_tiles: int,
    max_image_tiles: int,
) -> tuple[int, int]:
    """
    Given a minimum and maximum number of tiles, find the canvas with the closest aspect ratio to the
    original image aspect ratio.
    In case of tie-breaking condition when two canvases have the same aspect ratio difference, we favor the canvas with
    more tiles, until the area covered by the tiles is more than twice the target area, in order to avoid unnecessarily
    excessive tiling.
    """
    possible_tile_arrangements = get_all_supported_aspect_ratios(min_image_tiles, max_image_tiles)

    original_height, original_width = original_image_size
    target_tile_height, target_tile_width = target_tile_size
    aspect_ratio = original_width / original_height
    area = original_width * original_height

    best_ratio_diff = float("inf")
    best_grid = (1, 1)
    for grid in possible_tile_arrangements:
        grid_aspect_ratio = grid[0] / grid[1]
        ratio_diff = abs(aspect_ratio - grid_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_grid = grid
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * target_tile_height * target_tile_width * grid[0] * grid[1]:
                best_grid = grid

    return best_grid


@auto_docstring(
    custom_intro="""
    Splits a page into 512x512 tiles on the grid that best matches its aspect ratio (`crop_to_patches`, at most
    `max_patches` tiles and 16 per side), adds a thumbnail of the whole page when there is more than one tile, and
    pads the samples of a batch to the same number of tiles with all-zero tiles. `fine_route=True` marks every tile
    for the fine connector path (`tile_fine_mask`), which yields four times as many image tokens per tile.
    """
)
class GraniteForDoclingImageProcessor(GotOcr2ImageProcessor):
    valid_kwargs = GraniteForDoclingImageProcessorKwargs
    size = {"height": 512, "width": 512}
    crop_to_patches = True
    max_patches = 32
    fine_route = False

    def _prepare_images_structure(self, images: ImageInput, expected_ndims: int = 3) -> ImageInput:
        images = self.fetch_images(images)
        return make_nested_list_of_images(images, expected_ndims=expected_ndims)

    def _preprocess(
        self,
        images: list[list["torch.Tensor"]],
        size: SizeDict,
        resample,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        crop_to_patches: bool = True,
        min_patches: int = 1,
        max_patches: int = 32,
        fine_route: bool = False,
        **kwargs,
    ) -> BatchFeature:
        pixel_values, rows, cols = [], [], []
        for sample in images:
            sample_tiles, sample_rows, sample_cols = [], [], []
            for image in sample:
                if crop_to_patches:
                    num_cols, num_rows = get_optimal_tiled_canvas(
                        tuple(image.shape[-2:]), (size.height, size.width), min_patches, max_patches
                    )
                    tiles = self.crop_image_to_patches(
                        image[None], min_patches, max_patches, patch_size=size, resample=resample
                    )[0]
                else:
                    num_cols = num_rows = 1
                    tiles = self.resize(image, size, resample=resample)[None]
                sample_tiles.append(tiles)
                sample_rows.append(num_rows)
                sample_cols.append(num_cols)
            # A text-only sample of the batch has no tiles
            if sample_tiles:
                sample_tiles = self.rescale_and_normalize(
                    torch.cat(sample_tiles), do_rescale, rescale_factor, do_normalize, image_mean, image_std
                )
            pixel_values.append(sample_tiles)
            rows.append(sample_rows)
            cols.append(sample_cols)

        # Pad the samples to the same number of tiles with all-zero tiles, which the model discards.
        max_num_tiles = max(len(tiles) for tiles in pixel_values)
        first_tiles = next(tiles for tiles in pixel_values if len(tiles) > 0)
        padded_pixel_values = torch.zeros(
            len(pixel_values),
            max_num_tiles,
            *first_tiles.shape[1:],
            dtype=first_tiles.dtype,
            device=first_tiles.device,
        )
        tile_fine_mask = torch.zeros(len(pixel_values), max_num_tiles, dtype=torch.bool)
        for i, tiles in enumerate(pixel_values):
            if len(tiles) > 0:
                padded_pixel_values[i, : tiles.shape[0]] = tiles
                tile_fine_mask[i, : tiles.shape[0]] = fine_route

        data = {"pixel_values": padded_pixel_values}
        if fine_route:
            data["tile_fine_mask"] = tile_fine_mask
        encoding = BatchFeature(data=data, tensor_type=return_tensors)
        # Lists of different lengths, only needed by the processor to build the prompt
        encoding["rows"] = rows
        encoding["cols"] = cols
        return encoding

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs: dict | None = None
    ) -> tuple[int, int, int]:
        """
        A utility that returns the number of tiles for a given image size.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `tuple[int, int, int]`: Number of tiles (including the thumbnail) and the number of rows and columns
            they form.
        """
        images_kwargs = images_kwargs or {}
        if not images_kwargs.get("crop_to_patches", self.crop_to_patches):
            return 1, 1, 1
        min_patches = images_kwargs.get("min_patches", self.min_patches)
        max_patches = images_kwargs.get("max_patches", self.max_patches)
        size = images_kwargs.get("size", self.size)
        num_cols, num_rows = get_optimal_tiled_canvas(
            (height, width), (size["height"], size["width"]), min_patches, max_patches
        )
        num_patches = num_rows * num_cols
        if num_patches > 1:
            num_patches += 1
        return num_patches, num_rows, num_cols

    def to_dict(self):
        encoder_dict = super().to_dict()
        encoder_dict.pop("fine_route", None)
        return encoder_dict


@requires(backends=("vision",))
@auto_docstring(
    custom_intro="""
    PIL backend of [`GraniteForDoclingImageProcessor`]: the same 512x512 tiling, thumbnail, tile padding and
    `fine_route` option.
    """
)
class GraniteForDoclingImageProcessorPil(GotOcr2ImageProcessorPil):
    valid_kwargs = GraniteForDoclingImageProcessorKwargs
    size = {"height": 512, "width": 512}
    crop_to_patches = True
    max_patches = 32
    fine_route = False

    def _prepare_images_structure(self, images: ImageInput, expected_ndims: int = 3) -> ImageInput:
        images = self.fetch_images(images)
        return make_nested_list_of_images(images, expected_ndims=expected_ndims)

    def crop_image_to_patches(
        self,
        image: np.ndarray,
        min_patches: int,
        max_patches: int,
        use_thumbnail: bool = True,
        patch_size: SizeDict | None = None,
        resample: "PILImageResampling | int | None" = None,
    ):
        """
        Crop the image to patches and return a list of cropped images.
        The number of patches and their grid arrangement are determined by the original image size,
        the target patch size and the minimum and maximum number of patches.
        The aspect ratio of the patches grid is chosen to be the closest to the original image aspect ratio.
        """
        input_data_format = infer_channel_dimension_format(image)
        image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)

        patch_size_height, patch_size_width = patch_size.height, patch_size.width
        original_height, original_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        num_columns, num_rows = get_optimal_tiled_canvas(
            (original_height, original_width), (patch_size_height, patch_size_width), min_patches, max_patches
        )

        target_width = patch_size_width * num_columns
        target_height = patch_size_height * num_rows
        num_blocks = num_columns * num_rows

        resized_image = self.resize(image, SizeDict(height=target_height, width=target_width), resample=resample)
        processed_images = []
        for i in range(num_blocks):
            column = i % num_columns
            row = i // num_columns
            box = (
                column * patch_size_width,
                row * patch_size_height,
                (column + 1) * patch_size_width,
                (row + 1) * patch_size_height,
            )
            patch_image = resized_image[..., box[1] : box[3], box[0] : box[2]]
            patch_image = to_channel_dimension_format(patch_image, input_data_format, ChannelDimension.FIRST)
            processed_images.append(patch_image)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = self.resize(image, patch_size, resample=resample)
            thumbnail_img = to_channel_dimension_format(thumbnail_img, input_data_format, ChannelDimension.FIRST)
            processed_images.append(thumbnail_img)

        return processed_images

    def _preprocess(
        self,
        images: list[list[np.ndarray]],
        size: SizeDict,
        resample,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        crop_to_patches: bool = True,
        min_patches: int = 1,
        max_patches: int = 32,
        fine_route: bool = False,
        **kwargs,
    ) -> BatchFeature:
        pixel_values, rows, cols = [], [], []
        for sample in images:
            sample_tiles, sample_rows, sample_cols = [], [], []
            for image in sample:
                if crop_to_patches:
                    num_cols, num_rows = get_optimal_tiled_canvas(
                        tuple(image.shape[-2:]), (size.height, size.width), min_patches, max_patches
                    )
                    tiles = np.stack(
                        self.crop_image_to_patches(
                            image, min_patches, max_patches, patch_size=size, resample=resample
                        ),
                        axis=0,
                    )
                else:
                    num_cols = num_rows = 1
                    tiles = self.resize(image, size, resample=resample)[None]
                sample_tiles.append(tiles)
                sample_rows.append(num_rows)
                sample_cols.append(num_cols)
            if sample_tiles:
                sample_tiles = np.concatenate(sample_tiles, axis=0)
                processed_tiles = []
                for tile in sample_tiles:
                    if do_rescale:
                        tile = self.rescale(tile, rescale_factor)
                    if do_normalize:
                        tile = self.normalize(tile, image_mean, image_std)
                    processed_tiles.append(tile)
                sample_tiles = np.stack(processed_tiles, axis=0)
            pixel_values.append(sample_tiles)
            rows.append(sample_rows)
            cols.append(sample_cols)

        max_num_tiles = max(len(tiles) for tiles in pixel_values)
        first_tiles = next(tiles for tiles in pixel_values if len(tiles) > 0)
        padded_pixel_values = np.zeros(
            (len(pixel_values), max_num_tiles, *first_tiles.shape[1:]),
            dtype=first_tiles.dtype,
        )
        tile_fine_mask = np.zeros((len(pixel_values), max_num_tiles), dtype=bool)
        for i, tiles in enumerate(pixel_values):
            if len(tiles) > 0:
                padded_pixel_values[i, : tiles.shape[0]] = tiles
                tile_fine_mask[i, : tiles.shape[0]] = fine_route

        data = {"pixel_values": padded_pixel_values}
        if fine_route:
            data["tile_fine_mask"] = tile_fine_mask
        encoding = BatchFeature(data=data, tensor_type=return_tensors)
        encoding["rows"] = rows
        encoding["cols"] = cols
        return encoding

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs: dict | None = None
    ) -> tuple[int, int, int]:
        """
        A utility that returns the number of tiles for a given image size.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `tuple[int, int, int]`: Number of tiles (including the thumbnail) and the number of rows and columns
            they form.
        """
        images_kwargs = images_kwargs or {}
        if not images_kwargs.get("crop_to_patches", self.crop_to_patches):
            return 1, 1, 1
        min_patches = images_kwargs.get("min_patches", self.min_patches)
        max_patches = images_kwargs.get("max_patches", self.max_patches)
        size = images_kwargs.get("size", self.size)
        num_cols, num_rows = get_optimal_tiled_canvas(
            (height, width), (size["height"], size["width"]), min_patches, max_patches
        )
        num_patches = num_rows * num_cols
        if num_patches > 1:
            num_patches += 1
        return num_patches, num_rows, num_cols

    def to_dict(self):
        encoder_dict = super().to_dict()
        encoder_dict.pop("fine_route", None)
        return encoder_dict


class GraniteForDoclingProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}
    images_kwargs: GraniteForDoclingImageProcessorKwargs


@auto_docstring
class GraniteForDoclingProcessor(Idefics3Processor):
    valid_processor_kwargs = GraniteForDoclingProcessorKwargs

    def __init__(
        self, image_processor, tokenizer=None, image_seq_len: int = 64, chat_template: str | None = None, **kwargs
    ):
        r"""
        image_seq_len (`int`, *optional*, defaults to 64):
            The number of `<image>` tokens per tile on the coarse path. It is computed as
            `image_seq_len = int(((image_size // patch_size) ** 2) / (scale_factor**2))`.
        """
        self.fake_image_token = AddedToken("<fake_token_around_image>", normalized=False, special=True).content
        self.image_token = AddedToken("<image>", normalized=False, special=True).content
        self.global_image_tag = "<global-img>"
        self.image_seq_len = image_seq_len
        self.fake_image_token_id = tokenizer.convert_tokens_to_ids(self.fake_image_token)
        self.global_image_token_id = tokenizer.convert_tokens_to_ids(self.global_image_tag)
        tokenizer.add_special_tokens({"additional_special_tokens": [self.fake_image_token, self.image_token]})
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        ProcessorMixin.__init__(self, image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def replace_image_token(self, image_inputs: dict, image_idx: int, fine_route: bool = False, **kwargs) -> str:
        image_rows = [row for row_list in image_inputs["rows"] for row in row_list][image_idx]
        image_cols = [col for col_list in image_inputs["cols"] for col in col_list][image_idx]
        image_tokens = self.image_token * (self.image_seq_len * 4 if fine_route else self.image_seq_len)
        text_split_images = ""
        for n_h in range(image_rows):
            for n_w in range(image_cols):
                text_split_images += f"{self.fake_image_token}<row_{n_h + 1}_col_{n_w + 1}>{image_tokens}"
            text_split_images += "\n"
        # The thumbnail is only added when the image is split into several tiles
        if image_rows * image_cols > 1:
            text_split_images += (
                f"\n{self.fake_image_token}{self.global_image_tag}{image_tokens}{self.fake_image_token}"
            )
        return text_split_images

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.

        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """
        vision_data = {}
        if image_sizes is not None:
            num_image_tokens = []
            num_image_patches = []
            for height, width in image_sizes:
                num_patches, num_rows, num_cols = self.image_processor.get_number_of_image_patches(
                    height, width, kwargs
                )
                image_prompt = self.replace_image_token({"rows": [[num_rows]], "cols": [[num_cols]]}, 0, **kwargs)
                num_image_tokens.append(len(self.tokenizer(image_prompt, add_special_tokens=False)["input_ids"]))
                num_image_patches.append(num_patches)
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})
        return MultiModalData(**vision_data)


@auto_docstring(
    custom_intro="""
    Base class for GraniteForDocling model outputs, with potential hidden states and attentions.
    """
)
@dataclass
class GraniteForDoclingBaseModelOutputWithPast(Idefics3BaseModelOutputWithPast):
    r"""
    router_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
        Logits of the density router, one per sample. Returned when the model has a router and `pixel_values` are
        given.
    """

    router_logits: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    Base class for GraniteForDocling causal language model (or autoregressive) outputs.
    """
)
@dataclass
class GraniteForDoclingCausalLMOutputWithPast(Idefics3CausalLMOutputWithPast):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    router_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
        Logits of the density router, one per sample. Returned when the model has a router and `pixel_values` are
        given.
    """

    router_logits: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    Base class for GraniteForDocling image features.
    """
)
@dataclass
class GraniteForDoclingImageFeaturesOutput(BaseModelOutputWithPooling):
    r"""
    pooler_output (`torch.FloatTensor`):
        Image features projected to the text hidden size, of shape `(num_tiles, image_seq_len, hidden_size)`, or
        `(num_image_tokens, hidden_size)` when tiles are routed to different connector paths.
    deepstack_features (`list[torch.FloatTensor]`, *optional*):
        One projected intermediate vision encoder hidden state per entry of `config.deepstack_attn_layers`, with
        the same layout as `pooler_output`.
    router_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
        Logits of the density router, one per sample. Returned when the model has a router.
    """

    deepstack_features: list[torch.FloatTensor] | None = None
    router_logits: torch.FloatTensor | None = None


class GraniteForDoclingTextRMSNorm(GraniteRMSNorm):
    pass


class GraniteForDoclingTextMLP(GraniteMoeSharedMLP):
    pass


class GraniteForDoclingTextAttention(GraniteAttention):
    pass


class GraniteForDoclingTextRotaryEmbedding(GraniteRotaryEmbedding):
    pass


class GraniteForDoclingTextDecoderLayer(GraniteDecoderLayer):
    def __init__(self, config: GraniteForDoclingTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.mlp
        self.shared_mlp = GraniteForDoclingTextMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.shared_mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier
        return hidden_states


class GraniteForDoclingTextPreTrainedModel(GranitePreTrainedModel):
    pass


class GraniteForDoclingTextModel(GraniteModel):
    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        visual_pos_masks: torch.BoolTensor | None = None,
        deepstack_visual_embeds: dict[int, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        visual_pos_masks (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Positions of the image tokens that receive the DeepStack features.
        deepstack_visual_embeds (`dict[int, torch.Tensor]`, *optional*):
            DeepStack features of shape `(num_image_tokens, hidden_size)`, keyed by the index of the decoder layer
            after which they are added to the hidden states at `visual_pos_masks`.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds * self.embedding_multiplier

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if deepstack_visual_embeds is not None and layer_idx in deepstack_visual_embeds:
                visual_embeds = deepstack_visual_embeds[layer_idx].to(hidden_states.device, hidden_states.dtype)
                visual_pos_masks = visual_pos_masks.to(hidden_states.device)
                hidden_states[visual_pos_masks, :] = hidden_states[visual_pos_masks, :].clone() + visual_embeds

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


def build_2d_sincos_position_embedding(embed_dim: int, grid_size: int) -> torch.Tensor:
    """
    Returns fixed 2D sine-cosine position embeddings of shape `(grid_size**2, embed_dim)` for a square grid of tokens
    in row-major order.
    """
    omega = torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim // 4)
    omega = 1.0 / 10000**omega
    positions = torch.arange(grid_size, dtype=torch.float32)
    rows, cols = torch.meshgrid(positions, positions, indexing="ij")
    out_cols = cols.reshape(-1, 1) * omega
    out_rows = rows.reshape(-1, 1) * omega
    return torch.cat([out_cols.sin(), out_cols.cos(), out_rows.sin(), out_rows.cos()], dim=1)


def merge_tile_routes(tile_fine_mask: torch.Tensor, coarse: torch.Tensor, fine: torch.Tensor) -> torch.Tensor:
    """
    Interleaves per-tile features projected on the coarse and fine paths back into tile order, flattened to
    `(num_image_tokens, hidden_size)`.
    """
    tiles = [None] * tile_fine_mask.shape[0]
    for tile_idx, features in zip((~tile_fine_mask).nonzero(as_tuple=True)[0].tolist(), coarse):
        tiles[tile_idx] = features
    for tile_idx, features in zip(tile_fine_mask.nonzero(as_tuple=True)[0].tolist(), fine):
        tiles[tile_idx] = features
    return torch.cat(tiles, dim=0)


class GraniteForDoclingDeepStackMerger(nn.Module):
    def __init__(self, config: GraniteForDoclingConfig, scale_factor: int):
        super().__init__()
        merged_dim = config.vision_config.hidden_size * (scale_factor**2)
        self.norm = nn.LayerNorm(merged_dim)
        self.fc1 = nn.Linear(merged_dim, config.text_config.hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(hidden_states))))


class GraniteForDoclingConnector(Idefics3Connector):
    def __init__(self, config: GraniteForDoclingConfig):
        super().__init__(config)
        text_hidden_size = config.text_config.hidden_size
        tokens_per_tile = (config.vision_config.image_size // config.vision_config.patch_size) ** 2
        self.ln_in = nn.LayerNorm(config.vision_config.hidden_size)
        self.ln_mid = nn.LayerNorm(text_hidden_size)
        self.mlp_fc2 = nn.Linear(text_hidden_size, text_hidden_size, bias=False)
        self.ln_out = nn.LayerNorm(text_hidden_size)
        self.pos_embed_2d = nn.Buffer(
            build_2d_sincos_position_embedding(text_hidden_size, int(tokens_per_tile**0.5) // self.scale_factor),
            persistent=False,
        )
        self.deepstack_mergers = nn.ModuleList(
            [GraniteForDoclingDeepStackMerger(config, self.scale_factor) for _ in config.deepstack_visual_indexes]
        )

        # Fine path: pixel shuffle by half the factor, four times as many image tokens per tile
        self.fine_scale_factor = self.scale_factor // 2
        self.proj_fine = None
        self.deepstack_mergers_fine = None
        if config.use_fine_route:
            self.proj_fine = nn.Linear(
                config.vision_config.hidden_size * (self.fine_scale_factor**2), text_hidden_size, bias=False
            )
            self.pos_embed_2d_fine = nn.Buffer(
                build_2d_sincos_position_embedding(
                    text_hidden_size, int(tokens_per_tile**0.5) // self.fine_scale_factor
                ),
                persistent=False,
            )
            self.deepstack_mergers_fine = nn.ModuleList(
                [
                    GraniteForDoclingDeepStackMerger(config, self.fine_scale_factor)
                    for _ in config.deepstack_visual_indexes
                ]
            )

    def _project(self, image_hidden_states: torch.Tensor, fine: bool) -> torch.Tensor:
        hidden_states = self.ln_in(image_hidden_states)
        if fine:
            hidden_states = self.proj_fine(self.pixel_shuffle(hidden_states, self.fine_scale_factor))
            hidden_states = hidden_states + self.pos_embed_2d_fine.to(hidden_states.dtype)
        else:
            hidden_states = self.modality_projection(self.pixel_shuffle(hidden_states, self.scale_factor))
            hidden_states = hidden_states + self.pos_embed_2d.to(hidden_states.dtype)
        hidden_states = nn.functional.gelu(self.ln_mid(hidden_states))
        return self.ln_out(self.mlp_fc2(hidden_states))

    def _route(self, image_hidden_states: torch.Tensor, tile_fine_mask: torch.Tensor | None, coarse_fn, fine_fn):
        if tile_fine_mask is None or not tile_fine_mask.any():
            return coarse_fn(image_hidden_states)
        if self.proj_fine is None:
            raise ValueError("This model has no fine connector path (`config.use_fine_route=False`).")
        if tile_fine_mask.all():
            return fine_fn(image_hidden_states)
        return merge_tile_routes(
            tile_fine_mask,
            coarse_fn(image_hidden_states[~tile_fine_mask]),
            fine_fn(image_hidden_states[tile_fine_mask]),
        )

    def forward(self, image_hidden_states: torch.Tensor, tile_fine_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self._route(
            image_hidden_states,
            tile_fine_mask,
            lambda hidden_states: self._project(hidden_states, fine=False),
            lambda hidden_states: self._project(hidden_states, fine=True),
        )

    def deepstack(
        self, slot: int, image_hidden_states: torch.Tensor, tile_fine_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self._route(
            image_hidden_states,
            tile_fine_mask,
            lambda hidden_states: self.deepstack_mergers[slot](
                self.pixel_shuffle(hidden_states.contiguous(), self.scale_factor)
            ),
            lambda hidden_states: self.deepstack_mergers_fine[slot](
                self.pixel_shuffle(hidden_states.contiguous(), self.fine_scale_factor)
            ),
        )


class GraniteForDoclingDensityRouter(nn.Module):
    """
    Predicts from the vision encoder features of a sample's tiles whether the sample needs the fine connector path.
    """

    def __init__(self, config: GraniteForDoclingConfig):
        super().__init__()
        vision_hidden_size = config.vision_config.hidden_size
        self.att = nn.Linear(vision_hidden_size, 1)
        self.norm = nn.LayerNorm(vision_hidden_size)
        self.fc1 = nn.Linear(vision_hidden_size, config.density_router_hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.density_router_hidden_size, 1)

    def forward(
        self, image_hidden_states: torch.Tensor, tile_sample_index: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        # Attention-pool the patches of every tile, then average the tiles of every sample
        weights = torch.softmax(self.att(image_hidden_states), dim=1)
        tile_features = (image_hidden_states * weights).sum(dim=1)
        sample_features = tile_features.new_zeros(num_samples, tile_features.shape[-1])
        sample_features = sample_features.index_add(0, tile_sample_index, tile_features)
        num_tiles = tile_features.new_zeros(num_samples).index_add(
            0, tile_sample_index, tile_features.new_ones(tile_features.shape[0])
        )
        sample_features = sample_features / num_tiles.clamp(min=1).unsqueeze(-1)
        return self.fc2(self.act(self.fc1(self.norm(sample_features)))).squeeze(-1)


class GraniteForDoclingMTPBlock(GradientCheckpointingLayer):
    def __init__(self, config: GraniteForDoclingConfig):
        super().__init__()
        hidden_size = config.text_config.hidden_size
        self.input_norm = GraniteForDoclingTextRMSNorm(hidden_size, eps=config.text_config.rms_norm_eps)
        self.embed_norm = GraniteForDoclingTextRMSNorm(hidden_size, eps=config.text_config.rms_norm_eps)
        self.proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=config.mtp_num_attention_heads or config.text_config.num_attention_heads,
            dim_feedforward=config.mtp_intermediate_size or config.text_config.intermediate_size,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, hidden_states: torch.Tensor, token_embeddings: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(
            torch.cat([self.input_norm(hidden_states), self.embed_norm(token_embeddings)], dim=-1)
        )
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            hidden_states.shape[1], device=hidden_states.device, dtype=hidden_states.dtype
        )
        return self.transformer_layer(hidden_states, src_mask=causal_mask, is_causal=True)


class GraniteForDoclingMTP(nn.Module):
    """
    Multi-token prediction heads. Head `i` reads the hidden states of the previous head (the decoder for head 0) and
    the embeddings of the tokens `i + 1` positions ahead, and predicts the tokens `i + 2` positions ahead.
    """

    def __init__(self, config: GraniteForDoclingConfig):
        super().__init__()
        self.blocks = nn.ModuleList([GraniteForDoclingMTPBlock(config) for _ in range(config.num_mtp_layers)])

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.LongTensor, embed_tokens: nn.Module
    ) -> list[torch.Tensor]:
        """
        Returns the hidden states of every head. Position `t` of head `i` (of shape `(batch_size, sequence_length -
        i - 2, hidden_size)`) predicts token `t + i + 2`.
        """
        head_hidden_states = []
        for offset, block in enumerate(self.blocks, start=1):
            span = hidden_states.shape[1] - offset - 1
            if span <= 0:
                break
            hidden_states = block(hidden_states[:, :span], embed_tokens(input_ids[:, offset : offset + span]))
            head_hidden_states.append(hidden_states)
        return head_hidden_states


class GraniteForDoclingVisionEmbeddings(Idefics3VisionEmbeddings):
    """
    Patch embeddings with learned position embeddings for square tiles of `config.image_size` pixels.
    """


class GraniteForDoclingVisionAttention(Idefics3VisionAttention):
    def __init__(self, config):
        super().__init__(config)
        self.num_key_value_groups = 1  # needed for eager attention


class GraniteForDoclingPreTrainedModel(Idefics3PreTrainedModel):
    _no_split_modules = ["GraniteForDoclingVisionAttention", "GraniteForDoclingTextDecoderLayer"]

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, GraniteForDoclingConnector):
            for name in ("pos_embed_2d", "pos_embed_2d_fine"):
                buffer = getattr(module, name, None)
                if buffer is not None:
                    init.copy_(
                        buffer, build_2d_sincos_position_embedding(buffer.shape[-1], int(buffer.shape[0] ** 0.5))
                    )
        elif isinstance(module, GraniteForDoclingMTPBlock):
            # Start each head as the sum of the decoder hidden state and the token embedding
            eye = torch.eye(module.proj.out_features)
            init.copy_(module.proj.weight, torch.cat([eye, eye], dim=1))
        elif isinstance(module, nn.MultiheadAttention):
            init.normal_(module.in_proj_weight, mean=0.0, std=self.config.text_config.initializer_range)
            init.zeros_(module.in_proj_bias)


@auto_docstring(
    custom_intro="""
    GraniteForDocling model consisting of a vision encoder, a connector and a text decoder, outputting raw hidden
    states without any specific head on top.
    """
)
class GraniteForDoclingModel(Idefics3Model):
    def __init__(self, config: GraniteForDoclingConfig):
        super().__init__(config)
        del self.text_model
        self.text_model = GraniteForDoclingTextModel._from_config(config.text_config)
        self.density_router = None
        if config.density_router_hidden_size is not None:
            self.density_router = GraniteForDoclingDensityRouter(config)

    def inputs_merger(self, **super_kwargs):
        raise AttributeError("Not needed for GraniteForDocling")

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full((), self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        n_image_features = image_features.numel() // image_features.shape[-1]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask.to(inputs_embeds.device)

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        tile_fine_mask: torch.BoolTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | GraniteForDoclingImageFeaturesOutput:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_tiles, num_channels, image_size, image_size)`):
            The tensors corresponding to the input image tiles. All-zero tiles are padding and are discarded.
        tile_fine_mask (`torch.BoolTensor` of shape `(batch_size, num_tiles)`, *optional*):
            Tiles to route through the fine connector path, which yields four times as many image tokens per tile.
        """
        batch_size, num_tiles, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.to(dtype=self.dtype)  # fp16 compatibility
        pixel_values = pixel_values.view(batch_size * num_tiles, *pixel_values.shape[2:])

        # Remove padding tiles - padding tiles are full 0.
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
        pixel_values = pixel_values[real_images_inds].contiguous()
        if tile_fine_mask is not None:
            tile_fine_mask = tile_fine_mask.reshape(-1)[real_images_inds].to(pixel_values.device)

        # The intermediate hidden states feed the DeepStack mergers
        kwargs["output_hidden_states"] = True
        image_outputs = self.vision_model(pixel_values=pixel_values, return_dict=True, **kwargs)
        image_hidden_states = image_outputs.last_hidden_state
        image_features = self.connector(image_hidden_states, tile_fine_mask)
        # `hidden_states[0]` is the patch embedding output, so the output of vision layer `i` is `hidden_states[i + 1]`
        deepstack_features = [
            self.connector.deepstack(slot, image_outputs.hidden_states[vision_layer_idx + 1], tile_fine_mask)
            for slot, vision_layer_idx in enumerate(self.config.deepstack_visual_indexes)
        ]
        router_logits = None
        if self.density_router is not None:
            tile_sample_index = torch.arange(batch_size, device=pixel_values.device).repeat_interleave(num_tiles)
            router_logits = self.density_router(image_hidden_states, tile_sample_index[real_images_inds], batch_size)
        return GraniteForDoclingImageFeaturesOutput(
            last_hidden_state=image_hidden_states,
            pooler_output=image_features,
            hidden_states=image_outputs.hidden_states,
            attentions=image_outputs.attentions,
            deepstack_features=deepstack_features,
            router_logits=router_logits,
        )

    def predict_fine_route(self, pixel_values: torch.FloatTensor) -> torch.BoolTensor:
        r"""
        Predicts with the density router which samples need the fine connector path. Route them by calling the
        processor again with `fine_route=True`.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_tiles, num_channels, image_size, image_size)`):
                The tensors corresponding to the input image tiles.

        Returns:
            `torch.BoolTensor` of shape `(batch_size,)`: `True` for the samples that need the fine connector path.
        """
        if self.density_router is None:
            raise ValueError("This model has no density router, set `density_router_hidden_size` in the config.")
        router_logits = self.get_image_features(pixel_values, return_dict=True).router_logits
        return torch.sigmoid(router_logits) >= self.config.density_router_threshold

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_tiles, 3, image_size, image_size) where max_num_tiles
        is the maximum number of tiles among the batch_size samples in the batch. Padding tiles are all-zero and are
        discarded before the vision encoder.
        """
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        tile_fine_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | GraniteForDoclingBaseModelOutputWithPast:
        r"""
        tile_fine_mask (`torch.BoolTensor` of shape `(batch_size, num_tiles)`, *optional*):
            Tiles to route through the fine connector path, which yields four times as many image tokens per tile.
        image_hidden_states (`torch.FloatTensor` of shape `(num_tiles, image_seq_len, hidden_size)`, or `(num_tiles, 4 * image_seq_len, hidden_size)` on the fine route):
            The hidden states of the image encoder after modality projection. Pass this instead of `pixel_values`
            to reuse a previous call's vision-tower output and skip recomputing it (e.g. across turns in a
            conversation). Do not pass both at once: clear `pixel_values` from the inputs first, or a `ValueError`
            is raised.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(self.device)

        deepstack_visual_embeds = None
        visual_pos_masks = None
        router_logits = None
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError(
                "You cannot specify both pixel_values and image_hidden_states at the same time. To reuse a "
                "previous call's image_hidden_states, remove pixel_values from the inputs instead of passing both."
            )
        elif pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, tile_fine_mask=tile_fine_mask, return_dict=True)
            image_hidden_states = image_outputs.pooler_output
            router_logits = image_outputs.router_logits
            deepstack_visual_embeds = {
                layer_idx: features.reshape(-1, features.shape[-1])
                for layer_idx, features in zip(self.config.deepstack_attn_layers, image_outputs.deepstack_features)
            }

        if image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_hidden_states)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask.unsqueeze(-1), image_hidden_states)
            visual_pos_masks = special_image_mask

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return GraniteForDoclingBaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
            router_logits=router_logits,
        )


@auto_docstring(
    custom_intro="""
    The GraniteForDocling Model with a language modeling head. It is made up of a vision encoder, a connector and a
    text decoder, with a language modeling head on top.
    """
)
class GraniteForDoclingForConditionalGeneration(Idefics3ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.mtp = None
        if config.num_mtp_layers > 0:
            self.mtp = GraniteForDoclingMTP(config)

    def predict_fine_route(self, pixel_values: torch.FloatTensor) -> torch.BoolTensor:
        r"""
        Predicts with the density router which samples need the fine connector path. Route them by calling the
        processor again with `fine_route=True`.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_tiles, num_channels, image_size, image_size)`):
                The tensors corresponding to the input image tiles.

        Returns:
            `torch.BoolTensor` of shape `(batch_size,)`: `True` for the samples that need the fine connector path.
        """
        return self.model.predict_fine_route(pixel_values)

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        tile_fine_mask: torch.BoolTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | GraniteForDoclingImageFeaturesOutput:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_tiles, num_channels, image_size, image_size)`):
            The tensors corresponding to the input image tiles. All-zero tiles are padding and are discarded.
        tile_fine_mask (`torch.BoolTensor` of shape `(batch_size, num_tiles)`, *optional*):
            Tiles to route through the fine connector path, which yields four times as many image tokens per tile.
        """
        return self.model.get_image_features(
            pixel_values=pixel_values,
            tile_fine_mask=tile_fine_mask,
            **kwargs,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        tile_fine_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        router_labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | GraniteForDoclingCausalLMOutputWithPast:
        r"""
        tile_fine_mask (`torch.BoolTensor` of shape `(batch_size, num_tiles)`, *optional*):
            Tiles to route through the fine connector path, which yields four times as many image tokens per tile.
        image_hidden_states (`torch.FloatTensor` of shape `(num_tiles, image_seq_len, hidden_size)`, or `(num_tiles, 4 * image_seq_len, hidden_size)` on the fine route):
            The hidden states of the image encoder after modality projection.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or `model.image_token_id` (where `model` is your instance of `GraniteForDoclingForConditionalGeneration`).
            Tokens with indices set to `model.image_token_id` are ignored (masked), the loss is only
            computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        router_labels (`torch.Tensor` of shape `(batch_size,)`, *optional*):
            Targets of the density router, `1` (or `True`) for the samples that need the fine connector path and `0`
            otherwise. Given as float, bool or int; the router loss is a binary cross-entropy on the logits.

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, AutoModelForImageTextToText
        >>> from transformers.image_utils import load_image

        >>> image = load_image("https://huggingface.co/docling-project/granite-for-docling-500m/resolve/main/docling_technical_report_p1.png")

        >>> processor = AutoProcessor.from_pretrained("docling-project/granite-for-docling-500m")
        >>> model = AutoModelForImageTextToText.from_pretrained("docling-project/granite-for-docling-500m", dtype=torch.bfloat16, device_map="auto")

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image"},
        ...             {"type": "text", "text": "<doclang>"},
        ...         ],
        ...     }
        ... ]
        >>> prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        >>> inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

        >>> generated_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> print(processor.decode(generated_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            tile_fine_mask=tile_fine_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = logits / self.config.text_config.logits_scaling

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )
            if self.mtp is not None and input_ids is not None:
                # Head `offset` at position `t` predicts token `t + offset + 1`. `shift_labels`, when a sequence
                # parallel trainer passes it, already holds token `t + 1` at position `t` and replaces `labels`.
                shift_labels = kwargs.get("shift_labels")
                mtp_targets = labels[:, 1:] if shift_labels is None else shift_labels
                mtp_losses = []
                for offset, head_hidden_states in enumerate(
                    self.mtp(hidden_states, input_ids, self.get_input_embeddings()), start=1
                ):
                    head_logits = self.lm_head(head_hidden_states) / self.config.text_config.logits_scaling
                    head_labels = mtp_targets[:, offset : offset + head_hidden_states.shape[1]]
                    mtp_losses.append(
                        nn.functional.cross_entropy(
                            head_logits.reshape(-1, head_logits.shape[-1]).float(), head_labels.reshape(-1)
                        )
                    )
                if mtp_losses:
                    loss = loss + self.config.mtp_loss_weight * torch.stack(mtp_losses).mean()
        if router_labels is not None and outputs.router_logits is not None:
            router_loss = self.config.density_router_loss_weight * nn.functional.binary_cross_entropy_with_logits(
                outputs.router_logits.float(), router_labels.to(outputs.router_logits.device).float()
            )
            loss = router_loss if loss is None else loss + router_loss

        return GraniteForDoclingCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            router_logits=outputs.router_logits,
        )


__all__ = [
    "GraniteForDoclingConfig",
    "GraniteForDoclingForConditionalGeneration",
    "GraniteForDoclingImageProcessor",
    "GraniteForDoclingImageProcessorPil",
    "GraniteForDoclingModel",
    "GraniteForDoclingPreTrainedModel",
    "GraniteForDoclingProcessor",
    "GraniteForDoclingTextConfig",
    "GraniteForDoclingTextModel",
    "GraniteForDoclingTextPreTrainedModel",
    "GraniteForDoclingVisionConfig",
]
