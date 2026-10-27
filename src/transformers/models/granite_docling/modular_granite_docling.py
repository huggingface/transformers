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
"""PyTorch GraniteDocling model."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    SizeDict,
    infer_channel_dimension_format,
    is_valid_image,
    make_nested_list_of_images,
)
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.import_utils import torch_compilable_check
from ...utils.output_capturing import capture_outputs
from ..got_ocr2.image_processing_got_ocr2 import (
    GotOcr2ImageProcessor,
    GotOcr2ImageProcessorKwargs,
    get_optimal_tiled_canvas,
)
from ..got_ocr2.image_processing_pil_got_ocr2 import GotOcr2ImageProcessorPil
from ..granite.configuration_granite import GraniteConfig
from ..granite.modeling_granite import GraniteModel, GranitePreTrainedModel
from ..idefics3.configuration_idefics3 import Idefics3VisionConfig
from ..idefics3.modeling_idefics3 import (
    Idefics3BaseModelOutputWithPast,
    Idefics3CausalLMOutputWithPast,
    Idefics3Connector,
    Idefics3ForConditionalGeneration,
    Idefics3Model,
    Idefics3PreTrainedModel,
    Idefics3VisionAttention,
    Idefics3VisionTransformer,
)
from ..idefics3.processing_idefics3 import Idefics3Processor
from ..internvl.modeling_internvl import InternVLMultiModalProjector
from ..siglip.modeling_siglip import SiglipVisionEmbeddings
from ..vit_mae.modeling_vit_mae import build_2d_sinusoidal_position_embedding


logger = logging.get_logger(__name__)


@auto_docstring(
    custom_intro="""
    Configuration of the SigLIP-style vision encoder that embeds every 512x512 tile of a page, and of the connector
    that turns its outputs into image tokens.
    """,
    checkpoint="docling-project/granite-for-docling-500m",
)
@strict
class GraniteDoclingVisionConfig(Idefics3VisionConfig):
    r"""
    out_hidden_size (`int`, *optional*, defaults to 1024):
        Hidden size of the text decoder that the connector projects to.
    scale_factor (`int`, *optional*, defaults to 4):
        Pixel shuffle factor of the connector. Each tile of `(image_size // patch_size) ** 2` vision tokens is
        reduced by `scale_factor ** 2`.
    deepstack_visual_indexes (`list[int]`, *optional*, defaults to `[3, 7, 10]`):
        Indices of the vision encoder layers whose output is projected by the DeepStack mergers. Index 0 is the
        output of the first layer.
    use_fine_route (`bool`, *optional*, defaults to `True`):
        Whether to build the fine connector path, which shuffles pixels by half of `scale_factor` and yields four
        times as many image tokens per tile.

    Example:

    ```python
    >>> from transformers import GraniteDoclingVisionConfig, GraniteDoclingVisionModel

    >>> # Initializing a GraniteDoclingVisionConfig with docling-project/granite-for-docling-500m style configuration
    >>> configuration = GraniteDoclingVisionConfig()

    >>> # Initializing a GraniteDoclingVisionModel (with random weights) from the docling-project/granite-for-docling-500m style configuration
    >>> model = GraniteDoclingVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "granite_docling_vision"

    hidden_size: int = 768
    num_attention_heads: int = 12
    image_size: int | list[int] | tuple[int, int] = 512
    patch_size: int | list[int] | tuple[int, int] = 16
    out_hidden_size: int = 1024
    scale_factor: int = 4
    deepstack_visual_indexes: list[int] | None = None
    use_fine_route: bool = True

    def __post_init__(self, **kwargs):
        if self.deepstack_visual_indexes is None:
            self.deepstack_visual_indexes = [3, 7, 10]
        PreTrainedConfig.__post_init__(**kwargs)


@auto_docstring(checkpoint="docling-project/granite-for-docling-500m")
@strict
class GraniteDoclingTextConfig(GraniteConfig):
    model_type = "granite_docling_text"
    base_config_key = "text_config"

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


@auto_docstring(checkpoint="docling-project/granite-for-docling-500m")
@strict
class GraniteDoclingConfig(PreTrainedConfig):
    r"""
    density_router_hidden_size (`int`, *optional*):
        Hidden size of the density router, which predicts from the vision encoder features whether a page needs the
        fine connector path. No router is built when `None`.
    density_router_threshold (`float`, *optional*, defaults to 0.4):
        Probability above which the density router selects the fine connector path.
    density_router_loss_weight (`float`, *optional*, defaults to 0.1):
        Weight of the density router loss when `router_labels` are given.
    """

    model_type = "granite_docling"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=GraniteDoclingTextConfig),
        "vision_config": SubConfigSpec(config_class=GraniteDoclingVisionConfig),
    }

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 100337
    tie_word_embeddings: bool = True
    use_cache: bool = True
    pad_token_id: int | None = 100256
    density_router_hidden_size: int | None = None
    density_router_threshold: float = 0.4
    density_router_loss_weight: float = 0.1

    def validate_architecture(self):
        if self.vision_config.out_hidden_size != self.text_config.hidden_size:
            raise ValueError(
                f"vision_config.out_hidden_size ({self.vision_config.out_hidden_size}) must equal "
                f"text_config.hidden_size ({self.text_config.hidden_size})."
            )
        if self.density_router_hidden_size is not None and not self.vision_config.use_fine_route:
            raise ValueError(
                "The density router selects the fine connector path, it needs `vision_config.use_fine_route=True`."
            )


class GraniteDoclingImageProcessorKwargs(GotOcr2ImageProcessorKwargs):
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
        Whether the pages take the fine connector path, which yields four times as many image tokens per tile.
        Returned as `tile_fine_mask`, one value per sample.
    """

    fine_route: bool


@lru_cache(maxsize=10)
def get_all_supported_aspect_ratios(min_image_tiles: int, max_image_tiles: int) -> list[tuple[int, int]]:
    """
    Tile grids allowed between `min_image_tiles` and `max_image_tiles`. Each side stops at 16: the tokenizer only
    has `<row_i_col_j>` markers up to 16x16.
    """
    max_side = min(max_image_tiles, 16)
    ratios = [
        (width, height)
        for width in range(1, max_side + 1)
        for height in range(1, max_side + 1)
        if min_image_tiles <= width * height <= max_image_tiles
    ]
    return sorted(ratios, key=lambda grid: grid[0] * grid[1])


def get_page_tile_grid(
    height: int, width: int, size: SizeDict | dict, crop_to_patches: bool, min_patches: int, max_patches: int
) -> tuple[int, int]:
    """The `(num_rows, num_columns)` tile grid of a page. A page that is not split is a 1x1 grid."""
    if not crop_to_patches or max_patches <= 1:
        return 1, 1
    num_columns, num_rows = get_optimal_tiled_canvas(
        (height, width), (size["height"], size["width"]), min_patches, max_patches
    )
    return num_rows, num_columns


@auto_docstring(
    custom_intro="""
    Splits a page into 512x512 tiles on the grid that best matches its aspect ratio (`crop_to_patches`, at most
    `max_patches` tiles and 16 per side), adds a thumbnail of the whole page when there is more than one tile, and
    pads the samples of a batch to the same number of tiles. `pixel_attention_mask` marks the real tiles.
    `fine_route=True` marks the samples for the fine connector path (`tile_fine_mask`, one value per sample).
    """
)
class GraniteDoclingImageProcessor(GotOcr2ImageProcessor):
    valid_kwargs = GraniteDoclingImageProcessorKwargs
    size = {"height": 512, "width": 512}
    crop_to_patches = True
    max_patches = 32
    fine_route = False
    model_input_names = ["pixel_values", "pixel_attention_mask"]

    def _prepare_images_structure(self, images: ImageInput, expected_ndims: int = 3) -> ImageInput:
        images = self.fetch_images(images)
        return make_nested_list_of_images(images, expected_ndims=expected_ndims)

    def crop_image_to_patches(
        self,
        images: "torch.Tensor",
        num_rows: int,
        num_columns: int,
        patch_size: SizeDict,
        use_thumbnail: bool = True,
        resample=None,
    ):
        """GotOcr2's crop, on the `(num_rows, num_columns)` grid from `get_tile_grid`."""
        patch_size_height, patch_size_width = patch_size["height"], patch_size["width"]
        target_size = SizeDict(height=patch_size_height * num_rows, width=patch_size_width * num_columns)
        resized_image = self.resize(images, target_size, resample=resample)
        processed_images = []
        for i in range(num_columns * num_rows):
            column = i % num_columns
            row = i // num_columns
            box = (
                column * patch_size_width,
                row * patch_size_height,
                (column + 1) * patch_size_width,
                (row + 1) * patch_size_height,
            )
            processed_images.append(resized_image[..., box[1] : box[3], box[0] : box[2]])
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(self.resize(images, patch_size, resample=resample))
        return torch.stack(processed_images, dim=0).transpose(0, 1).contiguous()

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
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        crop_to_patches: bool = True,
        min_patches: int = 1,
        max_patches: int = 32,
        fine_route: bool = False,
        **kwargs,
    ) -> BatchFeature:
        grid_kwargs = {
            "crop_to_patches": crop_to_patches,
            "min_patches": min_patches,
            "max_patches": max_patches,
            "size": size,
        }
        # Pages of the same size get the same grid, so they are tiled together
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping, is_nested=True
        )
        tiles_grouped, grids_grouped = {}, {}
        for shape, stacked_images in grouped_images.items():
            num_rows, num_cols = self.get_tile_grid(*stacked_images.shape[-2:], grid_kwargs)
            stacked_tiles = self.crop_image_to_patches(stacked_images, num_rows, num_cols, size, resample=resample)
            stacked_tiles = self.rescale_and_normalize(
                stacked_tiles, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            tiles_grouped[shape] = stacked_tiles
            grids_grouped[shape] = [(num_rows, num_cols)] * stacked_tiles.shape[0]
        tiles = reorder_images(tiles_grouped, grouped_images_index, is_nested=True)
        grids = reorder_images(grids_grouped, grouped_images_index, is_nested=True)

        # One row of tiles per sample. A text-only sample in a mixed batch (`images=[[], [image]]`) has none.
        reference = next(iter(tiles_grouped.values()), None)
        if reference is not None:
            empty = reference.new_zeros(0, *reference.shape[2:])
        else:
            empty = torch.zeros(0, 3, size.height, size.width)
        sample_tiles = [torch.cat(tiles_list) if tiles_list else empty for tiles_list in tiles]
        data = {
            "pixel_values": pad_sequence(sample_tiles, batch_first=True),
            "pixel_attention_mask": pad_sequence(
                [torch.ones(len(tiles), dtype=torch.bool) for tiles in sample_tiles], batch_first=True
            ),
        }
        if fine_route:
            data["tile_fine_mask"] = torch.ones(len(sample_tiles), dtype=torch.bool)
        # The tile grids are lists of different lengths, only the processor reads them to build the prompt
        data["rows"] = [[grid[0] for grid in sample] for sample in grids]
        data["cols"] = [[grid[1] for grid in sample] for sample in grids]
        return BatchFeature(data=data, tensor_type=return_tensors, skip_tensor_conversion=["rows", "cols"])

    def get_tile_grid(self, height: int, width: int, images_kwargs: dict | None = None) -> tuple[int, int]:
        """The `(num_rows, num_columns)` tile grid a page of this size is split into."""
        images_kwargs = images_kwargs or {}
        return get_page_tile_grid(
            height,
            width,
            images_kwargs.get("size", self.size),
            images_kwargs.get("crop_to_patches", self.crop_to_patches),
            images_kwargs.get("min_patches", self.min_patches),
            images_kwargs.get("max_patches", self.max_patches),
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        """The number of tiles a page of this size yields, including the thumbnail when it is split."""
        num_rows, num_cols = self.get_tile_grid(height, width, images_kwargs)
        return num_rows * num_cols + 1 if num_rows * num_cols > 1 else 1


@auto_docstring(
    custom_intro="""
    PIL backend of [`GraniteDoclingImageProcessor`]: the same 512x512 tiling, thumbnail and tile padding.
    """
)
class GraniteDoclingImageProcessorPil(GotOcr2ImageProcessorPil):
    valid_kwargs = GraniteDoclingImageProcessorKwargs
    size = {"height": 512, "width": 512}
    crop_to_patches = True
    max_patches = 32
    fine_route = False
    model_input_names = ["pixel_values", "pixel_attention_mask"]

    def _prepare_images_structure(self, images: ImageInput, expected_ndims: int = 3) -> ImageInput:
        images = self.fetch_images(images)
        return make_nested_list_of_images(images, expected_ndims=expected_ndims)

    def crop_image_to_patches(
        self,
        image: np.ndarray,
        num_rows: int,
        num_columns: int,
        patch_size: SizeDict,
        use_thumbnail: bool = True,
        resample=None,
    ):
        """GotOcr2's PIL crop, on the `(num_rows, num_columns)` grid from `get_tile_grid`."""
        input_data_format = infer_channel_dimension_format(image)
        image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)
        patch_size_height, patch_size_width = patch_size["height"], patch_size["width"]
        target_size = SizeDict(height=patch_size_height * num_rows, width=patch_size_width * num_columns)
        resized_image = self.resize(image, target_size, resample=resample)
        processed_images = []
        for i in range(num_columns * num_rows):
            column = i % num_columns
            row = i // num_columns
            box = (
                column * patch_size_width,
                row * patch_size_height,
                (column + 1) * patch_size_width,
                (row + 1) * patch_size_height,
            )
            patch_image = resized_image[..., box[1] : box[3], box[0] : box[2]]
            processed_images.append(
                to_channel_dimension_format(patch_image, input_data_format, ChannelDimension.FIRST)
            )
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = self.resize(image, patch_size, resample=resample)
            processed_images.append(
                to_channel_dimension_format(thumbnail_img, input_data_format, ChannelDimension.FIRST)
            )
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
        grid_kwargs = {
            "crop_to_patches": crop_to_patches,
            "min_patches": min_patches,
            "max_patches": max_patches,
            "size": size,
        }
        # Pages are tiled one by one (grouping by shape is a torchvision-backend utility)
        tiles, rows, cols = [], [], []
        for sample in images:
            sample_tiles, sample_rows, sample_cols = [], [], []
            for image in sample:
                num_rows, num_cols = self.get_tile_grid(*image.shape[-2:], grid_kwargs)
                for tile in self.crop_image_to_patches(image, num_rows, num_cols, size, resample=resample):
                    if do_rescale:
                        tile = self.rescale(tile, rescale_factor)
                    if do_normalize:
                        tile = self.normalize(tile, image_mean, image_std)
                    sample_tiles.append(tile)
                sample_rows.append(num_rows)
                sample_cols.append(num_cols)
            tiles.append(sample_tiles)
            rows.append(sample_rows)
            cols.append(sample_cols)

        pixel_values = self._pad_for_batching(tiles, size)
        pixel_attention_mask = np.zeros(pixel_values.shape[:2], dtype=bool)
        for index, sample_tiles in enumerate(tiles):
            pixel_attention_mask[index, : len(sample_tiles)] = True
        data = {"pixel_values": pixel_values, "pixel_attention_mask": pixel_attention_mask}
        if fine_route:
            data["tile_fine_mask"] = np.ones(len(tiles), dtype=bool)
        # The tile grids are lists of different lengths, only the processor reads them to build the prompt
        data["rows"] = rows
        data["cols"] = cols
        return BatchFeature(data=data, tensor_type=return_tensors, skip_tensor_conversion=["rows", "cols"])

    def _pad_for_batching(self, images: list[list[np.ndarray]], size: SizeDict) -> np.ndarray:
        """Pads every sample with all-zero tiles up to the largest tile count in the batch."""
        max_num_tiles = max(len(sample_tiles) for sample_tiles in images)
        pixel_values = np.zeros((len(images), max_num_tiles, 3, size.height, size.width), dtype=np.float32)
        for index, sample_tiles in enumerate(images):
            if sample_tiles:
                pixel_values[index, : len(sample_tiles)] = np.stack(sample_tiles)
        return pixel_values

    def get_tile_grid(self, height: int, width: int, images_kwargs: dict | None = None) -> tuple[int, int]:
        """The `(num_rows, num_columns)` tile grid a page of this size is split into."""
        images_kwargs = images_kwargs or {}
        return get_page_tile_grid(
            height,
            width,
            images_kwargs.get("size", self.size),
            images_kwargs.get("crop_to_patches", self.crop_to_patches),
            images_kwargs.get("min_patches", self.min_patches),
            images_kwargs.get("max_patches", self.max_patches),
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        """The number of tiles a page of this size yields, including the thumbnail when it is split."""
        num_rows, num_cols = self.get_tile_grid(height, width, images_kwargs)
        return num_rows * num_cols + 1 if num_rows * num_cols > 1 else 1


class GraniteDoclingProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


@auto_docstring
class GraniteDoclingProcessor(Idefics3Processor):
    valid_processor_kwargs = GraniteDoclingProcessorKwargs
    # rows/cols build the prompt. They are ragged across a batch, so they stay Python lists and are dropped
    # before the processor returns.
    skip_tensor_conversion = ["video_metadata", "text_replacement_offsets", "rows", "cols"]

    def __init__(
        self, image_processor, tokenizer=None, image_seq_len: int = 64, chat_template: str | None = None, **kwargs
    ):
        r"""
        image_seq_len (`int`, *optional*, defaults to 64):
            The number of `<image>` tokens per tile on the coarse path. It is computed as
            `image_seq_len = int(((image_size // patch_size) ** 2) / (scale_factor**2))`.
        """
        self.fake_image_token = "<fake_token_around_image>"
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else "<image>"
        self.global_image_tag = "<global-img>"
        self.image_seq_len = image_seq_len
        self.fake_image_token_id = tokenizer.convert_tokens_to_ids(self.fake_image_token)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        ProcessorMixin.__init__(self, image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def __call__(self, images=None, text=None, **kwargs):
        outputs = super().__call__(images=images, text=text, **kwargs)
        # The tile grids only build the prompt
        outputs.pop("rows", None)
        outputs.pop("cols", None)
        return outputs

    def prepare_inputs_layout(self, images=None, text=None, videos=None, audio=None, **kwargs):
        """Idefics3's layout: one list of images per prompt, matched to the `<image>` placeholders."""
        if text is not None:
            if isinstance(text, str):
                text = [text]
            text = list(text).copy()

        if images is not None:
            images = self.image_processor.fetch_images(images)
            if is_valid_image(images):
                images = [[images]]
            elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
                if text is not None:
                    # Reorganize the images to match the prompts
                    n_images_in_text = [sample.count(self.image_token) for sample in text]
                    cumsum_images_in_text = [0] + list(accumulate(n_images_in_text))
                    split_images = [
                        images[cumsum_images_in_text[i] : cumsum_images_in_text[i + 1]]
                        for i in range(len(n_images_in_text))
                    ]
                    # Append the rest if any, we will error out when validating if they don't match with text
                    if len(images) > cumsum_images_in_text[-1]:
                        images = split_images + [images[cumsum_images_in_text[-1] :]]
                    else:
                        images = split_images
                else:
                    images = [images]

        return images, text, videos, audio

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        image_rows = [row for row_list in image_inputs["rows"] for row in row_list][image_idx]
        image_cols = [col for col_list in image_inputs["cols"] for col in col_list][image_idx]
        # The image processor decides the route; the prompt follows the mask it returned
        tokens_per_tile = self.image_seq_len * 4 if "tile_fine_mask" in image_inputs else self.image_seq_len
        image_tokens = self.image_token * tokens_per_tile
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
            fine_route = kwargs.get("fine_route", self.image_processor.fine_route)
            for height, width in image_sizes:
                num_rows, num_cols = self.image_processor.get_tile_grid(height, width, kwargs)
                image_inputs = {"rows": [[num_rows]], "cols": [[num_cols]]}
                if fine_route:
                    image_inputs["tile_fine_mask"] = None
                image_prompt = self.replace_image_token(image_inputs, 0)
                num_image_tokens.append(len(self.tokenizer(image_prompt, add_special_tokens=False)["input_ids"]))
                num_image_patches.append(self.image_processor.get_number_of_image_patches(height, width, kwargs))
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})
        return MultiModalData(**vision_data)


@auto_docstring(
    custom_intro="""
    Base class for GraniteDocling model outputs, with potential hidden states and attentions.
    """
)
@dataclass
class GraniteDoclingBaseModelOutputWithPast(Idefics3BaseModelOutputWithPast):
    r"""
    router_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
        Logits of the density router, one per sample. Returned when the model has a router and `pixel_values` are
        given.
    """

    router_logits: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    Base class for GraniteDocling causal language model (or autoregressive) outputs.
    """
)
@dataclass
class GraniteDoclingCausalLMOutputWithPast(Idefics3CausalLMOutputWithPast):
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
    Base class for GraniteDocling image features.
    """
)
@dataclass
class GraniteDoclingImageFeaturesOutput(BaseModelOutputWithPooling):
    r"""
    pooler_output (`tuple[torch.FloatTensor]` of length `batch_size`):
        Image tokens of each sample, in tile order, projected to the text hidden size. Each tensor has shape
        `(num_image_tokens, hidden_size)`. A tile yields `image_seq_len` tokens on the coarse connector path and four
        times as many on the fine path. Generation repeats these per-sample tensors across beams.
    deepstack_features (`list[tuple[torch.FloatTensor]]`, *optional*):
        One projected intermediate vision encoder hidden state per entry of `config.deepstack_visual_indexes`. Each
        entry is split per sample, with the same layout as `pooler_output`.
    router_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
        Logits of the density router, one per sample. Returned when the model has a router.
    """

    deepstack_features: list[torch.FloatTensor] | None = None
    router_logits: torch.FloatTensor | None = None


class GraniteDoclingTextPreTrainedModel(GranitePreTrainedModel):
    pass


class GraniteDoclingTextModel(GraniteModel):
    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

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
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        visual_pos_masks (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Positions of the image tokens that receive the DeepStack features.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            DeepStack features, each of shape `(num_image_tokens, hidden_size)`. Entry `i` is added to the hidden
            states at `visual_pos_masks` after decoder layer `i`.
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
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class GraniteDoclingPositionEmbedding(nn.Module):
    """
    Fixed 2D sine-cosine position embedding of the `grid_size x grid_size` image tokens of a tile, in row-major order.
    """

    def __init__(self, embed_dim: int, grid_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.pos_embed = nn.Buffer(self.build(embed_dim, grid_size), persistent=False)

    @staticmethod
    def build(embed_dim: int, grid_size: int) -> torch.Tensor:
        # vit_mae's layout is [rows | columns]; the checkpoints were trained with [columns | rows]
        return build_2d_sinusoidal_position_embedding(grid_size, grid_size, embed_dim).roll(embed_dim // 2, -1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.pos_embed.to(hidden_states.dtype)


class GraniteDoclingDeepStackMerger(InternVLMultiModalProjector):
    def __init__(self, config: GraniteDoclingVisionConfig, scale_factor: int):
        nn.Module.__init__(self)
        merged_dim = config.hidden_size * (scale_factor**2)
        self.layer_norm = nn.LayerNorm(merged_dim)
        self.linear_1 = nn.Linear(merged_dim, config.out_hidden_size)
        self.act = ACT2FN["gelu"]
        self.linear_2 = nn.Linear(config.out_hidden_size, config.out_hidden_size)


class GraniteDoclingConnectorPath(nn.Module):
    """Projection, fixed 2D position embedding and DeepStack mergers for one pixel-shuffle factor."""

    def __init__(self, config: GraniteDoclingVisionConfig, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor
        tokens_per_side = config.image_size // config.patch_size
        self.projection = nn.Linear(config.hidden_size * (scale_factor**2), config.out_hidden_size, bias=False)
        self.pos_embed = GraniteDoclingPositionEmbedding(config.out_hidden_size, tokens_per_side // scale_factor)
        self.deepstack_mergers = nn.ModuleList(
            [GraniteDoclingDeepStackMerger(config, scale_factor) for _ in config.deepstack_visual_indexes]
        )


class GraniteDoclingConnector(Idefics3Connector):
    """
    Turns the vision encoder output of each tile into image tokens, on the coarse path (`scale_factor`) or the fine
    path (half the factor, four times as many tokens), and projects the DeepStack taps the same way. The layer norms
    and the output projection are shared by both paths.
    """

    def __init__(self, config: GraniteDoclingVisionConfig):
        nn.Module.__init__(self)
        text_hidden_size = config.out_hidden_size
        self.ln_in = nn.LayerNorm(config.hidden_size)
        self.ln_mid = nn.LayerNorm(text_hidden_size)
        self.mlp_fc2 = nn.Linear(text_hidden_size, text_hidden_size, bias=False)
        self.ln_out = nn.LayerNorm(text_hidden_size)
        self.coarse = GraniteDoclingConnectorPath(config, config.scale_factor)
        self.fine = GraniteDoclingConnectorPath(config, config.scale_factor // 2) if config.use_fine_route else None

    def forward(
        self, image_hidden_states: torch.Tensor, deepstack_hidden_states: list[torch.Tensor], fine: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Image tokens and DeepStack features of the given tiles, all on one path, each flattened to
        `(num_image_tokens, hidden_size)` in tile order.
        """
        if fine and self.fine is None:
            raise ValueError("This model has no fine connector path (`config.use_fine_route=False`).")
        path = self.fine if fine else self.coarse

        hidden_states = self.ln_in(image_hidden_states)
        hidden_states = self.pixel_shuffle(hidden_states, path.scale_factor)
        hidden_states = path.projection(hidden_states)
        hidden_states = path.pos_embed(hidden_states)
        hidden_states = nn.functional.gelu(self.ln_mid(hidden_states))
        image_features = self.ln_out(self.mlp_fc2(hidden_states))

        deepstack_features = []
        for merger, deepstack_hidden in zip(path.deepstack_mergers, deepstack_hidden_states):
            deepstack_hidden = self.pixel_shuffle(deepstack_hidden.contiguous(), path.scale_factor)
            deepstack_features.append(merger(deepstack_hidden).flatten(0, 1))
        return image_features.flatten(0, 1), deepstack_features


class GraniteDoclingDensityRouter(nn.Module):
    """
    Predicts from the vision encoder features of a sample's tiles whether the sample needs the fine connector path.
    """

    def __init__(self, config: GraniteDoclingConfig):
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
        hidden = self.norm(sample_features)
        hidden = self.fc1(hidden)
        hidden = self.act(hidden)
        return self.fc2(hidden).squeeze(-1)


class GraniteDoclingVisionEmbeddings(SiglipVisionEmbeddings):
    """
    Patch embeddings with learned position embeddings for square tiles of `config.image_size` pixels.
    Tiles are always that size, so the positions are an arange and no patch mask is required.
    """


class GraniteDoclingVisionAttention(Idefics3VisionAttention):
    def __init__(self, config):
        super().__init__(config)
        self.num_key_value_groups = 1  # needed for eager attention


class GraniteDoclingPreTrainedModel(Idefics3PreTrainedModel):
    _no_split_modules = ["GraniteDoclingEncoderLayer", "GraniteDoclingTextDecoderLayer"]
    # Some checkpoints carry extra prediction heads under `mtp.*` for serving engines; they are not part of this model.
    _keys_to_ignore_on_load_unexpected = [r"^mtp\."]

    def _init_weights(self, module):
        # Idefics3 has no `_init_weights` of its own, the chain starts at PreTrainedModel
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, GraniteDoclingVisionEmbeddings):
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
        if isinstance(module, GraniteDoclingPositionEmbedding):
            init.copy_(module.pos_embed, module.build(module.embed_dim, module.grid_size))


@auto_docstring(
    custom_intro="""
    The vision encoder of GraniteDocling with its connector on top: embeds the tiles of a page and returns the
    image tokens of every tile together with the projected DeepStack features.
    """
)
class GraniteDoclingVisionModel(Idefics3VisionTransformer):
    config: GraniteDoclingVisionConfig

    def __init__(self, config: GraniteDoclingVisionConfig):
        super().__init__(config)
        self.connector = GraniteDoclingConnector(config)
        self.post_init()

    def encode(self, pixel_values: torch.FloatTensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Vision tokens and DeepStack taps, before the connector. The route does not change this."""
        hidden_states = self.embeddings(pixel_values)
        deepstack_hidden_states = []
        for layer_idx, encoder_layer in enumerate(self.encoder.layers):
            hidden_states = encoder_layer(hidden_states, attention_mask=None)
            if layer_idx in self.config.deepstack_visual_indexes:
                deepstack_hidden_states.append(hidden_states)
        return self.post_layernorm(hidden_states), deepstack_hidden_states

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        fine: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | GraniteDoclingImageFeaturesOutput:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_tiles, num_channels, image_size, image_size)`):
            Tiles flattened across the batch, without padding tiles.
        fine (`bool`, *optional*, defaults to `False`):
            Whether the tiles take the fine connector path, which yields four times as many image tokens per tile.

        Returns:
            Image tokens of every tile in tile order (`pooler_output`) and one projected DeepStack feature per
            tapped vision layer (`deepstack_features`), with the same tile order.
        """
        last_hidden_state, deepstack_hidden_states = self.encode(pixel_values)
        image_features, deepstack_features = self.connector(last_hidden_state, deepstack_hidden_states, fine=fine)
        return GraniteDoclingImageFeaturesOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=image_features,
            deepstack_features=deepstack_features,
        )


@auto_docstring(
    custom_intro="""
    GraniteDocling model consisting of a vision encoder, a connector and a text decoder, outputting raw hidden
    states without any specific head on top.
    """
)
class GraniteDoclingModel(Idefics3Model):
    def __init__(self, config: GraniteDoclingConfig):
        GraniteDoclingPreTrainedModel.__init__(self, config)
        self.padding_idx = self.config.text_config.pad_token_id
        self.vocab_size = self.config.text_config.vocab_size

        # The connector lives in the vision model, which returns image tokens and DeepStack features
        self.vision_model = GraniteDoclingVisionModel._from_config(config.vision_config)
        self.text_model = GraniteDoclingTextModel._from_config(config.text_config)
        self.density_router = None
        if config.density_router_hidden_size is not None:
            self.density_router = GraniteDoclingDensityRouter(config)

        self.image_seq_len = int(
            ((config.vision_config.image_size // config.vision_config.patch_size) ** 2)
            / (config.vision_config.scale_factor**2)
        )
        self.image_token_id = self.config.image_token_id

        self.post_init()

    def inputs_merger(self, **super_kwargs):
        raise AttributeError("Not needed for GraniteDocling")

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
        torch_compilable_check(
            n_image_tokens == n_image_features,
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}",
        )
        return special_image_mask.to(inputs_embeds.device)

    def _real_tiles(
        self, pixel_values: torch.FloatTensor, pixel_attention_mask: torch.BoolTensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Real tiles flattened across the batch, the sample each one belongs to, and the real-tile count per sample."""
        batch_size, num_tiles = pixel_values.shape[:2]
        flat_pixels = pixel_values.to(dtype=self.dtype).flatten(0, 1)  # fp16 compatibility
        if pixel_attention_mask is None:
            # Without a mask from the processor, all-zero tiles are the padding
            real_tiles = (flat_pixels != 0).flatten(1).any(-1)
        else:
            real_tiles = pixel_attention_mask.to(device=flat_pixels.device, dtype=torch.bool).reshape(-1)
        tile_sample_index = torch.arange(batch_size, device=flat_pixels.device).repeat_interleave(num_tiles)
        real_counts = real_tiles.view(batch_size, num_tiles).sum(-1)
        return flat_pixels[real_tiles].contiguous(), tile_sample_index[real_tiles], real_counts

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        tile_fine_mask: torch.BoolTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        encoder_outputs: dict | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | GraniteDoclingImageFeaturesOutput:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_tiles, num_channels, image_size, image_size)`):
            The tensors corresponding to the input image tiles.
        tile_fine_mask (`torch.BoolTensor` of shape `(batch_size,)`, *optional*):
            One value per sample. True sends that sample's tiles through the fine connector path, which yields four
            times as many image tokens per tile.
        pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_tiles)`, *optional*):
            Real tiles, as returned by the processor. Padding tiles are false. When omitted, all-zero tiles are
            treated as padding.
        encoder_outputs (`dict`, *optional*):
            Vision-encoder outputs of the real tiles (`last_hidden_state` and `deepstack_hidden_states`), as returned
            by `predict_fine_route(..., return_encoder_outputs=True)`. When given, only the connector runs.
        """
        batch_size = pixel_values.shape[0]
        tiles, tile_sample_index, real_counts = self._real_tiles(pixel_values, pixel_attention_mask)
        if tile_fine_mask is None:
            fine_samples = torch.zeros(batch_size, dtype=torch.bool, device=tiles.device)
        else:
            fine_samples = tile_fine_mask.to(device=tiles.device, dtype=torch.bool).reshape(-1)
            if fine_samples.numel() != batch_size:
                raise ValueError(
                    f"tile_fine_mask must have one value per sample, got shape {tuple(tile_fine_mask.shape)} for "
                    f"batch size {batch_size}."
                )
        fine_tiles = fine_samples[tile_sample_index]

        hidden_size = self.config.text_config.hidden_size
        empty = tiles.new_zeros(0, hidden_size)
        num_slots = len(self.config.vision_config.deepstack_visual_indexes)
        pooler_output = [empty] * batch_size
        deepstack_features = [[empty] * batch_size for _ in range(num_slots)]
        last_hidden_states, hidden_states, attentions, tile_order = [], [], [], []
        # A sample's tiles all take its route, so every route is one pass and splits back into whole samples
        for fine in (False, True):
            route_tiles = fine_tiles if fine else ~fine_tiles
            if not route_tiles.any():
                continue
            if encoder_outputs is None:
                vision_outputs = self.vision_model(
                    pixel_values=tiles[route_tiles], fine=fine, return_dict=True, **kwargs
                )
                last_hidden_state = vision_outputs.last_hidden_state
                route_features, route_deepstack = vision_outputs.pooler_output, vision_outputs.deepstack_features
                hidden_states.append(vision_outputs.hidden_states)
                attentions.append(vision_outputs.attentions)
            else:
                last_hidden_state = encoder_outputs["last_hidden_state"][route_tiles]
                route_features, route_deepstack = self.vision_model.connector(
                    last_hidden_state,
                    [states[route_tiles] for states in encoder_outputs["deepstack_hidden_states"]],
                    fine=fine,
                )
            last_hidden_states.append(last_hidden_state)
            tile_order.append(route_tiles.nonzero().flatten())

            samples = (fine_samples == fine).nonzero().flatten().tolist()
            tokens_per_tile = self.image_seq_len * 4 if fine else self.image_seq_len
            split_sizes = [int(real_counts[sample]) * tokens_per_tile for sample in samples]
            for sample, features in zip(samples, torch.split(route_features, split_sizes)):
                pooler_output[sample] = features
            for slot, features in enumerate(route_deepstack):
                for sample, sample_features in zip(samples, torch.split(features, split_sizes)):
                    deepstack_features[slot][sample] = sample_features

        last_hidden_state = None
        if tile_order:
            # Back to tile order when a batch mixes routes
            inverse_order = torch.cat(tile_order).argsort()
            last_hidden_state = torch.cat(last_hidden_states)[inverse_order]
            hidden_states = self._merge_routes(hidden_states, inverse_order)
            attentions = self._merge_routes(attentions, inverse_order)

        router_logits = None
        if self.density_router is not None and last_hidden_state is not None:
            router_logits = self.density_router(last_hidden_state, tile_sample_index, batch_size)
        return GraniteDoclingImageFeaturesOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=tuple(pooler_output),
            hidden_states=hidden_states,
            attentions=attentions,
            deepstack_features=[tuple(features) for features in deepstack_features],
            router_logits=router_logits,
        )

    @staticmethod
    def _merge_routes(per_route: list[tuple | None], inverse_order: torch.Tensor) -> tuple | None:
        """Joins the per-layer outputs of the route passes and puts the tiles back in order."""
        if not per_route or per_route[0] is None:
            return None
        return tuple(torch.cat(layers)[inverse_order] for layers in zip(*per_route))

    def predict_fine_route(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: torch.BoolTensor | None = None,
        return_encoder_outputs: bool = False,
    ) -> torch.BoolTensor | tuple[torch.BoolTensor, dict]:
        r"""
        Predicts with the density router which samples need the fine connector path. Route them by calling the
        processor again with `fine_route=True`.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_tiles, num_channels, image_size, image_size)`):
                The tensors corresponding to the input image tiles.
            pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_tiles)`, *optional*):
                Real tiles, as returned by the processor.
            return_encoder_outputs (`bool`, *optional*, defaults to `False`):
                Whether to also return the vision encoder outputs, to pass as `encoder_outputs` to
                `get_image_features` so the encoder does not run again.

        Returns:
            `torch.BoolTensor` of shape `(batch_size,)`: `True` for the samples that need the fine connector path,
            and the encoder outputs when `return_encoder_outputs=True`.
        """
        if self.density_router is None:
            raise ValueError("This model has no density router, set `density_router_hidden_size` in the config.")
        tiles, tile_sample_index, _ = self._real_tiles(pixel_values, pixel_attention_mask)
        last_hidden_state, deepstack_hidden_states = self.vision_model.encode(tiles)
        router_logits = self.density_router(last_hidden_state, tile_sample_index, pixel_values.shape[0])
        fine_route = torch.sigmoid(router_logits) >= self.config.density_router_threshold
        if return_encoder_outputs:
            encoder_outputs = {
                "last_hidden_state": last_hidden_state,
                "deepstack_hidden_states": deepstack_hidden_states,
            }
            return fine_route, encoder_outputs
        return fine_route

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
        pixel_attention_mask: torch.BoolTensor | None = None,
        mm_encoder_outputs: dict | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | GraniteDoclingBaseModelOutputWithPast:
        r"""
        tile_fine_mask (`torch.BoolTensor` of shape `(batch_size,)`, *optional*):
            One value per sample. True sends that sample's real tiles through the fine connector path, which yields four times as many image tokens per tile.
        pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_tiles)`, *optional*):
            Real tiles, as returned by the processor. Padding tiles are false. When omitted, all-zero tiles are
            treated as padding.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        kwargs.pop("router_labels", None)
        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(self.device)

        if pixel_values is not None and mm_encoder_outputs is not None:
            raise ValueError("You cannot specify both pixel_values and mm_encoder_outputs at the same time")
        if mm_encoder_outputs is not None and not isinstance(mm_encoder_outputs, dict):
            raise ValueError("mm_encoder_outputs must be a dict of cached encoder outputs.")

        image_hidden_states = None
        deepstack_visual_embeds = None
        visual_pos_masks = None
        router_logits = None
        mm_encoder_outputs = {} if mm_encoder_outputs is None else mm_encoder_outputs
        if mm_encoder_outputs.get("image") is not None or pixel_values is not None:
            if mm_encoder_outputs.get("image") is None:
                mm_encoder_outputs["image"] = self.get_image_features(
                    pixel_values,
                    tile_fine_mask=tile_fine_mask,
                    pixel_attention_mask=pixel_attention_mask,
                    return_dict=True,
                )
            image_outputs = mm_encoder_outputs["image"]
            image_hidden_states = torch.cat(image_outputs.pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            router_logits = image_outputs.router_logits
            deepstack_visual_embeds = image_outputs.deepstack_features
            if deepstack_visual_embeds is not None:
                deepstack_visual_embeds = [
                    torch.cat(features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                    for features in deepstack_visual_embeds
                ]
            special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_hidden_states)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask.unsqueeze(-1), image_hidden_states)
            visual_pos_masks = special_image_mask

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return GraniteDoclingBaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
            router_logits=router_logits,
        )


@auto_docstring(
    custom_intro="""
    The GraniteDocling Model with a language modeling head. It is made up of a vision encoder, a connector and a
    text decoder, with a language modeling head on top.
    """
)
class GraniteDoclingForConditionalGeneration(Idefics3ForConditionalGeneration):
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        tile_fine_mask: torch.BoolTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | GraniteDoclingImageFeaturesOutput:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_tiles, num_channels, image_size, image_size)`):
            The tensors corresponding to the input image tiles. All-zero tiles are padding and are discarded.
        tile_fine_mask (`torch.BoolTensor` of shape `(batch_size,)`, *optional*):
            One value per sample. True sends that sample's real tiles through the fine connector path, which yields four times as many image tokens per tile.
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
        pixel_attention_mask: torch.BoolTensor | None = None,
        mm_encoder_outputs: dict | None = None,
        labels: torch.LongTensor | None = None,
        router_labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | GraniteDoclingCausalLMOutputWithPast:
        r"""
        tile_fine_mask (`torch.BoolTensor` of shape `(batch_size,)`, *optional*):
            One value per sample. True sends that sample's real tiles through the fine connector path, which yields four times as many image tokens per tile.
        pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_tiles)`, *optional*):
            Real tiles, as returned by the processor. Padding tiles are false. When omitted, all-zero tiles are
            treated as padding.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or `model.image_token_id` (where `model` is your instance of `GraniteDoclingForConditionalGeneration`).
            Tokens with indices set to `model.image_token_id` are ignored (masked), the loss is only
            computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        router_labels (`torch.Tensor` of shape `(batch_size,)`, *optional*):
            Targets of the density router, `1` (or `True`) for the samples that need the fine connector path and `0`
            otherwise. Given as float, bool or int; the router loss is a binary cross-entropy on the logits.

        Example:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForImageTextToText
        >>> from transformers.image_utils import load_image

        >>> image = load_image("https://huggingface.co/docling-project/granite-for-docling-500m/resolve/main/docling_technical_report_p1.png")

        >>> processor = AutoProcessor.from_pretrained("docling-project/granite-for-docling-500m")
        >>> model = AutoModelForImageTextToText.from_pretrained("docling-project/granite-for-docling-500m", device_map="auto")

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

        >>> generated_ids = model.generate(**inputs, max_new_tokens=4096)
        >>> print(processor.decode(generated_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False))
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            tile_fine_mask=tile_fine_mask,
            pixel_attention_mask=pixel_attention_mask,
            mm_encoder_outputs=mm_encoder_outputs,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        router_logits = outputs.router_logits

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
        if router_labels is not None and router_logits is not None:
            router_labels = router_labels.to(router_logits.device, router_logits.dtype).reshape(-1)
            router_loss = self.config.density_router_loss_weight * nn.functional.binary_cross_entropy_with_logits(
                router_logits.reshape(-1), router_labels
            )
            loss = router_loss if loss is None else loss + router_loss

        return GraniteDoclingCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            router_logits=router_logits,
        )


__all__ = [
    "GraniteDoclingConfig",
    "GraniteDoclingForConditionalGeneration",
    "GraniteDoclingImageProcessor",
    "GraniteDoclingImageProcessorPil",
    "GraniteDoclingModel",
    "GraniteDoclingPreTrainedModel",
    "GraniteDoclingProcessor",
    "GraniteDoclingTextConfig",
    "GraniteDoclingTextModel",
    "GraniteDoclingTextPreTrainedModel",
    "GraniteDoclingVisionConfig",
    "GraniteDoclingVisionModel",
]
