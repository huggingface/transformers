# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
    infer_channel_dimension_format,
)
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.import_utils import requires
from ...utils.output_capturing import capture_outputs
from ..deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from ..deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MLP,
    DeepseekV2Model,
    DeepseekV2Moe,
    DeepseekV2PreTrainedModel,
)
from ..got_ocr2.image_processing_got_ocr2 import (
    GotOcr2ImageProcessor,
    GotOcr2ImageProcessorKwargs,
    get_optimal_tiled_canvas,
)
from ..got_ocr2.image_processing_pil_got_ocr2 import GotOcr2ImageProcessorPil
from ..llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
from ..llava_next.modeling_llava_next import (
    LlavaNextCausalLMOutputWithPast,
    LlavaNextForConditionalGeneration,
    LlavaNextModel,
    LlavaNextModelOutputWithPast,
    LlavaNextPreTrainedModel,
)
from ..qwen2.configuration_qwen2 import Qwen2Config
from ..qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2MLP,
    Qwen2Model,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)
from ..sam.configuration_sam import SamVisionConfig
from ..sam.modeling_sam import (
    SamPatchEmbeddings,
    SamVisionAttention,
    SamVisionEncoder,
    SamVisionLayer,
    SamVisionNeck,
)


logger = logging.get_logger(__name__)


class DeepseekOcr2ImageProcessorKwargs(GotOcr2ImageProcessorKwargs, total=False):
    r"""
    crop_to_patches (`bool`, *optional*, defaults to `self.crop_to_patches`):
        Whether to crop the image to patches. Can be overridden by the `crop_to_patches` parameter in the
        `preprocess` method.
    min_patches (`int`, *optional*, defaults to `self.min_patches`):
        The minimum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
        set to `True`. Can be overridden by the `min_patches` parameter in the `preprocess` method.
    max_patches (`int`, *optional*, defaults to `self.max_patches`):
        The maximum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
        set to `True`. Can be overridden by the `max_patches` parameter in the `preprocess` method.
    tile_size (`int`, *optional*, defaults to `768`):
        The size of each local tile. Must match the model's query embedding size.
    background_color (`list[int]`, *optional*, defaults to `[127, 127, 127]`):
        The background color for padding.
    """

    tile_size: int
    background_color: list[int]


@auto_docstring
class DeepseekOcr2ImageProcessor(GotOcr2ImageProcessor):
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 1024, "width": 1024}
    tile_size = 768
    crop_to_patches = True
    min_patches = 2
    max_patches = 6
    background_color = [127, 127, 127]
    model_input_names = ["pixel_values", "num_local_patches"]

    # Copied from transformers.models.llava.image_processing_llava.LlavaImageProcessor.pad_to_square
    def pad_to_square(
        self,
        images: "torch.Tensor",
        background_color: int | tuple[int, int, int] = 0,
    ) -> "torch.Tensor":
        """
        Pads an image to a square based on the longest edge.

        Args:
            images (`torch.Tensor`):
                The images to pad. Shape: (batch_size, num_channels, height, width) or (num_channels, height, width).
            background_color (`int` or `tuple[int, int, int]`, *optional*, defaults to 0):
                The color to use for the padding. Can be an integer for single channel or a
                tuple of integers representing for multi-channel images. If passed as integer
                in multi-channel mode, it will default to `0` in subsequent channels.
        Returns:
            `torch.Tensor`: The padded images.
        """
        height, width = images.shape[-2:]

        if height == width:
            return images

        num_channels = images.shape[1] if len(images.shape) == 4 else images.shape[0]
        if isinstance(background_color, int):
            background_color = [background_color] + [0] * (num_channels - 1)
        elif len(background_color) != num_channels:
            raise ValueError(
                f"background_color must have no more than {num_channels} elements to match the number of channels"
            )

        max_dim = max(height, width)
        paste_x_left = (max_dim - width) // 2
        paste_y_left = (max_dim - height) // 2
        paste_x_right = max_dim - width - paste_x_left
        paste_y_right = max_dim - height - paste_y_left
        padded_images = tvF.pad(
            images, padding=[paste_x_left, paste_y_left, paste_x_right, paste_y_right], fill=background_color
        )

        return padded_images

    def crop_image_to_patches(
        self,
        images: "torch.Tensor",
        min_patches: int,
        max_patches: int,
        tile_size: int,
        resample: PILImageResampling | None = None,
    ) -> tuple["torch.Tensor", int]:
        """
        Crop batched images to patches based on optimal tiling.

        Args:
            images (`torch.Tensor`):
                The images to crop, shape `(batch, channels, height, width)`.
            min_patches (`int`):
                Minimum number of patches.
            max_patches (`int`):
                Maximum number of patches.
            tile_size (`int`):
                The size of each tile.
            resample (`PILImageResampling`, *optional*):
                Resampling filter for resizing.

        Returns:
            `tuple[torch.Tensor, int]`: Stacked patches `(batch, num_patches, channels, tile_size, tile_size)`
            and number of patches per image.
        """
        original_height, original_width = images.shape[-2:]

        num_columns, num_rows = get_optimal_tiled_canvas(
            (original_height, original_width), (tile_size, tile_size), min_patches, max_patches
        )

        target_width = tile_size * num_columns
        target_height = tile_size * num_rows
        num_blocks = num_columns * num_rows

        resized = self.resize(images, SizeDict(height=target_height, width=target_width), resample=resample)

        patches = []
        for i in range(num_blocks):
            col = i % num_columns
            row = i // num_columns
            patch = resized[
                ...,
                row * tile_size : (row + 1) * tile_size,
                col * tile_size : (col + 1) * tile_size,
            ]
            patches.append(patch)

        stacked_patches = torch.stack(patches, dim=1)

        return stacked_patches, num_blocks

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        size: SizeDict,
        crop_to_patches: bool,
        min_patches: int,
        max_patches: int,
        tile_size: int,
        resample: PILImageResampling | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        # --- Local patches (batched by shape group) ---
        num_local_patches = {}
        local_patches_grouped = {}

        if crop_to_patches:
            grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)

            for shape, stacked_images in grouped_images.items():
                h, w = shape[-2:]
                if max(h, w) > tile_size:
                    stacked_patches, n_patches = self.crop_image_to_patches(
                        stacked_images,
                        min_patches=min_patches,
                        max_patches=max_patches,
                        tile_size=tile_size,
                        resample=resample,
                    )
                    flat_patches = stacked_patches.reshape(-1, *stacked_patches.shape[2:])
                    flat_patches = self.rescale_and_normalize(
                        flat_patches, do_rescale, rescale_factor, do_normalize, image_mean, image_std
                    )
                    local_patches_grouped[shape] = flat_patches.reshape(stacked_patches.shape)
                    num_local_patches[shape] = [n_patches] * stacked_images.shape[0]
                else:
                    local_patches_grouped[shape] = [None] * stacked_images.shape[0]
                    num_local_patches[shape] = [0] * stacked_images.shape[0]

            num_local_patches = reorder_images(num_local_patches, grouped_images_index)
            ordered_local = reorder_images(local_patches_grouped, grouped_images_index)
        else:
            num_local_patches = [0] * len(images)
            ordered_local = []

        flat_local_list = [patch for item in ordered_local if item is not None for patch in item]

        # --- Global view (batched by shape group) ---
        global_target_size = size.height if crop_to_patches else tile_size

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_global_grouped = {}
        for shape, stacked in grouped_images.items():
            h, w = shape[-2:]
            scale = global_target_size / max(h, w)
            new_h = round(h * scale)
            new_w = round(w * scale)
            stacked = self.resize(stacked, SizeDict(height=new_h, width=new_w), resample=resample)
            stacked = self.pad_to_square(stacked, background_color=self.background_color)
            stacked = self.rescale_and_normalize(
                stacked, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_global_grouped[shape] = stacked
        all_pixel_values_global = reorder_images(processed_global_grouped, grouped_images_index)

        data = {
            "pixel_values": all_pixel_values_global,
            "num_local_patches": num_local_patches,
        }
        if flat_local_list:
            data["pixel_values_local"] = flat_local_list

        return BatchFeature(data=data, tensor_type=return_tensors)

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> int:
        """
        Returns the number of image patches for a given image size (1 global + local patches).
        """
        if images_kwargs is None:
            images_kwargs = {}
        min_patches = images_kwargs.get("min_patches", self.min_patches)
        max_patches = images_kwargs.get("max_patches", self.max_patches)
        tile_size = images_kwargs.get("tile_size", self.tile_size)
        crop_to_patches = images_kwargs.get("crop_to_patches", self.crop_to_patches)

        num_patches = 1  # global view
        if crop_to_patches and max(height, width) > tile_size:
            num_columns, num_rows = get_optimal_tiled_canvas(
                (height, width), (tile_size, tile_size), min_patches, max_patches
            )
            num_patches += num_columns * num_rows

        return num_patches


@requires(backends=("vision",))
@auto_docstring
class DeepseekOcr2ImageProcessorPil(GotOcr2ImageProcessorPil):
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 1024, "width": 1024}
    tile_size = 768
    crop_to_patches = True
    min_patches = 2
    max_patches = 6
    background_color = [127, 127, 127]
    model_input_names = ["pixel_values", "num_local_patches"]

    def crop_image_to_patches(
        self,
        image: np.ndarray,
        min_patches: int,
        max_patches: int,
        tile_size: int,
        resample: "PILImageResampling | int | None" = None,
    ):
        """
        Crop the image to patches and return a list of cropped images.
        """
        input_data_format = infer_channel_dimension_format(image)
        image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)

        original_height, original_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)

        num_columns, num_rows = get_optimal_tiled_canvas(
            (original_height, original_width), (tile_size, tile_size), min_patches, max_patches
        )

        target_width = tile_size * num_columns
        target_height = tile_size * num_rows
        num_blocks = num_columns * num_rows

        resized_image = self.resize(image, SizeDict(height=target_height, width=target_width), resample=resample)

        processed_images = []
        for i in range(num_blocks):
            column = i % num_columns
            row = i // num_columns
            box = (
                column * tile_size,
                row * tile_size,
                (column + 1) * tile_size,
                (row + 1) * tile_size,
            )
            patch_image = resized_image[..., box[1] : box[3], box[0] : box[2]]
            patch_image = to_channel_dimension_format(patch_image, input_data_format, ChannelDimension.FIRST)
            processed_images.append(patch_image)

        return processed_images

    # Copied from transformers.models.llava.image_processing_pil_llava.LlavaImageProcessorPil.pad_to_square
    def pad_to_square(
        self,
        image: np.ndarray,
        background_color: int | tuple[int, int, int] = 0,
    ) -> np.ndarray:
        """
        Pads an image to a square based on the longest edge.

        Args:
            image (`np.ndarray`):
                The image to pad. Shape: (num_channels, height, width) - always channels_first in backend.
            background_color (`int` or `tuple[int, int, int]`, *optional*, defaults to 0):
                The color to use for the padding.

        Returns:
            `np.ndarray`: The padded image.
        """
        # Backend always uses channels_first format: (num_channels, height, width)
        num_channels, height, width = image.shape

        if height == width:
            return image

        max_dim = max(height, width)

        # Ensure background_color is the correct shape
        if isinstance(background_color, int):
            background_color = [background_color]
        elif len(background_color) != num_channels:
            raise ValueError(
                f"background_color must have no more than {num_channels} elements to match the number of channels"
            )

        result = np.zeros((num_channels, max_dim, max_dim), dtype=image.dtype)
        for i, color in enumerate(background_color):
            result[i, :, :] = color
        if width > height:
            start = (max_dim - height) // 2
            result[:, start : start + height, :] = image
        else:
            start = (max_dim - width) // 2
            result[:, :, start : start + width] = image

        return result

    def _preprocess(
        self,
        images: list[np.ndarray],
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        crop_to_patches: bool = True,
        min_patches: int = 2,
        max_patches: int = 6,
        tile_size: int = 768,
        background_color: list[int] | None = None,
        **kwargs,
    ) -> BatchFeature:
        if background_color is None:
            background_color = self.background_color

        all_pixel_values_local = []
        all_pixel_values_global = []
        num_local_patches = []

        for image in images:
            original_height, original_width = get_image_size(image)

            # --- Local patches ---
            if crop_to_patches and max(original_width, original_height) > tile_size:
                local_patches = self.crop_image_to_patches(
                    image,
                    min_patches=min_patches,
                    max_patches=max_patches,
                    tile_size=tile_size,
                    resample=resample,
                )
                for patch in local_patches:
                    if do_rescale:
                        patch = self.rescale(patch, rescale_factor)
                    if do_normalize:
                        patch = self.normalize(patch, image_mean, image_std)
                    all_pixel_values_local.append(patch)
                num_local_patches.append(len(local_patches))
            else:
                num_local_patches.append(0)

            # --- Global view ---
            global_target_size = size.height if crop_to_patches else tile_size
            scale = global_target_size / max(original_width, original_height)
            new_width = round(original_width * scale)
            new_height = round(original_height * scale)

            global_img = self.resize(image, SizeDict(height=new_height, width=new_width), resample=resample)
            global_img = self.pad_to_square(global_img, background_color=background_color)
            if do_rescale:
                global_img = self.rescale(global_img, rescale_factor)
            if do_normalize:
                global_img = self.normalize(global_img, image_mean, image_std)
            all_pixel_values_global.append(global_img)

        data = {
            "pixel_values": all_pixel_values_global,
            "num_local_patches": num_local_patches,
        }
        if all_pixel_values_local:
            data["pixel_values_local"] = all_pixel_values_local

        return BatchFeature(data=data, tensor_type=return_tensors)


@auto_docstring(checkpoint="thisisiron/DeepSeek-OCR-2-hf")
@strict
class DeepseekOcr2SamVisionConfig(SamVisionConfig):
    r"""
    output_channels (`int`, *optional*, defaults to 256):
        The number of output channels in the SAM neck.
    window_size (`int`, *optional*, defaults to 14):
        Window size for windowed attention layers.
    global_attn_indexes (`list[int]`, *optional*, defaults to `[2, 5, 8, 11]`):
        Indices of encoder layers that use global (non-windowed) attention.
    mlp_dim (`int`, *optional*):
        Dimensionality of the MLP layer in each vision encoder block. Defaults to `hidden_size * mlp_ratio`.
    downsample_channels (`list[int]`, *optional*):
        The channel dimensions for the multi-scale downsampling neck layers. Defaults to `[512, 896]`.
    """

    base_config_key = "sam_config"

    # Remove unused attribute inherited from SamVisionConfig
    num_pos_feats = AttributeError()

    downsample_channels: list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.downsample_channels is None:
            self.downsample_channels = [512, 896]
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="thisisiron/DeepSeek-OCR-2-hf")
@strict
class DeepseekOcr2VisionEncoderConfig(Qwen2Config):
    r"""
    Example:

    ```python
    >>> from transformers import DeepseekOcr2Config

    >>> config = DeepseekOcr2Config()
    >>> encoder_config = config.vision_config.encoder_config
    ```"""

    base_config_key = "encoder_config"


@auto_docstring(checkpoint="thisisiron/DeepSeek-OCR-2-hf")
@strict
class DeepseekOcr2VisionConfig(PreTrainedConfig):
    r"""
    sam_config (`dict` or `DeepseekOcr2SamVisionConfig`, *optional*):
        Configuration for the SAM vision encoder. Defaults to `DeepseekOcr2SamVisionConfig()`.
    encoder_config (`dict` or `DeepseekOcr2VisionEncoderConfig`, *optional*):
        Configuration for the DeepSeek-OCR-2 vision encoder. Defaults to `DeepseekOcr2VisionEncoderConfig()`.
    """

    base_config_key = "vision_config"
    sub_configs = {
        "sam_config": DeepseekOcr2SamVisionConfig,
        "encoder_config": DeepseekOcr2VisionEncoderConfig,
    }

    sam_config: dict | PreTrainedConfig | None = None
    encoder_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.sam_config is None:
            self.sam_config = DeepseekOcr2SamVisionConfig()
        elif isinstance(self.sam_config, dict):
            self.sam_config = DeepseekOcr2SamVisionConfig(**self.sam_config)

        if self.encoder_config is None:
            self.encoder_config = DeepseekOcr2VisionEncoderConfig()
        elif isinstance(self.encoder_config, dict):
            self.encoder_config = DeepseekOcr2VisionEncoderConfig(**self.encoder_config)

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="thisisiron/DeepSeek-OCR-2-hf")
@strict
class DeepseekOcr2TextConfig(DeepseekV2Config):
    r"""
    n_group (`int`, *optional*):
        Number of groups for grouped top-k expert routing.
    topk_method (`str`, *optional*, defaults to `"greedy"`):
        Method for selecting top-k experts in MoE layers.
    mlp_layer_types (`list[str]`, *optional*):
        MLP type (`"dense"` or `"sparse"`) for each decoder layer, e.g. `["dense", "sparse", "sparse", ...]`.
    """

    base_config_key = "text_config"
    mlp_layer_types: list[str] | None = None

    # Override DeepseekV2's MLA TP plan with standard MHA projections
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    # Remove unused attributes inherited from DeepseekV2Config
    first_k_dense_replace = AttributeError()
    kv_lora_rank = AttributeError()
    norm_topk_prob = AttributeError()
    q_lora_rank = AttributeError()
    qk_nope_head_dim = AttributeError()
    qk_rope_head_dim = AttributeError()
    v_head_dim = AttributeError()

    def __post_init__(self, **kwargs):
        self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring(checkpoint="thisisiron/DeepSeek-OCR-2-hf")
@strict
class DeepseekOcr2Config(PreTrainedConfig):
    r"""
    vision_config (`dict` or `DeepseekOcr2VisionConfig`, *optional*):
        Configuration for the vision encoders. Defaults to `DeepseekOcr2VisionConfig()`.
    """

    model_type = "deepseek_ocr2"
    sub_configs = {
        "vision_config": DeepseekOcr2VisionConfig,
        "text_config": DeepseekOcr2TextConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 128815
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = DeepseekOcr2VisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = DeepseekOcr2VisionConfig(**self.vision_config)

        if self.text_config is None:
            self.text_config = DeepseekOcr2TextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = DeepseekOcr2TextConfig(**self.text_config)

        super().__post_init__(**kwargs)


@dataclass
class DeepseekOcr2ModelOutputWithPooling(BaseModelOutputWithPooling):
    """
    local_last_hidden_state (`torch.FloatTensor` of shape `(total_local_patches, sequence_length, hidden_size)`, *optional*):
        Last hidden state from the vision encoder for local (cropped) patches.
    local_hidden_states (`torch.FloatTensor`, *optional*):
        Hidden states from all layers of the vision encoder for local patches.
    local_attentions (`torch.FloatTensor`, *optional*):
        Attention weights from all layers of the vision encoder for local patches.
    """

    local_last_hidden_state: torch.FloatTensor | None = None
    local_hidden_states: torch.FloatTensor | None = None
    local_attentions: torch.FloatTensor | None = None


class DeepseekOcr2ModelOutputWithPast(LlavaNextModelOutputWithPast):
    pass


class DeepseekOcr2CausalLMOutputWithPast(LlavaNextCausalLMOutputWithPast):
    pass


class DeepseekOcr2PreTrainedModel(LlavaNextPreTrainedModel):
    _no_split_modules = [
        "DeepseekOcr2SamVisionLayer",
        "DeepseekOcr2VisionEncoderLayer",
        "DeepseekOcr2TextDecoderLayer",
    ]
    # SAM uses rel-pos bias, incompatible with flash attention.
    _supports_flash_attn = False

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, DeepseekOcr2SamVisionAttention):
            if module.use_rel_pos:
                init.zeros_(module.rel_pos_h)
                init.zeros_(module.rel_pos_w)
        elif isinstance(module, DeepseekOcr2SamVisionEncoder):
            if module.pos_embed is not None:
                init.zeros_(module.pos_embed)
        elif isinstance(module, DeepseekOcr2Model):
            embed_std = 1 / math.sqrt(self.config.text_config.hidden_size)
            init.normal_(module.view_separator, mean=0.0, std=embed_std)


class DeepseekOcr2SamVisionAttention(SamVisionAttention):
    pass


class DeepseekOcr2SamVisionLayer(SamVisionLayer):
    pass


class DeepseekOcr2SamVisionNeck(SamVisionNeck):
    pass


class DeepseekOcr2SamPatchEmbeddings(SamPatchEmbeddings):
    def forward(self, pixel_values):
        embeddings = self.projection(pixel_values).permute(0, 2, 3, 1)
        return embeddings


class DeepseekOcr2SamVisionProj(nn.Module):
    """Neck and multi-scale downsampling for SAM ViT-B output."""

    def __init__(self, config: DeepseekOcr2SamVisionConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(
            config.output_channels,
            config.downsample_channels[0],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            config.downsample_channels[0],
            config.downsample_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        return hidden_states


class DeepseekOcr2SamVisionEncoder(SamVisionEncoder, DeepseekOcr2PreTrainedModel):
    def __init__(self, config: DeepseekOcr2SamVisionConfig):
        super().__init__(config)
        self.proj = DeepseekOcr2SamVisionProj(config)

    def interpolate_pos_encoding(self, height: int, width: int) -> torch.Tensor:
        """Interpolate the positional encoding to match the target spatial size using bicubic interpolation."""
        if not torch.jit.is_tracing() and self.pos_embed.shape[1] == height and self.pos_embed.shape[2] == width:
            return self.pos_embed

        target_dtype = self.pos_embed.dtype
        pos_embed = self.pos_embed.permute(0, 3, 1, 2)
        pos_embed = torch.nn.functional.interpolate(
            pos_embed.to(torch.float32),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).to(dtype=target_dtype)
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(self, pixel_values: torch.FloatTensor, **kwargs) -> BaseModelOutput:
        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.interpolate_pos_encoding(
                hidden_states.shape[1], hidden_states.shape[2]
            )

        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)

        hidden_states = self.neck(hidden_states)
        hidden_states = self.proj(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class DeepseekOcr2VisionMLP(Qwen2MLP):
    pass


class DeepseekOcr2VisionRMSNorm(Qwen2RMSNorm):
    pass


class DeepseekOcr2VisionRotaryEmbedding(Qwen2RotaryEmbedding):
    pass


class DeepseekOcr2VisionAttention(Qwen2Attention):
    pass


class DeepseekOcr2VisionEncoderLayer(Qwen2DecoderLayer):
    pass


@auto_docstring(custom_intro="Vision encoder for DeepSeek-OCR-2.")
class DeepseekOcr2VisionEncoder(Qwen2Model, DeepseekOcr2PreTrainedModel):
    _can_record_outputs = {
        "hidden_states": DeepseekOcr2VisionEncoderLayer,
        "attentions": DeepseekOcr2VisionAttention,
    }

    def __init__(self, config):
        super().__init__(config)
        del self.embed_tokens
        self.layers = nn.ModuleList(
            [DeepseekOcr2VisionEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        num_patches: int,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        num_patches (`int`):
            Number of image patch tokens at the beginning of the sequence. Used to build the hybrid attention mask
            (bidirectional over image tokens, causal over query tokens).
        """
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        bsz, seq_len, _ = inputs_embeds.shape
        token_type_ids = torch.cat(
            [
                torch.zeros(bsz, num_patches, dtype=torch.long, device=inputs_embeds.device),
                torch.ones(bsz, seq_len - num_patches, dtype=torch.long, device=inputs_embeds.device),
            ],
            dim=1,
        )
        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=None,
            or_mask_function=token_type_ids_mask_function(token_type_ids),
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


def token_type_ids_mask_function(token_type_ids: torch.Tensor):
    """
    Creates an or_mask_function for `create_causal_mask` that allows
    bidirectional attention between image tokens (type_id=0).

    Args:
        token_type_ids: `(batch_size, seq_len)` tensor where 0=image, 1=query.

    Returns:
        A mask function compatible with `create_causal_mask(or_mask_function=...)`.
    """
    is_image = token_type_ids == 0

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return is_image[batch_idx, q_idx] & is_image[batch_idx, kv_idx]

    return inner_mask


class DeepseekOcr2VisionModel(DeepseekOcr2PreTrainedModel):
    """Vision pipeline: SAM ViT-B (with neck)"""

    def __init__(self, config: DeepseekOcr2VisionConfig):
        super().__init__(config)
        self.sam_encoder = DeepseekOcr2SamVisionEncoder(config.sam_config)
        self.vision_encoder = DeepseekOcr2VisionEncoder(config.encoder_config)

        # Resolution-specific learnable queries
        self.query_768_resolution = nn.Embedding(144, config.encoder_config.hidden_size)  # 12x12 for 768px
        self.query_1024_resolution = nn.Embedding(256, config.encoder_config.hidden_size)  # 16x16 for 1024px
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(self, pixel_values: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> BaseModelOutput:
        sam_encoder_outputs = self.sam_encoder(pixel_values, **kwargs)
        hidden_states = sam_encoder_outputs.last_hidden_state.flatten(2).transpose(1, 2)
        bsz, num_patches, _ = hidden_states.shape

        queries = self.query_768_resolution.weight if num_patches <= 144 else self.query_1024_resolution.weight
        queries = queries.unsqueeze(0).expand(bsz, -1, -1)
        combined = torch.cat([hidden_states, queries], dim=1)

        encoder_outputs = self.vision_encoder(
            inputs_embeds=combined,
            num_patches=num_patches,
            **kwargs,
        )

        query_features = encoder_outputs.last_hidden_state[:, num_patches:, :]

        return BaseModelOutput(
            last_hidden_state=query_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class DeepseekOcr2TextRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class DeepseekOcr2TextAttention(LlamaAttention):
    pass


class DeepseekOcr2TextMLP(DeepseekV2MLP):
    pass


class DeepseekOcr2TextMoe(DeepseekV2Moe):
    pass


class DeepseekOcr2TextDecoderLayer(DeepseekV2DecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = DeepseekOcr2TextAttention(config=config, layer_idx=layer_idx)
        self.mlp = (
            DeepseekOcr2TextMoe(config)
            if config.mlp_layer_types[layer_idx] == "sparse"
            else DeepseekOcr2TextMLP(config)
        )


class DeepseekOcr2TextPreTrainedModel(DeepseekV2PreTrainedModel):
    pass


class DeepseekOcr2TextModel(DeepseekV2Model):
    def __init__(self, config: DeepseekOcr2TextConfig):
        super().__init__(config)
        # Use (cos/sin) RoPE instead of complex RoPE to match LlamaAttention (MHA)
        self.rotary_emb = DeepseekOcr2TextRotaryEmbedding(config=config)


class DeepseekOcr2Model(LlavaNextModel):
    def __init__(self, config: DeepseekOcr2Config):
        super().__init__(config)
        del embed_std  # noqa: F821
        del self.image_newline

        self.vision_tower = DeepseekOcr2VisionModel(config.vision_config)
        self.multi_modal_projector = nn.Linear(
            config.vision_config.encoder_config.hidden_size, config.text_config.hidden_size
        )

        # Learnable separator between local and global views (initialized in `_init_weights`).
        self.view_separator = nn.Parameter(torch.empty(config.text_config.hidden_size))

        self.language_model = DeepseekOcr2TextModel(config.text_config)

    def pack_image_features(self):
        raise NotImplementedError("DeepseekOcr2 does not use pack_image_features")

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_values_local: torch.FloatTensor | None = None,
        num_local_patches: list[int] | torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        pixel_values_local (`torch.FloatTensor` of shape `(total_patches, 3, height, width)`, *optional*):
            All local patches flattened across the batch, or `None` if no local views.
        num_local_patches (`list[int]` or `torch.Tensor`, *optional*):
            Number of local patches per image, e.g. `[6, 0, 4]`.
        """
        # torch.split requires list[int], not Tensor, for per-image variable-length splitting
        if isinstance(num_local_patches, torch.Tensor):
            num_local_patches = num_local_patches.tolist()

        batch_size = pixel_values.shape[0]

        global_vision_outputs = self.vision_tower(pixel_values, **kwargs)
        global_features = self.multi_modal_projector(global_vision_outputs.last_hidden_state)

        local_outputs = {}
        if pixel_values_local is not None:
            local_vision_outputs = self.vision_tower(pixel_values_local, **kwargs)
            all_local_features = self.multi_modal_projector(local_vision_outputs.last_hidden_state)
            per_image_local = torch.split(all_local_features, num_local_patches, dim=0)
            local_outputs = {
                "local_last_hidden_state": local_vision_outputs.last_hidden_state,
                "local_hidden_states": local_vision_outputs.hidden_states,
                "local_attentions": local_vision_outputs.attentions,
            }
        else:
            per_image_local = [None] * batch_size

        all_features = []
        view_sep = self.view_separator.to(global_features.device).unsqueeze(0)
        for idx in range(batch_size):
            global_flat = global_features[idx].reshape(-1, global_features.shape[-1])

            if per_image_local[idx] is not None:
                local_flat = per_image_local[idx].reshape(-1, per_image_local[idx].shape[-1])
                all_features.append(torch.cat([local_flat, global_flat, view_sep], dim=0))
            else:
                all_features.append(torch.cat([global_flat, view_sep], dim=0))

        image_features = torch.cat(all_features, dim=0)
        return DeepseekOcr2ModelOutputWithPooling(
            last_hidden_state=global_vision_outputs.last_hidden_state,
            pooler_output=image_features,
            hidden_states=global_vision_outputs.hidden_states,
            attentions=global_vision_outputs.attentions,
            **local_outputs,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_local: torch.FloatTensor | None = None,
        num_local_patches: list[int] | torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | DeepseekOcr2ModelOutputWithPast:
        r"""
        pixel_values_local (`torch.FloatTensor`, *optional*):
            Local patch pixel values of shape `(total_patches, 3, H, W)`.
        num_local_patches (`list[int]` or `torch.Tensor`, *optional*):
            Number of local patches per image in the batch.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values, pixel_values_local, num_local_patches, return_dict=True
            ).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

            special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        return DeepseekOcr2ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )


@auto_docstring
class DeepseekOcr2ForConditionalGeneration(LlavaNextForConditionalGeneration):
    def pack_image_features(self):
        raise NotImplementedError("DeepseekOcr2 does not use pack_image_features")

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_values_local: torch.FloatTensor | None = None,
        num_local_patches: list[int] | torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, 3, height, width)`):
            The tensors corresponding to the global view input images.
        pixel_values_local (`torch.FloatTensor` of shape `(total_patches, 3, height, width)`, *optional*):
            All local patches flattened across the batch, or `None` if no local views.
        num_local_patches (`list[int]` or `torch.Tensor`, *optional*):
            Number of local patches per image, e.g. `[6, 0, 4]`.
        """
        return self.model.get_image_features(
            pixel_values=pixel_values,
            pixel_values_local=pixel_values_local,
            num_local_patches=num_local_patches,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_local=None,
        num_local_patches=None,
        attention_mask=None,
        logits_to_keep=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if is_first_iteration or not kwargs.get("use_cache", True):
            model_inputs["pixel_values"] = pixel_values
            model_inputs["pixel_values_local"] = pixel_values_local
            model_inputs["num_local_patches"] = num_local_patches

        return model_inputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_local: torch.FloatTensor | None = None,
        num_local_patches: list[int] | torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | DeepseekOcr2CausalLMOutputWithPast:
        r"""
        pixel_values_local (`torch.FloatTensor`, *optional*):
            Local patch pixel values of shape `(total_patches, 3, H, W)`.
        num_local_patches (`list[int]` or `torch.Tensor`, *optional*):
            Number of local patches per image in the batch.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_local=pixel_values_local,
            num_local_patches=num_local_patches,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

        return DeepseekOcr2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = [
    "DeepseekOcr2Config",
    "DeepseekOcr2TextConfig",
    "DeepseekOcr2VisionConfig",
    "DeepseekOcr2VisionEncoderConfig",
    "DeepseekOcr2SamVisionConfig",
    "DeepseekOcr2ForConditionalGeneration",
    "DeepseekOcr2ImageProcessor",
    "DeepseekOcr2ImageProcessorPil",
    "DeepseekOcr2Model",
    "DeepseekOcr2PreTrainedModel",
    "DeepseekOcr2TextModel",
    "DeepseekOcr2TextPreTrainedModel",
    "DeepseekOcr2VisionModel",
]
