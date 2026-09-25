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

"""PyTorch Molmo2 model."""

import itertools
import math
from collections.abc import Callable

import torch
from huggingface_hub.dataclasses import strict
from torch import nn
from torch.nn import functional as F

from ... import initialization as init
from ...activations import ACT2FN
from ...backbone_utils import filter_output_hidden_states
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...masking_utils import create_bidirectional_mask, create_causal_mask, create_masks_for_generate
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import (
    ImagesKwargs,
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    VideosKwargs,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import VideoInput, VideoMetadata
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaModel,
    LlavaModelOutputWithPast,
)
from ..olmo.modeling_olmo import OlmoMLP
from ..olmo2.modeling_olmo2 import Olmo2Attention
from ..siglip2.modeling_siglip2 import (
    Siglip2EncoderLayer,
    Siglip2MLP,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="allenai/Molmo2-8B")
@strict
class Molmo2VisionConfig(PreTrainedConfig):
    r"""
    image_size (`list[int]`, *optional*):
        Input image size as (height, width), `[378, 378]` when not provided.
    num_position_embeddings (`int`, *optional*):
        Number of positional embeddings, one per image patch. Derived from `image_size` and `patch_size` when not
        provided.
    """

    model_type = "molmo2"
    base_config_key = "vision_config"
    # Keys of the released checkpoints' `config.json`.
    attribute_map = {
        "image_default_input_size": "image_size",
        "image_patch_size": "patch_size",
        "image_num_pos": "num_position_embeddings",
    }

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    image_size: list[int] | None = None
    patch_size: int = 14
    num_position_embeddings: int | None = None
    num_channels: int = 3
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.image_size is None:
            self.image_size = [378, 378]
        if self.num_position_embeddings is None:
            self.num_position_embeddings = (self.image_size[0] // self.patch_size) * (
                self.image_size[1] // self.patch_size
            )
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="allenai/Molmo2-8B")
@strict
class Molmo2AdapterConfig(PreTrainedConfig):
    r"""
    vision_feature_layer (`list[int]`, *optional*):
        Indices of the ViT layers whose outputs are concatenated and pooled, `[-3, -9]` when not provided.
    text_hidden_size (`int`, *optional*, defaults to 3584):
        Hidden size of the text model (used for projection).
    image_feature_dropout (`float`, *optional*, defaults to 0.0):
        Dropout rate for image features.
    """

    model_type = "molmo2"
    base_config_key = "adapter_config"
    attribute_map = {"vit_layers": "vision_feature_layer"}

    vision_feature_layer: list[int] | None = None
    hidden_size: int = 1152
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    hidden_act: str = "silu"
    intermediate_size: int = 18944
    text_hidden_size: int = 3584
    image_feature_dropout: float = 0.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.vision_feature_layer is None:
            self.vision_feature_layer = [-3, -9]
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="allenai/Molmo2-8B")
@strict
class Molmo2TextConfig(PreTrainedConfig):
    r"""
    additional_vocab_size (`int`, *optional*, defaults to 128):
        Number of additional vocabulary tokens beyond the base vocabulary.
    qk_norm_type (`str`, *optional*, defaults to `"qwen3"`):
        Query/key normalization layout used by the checkpoint. `"qwen3"` normalizes per head; `"olmo"` normalizes the
        full projected query/key tensors.
    embedding_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the embedding layer.
    residual_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio applied after residual connections.
    rope_parameters (`RopeParameters`, *optional*):
        RoPE parameters for the model.
    rope_scaling_layers (`list[int]`, *optional*):
        Indices of the layers that apply the scaled RoPE described by `rope_parameters`. The remaining layers use an
        unscaled RoPE with the same theta. All layers are scaled when not provided.
    norm_after (`bool`, *optional*, defaults to `False`):
        Whether to apply layer normalization after the attention/FFN blocks instead of before.
    """

    model_type = "molmo2_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"qkv_bias": "attention_bias", "layer_norm_eps": "rms_norm_eps"}
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_gather_output",
        "layers.*.self_attn.k_proj": "colwise_gather_output",
        "layers.*.self_attn.v_proj": "colwise_gather_output",
        "layers.*.self_attn.o_proj": "rowwise_split_input",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    additional_vocab_size: int = 128
    attention_bias: bool = False
    qk_norm_type: str = "qwen3"
    num_hidden_layers: int = 36
    intermediate_size: int = 12288
    hidden_act: str = "silu"
    embedding_dropout: float = 0.0
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    max_position_embeddings: int = 36864
    rope_parameters: RopeParameters | dict | None = None
    rope_scaling_layers: list[int] | None = None
    rms_norm_eps: float = 1e-6
    norm_after: bool = False
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if self.rope_scaling_layers is None:
            self.rope_scaling_layers = list(range(self.num_hidden_layers))
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        super().validate_architecture()
        if self.qk_norm_type not in ("qwen3", "olmo"):
            raise ValueError(f"Unsupported `qk_norm_type`: {self.qk_norm_type}")


@auto_docstring(checkpoint="allenai/Molmo2-8B")
@strict
class Molmo2Config(PreTrainedConfig):
    r"""
    vision_config (`Molmo2VisionConfig`, *optional*):
        Configuration for the vision transformer backbone.
    adapter_config (`Molmo2AdapterConfig`, *optional*):
        Configuration for the vision-to-language adapter.
    image_start_token_id (`int`, *optional*, defaults to 151936):
        Token ID marking the start of an image region.
    low_res_image_start_token_id (`int`, *optional*, defaults to 151940):
        Token ID marking the start of a low-resolution image crop.
    image_end_token_id (`int`, *optional*, defaults to 151937):
        Token ID marking the end of an image region.
    image_patch_id (`int`, *optional*, defaults to 151938):
        Token ID for image patches.
    frame_start_token_id (`int`, *optional*, defaults to 151943):
        Token ID marking the start of a video frame.
    frame_end_token_id (`int`, *optional*, defaults to 151944):
        Token ID marking the end of a video frame.
    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
        Whether the model's input and output word embeddings should be tied.
    """

    model_type = "molmo2"
    attribute_map = {"image_token_id": "image_patch_id"}
    sub_configs = {
        "text_config": Molmo2TextConfig,
        "vision_config": Molmo2VisionConfig,
        "adapter_config": Molmo2AdapterConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    adapter_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_start_token_id: int = 151936
    low_res_image_start_token_id: int = 151940
    image_end_token_id: int = 151937
    image_patch_id: int = 151938
    frame_start_token_id: int = 151943
    frame_end_token_id: int = 151944
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        # Checkpoints serialized before the rename store the vision sub-config under `vit_config`.
        legacy_vision_config = kwargs.pop("vit_config", None)
        if self.vision_config is None and legacy_vision_config is not None:
            self.vision_config = legacy_vision_config

        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.adapter_config, dict):
            self.adapter_config = self.sub_configs["adapter_config"](**self.adapter_config)
        elif self.adapter_config is None:
            self.adapter_config = self.sub_configs["adapter_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        # Normalize negative `vision_feature_layer` indices and trim the ViT to the deepest layer the adapter reads.
        num_vit_layers = self.vision_config.num_hidden_layers
        self.adapter_config.vision_feature_layer = [
            layer if layer >= 0 else layer + num_vit_layers for layer in self.adapter_config.vision_feature_layer
        ]
        last_layer_needed = max(self.adapter_config.vision_feature_layer) + 1
        if last_layer_needed < num_vit_layers:
            self.vision_config.num_hidden_layers = last_layer_needed

        super().__post_init__(**kwargs)


def select_tiling(height: int, width: int, patch_size: int, max_num_crops: int) -> tuple[int, int]:
    """Same as `get_optimal_tiled_canvas` in cohere2_vision, with the candidates in (height, width) order and ties
    broken on height, as in the original Molmo2 processor."""
    tilings = [
        (tile_height, tile_width)
        for tile_height, tile_width in itertools.product(range(1, max_num_crops + 1), repeat=2)
        if tile_height * tile_width <= max_num_crops
    ]
    tilings.sort(key=lambda x: (x[0] * x[1], x[0]))

    candidate_resolutions = torch.tensor(tilings, dtype=torch.int32) * patch_size
    original_size = torch.tensor([height, width], dtype=torch.float32)

    required_scales = candidate_resolutions.to(torch.float32) / original_size
    required_scale = required_scales.amin(dim=-1, keepdim=True)

    if torch.all(required_scale < 1):
        return tilings[int(required_scale.argmax())]

    required_scale = torch.where(required_scale < 1.0, 10e9, required_scale)
    return tilings[int(required_scale.argmin())]


def compute_crop_geometry(
    height: int,
    width: int,
    max_crops: int,
    overlap_margins: list[int],
    crop_size: int,
    patch_size: int,
) -> tuple[int, int, int, int]:
    """Tiling `(tiling_h, tiling_w)` and canvas pixel size `(canvas_height, canvas_width)` for one image."""
    left_margin, right_margin = overlap_margins
    crop_patches = crop_size // patch_size
    window_patches = crop_patches - (left_margin + right_margin)
    window_size = window_patches * patch_size
    margin_size = (left_margin + right_margin) * patch_size
    tiling_h, tiling_w = select_tiling(height - margin_size, width - margin_size, window_size, max_crops)
    canvas_height = tiling_h * window_size + margin_size
    canvas_width = tiling_w * window_size + margin_size
    return tiling_h, tiling_w, canvas_height, canvas_width


def batch_pixels_to_patches(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Reshape images of [num_images, h, w, channels] -> [num_images, n_patches, pixels_per_patch]"""
    num_crops, height, width, channels = images.shape
    num_patches_height = height // patch_size
    num_patches_width = width // patch_size
    images = images.reshape(num_crops, num_patches_height, patch_size, num_patches_width, patch_size, channels)
    images = images.transpose(2, 3)
    images = images.reshape(num_crops, num_patches_height * num_patches_width, patch_size * patch_size * channels)
    return images


def arange_for_pooling(
    index_grid: torch.Tensor,
    pool_h: int,
    pool_w: int,
) -> tuple[torch.Tensor, int, int]:
    """Group a [h, w] patch-index grid into pooling windows: `[num_rows * num_cols, pool_h * pool_w]`, padded
    with -1. Also returns the pooled grid's `(num_rows, num_cols)`."""
    height_padding = pool_h * ((index_grid.shape[0] + pool_h - 1) // pool_h) - index_grid.shape[0]
    width_padding = pool_w * ((index_grid.shape[1] + pool_w - 1) // pool_w) - index_grid.shape[1]
    index_grid = F.pad(
        index_grid,
        (width_padding // 2, (width_padding + 1) // 2, height_padding // 2, (height_padding + 1) // 2),
        mode="constant",
        value=-1,
    )
    num_rows, num_cols = index_grid.shape[0] // pool_h, index_grid.shape[1] // pool_w
    pooling_indices = (
        index_grid.reshape(num_rows, pool_h, num_cols, pool_w)
        .permute(0, 2, 1, 3)
        .reshape(num_rows * num_cols, pool_h * pool_w)
    )
    return pooling_indices, num_rows, num_cols


def resize_and_normalize_image(
    backend: "TorchvisionBackend",
    image: torch.Tensor,
    output_size: list[int],
    resample: PILImageResampling,
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: list[float],
    image_std: list[float],
) -> torch.Tensor:
    # Under `torch.compile`, torchvision's uint8 bilinear resize can produce values slightly outside 0-255,
    # which overflow and wrap around (-1 becomes 255, so a near-black pixel becomes white). Resizing in
    # float, then rounding and clamping back, avoids that and gives the exact same bytes in eager mode.
    is_integer_input = not image.dtype.is_floating_point
    resized = backend.resize(
        image.float() if is_integer_input else image,
        size=SizeDict(height=output_size[0], width=output_size[1]),
        resample=resample,
        antialias=False,
    )
    if is_integer_input:
        resized = resized.round().clamp(0, 255).to(image.dtype)
    return backend.rescale_and_normalize(resized, do_rescale, rescale_factor, do_normalize, image_mean, image_std)


def build_resized_image(
    backend: "TorchvisionBackend",
    images: torch.Tensor,
    crop_size: int,
    resample: PILImageResampling,
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: list[float],
    image_std: list[float],
    image_patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # `images`: a batch of same-shape images `[N, C, H, W]`; resize the whole batch at once.
    resized = resize_and_normalize_image(
        backend,
        images,
        [crop_size, crop_size],
        resample,
        do_rescale=do_rescale,
        rescale_factor=rescale_factor,
        do_normalize=do_normalize,
        image_mean=image_mean,
        image_std=image_std,
    )
    # [N, C, S, S] -> [N, 1, S, S, C]: one global (low-res) view per image.
    resized = resized.permute(0, 2, 3, 1).unsqueeze(1)
    # The per-patch index grid depends only on the (shared) shape, so it is built once.
    patches_per_side = crop_size // image_patch_size
    resized_index_grid = torch.arange(patches_per_side**2, dtype=torch.int32, device=images.device).reshape(
        patches_per_side, patches_per_side
    )
    return resized, resized_index_grid


class Molmo2ImagesKwargs(ImagesKwargs, total=False):
    """
    max_crops (`int`, *optional*, defaults to 8):
        Maximum number of crops to use per image.
    overlap_margins (`list[int]`, *optional*, defaults to `[4, 4]`):
        Overlap margins (in patches) for overlapping crop extraction.
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    pooling_size (`list[int]`, *optional*, defaults to `[2, 2]`):
        The pooling size of the vision adapter.
    """

    max_crops: int
    overlap_margins: list[int]
    patch_size: int
    pooling_size: list[int]


@auto_docstring
class Molmo2ImageProcessor(TorchvisionBackend):
    valid_kwargs = Molmo2ImagesKwargs
    model_input_names = ["pixel_values", "image_token_pooling", "image_grids", "image_num_crops"]
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 378, "width": 378}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    max_crops = 8
    overlap_margins = [4, 4]
    patch_size = 14
    pooling_size = [2, 2]

    def __init__(self, **kwargs: Unpack[Molmo2ImagesKwargs]):
        super().__init__(**kwargs)

    def _build_overlapping_crops(
        self,
        images: torch.Tensor,
        max_crops: int,
        overlap_margins: list[int],
        crop_size: int,
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: list[float],
        image_std: list[float],
        image_patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tile a batch of same-shape images `[N, C, H, W]` into overlapping square crops. The tiling and the per-patch index grid depend only on the (shared) shape, so they are computed once and the resize/unfold are batched over N."""
        left_margin, right_margin = overlap_margins
        crop_patches = crop_size // image_patch_size
        window_patches = crop_patches - (left_margin + right_margin)
        window_size = window_patches * image_patch_size

        num_images, num_channels, original_height, original_width = images.shape
        tiling_h, tiling_w, canvas_height, canvas_width = compute_crop_geometry(
            original_height,
            original_width,
            max_crops=max_crops,
            overlap_margins=overlap_margins,
            crop_size=crop_size,
            patch_size=image_patch_size,
        )

        canvas = resize_and_normalize_image(
            self,
            images,
            [canvas_height, canvas_width],
            resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
        )

        # [N, C, canvas_height, canvas_width] -> unfold spatial dims -> [N, C, tiling_h, tiling_w, crop, crop]
        crops = canvas.unfold(2, crop_size, window_size).unfold(3, crop_size, window_size)
        crops = (
            crops.permute(0, 2, 3, 4, 5, 1)
            .reshape(num_images, tiling_h * tiling_w, crop_size, crop_size, num_channels)
            .contiguous()
        )

        patch_index_grid = torch.arange(
            tiling_h * tiling_w * crop_patches * crop_patches, dtype=torch.int32, device=images.device
        ).reshape(tiling_h, tiling_w, crop_patches, crop_patches)
        if left_margin:
            patch_index_grid[1:, :, :left_margin, :] = -1
            patch_index_grid[:, 1:, :, :left_margin] = -1
        if right_margin:
            patch_index_grid[:-1, :, -right_margin:, :] = -1
            patch_index_grid[:, :-1, :, -right_margin:] = -1

        patch_index_grid = patch_index_grid.permute(0, 2, 1, 3).reshape(-1)
        patch_index_grid = patch_index_grid[patch_index_grid >= 0].reshape(
            canvas_height // image_patch_size, canvas_width // image_patch_size
        )

        return crops, patch_index_grid

    def _image_batch_to_patches_and_grids(
        self,
        images: torch.Tensor,
        max_crops: int,
        overlap_margins: list[int],
        crop_size: int,
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: list[float],
        image_std: list[float],
        image_patch_size: int,
        image_pooling_w: int,
        image_pooling_h: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a batch of same-shape images `[N, C, H, W]`. The grid and pooling indices are
        shape-dependent (shared across the batch) and returned expanded to `[N, ...]` so the caller
        can reorder them per-image alongside the batched patch tensor."""
        patches_per_crop = (crop_size // image_patch_size) ** 2

        crops, patch_index_grid = self._build_overlapping_crops(
            images,
            max_crops=max_crops,
            overlap_margins=overlap_margins,
            crop_size=crop_size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            image_patch_size=image_patch_size,
        )
        pooling_indices, num_patch_rows, num_patch_cols = arange_for_pooling(
            patch_index_grid, image_pooling_h, image_pooling_w
        )

        resized, resized_index_grid = build_resized_image(
            self,
            images,
            crop_size=crop_size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            image_patch_size=image_patch_size,
        )
        # [N, 1, S, S, C] + [N, ncrops, S, S, C] -> [N, 1 + ncrops, S, S, C]
        crops = torch.cat([resized, crops], dim=1)

        resized_pooling_indices, resized_h, resized_w = arange_for_pooling(
            resized_index_grid, image_pooling_h, image_pooling_w
        )

        pooling_indices = torch.where(pooling_indices >= 0, pooling_indices + patches_per_crop, -1)
        pooling_indices = torch.cat([resized_pooling_indices, pooling_indices])
        image_grid = torch.tensor(
            [[resized_h, resized_w, num_patch_rows, num_patch_cols]], dtype=torch.int64, device=images.device
        )

        # [N, total_crops, S, S, C] -> patches [N, total_crops, n_patch, pixels_per_patch]
        num_images, total_crops = crops.shape[0], crops.shape[1]
        pixels_per_patch = image_patch_size * image_patch_size * crops.shape[-1]
        patches = batch_pixels_to_patches(
            crops.reshape(num_images * total_crops, *crops.shape[2:]), image_patch_size
        ).reshape(num_images, total_crops, -1, pixels_per_patch)

        # The grid and pooling indices are shared by all images of this shape; expand them to one row
        # per image so the batch can later be restored to the original image order.
        return image_grid.expand(num_images, -1), patches, pooling_indices.unsqueeze(0).expand(num_images, -1, -1)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[Molmo2ImagesKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: list[float],
        image_std: list[float],
        do_convert_rgb: bool,
        max_crops: int,
        overlap_margins: list[int],
        patch_size: int,
        pooling_size: list[int],
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        if size.height != size.width:
            raise ValueError(f"Molmo2 only supports a square `size`, got height={size.height}, width={size.width}.")
        crop_size = size.height
        image_pooling_h, image_pooling_w = pooling_size

        # Group images by shape and process each unique shape as a single batch (the tiling and all
        # patch/pooling indices are shape-dependent only), then restore the original order.
        device = images[0].device
        grouped_images, grouped_index = group_images_by_shape(images, disable_grouping=disable_grouping)

        grids_grouped: dict = {}
        patches_grouped: dict = {}
        pooled_grouped: dict = {}
        for shape, stacked_images in grouped_images.items():
            image_grid, patches, pooled_indices = self._image_batch_to_patches_and_grids(
                stacked_images,
                max_crops=max_crops,
                overlap_margins=overlap_margins,
                crop_size=crop_size,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                image_patch_size=patch_size,
                image_pooling_w=image_pooling_w,
                image_pooling_h=image_pooling_h,
            )
            grids_grouped[shape] = image_grid
            patches_grouped[shape] = patches
            pooled_grouped[shape] = pooled_indices

        grids = reorder_images(grids_grouped, grouped_index)
        patches = reorder_images(patches_grouped, grouped_index)
        pooled = reorder_images(pooled_grouped, grouped_index)

        all_crops: list[torch.Tensor] = []
        all_pooled: list[torch.Tensor] = []
        patch_offset = 0
        for crops, pooled_indices in zip(patches, pooled):
            all_pooled.append(torch.where(pooled_indices >= 0, pooled_indices + patch_offset, pooled_indices))
            all_crops.append(crops)
            patch_offset += crops.shape[0] * crops.shape[1]

        data = {
            "pixel_values": torch.cat(all_crops, dim=0),
            "image_token_pooling": torch.cat(all_pooled, dim=0),
            "image_grids": torch.stack(grids, dim=0),
            "image_num_crops": torch.tensor([crops.shape[0] for crops in patches], dtype=torch.int64, device=device),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> int:
        image_grid, _ = self.get_image_grid_and_crops(height, width, images_kwargs)
        resized_h, resized_w, high_res_rows, high_res_cols = image_grid
        return resized_h * resized_w + high_res_rows * high_res_cols

    def get_image_grid_and_crops(self, height: int, width: int, images_kwargs=None) -> tuple[list[int], int]:
        """Return the `[low_res_h, low_res_w, high_res_h, high_res_w]` pooled grid and the crop count for one image."""
        if images_kwargs is None:
            images_kwargs = {}
        max_crops = images_kwargs.get("max_crops", self.max_crops)
        overlap_margins = images_kwargs.get("overlap_margins", self.overlap_margins)
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        pooling_size = images_kwargs.get("pooling_size", self.pooling_size)
        size = images_kwargs.get("size", self.size)

        base_h = size["height"] if isinstance(size, dict) else size.height
        base_w = size["width"] if isinstance(size, dict) else size.width
        pooling_h, pooling_w = pooling_size

        tiling_h, tiling_w, canvas_height, canvas_width = compute_crop_geometry(
            height,
            width,
            max_crops=max_crops,
            overlap_margins=overlap_margins,
            crop_size=base_h,
            patch_size=patch_size,
        )
        num_patch_rows_high = math.ceil((canvas_height // patch_size) / pooling_h)
        num_patch_cols_high = math.ceil((canvas_width // patch_size) / pooling_w)

        resized_h = math.ceil((base_h // patch_size) / pooling_h)
        resized_w = math.ceil((base_w // patch_size) / pooling_w)

        return [resized_h, resized_w, num_patch_rows_high, num_patch_cols_high], tiling_h * tiling_w + 1


class Molmo2VideosKwargs(VideosKwargs, total=False):
    """
    patch_size (`int`, *optional*):
        Side length in pixels of each ViT patch for video frames.
    pooling_size (`list[int]`, *optional*):
        `[pool_h, pool_w]` pooling window applied to video patch features.
    max_fps (`int`, *optional*):
        Maximum sampling rate in frames per second for short videos.
    """

    patch_size: int
    pooling_size: list[int]
    max_fps: int


@auto_docstring
class Molmo2VideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR
    size = {"height": 378, "width": 378}
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    patch_size = 14
    pooling_size = [3, 3]
    num_frames = 64
    do_sample_frames = True
    max_fps = 2
    valid_kwargs = Molmo2VideosKwargs
    model_input_names = ["pixel_values_videos", "video_token_pooling", "video_grids"]

    def __init__(self, **kwargs: Unpack[Molmo2VideosKwargs]):
        super().__init__(**kwargs)
        if self.size is not None and (self.size.get("height", None) is None or self.size.get("width", None) is None):
            raise ValueError("size must contain 'height' and 'width' keys.")

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: int | float | None = None,
        max_fps: int | float | None = None,
        **kwargs,
    ):
        if fps is not None and num_frames is not None:
            raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

        max_fps = max_fps if max_fps is not None else self.max_fps

        if metadata.fps is None:
            metadata.fps = fps if fps is not None else max_fps
            logger.warning_once(
                "Molmo2 inserts frame timestamps into video prompts, but the input video's `fps` was not provided "
                f"or could not be inferred. Defaulting to `fps={metadata.fps}`. Please provide `video_metadata` "
                "for more accurate timestamps."
            )
        if metadata.duration is None:
            metadata.duration = metadata.total_num_frames / metadata.fps

        if fps is not None:
            target_num_frames = int(metadata.duration * fps)
        else:
            target_num_frames = num_frames if num_frames is not None else self.num_frames
            if max_fps is not None and metadata.fps > max_fps:
                target_num_frames = min(target_num_frames, int(metadata.duration * max_fps))

        target_num_frames = max(min(target_num_frames, metadata.total_num_frames), 1)
        total = metadata.total_num_frames
        return torch.arange(0, total, total / target_num_frames).int()

    def _build_video_patches(
        self,
        video: torch.Tensor,
        crop_size: int,
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: list[float],
        image_std: list[float],
        image_patch_size: int,
        image_pooling_h: int,
        image_pooling_w: int,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        # `build_resized_image` takes a whole `[N, C, H, W]` batch, so all frames of a video are resized in one call.
        resized_frames, resized_index_grid = build_resized_image(
            self,
            video,
            crop_size=crop_size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            image_patch_size=image_patch_size,
        )
        # The pooling index grid is shape-dependent only, hence shared by every frame.
        pooling_indices, num_patch_rows, num_patch_cols = arange_for_pooling(
            resized_index_grid, image_pooling_h, image_pooling_w
        )
        # [T, 1, S, S, C] -> [T, n_patch, pixels_per_patch]
        patches = batch_pixels_to_patches(resized_frames.squeeze(1), image_patch_size)
        return [num_patch_rows, num_patch_cols], patches, pooling_indices

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        size: SizeDict,
        resample: PILImageResampling | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        pooling_size: list[int],
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        if size.height != size.width:
            raise ValueError(f"Molmo2 only supports a square `size`, got height={size.height}, width={size.width}.")
        crop_size = size.height
        image_pooling_h, image_pooling_w = pooling_size

        all_crops: list[torch.Tensor] = []
        all_pooled: list[torch.Tensor] = []
        all_grids: list[torch.Tensor] = []
        patch_offset = 0

        for video in videos:
            image_grid, patches, pooling_indices = self._build_video_patches(
                video,
                crop_size=crop_size,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                image_patch_size=patch_size,
                image_pooling_h=image_pooling_h,
                image_pooling_w=image_pooling_w,
            )
            num_frames, patches_per_frame = patches.shape[:2]
            frame_offsets = (
                patch_offset + torch.arange(num_frames, device=pooling_indices.device) * patches_per_frame
            ).view(-1, 1, 1)
            pooled_indices = torch.where(pooling_indices >= 0, pooling_indices + frame_offsets, pooling_indices)
            all_pooled.append(pooled_indices.reshape(-1, pooling_indices.shape[-1]))
            all_crops.append(patches)
            patch_offset += num_frames * patches_per_frame

            all_grids.append(torch.tensor([num_frames, image_grid[0], image_grid[1]], dtype=torch.int64))

        data = {
            "pixel_values_videos": torch.cat(all_crops, dim=0),
            "video_token_pooling": torch.cat(all_pooled, dim=0),
            "video_grids": torch.stack(all_grids, dim=0),
        }
        return BatchFeature(data, tensor_type=return_tensors)


class Molmo2ProcessorKwargs(ProcessingKwargs, total=False):
    """Molmo2 processor kwargs"""

    images_kwargs: Molmo2ImagesKwargs
    videos_kwargs: Molmo2VideosKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": True,
        },
        "videos_kwargs": {"return_metadata": True},
    }


@auto_docstring
class Molmo2Processor(ProcessorMixin):
    valid_processor_kwargs = Molmo2ProcessorKwargs
    image_token = "<|image|>"
    video_token = "<|video|>"

    @property
    def model_input_names(self):
        return super().model_input_names + ["mm_token_type_ids"]

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        tokenizer=None,
        chat_template: str | None = None,
        image_use_col_tokens: bool | None = True,
        use_single_crop_col_tokens: bool | None = None,
        use_single_crop_start_token: bool | None = True,
        video_use_col_tokens: bool | None = False,
        use_frame_special_tokens: bool | None = True,
        **kwargs,
    ) -> None:
        r"""
        image_use_col_tokens (`bool`, *optional*, defaults to `True`):
            Whether to append column-separator tokens (`<im_col>`) after each patch row of the high-resolution image
            view.
        use_single_crop_col_tokens (`bool`, *optional*):
            Whether to append column-separator tokens after each patch row of the low-resolution (single-crop) image
            view. If `None`, falls back to `image_use_col_tokens`.
        use_single_crop_start_token (`bool`, *optional*, defaults to `True`):
            Whether to start the low-resolution image view with `<low_res_im_start>` instead of the regular
            `<im_start>`.
        video_use_col_tokens (`bool`, *optional*, defaults to `False`):
            Whether to append column-separator tokens after each patch row of video frames.
        use_frame_special_tokens (`bool`, *optional*, defaults to `True`):
            Whether to wrap each video frame with `<frame_start>` / `<frame_end>` tokens. If `False`, falls back to
            `<im_start>` / `<im_end>`.
        """
        self.image_use_col_tokens = image_use_col_tokens
        self.use_single_crop_col_tokens = use_single_crop_col_tokens
        self.use_single_crop_start_token = use_single_crop_start_token
        self.video_use_col_tokens = video_use_col_tokens
        self.use_frame_special_tokens = use_frame_special_tokens
        self.image_token = getattr(tokenizer, "image_token", self.image_token)
        self.video_token = getattr(tokenizer, "video_token", self.video_token)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.image_token_ids = tokenizer.convert_tokens_to_ids(
            [
                "<im_patch>",
                "<im_col>",
                "<im_start>",
                "<low_res_im_start>",
                "<frame_start>",
                "<im_end>",
                "<frame_end>",
                "<im_low>",
            ]
        )
        super().__init__(image_processor, video_processor, tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[Molmo2ProcessorKwargs],
    ):
        # TODO(molbap): remove once the allenai chat templates emit `bos_token`
        # (hub PRs allenai/Molmo2-8B#11, allenai/Molmo2-4B#4, allenai/Molmo2-O-7B#2)
        bos_token = self.tokenizer.bos_token or self.tokenizer.eos_token
        if isinstance(text, str):
            text = [text]
        if text is not None:
            text = [prompt if prompt.startswith(bos_token) else bos_token + prompt for prompt in text]
        return super().__call__(images=images, text=text, videos=videos, **kwargs)

    def get_video_string(self, video_grid, timestamps) -> str:
        if hasattr(video_grid, "tolist"):
            video_grid = video_grid.tolist()
        start_token = "<frame_start>" if self.use_frame_special_tokens else "<im_start>"
        end_token = "<frame_end>" if self.use_frame_special_tokens else "<im_end>"

        num_frames, num_patch_rows, num_patch_cols = video_grid
        video_string = ""
        for frame_idx, frame_time in enumerate(timestamps):
            prev_space = " " if frame_idx > 0 else ""
            video_string += prev_space + f"{frame_time:.1f} "
            tokens_per_row = ["<im_patch>"] * num_patch_cols
            if self.video_use_col_tokens:
                tokens_per_row = tokens_per_row + ["<im_col>"]
            video_string += "".join([start_token] + tokens_per_row * num_patch_rows + [end_token])

        return video_string

    def validate_inputs(
        self,
        images=None,
        text=None,
        videos=None,
        audio=None,
        **kwargs,
    ):
        super().validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)
        if videos is not None and text is not None:
            for sample in text:
                if sample.count(self.video_token) > 1:
                    raise ValueError("At most one video is supported per sample.")

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        image_grid = image_inputs["image_grids"][image_idx]
        if hasattr(image_grid, "tolist"):
            image_grid = image_grid.tolist()
        resized_h, resized_w, height, width = image_grid

        tokens_per_row = ["<im_patch>"] * width
        if self.image_use_col_tokens:
            tokens_per_row = tokens_per_row + ["<im_col>"]
        high_res_tokens = ["<im_start>"] + tokens_per_row * height + ["<im_end>"]

        tokens_per_row = ["<im_patch>"] * resized_w
        use_single_crop_col_tokens = (
            self.image_use_col_tokens if self.use_single_crop_col_tokens is None else self.use_single_crop_col_tokens
        )
        image_start_token = "<low_res_im_start>" if self.use_single_crop_start_token else "<im_start>"
        if use_single_crop_col_tokens:
            tokens_per_row = tokens_per_row + ["<im_col>"]
        low_res_tokens = [image_start_token] + tokens_per_row * resized_h + ["<im_end>"]

        return "".join(low_res_tokens + high_res_tokens)

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
            images_kwargs = Molmo2ProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)

            num_image_tokens, num_image_patches = [], []
            for image_size in image_sizes:
                image_grid, num_crops = self.image_processor.get_image_grid_and_crops(*image_size, images_kwargs)
                image_string = self.replace_image_token({"image_grids": [image_grid]}, image_idx=0)
                num_image_tokens.append(len(self.tokenizer.tokenize(image_string)))
                num_image_patches.append(num_crops)
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        video_grid = video_inputs["video_grids"][video_idx]
        video_metadata = video_inputs.get("video_metadata", [])
        metadata = video_metadata[video_idx] if video_idx < len(video_metadata) else None

        frames_indices = getattr(metadata, "frames_indices", None)
        if frames_indices is None:
            frames_indices = range(int(video_grid[0].item()))
        fps = getattr(metadata, "fps", None)
        if fps is None:
            fps = self.video_processor.max_fps
            logger.warning_once(
                "Molmo2 inserts frame timestamps into video prompts, but the input video's `fps` was not "
                f"provided or could not be inferred. Defaulting to `fps={fps}`. Please provide "
                "`video_metadata` for more accurate timestamps."
            )
        timestamps = [frame_idx / fps for frame_idx in frames_indices]
        return self.get_video_string(video_grid, timestamps)


class Molmo2CausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


class Molmo2ModelOutputWithPast(LlavaModelOutputWithPast):
    pass


class Molmo2RotaryEmbedding(LlamaRotaryEmbedding):
    """RoPE from the flat `rope_parameters`; `scaled=False` keeps the theta but skips the scaling, for the
    layers outside `rope_scaling_layers`."""

    def __init__(self, config: Molmo2TextConfig, device=None, scaled: bool = True):
        super().__init__(config, device)
        if not scaled:
            # `rope_type = "default"` also keeps the generic `_init_weights` rotary re-init on the unscaled table.
            self.rope_type = "default"
            inv_freq, self.attention_scaling = self.compute_default_rope_parameters(config, device=device)
            self.inv_freq = inv_freq
            self.original_inv_freq = inv_freq.clone()


class Molmo2RMSNorm(LlamaRMSNorm):
    pass


class Molmo2Attention(Olmo2Attention):
    """Olmo2 attention whose q/k RMSNorm runs either over the full projection (`olmo`) or per head (`qwen3`)."""

    def __init__(self, config: Molmo2TextConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.qk_norm_type = config.qk_norm_type
        if self.qk_norm_type == "qwen3":
            self.q_norm = Molmo2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Molmo2RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.qk_norm_type == "olmo":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        if self.qk_norm_type == "qwen3":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Molmo2MLP(OlmoMLP):
    pass


class Molmo2DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: Molmo2TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.norm_after = config.norm_after
        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # `norm_after=True` normalizes each sublayer's output (post-norm) instead of its input (pre-norm)
        residual = hidden_states
        if not self.norm_after:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        if self.norm_after:
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        # Fully Connected
        residual = hidden_states
        if not self.norm_after:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.norm_after:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states


class Molmo2PreTrainedModel(LlamaPreTrainedModel):
    config: Molmo2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Molmo2DecoderLayer",
        "Molmo2VisionEncoderLayer",
        "Molmo2VisionAttention",
    ]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Molmo2DecoderLayer,
        "attentions": Molmo2Attention,
    }

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Molmo2VisionModel):
            init.normal_(module.positional_embedding, mean=0.0, std=self.config.initializer_range)


class Molmo2VisionMLP(Siglip2MLP):
    pass


class Molmo2VisionAttention(nn.Module):
    def __init__(self, config: Molmo2VisionConfig | Molmo2AdapterConfig, hidden_size: int | None = None):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size or config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        key_value_shape = (*encoder_hidden_states.shape[:-1], -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(encoder_hidden_states).view(key_value_shape).transpose(1, 2)
        value_states = self.v_proj(encoder_hidden_states).view(key_value_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Molmo2VisionEncoderLayer(Siglip2EncoderLayer):
    def __init__(self, config: Molmo2VisionConfig):
        super().__init__(config)
        self.self_attn = Molmo2VisionAttention(config)
        self.mlp = Molmo2VisionMLP(config)


@auto_docstring
class Molmo2VisionModel(Molmo2PreTrainedModel):
    config_class = Molmo2VisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _no_split_modules = ["Molmo2VisionEncoderLayer"]
    _can_record_outputs = {
        "hidden_states": Molmo2VisionEncoderLayer,
        "attentions": Molmo2VisionAttention,
    }

    def __init__(self, config: Molmo2VisionConfig):
        super().__init__(config)
        self.positional_embedding = nn.Parameter(torch.zeros(config.num_position_embeddings, config.hidden_size))
        self.patch_embedding = nn.Linear(
            config.patch_size * config.patch_size * config.num_channels, config.hidden_size
        )
        # trf-ignore: TRF034 (false positive: Siglip2EncoderLayer subclasses GradientCheckpointingLayer)
        self.layers = nn.ModuleList([Molmo2VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(self, pixel_values: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> BaseModelOutputWithPooling:
        hidden_states = self.patch_embedding(pixel_values.to(dtype=self.dtype))
        # patch count == num_position_embeddings, locked by config; only retraining with a different grid breaks this.
        hidden_states = hidden_states + self.positional_embedding[None, :, :].to(hidden_states.dtype)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, None, **kwargs)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=hidden_states.mean(dim=1),
        )


class Molmo2ImageProjectorMLP(LlamaMLP):
    def __init__(self, config: Molmo2AdapterConfig):
        nn.Module.__init__(self)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.text_hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]


@auto_docstring(
    custom_intro="""
    The Molmo2 vision adapter: pools ViT patch features with a cross-attention layer and projects them into the
    language model's embedding space.
    """
)
class Molmo2Adapter(Molmo2PreTrainedModel):
    config_class = Molmo2AdapterConfig
    input_modalities = ("image",)
    _no_split_modules = ["Molmo2VisionAttention"]

    def __init__(self, config: Molmo2AdapterConfig):
        super().__init__(config)
        pooling_input_dim = config.hidden_size * len(config.vision_feature_layer)
        self.image_pooling_2d = Molmo2VisionAttention(config, hidden_size=pooling_input_dim)
        self.image_projector = Molmo2ImageProjectorMLP(config)
        self.image_feature_dropout = nn.Dropout(config.image_feature_dropout)
        self.post_init()

    @merge_with_config_defaults
    @auto_docstring
    def forward(self, image_features: torch.Tensor, pooled_patches_idx: torch.Tensor, **kwargs) -> BaseModelOutput:
        r"""
        image_features (`torch.Tensor` of shape `(num_crops, num_patches, hidden_size * len(vision_feature_layer))`):
            Concatenated intermediate ViT features of every crop.
        pooled_patches_idx (`torch.Tensor` of shape `(num_tokens, pool_h * pool_w)`):
            Indices into the flattened patch sequence pooled by each output token; `-1` marks padding slots.
        """
        image_features = self.image_feature_dropout(image_features)
        flat_features = image_features.reshape(-1, image_features.shape[-1])

        valid_mask = pooled_patches_idx >= 0
        valid_token_mask = torch.any(valid_mask, -1)

        patches_to_pool = flat_features[torch.clip(pooled_patches_idx, 0)]
        patches_to_pool = patches_to_pool * valid_mask.to(patches_to_pool.dtype)[..., None]

        num_valid_patches = valid_mask.float().sum(-1)
        num_valid_patches = torch.where(num_valid_patches == 0, 1, num_valid_patches)
        query = patches_to_pool.sum(-2, keepdim=True) / num_valid_patches[:, None, None].to(patches_to_pool.dtype)

        attention_mask = create_bidirectional_mask(
            config=self.config, inputs_embeds=query, attention_mask=valid_mask, encoder_hidden_states=patches_to_pool
        )
        pooled_features, _ = self.image_pooling_2d(query, patches_to_pool, attention_mask=attention_mask)
        pooled_features = pooled_features.squeeze(1)
        pooled_features = self.image_projector(pooled_features)
        return BaseModelOutput(last_hidden_state=pooled_features[valid_token_mask])


class Molmo2TextModel(LlamaModel):
    config: Molmo2TextConfig

    def __init__(self, config: Molmo2TextConfig):
        Molmo2PreTrainedModel.__init__(self, config)
        self.vocab_size = config.vocab_size
        # The checkpoint's extra-vocabulary table is concatenated onto the base one at load time
        # (see `conversion_mapping.py`), so the embedding covers `vocab_size + additional_vocab_size`.
        self.embed_tokens = nn.Embedding(config.vocab_size + (config.additional_vocab_size or 0), config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        # trf-ignore: TRF034 (false positive: LlamaDecoderLayer subclasses GradientCheckpointingLayer)
        self.layers = nn.ModuleList(
            [Molmo2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Molmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Molmo2RotaryEmbedding(config)
        self.rotary_emb_unscaled = Molmo2RotaryEmbedding(config, scaled=False)
        # CODEPATH: O-7B lists 24 of its 32 layers in `rope_scaling_layers` (YaRN there, plain RoPE elsewhere);
        # 4B/8B scale every layer.
        self.rope_types = [
            "scaled" if layer_idx in config.rope_scaling_layers else "unscaled"
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

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
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = self.embedding_dropout(inputs_embeds)

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
        position_embeddings = {
            "scaled": self.rotary_emb(hidden_states, position_ids=position_ids),
            "unscaled": self.rotary_emb_unscaled(hidden_states, position_ids=position_ids),
        }

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings[self.rope_types[layer_idx]],
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring(
    custom_intro="""
    The Molmo2 model which consists of a vision backbone, a pooling adapter and a language model, without a language
    modeling head.
    """
)
class Molmo2Model(LlavaModel):
    config: Molmo2Config

    def __init__(self, config: Molmo2Config):
        Molmo2PreTrainedModel.__init__(self, config)
        self.vision_tower = Molmo2VisionModel(config.vision_config)
        self.multi_modal_projector = Molmo2Adapter(config.adapter_config)
        self.language_model = Molmo2TextModel(config.text_config)
        self.post_init()

    @can_return_tuple
    @filter_output_hidden_states
    @auto_docstring(
        custom_intro="Obtains pooled image features from the vision tower and the adapter: the `pooler_output` is a "
        "tuple with one `(num_image_tokens, hidden_size)` tensor per image."
    )
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_token_pooling: torch.Tensor,
        image_grids: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        image_shape = None
        if pixel_values.dim() == 4:
            image_shape = pixel_values.shape[:3]
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-2], pixel_values.shape[-1])

        image_outputs: BaseModelOutputWithPooling = self.vision_tower(pixel_values, **kwargs)
        # `vision_feature_layer` indices are normalized to non-negative in `Molmo2Config.__post_init__`
        image_features = torch.cat(
            [image_outputs.hidden_states[layer + 1] for layer in self.config.adapter_config.vision_feature_layer],
            dim=-1,
        )

        if image_shape is not None:
            image_features = image_features.reshape(*image_shape, -1)
        pooled_features = self.multi_modal_projector(image_features, image_token_pooling).last_hidden_state
        split_sizes = (image_grids[:, 0] * image_grids[:, 1] + image_grids[:, 2] * image_grids[:, 3]).tolist()
        image_outputs.pooler_output = torch.split(pooled_features, split_sizes)
        return image_outputs

    @can_return_tuple
    @auto_docstring(
        custom_intro="Obtains pooled video features from the vision tower and the adapter: the `pooler_output` is a "
        "tuple with one `(num_video_tokens, hidden_size)` tensor per video."
    )
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_token_pooling: torch.Tensor,
        video_grids: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        # Video frames only go through the low-res path: `num_frames` frames of `rows x cols` tokens, no high-res crops.
        num_frames, rows, cols = video_grids.unbind(-1)
        image_grids = torch.stack([num_frames * rows, cols, torch.zeros_like(cols), torch.zeros_like(cols)], dim=-1)
        return self.get_image_features(pixel_values_videos, video_token_pooling, image_grids, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Molmo2ModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if pixel_values is not None and pixel_values_videos is not None:
            raise ValueError("pixel_values and pixel_values_videos are provided at the same time")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features: torch.FloatTensor | None = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_token_pooling, image_grids).pooler_output
        elif pixel_values_videos is not None:
            image_features = self.get_video_features(
                pixel_values_videos, video_token_pooling, video_grids
            ).pooler_output

        if image_features is not None:
            image_features = torch.cat(image_features, dim=0)
            # `get_placeholder_mask` returns a [batch, seq, 1] mask; Molmo2 *adds* the image features onto
            # the placeholder-token embeddings (residual), which means we index `inputs_embeds` directly.
            # Boolean indexing does not broadcast, so the mask must be expanded to the hidden dim first.
            special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            special_image_mask = special_image_mask.expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask,
                inputs_embeds[special_image_mask] + image_features.reshape(-1),
            )

        if self.training and mm_token_type_ids is None:
            raise ValueError("`mm_token_type_ids` is required as a model input when training")

        # An already prepared 4D mask (e.g. from `generate`) is returned as is by `create_causal_mask`
        mask_kwargs = {
            "config": self.config.get_text_config(),
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        is_prefill = past_key_values is None or not past_key_values.is_initialized or image_features is not None
        if mm_token_type_ids is not None and is_prefill:
            # every image or frame token attends to every other one, so they all share a single block
            mask_kwargs["block_sequence_ids"] = torch.where(mm_token_type_ids.to(inputs_embeds.device) == 1, 0, -1)
        causal_mask = create_causal_mask(**mask_kwargs)

        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return Molmo2ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )


@auto_docstring(
    custom_intro="""
    The Molmo2 model which consists of a vision backbone, a pooling adapter and a language model.
    """
)
class Molmo2ForConditionalGeneration(Molmo2PreTrainedModel, GenerationMixin):
    # the loss must not be normalized by `num_items_in_batch`: logits/labels are filtered before computing it
    accepts_loss_kwargs = False
    config: Molmo2Config

    def __init__(self, config: Molmo2Config):
        super().__init__(config)

        self.model = Molmo2Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.vocab_size = config.text_config.vocab_size

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.Tensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Molmo2CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Molmo2ForConditionalGeneration

        >>> model = Molmo2ForConditionalGeneration.from_pretrained("allenai/Molmo2-8B")
        >>> processor = AutoProcessor.from_pretrained("allenai/Molmo2-8B")

        >>> prompt = "What's the content of the image?"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]}]

        >>> inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=15)
        >>> generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
        >>> processor.post_process_image_text_to_text(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a bustling street scene in what appears to be a Chinatown area. There's ..."
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs)

        return Molmo2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ):
        visual_keys = (
            "pixel_values",
            "image_token_pooling",
            "image_grids",
            "image_num_crops",
            "pixel_values_videos",
            "video_token_pooling",
            "video_grids",
        )
        visual = {k: model_kwargs.pop(k) for k in visual_keys if k in model_kwargs}
        original_input_ids = input_ids
        input_ids, model_kwargs = super()._expand_inputs_for_generation(
            expand_size=expand_size,
            is_encoder_decoder=is_encoder_decoder,
            input_ids=input_ids,
            **model_kwargs,
        )
        if expand_size != 1 and original_input_ids is not None:
            # image and video patches share `image_token_id` in `input_ids`, so the per-sample count
            # covers whichever pooling tensor is present
            patch_counts = (original_input_ids == self.config.image_token_id).sum(dim=-1).tolist()
            for pooling_key in ("image_token_pooling", "video_token_pooling"):
                if visual.get(pooling_key) is not None:
                    chunks = visual[pooling_key].split(patch_counts)
                    visual[pooling_key] = torch.cat([chunk for chunk in chunks for _ in range(expand_size)], dim=0)
            for grid_key in ("image_grids", "video_grids"):
                if visual.get(grid_key) is not None:
                    visual[grid_key] = visual[grid_key].repeat_interleave(expand_size, dim=0)
        model_kwargs.update(visual)
        return input_ids, model_kwargs

    @staticmethod
    def create_masks_for_generate(
        config: PreTrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        mask_kwargs = {
            "config": config.get_text_config(),
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        if mm_token_type_ids is not None and inputs_embeds.shape[1] != 1:
            mask_kwargs["block_sequence_ids"] = torch.where(mm_token_type_ids.to(inputs_embeds.device) == 1, 0, -1)

        return create_masks_for_generate(**mask_kwargs)


__all__ = [
    "Molmo2AdapterConfig",
    "Molmo2Config",
    "Molmo2TextConfig",
    "Molmo2VisionConfig",
    "Molmo2Adapter",
    "Molmo2ForConditionalGeneration",
    "Molmo2ImageProcessor",
    "Molmo2Model",
    "Molmo2PreTrainedModel",
    "Molmo2Processor",
    "Molmo2TextModel",
    "Molmo2VideoProcessor",
    "Molmo2VisionModel",
]
