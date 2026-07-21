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

import math
from collections.abc import Callable
from dataclasses import field

import torch
from huggingface_hub.dataclasses import strict
from torch import nn
from torch.nn import functional as F

from ... import initialization as init
from ...activations import ACT2FN
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
from ...masking_utils import create_causal_mask, create_masks_for_generate
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
    torch_compilable_check,
)
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import VideoMetadata
from ..gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from ..llama.modeling_llama import (
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaModelOutputWithPast,
)
from ..olmo2.modeling_olmo2 import Olmo2Attention
from ..phi3.modeling_phi3 import (
    Phi3DecoderLayer,
    Phi3MLP,
)
from ..siglip2.modeling_siglip2 import (
    Siglip2Encoder,
    Siglip2EncoderLayer,
    Siglip2MLP,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="allenai/Molmo2-8B")
@strict
class Molmo2VisionConfig(PreTrainedConfig):
    r"""
    image_default_input_size (`list[int]`, *optional*, defaults to `[378, 378]`):
        Default input image size (height, width).
    image_patch_size (`int`, *optional*, defaults to 14):
        Size of each image patch.
    image_num_pos (`int`, *optional*, defaults to 577):
        Number of positional embeddings for the image.
    """

    model_type = "molmo2"
    base_config_key = "vision_config"

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    image_default_input_size: list[int] = field(default_factory=lambda: [378, 378])
    image_patch_size: int = 14
    image_num_pos: int = 577
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    initializer_range: float = 0.02


@auto_docstring(checkpoint="allenai/Molmo2-8B")
@strict
class Molmo2AdapterConfig(PreTrainedConfig):
    r"""
    vit_layers (`list[int]`, *optional*, defaults to `[-3, -9]`):
        Indices of ViT layers to extract features from.
    text_hidden_size (`int`, *optional*, defaults to 3584):
        Hidden size of the text model (used for projection).
    image_feature_dropout (`float`, *optional*, defaults to 0.0):
        Dropout rate for image features.
    """

    model_type = "molmo2"
    base_config_key = "adapter_config"

    vit_layers: list[int] = field(default_factory=lambda: [-3, -9])
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


@auto_docstring(checkpoint="allenai/Molmo2-8B")
@strict
class Molmo2TextConfig(PreTrainedConfig):
    r"""
    layer_types (`list[str]`, *optional*):
        List of layer types to use for the model, `"full_attention"` for every layer when not provided.
    rope_layer_types (`list[str]`, *optional*):
        Per-layer RoPE type keying into a nested `rope_parameters`. Built from the checkpoint's `rope_scaling_layers`
        when not provided (`"scaling"` for the scaled layers, `"default"` otherwise).
    additional_vocab_size (`int`, *optional*, defaults to 128):
        Number of additional vocabulary tokens beyond the base vocabulary.
    qkv_bias (`bool`, *optional*, defaults to `True`):
        Whether to use bias in query, key, and value projections.
    use_qk_norm (`bool`, *optional*, defaults to `True`):
        Serialized compatibility flag for checkpoints that use query/key normalization.
    qk_norm_type (`str`, *optional*, defaults to `"qwen3"`):
        Query/key normalization layout used by the checkpoint. `"qwen3"` normalizes per head; `"olmo"` normalizes the
        full projected query/key tensors.
    embedding_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the embedding layer.
    residual_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio applied after residual connections.
    rope_parameters (`RopeParameters`, *optional*):
        RoPE parameters for the model.
    norm_after (`bool`, *optional*, defaults to `False`):
        Whether to apply layer normalization after the attention/FFN blocks instead of before.
    """

    model_type = "molmo2_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.att_proj": "colwise",
        "layers.*.self_attn.attn_out": "rowwise",
        "layers.*.mlp.ff_proj": "colwise",
        "layers.*.mlp.ff_out": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    hidden_size: int = 3584
    num_attention_heads: int = 28
    num_key_value_heads: int | None = 4
    head_dim: int = 128
    vocab_size: int = 152064
    additional_vocab_size: int = 128
    qkv_bias: bool = True
    use_qk_norm: bool = True
    qk_norm_type: str = "qwen3"
    num_hidden_layers: int = 48
    intermediate_size: int = 18944
    hidden_act: str = "silu"
    embedding_dropout: float = 0.0
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    max_position_embeddings: int = 4096
    rope_parameters: RopeParameters | dict | None = None
    layer_types: list[str] | None = None
    rope_layer_types: list[str] | None = None
    layer_norm_eps: float = 1e-6
    norm_after: bool = False
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        super().validate_architecture()
        if not self.use_qk_norm:
            raise ValueError(
                "Molmo2 requires `use_qk_norm=True`; all shipped checkpoints apply Q/K norm and "
                "no q_norm/k_norm-less path is provided."
            )
        if self.qk_norm_type not in ("qwen3", "olmo"):
            raise ValueError(f"Unsupported `qk_norm_type`: {self.qk_norm_type}")

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        rope_scaling_layers = kwargs.pop("rope_scaling_layers", None)
        rope_theta = kwargs.pop("rope_theta", None)
        if self.rope_layer_types is None:
            self.rope_layer_types = [
                "scaling" if rope_scaling_layers is not None and layer_idx in rope_scaling_layers else "default"
                for layer_idx in range(self.num_hidden_layers)
            ]
        rope_parameters = rope_scaling or self.rope_parameters or {}
        if rope_parameters and set(rope_parameters).issubset(self.rope_layer_types):
            self.rope_parameters = rope_parameters
        else:
            scaling_parameters = dict(rope_parameters)
            scaling_parameters.setdefault("rope_type", "default")
            if rope_theta is not None:
                scaling_parameters.setdefault("rope_theta", rope_theta)
            default_parameters = {"rope_type": "default"}
            if "rope_theta" in scaling_parameters:
                default_parameters["rope_theta"] = scaling_parameters["rope_theta"]
            self.rope_parameters = {
                layer_type: scaling_parameters if layer_type == "scaling" else default_parameters
                for layer_type in set(self.rope_layer_types)
            }
        self.standardize_rope_params()
        return kwargs

    def standardize_rope_params(self):
        for rope_parameters in (self.rope_parameters or {}).values():
            rope_parameters.setdefault("rope_type", "default")
            if rope_parameters["rope_type"] in ("llama3", "yarn", "longrope"):
                rope_parameters.setdefault("original_max_position_embeddings", self.max_position_embeddings)

    def validate_rope(self):
        for rope_parameters in (self.rope_parameters or {}).values():
            rope_type = rope_parameters["rope_type"]
            validation_fn = getattr(self, f"_validate_{rope_type}_rope_parameters", None)
            if validation_fn is not None:
                validation_fn(rope_parameters, ignore_keys=self.ignore_keys_at_rope_validation)


@auto_docstring(checkpoint="allenai/Molmo2-8B")
@strict
class Molmo2Config(PreTrainedConfig):
    r"""
    vision_config (`Molmo2VisionConfig`, *optional*):
        Configuration for the vision transformer backbone.
    adapter_config (`Molmo2AdapterConfig`, *optional*):
        Configuration for the vision-to-language adapter.
    image_start_token_id (`int`, *optional*):
        Token ID marking the start of an image region.
    low_res_image_start_token_id (`int`, *optional*):
        Token ID marking the start of a low-resolution image crop.
    image_end_token_id (`int`, *optional*):
        Token ID marking the end of an image region.
    image_low_res_id (`int`, *optional*):
        Token ID for low-resolution image patches.
    image_patch_id (`int`, *optional*):
        Token ID for image patches.
    image_col_id (`int`, *optional*):
        Token ID for column separators in image patch sequences.
    frame_start_token_id (`int`, *optional*):
        Token ID marking the start of a video frame.
    frame_end_token_id (`int`, *optional*):
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
    image_start_token_id: int | None = None
    low_res_image_start_token_id: int | None = None
    image_end_token_id: int | None = None
    image_low_res_id: int | None = None
    image_patch_id: int | None = None
    image_col_id: int | None = None
    frame_start_token_id: int | None = None
    frame_end_token_id: int | None = None
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

        # Normalize negative `vit_layers` indices and trim the ViT to the deepest layer the adapter actually reads.
        num_vit_layers = self.vision_config.num_hidden_layers
        self.adapter_config.vit_layers = [
            layer if layer >= 0 else layer + num_vit_layers for layer in self.adapter_config.vit_layers
        ]
        last_layer_needed = max(self.adapter_config.vit_layers) + 1
        if last_layer_needed < num_vit_layers:
            self.vision_config.num_hidden_layers = last_layer_needed

        super().__post_init__(**kwargs)


def select_tiling(height: int, width: int, patch_size: int, max_num_crops: int) -> tuple[int, int]:
    """Select the image tiling in the same height/width order as the original Molmo2 processor."""
    tilings = []
    for tile_height in range(1, max_num_crops + 1):
        for tile_width in range(1, max_num_crops + 1):
            if tile_height * tile_width <= max_num_crops:
                tilings.append((tile_height, tile_width))
    tilings.sort(key=lambda x: (x[0] * x[1], x[0]))

    candidate_resolutions = torch.tensor(tilings, dtype=torch.int32) * patch_size
    original_size = torch.tensor([height, width], dtype=torch.float32)

    required_scales = candidate_resolutions.to(torch.float32) / original_size
    required_scale = required_scales.amin(dim=-1, keepdim=True)

    if torch.all(required_scale < 1):
        return tilings[int(required_scale.argmax())]

    required_scale = torch.where(required_scale < 1.0, 10e9, required_scale)
    return tilings[int(required_scale.argmin())]


def batch_pixels_to_patches(array: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Reshape images of [n_images, h, w, 3] -> [n_images, n_patches, pixels_per_patch]"""
    if len(array.shape) == 3:
        n_crops, height, width = array.shape
        h_patches = height // patch_size
        w_patches = width // patch_size
        array = array.reshape(n_crops, h_patches, patch_size, w_patches, patch_size)
        array = array.permute(0, 1, 3, 2, 4)
        array = array.reshape(n_crops, h_patches * w_patches, patch_size * patch_size)
        return array
    else:
        n_crops, height, width, channels = array.shape
        h_patches = height // patch_size
        w_patches = width // patch_size
        array = array.reshape(n_crops, h_patches, patch_size, w_patches, patch_size, channels)
        array = array.permute(0, 1, 3, 2, 4, 5)
        array = array.reshape(n_crops, h_patches * w_patches, patch_size * patch_size * channels)
        return array


def arange_for_pooling(
    idx_arr: torch.Tensor,
    pool_h: int,
    pool_w: int,
) -> torch.Tensor:
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = F.pad(
        idx_arr,
        (w_pad // 2, (w_pad + 1) // 2, h_pad // 2, (h_pad + 1) // 2),
        mode="constant",
        value=-1,
    )
    num_rows, num_cols = idx_arr.shape[0] // pool_h, idx_arr.shape[1] // pool_w
    return (
        idx_arr.reshape(num_rows, pool_h, num_cols, pool_w)
        .permute(0, 2, 1, 3)
        .reshape(num_rows, num_cols, pool_h * pool_w)
    )


def resize_and_normalize_image(
    backend: "TorchvisionBackend",
    image_chw: torch.Tensor,
    output_size: list[int],
    resample: PILImageResampling,
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: list[float],
    image_std: list[float],
) -> torch.Tensor:
    resized = backend.resize(
        image_chw,
        size=SizeDict(height=output_size[0], width=output_size[1]),
        resample=resample,
        antialias=False,
    )
    return backend.rescale_and_normalize(resized, do_rescale, rescale_factor, do_normalize, image_mean, image_std)


def build_resized_image(
    backend: "TorchvisionBackend",
    images_nchw: torch.Tensor,
    base_image_input_size: int,
    resample: PILImageResampling,
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: list[float],
    image_std: list[float],
    image_patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # `images_nchw`: a batch of same-shape images `[N, C, H, W]`; resize the whole batch at once.
    resized = resize_and_normalize_image(
        backend,
        images_nchw,
        [base_image_input_size, base_image_input_size],
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
    crop_patch_h = crop_patch_w = base_image_input_size // image_patch_size
    resize_idx = torch.arange(crop_patch_w * crop_patch_h, dtype=torch.int32, device=images_nchw.device).reshape(
        crop_patch_h, crop_patch_w
    )
    return resized, resize_idx


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

    max_crops: int | None
    overlap_margins: list[int] | None
    patch_size: int | None
    pooling_size: list[int] | None


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
        images_nchw: torch.Tensor,
        max_crops: int,
        overlap_margins: list[int],
        base_image_input_size: int,
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: list[float],
        image_std: list[float],
        image_patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tile a batch of same-shape images `[N, C, H, W]` into overlapping square crops. The tiling and the per-patch index grid depend only on the (shared) shape, so they are computed once and the resize/unfold are batched over N."""
        crop_size = base_image_input_size
        left_margin, right_margin = overlap_margins
        crop_patches = crop_size // image_patch_size
        window_patches = crop_patches - (left_margin + right_margin)
        window_size = window_patches * image_patch_size
        margin_size = (left_margin + right_margin) * image_patch_size

        _, _, original_height, original_width = images_nchw.shape
        tiling_h, tiling_w = select_tiling(
            original_height - margin_size, original_width - margin_size, window_size, max_crops
        )
        src_height = tiling_h * window_size + margin_size
        src_width = tiling_w * window_size + margin_size

        src = resize_and_normalize_image(
            self,
            images_nchw,
            [src_height, src_width],
            resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
        )

        # [N, C, src_h, src_w] -> unfold spatial dims -> [N, C, tiling_h, tiling_w, crop, crop]
        crops = src.unfold(2, crop_size, window_size).unfold(3, crop_size, window_size)
        crops = (
            crops.permute(0, 2, 3, 4, 5, 1)
            .reshape(src.shape[0], tiling_h * tiling_w, crop_size, crop_size, 3)
            .contiguous()
        )

        patch_idx = torch.arange(
            tiling_h * tiling_w * crop_patches * crop_patches, dtype=torch.int32, device=images_nchw.device
        ).reshape(tiling_h, tiling_w, crop_patches, crop_patches)
        if left_margin:
            patch_idx[1:, :, :left_margin, :] = -1
            patch_idx[:, 1:, :, :left_margin] = -1
        if right_margin:
            patch_idx[:-1, :, -right_margin:, :] = -1
            patch_idx[:, :-1, :, -right_margin:] = -1

        patch_idx = patch_idx.permute(0, 2, 1, 3).reshape(-1)
        patch_idx = patch_idx[patch_idx >= 0].reshape(src_height // image_patch_size, src_width // image_patch_size)

        return crops, patch_idx

    def _image_batch_to_patches_and_grids(
        self,
        images_nchw: torch.Tensor,
        max_crops: int,
        overlap_margins: list[int],
        base_image_input_size: int,
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
        base_image_input_d = image_patch_size
        pooling_w = image_pooling_w
        pooling_h = image_pooling_h
        crop_patch_h = crop_patch_w = base_image_input_size // base_image_input_d

        crop_arr, patch_idx_arr = self._build_overlapping_crops(
            images_nchw,
            max_crops,
            overlap_margins,
            base_image_input_size,
            resample,
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
            image_patch_size,
        )
        pooling_idx = arange_for_pooling(patch_idx_arr, pooling_h, pooling_w)
        num_patch_rows, num_patch_cols = pooling_idx.shape[:2]
        pooling_idx = pooling_idx.reshape([-1, pooling_h * pooling_w])

        resized, resize_idx = build_resized_image(
            self,
            images_nchw,
            base_image_input_size,
            resample,
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
            image_patch_size,
        )
        # [N, 1, S, S, C] + [N, ncrops, S, S, C] -> [N, 1 + ncrops, S, S, C]
        crop_arr = torch.cat([resized, crop_arr], dim=1)

        resize_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
        resized_h, resized_w = resize_idx.shape[:2]
        resize_idx = resize_idx.reshape([-1, pooling_h * pooling_w])

        pooling_idx = torch.where(pooling_idx >= 0, pooling_idx + crop_patch_h * crop_patch_w, -1)
        pooling_idx = torch.cat([resize_idx, pooling_idx])
        image_grid = torch.tensor(
            [[resized_h, resized_w, num_patch_rows, num_patch_cols]], dtype=torch.int64, device=images_nchw.device
        )

        # [N, total_crops, S, S, C] -> patches [N, total_crops, n_patch, pixels_per_patch]
        n_images, total_crops = crop_arr.shape[0], crop_arr.shape[1]
        patches = batch_pixels_to_patches(
            crop_arr.reshape(n_images * total_crops, *crop_arr.shape[2:]), image_patch_size
        ).reshape(n_images, total_crops, -1, image_patch_size * image_patch_size * 3)

        # Expand the shared grid / pooling indices to the batch so they reorder per-image.
        return image_grid.expand(n_images, -1), patches, pooling_idx.unsqueeze(0).expand(n_images, -1, -1)

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
        disable_grouping: bool | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        if size.height != size.width:
            raise ValueError(f"Molmo2 only supports a square `size`, got height={size.height}, width={size.width}.")
        base_image_input_size = size.height
        image_pooling_h, image_pooling_w = pooling_size

        # Group images by shape and process each unique shape as a single batch (the tiling and all
        # patch/pooling indices are shape-dependent only), then restore the original order.
        device = images[0].device
        grouped_images, grouped_index = group_images_by_shape(images, disable_grouping=disable_grouping)

        grids_grouped: dict = {}
        patches_grouped: dict = {}
        pooled_grouped: dict = {}
        for shape, stacked_images in grouped_images.items():
            image_grid, patches, pooled_idx = self._image_batch_to_patches_and_grids(
                stacked_images,
                max_crops,
                overlap_margins,
                base_image_input_size,
                resample,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
                patch_size,
                image_pooling_w,
                image_pooling_h,
            )
            grids_grouped[shape] = image_grid
            patches_grouped[shape] = patches
            pooled_grouped[shape] = pooled_idx

        grids = reorder_images(grids_grouped, grouped_index)
        patches = reorder_images(patches_grouped, grouped_index)
        pooled = reorder_images(pooled_grouped, grouped_index)

        all_crops: list[torch.Tensor] = []
        all_pooled: list[torch.Tensor] = []
        patch_offset = 0
        for crops, pooled_idx in zip(patches, pooled):
            all_pooled.append(torch.where(pooled_idx >= 0, pooled_idx + patch_offset, pooled_idx))
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
        if images_kwargs is None:
            images_kwargs = {}
        max_crops = images_kwargs.get("max_crops", self.max_crops)
        overlap_margins = images_kwargs.get("overlap_margins", self.overlap_margins)
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        pooling_size = images_kwargs.get("pooling_size", self.pooling_size)
        size = images_kwargs.get("size", self.size)

        base_h = size["height"] if isinstance(size, dict) else size.height
        base_w = size["width"] if isinstance(size, dict) else size.width
        left_margin, right_margin = overlap_margins
        pooling_h, pooling_w = pooling_size

        total_margin_pixels = patch_size * (left_margin + right_margin)
        crop_patches = base_h // patch_size
        crop_window_patches = crop_patches - (left_margin + right_margin)
        crop_window_size = crop_window_patches * patch_size

        effective_h = height - total_margin_pixels
        effective_w = width - total_margin_pixels
        tiling_h, tiling_w = select_tiling(effective_h, effective_w, crop_window_size, max_crops)

        high_res_h = tiling_h * crop_window_patches + left_margin + right_margin
        high_res_w = tiling_w * crop_window_patches + left_margin + right_margin
        num_patch_rows_high = math.ceil(high_res_h / pooling_h)
        num_patch_cols_high = math.ceil(high_res_w / pooling_w)

        crop_patch_h = base_h // patch_size
        crop_patch_w = base_w // patch_size
        resized_h = math.ceil(crop_patch_h / pooling_h)
        resized_w = math.ceil(crop_patch_w / pooling_w)

        return resized_h * resized_w + num_patch_rows_high * num_patch_cols_high


class Molmo2VideosKwargs(VideosKwargs, total=False):
    """
    patch_size (`int`, *optional*):
        Side length in pixels of each ViT patch for video frames.
    pooling_size (`list[int]`, *optional*):
        `[pool_h, pool_w]` pooling window applied to video patch features.
    max_fps (`int`, *optional*):
        Maximum sampling rate in frames per second for short videos.
    """

    patch_size: int | None
    pooling_size: list[int] | None
    max_fps: int | None


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
        video_tchw: torch.Tensor,
        base_image_input_size: int,
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
        # `build_resized_image` is batch-native (`[N, C, H, W]`); all frames of a video are one batch.
        hwc, resize_idx = build_resized_image(
            self,
            video_tchw,
            base_image_input_size,
            resample,
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
            image_patch_size,
        )
        # The pooling index grid is shape-dependent only, hence shared by every frame.
        pooling_idx = arange_for_pooling(resize_idx, image_pooling_h, image_pooling_w)
        num_patch_rows, num_patch_cols = pooling_idx.shape[:2]
        pooling_idx = pooling_idx.reshape([-1, image_pooling_h * image_pooling_w])
        # [T, 1, S, S, C] -> [T, n_patch, pixels_per_patch]
        patches = batch_pixels_to_patches(hwc.squeeze(1), image_patch_size)
        return [num_patch_rows, num_patch_cols], patches, pooling_idx

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        size: SizeDict | None = None,
        resample: PILImageResampling | None = None,
        do_rescale: bool = True,
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_size: int | None = None,
        pooling_size: list[int] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        if size.height != size.width:
            raise ValueError(f"Molmo2 only supports a square `size`, got height={size.height}, width={size.width}.")
        base_image_input_size = size.height
        image_pooling_h, image_pooling_w = pooling_size

        all_crops: list[torch.Tensor] = []
        all_pooled: list[torch.Tensor] = []
        all_grids: list[torch.Tensor] = []
        patch_offset = 0

        for video in videos:
            image_grid, patches, pooling_idx = self._build_video_patches(
                video,
                base_image_input_size,
                resample,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
                patch_size,
                image_pooling_h,
                image_pooling_w,
            )
            num_frames, patches_per_frame = patches.shape[:2]
            frame_offsets = (
                patch_offset + torch.arange(num_frames, device=pooling_idx.device) * patches_per_frame
            ).view(-1, 1, 1)
            pooled_idx = torch.where(pooling_idx >= 0, pooling_idx + frame_offsets, pooling_idx)
            all_pooled.append(pooled_idx.reshape(-1, pooling_idx.shape[-1]))
            all_crops.append(patches)
            patch_offset += num_frames * patches_per_frame

            all_grids.append(torch.tensor([num_frames, image_grid[0], image_grid[1]], dtype=torch.int64))

        data = {
            "pixel_values_videos": torch.cat(all_crops, dim=0),
            "video_token_pooling": torch.cat(all_pooled, dim=0),
            "video_grids": torch.stack(all_grids, dim=0),
        }
        return BatchFeature(data, tensor_type=return_tensors)


class Molmo2ProcessorImagesKwargs(Molmo2ImagesKwargs, total=False):
    """
    max_crops (`int`, *optional*):
        Maximum number of image crops produced by the image processor.
    overlap_margins (`list[int]`, *optional*):
        Pixel margins `[left_right, top_bottom]` to overlap between neighboring crops.
    patch_size (`int`, *optional*):
        Side length in pixels of each ViT patch.
    pooling_size (`list[int]`, *optional*):
        `[pool_h, pool_w]` pooling window applied to patch features in the vision adapter.
    """


class Molmo2ProcessorKwargs(ProcessingKwargs, total=False):
    """Molmo2 processor kwargs"""

    images_kwargs: Molmo2ProcessorImagesKwargs
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
    image_patch_tokens = (
        "<im_patch>",
        "<im_col>",
        "<im_start>",
        "<low_res_im_start>",
        "<frame_start>",
        "<im_end>",
        "<frame_end>",
        "<im_low>",
    )

    @property
    def model_input_names(self):
        return super().model_input_names + ["mm_token_type_ids"]

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.image_token)

    @property
    def video_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.video_token)

    @property
    def image_token_ids(self) -> list[int]:
        return [self.tokenizer.convert_tokens_to_ids(token) for token in self.image_patch_tokens]

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
        super().__init__(image_processor, video_processor, tokenizer, chat_template=chat_template)

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
            per_row = ["<im_patch>"] * num_patch_cols
            if self.video_use_col_tokens:
                per_row = per_row + ["<im_col>"]
            video_string += "".join([start_token] + per_row * num_patch_rows + [end_token])

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

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        image_grid = image_inputs["image_grids"][image_idx]
        if hasattr(image_grid, "tolist"):
            image_grid = image_grid.tolist()
        resized_h, resized_w, height, width = image_grid

        per_row = ["<im_patch>"] * width
        if self.image_use_col_tokens:
            per_row = per_row + ["<im_col>"]
        high_res_tokens = ["<im_start>"] + per_row * height + ["<im_end>"]

        per_row = ["<im_patch>"] * resized_w
        use_single_crop_col_tokens = (
            self.image_use_col_tokens if self.use_single_crop_col_tokens is None else self.use_single_crop_col_tokens
        )
        image_start_token = "<low_res_im_start>" if self.use_single_crop_start_token else "<im_start>"
        if use_single_crop_col_tokens:
            per_row = per_row + ["<im_col>"]
        low_res_tokens = [image_start_token] + per_row * resized_h + ["<im_end>"]

        return "".join(low_res_tokens + high_res_tokens)

    def replace_video_token(self, video_inputs: dict, video_idx: int) -> str:
        video_grid = video_inputs["video_grids"][video_idx]
        video_metadata = video_inputs.get("video_metadata", [])
        metadata = video_metadata[video_idx] if video_idx < len(video_metadata) else None
        if metadata is not None:
            if metadata.frames_indices is None:
                metadata.frames_indices = list(range(int(video_grid[0].item())))
            if metadata.fps is None:
                metadata.fps = self.video_processor.max_fps or 2
                logger.warning_once(
                    "Molmo2 inserts frame timestamps into video prompts, but the input video's `fps` was not "
                    f"provided or could not be inferred. Defaulting to `fps={metadata.fps}`. Please provide "
                    "`video_metadata` for more accurate timestamps."
                )
            timestamps = metadata.timestamps
        else:
            fps = self.video_processor.max_fps or 2
            num_frames = int(video_grid[0].item())
            timestamps = [i / fps for i in range(num_frames)]
        return self.get_video_string(video_grid, timestamps)

    def apply_chat_template(
        self,
        conversation,
        chat_template: str | None = None,
        **kwargs,
    ):
        uses_default_template = chat_template is None
        if chat_template is None:
            if isinstance(self.chat_template, dict):
                chat_template = self.chat_template.get("default")
            else:
                chat_template = self.chat_template
        elif isinstance(self.chat_template, dict) and chat_template in self.chat_template:
            uses_default_template = True
            chat_template = self.chat_template[chat_template]

        if (
            uses_default_template
            and isinstance(chat_template, str)
            and self.tokenizer.bos_token is not None
            and "{{ bos_token" not in chat_template
            and not chat_template.lstrip().startswith(self.tokenizer.bos_token)
        ):
            chat_template = "{{ bos_token }}" + chat_template

        return super().apply_chat_template(conversation, chat_template=chat_template, **kwargs)


class Molmo2CausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


class Molmo2ModelOutputWithPast(LlavaModelOutputWithPast):
    pass


class Molmo2VisionMLP(Siglip2MLP):
    pass


class Molmo2VisionAttention(nn.Module):
    def __init__(
        self,
        config: Molmo2VisionConfig | Molmo2AdapterConfig,
        input_dim: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        input_dim = config.hidden_size if input_dim is None else input_dim
        self.q_proj = nn.Linear(input_dim, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(input_dim, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(input_dim, self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        key_value_states = hidden_states if key_value_states is None else key_value_states

        batch_size = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(key_value_states)
        value_states = self.v_proj(key_value_states)

        query_states = query_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

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
        )

        attn_output = attn_output.reshape(batch_size, -1, self.num_heads * self.head_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class Molmo2VisionEncoderLayer(Siglip2EncoderLayer):
    def __init__(self, config: Molmo2VisionConfig):
        super().__init__(config)
        self.self_attn = Molmo2VisionAttention(config)
        self.mlp = Molmo2VisionMLP(config)


class Molmo2VisionEncoder(Siglip2Encoder):
    def __init__(self, config: Molmo2VisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([Molmo2VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])


@auto_docstring
class Molmo2VisionModel(PreTrainedModel):
    config_class = Molmo2VisionConfig
    main_input_name = "pixel_values"
    input_modalities = "image"
    _no_split_modules = ["Molmo2VisionEncoderLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _can_record_outputs = {
        "hidden_states": Molmo2VisionEncoderLayer,
        "attentions": Molmo2VisionAttention,
    }

    def _init_weights(self, module):
        if isinstance(module, Molmo2VisionModel):
            init.normal_(module.positional_embedding, mean=0.0, std=self.config.initializer_range)
        else:
            super()._init_weights(module)

    def __init__(self, config: Molmo2VisionConfig):
        super().__init__(config)
        self.config = config
        self.image_default_input_size = config.image_default_input_size

        self.positional_embedding = nn.Parameter(
            torch.zeros(config.image_num_pos, config.hidden_size),
        )

        image_patch_size = config.image_patch_size
        self.patch_embedding = nn.Linear(
            image_patch_size * image_patch_size * 3,
            config.hidden_size,
            bias=True,
        )

        self.encoder = Molmo2VisionEncoder(config)

        self.post_init()

    @capture_outputs(tie_last_hidden_states=False)
    def forward(self, pixel_values: torch.Tensor, **kwargs) -> BaseModelOutputWithPooling:
        hidden_states = self.patch_embedding(pixel_values.to(dtype=self.dtype))
        # patch count == image_num_pos, locked by config; only retraining with a different grid breaks this.
        hidden_states = hidden_states + self.positional_embedding[None, :, :].to(hidden_states.dtype)

        encoder_outputs = self.encoder(hidden_states, **kwargs)
        last_hidden_state = encoder_outputs.last_hidden_state
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=last_hidden_state.mean(dim=1),
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
class Molmo2Adapter(PreTrainedModel):
    config_class = Molmo2AdapterConfig
    input_modalities = "image"
    _no_split_modules = ["Molmo2VisionAttention"]
    _supports_sdpa = True
    _supports_flash_attn = True

    def __init__(self, config: Molmo2AdapterConfig):
        super().__init__(config)
        pool_dim = config.hidden_size * len(config.vit_layers)
        self.image_pooling_2d = Molmo2VisionAttention(config, input_dim=pool_dim)
        self.image_projector = Molmo2ImageProjectorMLP(config)
        self.image_feature_dropout = nn.Dropout(config.image_feature_dropout)
        self.post_init()

    def forward(self, image_features: torch.Tensor, pooled_patches_idx: torch.Tensor) -> torch.Tensor:
        image_features = self.image_feature_dropout(image_features)
        flat_features = image_features.reshape(-1, image_features.shape[-1])

        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, -1)

        to_pool = flat_features[torch.clip(pooled_patches_idx, 0)]
        to_pool = to_pool * valid.to(to_pool.dtype)[..., None]

        keep_mask = valid.reshape(-1, 1, 1, valid.shape[-1])
        attention_mask = torch.zeros_like(keep_mask, dtype=to_pool.dtype).masked_fill_(
            ~keep_mask, torch.finfo(to_pool.dtype).min
        )
        denom = valid.float().sum(-1)
        denom = torch.where(denom == 0, 1, denom)
        query = to_pool.sum(-2, keepdim=True) / denom[:, None, None].to(to_pool.dtype)

        pooled_features, _ = self.image_pooling_2d(query, to_pool, attention_mask=attention_mask)
        pooled_features = pooled_features.squeeze(1)
        pooled_features = self.image_projector(pooled_features)
        return pooled_features[valid_token]


class Molmo2RotaryEmbedding(Gemma3RotaryEmbedding):
    def __init__(self, config: Molmo2TextConfig):
        nn.Module.__init__(self)
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.layer_types = sorted(set(config.rope_layer_types))
        self.rope_type = {}
        for layer_type in self.layer_types:
            rope_parameters = config.rope_parameters[layer_type]
            self.rope_type[layer_type] = rope_parameters["rope_type"]
            rope_init_fn: Callable = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            inv_freq, attention_scaling = rope_init_fn(self.config, layer_type=layer_type)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)


class Molmo2RMSNorm(LlamaRMSNorm):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__(hidden_size, eps=eps)
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=hidden_states.device.type):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Molmo2Attention(Olmo2Attention):
    """Molmo2 attention: Olmo2-style q/k RMSNorm with a fused QKV projection and renamed output projection."""

    def __init__(self, config: Molmo2TextConfig, layer_idx: int) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.fused_dims = (
            config.num_attention_heads * config.head_dim,
            config.head_dim * config.num_key_value_heads,
            config.head_dim * config.num_key_value_heads,
        )
        self.att_proj = nn.Linear(config.hidden_size, sum(self.fused_dims), bias=config.qkv_bias)
        self.attn_out = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)

        self.qk_norm_type = config.qk_norm_type
        if self.qk_norm_type == "qwen3":
            q_norm_dim = config.head_dim
            k_norm_dim = config.head_dim
        else:
            q_norm_dim = config.num_attention_heads * config.head_dim
            k_norm_dim = config.num_key_value_heads * config.head_dim
        self.q_norm = Molmo2RMSNorm(q_norm_dim, eps=config.layer_norm_eps)
        self.k_norm = Molmo2RMSNorm(k_norm_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        q_shape = (*input_shape, self.num_heads, self.head_dim)
        kv_shape = (*input_shape, self.num_key_value_heads, self.head_dim)

        qkv = self.att_proj(hidden_states)
        query_states, key_states, value_states = torch.split(qkv, self.fused_dims, dim=-1)

        value_states = value_states.view(kv_shape)

        if self.qk_norm_type == "olmo":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(q_shape)
        key_states = key_states.view(kv_shape)

        if self.qk_norm_type == "qwen3":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

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
        attn_output = self.attn_out(attn_output)
        return attn_output, attn_weights


class Molmo2MLP(Phi3MLP):
    def __init__(self, config: Molmo2TextConfig):
        nn.Module.__init__(self)
        self.ff_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.ff_out = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ff_proj(hidden_states)
        up_states, gate = hidden_states.chunk(2, dim=-1)
        hidden_states = self.act(gate) * up_states
        return self.ff_out(hidden_states)


class Molmo2DecoderLayer(Phi3DecoderLayer):
    def __init__(self, config: Molmo2TextConfig, layer_idx: int | None = None):
        GradientCheckpointingLayer.__init__(self)
        self.config = config
        self.norm_after = config.norm_after
        self.self_attn = Molmo2Attention(config, layer_idx)
        self.attn_norm = Molmo2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.residual_dropout)
        self.mlp = Molmo2MLP(config)
        self.ff_norm = Molmo2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

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
        # `norm_after` selects post-norm (normalize each sublayer's *output*) over the default
        # pre-norm (normalize each sublayer's *input*) -- a config flag instead of a subclass.
        residual = hidden_states
        if not self.norm_after:
            hidden_states = self.attn_norm(hidden_states)

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
            hidden_states = self.attn_norm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        # Fully Connected
        residual = hidden_states
        if not self.norm_after:
            hidden_states = self.ff_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.norm_after:
            hidden_states = self.ff_norm(hidden_states)
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

    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Molmo2DecoderLayer,
        "attentions": Molmo2Attention,
    }

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, Molmo2VisionModel):
            init.normal_(module.positional_embedding, mean=0.0, std=std)
        elif isinstance(module, Molmo2RotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), inv_freq)
        else:
            super()._init_weights(module)


class Molmo2TextModel(LlamaModel):
    config: Molmo2TextConfig

    def __init__(self, config: Molmo2TextConfig):
        Molmo2PreTrainedModel.__init__(self, config)
        self.vocab_size = config.vocab_size
        # The checkpoint's extra-vocabulary table is concatenated onto the base one at load time
        # (see `conversion_mapping.py`), so the embedding covers `vocab_size + additional_vocab_size`.
        self.embed_tokens = nn.Embedding(config.vocab_size + (config.additional_vocab_size or 0), config.hidden_size)
        self.emb_drop = nn.Dropout(config.embedding_dropout)
        self.layers = nn.ModuleList(
            [Molmo2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Molmo2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = Molmo2RotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
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

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = self.emb_drop(inputs_embeds)

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            ).unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            causal_mask_mapping = {
                "full_attention": create_causal_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
            }

        hidden_states = inputs_embeds

        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in self.rotary_emb.layer_types
        }

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[layer_idx]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings[self.config.rope_layer_types[layer_idx]],
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


def token_type_ids_mask_function(token_type_ids: torch.Tensor | None = None) -> Callable | None:
    if token_type_ids is None:
        return None

    seq_len = token_type_ids.shape[1]

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # Indices past `token_type_ids` (static cache, assisted decoding) are generated text: type 0.
        q_type = torch.where(q_idx < seq_len, token_type_ids[batch_idx, q_idx.clamp(max=seq_len - 1)], 0)
        kv_type = torch.where(kv_idx < seq_len, token_type_ids[batch_idx, kv_idx.clamp(max=seq_len - 1)], 0)
        return (q_type == 1) & (kv_type == 1)

    return inner_mask


class Molmo2Model(Molmo2PreTrainedModel):
    config: Molmo2Config

    def __init__(self, config: Molmo2Config):
        super().__init__(config)
        self.language_model: Molmo2TextModel = Molmo2TextModel(config.text_config)
        self.image_col_id = config.image_col_id
        self.image_low_res_id = config.image_low_res_id
        # `vision_config.num_hidden_layers` and `adapter_config.vit_layers` are normalized in
        # `Molmo2Config.__post_init__`.
        self.vit_layers = list(config.adapter_config.vit_layers)
        self.vision_tower = Molmo2VisionModel(config.vision_config)
        self.multi_modal_projector = Molmo2Adapter(config.adapter_config)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_token_pooling: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        image_shape = None
        if pixel_values.dim() == 4:
            image_shape = pixel_values.shape[:3]
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-2], pixel_values.shape[-1])

        # The adapter pools a concatenation of intermediate layers, so hidden states are always needed.
        kwargs["output_hidden_states"] = True
        image_outputs: BaseModelOutputWithPooling = self.vision_tower(pixel_values, **kwargs)
        image_features = torch.cat([image_outputs.hidden_states[layer + 1] for layer in self.vit_layers], dim=-1)

        if image_shape is not None:
            image_features = image_features.reshape(*image_shape, -1)
        image_outputs.pooler_output = self.multi_modal_projector(image_features, image_token_pooling)
        return image_outputs

    @can_return_tuple
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_token_pooling: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        return self.get_image_features(pixel_values_videos, video_token_pooling, **kwargs)

    # Copied from transformers.models.llava.modeling_llava.LlavaModel.get_placeholder_mask
    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        n_image_features = image_features.shape[0] * image_features.shape[1]
        special_image_mask = special_image_mask.unsqueeze(-1).to(inputs_embeds.device)
        torch_compilable_check(
            n_image_tokens * inputs_embeds.shape[-1] == image_features.numel(),
            f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {n_image_features}",
        )
        return special_image_mask

    @can_return_tuple
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
            image_features = self.get_image_features(pixel_values, image_token_pooling).pooler_output
        elif pixel_values_videos is not None:
            image_features = self.get_video_features(pixel_values_videos, video_token_pooling).pooler_output

        if image_features is not None:
            # `get_placeholder_mask` returns a [batch, seq, 1] mask; Molmo2 *adds* the image features onto
            # the placeholder-token embeddings (residual), which means we index `inputs_embeds` directly.
            # Boolean indexing does not broadcast, so the mask must be expanded to the hidden dim first.
            special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            special_image_mask = special_image_mask.expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask,
                inputs_embeds[special_image_mask] + image_features.reshape(-1),
            )

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            if self.training and mm_token_type_ids is None:
                raise ValueError("`mm_token_type_ids` is required as a model input when training")

            mask_kwargs = {
                "config": self.config.get_text_config(),
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            is_prefill = past_key_values is None or not past_key_values.is_initialized or image_features is not None
            if mm_token_type_ids is not None and is_prefill:
                mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                    mm_token_type_ids.to(inputs_embeds.device)
                )
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}

        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
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
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size)

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
        model_kwargs.update(visual)
        return input_ids, model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor | None = None,
        is_first_iteration: bool = False,
        use_cache: bool = True,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            mm_token_type_ids=mm_token_type_ids,
            is_first_iteration=is_first_iteration,
            use_cache=use_cache,
            **kwargs,
        )

        if is_first_iteration or not use_cache:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_token_pooling"] = image_token_pooling
            model_inputs["image_grids"] = image_grids
            model_inputs["image_num_crops"] = image_num_crops
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["video_token_pooling"] = video_token_pooling
            model_inputs["video_grids"] = video_grids

        return model_inputs

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
            mask_kwargs["or_mask_function"] = token_type_ids_mask_function(mm_token_type_ids.to(inputs_embeds.device))

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
