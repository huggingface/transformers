# Copyright 2025 Perceptron, Inc and The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import Any

from ... import initialization as init
from ...cache_utils import DynamicCache
from ...configuration_utils import PretrainedConfig, layer_type_validation
from ...feature_extraction_utils import BatchFeature
from ...generation.utils import GenerationMixin
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    ImagesKwargs,
    SizeDict,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    PILImageResampling,
)
from ...masking_utils import create_bidirectional_mask, create_masks_for_generate
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...models.qwen3.configuration_qwen3 import Qwen3Config
from ...models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3PreTrainedModel,
)
from ...processing_utils import ProcessorMixin, Unpack
from ...utils import TensorType, auto_docstring
from ...utils.constants import IMAGENET_STANDARD_MEAN as VISION_MEAN
from ...utils.constants import IMAGENET_STANDARD_STD as VISION_STD
from ...utils.generic import TransformersKwargs, can_return_tuple, maybe_autocast, merge_with_config_defaults
from ...utils.import_utils import (
    is_torch_available,
    is_torchdynamo_compiling,
    is_torchvision_available,
    is_vision_available,
)
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRotaryEmbedding
from ..siglip2.configuration_siglip2 import Siglip2VisionConfig
from ..siglip2.modeling_siglip2 import (
    Siglip2Attention,
    Siglip2Encoder,
    Siglip2EncoderLayer,
    Siglip2VisionEmbeddings,
)


if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
if is_vision_available():
    from PIL.Image import Image
else:
    Image = None
if is_torchvision_available():
    from ..pix2struct.image_processing_pix2struct_fast import torch_extract_patches


class ModalityType(IntEnum):
    """
    Modality identifiers for events.

    Members:
        image: Vision tokens (e.g., patches).
        text: Textual tokens.
    """

    image = 0
    text = 1


class IsaacVisionConfig(Siglip2VisionConfig):
    """Vision configuration for Isaac with Pixel Shuffle support.

    Extends Siglip2VisionConfig with additional fields for pixel shuffle.

    Args:
        pixel_shuffle_scale_factor (`int`, *optional*, defaults to 1):
            Spatial factor applied before pixel shuffle reduces the resolution.
        num_patches (`int`, *optional*, defaults to 256):
            Maximum number of learnable positional embeddings to initialize.
    """

    model_type = "isaac_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        num_patches=256,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        pixel_shuffle_scale_factor=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.num_patches = num_patches

        # Add our custom fields
        self.pixel_shuffle_scale_factor = pixel_shuffle_scale_factor

        # Ensure a sensible default attention backend
        if getattr(self, "_attn_implementation", None) is None:
            self._attn_implementation = "sdpa"


class IsaacImageProcessorFastKwargs(ImagesKwargs, total=False):
    """
    patch_size (`int`, *optional*):
        Side length (in pixels) for square patches extracted from resized images.
    max_num_patches (`int`, *optional*):
        Upper bound on extracted patches per image after resizing.
    min_num_patches (`int`, *optional*):
        Lower bound on extracted patches per image after resizing.
    pixel_shuffle_scale (`int`, *optional*):
        Pixel-shuffle reduction factor applied in the vision tower.
    """

    patch_size: int | None
    max_num_patches: int | None
    min_num_patches: int | None
    pixel_shuffle_scale: int | None


@auto_docstring
class IsaacImageProcessorFast(BaseImageProcessorFast):
    MAX_PIXELS = 60_000_000  # 60‑megapixel ceiling ≈ 8200 × 7300 px

    resample = PILImageResampling.BILINEAR
    model_input_names = ["patches", "token_grids"]
    valid_kwargs = IsaacImageProcessorFastKwargs
    unused_kwargs = ["size", "do_center_crop", "crop_size", "pad_size", "do_pad"]

    do_resize = True
    do_center_crop = False
    patch_size: int | None = 16
    max_num_patches: int | None = 256
    min_num_patches: int | None = None
    pixel_shuffle_scale: int | None = 1
    do_pad = False
    do_rescale = True
    do_normalize = True
    image_mean = list(VISION_MEAN)
    image_std = list(VISION_STD)
    do_convert_rgb = True
    disable_grouping = False

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def _validate_preprocess_kwargs(self, **kwargs):
        # Allow callers to omit resize-related placeholders that BaseImageProcessorFast checks for.
        kwargs.pop("do_resize", None)
        return super()._validate_preprocess_kwargs(**kwargs)

    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        **kwargs,
    ) -> torch.Tensor:
        resize_kwargs: dict[str, Any] = {"align_corners": False}
        resize_mode = "bilinear"

        return F.interpolate(
            image,
            size=(size.height, size.width),
            mode=resize_mode,
            **resize_kwargs,
        )

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        interpolation: Any | None,
        do_rescale: bool | None,
        rescale_factor: float | None,
        do_normalize: bool | None,
        image_mean: float | Sequence[float] | None,
        image_std: float | Sequence[float] | None,
        disable_grouping: bool | None = None,
        return_tensors: str | TensorType | None = None,
        *,
        patch_size: int | None = None,
        max_num_patches: int | None = None,
        min_num_patches: int | None = None,
        pixel_shuffle_scale: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)

        grouped_outputs = {}

        for shape, stacked_images in grouped_images.items():
            batch_size, channels, original_height, original_width = stacked_images.shape

            if bool(self.do_convert_rgb) and channels == 1:
                stacked_images = stacked_images.repeat(1, 3, 1, 1)

            target_height, target_width = get_image_size_for_max_num_patches(
                original_height,
                original_width,
                patch_size,
                max_num_patches,
                min_num_patches=min_num_patches,
                pixel_shuffle_scale=pixel_shuffle_scale,
            )
            if do_resize:
                image_batch = self.resize(
                    stacked_images, SizeDict(height=target_height, width=target_width), interpolation=interpolation
                )
            else:
                if (original_height % patch_size) or (original_width % patch_size):
                    raise ValueError(
                        f"Image dimensions (h={original_height}, w={original_width}) must be divisible by patch_size={patch_size} when resize is disabled; enable resizing or adjust the input resolution."
                    )
                image_batch, target_height, target_width = stacked_images, original_height, original_width

            image_batch = self.rescale_and_normalize(
                image_batch,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
            )

            patches = torch_extract_patches(image_batch, patch_size, patch_size)
            _, height_tokens, width_tokens, _ = patches.shape

            token_grid = (
                torch.tensor([height_tokens, width_tokens], device=patches.device).long().expand(batch_size, 2)
            )

            real_dim = (
                torch.tensor(
                    [1, height_tokens, width_tokens],
                    dtype=torch.long,
                    device=patches.device,
                )
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )

            if (height_tokens % pixel_shuffle_scale) or (width_tokens % pixel_shuffle_scale):
                raise ValueError(
                    f"Token grid (h={height_tokens}, w={width_tokens}) must be divisible by pixel_shuffle_scale={pixel_shuffle_scale}; adjust resize/patch parameters or disable pixel shuffle."
                )
            virtual_height = height_tokens // pixel_shuffle_scale
            virtual_width = width_tokens // pixel_shuffle_scale

            virtual_dim = (
                torch.tensor(
                    [1, virtual_height, virtual_width],
                    dtype=torch.long,
                    device=patches.device,
                )
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )
            grouped_outputs[shape] = (patches, token_grid, virtual_dim, real_dim)

        keys = ("patches", "token_grids", "virtual_pixel_size", "real_pixel_size")
        tensors: dict[str, torch.Tensor] = {}

        for i, key in enumerate(keys):
            slices = reorder_images(
                {shape: values[i] for shape, values in grouped_outputs.items()},
                grouped_images_index,
            )
            tensors[key] = torch.stack(slices, dim=0)

        return BatchFeature(data=tensors, tensor_type=return_tensors)


class IsaacVisionEmbeddings(Siglip2VisionEmbeddings):
    """Adapter around SigLIP2 vision embeddings that consumes packed patch sequences.

    Isaac accepts variable-resolution vision inputs as a single packed sequence with per-image
    `token_grids`; packing/unpacking here reconstructs per-image shapes so we can resize positional
    embeddings and build `cu_seqlens` for variable-length attention (not generic generation packing).
    """

    def __init__(self, config: IsaacVisionConfig):
        super().__init__(config)
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Parameter(
            torch.empty(
                self.position_embedding_size,
                self.position_embedding_size,
                self.embed_dim,
            )
        )
        nn.init.normal_(self.position_embedding)

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # pixel_values: (num_images, max_patches, patch_dim)
        if pixel_values.numel() == 0:
            return pixel_values.new_zeros((0, 0, self.embed_dim))

        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        resized_positional_embeddings = self.resize_positional_embeddings(
            self.position_embedding,
            spatial_shapes,
            max_length=pixel_values.shape[1],
        )
        embeddings = patch_embeds + resized_positional_embeddings

        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1).to(dtype=embeddings.dtype)

        return embeddings


class IsaacVisionAttention(Siglip2Attention):
    """Custom attention that supports variable-length sequences with flash/SDPA backends."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        **kwargs,
    ):
        batch_size, seq_length, embed_dim = hidden_states.shape
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_impl = self.config._attn_implementation or "sdpa"
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS[attn_impl]

        attention_kwargs: dict[str, Any] = {
            "is_causal": False,
            "scaling": self.scale,
        }

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            **attention_kwargs,
        )
        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class IsaacVisionEncoderLayer(Siglip2EncoderLayer):
    """Isaac vision encoder layer using the shared attention interfaces."""

    def __init__(self, config: IsaacVisionConfig):
        super().__init__(config)
        self.self_attn = IsaacVisionAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_output, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs,
        )

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class IsaacVisionEncoder(Siglip2Encoder):
    """Encoder using Isaac encoder layers."""

    def __init__(self, config: IsaacVisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([IsaacVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])


def pixel_shuffle_padded(
    x: torch.Tensor,
    token_grids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    scale_factor: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply pixel shuffle per image on padded batched vision embeddings.

    Args:
        x (`torch.Tensor`):
            Vision embeddings of shape `(num_images, max_patches, hidden_size)`.
        token_grids (`torch.Tensor`):
            Grid sizes `(height, width)` per image, shape `(num_images, 2)`.
        attention_mask (`torch.Tensor`, *optional*):
            Patch validity mask of shape `(num_images, max_patches)`.
        scale_factor (`int`, *optional*, defaults to 1):
            Spatial down-sampling factor.

    Returns:
        Tuple of:
            - pixel-shuffled embeddings `(num_images, max_tokens, hidden_size * scale_factor**2)`
            - attention mask `(num_images, max_tokens)`
            - per-image valid token lengths `(num_images,)`
    """
    num_images, _, embed_dim = x.shape

    output_lengths: list[int] = []
    for image_idx in range(num_images):
        height_tokens, width_tokens = token_grids[image_idx].to(torch.long).tolist()
        if height_tokens == 0 or width_tokens == 0:
            output_lengths.append(0)
            continue
        if not is_torchdynamo_compiling() and ((height_tokens % scale_factor) or (width_tokens % scale_factor)):
            raise ValueError(
                f"Every (H, W) grid must be divisible by pixel_shuffle_scale={scale_factor}, got {(height_tokens, width_tokens)}."
            )
        output_lengths.append((height_tokens // scale_factor) * (width_tokens // scale_factor))

    max_output_tokens = max(output_lengths, default=0)
    output_dim = embed_dim * scale_factor * scale_factor

    shuffled = x.new_zeros((num_images, max_output_tokens, output_dim))
    shuffled_attention_mask = torch.zeros((num_images, max_output_tokens), device=x.device, dtype=torch.long)

    for image_idx in range(num_images):
        out_length = output_lengths[image_idx]
        if out_length == 0:
            continue

        height_tokens, width_tokens = token_grids[image_idx].to(torch.long).tolist()
        seq_length = height_tokens * width_tokens
        if attention_mask is not None:
            seq_length = min(seq_length, int(attention_mask[image_idx].sum().item()))
        if seq_length == 0:
            continue

        # Vision patches are contiguous in row-major order.
        height_blocks = height_tokens // scale_factor
        width_blocks = width_tokens // scale_factor
        tokens = x[image_idx, :seq_length]
        tokens = tokens.view(height_tokens, width_tokens, embed_dim).permute(2, 0, 1).unsqueeze(0)
        tokens = F.pixel_unshuffle(tokens, downscale_factor=scale_factor)
        tokens = tokens.view(1, embed_dim, scale_factor, scale_factor, height_blocks, width_blocks)
        tokens = tokens.permute(0, 4, 5, 2, 3, 1).contiguous().view(out_length, output_dim)
        shuffled[image_idx, :out_length] = tokens
        shuffled_attention_mask[image_idx, :out_length] = 1

    shuffled_lengths = torch.tensor(output_lengths, device=x.device, dtype=torch.long)
    return shuffled, shuffled_attention_mask, shuffled_lengths


class IsaacVisionTransformer(PreTrainedModel):
    """Vision tower for padded variable-resolution patches with per-image masks.

    Args:
        config (IsaacVisionConfig): Vision configuration with pixel-shuffle and patching parameters.

    Inputs:
        vision_tokens (Tuple[Tensor, Tensor, Optional[Tensor]]):
            `(patches, token_grids, patch_attention_mask)` where:
            - `patches`: `(num_images, max_patches, patch_dim)`
            - `token_grids`: `(num_images, 2)` with per-image `(H_tokens, W_tokens)`
            - `patch_attention_mask`: `(num_images, max_patches)` or `None`

    Returns:
        Tuple of `(pixel_shuffled_features, attention_mask, token_lengths)`.
    """

    _supports_sdpa = True
    _supports_flash_attn = True

    def __init__(self, config: IsaacVisionConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = IsaacVisionEmbeddings(config)
        self.encoder = IsaacVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pixel_shuffle_scale_factor = config.pixel_shuffle_scale_factor

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, IsaacVisionEmbeddings):
            init.zeros_(module.position_embedding)

    def forward(
        self,
        vision_tokens: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(vision_tokens) == 2:
            seq_patches, token_grids = vision_tokens
            vision_patch_attention_mask = None
        else:
            seq_patches, token_grids, vision_patch_attention_mask = vision_tokens

        if seq_patches.numel() == 0:
            hidden_dim = self.config.hidden_size * (self.pixel_shuffle_scale_factor**2)
            empty_hidden = seq_patches.new_zeros((0, 0, hidden_dim))
            empty_mask = torch.zeros((0, 0), device=seq_patches.device, dtype=torch.long)
            empty_lengths = torch.zeros((0,), device=seq_patches.device, dtype=torch.long)
            return empty_hidden, empty_mask, empty_lengths

        hidden_states = self.embeddings(
            seq_patches,
            token_grids,
            attention_mask=vision_patch_attention_mask,
        )

        encoder_attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=vision_patch_attention_mask,
        )
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, attention_mask=encoder_attention_mask)
        hidden_states = self.post_layernorm(encoder_outputs.last_hidden_state)

        return pixel_shuffle_padded(
            x=hidden_states,
            token_grids=token_grids,
            attention_mask=vision_patch_attention_mask,
            scale_factor=self.pixel_shuffle_scale_factor,
        )


class IsaacMultiModalProjector(nn.Module):
    """Maps vision tower outputs to the text hidden size with a SiLU MLP."""

    def __init__(self, config: IsaacConfig):
        super().__init__()
        self.vision_hidden_size = config.vision_config.hidden_size * (
            config.vision_config.pixel_shuffle_scale_factor**2
        )
        self.backbone_hidden_size = config.hidden_size
        self.linear_1 = nn.Linear(self.vision_hidden_size, 4 * self.vision_hidden_size, bias=False)
        self.silu = nn.SiLU()
        self.linear_2 = nn.Linear(4 * self.vision_hidden_size, self.backbone_hidden_size, bias=False)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.silu(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class IsaacVisionEmbedding(nn.Module):
    def __init__(self, config: IsaacConfig):
        super().__init__()
        vision_cfg = config.vision_config

        self.vision_tower = IsaacVisionTransformer(vision_cfg)
        self.multimodal_projector = IsaacMultiModalProjector(config)

    def forward(
        self,
        vision_tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vision_patches, token_grids, vision_patch_attention_mask = vision_tokens
        hidden_states, token_attention_mask, token_lengths = self.vision_tower(
            (vision_patches, token_grids, vision_patch_attention_mask)
        )
        projected = self.multimodal_projector(hidden_states)
        return projected, token_attention_mask, token_lengths


def get_scaled_image_size(
    scale: float,
    original_size: int,
    patch_size: int,
    pixel_shuffle_scale: int,
) -> int:
    scaled_size = scale * original_size
    divisor = patch_size * pixel_shuffle_scale
    scaled_size = math.ceil(scaled_size / divisor) * divisor
    scaled_size = max(divisor, scaled_size)
    return int(scaled_size)


def get_image_size_for_max_num_patches(
    image_height: int,
    image_width: int,
    patch_size: int,
    max_num_patches: int,
    min_num_patches: int | None = None,
    eps: float = 1e-5,
    pixel_shuffle_scale: int = 1,
) -> tuple[int, int]:
    r"""Compute a target resolution whose patch grid satisfies patching parametrization.

    Args:
        image_height (`int`):
            Height in pixels of the source image prior to any resizing.
        image_width (`int`):
            Width in pixels of the source image prior to any resizing.
        patch_size (`int`):
            Size of the square patch used by the vision encoder.
        max_num_patches (`int`):
            Upper bound on `(height / patch_size) * (width / patch_size)` after resizing.
        min_num_patches (`int`, *optional*):
            Lower bound on the number of patches. When provided the image will be scaled up if necessary.
        eps (`float`, *optional*, defaults to 1e-5):
            Convergence tolerance for the internal binary search to determing the target dimensions.
        pixel_shuffle_scale (`int`, *optional*, defaults to 1):
            Additional stride multiplier applied when pixel shuffle later reduces spatial resolution.

    Returns:
        `tuple[int, int]`: Height and width (in pixels) that are multiples of `patch_size * pixel_shuffle_scale`
        and respect both the maximum and optional minimum patch-count constraints.
    """

    # Ensure divisibility
    divisor = patch_size * pixel_shuffle_scale
    adjusted_height = math.ceil(image_height / divisor) * divisor
    adjusted_height = max(divisor, adjusted_height)
    adjusted_width = math.ceil(image_width / divisor) * divisor
    adjusted_width = max(divisor, adjusted_width)

    num_patches = (adjusted_height / patch_size) * (adjusted_width / patch_size)

    if min_num_patches is not None and num_patches < min_num_patches:
        # Scale up via binary search to satisfy the minimum patch budget while
        # preserving divisibility by patch_size * pixel_shuffle_scale.
        scale_min, scale_max = 1.0, 100.0
        while (scale_max - scale_min) >= eps:
            scale = (scale_min + scale_max) / 2
            target_height = get_scaled_image_size(scale, image_height, patch_size, pixel_shuffle_scale)
            target_width = get_scaled_image_size(scale, image_width, patch_size, pixel_shuffle_scale)
            num_patches = (target_height / patch_size) * (target_width / patch_size)
            if num_patches >= min_num_patches:
                scale_max = scale
            else:
                scale_min = scale
        scale = scale_max
        target_height = get_scaled_image_size(scale, image_height, patch_size, pixel_shuffle_scale)
        target_width = get_scaled_image_size(scale, image_width, patch_size, pixel_shuffle_scale)
        return target_height, target_width
    elif num_patches <= max_num_patches:
        return adjusted_height, adjusted_width
    else:
        # Scale down
        scale_min, scale_max = eps / 10, 1.0
        while (scale_max - scale_min) >= eps:
            scale = (scale_min + scale_max) / 2
            target_height = get_scaled_image_size(scale, image_height, patch_size, pixel_shuffle_scale)
            target_width = get_scaled_image_size(scale, image_width, patch_size, pixel_shuffle_scale)
            num_patches = (target_height / patch_size) * (target_width / patch_size)
            if num_patches <= max_num_patches:
                scale_min = scale
            else:
                scale_max = scale
        scale = scale_min
        target_height = get_scaled_image_size(scale, image_height, patch_size, pixel_shuffle_scale)
        target_width = get_scaled_image_size(scale, image_width, patch_size, pixel_shuffle_scale)
        return target_height, target_width


class IsaacConfig(PretrainedConfig):
    """Configuration class for Isaac multimodal model.

    This configuration corresponds to checkpoints such as
    [Perceptron/isaac-base](https://huggingface.co/Perceptron/isaac-base).
    """

    model_type = "isaac"
    sub_configs = {"vision_config": IsaacVisionConfig, "text_config": Qwen3Config}
    image_processor_type = "IsaacImageProcessor"

    def __init__(
        self,
        vision_config: IsaacVisionConfig | None = None,
        text_config: Qwen3Config | dict | None = None,
        vision_rescale_factor: float = 1 / 255,
        max_sequence_length: int = 16384,
        vision_token: str = "<image>",
        **kwargs,
    ):
        attn_implementation = kwargs.get("attn_implementation")

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif isinstance(text_config, Qwen3Config):
            self.text_config = text_config
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        # Seed RoPE parameters before base init so the shared mixin can standardize/validate them.
        self.rope_parameters = getattr(self.text_config, "rope_parameters", None)
        self.layer_types = getattr(self.text_config, "layer_types", None)

        super().__init__(**kwargs)

        # Keep rope parameters aligned between the composite and text sub-configs.
        self.text_config.rope_parameters = self.rope_parameters

        # Mirror frequently accessed Qwen3 attributes at the composite config level
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.num_attention_heads = self.text_config.num_attention_heads
        self.head_dim = self.text_config.head_dim
        self.hidden_act = self.text_config.hidden_act
        self.use_cache = self.text_config.use_cache
        self.rope_theta = self.rope_parameters["rope_theta"]
        self.max_position_embeddings = getattr(self.text_config, "max_position_embeddings", max_sequence_length)
        self.text_config.max_position_embeddings = self.max_position_embeddings

        self.layer_types = getattr(self.text_config, "layer_types", None)
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        # Handle vision config - either dict or IsaacVisionConfig instance
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif isinstance(vision_config, IsaacVisionConfig):
            self.vision_config = vision_config
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        # Propagate user-requested attention backend to the vision sub-config when provided.
        if attn_implementation is not None:
            if isinstance(attn_implementation, dict):
                vision_attn = attn_implementation.get("vision_config", attn_implementation.get("", None))
            else:
                vision_attn = attn_implementation
            if vision_attn is not None:
                self.vision_config._attn_implementation = vision_attn

        if getattr(self, "_attn_implementation", None) is None:
            self._attn_implementation = "sdpa"
        # Vision normalization parameters
        self.vision_rescale_factor = float(vision_rescale_factor)

        # Processing parameters
        self.max_sequence_length = max_sequence_length
        self.vision_token = vision_token

    def to_dict(self):
        output = super().to_dict()
        # Ensure nested configs round-trip through dict serialization
        if hasattr(self, "text_config") and self.text_config is not None:
            output["text_config"] = self.text_config.to_dict()
        if hasattr(self, "vision_config") and self.vision_config is not None:
            output["vision_config"] = self.vision_config.to_dict()
        return output


class IsaacProcessor(ProcessorMixin):
    """Processor that pairs the Isaac image processor with the Qwen2 tokenizer.

    Args:
        image_processor: Vision preprocessor (fast) used for patch extraction.
        tokenizer: Qwen2 tokenizer instance.
        vision_token (str, optional): Placeholder token marking image locations. Defaults to "<image>".
        max_sequence_length (int, optional): Maximum combined text+vision tokens kept. Defaults to 16384.
        rescale_factor (float, optional): Image rescale factor; defaults to 1/255.
        config (IsaacConfig | dict, optional): If provided, overrides processor defaults from the model config.

    Returns:
        BatchFeature: Top-level batched text and vision tensors.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = ("IsaacImageProcessorFast",)
    tokenizer_class = ("Qwen2Tokenizer",)
    pad_token_id = 151643

    def __init__(
        self,
        image_processor,
        tokenizer,
        *,
        vision_token: str = "<image>",
        max_sequence_length: int = 16384,
        rescale_factor: float | None = None,
        config: IsaacConfig | dict | None = None,
    ) -> None:
        if isinstance(config, dict):
            config = IsaacConfig(**config)

        if config is not None:
            vision_token = config.vision_token
            max_sequence_length = config.max_sequence_length
            rescale_factor = config.vision_rescale_factor

        resolved_rescale_factor = float(rescale_factor) if rescale_factor is not None else float(1 / 255)
        if config is not None:
            config.vision_rescale_factor = resolved_rescale_factor

        self.image_processor = image_processor
        super().__init__(image_processor, tokenizer)

        text_pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        image_pad_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        self.text_pad_token_id = int(text_pad_token_id)
        self.image_pad_token_id = int(image_pad_token_id)
        self.pad_token_id = self.text_pad_token_id

        self.current_processor = self.image_processor
        self.config = config
        self.chat_template = getattr(self.tokenizer, "chat_template", None)
        self.vision_token = vision_token
        self.max_sequence_length = max_sequence_length

    def _parse_sample(self, text: str, images: list[Image] | None) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        segments = text.split(self.vision_token)  # Parse by vision_token; interleave text segments and image segments.
        num_images = len(segments) - 1
        items: list[dict[str, Any]] = []
        total = 0
        num_provided_images = len(images) if images is not None else 0
        if not num_images == num_provided_images:
            raise ValueError(
                f"IsaacProcessor expects one image per image token, got {num_images} tokens and {num_provided_images} images in sample with text {text} "
            )

        for index, segment in enumerate(segments):
            if segment:
                tok = (
                    self.tokenizer.encode(segment, add_special_tokens=False, return_tensors="pt")
                    .squeeze(0)
                    .to(torch.long)
                )
                segment_length = int(tok.numel())
                items.append({"type": "text", "segment_length": segment_length, "tok": tok})
                total += segment_length

            if index < num_images:
                feat = self.image_processor(images=images[index], return_tensors=TensorType.PYTORCH)
                patches = feat["patches"][0].reshape(-1, feat["patches"].shape[-1])

                virtual_pixel_size = feat["virtual_pixel_size"][0].to(torch.long).tolist()
                real_pixel_size = feat["real_pixel_size"][0].to(torch.long).tolist()
                dims = tuple((virtual_pixel_size + [1, 1, 1])[:3])  # (T,H,W) in virtual space
                segment_length = int(dims[0] * dims[1] * dims[2])

                items.append(
                    {
                        "type": "image",
                        "segment_length": segment_length,
                        "dims": dims,
                        "patches": patches,
                        "grid": (int(real_pixel_size[1]), int(real_pixel_size[2])),
                    }
                )
                total += segment_length

        # Tail crop window.
        start = max(0, total - self.max_sequence_length)
        end = total

        image_pad_value = self.image_pad_token_id
        base_device: torch.device | None = None
        position_ids, modality, input_ids = [], [], []
        vpatches, grids, vision_token_offsets, vision_token_lengths = [], [], [], []

        global_offset = 0
        position_offset = 0

        for item in items:
            segment_length = int(item["segment_length"])
            current_window_start = max(start, global_offset)
            current_window_end = min(end, global_offset + segment_length)
            has_overlap = current_window_end > current_window_start

            if has_overlap and base_device is None:
                base_device = item["patches"].device if item["type"] == "image" else item["tok"].device

            if has_overlap:
                segment_local_start = int(current_window_start - global_offset)
                segment_local_end = int(current_window_end - global_offset)
                segment_local_indices = torch.arange(
                    segment_local_start, segment_local_end, device=base_device, dtype=torch.long
                )
                segment_kept_length = segment_local_end - segment_local_start

                if item["type"] == "text":
                    slice_index = segment_local_indices + position_offset
                    zero_axis_pad = torch.zeros_like(slice_index)
                    position_ids.append(torch.stack((slice_index, zero_axis_pad, zero_axis_pad), -1))
                    modality.append(
                        torch.full(
                            (segment_kept_length,), ModalityType.text.value, device=base_device, dtype=torch.long
                        )
                    )
                    input_ids.append(item["tok"].to(base_device)[segment_local_start:segment_local_end])
                    position_offset += segment_length
                else:
                    num_pos_slices, grid_height_tokens, grid_width_tokens = item["dims"]
                    hw = grid_height_tokens * grid_width_tokens
                    slice_index = (segment_local_indices // hw) + position_offset
                    rem = segment_local_indices % hw
                    row_index = rem // grid_width_tokens
                    col_index = rem % grid_width_tokens
                    position_ids.append(torch.stack((slice_index, row_index, col_index), -1))
                    modality.append(
                        torch.full(
                            (segment_kept_length,), ModalityType.image.value, device=base_device, dtype=torch.long
                        )
                    )
                    input_ids.append(
                        torch.full((segment_kept_length,), image_pad_value, device=base_device, dtype=torch.long)
                    )

                    vpatches.append(item["patches"].to(base_device))  # full patches; slice later via offsets/lengths
                    # Record per-image slice boundaries so we can drop cropped virtual tokens
                    # after pixel shuffle without re-packing the entire vision stream.
                    grids.append(item["grid"])
                    vision_token_offsets.append(segment_local_start)
                    vision_token_lengths.append(segment_kept_length)

                    position_offset += int(num_pos_slices)

            else:
                position_offset += segment_length if item["type"] == "text" else int(item["dims"][0])

            global_offset += segment_length

        if base_device is None:
            base_device = torch.device("cpu")

        modality_tensor = (
            torch.cat(modality, 0).unsqueeze(0)
            if modality
            else torch.zeros((1, 0), device=base_device, dtype=torch.long)
        )
        position_ids = (
            torch.cat(position_ids, 0).unsqueeze(0)
            if position_ids
            else torch.zeros((1, 0, 3), device=base_device, dtype=torch.long)
        )
        input_ids = (
            torch.cat(input_ids, 0).unsqueeze(0)
            if input_ids
            else torch.zeros((1, 0), device=base_device, dtype=torch.long)
        )

        if vpatches:
            vision_token_grids = torch.tensor(grids, device=base_device, dtype=torch.long)
            vision_token_offsets = torch.tensor(vision_token_offsets, device=base_device, dtype=torch.long)
            vision_token_lengths = torch.tensor(vision_token_lengths, device=base_device, dtype=torch.long)
        else:
            vision_token_grids = torch.zeros((0, 2), device=base_device, dtype=torch.long)
            vision_token_offsets = torch.zeros((0,), device=base_device, dtype=torch.long)
            vision_token_lengths = torch.zeros((0,), device=base_device, dtype=torch.long)

        return {
            "input_ids": input_ids.squeeze(0),
            "vision_patches": vpatches,
            "vision_token_grids": vision_token_grids,
            "vision_token_offsets": vision_token_offsets,
            "vision_token_lengths": vision_token_lengths,
            "modality_tensor": modality_tensor.squeeze(0),
            "position_ids": position_ids.squeeze(0),
        }

    def _collate_batch(
        self, per_sample: list[dict[str, torch.Tensor | list[torch.Tensor]]]
    ) -> dict[str, torch.Tensor | None]:
        lengths = [int(sample["input_ids"].shape[0]) for sample in per_sample]
        max_len = max(lengths, default=0)
        batch_size = len(per_sample)

        base_device = torch.device("cpu")
        for sample in per_sample:
            if sample["input_ids"].numel() > 0:
                base_device = sample["input_ids"].device
                break

        input_ids = torch.full((batch_size, max_len), self.text_pad_token_id, device=base_device, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), device=base_device, dtype=torch.long)
        modality_tensor = torch.full(
            (batch_size, max_len), ModalityType.text.value, device=base_device, dtype=torch.long
        )
        position_ids = torch.zeros((batch_size, max_len, 3), device=base_device, dtype=torch.long)

        for batch_idx, (sample, length) in enumerate(zip(per_sample, lengths)):
            if length == 0:
                continue
            input_ids[batch_idx, -length:] = sample["input_ids"]
            attention_mask[batch_idx, -length:] = 1
            modality_tensor[batch_idx, -length:] = sample["modality_tensor"]
            position_ids[batch_idx, -length:] = sample["position_ids"]

        image_counts = [len(sample["vision_patches"]) for sample in per_sample]
        max_images = max(image_counts, default=0)

        if max_images == 0:
            vision_patches = None
            vision_patch_attention_mask = None
            vision_token_grids = None
            vision_token_offsets = None
            vision_token_lengths = None
            vision_image_attention_mask = None
        else:
            first_patch = None
            for sample in per_sample:
                if len(sample["vision_patches"]) > 0:
                    first_patch = sample["vision_patches"][0]
                    break

            patch_dim = first_patch.shape[-1]
            patch_dtype = first_patch.dtype
            max_patches = max(
                (patches.shape[0] for sample in per_sample for patches in sample["vision_patches"]),
                default=0,
            )

            vision_patches = torch.zeros(
                (batch_size, max_images, max_patches, patch_dim),
                device=base_device,
                dtype=patch_dtype,
            )
            vision_patch_attention_mask = torch.zeros(
                (batch_size, max_images, max_patches),
                device=base_device,
                dtype=torch.long,
            )
            vision_token_grids = torch.zeros((batch_size, max_images, 2), device=base_device, dtype=torch.long)
            vision_token_offsets = torch.zeros((batch_size, max_images), device=base_device, dtype=torch.long)
            vision_token_lengths = torch.zeros((batch_size, max_images), device=base_device, dtype=torch.long)
            vision_image_attention_mask = torch.zeros((batch_size, max_images), device=base_device, dtype=torch.long)

            for batch_idx, sample in enumerate(per_sample):
                sample_patches = sample["vision_patches"]
                sample_image_count = len(sample_patches)
                if sample_image_count == 0:
                    continue

                sample_grids = sample["vision_token_grids"]
                sample_offsets = sample["vision_token_offsets"]
                sample_lengths = sample["vision_token_lengths"]

                vision_token_grids[batch_idx, :sample_image_count] = sample_grids
                vision_token_offsets[batch_idx, :sample_image_count] = sample_offsets
                vision_token_lengths[batch_idx, :sample_image_count] = sample_lengths
                vision_image_attention_mask[batch_idx, :sample_image_count] = 1

                for image_idx, patches in enumerate(sample_patches):
                    patch_count = int(patches.shape[0])
                    vision_patches[batch_idx, image_idx, :patch_count] = patches
                    vision_patch_attention_mask[batch_idx, image_idx, :patch_count] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "modality_tensor": modality_tensor,
            "vision_patches": vision_patches,
            "vision_patch_attention_mask": vision_patch_attention_mask,
            "vision_token_grids": vision_token_grids,
            "vision_token_offsets": vision_token_offsets,
            "vision_token_lengths": vision_token_lengths,
            "vision_image_attention_mask": vision_image_attention_mask,
        }

    def __call__(
        self,
        text: str | list[str],
        images: Image | list[Image] | None = None,
        return_tensors: str | TensorType | None = TensorType.PYTORCH,
        **kwargs,
    ) -> BatchFeature:
        texts = [text] if isinstance(text, str) else text
        images_list: list[list[Image] | None] | None = None
        if images is not None:
            if isinstance(images, list) and len(images) == len(texts):
                if not images:
                    images_list = []
                elif isinstance(images[0], list):
                    images_list = images  # already per-sample
                else:
                    images_list = [[img] for img in images]  # list of images, one per sample
            else:
                images_list = []
                for t in texts:
                    n_tok = t.count(self.vision_token)
                    if n_tok == 0:
                        images_list.append(None)
                    else:
                        if isinstance(images, list):
                            images_list.append(images)
                        else:
                            images_list.append([images])

        pairs = (
            ((text_value, None) for text_value in texts)
            if images_list is None
            else zip(texts, images_list, strict=True)
        )
        per_sample = [self._parse_sample(text_value, sample_images) for text_value, sample_images in pairs]
        return BatchFeature(data=self._collate_batch(per_sample), tensor_type=return_tensors)


class IsaacRotaryEmbedding(Qwen2_5_VLRotaryEmbedding):
    def __init__(self, config: IsaacConfig, device=None):
        rope_source_cfg = config.get_text_config() if hasattr(config, "get_text_config") else config
        rope_scaling = getattr(rope_source_cfg, "rope_scaling", None) or {}
        config_for_rope = copy.copy(rope_source_cfg)
        config_for_rope.rope_scaling = rope_scaling

        super().__init__(
            config_for_rope,
            device=device if device is not None and getattr(device, "type", None) != "meta" else None,
        )

        rotary_half_dim = self.inv_freq.shape[0]
        self.mrope_section = self._resolve_mrope_section(rope_scaling.get("mrope_section"), rotary_half_dim)
        self.hidden_size = getattr(rope_source_cfg, "hidden_size", None) or config.hidden_size

    @staticmethod
    def _resolve_mrope_section(section: list[int] | None, rotary_half_dim: int) -> list[int]:
        if section is None:
            weights = (2, 1, 1)
            base = [rotary_half_dim * w // sum(weights) for w in weights]
            base[0] += rotary_half_dim - sum(base)
            return base

        section = [int(v) for v in section]
        return section

    def _combine_axes(self, tensor: torch.Tensor) -> torch.Tensor:
        split_sections = tuple(self.mrope_section * 2)
        chunks = tensor.split(split_sections, dim=-1)
        return torch.cat([chunk[i % 3] for i, chunk in enumerate(chunks)], dim=-1)

    def forward(
        self,
        position_ids: torch.Tensor,
        modality_tensor: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states is None:
            batch, seq_len, _ = position_ids.shape
            hidden_states = torch.zeros(
                batch,
                seq_len,
                self.hidden_size,
                dtype=torch.float32,
                device=position_ids.device,
            )

        with torch.no_grad():
            pos = position_ids.clone()
            not_spatial = modality_tensor == 1
            data_1d = pos[not_spatial][..., 0].unsqueeze(-1)  # Collapse non-vision modalities to 1D positions
            pos[not_spatial] = data_1d.expand(-1, pos.shape[-1])
            pos_axes = pos.permute(2, 0, 1).contiguous()

        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, pos_axes.shape[1], -1, 1)
        pos_axes_expanded = pos_axes[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = (
            hidden_states.device.type
            if isinstance(hidden_states.device.type, str) and hidden_states.device.type != "mps"
            else "cpu"
        )
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ pos_axes_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        cos_axes, sin_axes = cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)
        cos_combined, sin_combined = self._combine_axes(cos_axes), self._combine_axes(sin_axes)

        return cos_combined, sin_combined


@auto_docstring
class IsaacModel(Qwen3PreTrainedModel):
    supports_gradient_checkpointing = True
    _can_compile_fullgraph = False
    _supports_flex_attn = False
    _can_record_outputs = {
        "hidden_states": OutputRecorder(Qwen3DecoderLayer),
        "attentions": Qwen3Attention,
        "vision_attentions": IsaacVisionAttention,
    }
    all_tied_weights_keys: dict[str, str] = {}

    def __init__(self, config: IsaacConfig):
        Qwen3PreTrainedModel.__init__(self, config)
        self.text_model = Qwen3Model._from_config(config.text_config)

        self.rotary_emb = IsaacRotaryEmbedding(config, device=self.device)

        self.vision_embedding = IsaacVisionEmbedding(config)
        self.max_sequence_length = config.max_sequence_length
        self.vision_rescale_factor = config.vision_rescale_factor
        self.vision_token = config.vision_token
        self.rope_deltas = None

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.text_model.set_input_embeddings(value)
        vocab_size = getattr(value, "num_embeddings", None)
        if vocab_size is not None:
            self.config.vocab_size = vocab_size
            if hasattr(self.config, "text_config"):
                self.config.text_config.vocab_size = vocab_size
            self.text_model.config.vocab_size = vocab_size

    @property
    def final_norm(self) -> nn.Module:
        return self.text_model.norm

    @property
    def embed_tokens(self) -> nn.Module:
        return self.text_model.embed_tokens

    @embed_tokens.setter
    def embed_tokens(self, value: nn.Module) -> None:
        self.text_model.embed_tokens = value

    def embed_multimodal_inputs(
        self,
        input_ids: torch.Tensor,
        modality_tensor: torch.Tensor,
        vision_patches: torch.Tensor | None = None,
        vision_patch_attention_mask: torch.Tensor | None = None,
        vision_token_grids: torch.Tensor | None = None,
        vision_token_offsets: torch.Tensor | None = None,
        vision_token_lengths: torch.Tensor | None = None,
        vision_image_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        modality = modality_tensor.to(device=input_ids.device, dtype=torch.long)
        embeds = self.text_model.embed_tokens(input_ids)
        image_token_mask = modality == ModalityType.image.value

        if vision_patches is None or vision_token_grids is None:
            if torch.any(image_token_mask):
                raise ValueError("Image placeholders require `vision_patches` and `vision_token_grids`.")
            return embeds, modality

        vision_patches = vision_patches.to(device=embeds.device)
        token_grids = vision_token_grids.to(device=embeds.device, dtype=torch.long)
        image_attention_mask = (
            vision_image_attention_mask.to(device=embeds.device, dtype=torch.bool)
            if vision_image_attention_mask is not None
            else torch.ones(token_grids.shape[:2], device=embeds.device, dtype=torch.bool)
        )
        patch_attention_mask = (
            vision_patch_attention_mask.to(device=embeds.device, dtype=torch.long)
            if vision_patch_attention_mask is not None
            else torch.ones(vision_patches.shape[:3], device=embeds.device, dtype=torch.long)
        )
        offsets = (
            vision_token_offsets.to(device=embeds.device, dtype=torch.long)
            if vision_token_offsets is not None
            else torch.zeros(token_grids.shape[:2], device=embeds.device, dtype=torch.long)
        )
        reduction_factor = int(self.config.vision_config.pixel_shuffle_scale_factor) ** 2
        lengths = (
            vision_token_lengths.to(device=embeds.device, dtype=torch.long)
            if vision_token_lengths is not None
            else token_grids.prod(-1).div(reduction_factor, rounding_mode="floor").to(dtype=torch.long)
        )

        valid_images = image_attention_mask.bool()
        if not torch.any(valid_images):
            if torch.any(image_token_mask):
                raise ValueError("Image placeholders are present but no valid images were provided.")
            return embeds, modality

        flat_vision_patches = vision_patches[valid_images]
        flat_patch_attention_mask = patch_attention_mask[valid_images]
        flat_token_grids = token_grids[valid_images]
        flat_offsets = offsets[valid_images]
        flat_lengths = lengths[valid_images]

        vision_embeddings, _, per_image_lengths = self.vision_embedding(
            (flat_vision_patches, flat_token_grids, flat_patch_attention_mask)
        )

        if torch.any(flat_offsets < 0) or torch.any(flat_lengths < 0):
            raise ValueError("`vision_token_offsets` and `vision_token_lengths` must be non-negative.")
        if torch.any(flat_offsets + flat_lengths > per_image_lengths):
            raise ValueError("Requested vision token slices exceed available per-image vision embeddings.")

        max_chunk_length = int(flat_lengths.max().item()) if flat_lengths.numel() > 0 else 0
        if max_chunk_length == 0:
            if torch.any(image_token_mask):
                raise ValueError("Image placeholders are present but all selected vision chunks are empty.")
            return embeds, modality

        token_positions = torch.arange(max_chunk_length, device=embeds.device, dtype=torch.long)
        gather_positions = flat_offsets[:, None] + token_positions[None, :]
        gather_mask = token_positions[None, :] < flat_lengths[:, None]
        image_features = vision_embeddings[
            torch.arange(vision_embeddings.shape[0], device=embeds.device, dtype=torch.long)[:, None],
            gather_positions,
        ][gather_mask]
        image_features = image_features.to(device=embeds.device, dtype=embeds.dtype)

        expected_image_tokens = int(image_token_mask.sum().item())
        if image_features.shape[0] != expected_image_tokens:
            raise ValueError(
                f"Image features and image placeholders do not match: tokens={expected_image_tokens}, features={image_features.shape[0]}."
            )

        scatter_mask = image_token_mask.unsqueeze(-1).expand_as(embeds)
        embeds = embeds.masked_scatter(scatter_mask, image_features)

        return embeds, modality

    def get_rope_index(
        self,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare multimodal RoPE positions and carry forward per-batch offsets.

        Unlike vanilla 1D RoPE, Isaac builds 3-axis indices for text and vision tokens.
        If callers do not supply positions, we synthesize them from `cache_position` and
        use `attention_mask` to strip left padding so pad tokens never consume RoPE slots.
        The returned `rope_deltas` capture any custom offset (i.e., prefill length) and
        are reused across generation steps so newly decoded tokens keep counting forward
        after the cached prefix."""

        device = inputs_embeds.device
        batch_size, seq_len = inputs_embeds.shape[:2]

        if position_ids is None:
            cp = cache_position.to(device=device, dtype=torch.long)
            if cp.ndim == 1:
                cp = cp.view(1, -1).expand(batch_size or 1, -1)

            is_new_prefill = bool(torch.all(cp[:, :1] == 0))
            base_delta = torch.as_tensor(
                0 if is_new_prefill or self.rope_deltas is None else self.rope_deltas,
                device=device,
                dtype=torch.long,
            ).reshape(-1, 1)
            base_delta = torch.broadcast_to(base_delta, (batch_size, 1))

            mask_delta = attention_mask.to(device=device, dtype=torch.long).sum(1, keepdim=True) - attention_mask.size(
                1
            )
            rope_position = cp + base_delta + mask_delta
            pos_3d = rope_position.unsqueeze(-1).expand(-1, -1, 3)
            return pos_3d, base_delta

        position_ids = position_ids.to(device=device)
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(-1).expand(-1, -1, 3)

        if position_ids.shape[1] != seq_len:
            start_positions = position_ids[:, :1, 0]
            position_ids = torch.arange(seq_len, device=position_ids.device).view(1, -1) + start_positions
            position_ids = position_ids.unsqueeze(-1).expand(-1, -1, 3)

        attn = attention_mask.to(device=device, dtype=torch.long)
        m_per_batch = position_ids.amax(dim=(1, 2))
        seq_lens = attn.eq(1).sum(dim=-1).to(dtype=m_per_batch.dtype, device=device)
        rope_deltas = (m_per_batch + 1 - seq_lens).to(dtype=position_ids.dtype).unsqueeze(1)
        return position_ids, rope_deltas

    @auto_docstring
    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        modality_tensor: torch.LongTensor | None = None,
        vision_patches: torch.Tensor | None = None,
        vision_patch_attention_mask: torch.Tensor | None = None,
        vision_token_grids: torch.LongTensor | None = None,
        vision_token_offsets: torch.LongTensor | None = None,
        vision_token_lengths: torch.LongTensor | None = None,
        vision_image_attention_mask: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        """
        Forward pass with MRoPE position embeddings.

        Computes position embeddings once and passes them through all layers.

        Args:
            modality_tensor (`torch.LongTensor`, *optional*):
                Modality identifiers aligned with the embedded sequence, shaped `(batch_size, seq_len)` and containing
                values from `ModalityType`. Treated as text-only when omitted.
            vision_patches (`torch.FloatTensor`, *optional*):
                Padded per-image patch vectors of shape `(batch_size, max_images, max_patches, patch_dim)`.
            vision_patch_attention_mask (`torch.LongTensor`, *optional*):
                Mask for valid patch entries in `vision_patches`, shaped `(batch_size, max_images, max_patches)`.
            vision_token_grids (`torch.LongTensor`, *optional*):
                Per-image patch grids `(h, w)` with shape `(batch_size, max_images, 2)`.
            vision_token_offsets (`torch.LongTensor`, *optional*):
                Start offsets inside the per-image vision embedding sequence, shape `(batch_size, max_images)`.
            vision_token_lengths (`torch.LongTensor`, *optional*):
                Number of vision tokens to consume per image, shape `(batch_size, max_images)`.
            vision_image_attention_mask (`torch.LongTensor`, *optional*):
                Mask indicating which image slots are populated, shape `(batch_size, max_images)`.
        """

        if inputs_embeds is None and input_ids is None:
            raise ValueError("`input_ids` or `inputs_embeds` must be provided.")

        if inputs_embeds is None and input_ids is not None:
            if modality_tensor is not None or vision_patches is not None:
                if modality_tensor is None:
                    modality_tensor = torch.full_like(input_ids, ModalityType.text.value)
                inputs_embeds, modality_tensor = self.embed_multimodal_inputs(
                    input_ids=input_ids,
                    modality_tensor=modality_tensor,
                    vision_patches=vision_patches,
                    vision_patch_attention_mask=vision_patch_attention_mask,
                    vision_token_grids=vision_token_grids,
                    vision_token_offsets=vision_token_offsets,
                    vision_token_lengths=vision_token_lengths,
                    vision_image_attention_mask=vision_image_attention_mask,
                )
            else:
                inputs_embeds = self.text_model.embed_tokens(input_ids)
        elif inputs_embeds is not None and modality_tensor is None:
            batch_size, seq_len = inputs_embeds.shape[:2]
            modality_tensor = torch.full(
                (batch_size, seq_len), ModalityType.text.value, device=inputs_embeds.device, dtype=torch.long
            )

        device = inputs_embeds.device
        batch_size, seq_len = inputs_embeds.shape[:2]

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config.get_text_config())

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_len, device=device)

        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long)

        position_ids, rope_deltas = self.get_rope_index(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
        )
        self.rope_deltas = rope_deltas

        if modality_tensor is None:
            modality_tensor = torch.full(
                (batch_size, seq_len), ModalityType.text.value, device=device, dtype=torch.long
            )

        cos, sin = self.rotary_emb(position_ids, modality_tensor, hidden_states=inputs_embeds)

        decoder_position_ids = position_ids[..., 0] if position_ids.ndim == 3 else position_ids

        if not isinstance(attention_mask, dict):
            attention_mask = create_masks_for_generate(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=decoder_position_ids,
            )

        is_mask_dict = isinstance(attention_mask, dict)
        hidden_states = inputs_embeds

        for layer in self.text_model.layers:
            layer_mask = attention_mask[layer.attention_type] if is_mask_dict else attention_mask
            layer_outputs = layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=(cos, sin),
                **kwargs,
            )

            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        hidden_states = self.final_norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class IsaacForConditionalGeneration(Qwen3ForCausalLM, GenerationMixin):
    config_class = IsaacConfig
    _can_compile_fullgraph = False
    _tied_weights_keys = {"lm_head.weight": "model.text_model.embed_tokens.weight"}
    all_tied_weights_keys: dict[str, str] = {"lm_head.weight": "model.text_model.embed_tokens.weight"}

    def __init__(self, config: IsaacConfig):
        super().__init__(config)
        self.model = IsaacModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @auto_docstring
    @can_return_tuple
    @merge_with_config_defaults
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        modality_tensor: torch.LongTensor | None = None,
        vision_patches: torch.Tensor | None = None,
        vision_patch_attention_mask: torch.Tensor | None = None,
        vision_token_grids: torch.LongTensor | None = None,
        vision_token_offsets: torch.LongTensor | None = None,
        vision_token_lengths: torch.LongTensor | None = None,
        vision_image_attention_mask: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        modality_tensor (`torch.LongTensor`, *optional*):
            Modality identifiers aligned with the token sequence, shaped `(batch_size, seq_len)`.
        vision_patches (`torch.FloatTensor`, *optional*):
            Padded per-image patch vectors of shape `(batch_size, max_images, max_patches, patch_dim)`.
        vision_patch_attention_mask (`torch.LongTensor`, *optional*):
            Mask for valid patch entries in `vision_patches`, shaped `(batch_size, max_images, max_patches)`.
        vision_token_grids (`torch.LongTensor`, *optional*):
            Per-image patch grids `(h, w)` with shape `(batch_size, max_images, 2)`.
        vision_token_offsets (`torch.LongTensor`, *optional*):
            Start offsets inside the per-image vision embedding sequence, shape `(batch_size, max_images)`.
        vision_token_lengths (`torch.LongTensor`, *optional*):
            Number of vision tokens to consume per image, shape `(batch_size, max_images)`.
        vision_image_attention_mask (`torch.LongTensor`, *optional*):
            Mask indicating which image slots are populated, shape `(batch_size, max_images)`.
        """
        outputs = self.model(
            input_ids=input_ids,
            modality_tensor=modality_tensor,
            vision_patches=vision_patches,
            vision_patch_attention_mask=vision_patch_attention_mask,
            vision_token_grids=vision_token_grids,
            vision_token_offsets=vision_token_offsets,
            vision_token_lengths=vision_token_lengths,
            vision_image_attention_mask=vision_image_attention_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: list[torch.FloatTensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        modality_tensor: torch.LongTensor | None = None,
        vision_patches: torch.Tensor | None = None,
        vision_patch_attention_mask: torch.Tensor | None = None,
        vision_token_grids: torch.LongTensor | None = None,
        vision_token_offsets: torch.LongTensor | None = None,
        vision_token_lengths: torch.LongTensor | None = None,
        vision_image_attention_mask: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=None,
            **kwargs,
        )
        has_multimodal = (
            modality_tensor is not None
            or vision_patches is not None
            or vision_patch_attention_mask is not None
            or vision_token_grids is not None
            or vision_token_offsets is not None
            or vision_token_lengths is not None
            or vision_image_attention_mask is not None
        )
        if not has_multimodal:
            return model_inputs

        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        first_step = past_len == 0
        model_inputs["modality_tensor"] = modality_tensor if first_step else None
        model_inputs["vision_patches"] = vision_patches if first_step else None
        model_inputs["vision_patch_attention_mask"] = vision_patch_attention_mask if first_step else None
        model_inputs["vision_token_grids"] = vision_token_grids if first_step else None
        model_inputs["vision_token_offsets"] = vision_token_offsets if first_step else None
        model_inputs["vision_token_lengths"] = vision_token_lengths if first_step else None
        model_inputs["vision_image_attention_mask"] = vision_image_attention_mask if first_step else None
        model_inputs["position_ids"] = position_ids if first_step else None

        return model_inputs

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.set_input_embeddings(value)
        vocab_size = getattr(value, "num_embeddings", None)
        self.config.vocab_size = vocab_size
        self.model.config.vocab_size = vocab_size
        self.model.text_model.config.vocab_size = vocab_size
        if self.lm_head.weight.shape[0] != vocab_size:
            self.lm_head = nn.Linear(self.config.hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.model.text_model.embed_tokens.weight


__all__ = [
    "IsaacConfig",
    "IsaacModel",
    "IsaacPreTrainedModel",  # noqa: F822
    "IsaacForConditionalGeneration",
    "IsaacImageProcessorFast",
    "IsaacProcessor",
]
