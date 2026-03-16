# Copyright 2026 Perceptron, Inc and The HuggingFace Team. All rights reserved.
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
from collections.abc import Sequence
from typing import Any

from ... import initialization as init
from ...configuration_utils import PretrainedConfig
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
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...models.qwen3.configuration_qwen3 import Qwen3Config
from ...models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3PreTrainedModel,
)
from ...processing_utils import ProcessorMixin, Unpack
from ...utils import TensorType, auto_docstring, torch_compilable_check
from ...utils.constants import IMAGENET_STANDARD_MEAN as VISION_MEAN
from ...utils.constants import IMAGENET_STANDARD_STD as VISION_STD
from ...utils.generic import TransformersKwargs, can_return_tuple, merge_with_config_defaults
from ...utils.import_utils import (
    is_torch_available,
    is_torchdynamo_compiling,
    is_torchvision_available,
    is_vision_available,
)
from ...utils.output_capturing import capture_outputs
from ..qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    Qwen3VLTextRotaryEmbedding,
)
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


MM_TOKEN_TYPE_TEXT = 0
MM_TOKEN_TYPE_IMAGE = 1


class IsaacVisionConfig(Siglip2VisionConfig):
    """Vision configuration for Isaac with Pixel Shuffle support.

    Extends Siglip2VisionConfig with additional fields for pixel shuffle.

    Args:
        pixel_shuffle_scale_factor (`int`, *optional*, defaults to 1):
            Spatial factor applied before pixel shuffle reduces the resolution.
    """

    model_type = "isaac_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        pixel_shuffle_scale_factor=1,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        # Add our custom fields
        self.pixel_shuffle_scale_factor = pixel_shuffle_scale_factor


class IsaacTextConfig(Qwen3Config):
    model_type = "isaac_text"

    def __init__(self, **super_kwargs):
        super().__init__(ignore_keys_at_rope_validation={"mrope_section", "mrope_interleaved"}, **super_kwargs)


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

    patch_size: int
    max_num_patches: int
    min_num_patches: int
    pixel_shuffle_scale: int


@auto_docstring
class IsaacImageProcessorFast(BaseImageProcessorFast):
    MAX_PIXELS = 60_000_000  # 60‑megapixel ceiling ≈ 8200 × 7300 px

    resample = PILImageResampling.BILINEAR
    model_input_names = ["patches", "token_grids", "mm_token_type_ids"]
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
        return F.interpolate(image, size=(size.height, size.width), mode="bilinear", align_corners=False)

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
        self.position_embedding = nn.Parameter(
            torch.empty(
                self.position_embedding_size,
                self.position_embedding_size,
                self.embed_dim,
            )
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # pixel_values: (num_images, max_patches, patch_dim)
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

    pass


class IsaacVisionEncoderLayer(Siglip2EncoderLayer):
    """Isaac vision encoder layer using the shared attention interfaces."""

    def __init__(self, config: IsaacVisionConfig):
        super().__init__(config)
        self.self_attn = IsaacVisionAttention(config)


class IsaacVisionEncoder(Siglip2Encoder):
    """Encoder using Isaac encoder layers."""

    def __init__(self, config: IsaacVisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([IsaacVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])


def pixel_shuffle_padded(
    x: torch.Tensor,
    token_grids: torch.Tensor,
    scale_factor: int = 1,
) -> torch.Tensor:
    """Apply pixel shuffle per image on padded batched vision embeddings.

    Args:
        x (`torch.Tensor`):
            Vision embeddings of shape `(num_images, max_patches, hidden_size)`.
        token_grids (`torch.Tensor`):
            Grid sizes `(height, width)` per image, shape `(num_images, 2)`.
        scale_factor (`int`, *optional*, defaults to 1):
            Spatial down-sampling factor.

    Returns:
        Tuple of:
            - pixel-shuffled embeddings `(num_images, max_tokens, hidden_size * scale_factor**2)`
            - attention mask `(num_images, max_tokens)`
            - per-image valid token lengths `(num_images,)`
    """
    num_images, max_patches, embed_dim = x.shape
    output_dim = embed_dim * scale_factor * scale_factor

    token_grids = token_grids.to(device=x.device, dtype=torch.long)
    heights = token_grids[:, 0]
    widths = token_grids[:, 1]
    full_lengths = heights * widths

    non_empty = full_lengths > 0
    if not is_torchdynamo_compiling():
        divisible = ((heights % scale_factor) == 0) & ((widths % scale_factor) == 0)
        torch_compilable_check(
            (~non_empty) | divisible,
            f"Every non-empty (H, W) grid must be divisible by pixel_shuffle_scale={scale_factor}.",
        )

    output_lengths = (heights // scale_factor) * (widths // scale_factor)
    max_output_tokens = output_lengths.max()
    shuffled_4d = x.new_zeros((num_images, max_output_tokens, scale_factor * scale_factor, embed_dim))

    token_positions = torch.arange(max_patches, device=x.device, dtype=torch.long).unsqueeze(0).expand(num_images, -1)
    valid_token_mask = token_positions < full_lengths.unsqueeze(1)

    safe_widths = torch.where(widths > 0, widths, torch.ones_like(widths))
    row_index = torch.div(token_positions, safe_widths.unsqueeze(1), rounding_mode="floor")
    col_index = token_positions.remainder(safe_widths.unsqueeze(1))

    output_widths = widths.div(scale_factor, rounding_mode="floor")
    output_index = row_index.div(scale_factor, rounding_mode="floor") * output_widths.unsqueeze(1)
    output_index = output_index + col_index.div(scale_factor, rounding_mode="floor")
    sub_index = row_index.remainder(scale_factor) * scale_factor + col_index.remainder(scale_factor)

    batch_index = torch.arange(num_images, device=x.device, dtype=torch.long).unsqueeze(1).expand_as(token_positions)
    shuffled_4d[batch_index[valid_token_mask], output_index[valid_token_mask], sub_index[valid_token_mask]] = x[
        valid_token_mask
    ]

    shuffled = shuffled_4d.view(num_images, max_output_tokens, output_dim)
    return shuffled


class IsaacVisionTransformer(PreTrainedModel):
    """Vision tower for padded variable-resolution patches with per-image masks.

    Args:
        config (IsaacVisionConfig): Vision configuration with pixel-shuffle and patching parameters.

    """

    _supports_sdpa = True
    _supports_flash_attn = True
    _can_record_outputs = {
        "hidden_states": IsaacVisionEncoderLayer,
        "attentions": IsaacVisionAttention,
    }

    def __init__(self, config: IsaacVisionConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = IsaacVisionEmbeddings(config)
        self.encoder = IsaacVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pixel_shuffle_scale_factor = config.pixel_shuffle_scale_factor

        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, IsaacVisionEmbeddings):
            init.zeros_(module.position_embedding)

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    def forward(
        self,
        vision_patches: torch.Tensor,
        vision_token_grids: torch.Tensor,
        vision_patch_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        """
        Inputs:
            vision_patches (`torch.Tensor`):
                Patches shaped `(num_images, max_patches, patch_dim)`.
            vision_token_grids (`torch.Tensor`):
                Token grids shaped `(num_images, 2)` with per-image `(H_tokens, W_tokens)`.
            vision_patch_attention_mask (`torch.Tensor`, *optional*):
                Patch mask shaped `(num_images, max_patches)`.

        Returns:
            `BaseModelOutputWithPooling` with pixel-shuffled embeddings in `last_hidden_state`.
        """
        hidden_states = self.embeddings(
            vision_patches,
            vision_token_grids,
            attention_mask=vision_patch_attention_mask,
        )

        encoder_attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=vision_patch_attention_mask,
        )
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, attention_mask=encoder_attention_mask, **kwargs)
        hidden_states = self.post_layernorm(encoder_outputs.last_hidden_state)

        hidden_states = pixel_shuffle_padded(
            x=hidden_states,
            token_grids=vision_token_grids,
            scale_factor=self.pixel_shuffle_scale_factor,
        )

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class IsaacMultiModalProjector(nn.Module):
    """Maps vision tower outputs to the text hidden size with a SiLU MLP."""

    def __init__(self, config: IsaacConfig):
        super().__init__()
        text_config = config.get_text_config()
        vision_hidden_size = config.vision_config.hidden_size * (config.vision_config.pixel_shuffle_scale_factor**2)
        backbone_hidden_size = text_config.hidden_size
        self.linear_1 = nn.Linear(vision_hidden_size, 4 * vision_hidden_size, bias=False)
        self.silu = nn.SiLU()
        self.linear_2 = nn.Linear(4 * vision_hidden_size, backbone_hidden_size, bias=False)

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
        vision_patches: torch.Tensor,
        vision_token_grids: torch.Tensor,
        vision_patch_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vision_outputs = self.vision_tower(
            vision_patches=vision_patches,
            vision_token_grids=vision_token_grids,
            vision_patch_attention_mask=vision_patch_attention_mask,
            return_dict=True,
        )
        projected = self.multimodal_projector(vision_outputs.last_hidden_state)
        return projected


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
    sub_configs = {"vision_config": IsaacVisionConfig, "text_config": IsaacTextConfig}

    def __init__(
        self,
        vision_config: IsaacVisionConfig | None = None,
        text_config: IsaacTextConfig | dict | None = None,
        vision_rescale_factor: float = 1 / 255,
        max_sequence_length: int = 16384,
        vision_token: str = "<image>",
        **kwargs,
    ):
        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif isinstance(text_config, IsaacTextConfig):
            self.text_config = text_config
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif isinstance(vision_config, IsaacVisionConfig):
            self.vision_config = vision_config
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        super().__init__(**kwargs)

        # Mirror frequently accessed composite-level attributes.
        self.use_cache = self.text_config.use_cache
        self.rope_theta = self.text_config.rope_parameters["rope_theta"]
        self.max_position_embeddings = getattr(self.text_config, "max_position_embeddings", max_sequence_length)
        self.text_config.max_position_embeddings = self.max_position_embeddings

        # Vision normalization parameters
        self.vision_rescale_factor = float(vision_rescale_factor)

        # Processing parameters
        self.max_sequence_length = max_sequence_length
        self.vision_token = vision_token


@auto_docstring
class IsaacProcessor(ProcessorMixin):
    """Processor that pairs the Isaac image processor with the Qwen2 tokenizer."""

    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template: str | dict[str, str] | None = None,
        vision_token: str = "<image>",
        max_sequence_length: int = 16384,
        rescale_factor: float | None = None,
    ) -> None:
        """
        Args:
            image_processor:
                Image processor used to convert images into Isaac patch tensors.
            tokenizer:
                Tokenizer used for the text side of the multimodal prompt.
            chat_template (`str` or `dict[str, str]`, *optional*):
                Chat template override forwarded to [`~processing_utils.ProcessorMixin`].
            vision_token (`str`, *optional*, defaults to `"<image>"`):
                Placeholder token used inside text prompts to mark image positions.
            max_sequence_length (`int`, *optional*, defaults to 16384):
                Maximum packed multimodal sequence length produced by the processor.
            rescale_factor (`float`, *optional*):
                Deprecated compatibility argument accepted for backward compatibility.
        """
        if chat_template is None:
            chat_template = getattr(tokenizer, "chat_template", None)

        self.image_processor = image_processor
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        text_pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        image_pad_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        self.text_pad_token_id = int(text_pad_token_id)
        self.image_pad_token_id = int(image_pad_token_id)
        self.pad_token_id = self.text_pad_token_id

        self.vision_token = vision_token
        self.max_sequence_length = max_sequence_length

    def _build_batch(
        self,
        text: str | list[str],
        images: Image | list[Image] | None = None,
    ) -> dict[str, torch.Tensor | None]:
        texts = [text] if isinstance(text, str) else text
        if images is None:
            pairs = ((text_value, None) for text_value in texts)
        elif isinstance(images, list) and len(images) == len(texts):
            if not images:
                images_list = []
            elif isinstance(images[0], list):
                images_list = images
            else:
                images_list = [[image] for image in images]
            pairs = zip(texts, images_list, strict=True)
        else:
            pairs = (
                (
                    text_value,
                    None
                    if text_value.count(self.vision_token) == 0
                    else images
                    if isinstance(images, list)
                    else [images],
                )
                for text_value in texts
            )

        sample_input_ids: list[torch.Tensor] = []
        sample_mm_token_type_ids: list[torch.Tensor] = []
        sample_position_ids: list[torch.Tensor] = []
        sample_vision_patches: list[list[torch.Tensor]] = []
        sample_vision_grids: list[torch.Tensor] = []
        sample_vision_offsets: list[torch.Tensor] = []
        sample_vision_lengths: list[torch.Tensor] = []

        for text_value, sample_images in pairs:
            segments = text_value.split(self.vision_token)
            num_images = len(segments) - 1
            num_provided_images = len(sample_images) if sample_images is not None else 0
            if num_images != num_provided_images:
                raise ValueError(
                    f"IsaacProcessor expects one image per image token, got {num_images} tokens and {num_provided_images} images in sample with text {text_value} "
                )

            items: list[dict[str, Any]] = []
            total = 0
            for index, segment in enumerate(segments):
                if segment:
                    text_tokens = (
                        self.tokenizer.encode(segment, add_special_tokens=False, return_tensors="pt")
                        .squeeze(0)
                        .to(torch.long)
                    )
                    segment_length = int(text_tokens.numel())
                    items.append({"type": "text", "segment_length": segment_length, "tokens": text_tokens})
                    total += segment_length

                if index < num_images:
                    feature = self.image_processor(images=sample_images[index], return_tensors=TensorType.PYTORCH)
                    patches = feature["patches"][0].reshape(-1, feature["patches"].shape[-1])
                    virtual_pixel_size = feature["virtual_pixel_size"][0].to(torch.long).tolist()
                    real_pixel_size = feature["real_pixel_size"][0].to(torch.long).tolist()
                    dims = tuple((virtual_pixel_size + [1, 1, 1])[:3])
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

            start = max(0, total - self.max_sequence_length)
            end = total
            base_device: torch.device | None = None
            input_ids_chunks, mm_token_type_chunks, position_chunks = [], [], []
            vision_patches, vision_grids, vision_offsets, vision_lengths = [], [], [], []
            global_offset = 0
            position_offset = 0

            for item in items:
                segment_length = int(item["segment_length"])
                current_window_start = max(start, global_offset)
                current_window_end = min(end, global_offset + segment_length)
                has_overlap = current_window_end > current_window_start

                if has_overlap and base_device is None:
                    base_device = item["patches"].device if item["type"] == "image" else item["tokens"].device

                if has_overlap:
                    segment_local_start = int(current_window_start - global_offset)
                    segment_local_end = int(current_window_end - global_offset)
                    segment_local_indices = torch.arange(
                        segment_local_start, segment_local_end, device=base_device, dtype=torch.long
                    )
                    segment_kept_length = segment_local_end - segment_local_start

                    if item["type"] == "text":
                        slice_index = segment_local_indices + position_offset
                        zero_axis = torch.zeros_like(slice_index)
                        position_chunks.append(torch.stack((slice_index, zero_axis, zero_axis), -1))
                        mm_token_type_chunks.append(
                            torch.full(
                                (segment_kept_length,), MM_TOKEN_TYPE_TEXT, device=base_device, dtype=torch.long
                            )
                        )
                        input_ids_chunks.append(item["tokens"].to(base_device)[segment_local_start:segment_local_end])
                        position_offset += segment_length
                    else:
                        num_pos_slices, grid_height_tokens, grid_width_tokens = item["dims"]
                        hw = grid_height_tokens * grid_width_tokens
                        slice_index = (segment_local_indices // hw) + position_offset
                        rem = segment_local_indices % hw
                        position_chunks.append(
                            torch.stack((slice_index, rem // grid_width_tokens, rem % grid_width_tokens), -1)
                        )
                        mm_token_type_chunks.append(
                            torch.full(
                                (segment_kept_length,), MM_TOKEN_TYPE_IMAGE, device=base_device, dtype=torch.long
                            )
                        )
                        input_ids_chunks.append(
                            torch.full(
                                (segment_kept_length,), self.image_pad_token_id, device=base_device, dtype=torch.long
                            )
                        )

                        vision_patches.append(item["patches"].to(base_device))
                        vision_grids.append(item["grid"])
                        vision_offsets.append(segment_local_start)
                        vision_lengths.append(segment_kept_length)
                        position_offset += int(num_pos_slices)
                else:
                    position_offset += segment_length if item["type"] == "text" else int(item["dims"][0])

                global_offset += segment_length

            if base_device is None:
                base_device = torch.device("cpu")

            sample_input_ids.append(
                torch.cat(input_ids_chunks, 0)
                if input_ids_chunks
                else torch.zeros((0,), device=base_device, dtype=torch.long)
            )
            sample_mm_token_type_ids.append(
                torch.cat(mm_token_type_chunks, 0)
                if mm_token_type_chunks
                else torch.zeros((0,), device=base_device, dtype=torch.long)
            )
            sample_position_ids.append(
                torch.cat(position_chunks, 0)
                if position_chunks
                else torch.zeros((0, 3), device=base_device, dtype=torch.long)
            )
            sample_vision_patches.append(vision_patches)
            if vision_patches:
                sample_vision_grids.append(torch.tensor(vision_grids, device=base_device, dtype=torch.long))
                sample_vision_offsets.append(torch.tensor(vision_offsets, device=base_device, dtype=torch.long))
                sample_vision_lengths.append(torch.tensor(vision_lengths, device=base_device, dtype=torch.long))
            else:
                sample_vision_grids.append(torch.zeros((0, 2), device=base_device, dtype=torch.long))
                sample_vision_offsets.append(torch.zeros((0,), device=base_device, dtype=torch.long))
                sample_vision_lengths.append(torch.zeros((0,), device=base_device, dtype=torch.long))

        batch_size = len(sample_input_ids)
        lengths = [int(sample_input.shape[0]) for sample_input in sample_input_ids]
        max_len = max(lengths, default=0)
        base_device = next(
            (sample_input.device for sample_input in sample_input_ids if sample_input.numel() > 0),
            torch.device("cpu"),
        )

        input_ids = torch.full((batch_size, max_len), self.text_pad_token_id, device=base_device, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), device=base_device, dtype=torch.long)
        mm_token_type_ids = torch.full((batch_size, max_len), MM_TOKEN_TYPE_TEXT, device=base_device, dtype=torch.long)
        position_ids = torch.zeros((batch_size, max_len, 3), device=base_device, dtype=torch.long)

        for batch_idx, length in enumerate(lengths):
            if length == 0:
                continue
            input_ids[batch_idx, -length:] = sample_input_ids[batch_idx]
            attention_mask[batch_idx, -length:] = 1
            mm_token_type_ids[batch_idx, -length:] = sample_mm_token_type_ids[batch_idx]
            position_ids[batch_idx, -length:] = sample_position_ids[batch_idx]

        image_counts = [len(patches) for patches in sample_vision_patches]
        max_images = max(image_counts, default=0)
        if max_images == 0:
            vision_patches = None
            vision_patch_attention_mask = None
            vision_token_grids = None
            vision_token_offsets = None
            vision_token_lengths = None
            vision_image_attention_mask = None
        else:
            first_patch = next((patches[0] for patches in sample_vision_patches if patches), None)
            patch_dim = first_patch.shape[-1]
            patch_dtype = first_patch.dtype
            max_patches = max((patch.shape[0] for patches in sample_vision_patches for patch in patches), default=0)

            vision_patches = torch.zeros(
                (batch_size, max_images, max_patches, patch_dim), device=base_device, dtype=patch_dtype
            )
            vision_patch_attention_mask = torch.zeros(
                (batch_size, max_images, max_patches), device=base_device, dtype=torch.long
            )
            vision_token_grids = torch.zeros((batch_size, max_images, 2), device=base_device, dtype=torch.long)
            vision_token_offsets = torch.zeros((batch_size, max_images), device=base_device, dtype=torch.long)
            vision_token_lengths = torch.zeros((batch_size, max_images), device=base_device, dtype=torch.long)
            vision_image_attention_mask = torch.zeros((batch_size, max_images), device=base_device, dtype=torch.long)

            for batch_idx, sample_patches in enumerate(sample_vision_patches):
                sample_image_count = len(sample_patches)
                if sample_image_count == 0:
                    continue
                vision_token_grids[batch_idx, :sample_image_count] = sample_vision_grids[batch_idx]
                vision_token_offsets[batch_idx, :sample_image_count] = sample_vision_offsets[batch_idx]
                vision_token_lengths[batch_idx, :sample_image_count] = sample_vision_lengths[batch_idx]
                vision_image_attention_mask[batch_idx, :sample_image_count] = 1

                for image_idx, patches in enumerate(sample_patches):
                    patch_count = int(patches.shape[0])
                    vision_patches[batch_idx, image_idx, :patch_count] = patches
                    vision_patch_attention_mask[batch_idx, image_idx, :patch_count] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "mm_token_type_ids": mm_token_type_ids,
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
        return BatchFeature(data=self._build_batch(text=text, images=images), tensor_type=return_tensors)


class IsaacRotaryEmbedding(Qwen3VLTextRotaryEmbedding):
    def __init__(self, config: IsaacConfig | IsaacTextConfig, device=None):
        rope_source_cfg = config.get_text_config()
        config_for_rope = copy.copy(rope_source_cfg)
        rope_scaling = getattr(rope_source_cfg, "rope_scaling", None) or {}
        config_for_rope.rope_scaling = rope_scaling

        super().__init__(
            config_for_rope,
            device=device if device is not None and getattr(device, "type", None) != "meta" else None,
        )

        self.mrope_section = self._resolve_mrope_section(rope_scaling.get("mrope_section"), self.inv_freq.shape[0])
        self.hidden_size = getattr(rope_source_cfg, "hidden_size", None)

    @staticmethod
    def _resolve_mrope_section(section: list[int] | None, rotary_half_dim: int) -> list[int]:
        if section is None:
            weights = (2, 1, 1)
            base = [rotary_half_dim * w // sum(weights) for w in weights]
            base[0] += rotary_half_dim - sum(base)
            return base

        section = [int(v) for v in section]
        return section

    def apply_interleaved_mrope(self, freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
        chunks = freqs.split(tuple(mrope_section), dim=-1)
        return torch.cat([chunk[i % 3] for i, chunk in enumerate(chunks)], dim=-1)


class IsaacTextAttention(Qwen3VLTextAttention):
    pass


class IsaacTextDecoderLayer(Qwen3VLTextDecoderLayer):
    pass


class IsaacTextModel(Qwen3VLTextModel):
    def __init__(self, config: IsaacTextConfig):
        super().__init__(config)
        self.rotary_emb = IsaacRotaryEmbedding(config=config, device=self.device)


@auto_docstring
class IsaacModel(Qwen3PreTrainedModel):
    supports_gradient_checkpointing = True
    _can_compile_fullgraph = False
    _supports_flex_attn = False
    _tied_weights_keys = {}

    def __init__(self, config: IsaacConfig):
        Qwen3PreTrainedModel.__init__(self, config)
        self.text_model = IsaacTextModel._from_config(config.text_config)

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

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_token_grids: torch.Tensor,
        image_patch_attention_mask: torch.Tensor | None = None,
        image_token_offsets: torch.Tensor | None = None,
        image_token_lengths: torch.Tensor | None = None,
        image_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        """
        Args:
            pixel_values (`torch.Tensor`):
                Padded per-image patch vectors with shape `(batch_size, max_images, max_patches, patch_dim)`.
            image_token_grids (`torch.Tensor`):
                Per-image token grids shaped `(batch_size, max_images, 2)` with `(height, width)` entries.
            image_patch_attention_mask (`torch.Tensor`, *optional*):
                Mask for valid patch rows in `pixel_values`, shaped `(batch_size, max_images, max_patches)`.
            image_token_offsets (`torch.Tensor`, *optional*):
                Start offsets inside each per-image embedding sequence, shaped `(batch_size, max_images)`.
            image_token_lengths (`torch.Tensor`, *optional*):
                Number of image tokens to gather per image for placeholder scattering, shaped `(batch_size, max_images)`.
            image_attention_mask (`torch.Tensor`, *optional*):
                Mask indicating which image slots are populated, shaped `(batch_size, max_images)`.
        """
        device = self.text_model.embed_tokens.weight.device
        pixel_values = pixel_values.to(device=device)
        image_token_grids = image_token_grids.to(device=device, dtype=torch.long)
        patch_attention_mask = (
            image_patch_attention_mask.to(device=device, dtype=torch.long)
            if image_patch_attention_mask is not None
            else torch.ones(pixel_values.shape[:3], device=device, dtype=torch.long)
        )
        image_attention_mask = (
            image_attention_mask.to(device=device, dtype=torch.bool)
            if image_attention_mask is not None
            else torch.ones(image_token_grids.shape[:2], device=device, dtype=torch.bool)
        )

        batch_size, max_images = pixel_values.shape[:2]
        hidden_size = self.config.get_text_config().hidden_size

        if image_attention_mask.any():
            vision_kwargs = {
                key: value
                for key in ("output_hidden_states", "output_attentions")
                if (value := kwargs.get(key)) is not None
            }
            vision_outputs = self.vision_embedding.vision_tower(
                vision_patches=pixel_values[image_attention_mask],
                vision_token_grids=image_token_grids[image_attention_mask],
                vision_patch_attention_mask=patch_attention_mask[image_attention_mask],
                return_dict=True,
                **vision_kwargs,
            )
            flat_projected_features = self.vision_embedding.multimodal_projector(vision_outputs.last_hidden_state)
            max_tokens = flat_projected_features.shape[1]
            projected_features = flat_projected_features.new_zeros((batch_size, max_images, max_tokens, hidden_size))
            projected_features[image_attention_mask] = flat_projected_features
            offsets = (
                image_token_offsets.to(device=device, dtype=torch.long)
                if image_token_offsets is not None
                else torch.zeros((batch_size, max_images), device=device, dtype=torch.long)
            )
            lengths = (
                image_token_lengths.to(device=device, dtype=torch.long)
                if image_token_lengths is not None
                else torch.full((batch_size, max_images), max_tokens, device=device, dtype=torch.long)
            )
            flat_offsets = offsets[image_attention_mask]
            flat_lengths = lengths[image_attention_mask]
            token_positions = torch.arange(flat_lengths.max(), device=device, dtype=torch.long)
            gather_positions = flat_offsets[:, None] + token_positions[None, :]
            gather_mask = token_positions[None, :] < flat_lengths[:, None]
            image_features = flat_projected_features[
                torch.arange(flat_projected_features.shape[0], device=device, dtype=torch.long)[:, None],
                gather_positions,
            ][gather_mask]
            hidden_states = vision_outputs.hidden_states
            attentions = vision_outputs.attentions
        else:
            projected_features = pixel_values.new_zeros((batch_size, max_images, 0, hidden_size))
            image_features = pixel_values.new_zeros((0, hidden_size))
            hidden_states = None
            attentions = None

        return BaseModelOutputWithPooling(
            last_hidden_state=projected_features,
            pooler_output=image_features,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    def get_placeholder_mask(
        self,
        mm_token_type_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ) -> torch.BoolTensor:
        image_token_mask = mm_token_type_ids.to(device=inputs_embeds.device, dtype=torch.long) == MM_TOKEN_TYPE_IMAGE
        n_image_tokens = image_token_mask.sum()
        image_token_mask = image_token_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        torch_compilable_check(
            inputs_embeds[image_token_mask].numel() == image_features.numel(),
            f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
        )
        return image_token_mask

    def get_rope_index(
        self,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare multimodal RoPE positions for the current prefill sequence.

        Unlike vanilla 1D RoPE, Isaac builds 3-axis indices for text and vision tokens.
        If callers do not supply positions, we synthesize text-style positions from
        `attention_mask`. The returned `rope_deltas` capture any custom offset between
        the attended sequence length and Isaac's multimodal positions so decode steps can
        keep counting forward from the cached prefix."""

        device = inputs_embeds.device
        batch_size, seq_len = inputs_embeds.shape[:2]
        attention_mask = attention_mask.to(device=device, dtype=torch.long)

        if position_ids is None:
            rope_position = attention_mask.cumsum(dim=-1) - 1
            rope_position = rope_position.masked_fill(attention_mask == 0, 0)
            rope_position = rope_position[:, -seq_len:]
            pos_3d = rope_position.unsqueeze(-1).expand(-1, -1, 3)
            rope_deltas = torch.zeros((batch_size, 1), device=device, dtype=torch.long)
            return pos_3d, rope_deltas

        position_ids = position_ids.to(device=device)
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(-1).expand(-1, -1, 3)

        if position_ids.shape[1] != seq_len:
            start_positions = position_ids[:, :1, 0]
            position_ids = torch.arange(seq_len, device=position_ids.device).view(1, -1) + start_positions
            position_ids = position_ids.unsqueeze(-1).expand(-1, -1, 3)

        m_per_batch = position_ids.amax(dim=(1, 2))
        seq_lens = attention_mask.eq(1).sum(dim=-1).to(dtype=m_per_batch.dtype, device=device)
        rope_deltas = (m_per_batch + 1 - seq_lens).to(dtype=position_ids.dtype).unsqueeze(1)
        return position_ids, rope_deltas

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        past_seen_tokens = 0 if past_key_values is None else past_key_values.get_seq_length()
        if past_seen_tokens == 0:
            position_ids, rope_deltas = self.get_rope_index(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
            self.rope_deltas = rope_deltas
            return position_ids

        if self.rope_deltas is None:
            return None

        rope_deltas = torch.as_tensor(self.rope_deltas, device=inputs_embeds.device, dtype=torch.long).reshape(-1, 1)
        if rope_deltas.shape[0] != inputs_embeds.shape[0]:
            if inputs_embeds.shape[0] % rope_deltas.shape[0] == 0:
                rope_deltas = rope_deltas.repeat_interleave(inputs_embeds.shape[0] // rope_deltas.shape[0], dim=0)
            else:
                rope_deltas = rope_deltas[:1].expand(inputs_embeds.shape[0], -1)

        if attention_mask.shape[-1] > inputs_embeds.shape[1]:
            rope_position = attention_mask.long().cumsum(dim=-1) - 1
            rope_position = rope_position.masked_fill(attention_mask == 0, 0)
            rope_position = rope_position[:, -inputs_embeds.shape[1] :]
        else:
            rope_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
                dtype=torch.long,
            ).view(1, -1)
            rope_position = rope_position.expand(inputs_embeds.shape[0], -1)

        return rope_position.unsqueeze(-1).expand(-1, -1, 3) + rope_deltas.unsqueeze(-1)

    @auto_docstring
    @can_return_tuple
    @merge_with_config_defaults
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        vision_patches: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        vision_patch_attention_mask: torch.Tensor | None = None,
        image_patch_attention_mask: torch.Tensor | None = None,
        vision_token_grids: torch.LongTensor | None = None,
        image_token_grids: torch.LongTensor | None = None,
        vision_token_offsets: torch.LongTensor | None = None,
        vision_token_lengths: torch.LongTensor | None = None,
        vision_image_attention_mask: torch.LongTensor | None = None,
        image_attention_mask: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        """
        Forward pass with MRoPE position embeddings.

        Computes position embeddings once and passes them through all layers.

        Args:
            mm_token_type_ids (`torch.LongTensor`, *optional*):
                Multimodal token type ids aligned with the embedded sequence, shaped `(batch_size, seq_len)`. Isaac
                follows the standard convention `0 -> text`, `1 -> image`. Treated as text-only when omitted.
            vision_patches (`torch.FloatTensor`, *optional*):
                Padded per-image patch vectors of shape `(batch_size, max_images, max_patches, patch_dim)`.
            pixel_values (`torch.FloatTensor`, *optional*):
                Alias for `vision_patches` accepted by generic image-feature and generation helpers.
            vision_patch_attention_mask (`torch.LongTensor`, *optional*):
                Mask for valid patch entries in `vision_patches`, shaped `(batch_size, max_images, max_patches)`.
            image_patch_attention_mask (`torch.LongTensor`, *optional*):
                Alias for `vision_patch_attention_mask`.
            vision_token_grids (`torch.LongTensor`, *optional*):
                Per-image patch grids `(h, w)` with shape `(batch_size, max_images, 2)`.
            image_token_grids (`torch.LongTensor`, *optional*):
                Alias for `vision_token_grids`.
            vision_token_offsets (`torch.LongTensor`, *optional*):
                Start offsets inside the per-image vision embedding sequence, shape `(batch_size, max_images)`.
            vision_token_lengths (`torch.LongTensor`, *optional*):
                Number of vision tokens to consume per image, shape `(batch_size, max_images)`.
            vision_image_attention_mask (`torch.LongTensor`, *optional*):
                Mask indicating which image slots are populated, shape `(batch_size, max_images)`.
            image_attention_mask (`torch.LongTensor`, *optional*):
                Alias for `vision_image_attention_mask`.
        """
        if vision_patches is None and pixel_values is not None:
            vision_patches = pixel_values
            vision_patch_attention_mask = (
                image_patch_attention_mask if vision_patch_attention_mask is None else vision_patch_attention_mask
            )
            vision_token_grids = image_token_grids if vision_token_grids is None else vision_token_grids
            vision_image_attention_mask = (
                image_attention_mask if vision_image_attention_mask is None else vision_image_attention_mask
            )

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("`input_ids` or `inputs_embeds` must be provided.")

            has_vision_inputs = vision_patches is not None
            if mm_token_type_ids is not None or has_vision_inputs:
                if mm_token_type_ids is None:
                    mm_token_type_ids = torch.full_like(input_ids, MM_TOKEN_TYPE_TEXT)
                mm_token_type_ids = mm_token_type_ids.to(device=input_ids.device, dtype=torch.long)
                inputs_embeds = self.text_model.embed_tokens(input_ids)
                image_token_mask = mm_token_type_ids == MM_TOKEN_TYPE_IMAGE
                if torch.any(image_token_mask) and (vision_patches is None or vision_token_grids is None):
                    raise ValueError("Image placeholders require `vision_patches` and `vision_token_grids`.")
                if torch.any(image_token_mask):
                    image_outputs = self.get_image_features(
                        pixel_values=vision_patches,
                        image_token_grids=vision_token_grids,
                        image_patch_attention_mask=vision_patch_attention_mask,
                        image_token_offsets=vision_token_offsets,
                        image_token_lengths=vision_token_lengths,
                        image_attention_mask=vision_image_attention_mask,
                        return_dict=True,
                    )
                    image_features = image_outputs.pooler_output.to(
                        device=inputs_embeds.device, dtype=inputs_embeds.dtype
                    )
                    scatter_mask = self.get_placeholder_mask(
                        mm_token_type_ids=mm_token_type_ids,
                        inputs_embeds=inputs_embeds,
                        image_features=image_features,
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(scatter_mask, image_features)
            else:
                inputs_embeds = self.text_model.embed_tokens(input_ids)

        if mm_token_type_ids is None:
            batch_size, seq_len = inputs_embeds.shape[:2]
            mm_token_type_ids = torch.full(
                (batch_size, seq_len), MM_TOKEN_TYPE_TEXT, device=inputs_embeds.device, dtype=torch.long
            )

        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long)

        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        if position_ids is None:
            text_position_ids = None
        else:
            rope_position_ids = position_ids.clone()
            text_mask = mm_token_type_ids == MM_TOKEN_TYPE_TEXT
            text_positions = rope_position_ids[text_mask][..., :1]
            rope_position_ids[text_mask] = text_positions.expand(-1, rope_position_ids.shape[-1])
            rope_position_ids = rope_position_ids.permute(2, 0, 1).contiguous()
            text_position_ids = torch.cat((position_ids[..., 0].unsqueeze(0), rope_position_ids), dim=0)

        if isinstance(attention_mask, dict):
            attention_mask = attention_mask.get("full_attention", next(iter(attention_mask.values())))

        text_model_outputs = self.text_model(
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=text_model_outputs.last_hidden_state,
            past_key_values=text_model_outputs.past_key_values,
            hidden_states=text_model_outputs.hidden_states,
            attentions=text_model_outputs.attentions,
        )


@auto_docstring
class IsaacForConditionalGeneration(Qwen3ForCausalLM, GenerationMixin):
    config_class = IsaacConfig
    _can_compile_fullgraph = False
    _tied_weights_keys = {"lm_head.weight": "model.text_model.embed_tokens.weight"}

    def __init__(self, config: IsaacConfig):
        super().__init__(config)
        self.model = IsaacModel(config)
        self.vocab_size = config.get_text_config().vocab_size
        self.lm_head = nn.Linear(config.get_text_config().hidden_size, config.get_text_config().vocab_size, bias=False)

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        vision_patches: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        vision_patch_attention_mask: torch.Tensor | None = None,
        image_patch_attention_mask: torch.Tensor | None = None,
        vision_token_grids: torch.LongTensor | None = None,
        image_token_grids: torch.LongTensor | None = None,
        vision_token_offsets: torch.LongTensor | None = None,
        vision_token_lengths: torch.LongTensor | None = None,
        vision_image_attention_mask: torch.LongTensor | None = None,
        image_attention_mask: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        mm_token_type_ids (`torch.LongTensor`, *optional*):
            Multimodal token type ids aligned with the token sequence, shaped `(batch_size, seq_len)`, using
            `0 -> text` and `1 -> image`.
        vision_patches (`torch.FloatTensor`, *optional*):
            Padded per-image patch vectors of shape `(batch_size, max_images, max_patches, patch_dim)`.
        pixel_values (`torch.FloatTensor`, *optional*):
            Alias for `vision_patches` accepted by generic image-feature and generation helpers.
        vision_patch_attention_mask (`torch.LongTensor`, *optional*):
            Mask for valid patch entries in `vision_patches`, shaped `(batch_size, max_images, max_patches)`.
        image_patch_attention_mask (`torch.LongTensor`, *optional*):
            Alias for `vision_patch_attention_mask`.
        vision_token_grids (`torch.LongTensor`, *optional*):
            Per-image patch grids `(h, w)` with shape `(batch_size, max_images, 2)`.
        image_token_grids (`torch.LongTensor`, *optional*):
            Alias for `vision_token_grids`.
        vision_token_offsets (`torch.LongTensor`, *optional*):
            Start offsets inside the per-image vision embedding sequence, shape `(batch_size, max_images)`.
        vision_token_lengths (`torch.LongTensor`, *optional*):
            Number of vision tokens to consume per image, shape `(batch_size, max_images)`.
        vision_image_attention_mask (`torch.LongTensor`, *optional*):
            Mask indicating which image slots are populated, shape `(batch_size, max_images)`.
        image_attention_mask (`torch.LongTensor`, *optional*):
            Alias for `vision_image_attention_mask`.
        """
        if vision_patches is None and pixel_values is not None:
            vision_patches = pixel_values
            vision_patch_attention_mask = (
                image_patch_attention_mask if vision_patch_attention_mask is None else vision_patch_attention_mask
            )
            vision_token_grids = image_token_grids if vision_token_grids is None else vision_token_grids
            vision_image_attention_mask = (
                image_attention_mask if vision_image_attention_mask is None else vision_image_attention_mask
            )
        outputs = self.model(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
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
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=logits.shape[-1])

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
        mm_token_type_ids: torch.LongTensor | None = None,
        vision_patches: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        vision_patch_attention_mask: torch.Tensor | None = None,
        image_patch_attention_mask: torch.Tensor | None = None,
        vision_token_grids: torch.LongTensor | None = None,
        image_token_grids: torch.LongTensor | None = None,
        vision_token_offsets: torch.LongTensor | None = None,
        vision_token_lengths: torch.LongTensor | None = None,
        vision_image_attention_mask: torch.LongTensor | None = None,
        image_attention_mask: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        is_first_iteration=False,
        use_cache=True,
        **kwargs,
    ) -> dict[str, Any]:
        if vision_patches is None and pixel_values is not None:
            vision_patches = pixel_values
            vision_patch_attention_mask = (
                image_patch_attention_mask if vision_patch_attention_mask is None else vision_patch_attention_mask
            )
            vision_token_grids = image_token_grids if vision_token_grids is None else vision_token_grids
            vision_image_attention_mask = (
                image_attention_mask if vision_image_attention_mask is None else vision_image_attention_mask
            )
        custom_position_ids = (
            position_ids
            if position_ids is not None and position_ids.ndim == 3 and vision_patches is not None
            else None
        )
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=None,
            is_first_iteration=is_first_iteration,
            use_cache=use_cache,
            **kwargs,
        )
        multimodal_inputs = {
            "mm_token_type_ids": mm_token_type_ids,
            "vision_patches": vision_patches,
            "vision_patch_attention_mask": vision_patch_attention_mask,
            "vision_token_grids": vision_token_grids,
            "vision_token_offsets": vision_token_offsets,
            "vision_token_lengths": vision_token_lengths,
            "vision_image_attention_mask": vision_image_attention_mask,
        }
        is_prefill = is_first_iteration or not use_cache
        for key, value in multimodal_inputs.items():
            model_inputs[key] = value if is_prefill else None
        if custom_position_ids is not None:
            model_inputs["position_ids"] = custom_position_ids if is_prefill else None

        return model_inputs

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()


__all__ = [
    "IsaacConfig",
    "IsaacTextConfig",
    "IsaacTextModel",
    "IsaacVisionConfig",
    "IsaacModel",
    "IsaacPreTrainedModel",  # noqa: F822
    "IsaacForConditionalGeneration",
    "IsaacImageProcessorFast",
    "IsaacProcessor",
]
