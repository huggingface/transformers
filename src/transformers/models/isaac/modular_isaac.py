# coding=utf-8
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
from typing import Any, Optional, Union

from ...utils.import_utils import (
    is_torch_available,
    is_torchdynamo_compiling,
    is_torchvision_available,
    is_vision_available,
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

from enum import IntEnum

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
from ...masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, create_masks_for_generate, packed_sequence_mask_function
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...models.qwen3.configuration_qwen3 import Qwen3Config
from ...models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model, Qwen3PreTrainedModel
from ...processing_utils import ProcessorMixin, Unpack
from ...utils import TensorType, auto_docstring

# Vision preprocessing constants
from ...utils.constants import IMAGENET_STANDARD_MEAN as VISION_MEAN
from ...utils.constants import IMAGENET_STANDARD_STD as VISION_STD
from ...utils.generic import (
    OutputRecorder,
    TransformersKwargs,
    can_return_tuple,
    check_model_inputs,
)
from ..qwen2_5_vl import modeling_qwen2_5_vl as qwen2_5_vl_modeling
from ..siglip2.configuration_siglip2 import Siglip2VisionConfig
from ..siglip2.modeling_siglip2 import (
    Siglip2Attention,
    Siglip2Encoder,
    Siglip2EncoderLayer,
    Siglip2VisionEmbeddings,
)


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
    patch_size: Optional[int]
    max_num_patches: Optional[int]
    min_num_patches: Optional[int]
    pixel_shuffle_scale: Optional[int]


@auto_docstring
class IsaacImageProcessorFast(BaseImageProcessorFast):
    MAX_PIXELS = 60_000_000  # 60‑megapixel ceiling ≈ 8200 × 7300 px
    r"""Fast torch-based image processor for Isaac vision inputs."""

    resample = PILImageResampling.BILINEAR
    model_input_names = ["patches", "token_grids"]
    valid_kwargs = IsaacImageProcessorFastKwargs
    unused_kwargs = ["size", "do_center_crop", "crop_size", "pad_size", "do_pad"]

    do_resize = True
    do_center_crop = False
    patch_size: Optional[int] = 16
    max_num_patches: Optional[int] = 256
    min_num_patches: Optional[int] = None
    pixel_shuffle_scale: Optional[int] = 1
    do_pad = False
    do_rescale = True
    do_normalize = True
    image_mean = list(VISION_MEAN)
    image_std = list(VISION_STD)
    do_convert_rgb = True
    disable_grouping = False

    def __init__(
        self,
        **kwargs: Unpack[IsaacImageProcessorFastKwargs],
    ) -> None:
        super().__init__(**kwargs)

    def _validate_preprocess_kwargs(self, **kwargs):
        # Allow callers to omit resize-related placeholders that BaseImageProcessorFast checks for.
        kwargs.pop("do_resize", None)
        kwargs.pop("size", None)
        kwargs.pop("do_center_crop", None)
        kwargs.pop("crop_size", None)
        kwargs.pop("disable_grouping", None)
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
        interpolation: Optional[Any],
        do_rescale: Optional[bool],
        rescale_factor: Optional[float],
        do_normalize: Optional[bool],
        image_mean: Optional[Union[float, Sequence[float]]],
        image_std: Optional[Union[float, Sequence[float]]],
        disable_grouping: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        *,
        patch_size: Optional[int] = None,
        max_num_patches: Optional[int] = None,
        min_num_patches: Optional[int] = None,
        pixel_shuffle_scale: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)

        grouped_outputs = {}

        for shape, stacked_images in grouped_images.items():
            if stacked_images.ndim != 4:
                raise ValueError(
                    f"Expected images shaped as (batch, channels, height, width); got shape {tuple(stacked_images.shape)}."
                )

            batch_size, channels, original_height, original_width = stacked_images.shape

            if bool(self.do_convert_rgb) and channels == 1:
                stacked_images = stacked_images.repeat(1, 3, 1, 1)
                channels = 3

            if original_height * original_width > self.MAX_PIXELS:
                raise ValueError(
                    f"Image area {original_height * original_width} (h={original_height}, w={original_width}) exceeds MAX_PIXELS={self.MAX_PIXELS}; enable resizing or provide smaller inputs."
                )

            target_height, target_width = get_image_size_for_max_num_patches(
                original_height,
                original_width,
                patch_size,
                max_num_patches,
                min_num_patches=min_num_patches,
                pixel_shuffle_scale=pixel_shuffle_scale,
            )

            if do_resize:
                resize_size = SizeDict(height=target_height, width=target_width)
                image_batch = self.resize(
                    image=stacked_images,
                    size=resize_size,
                    interpolation=interpolation,
                )
            else:
                if ((original_height % patch_size) != 0) or ((original_width % patch_size) != 0):
                    raise ValueError(
                        f"Image dimensions (h={original_height}, w={original_width}) must be divisible by patch_size={patch_size} when resize is disabled; enable resizing or adjust the input resolution."
                    )
                image_batch = stacked_images
                target_height, target_width = original_height, original_width

            if do_rescale:
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
                torch.tensor(
                    [height_tokens, width_tokens],
                    dtype=torch.long,
                    device=patches.device,
                )
                .unsqueeze(0)
                .repeat(batch_size, 1)
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

        # Helper to reorder a single item of the tuple payloads using the same grouped_images_index
        def _reorder_grouped_item(
            grouped: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
            grouped_index: dict[tuple[int, ...], list[int]],
            item_idx: int,
        ) -> list[torch.Tensor]:
            return reorder_images({k: v[item_idx] for k, v in grouped.items()}, grouped_index)

        keys = ("patches", "token_grids", "virtual_pixel_size", "real_pixel_size")
        tensors: dict[str, torch.Tensor] = {}

        for i, key in enumerate(keys):
            slices = _reorder_grouped_item(grouped_outputs, grouped_images_index, i)
            tensors[key] = torch.stack(slices, dim=0)

        return BatchFeature(data=tensors, tensor_type=return_tensors)


def create_document_attention_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
) -> Optional[Union[torch.Tensor, Any]]:
    """
    Materialize a backend-specific block-diagonal attention mask from packed cu_seqlens.

    Returns None if cu_seqlens is missing/degenerate.
    """
    if cu_seqlens is None or cu_seqlens.numel() < 2:
        return None  # Degenerate input: nothing to mask

    seq_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    if seq_sizes.numel() == 0 or int(seq_sizes.sum()) == 0:
        return None  # All-empty segments produce no attention blocks

    seg_ids = torch.repeat_interleave(
        torch.arange(seq_sizes.numel(), device=cu_seqlens.device),
        seq_sizes,
    )
    mask_function = packed_sequence_mask_function(seg_ids.view(1, -1))

    seq_len = input_embeds.shape[1]
    cache_position = torch.arange(seq_len, device=input_embeds.device, dtype=torch.long)

    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]
    return mask_interface(
        batch_size=input_embeds.shape[0],
        cache_position=cache_position,
        kv_length=seq_len,
        kv_offset=0,
        mask_function=mask_function,
        attention_mask=None,
        allow_is_causal_skip=False,
        allow_is_bidirectional_skip=False,
        dtype=input_embeds.dtype,
        config=config,
        use_vmap=False,
    )


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
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    @check_model_inputs
    def forward(self, seq_patches: torch.Tensor, spatial_shapes: torch.Tensor) -> torch.Tensor:
        # Rebatch packed variable-resolution patches to resize per-image position embeddings
        # and track lengths for varlen attention metadata.
        packed_pixel_values, seq_lengths = self._pack_to_batch(seq_patches, spatial_shapes)
        if packed_pixel_values is None:
            return seq_patches.new_zeros((0, self.embed_dim))

        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(packed_pixel_values.to(dtype=target_dtype))

        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size,
            self.position_embedding_size,
            -1,
        )
        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings, spatial_shapes, max_length=packed_pixel_values.shape[1]
        )

        embeddings = patch_embeds + resized_positional_embeddings
        return self._unpack_from_batch(embeddings, seq_lengths)

    def _pack_to_batch(
        self,
        seq_patches: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """Rebatch a packed patch sequence using per-image grids to align embeddings.

        Args:
            seq_patches: Packed patches of shape (total_patches, patch_dim).
            spatial_shapes: Per-image patch grids of shape (num_images, 2) as (H_tokens, W_tokens).

        Returns:
            (packed_pixel_values, seq_lengths) where:
            - packed_pixel_values: (batch, max_len, patch_dim) padded with zeros, or None if batch_size == 0
            - seq_lengths: (batch,) lengths for each image
        """
        # Per-image token counts
        seq_lengths = spatial_shapes.long().prod(dim=-1)  # (B,)
        batch_size = int(seq_lengths.numel())
        if batch_size == 0:
            return None, seq_lengths

        # Split the packed sequence into per-image chunks, then pad to a batch
        lengths_list = seq_lengths.tolist()
        chunks = seq_patches.split(lengths_list, dim=0)
        packed_pixel_values = nn.utils.rnn.pad_sequence(chunks, batch_first=True)  # zero-padded by default
        return packed_pixel_values, seq_lengths

    def _unpack_from_batch(self, embeddings: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        """Flatten a padded batch back to packed sequence order using `seq_lengths`."""
        lengths = seq_lengths.to(device=embeddings.device).tolist()
        chunks = [embeddings[i, :l] for i, l in enumerate(lengths) if l > 0]
        if not chunks:
            return embeddings.new_zeros((0, embeddings.size(-1)))
        return torch.cat(chunks, dim=0)


class IsaacVisionAttention(Siglip2Attention):
    """Custom attention that supports variable-length sequences with flash attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ):
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)

        batch_size, seq_length, embed_dim = hidden_states.shape
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_impl = self.config._attn_implementation
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS["sdpa"]
        if attn_impl != "sdpa":
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]

        attention_kwargs: dict[str, Any] = {
            "is_causal": False,
            "scaling": self.scale,
        }

        supports_varlen = cu_seqlens is not None and attn_impl in {
            "flash_attention_2",
            "flash_attention_3",
            "flex_attention",
            "paged|flash_attention_2",
            "paged|flash_attention_3",
        }
        if supports_varlen:
            if max_seqlen is not None:
                max_q = max_k = int(max_seqlen)
            elif cu_seqlens.numel() >= 2:
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                max_q = max_k = lengths.max() if lengths.numel() > 0 else seq_length
            else:
                max_q = max_k = seq_length

            attention_kwargs.update(
                {
                    "cu_seq_lens_q": cu_seqlens,
                    "cu_seq_lens_k": cu_seqlens,
                    "max_length_q": max_q,
                    "max_length_k": max_k,
                }
            )

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
    """Isaac vision encoder layer with variable-length attention."""

    def __init__(self, config: IsaacVisionConfig):
        super().__init__(config)
        self.self_attn = IsaacVisionAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        output_attentions: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        cu_seqlens (`torch.Tensor`, *optional*):
            Prefix-sum tensor whose length equals the number of documents + 1. The difference between successive
            entries gives each document's token count and enables block-diagonal attention masking for packed batches.
        max_seqlen (`int`, *optional*):
            Maximum document length referenced by `cu_seqlens`. Passed to FlashAttention so it can size temporary
            buffers for packed variable-length attention.
        """
        # Run attention directly so variable-length metadata reaches FlashAttention.
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_output, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            **kwargs,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class IsaacVisionEncoder(Siglip2Encoder):
    """Encoder using Isaac encoder layers with variable-length attention support."""

    def __init__(self, config: IsaacVisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([IsaacVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])


def create_pixel_shuffle_index_map(
    seq_sizes: torch.Tensor,
    token_grids: torch.Tensor,
    scale_factor: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build a gather-index map that tells us, for every *output* token after
    pixel-shuffle, which `scale_factor**2` *input* tokens are being merged.

    Args
    ----
    seq_sizes     : (num_images,)  - #patches in each image (row-major order)
    token_grids   : (num_images,2) - (height, width) for every image
    scale_factor  : spatial down-scale factor (≥2)
    device        : (optional) overrides `seq_sizes.device`

    Returns
    -------
    gather_idx : (new_total_seq_len, scale_factor**2) int64 tensor.
                 gather_idx[i, j] is the *flat* index into the *original*
                 packed sequence for the j-th sub-patch that forms the
                 i-th output token.
    """
    if not is_torchdynamo_compiling():
        if (token_grids % scale_factor).any():
            raise AssertionError(
                f"Every (H,W) in token_grids must be divisible by scale_factor={scale_factor}, got {token_grids.tolist()}"
            )

    gather_chunks: list[torch.Tensor] = []
    tok_offset = 0
    for seq_len, (h, w) in zip(seq_sizes.tolist(), token_grids.tolist()):
        # Flat indices for this image's packed segment
        grid = torch.arange(seq_len, device=device, dtype=torch.int64).view(h, w) + tok_offset

        # Block into (H/s, W/s) groups; each group contributes s*s indices
        grid = (
            grid.view(h // scale_factor, scale_factor, w // scale_factor, scale_factor)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        gather_chunks.append(grid.view(-1, scale_factor * scale_factor))

        tok_offset += seq_len

    return torch.cat(gather_chunks, dim=0)


def pixel_shuffle_varlen(
    x: torch.Tensor,
    token_grids: torch.Tensor,
    scale_factor: int = 1,
) -> torch.Tensor:
    r"""Apply pixel shuffle to a packed vision sequence without unpacking per image.

    Args:
        x (`torch.Tensor`):
            Concatenated vision embeddings. Accepts `(seq_len, hidden_size)` or `(1, seq_len, hidden_size)` shapes
            produced by stacking image patches.
        token_grids (`torch.Tensor`):
            Integer tensor of shape `(num_images, 2)` whose rows give the `(height, width)` patch grid sizes
            corresponding to each image segment inside `x`.
        scale_factor (`int`, *optional*, defaults to 1):
            Spatial down-sampling factor specific to pixel shuffle. Values greater than one merge `scale_factor**2` neighboring patches into a
            single embedding channel-group.

    Returns:
        `torch.Tensor`: Pixel-shuffled embeddings with shape matching the input convention:
        `(seq_len, hidden_size * scale_factor**2)` when the input was 2D, or `(1, seq_len, hidden_size * scale_factor**2)`
        if the singleton batch dimension was present.

    Raises:
        ValueError: If more than one batch item is provided.
    """
    return_with_batch_dim = x.dim() == 3
    if return_with_batch_dim:
        if x.size(0) != 1:
            raise ValueError(
                f"Packed vision sequences expect a singleton batch dimension; received batch_size={x.size(0)}."
            )
        embeddings = x.squeeze(0)  # (seq, embed)
    else:
        embeddings = x  # (seq, embed)

    embed_dim = embeddings.size(-1)
    scale_factor = int(scale_factor)

    # Calculate seq_sizes from token_grids
    seq_sizes = torch.prod(token_grids, dim=-1)

    # Build a single gather index so pixel shuffle works on the packed stream
    # without unpacking per-image grids.
    gather_idx = create_pixel_shuffle_index_map(
        seq_sizes=seq_sizes,
        token_grids=token_grids,
        scale_factor=scale_factor,
        device=embeddings.device,
    )  # (new_seq, scale_factor**2)

    # Gather → (new_seq, scale_factor**2, embed_dim)
    gathered = embeddings[gather_idx]  # fancy indexing keeps gradient

    # Merge the scale_factor**2 group dimension into channels to finish the shuffle
    out = gathered.reshape(gathered.size(0), embed_dim * scale_factor * scale_factor)

    # Restore batch dimension if needed
    if return_with_batch_dim:
        out = out.unsqueeze(0)
    return out


class IsaacVisionTransformer(nn.Module):
    """Vision tower that packs variable-resolution patches, applies varlen attention, and pixel-shuffles outputs.

    Args:
        config (IsaacVisionConfig): Vision configuration with pixel-shuffle and patching parameters.

    Inputs:
        packed_seq_patches (Tuple[Tensor, Tensor]): ``(patches, token_grids)`` where ``patches`` is a packed
            patch sequence and ``token_grids`` holds per-image (H_tokens, W_tokens).

    Returns:
        torch.Tensor: Vision embeddings after encoder + pixel shuffle, shaped ``(seq_len, hidden_size * s^2)``.
    """

    _supports_sdpa = True

    def __init__(self, config: IsaacVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = IsaacVisionEmbeddings(config)
        self.encoder = IsaacVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pixel_shuffle_scale_factor = config.pixel_shuffle_scale_factor

    def forward(self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor]):
        seq_patches, token_grids = packed_seq_patches
        seq_sizes = torch.prod(token_grids, dim=-1)

        # Get embeddings from packed sequence
        hidden_states = self.embeddings(seq_patches, token_grids)

        # Add a pseudo batch dimension so we can reuse the batch-first encoder stack
        # while still driving per-image cu_seqlens through the varlen attention path.
        hidden_states = hidden_states.unsqueeze(0)

        # Generate cumulative sequence lengths for variable-length attention
        cu_seqlens = F.pad(seq_sizes.cumsum(0).to(torch.int32), (1, 0))

        attention_mask = create_document_attention_mask(self.config, hidden_states, cu_seqlens)

        # Pass through encoder with variable-length attention parameters
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
        )
        hidden_states = encoder_outputs.last_hidden_state

        # Apply final layer normalization
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = pixel_shuffle_varlen(
            x=hidden_states,
            token_grids=token_grids,
            scale_factor=self.pixel_shuffle_scale_factor,
        )
        # Remove the pseudo batch dimension we added earlier
        hidden_states = hidden_states.squeeze(0)

        # Return the full sequence of embeddings
        return hidden_states


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
    """Wraps the vision tower plus projection into the text hidden size.

    Args:
        config (IsaacConfig): Composite config containing both vision and text settings.

    Inputs:
        vision_tokens (Tuple[Tensor, Tensor]): Packed vision patches and token grids.

    Returns:
        torch.Tensor: Projected vision embeddings aligned to the text hidden size.
    """

    _supports_sdpa = True

    def __init__(self, config: IsaacConfig):
        super().__init__()
        vision_cfg = config.vision_config

        self.vision_tower = IsaacVisionTransformer(vision_cfg)
        self.multimodal_projector = IsaacMultiModalProjector(config)

    def forward(self, vision_tokens: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hidden_states = self.vision_tower(vision_tokens)
        return self.multimodal_projector(hidden_states)


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
    min_num_patches: Optional[int] = None,
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
        vision_config: Optional[IsaacVisionConfig] = None,
        text_config: Optional[Union[Qwen3Config, dict]] = None,
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
        BatchFeature: Contains ``input_ids`` and ``packed_inputs`` (patch tensors, grids, offsets, lengths, modality, positions).
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
        rescale_factor: Optional[float] = None,
        config: Optional[Union[IsaacConfig, dict]] = None,
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

        self.current_processor = self.image_processor
        self.config = config
        self.chat_template = getattr(self.tokenizer, "chat_template", None)
        self.vision_token = vision_token
        self.max_sequence_length = max_sequence_length

    def _pack_single(self, text: str, images: Optional[list[Image]]) -> dict[str, Optional[torch.Tensor]]:
        # Parse by vision_token; interleave text segments and image segments.
        segments = text.split(self.vision_token)
        num_images = len(segments) - 1
        if num_images and (images is None or len(images) != num_images):
            raise ValueError(
                f"Expected one image per '{self.vision_token}' token: found {num_images} token(s) but received {0 if images is None else len(images)} image(s)."
            )

        items: list[dict[str, Any]] = []
        total = 0

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

        fill_value = self.pad_token_id
        base_device: Optional[torch.device] = None
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
                        torch.full((segment_kept_length,), fill_value, device=base_device, dtype=torch.long)
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
            vision_patches = torch.cat(vpatches, 0)
            vision_token_grids = torch.tensor(grids, device=base_device, dtype=torch.long)
            vision_token_offsets = torch.tensor(vision_token_offsets, device=base_device, dtype=torch.long)
            vision_token_lengths = torch.tensor(vision_token_lengths, device=base_device, dtype=torch.long)
        else:
            vision_patches = vision_token_grids = vision_token_offsets = vision_token_lengths = None

        return {
            "input_ids": input_ids,
            "vision_patches": vision_patches,
            "vision_token_grids": vision_token_grids,
            "vision_token_offsets": vision_token_offsets,
            "vision_token_lengths": vision_token_lengths,
            "modality_tensor": modality_tensor,
            "position_ids": position_ids,
        }

    def __call__(
        self,
        text: Union[str, list[str]],
        images: Optional[Union[Image, list[Image]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        **kwargs,
    ) -> BatchFeature:
        texts = [text] if isinstance(text, str) else text
        if len(texts) != 1:
            raise ValueError(
                f"IsaacProcessor currently supports batch_size=1; received {len(texts)} text prompts. Split the batch and call the processor per sample."
            )

        images_list = None
        if images is not None:
            images_list = [images] if isinstance(images, Image) else images
            n_tok = texts[0].count(self.vision_token)
            if n_tok != len(images_list):
                raise ValueError(
                    f"Expected {len(images_list)} occurrences of '{self.vision_token}' (one per provided image), but found {n_tok} in the text."
                )

        packed = self._pack_single(texts[0], images_list)
        input_ids = packed.pop("input_ids")
        return BatchFeature(data={"input_ids": input_ids, "packed_inputs": packed})


class IsaacRotaryEmbedding(qwen2_5_vl_modeling.Qwen2_5_VLRotaryEmbedding):
    def __init__(self, config: IsaacConfig, device=None):
        rope_source_cfg = config.get_text_config() if hasattr(config, "get_text_config") else config
        rope_scaling = getattr(rope_source_cfg, "rope_scaling", None) or {}
        config_for_rope = copy.copy(rope_source_cfg)
        config_for_rope.rope_scaling = rope_scaling

        init_device = device if device is not None and getattr(device, "type", None) != "meta" else None
        super().__init__(config_for_rope, device=init_device)

        rotary_half_dim = self.inv_freq.shape[0]
        self.mrope_section = self._resolve_mrope_section(rope_scaling.get("mrope_section"), rotary_half_dim)
        self.hidden_size = getattr(rope_source_cfg, "hidden_size", None) or config.hidden_size

    @staticmethod
    def _resolve_mrope_section(section: Optional[list[int]], rotary_half_dim: int) -> list[int]:
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
        hidden_states: Optional[torch.Tensor] = None,
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
            not_spatial = modality_tensor != ModalityType.image.value
            if not_spatial.any():
                # Collapse non-vision modalities to 1D positions so rotary embedding
                # treats them like text tokens while keeping image tokens 3D.
                data_1d = pos[not_spatial][..., 0].unsqueeze(-1)
                pos[not_spatial] = data_1d.expand(-1, pos.shape[-1])

            pos_axes = pos.permute(2, 0, 1).contiguous()

        cos_axes, sin_axes = super().forward(hidden_states, pos_axes)
        cos_axes = cos_axes.to(hidden_states.dtype)
        sin_axes = sin_axes.to(hidden_states.dtype)
        cos_combined, sin_combined = self._combine_axes(cos_axes), self._combine_axes(sin_axes)

        return cos_combined, sin_combined


@auto_docstring
class IsaacModel(Qwen3PreTrainedModel):
    supports_gradient_checkpointing = True
    _can_compile_fullgraph = False
    _supports_flex_attn = False
    _can_record_outputs = {"attentions": OutputRecorder(IsaacVisionAttention, index=1)}
    all_tied_weights_keys: dict[str, str] = {}

    def __init__(self, config: IsaacConfig):
        Qwen3PreTrainedModel.__init__(self, config)

        text_cfg_source = config.text_config
        text_cfg = copy.deepcopy(text_cfg_source)
        self.text_model = Qwen3Model._from_config(text_cfg)
        self.text_model.config = config  # Ensure downstream callers observe the composed config

        self.rotary_emb = IsaacRotaryEmbedding(config, device=self.device)

        self.vision_embedding = IsaacVisionEmbedding(config)
        self.vision_embedding._supports_sdpa = True
        self.max_sequence_length = config.max_sequence_length
        self.vision_rescale_factor = config.vision_rescale_factor
        self.vision_token = config.vision_token

        self.post_init()

        # Respect config-specified gradient checkpointing
        if getattr(config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()

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
    def embed_tokens(self) -> nn.Module:
        return self.text_model.embed_tokens

    @embed_tokens.setter
    def embed_tokens(self, value: nn.Module) -> None:
        self.text_model.embed_tokens = value

    @property
    def vision_model(self) -> nn.Module:
        return self.vision_embedding.vision_tower

    def embed_packed_inputs(
        self, input_ids: torch.Tensor, packed_inputs: dict[str, Optional[torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Expects input_ids for text tokens and packed_inputs containing:
        - modality_tensor: (batch, seq_len) modality ids aligned to the sequence
        - position_ids: (batch, seq_len, 3) MRoPE coordinates (optional)
        - vision_patches: concatenated vision tokens shaped (total_tokens, embed_dim) or None
        - vision_token_grids: (num_images, 2) token grid sizes or None
        - vision_token_offsets: (num_images,) offsets into each image's virtual token span (optional)
        - vision_token_lengths: (num_images,) surviving virtual token lengths per image (optional)
        """
        modality = packed_inputs["modality_tensor"].to(device=input_ids.device, dtype=torch.long)
        embeds = self.text_model.embed_tokens(input_ids)

        vision_patches = packed_inputs.get("vision_patches")
        if vision_patches is None:
            return embeds, modality

        token_grids = packed_inputs["vision_token_grids"].to(device=vision_patches.device, dtype=torch.long)
        vision = self.vision_embedding((vision_patches, token_grids))  # (total_tokens, hidden)

        # per-image token counts AFTER pixel-shuffle
        s = int(self.config.vision_config.pixel_shuffle_scale_factor)
        sizes = token_grids.prod(-1).div(s * s, rounding_mode="floor").tolist()
        offsets = packed_inputs.get("vision_token_offsets")
        lengths = packed_inputs.get("vision_token_lengths")

        if offsets is not None or lengths is not None:
            off = (
                offsets.to(device=vision.device, dtype=torch.long)
                if offsets is not None
                else torch.zeros(len(sizes), device=vision.device, dtype=torch.long)
            )
            ln = (
                lengths.to(device=vision.device, dtype=torch.long)
                if lengths is not None
                else torch.tensor(sizes, device=vision.device, dtype=torch.long)
            )

            # Honor per-image crop windows (after pixel shuffle) so we only splice back
            # the surviving virtual tokens instead of the full vision span.
            chunks = vision.split(sizes, dim=0)
            picked: list[torch.Tensor] = []
            for c, n, o, l in zip(chunks, sizes, off.tolist(), ln.tolist()):
                if n <= 0:
                    continue
                o = max(0, min(int(o), n))
                l = max(0, min(int(l), n - o))
                if l:
                    picked.append(c[o : o + l])
            vision = torch.cat(picked, 0) if picked else vision.new_zeros((0, vision.size(-1)))

        m = modality == ModalityType.image.value
        embeds = embeds.clone()
        embeds[m] = vision.to(device=embeds.device, dtype=embeds.dtype)

        return embeds, modality

    @auto_docstring
    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        packed_inputs: Optional[dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        """
        Forward pass with MRoPE position embeddings.

        Computes position embeddings once and passes them through all layers.

        Args:
            packed_inputs (`dict`, *optional*):
                Plain tensor payloads extracted from a TensorStream. When provided, it replaces the TensorStream path
                and requires `input_ids` for text tokens (or `text_token_ids` so `input_ids` can be rebuilt).
            modality_tensor (`torch.LongTensor`, *optional*):
                Modality identifiers aligned with the embedded sequence, shaped `(batch_size, seq_len)` and containing
                values from `ModalityType`. Automatically built from `packed_inputs` or treated as text-only when omitted.
        """

        output_attentions = kwargs.pop("output_attentions", None)

        # Resolve the input source (prefer packed_inputs > ids > embeds).
        modality_tensor: Optional[torch.Tensor] = None
        precomputed_position_ids: Optional[torch.Tensor] = None

        if packed_inputs is not None:
            inputs_embeds, modality_tensor = self.embed_packed_inputs(input_ids, packed_inputs)
            precomputed_position_ids = packed_inputs.get("position_ids")
            if precomputed_position_ids is not None:
                precomputed_position_ids = precomputed_position_ids.to(inputs_embeds.device)
        elif input_ids is not None:
            inputs_embeds = self.text_model.embed_tokens(input_ids)

        device = inputs_embeds.device
        batch_size, seq_len = inputs_embeds.shape[:2]

        # Ensure cache exists when requested
        if use_cache and past_key_values is None:
            cache_config = self.config.get_text_config() if hasattr(self.config, "get_text_config") else self.config
            past_key_values = DynamicCache(config=cache_config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_len, device=device)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)

        position_ids = position_ids if position_ids is not None else precomputed_position_ids
        if position_ids is None:
            position_ids = cache_position.view(1, -1).expand(batch_size, -1)

        if modality_tensor is None:
            modality_tensor = torch.full(
                (batch_size, seq_len), ModalityType.text.value, device=device, dtype=torch.long
            )
        else:
            modality_tensor = modality_tensor.to(device=device, dtype=torch.long)

        position_ids = position_ids.to(device=device)

        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(-1).expand(-1, -1, 3)
        if position_ids.shape[1] != seq_len:
            start_positions = position_ids[:, :1, 0]
            position_ids = torch.arange(seq_len, device=device).view(1, -1) + start_positions
            position_ids = position_ids.unsqueeze(-1).expand(-1, -1, 3)

        cos, sin = self.rotary_emb(position_ids, modality_tensor, hidden_states=inputs_embeds)

        decoder_position_ids = position_ids[..., 0] if position_ids.ndim == 3 else position_ids

        if not isinstance(attention_mask, dict):  # Prepare attention mask
            attention_mask = create_masks_for_generate(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=decoder_position_ids,
            )

        is_mask_dict = isinstance(attention_mask, dict)
        hidden_states = inputs_embeds
        all_attentions = [] if output_attentions else None

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
                output_attentions=output_attentions,
                **kwargs,
            )

            layer_outputs_is_tuple = isinstance(layer_outputs, tuple)
            hidden_states = layer_outputs[0] if layer_outputs_is_tuple else layer_outputs
            if output_attentions and layer_outputs_is_tuple:
                all_attentions.append(layer_outputs[1])

        hidden_states = self.text_model.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=(hidden_states,),
            attentions=tuple(all_attentions) if output_attentions else None,
        )


@auto_docstring
class IsaacForConditionalGeneration(Qwen3ForCausalLM, GenerationMixin):
    """Isaac multimodal model for conditional generation."""

    config_class = IsaacConfig
    _can_compile_fullgraph = False
    _tied_weights_keys = {"lm_head.weight": "model.text_model.embed_tokens.weight"}
    all_tied_weights_keys: dict[str, str] = {"lm_head.weight": "model.text_model.embed_tokens.weight"}

    def __init__(self, config: IsaacConfig):
        super().__init__(config)
        self.model = IsaacModel(config)  # Use our custom model
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None

    @auto_docstring
    @can_return_tuple
    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        packed_inputs: Optional[dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        """Run multimodal CausalLM forward, accepting packed vision/text inputs.

        Args:
            input_ids: Text token ids.
            packed_inputs (`dict`, *optional*):
                Packed vision/text payload from ``IsaacProcessor`` containing modality ids, MRoPE position ids, and
                vision patch tensors/grids (with optional offsets/lengths) used to rebuild embeddings.
            attention_mask: Attention mask or mask dict; created if not provided.
            position_ids: Optional 3D MRoPE positions; auto-derived when absent.
            past_key_values: Cache for decoding.
            inputs_embeds: Precomputed embeddings (bypass embedding layer).
            labels: Target ids for computing language modeling loss.
            use_cache: Whether to return caches.
            cache_position: Positions for cache-aware generation.

        Returns:
            CausalLMOutputWithPast: logits, optional loss, caches, hidden states, attentions.
        """
        output_attentions = kwargs.pop("output_attentions", None)

        if position_ids is None and packed_inputs is not None:
            pos_3d = packed_inputs.get("position_ids")
            if pos_3d is not None:
                position_ids, self.rope_deltas = self.get_rope_index(
                    position_ids=pos_3d,
                    attention_mask=attention_mask,
                )

        elif position_ids is None and cache_position is not None and self.rope_deltas is not None:
            if input_ids is not None:
                base_position_ids = torch.arange(input_ids.size(1), device=input_ids.device)[None, :, None].expand(
                    input_ids.size(0), -1, 3
                )
            else:
                batch_size, seq_len = inputs_embeds.shape[:2]
                dummy_ids = torch.zeros((batch_size, seq_len), device=inputs_embeds.device, dtype=torch.long)
                base_position_ids = torch.arange(dummy_ids.size(1), device=dummy_ids.device)[None, :, None].expand(
                    dummy_ids.size(0), -1, 3
                )

            rope_delta = (cache_position[0] + self.rope_deltas).to(base_position_ids.device)
            if not isinstance(rope_delta, int):
                rope_delta = rope_delta.repeat_interleave(base_position_ids.shape[0] // rope_delta.shape[0], dim=0)
            position_ids = base_position_ids.add(rope_delta)

        outputs = self.model(
            input_ids=input_ids,
            packed_inputs=packed_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
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
            attentions=outputs.attentions if output_attentions else None,
        )

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.set_input_embeddings(value)
        vocab_size = getattr(value, "num_embeddings", None)
        self.config.vocab_size = vocab_size
        self.model.config.vocab_size = vocab_size
        self.model.text_model.config.vocab_size = vocab_size
        if self.lm_head.weight.shape[0] != vocab_size:
            self.lm_head = nn.Linear(self.config.hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.model.text_model.embed_tokens.weight

    def get_rope_index(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute (position_ids_3d, rope_deltas) without TensorStream.

        - If `position_ids` is provided, it must be shape (B, L, 3).
        - Else, if `input_ids` is provided, position ids are synthesized as (B, L, 3).
        - `rope_deltas` is (B, 1) used to advance positions during decode.
        """

        if position_ids is None:
            pos_3d = torch.arange(input_ids.size(1), device=input_ids.device)[None, :, None].expand(
                input_ids.size(0), -1, 3
            )
        else:
            pos_3d = position_ids
            if pos_3d.ndim != 3 or pos_3d.size(-1) != 3:
                raise ValueError(
                    f"`position_ids` must have shape (batch, seq_len, 3) for MRoPE; got shape {tuple(pos_3d.shape)}."
                )

        B, L, _ = pos_3d.shape
        m_per_batch = pos_3d.amax(dim=(1, 2))

        if attention_mask is None:
            seq_lens = torch.full((B,), L, device=pos_3d.device, dtype=m_per_batch.dtype)
        else:
            seq_lens = attention_mask.eq(1).sum(dim=-1).to(dtype=m_per_batch.dtype, device=m_per_batch.device)

        rope_deltas = (m_per_batch + 1 - seq_lens).to(dtype=pos_3d.dtype).unsqueeze(1)
        return pos_3d, rope_deltas

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        packed_inputs: Optional[dict[str, torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            **kwargs,
        )
        if packed_inputs is None:
            return model_inputs

        cache_position = model_inputs.get("cache_position", cache_position)
        first_step = cache_position is None or cache_position[0] == 0
        model_inputs["packed_inputs"] = packed_inputs if first_step else None
        model_inputs["position_ids"] = None

        return model_inputs

    @classmethod
    def can_generate(cls) -> bool:
        return True


__all__ = [
    "IsaacConfig",
    "IsaacModel",
    "IsaacPreTrainedModel",  # noqa: F822
    "IsaacForConditionalGeneration",
    "IsaacImageProcessorFast",
    "IsaacProcessor",
]
