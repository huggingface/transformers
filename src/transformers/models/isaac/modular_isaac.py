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
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

from ...utils.import_utils import (
    is_perceptron_available,
    is_torch_available,
    is_torchdynamo_compiling,
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


if is_perceptron_available():
    from perceptron.tensorstream.ops import (
        compute_mrope_pos_tensor,
        modality_mask,
        reconstruct_tensor_stream_from_compact_dict,
        tensor_stream_token_view,
    )
    from perceptron.tensorstream.ops import (
        slice as ts_slice,
    )
    from perceptron.tensorstream.tensorstream import (
        Event,
        Stream,
        TensorStream,
        TextType,
        VisionType,
        create_stream,
        group_streams,
    )
else:
    ts_slice = None
    Event = None
    Stream = None
    TensorStream = None
    TextType = None
    VisionType = None
    create_stream = None
    group_streams = None


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
    ChannelDimension,
    PILImageResampling,
)
from ...masking_utils import create_masks_for_generate, eager_mask, packed_sequence_mask_function, sdpa_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import rope_config_validation
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...models.auto.modeling_auto import AutoModel
from ...models.auto.tokenization_auto import AutoTokenizer
from ...models.qwen3.configuration_qwen3 import Qwen3Config
from ...models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3PreTrainedModel
from ...processing_utils import ProcessorMixin, Unpack
from ...utils import TensorType, auto_docstring

# Vision preprocessing constants
from ...utils.constants import IMAGENET_STANDARD_MEAN as VISION_MEAN
from ...utils.constants import IMAGENET_STANDARD_STD as VISION_STD
from ...utils.generic import TransformersKwargs, can_return_tuple, check_model_inputs
from ..qwen2_5_vl import modeling_qwen2_5_vl as qwen2_5_vl_modeling
from ..siglip2.configuration_siglip2 import Siglip2VisionConfig
from ..siglip2.modeling_siglip2 import (
    Siglip2Attention,
    Siglip2Encoder,
    Siglip2EncoderLayer,
)


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
            self._attn_implementation = "eager"


class IsaacImageProcessorKwargs(ImagesKwargs, total=False):
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
    valid_kwargs = IsaacImageProcessorKwargs
    unused_kwargs = ["size", "do_center_crop", "crop_size"]

    do_resize = True
    size: Optional[SizeDict] = None
    default_to_square: Optional[bool] = None
    do_center_crop = False
    crop_size: Optional[SizeDict] = None
    patch_size: Optional[int] = 16
    max_num_patches: Optional[int] = 256
    min_num_patches: Optional[int] = None
    pixel_shuffle_scale: Optional[int] = 1
    do_pad = False
    pad_size: Optional[SizeDict] = None
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = list(VISION_MEAN)
    image_std = list(VISION_STD)
    do_convert_rgb = True
    return_tensors = None
    data_format = ChannelDimension.FIRST
    input_data_format = None
    device = None
    disable_grouping = False
    size_divisor: Optional[int] = None

    def __init__(
        self,
        **kwargs: Unpack[IsaacImageProcessorKwargs],
    ) -> None:
        super().__init__(**kwargs)

        pixel_shuffle_scale = 1 if self.pixel_shuffle_scale is None else int(self.pixel_shuffle_scale)
        if pixel_shuffle_scale < 1:
            raise ValueError("`pixel_shuffle_scale` must be >= 1")
        self.pixel_shuffle_scale = pixel_shuffle_scale

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
        interpolation: Optional[Any] = None,
        antialias: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if size.height is None or size.width is None:
            raise ValueError("IsaacImageProcessorFast requires explicit `height` and `width` when resizing.")

        resize_mode: Any = interpolation
        if hasattr(resize_mode, "value"):
            resize_mode = resize_mode.value
        elif hasattr(resize_mode, "name"):
            resize_mode = resize_mode.name.lower()
        elif resize_mode is None:
            resize_mode = "bilinear"

        if isinstance(resize_mode, str):
            mode_key = resize_mode.lower()
        else:
            mode_key = resize_mode

        resize_kwargs: dict[str, Any] = {}
        if mode_key in {"linear", "bilinear", "bicubic", "trilinear"}:
            resize_kwargs["align_corners"] = False

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
        size: Optional[SizeDict],
        interpolation: Optional[Any],
        do_center_crop: bool,
        crop_size: Optional[SizeDict],
        do_rescale: Optional[bool],
        rescale_factor: Optional[float],
        do_normalize: Optional[bool],
        image_mean: Optional[Union[float, Sequence[float]]],
        image_std: Optional[Union[float, Sequence[float]]],
        disable_grouping: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        do_pad: Optional[bool] = None,
        pad_size: Optional[SizeDict] = None,
        *,
        patch_size: Optional[int] = None,
        max_num_patches: Optional[int] = None,
        min_num_patches: Optional[int] = None,
        pixel_shuffle_scale: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        if do_center_crop:
            raise ValueError("`do_center_crop` is not supported by IsaacImageProcessorFast.")
        if do_pad:
            raise ValueError("`do_pad` is not supported by IsaacImageProcessorFast.")

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_patches_grouped: dict[tuple[int, ...], torch.Tensor] = {}
        token_grids_grouped: dict[tuple[int, ...], torch.Tensor] = {}
        virtual_dims_grouped: dict[tuple[int, ...], torch.Tensor] = {}
        real_dims_grouped: dict[tuple[int, ...], torch.Tensor] = {}

        for shape, stacked_images in grouped_images.items():
            if stacked_images.ndim != 4:
                raise ValueError("Expected batched channel-first image tensors.")

            batch_size, channels, original_height, original_width = stacked_images.shape

            if bool(self.do_convert_rgb) and channels == 1:
                stacked_images = stacked_images.repeat(1, 3, 1, 1)
                channels = 3

            if original_height * original_width > self.MAX_PIXELS:
                raise ValueError(f"Image (w={original_width}, h={original_height}) > MAX=`{self.MAX_PIXELS}`")

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
                    raise ValueError("Image dimensions must be divisible by patch_size when resize is disabled.")
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

            nhwc_images = image_batch.permute(0, 2, 3, 1)
            nhwc_images = _compute_residual_p_frames(nhwc_images, is_p_frame=[False] * batch_size)

            patches = patchify_vision(nhwc_images, patch_size=patch_size)
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
                    "Spatial dimensions must be divisible by pixel_shuffle_scale when pixel shuffle is enabled."
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

            processed_patches_grouped[shape] = patches
            token_grids_grouped[shape] = token_grid
            virtual_dims_grouped[shape] = virtual_dim
            real_dims_grouped[shape] = real_dim

        patches_slices = reorder_images(processed_patches_grouped, grouped_images_index)
        token_grid_slices = reorder_images(token_grids_grouped, grouped_images_index)
        virtual_dim_slices = reorder_images(virtual_dims_grouped, grouped_images_index)
        real_dim_slices = reorder_images(real_dims_grouped, grouped_images_index)

        patches_tensor = torch.stack(patches_slices, dim=0)
        token_grids_tensor = torch.stack(token_grid_slices, dim=0)
        virtual_dims_tensor = torch.stack(virtual_dim_slices, dim=0)
        real_dims_tensor = torch.stack(real_dim_slices, dim=0)

        return BatchFeature(
            data={
                "patches": patches_tensor,
                "token_grids": token_grids_tensor,
                "virtual_pixel_size": virtual_dims_tensor,
                "real_pixel_size": real_dims_tensor,
            },
            tensor_type=return_tensors,
        )


def document_mask_function_from_cu_seqlens(cu_seqlens: Optional[torch.Tensor]) -> Optional[Callable]:
    """Return a mask function that blocks cross-document attention from packed ``cu_seqlens``.

    The returned callable matches the signature expected by ``masking_utils`` mask factories and
    yields ``True`` only when query/key positions belong to the same packed segment.
    """

    if cu_seqlens is None:
        return None

    if cu_seqlens.numel() < 2:
        return None

    seq_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    if seq_sizes.numel() == 0:
        return None

    total_tokens = int(seq_sizes.sum().item())
    seg_ids = torch.repeat_interleave(torch.arange(seq_sizes.numel(), device=cu_seqlens.device), seq_sizes)
    packed_sequence_mask = seg_ids.view(1, total_tokens)
    return packed_sequence_mask_function(packed_sequence_mask)


def ensure_document_attention_mask(
    attention_mask: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    total_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    return_mask_function: bool = False,
) -> Optional[Union[torch.Tensor, Callable]]:
    """Return the provided mask, a callable mask from ``cu_seqlens``, or ``None``.

    ``return_mask_function=True`` yields a callable suitable for ``masking_utils``; otherwise
    ``None`` is returned when no explicit ``attention_mask`` is provided. The legacy additive mask
    has been removed in favor of the callable-based path.
    """

    if attention_mask is not None:
        return attention_mask

    if cu_seqlens is None:
        return None

    if return_mask_function:
        return document_mask_function_from_cu_seqlens(cu_seqlens)

    return None


class IsaacVisionEmbeddings(nn.Module):
    """Adapter around SigLIP2 vision embeddings that consumes packed patch sequences."""

    def __init__(self, config: IsaacVisionConfig):
        super().__init__()
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

    def forward(self, seq_patches: torch.Tensor, spatial_shapes: torch.Tensor) -> torch.Tensor:
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

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor:
        """
        Resize positional embeddings to image-specific size and pad to a fixed size.

        Args:
            positional_embeddings (`torch.Tensor`):
                Position embeddings of shape (height, width, embed_dim)
            spatial_shapes (`torch.LongTensor`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
            max_length (`int`):
                Maximum length of the positional embeddings to pad resized positional embeddings to

        Returns:
            `torch.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
        """
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = torch.empty(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 on CPU because antialias is not supported for bfloat16/float16 on CPU
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for i in range(batch_size):
            # (1, dim, height, width) -> (1, dim, target_height, target_width)
            height, width = spatial_shapes[i]
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
            resized_embeddings = resized_embeddings.reshape(embed_dim, height * width).transpose(0, 1)

            # Cast to original dtype
            resized_embeddings = resized_embeddings.to(source_dtype)

            resulted_positional_embeddings[i, : height * width] = resized_embeddings
            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings

    def _pack_to_batch(
        self,
        seq_patches: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        if seq_patches.ndim != 2:
            raise ValueError("`seq_patches` is expected to be 2D (total_patches, patch_dim).")
        if spatial_shapes.ndim != 2 or spatial_shapes.size(-1) != 2:
            raise ValueError("`spatial_shapes` must have shape (num_images, 2) with (height_tokens, width_tokens).")

        seq_lengths = spatial_shapes.long().prod(dim=-1)
        total_patches = int(seq_lengths.sum().item())
        if total_patches != seq_patches.size(0):
            raise ValueError(
                "Mismatch between packed patches and spatial shapes: got "
                f"{seq_patches.size(0)} patches but spatial shapes imply {total_patches}."
            )

        batch_size = spatial_shapes.size(0)
        if batch_size == 0:
            return None, seq_lengths

        max_length = int(seq_lengths.max().item())
        patch_dim = seq_patches.size(-1)
        device = seq_patches.device

        packed_pixel_values = seq_patches.new_zeros((batch_size, max_length, patch_dim), device=device)

        start = 0
        for batch_idx, length in enumerate(seq_lengths.tolist()):
            if length == 0:
                continue
            end = start + length
            packed_pixel_values[batch_idx, :length] = seq_patches[start:end]
            start = end

        return packed_pixel_values, seq_lengths

    def _unpack_from_batch(self, embeddings: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        output_chunks: list[torch.Tensor] = []
        for batch_idx, length in enumerate(seq_lengths.tolist()):
            if length == 0:
                continue
            output_chunks.append(embeddings[batch_idx, :length])

        if not output_chunks:
            return embeddings.new_zeros((0, embeddings.size(-1)))

        return torch.cat(output_chunks, dim=0)


class IsaacVisionAttention(Siglip2Attention):
    """Custom attention that supports variable-length sequences with flash attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        is_causal: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ):
        # Ignore unused arguments for interface compatibility
        _ = position_ids
        _ = past_key_value
        _ = is_causal
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)

        batch_size, seq_length, embed_dim = hidden_states.shape
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        if not queries.is_contiguous():
            queries = queries.contiguous()
        if not keys.is_contiguous():
            keys = keys.contiguous()
        if not values.is_contiguous():
            values = values.contiguous()

        L = queries.size(0)
        if max_seqlen is not None:
            max_q = max_k = int(max_seqlen)
        else:
            max_q = max_k = self._max_from_cu(cu_seqlens, L)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS["sdpa"]
        if self.config._attn_implementation != "sdpa":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        dropout = 0.0 if not self.training else self.dropout
        attention_kwargs: dict[str, Any] = {
            "is_causal": False,
            "scaling": self.scale,
            "dropout": dropout,
        }
        if cu_seqlens is not None:
            attention_kwargs["cu_seq_lens_q"] = cu_seqlens
            attention_kwargs["cu_seq_lens_k"] = cu_seqlens
        if max_seqlen is not None:
            attention_kwargs["max_length_q"] = max_q
            attention_kwargs["max_length_k"] = max_k
        if output_attentions:
            attention_kwargs["output_attentions"] = True

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            **attention_kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()

        # Align projection inputs with parameter dtype to avoid mixed-dtype matmul errors
        out_proj_dtype = self.out_proj.weight.dtype
        if attn_output.dtype != out_proj_dtype:
            attn_output = attn_output.to(out_proj_dtype)

        attn_output = self.out_proj(attn_output)
        if attn_output.dtype != hidden_states.dtype:
            attn_output = attn_output.to(hidden_states.dtype)

        return attn_output, attn_weights

    @staticmethod
    def _max_from_cu(cu: Optional[torch.Tensor], fallback: int) -> int:
        if cu is None or cu.numel() < 2:
            return fallback
        return int((cu[1:] - cu[:-1]).max().item())


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
        attention_mask = ensure_document_attention_mask(
            attention_mask,
            cu_seqlens,
            hidden_states.size(1),
            hidden_states.dtype,
            hidden_states.device,
            return_mask_function=False,
        )

        # Run attention directly so variable-length metadata reaches FlashAttention.
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_attentions=output_attentions,
            **kwargs,
        )
        if isinstance(attn_outputs, tuple):
            attn_output, attn_weights = attn_outputs
        else:
            attn_output, attn_weights = attn_outputs, None
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if output_attentions:
            return hidden_states, attn_weights
        return hidden_states


class IsaacVisionEncoder(Siglip2Encoder):
    """Encoder using Isaac encoder layers with variable-length attention support."""

    def __init__(self, config: IsaacVisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([IsaacVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    @can_return_tuple
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        attention_mask = ensure_document_attention_mask(
            attention_mask,
            cu_seqlens,
            inputs_embeds.size(1),
            inputs_embeds.dtype,
            inputs_embeds.device,
            return_mask_function=False,
        )

        return super().forward(
            inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            **kwargs,
        )


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
    if device is None:
        device = seq_sizes.device

    scale_factor = int(scale_factor)
    if scale_factor < 2:
        raise ValueError("`scale_factor` must be ≥ 2")

    # Safety: all spatial dims must be divisible by the scale factor
    # Cannot run under torch compile fullgraph mode hence
    if not is_torchdynamo_compiling():
        if not ((token_grids[:, 0] % scale_factor == 0).all() and (token_grids[:, 1] % scale_factor == 0).all()):
            raise AssertionError(
                "Every (H,W) in `token_grids` must be divisible by "
                f"scale_factor={scale_factor}, got {token_grids.tolist()}"
            )

    gather_chunks: list[torch.Tensor] = []
    tok_offset = 0

    for seq_len, (h, w) in zip(seq_sizes.tolist(), token_grids.tolist(), strict=False):
        # Build the (H, W) grid of flat indices for this image
        grid = torch.arange(seq_len, device=device, dtype=torch.int64) + tok_offset
        grid = grid.view(h, w)  # (H, W)

        # -------- identical ordering to your fixed-res routine --------
        # Step 1: split width into blocks of scale_factor
        grid = grid.view(h, w // scale_factor, scale_factor)  # (H, W/scale_factor, scale_factor)
        # Step 2: now split height into blocks of scale_factor
        grid = grid.view(h // scale_factor, scale_factor, w // scale_factor, scale_factor)
        # (H/scale_factor, scale_factor, W/scale_factor, scale_factor)
        # Step 3: final permutation to (H/scale_factor, W/scale_factor, scale_factor, scale_factor)
        grid = grid.permute(0, 2, 1, 3).contiguous()  # (H/scale_factor, W/scale_factor, scale_factor, scale_factor)
        # Step 4: each (scale_factor, scale_factor) block forms one output token
        gather_chunks.append(grid.reshape(-1, scale_factor * scale_factor))
        # (H*W / scale_factor**2, scale_factor**2)

        tok_offset += seq_len

    # Concatenate over all images in the packed batch
    gather_idx = torch.cat(gather_chunks, dim=0)  # (Σ_i HᵢWᵢ/scale_factor**2, scale_factor**2)
    return gather_idx


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
    keep_batch_dim = x.dim() == 3
    if keep_batch_dim:
        if x.size(0) != 1:
            raise AssertionError("Packed sequence is expected to have batch_size == 1")
        x_ = x.squeeze(0)  # (seq, embed)
    else:
        x_ = x  # (seq, embed)

    embed_dim = x_.size(-1)
    scale_factor = int(scale_factor)

    # Calculate seq_sizes from token_grids
    seq_sizes = torch.prod(token_grids, dim=-1)

    # Build index map and gather in one go
    gather_idx = create_pixel_shuffle_index_map(
        seq_sizes=seq_sizes,
        token_grids=token_grids,
        scale_factor=scale_factor,
        device=x_.device,
    )  # (new_seq, scale_factor**2)

    # Gather → (new_seq, scale_factor**2, embed_dim)
    gathered = x_[gather_idx]  # fancy indexing keeps gradient

    # Merge the scale_factor**2 group dimension into channels to finish the shuffle
    out = gathered.reshape(gathered.size(0), embed_dim * scale_factor * scale_factor)

    # Restore batch dimension if needed
    if keep_batch_dim:
        out = out.unsqueeze(0)
    return out


class IsaacVisionTransformer(nn.Module):
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

        # Add a pseudo batch dimension for the encoder
        hidden_states = hidden_states.unsqueeze(0)

        # Generate cumulative sequence lengths for variable-length attention
        cu_seqlens = torch.zeros(seq_sizes.size(0) + 1, dtype=torch.int32, device=hidden_states.device)
        cu_seqlens[1:] = seq_sizes.cumsum(0)
        max_seqlen = int(seq_sizes.max().item()) if seq_sizes.numel() > 0 else 0

        # Pass through encoder with variable-length attention parameters
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            return_dict=True,
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


class IsaacVisionEmbedding(nn.Module):
    """Vision embedding wrapper exposing tower and projector."""

    def __init__(self, config: IsaacConfig):
        super().__init__()
        vision_cfg = config.vision_config
        hidden_dim = vision_cfg.hidden_size * (vision_cfg.pixel_shuffle_scale_factor**2)

        self.vision_tower = IsaacVisionTransformer(vision_cfg)
        self.multimodal_projector = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, config.hidden_size, bias=False),
        )

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
        # Scale up
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


def patchify_vision(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    r"""Convert normalized images into flattened ViT-style patches.

    Args:
        image (`torch.Tensor`):
            Tensor of shape `(num_images, height, width, channels)`.
        patch_size (`int`):
            Edge length of the square patches

    Returns:
        `torch.Tensor`:
            Patch tensor where each position stores the flattened pixels belonging to that patch.

    Raises:
        ValueError: If `height` or `width` is not divisible by `patch_size`.
    """
    num_images, height, width, channels = image.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"Dimensions of images {image.shape} are not divisible by patch_size={patch_size}.")
    patches = image.reshape(num_images, height // patch_size, patch_size, width // patch_size, patch_size, channels)
    patches = patches.permute(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(
        num_images, height // patch_size, width // patch_size, channels * patch_size * patch_size
    )
    return patches


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
        self._rope_parameters: Optional[dict[str, Any]] = None

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        super().__init__(**kwargs)

        if self._rope_scaling is None:
            self._rope_scaling = getattr(self.text_config, "rope_scaling", None)
        else:
            self.text_config.rope_scaling = self._rope_scaling

        # Keep rope parameters alias in sync with upstream expectations
        self._rope_parameters = self._rope_scaling

        # Mirror frequently accessed Qwen3 attributes at the composite config level for BC.
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.num_attention_heads = self.text_config.num_attention_heads
        self.head_dim = self.text_config.head_dim
        self.hidden_act = self.text_config.hidden_act
        self.use_cache = self.text_config.use_cache
        self.rope_theta = self.text_config.rope_parameters["rope_theta"]

        # Validate rotary parameters now that they have been mirrored locally.
        rope_config_validation(self)

        self.layer_types = getattr(self.text_config, "layer_types", None)
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        # Handle vision config - either dict or IsaacVisionConfig instance
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif isinstance(vision_config, IsaacVisionConfig):
            self.vision_config = vision_config
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        # Vision normalization parameters
        self.vision_rescale_factor = float(vision_rescale_factor)

        # Processing parameters
        self.max_sequence_length = max_sequence_length
        self.vision_token = vision_token

        # Default and propagate attention implementation
        attn_impl = getattr(self, "_attn_implementation", None)
        if attn_impl is None:
            attn_impl = "eager"
            self._attn_implementation = attn_impl
        if hasattr(self, "text_config") and self.text_config is not None:
            self.text_config._attn_implementation = attn_impl
        if hasattr(self, "vision_config") and self.vision_config is not None:
            self.vision_config._attn_implementation = attn_impl

    @property
    def rope_scaling(self):
        if hasattr(self, "text_config") and self.text_config is not None:
            return getattr(self.text_config, "rope_scaling", None)
        return self._rope_scaling

    @rope_scaling.setter
    def rope_scaling(self, value):
        self._rope_scaling = value
        if hasattr(self, "text_config") and self.text_config is not None:
            self.text_config.rope_scaling = value

    @property
    def rope_parameters(self) -> dict[str, Any] | None:
        """Alias introduced upstream for rope scaling dictionaries."""
        value = self._rope_parameters
        if value is None:
            value = self.rope_scaling
        if value is None:
            return {"rope_type": "default"}
        return value

    @rope_parameters.setter
    def rope_parameters(self, value: dict[str, Any] | None) -> None:
        self._rope_parameters = value
        self.rope_scaling = value

    def to_dict(self):
        output = super().to_dict()
        # Ensure nested configs round-trip through dict serialization
        if hasattr(self, "text_config") and self.text_config is not None:
            output["text_config"] = self.text_config.to_dict()
        if hasattr(self, "vision_config") and self.vision_config is not None:
            output["vision_config"] = self.vision_config.to_dict()
        return output

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "_attn_implementation":
            if hasattr(self, "text_config") and self.text_config is not None:
                setattr(self.text_config, "_attn_implementation", value)
            if hasattr(self, "vision_config") and self.vision_config is not None:
                setattr(self.vision_config, "_attn_implementation", value)


# ============================================================================
# Processor Components
# ============================================================================


def create_text_event(tokenizer: AutoTokenizer, text: str, time: float = 0.0) -> Event:
    r"""Wrap a text into an `Event` compatible with the multimodal TensorStream.

    Args:
        tokenizer (`AutoTokenizer`):
            Tokenizer used to convert text into model vocabulary ids.
        text (`str`):
            Plain-text fragment to encode.
        time (`float`, *optional*, defaults to 0.0):
            Timeline coordinate associated with the event. Both start and end times use the same value because text
            segments are instantaneous in the scheduler.

    Returns:
        `Event`: Event carrying a `(num_tokens, 1)` tensor of token ids with matching
        metadata so that downstream processors can compute modality-specific embeddings.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").squeeze(0)

    # Calculate dimensions for the event
    num_tokens = len(tokens)
    dims_virtual = [num_tokens, 1]  # [sequence_length, 1]
    dims_real = dims_virtual.copy()

    # Ensure tokens has the right shape for tensor_stream_token_view
    # It expects a 2D tensor where sum(dim=-1) gives the token IDs
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(-1)

    return Event(
        data=tokens,
        type=TextType.text,
        time=(time, time),
        dims_virtual=dims_virtual,
        dims_real=dims_real,
        idx_range=(0, num_tokens),
    )


# ============================================================================
# Processor
# ============================================================================


class IsaacProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = ("IsaacImageProcessorFast",)
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

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
        if tokenizer is None:
            raise ValueError("`tokenizer` must be provided to initialize IsaacProcessor.")

        if isinstance(config, dict):
            config = IsaacConfig(**config)

        if config is not None:
            max_sequence_length = config.max_sequence_length
            vision_token = config.vision_token
            rescale_factor = config.vision_rescale_factor

        resolved_rescale_factor = float(rescale_factor) if rescale_factor is not None else float(1 / 255)

        if config is not None:
            config.vision_rescale_factor = resolved_rescale_factor

        self.image_processor = image_processor

        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self.config = config

        # Mirror tokenizer chat template so ProcessorMixin.apply_chat_template works.
        self.chat_template = getattr(self.tokenizer, "chat_template", None)

        self.vision_token = vision_token
        self.max_sequence_length = max_sequence_length

    def build_event_stream_simple(
        self,
        text: str,
        images: Optional[list[Image]] = None,
    ) -> Stream:
        events = []
        # Process text and images
        # Find all occurrences of vision token

        pattern = re.escape(self.vision_token)
        parts = re.split(f"({pattern})", text)  # Keep the delimiter in the result

        image_idx = 0
        for current_time, part in enumerate(parts):
            if part == self.vision_token:
                # Replace vision token with image event
                if images is None or image_idx >= len(images):
                    raise ValueError("Encountered vision token without a corresponding image.")

                features = self.image_processor(
                    images=images[image_idx],
                    return_tensors=TensorType.PYTORCH,
                )

                patches = features["patches"][0]  # (H_tokens, W_tokens, embed)
                virtual_dims = features["virtual_pixel_size"][0].tolist()
                real_dims = features["real_pixel_size"][0].tolist()

                vision_event = Event(
                    data=patches.reshape(-1, patches.shape[-1]),
                    type=VisionType.image,
                    time=(current_time, current_time),
                    dims_virtual=virtual_dims,
                    dims_real=real_dims,
                    idx_range=(0, math.prod(virtual_dims)),
                )
                events.append(vision_event)
                image_idx += 1
            elif part:  # Non-empty text part
                # tokens = self.text_processor.tokenize(part, add_special_tokens=False)
                text_event = create_text_event(self.tokenizer, part, time=current_time)
                events.append(text_event)

        # Create stream without scheduling (events already in order)
        return create_stream(events, priority=[TextType.text, VisionType.image], schedule=True)

    def __call__(
        self,
        text: Union[str, list[str]],
        images: Optional[Union[Image, list[Image]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        **kwargs,
    ) -> BatchFeature:
        """
        Process text and images into TensorStream format.
        Args:
            text: Input text or list of texts with vision tokens
            images: PIL image or list of images (optional)
            return_tensors: Format for output tensors

        Returns:
            BatchFeature with input_ids and tensor_stream
        """
        # Normalize inputs to lists
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        if images is not None:
            if isinstance(images, Image):
                images_list = [images]
            else:
                images_list = images
        else:
            images_list = None

        if len(texts) != 1:
            raise ValueError("IsaacProcessor currently supports batch_size=1")
        if images_list is not None:
            # Count vision tokens in text to validate image count
            vision_token_count = texts[0].count(self.vision_token)
            if vision_token_count != len(images_list):
                raise ValueError(
                    f"Number of {self.vision_token} tokens in text ({vision_token_count}) "
                    f"must match number of images ({len(images_list)})"
                )

        # Build event stream
        stream = self.build_event_stream_simple(
            text=texts[0],
            images=images_list,
        )

        # Create TensorStream
        tensor_stream = TensorStream([stream])

        # Slice to max length if needed
        _, T = tensor_stream.shape
        if T > self.max_sequence_length:
            tensor_stream = ts_slice(tensor_stream, start=T - self.max_sequence_length, end=T)

        # Get token view
        tokens = tensor_stream_token_view(tensor_stream)
        if return_tensors in (TensorType.PYTORCH, "pt"):
            input_ids = torch.as_tensor(tokens, dtype=torch.long)
        else:
            input_ids = tokens

        data = {
            "input_ids": input_ids,
            "tensor_stream": tensor_stream,
        }

        return BatchFeature(data=data)


# ============================================================================
# Model
# ============================================================================


def compute_position_ids_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    r"""Create 3D positional indices for token input.

    Args:
        input_ids (`torch.Tensor`):
            Tensor of shape `(batch_size, seq_len)` containing token ids.

    Returns:
        `torch.Tensor`: Positional indices with shape `(batch_size, seq_len, 3)` where each channel duplicates the
        1D position so it can be consumed by the 3-axis MRoPE rotary embedding.
    """
    batch_size, seq_length = input_ids.shape
    position_ids = torch.arange(seq_length, device=input_ids.device)
    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
    position_ids = position_ids.unsqueeze(2).expand(-1, -1, 3)  # Add 3D for MRoPE
    return position_ids


class IsaacRotaryEmbedding(nn.Module):
    EXTRA_ROPE_KEYS = {"mrope_section", "mrope_interleaved"}

    def __init__(self, config: IsaacConfig, device=None):
        super().__init__()

        rope_source_cfg = config.get_text_config() if hasattr(config, "get_text_config") else config
        rope_scaling = getattr(rope_source_cfg, "rope_scaling", None) or {}

        sanitized_scaling = {k: v for k, v in rope_scaling.items() if k not in self.EXTRA_ROPE_KEYS}
        config_for_rope = copy.copy(rope_source_cfg)
        config_for_rope.rope_scaling = sanitized_scaling if sanitized_scaling else None

        init_device = device if device is not None and getattr(device, "type", None) != "meta" else None
        self._qwen_rotary = qwen2_5_vl_modeling.Qwen2_5_VLRotaryEmbedding(config_for_rope, device=init_device)

        rotary_half_dim = self._qwen_rotary.inv_freq.shape[0]
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
        if len(section) != 3:
            raise ValueError("`mrope_section` must contain exactly three elements (temporal, height, width)")
        if sum(section) != rotary_half_dim:
            raise ValueError(
                f"`mrope_section` must sum to the rotary half-dimension ({rotary_half_dim}). Received {section}."
            )
        return section

    def _combine_axes(self, tensor: torch.Tensor) -> torch.Tensor:
        split_sections = tuple(self.mrope_section * 2)
        chunks = tensor.split(split_sections, dim=-1)
        return torch.cat([chunk[i % 3] for i, chunk in enumerate(chunks)], dim=-1)

    @property
    def inv_freq(self) -> torch.Tensor:
        return self._qwen_rotary.inv_freq

    def forward(
        self,
        position_ids: torch.Tensor,
        modality_tensor: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 3 or position_ids.size(-1) != 3:
            raise ValueError("`position_ids` must have shape (batch, seq_len, 3) for MRoPE")
        if modality_tensor.shape != position_ids.shape[:2]:
            raise ValueError("`modality_tensor` must align with the first two dims of `position_ids`")

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
            image_value = VisionType.image.value if VisionType is not None else 1
            not_spatial = modality_tensor != image_value
            if not_spatial.any():
                data_1d = pos[not_spatial][..., 0].unsqueeze(-1)
                pos[not_spatial] = data_1d.expand(-1, pos.shape[-1])

            pos_axes = pos.permute(2, 0, 1).contiguous()

        cos_axes, sin_axes = self._qwen_rotary(hidden_states, pos_axes)

        cos_axes = cos_axes.to(hidden_states.dtype)
        sin_axes = sin_axes.to(hidden_states.dtype)

        cos_combined = self._combine_axes(cos_axes)
        sin_combined = self._combine_axes(sin_axes)

        return cos_combined, sin_combined


class IsaacModel(Qwen3PreTrainedModel):
    supports_gradient_checkpointing = True
    _can_compile_fullgraph = False
    _supports_flex_attn = False
    # Expose tied-weights mapping even if empty for base model tests.
    all_tied_weights_keys: dict[str, str] = {}

    def __init__(self, config: IsaacConfig):
        Qwen3PreTrainedModel.__init__(self, config)

        text_cfg_source = config.text_config
        text_cfg = copy.deepcopy(text_cfg_source)
        self.text_model = AutoModel.from_config(text_cfg)
        # Ensure downstream callers observe the composed config
        self.text_model.config = config

        self.rotary_emb = IsaacRotaryEmbedding(config, device=self.device)

        if config.vision_config is None:
            raise ValueError("IsaacConfig should always have vision_config")

        self.vision_embedding = IsaacVisionEmbedding(config)

        # Dispatch table for TensorStream balanced embedding (text + vision)
        self.embed_fns = {
            TextType: self.embed_text_tokens,
            VisionType: self.embed_vision,
        }

        # Keep track of config attributes that downstream utilities may query directly on the model.
        self.max_sequence_length = config.max_sequence_length
        self.vision_rescale_factor = config.vision_rescale_factor
        self.vision_token = config.vision_token

        # Initialize weights and parallel plans (including tp_plan from the text model)
        self.post_init()

        # Respect config-specified gradient checkpointing
        if getattr(config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize vision modules with SigLIP-style defaults while preserving text init via super()."""
        super()._init_weights(module)

        if isinstance(module, IsaacVisionEmbeddings):
            nn.init.xavier_uniform_(module.patch_embedding.weight)
            if module.patch_embedding.bias is not None:
                nn.init.zeros_(module.patch_embedding.bias)
            hidden_size = module.embed_dim
            nn.init.normal_(module.position_embedding.weight, mean=0.0, std=1.0 / math.sqrt(hidden_size))
            return

        if isinstance(module, IsaacVisionAttention):
            for proj in (module.q_proj, module.k_proj, module.v_proj, module.out_proj):
                nn.init.xavier_uniform_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
            return

        # Initialize only the multimodal projector linears to avoid touching text weights.
        if isinstance(module, nn.Linear):
            projector = getattr(self, "vision_embedding", None)
            if projector is not None and isinstance(getattr(projector, "multimodal_projector", None), nn.Sequential):
                if any(module is layer for layer in projector.multimodal_projector if isinstance(layer, nn.Linear)):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                    return

        if isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

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
    def layers(self) -> nn.ModuleList:
        return self.text_model.layers

    @property
    def norm(self) -> nn.Module:
        return self.text_model.norm

    def embed_text_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed text tokens, squeezing singleton dimensions."""
        # Text events are shaped as (..., 1); squeeze the singleton index dim
        h = self.text_model.embed_tokens(token_ids)
        if h.dim() >= 2 and h.size(-2) == 1:
            h = h[..., 0, :]
        return h

    def embed_vision(self, vision_tokens: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Embed vision tokens using the vision encoder."""
        # vision tokens is (seq_patches, token_grids)
        return self.vision_embedding(vision_tokens)

    def embed_stream(self, tensor_stream: TensorStream) -> torch.Tensor:
        """
        Embed each modality stream independently, preserving the original TensorStream
        structure.
        """
        flat_stream = tensor_stream.flat_stream()
        per_modality_stream = group_streams(flat_stream, group_fn=lambda ev: ev.type, schedule=False)
        per_modality_compact_stream = {k: v.compact() for k, v in per_modality_stream.items()}

        # Collect per-event grids for vision tokens (H, W like dims sans time)
        token_grids = defaultdict(list)
        for stream in tensor_stream.streams:
            for event in stream:
                token_grids[event.type].append(event.dims(virtual=False))

        embedded_compact = {}
        for stream_type, modality_payload_tensor in per_modality_compact_stream.items():
            if stream_type.modality == VisionType:
                # Build a (N_events, 2) grid tensor with spatial dims only
                grids = token_grids.get(stream_type, [])
                if len(grids) == 0:
                    input_tensor = modality_payload_tensor
                else:
                    token_grids_tensor = torch.tensor(grids, dtype=torch.long, device=tensor_stream.device)[:, 1:]
                    input_tensor = (modality_payload_tensor, token_grids_tensor)
                embedded_compact[stream_type] = self.embed_fns[stream_type.modality](input_tensor)
            else:
                embedded_compact[stream_type] = self.embed_fns[stream_type.modality](modality_payload_tensor)

        # Reconstruct a TensorStream with embedded payloads and compact
        embedded_ts = reconstruct_tensor_stream_from_compact_dict(tensor_stream, embedded_compact)
        h = embedded_ts.compact()  # (B, T, D)
        return h

    @auto_docstring
    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        tensor_stream: Optional[TensorStream] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        """
        Forward pass with MRoPE position embeddings.

        Computes position embeddings once and passes them through all layers.

        Args:
            tensor_stream (`TensorStream`, *optional*):
                Packed multimodal stream of text and vision events to embed directly. Mutually exclusive with
                `input_ids` and `inputs_embeds`. When provided, the method derives `position_ids` and `modality_tensor`
                if they are not supplied.
            modality_tensor (`torch.LongTensor`, *optional*):
                Modality identifiers aligned with the embedded sequence, shaped `(batch_size, seq_len)` and containing
                values from `TextType`/`VisionType`. Automatically built from `tensor_stream` or `input_ids` when
                omitted.
        """

        modality_tensor = kwargs.pop("modality_tensor", None)
        if modality_tensor is not None:
            modality_tensor = modality_tensor.to(dtype=torch.long)
        text_value = TextType.text.value if TextType is not None else 0

        # Get inputs

        if tensor_stream is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both tensor_stream and inputs_embeds")
        elif tensor_stream is not None:
            # Embed TensorStream directly
            inputs_embeds = self.embed_stream(tensor_stream)
            # Create modality tensor if not provided
            if modality_tensor is None:
                modality_tensor = modality_mask(tensor_stream)
        elif input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            inputs_embeds = self.text_model.embed_tokens(input_ids)
            # Create text modality tensor if not provided
            if modality_tensor is None:
                batch_size, seq_length = input_ids.shape
                modality_tensor = torch.full(
                    (batch_size, seq_length), text_value, device=input_ids.device, dtype=torch.long
                )
        elif inputs_embeds is not None:
            # Inputs provided directly as embeddings (no input_ids/tensor_stream)
            if modality_tensor is None:
                batch_size, seq_length = inputs_embeds.shape[:2]
                modality_tensor = torch.full(
                    (batch_size, seq_length), text_value, device=inputs_embeds.device, dtype=torch.long
                )
            if attention_mask is None:
                attention_mask = torch.ones(
                    (inputs_embeds.shape[0], inputs_embeds.shape[1]), device=inputs_embeds.device, dtype=torch.long
                )
        else:
            raise ValueError("You have to specify either tensor_stream, input_ids or inputs_embeds")

        # Ensure cache exists when requested
        if use_cache and past_key_values is None:
            cache_config = self.config.get_text_config() if hasattr(self.config, "get_text_config") else self.config
            past_key_values = DynamicCache(config=cache_config)

        if cache_position is None and (past_key_values is not None or use_cache):
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # Create default position_ids if not provided
        if position_ids is None:
            if tensor_stream is not None:
                position_ids = compute_mrope_pos_tensor(tensor_stream)  # (B,L,3)
            elif input_ids is not None:
                position_ids = compute_position_ids_input_ids(input_ids)
            else:
                batch_size, seq_length = inputs_embeds.shape[:2]
                dummy_ids = torch.zeros((batch_size, seq_length), device=inputs_embeds.device, dtype=torch.long)
                position_ids = compute_position_ids_input_ids(dummy_ids)

        if attention_mask is None:
            attention_mask = torch.ones(
                (inputs_embeds.shape[0], inputs_embeds.shape[1]), device=inputs_embeds.device, dtype=torch.long
            )

        # Compute MRoPE position embeddings if we have custom rotary_emb
        cos, sin = self.rotary_emb(
            position_ids,
            modality_tensor,
            hidden_states=inputs_embeds,
        )
        cos = cos.to(inputs_embeds.dtype)
        sin = sin.to(inputs_embeds.dtype)

        # Flash attention expects 1D position_ids; keep 3D only for rotary phases
        decoder_position_ids = position_ids
        if position_ids is not None and position_ids.ndim == 3:
            decoder_position_ids = position_ids[..., 0]

        # Prepare attention mask

        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": decoder_position_ids,
            }
            attention_mask = create_masks_for_generate(**mask_kwargs)

        # Initialize hidden states
        hidden_states = inputs_embeds
        all_attentions = [] if output_attentions else None

        for decoder_layer in self.text_model.layers:
            layer_attention_mask = (
                attention_mask[decoder_layer.attention_type] if isinstance(attention_mask, dict) else attention_mask
            )
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_attention_mask,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=(cos, sin),
                output_attentions=output_attentions,
                **kwargs,
            )

            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_attentions.append(layer_outputs[1])
            else:
                hidden_states = layer_outputs

        # Final layer norm
        hidden_states = self.text_model.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=(hidden_states,),
            attentions=tuple(all_attentions) if output_attentions else None,
        )


class IsaacForConditionalGeneration(Qwen3ForCausalLM, GenerationMixin):
    """Isaac multimodal model for conditional generation."""

    config_class = IsaacConfig
    _can_compile_fullgraph = False
    _tied_weights_keys = {"lm_head.weight": "model.text_model.embed_tokens.weight"}
    all_tied_weights_keys: dict[str, str] = {"lm_head.weight": "model.text_model.embed_tokens.weight"}

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.set_input_embeddings(value)
        vocab_size = getattr(value, "num_embeddings", None)
        if vocab_size is not None:
            self.config.vocab_size = vocab_size
            self.model.config.vocab_size = vocab_size
            if hasattr(self.model, "text_model"):
                self.model.text_model.config.vocab_size = vocab_size
            if self.lm_head.weight.shape[0] != vocab_size:
                self.lm_head = nn.Linear(self.config.hidden_size, vocab_size, bias=False)
            if hasattr(self.model, "embed_tokens"):
                self.lm_head.weight = self.model.text_model.embed_tokens.weight

    def __init__(self, config: IsaacConfig):
        super().__init__(config)
        self.model = IsaacModel(config)  # Use our custom model
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Tracks rotary position offsets computed during a full forward pass so decode steps can reuse them.
        self.rope_deltas = None

    def get_rope_index(
        self,
        input_ids: Optional[torch.Tensor],
        tensor_stream: Optional[TensorStream],
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MRoPE position ids from a TensorStream (or 1D fallback).

        Returns (position_ids, rope_deltas). position_ids is (B,L,3) for MRoPE.
        rope_deltas is (B,1) used to advance positions in decode.
        """
        # tensor_stream present: compute 3D coords
        if tensor_stream is None and input_ids is None:
            raise ValueError("`tensor_stream` or `input_ids` must be provided to compute rope indices")

        if tensor_stream is not None:
            pos_3d = compute_mrope_pos_tensor(tensor_stream)  # (B,L,3)
        else:
            pos_3d = compute_position_ids_input_ids(input_ids)
        B, L, _ = pos_3d.shape

        # Max position per batch across the 3 planes and sequence dimension: (B,)
        m_per_batch = pos_3d.amax(dim=(1, 2))

        # Sequence lengths per batch: (B,)
        if attention_mask is None:
            seq_lens = torch.full_like(m_per_batch, L)
        else:
            seq_lens = attention_mask.eq(1).sum(dim=-1).to(dtype=m_per_batch.dtype, device=m_per_batch.device)

        rope_deltas = (m_per_batch + 1 - seq_lens).to(dtype=pos_3d.dtype).unsqueeze(1)
        return pos_3d, rope_deltas

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        tensor_stream: Optional[TensorStream] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        Forward pass for conditional generation supporting both standard inputs and TensorStream.

        tensor_stream (`TensorStream`, *optional*):
            Packed multimodal stream (text, vision, audio tokens) that already encodes spatial metadata. When provided,
            the model derives embeddings, modality masks, and 3D rotary coordinates directly from the stream instead of
            `input_ids`.
        """

        # Don't compute embeddings here - let the model handle it
        if tensor_stream is not None:
            input_ids = None
        if input_ids is None and inputs_embeds is None and tensor_stream is None:
            raise ValueError("Either input_ids, inputs_embeds, or tensor_stream must be provided.")

        # Build position ids (MRoPE) if needed and tensor_stream is available
        # During decode we reuse `self.rope_deltas` computed on the initial forward pass; `rope_delta` captures how far
        # cached rotary phases have progressed so we can advance `position_ids` without rebuilding the TensorStream.
        if position_ids is None and tensor_stream is not None:
            position_ids, self.rope_deltas = self.get_rope_index(input_ids, tensor_stream, attention_mask)
        elif position_ids is None and input_ids is not None:
            # For text inputs build position ids and modality tensor
            position_ids = compute_position_ids_input_ids(input_ids)
            if cache_position is not None and self.rope_deltas is not None:
                # Combine the incremental decode step (`cache_position`) with cached offsets so hidden states continue
                # rotating in lockstep across generation steps.
                rope_delta = (cache_position[0] + self.rope_deltas).to(input_ids.device)
            else:
                rope_delta = 0
            if cache_position is not None and not isinstance(rope_delta, int):  # otherwise `deltas` is an int `0`
                batch_size = input_ids.shape[0]
                rope_delta = rope_delta.repeat_interleave(batch_size // rope_delta.shape[0], dim=0)
            position_ids = position_ids.add(rope_delta)
        elif position_ids is None and inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
            dummy_ids = torch.zeros((batch_size, seq_len), device=inputs_embeds.device, dtype=torch.long)
            position_ids = compute_position_ids_input_ids(dummy_ids)

        if attention_mask is None:
            if input_ids is not None:
                batch_size, seq_len = input_ids.shape
                attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device, dtype=torch.long)
            else:
                batch_size, seq_len = inputs_embeds.shape[:2]
                attention_mask = torch.ones((batch_size, seq_len), device=inputs_embeds.device, dtype=torch.long)

        text_value = TextType.text.value if TextType is not None else 0

        if tensor_stream is not None:
            modality_tensor = modality_mask(tensor_stream)
        elif input_ids is not None:
            batch_size, seq_len = input_ids.shape
            modality_tensor = torch.full(
                (batch_size, seq_len), text_value, device=position_ids.device, dtype=torch.long
            )
        else:
            batch_size, seq_len = inputs_embeds.shape[:2]
            modality_tensor = torch.full(
                (batch_size, seq_len), text_value, device=position_ids.device, dtype=torch.long
            )

        outputs = self.model(
            input_ids=input_ids,
            tensor_stream=tensor_stream,
            attention_mask=attention_mask,
            position_ids=position_ids,
            modality_tensor=modality_tensor,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
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

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        tensor_stream: Optional[TensorStream] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Prepare inputs for generation, handling TensorStream inputs properly.
        """
        if cache_position is None:
            seq_length = None
            device = None
            if input_ids is not None:
                seq_length = input_ids.shape[1]
                device = input_ids.device
            elif inputs_embeds is not None:
                seq_length = inputs_embeds.shape[1]
                device = inputs_embeds.device
            elif tensor_stream is not None:
                _, seq_length = tensor_stream.shape
                device = tensor_stream.device
            if seq_length is not None:
                # prepare_inputs_for_generation may be invoked outside `generate`, so synthesize the
                # same cache positions that GenerationMixin would have created during prefill.
                cache_position = torch.arange(seq_length, dtype=torch.long, device=device)

        # Call parent preparation
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )

        cache_position = model_inputs.get("cache_position", cache_position)

        # Handle TensorStream for first forward pass only
        if tensor_stream is not None and (cache_position is None or cache_position[0] == 0):
            model_inputs["tensor_stream"] = tensor_stream
        # Let forward rebuild position_ids using cached deltas during decode
        model_inputs["position_ids"] = None
        # Drop tensor_stream after step 0
        if cache_position is not None and cache_position[0] != 0:
            model_inputs["tensor_stream"] = None
        return model_inputs

    @classmethod
    def can_generate(cls) -> bool:
        return True


def _compute_residual_p_frames(frames: torch.Tensor, is_p_frame: list[bool]) -> torch.Tensor:
    """Compute residuals for P-frames to stay in sync with the training pipeline."""
    if not any(is_p_frame):
        return frames

    frame_indices = torch.arange(len(is_p_frame), device=frames.device)
    i_frame_mask = torch.tensor([not flag for flag in is_p_frame], device=frames.device)
    last_i_indices = torch.cummax((i_frame_mask * (1 + frame_indices)), dim=0).values.long() - 1
    p_indices = frame_indices[torch.tensor(is_p_frame, device=frames.device)]
    frames[p_indices] = frames[p_indices] - frames[last_i_indices[p_indices]]
    return frames


__all__ = [
    "IsaacConfig",
    "IsaacModel",
    "IsaacPreTrainedModel",  # noqa: F822
    "IsaacForConditionalGeneration",
    "IsaacImageProcessorFast",
    "IsaacProcessor",
]
