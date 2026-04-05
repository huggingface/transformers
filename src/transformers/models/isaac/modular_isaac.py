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

import itertools
import math
import re
from collections import defaultdict
from typing import Any, NamedTuple

from huggingface_hub.dataclasses import strict

from ... import TorchvisionBackend
from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...generation.utils import GenerationMixin
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import ImageInput, PILImageResampling, SizeDict, make_nested_list_of_images
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...models.qwen3.configuration_qwen3 import Qwen3Config
from ...processing_utils import ImagesKwargs, MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...utils import TensorType, auto_docstring, torch_compilable_check
from ...utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from ...utils.generic import TransformersKwargs, can_return_tuple, merge_with_config_defaults
from ...utils.import_utils import (
    is_torch_available,
    is_torchdynamo_compiling,
    is_torchvision_available,
)
from ...utils.output_capturing import capture_outputs
from ..qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
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
    from torchvision.transforms.v2 import functional as tvF

if is_torchvision_available():
    from ..pix2struct.image_processing_pix2struct import torch_extract_patches


@auto_docstring(checkpoint="PerceptronAI/Isaac-0.1-Base")
@strict
class IsaacVisionConfig(Siglip2VisionConfig):
    r"""
    num_patches (`int`, *optional*, defaults to 256):
        The number of patches in the image with the size of (`patch_size`, `patch_size`). The image is resized to
        fill a maximum of this number of patches while preserving the aspect ratio. If the resulting number of patches
        is lower, the image is padded in the patch dimension.
    pixel_shuffle_scale_factor (`int`, *optional*, defaults to 1):
        Spatial factor applied before pixel shuffle reduces the resolution.
    """

    model_type = "isaac_vision"
    base_config_key = "vision_config"
    pixel_shuffle_scale_factor: int = 1


@auto_docstring(checkpoint="PerceptronAI/Isaac-0.1-Base")
@strict
class IsaacTextConfig(Qwen3Config):
    r"""
    Example:

    ```python
    >>> from transformers import IsaacTextConfig, IsaacTextModel

    >>> configuration = IsaacTextConfig()
    >>> model = IsaacTextModel(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "isaac_text"
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}
    max_position_embeddings: int = 32768
    sliding_window = AttributeError()
    layer_types = AttributeError()

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        PretrainedConfig.__post_init__(self, **kwargs)


@auto_docstring(checkpoint="PerceptronAI/Isaac-0.1-Base")
@strict
class IsaacConfig(PretrainedConfig):
    r"""
    vision_config (`IsaacVisionConfig` or `dict`, *optional*):
        Configuration for the Isaac vision tower. Dictionaries are converted to [`IsaacVisionConfig`]. If unset,
        the default [`IsaacVisionConfig`] is used.
    text_config (`IsaacTextConfig` or `dict`, *optional*):
        Configuration for the text backbone. Dictionaries are converted to [`IsaacTextConfig`].
    vision_rescale_factor (`float`, *optional*, defaults to 1 / 255):
        Rescale factor applied by the image processor before normalization.
    max_sequence_length (`int`, *optional*, defaults to 16384):
        Maximum multimodal sequence length produced by the processor and expected by the model.
    Example:

    ```python
    >>> from transformers import IsaacConfig, IsaacModel

    >>> configuration = IsaacConfig()
    >>> model = IsaacModel(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "isaac"
    sub_configs = {"vision_config": IsaacVisionConfig, "text_config": IsaacTextConfig}
    vision_config: IsaacVisionConfig | dict | None = None
    text_config: IsaacTextConfig | dict | None = None
    vision_rescale_factor: float = 1 / 255
    max_sequence_length: int = 16384

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        elif not isinstance(self.text_config, IsaacTextConfig):
            raise TypeError(
                f"text_config must be a dict or an IsaacTextConfig instance, got {type(self.text_config).__name__}."
            )

        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        elif not isinstance(self.vision_config, IsaacVisionConfig):
            raise TypeError(
                f"vision_config must be a dict or an IsaacVisionConfig instance, got {type(self.vision_config).__name__}."
            )

        self.vision_rescale_factor = float(self.vision_rescale_factor)
        super().__post_init__(**kwargs)


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
        image_grid_thw: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # pixel_values: (num_images, max_patches, patch_dim)
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        resized_positional_embeddings = self.resize_positional_embeddings(
            self.position_embedding,
            image_grid_thw[:, 1:],
            max_length=pixel_values.shape[1],
        )
        resized_positional_embeddings = resized_positional_embeddings.to(
            device=patch_embeds.device, dtype=patch_embeds.dtype
        )
        embeddings = patch_embeds + resized_positional_embeddings

        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1).to(device=embeddings.device, dtype=embeddings.dtype)

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


@auto_docstring
class IsaacVisionModel(PreTrainedModel):
    config: IsaacVisionConfig
    _supports_sdpa = True
    _supports_flash_attn = True
    _can_record_outputs = {
        "hidden_states": IsaacVisionEncoderLayer,
        "attentions": IsaacVisionAttention,
    }

    def __init__(self, config: IsaacVisionConfig):
        super().__init__(config)
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

    def pixel_shuffle_padded(
        self,
        hidden_states: torch.Tensor,
        token_grids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply pixel shuffle per image on padded batched vision embeddings.

        Args:
            hidden_states (`torch.Tensor`):
                Vision embeddings of shape `(num_images, max_patches, hidden_size)`.
            token_grids (`torch.Tensor`):
                Grid sizes `(height, width)` per image, shape `(num_images, 2)`.

        Returns:
            `torch.Tensor`: Pixel-shuffled embeddings of shape
            `(num_images, max_tokens, hidden_size * scale_factor**2)`.
        """
        scale_factor = self.pixel_shuffle_scale_factor
        num_images, max_patches, embed_dim = hidden_states.shape
        output_dim = embed_dim * scale_factor * scale_factor

        token_grids = token_grids.to(device=hidden_states.device, dtype=torch.long)
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
        shuffled_4d = hidden_states.new_zeros((num_images, max_output_tokens, scale_factor * scale_factor, embed_dim))

        token_positions = (
            torch.arange(max_patches, device=hidden_states.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(num_images, -1)
        )
        valid_token_mask = token_positions < full_lengths.unsqueeze(1)

        safe_widths = torch.where(widths > 0, widths, torch.ones_like(widths))
        row_index = torch.div(token_positions, safe_widths.unsqueeze(1), rounding_mode="floor")
        col_index = token_positions.remainder(safe_widths.unsqueeze(1))

        output_widths = widths.div(scale_factor, rounding_mode="floor")
        output_index = row_index.div(scale_factor, rounding_mode="floor") * output_widths.unsqueeze(1)
        output_index = output_index + col_index.div(scale_factor, rounding_mode="floor")
        sub_index = row_index.remainder(scale_factor) * scale_factor + col_index.remainder(scale_factor)

        batch_index = (
            torch.arange(num_images, device=hidden_states.device, dtype=torch.long)
            .unsqueeze(1)
            .expand_as(token_positions)
        )
        shuffled_4d[batch_index[valid_token_mask], output_index[valid_token_mask], sub_index[valid_token_mask]] = (
            hidden_states[valid_token_mask]
        )

        shuffled = shuffled_4d.view(num_images, max_output_tokens, output_dim)
        return shuffled

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        image_grid_thw (`torch.LongTensor`, *optional*):
            Batch-major per-slot grids shaped `(batch_size, max_images, 3)` with `(T=1, H, W)` entries.
        """
        full_lengths = image_grid_thw[:, 1] * image_grid_thw[:, 2]
        token_positions = torch.arange(pixel_values.shape[1], device=pixel_values.device, dtype=torch.long)
        image_patch_attention_mask = token_positions.unsqueeze(0) < full_lengths.unsqueeze(1)
        image_patch_attention_mask = image_patch_attention_mask.to(dtype=torch.long)
        hidden_states = self.embeddings(
            pixel_values,
            image_grid_thw,
            attention_mask=image_patch_attention_mask,
        )

        encoder_attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=image_patch_attention_mask,
        )
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, attention_mask=encoder_attention_mask, **kwargs)
        hidden_states = self.post_layernorm(encoder_outputs.last_hidden_state)

        hidden_states = self.pixel_shuffle_padded(
            hidden_states=hidden_states,
            token_grids=image_grid_thw[:, 1:],
        )

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class IsaacRotaryEmbedding(Qwen3VLTextRotaryEmbedding):
    def __init__(self, config: IsaacTextConfig, device=None):
        super().__init__(config, device=device)
        self.mrope_section = config.rope_parameters.get("mrope_section")
        if self.mrope_section is None:
            weights = (2, 1, 1)
            self.mrope_section = [self.inv_freq.shape[0] * w // sum(weights) for w in weights]
            self.mrope_section[0] += self.inv_freq.shape[0] - sum(self.mrope_section)

    def apply_interleaved_mrope(self, freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
        chunks = freqs.split(tuple(mrope_section), dim=-1)
        return torch.cat([chunk[i % 3] for i, chunk in enumerate(chunks)], dim=-1)


class IsaacTextDecoderLayer(Qwen3VLTextDecoderLayer):
    pass


class IsaacTextModel(Qwen3VLTextModel):
    def __init__(self, config: IsaacTextConfig):
        super().__init__(config)
        self.rotary_emb = IsaacRotaryEmbedding(config=config, device=self.device)


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


@auto_docstring
class IsaacModel(Qwen3VLModel):
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["IsaacTextDecoderLayer", "IsaacVisionEncoderLayer"]
    _can_compile_fullgraph = False
    _supports_flex_attn = False
    _tied_weights_keys = {}
    _input_embed_layer = "language_model.embed_tokens"

    def __init__(self, config: IsaacConfig):
        PreTrainedModel.__init__(self, config)
        self.language_model = IsaacTextModel._from_config(config.text_config)
        self.visual = IsaacVisionModel(config.vision_config)
        self.multimodal_projector = IsaacMultiModalProjector(config)
        self.max_sequence_length = config.max_sequence_length
        self.vision_rescale_factor = config.vision_rescale_factor
        self.rope_deltas = None

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_metadata: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor`, *optional*):
            Batch-major patch vectors shaped `(batch_size, max_images, max_patches, patch_dim)`.
        image_grid_thw (`torch.LongTensor`, *optional*):
            Batch-major per-slot grids shaped `(batch_size, max_images, 3)` with `(T=1, H, W)` entries.
        image_metadata (`torch.Tensor`, *optional*):
            Batch-major per-slot metadata `(offset, length)` shaped `(batch_size, max_images, 2)`.
        """
        active_slot_mask = image_grid_thw[..., 0].eq(1)
        flat_pixel_values = pixel_values[active_slot_mask]
        flat_image_grid_thw = image_grid_thw[active_slot_mask]

        vision_outputs: BaseModelOutputWithPooling = self.visual(
            pixel_values=flat_pixel_values,
            image_grid_thw=flat_image_grid_thw,
            return_dict=True,
            **kwargs,
        )
        projected_features = self.multimodal_projector(vision_outputs.last_hidden_state)

        # Truncate image features using offset and length
        if image_metadata is None:
            pixel_shuffle_scale = self.config.vision_config.pixel_shuffle_scale_factor
            downsampled_height = flat_image_grid_thw[:, 1].div(pixel_shuffle_scale, rounding_mode="floor")
            downsampled_width = flat_image_grid_thw[:, 2].div(pixel_shuffle_scale, rounding_mode="floor")
            lengths = downsampled_height * downsampled_width
            offsets = torch.zeros_like(lengths)
        else:
            torch_compilable_check(
                image_metadata.shape[:2] == image_grid_thw.shape[:2],
                "IsaacModel.get_image_features expects batch-major metadata aligned with `image_grid_thw`.",
            )
            offsets = image_metadata[active_slot_mask][:, 0]
            lengths = image_metadata[active_slot_mask][:, 1]

        image_features = tuple(
            projected_features[image_idx, offset : offset + length]
            for image_idx, (offset, length) in enumerate(zip(offsets.tolist(), lengths.tolist(), strict=True))
        )

        return BaseModelOutputWithPooling(
            last_hidden_state=projected_features,
            pooler_output=image_features,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
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
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None:
            torch_compilable_check(
                inputs_embeds[special_image_mask].numel() == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
            )

        return special_image_mask

    def get_video_features(self, **super_kwargs):
        raise AttributeError("Isaac is image-only and does not support `pixel_values_videos` or `video_grid_thw`.")

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_metadata: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pixel_shuffle_scale = self.config.vision_config.pixel_shuffle_scale_factor

        mrope_position_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        # shape: [batch_size, max_num_images]
        active_slot_mask = image_grid_thw[..., 0].eq(1)

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]

            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            image_idx = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                # text == 0
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                # image == 1
                else:
                    while image_idx < image_metadata[batch_idx].shape[0] and (
                        not active_slot_mask[batch_idx][image_idx] or image_metadata[batch_idx][image_idx, 1] == 0
                    ):
                        image_idx += 1
                    grid_thw = image_grid_thw[batch_idx][image_idx]
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos, grid_thw, 1, pixel_shuffle_scale, device=input_ids.device
                    )
                    token_offset = image_metadata[batch_idx][image_idx][0].item()
                    token_length = image_metadata[batch_idx][image_idx][1].item()
                    vision_position_ids = vision_position_ids[:, token_offset : token_offset + token_length]
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += max(grid_thw[1], grid_thw[2]) // pixel_shuffle_scale
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        image_metadata: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        has_multimodal = image_grid_thw is not None and image_metadata is not None

        if has_multimodal and mm_token_type_ids is None and input_ids is not None:
            raise ValueError(
                "Multimodal data was passed (via `image_grid_thw` or `image_metadata`) but `mm_token_type_ids` is "
                "missing. Please pass `mm_token_type_ids` to the model so that multimodal RoPE (M-RoPE) can be "
                "computed correctly. `mm_token_type_ids` is returned by the processor alongside `input_ids`."
            )
        can_compute_mrope = input_ids is not None and mm_token_type_ids is not None and has_multimodal

        if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                image_metadata=image_metadata,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.rope_deltas = rope_deltas
        # Use pre-calculated rope-deltas to infer correct 3D position ids during incremental
        # generation (past_key_values_length > 0) or when only inputs_embeds is provided (no input_ids
        # to recompute from). Skip when input_ids is provided without past_key_values to avoid shape
        # mismatches from stale rope_deltas (e.g., training forward pass after generation).
        elif self.rope_deltas is not None and (past_key_values_length > 0 or input_ids is None):
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1).to(inputs_embeds.device)
            else:
                position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_length)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).to(inputs_embeds.device)
            delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
            position_ids = position_ids + delta.to(device=inputs_embeds.device)
        else:
            # Can't build correct 3D positions. Let the model infer it
            position_ids = None
        return position_ids

    @auto_docstring(
        custom_intro="""
        Forward pass with multimodal MRoPE position ids.

        When image placeholders are present, Isaac computes vision features, scatters them into the token
        embeddings, and runs the shared text backbone on the mixed sequence.
        """,
    )
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        image_metadata: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        mm_token_type_ids (`torch.LongTensor`, *optional*):
            Multimodal token type ids aligned with the token sequence, using `0 -> text` and `1 -> image`.
        pixel_values (`torch.FloatTensor`, *optional*):
            Batch-major patch vectors shaped `(batch_size, max_images, max_patches, patch_dim)`.
        image_grid_thw (`torch.LongTensor`, *optional*):
            Batch-major per-slot grids shaped `(batch_size, max_images, 3)` with `(T=1, H, W)` entries.
        image_metadata (`torch.LongTensor`, *optional*):
            Batch-major per-slot metadata shaped `(batch_size, max_images, 2)` with `(offset, length)`.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        if pixel_values is not None and image_grid_thw is not None:
            image_outputs = self.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                image_metadata=image_metadata,
                return_dict=True,
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            image_metadata=image_metadata,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=image_mask[..., 0] if image_mask is not None else None,
            deepstack_visual_embeds=None,
            use_cache=use_cache,
            **kwargs,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class IsaacForConditionalGeneration(Qwen3VLForConditionalGeneration, GenerationMixin):
    config_class = IsaacConfig
    input_modalities = ("image", "text")
    _no_split_modules = ["IsaacTextDecoderLayer", "IsaacVisionEncoderLayer"]
    _can_compile_fullgraph = False
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: IsaacConfig):
        PreTrainedModel.__init__(self, config)
        self.model = IsaacModel(config)
        self.vocab_size = config.get_text_config().vocab_size
        self.lm_head = nn.Linear(config.get_text_config().hidden_size, config.get_text_config().vocab_size, bias=False)
        self.post_init()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values: Cache = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        image_metadata: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        is_first_iteration: bool = False,
        use_cache: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            is_first_iteration=is_first_iteration,
            use_cache=use_cache,
            **kwargs,
        )

        multimodal_inputs = {
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "image_metadata": image_metadata,
        }
        is_prefill = is_first_iteration or not use_cache
        for key, value in multimodal_inputs.items():
            model_inputs[key] = value if is_prefill else None

        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        text_positions = GenerationMixin._prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs)

        past_length = 0
        if (cache := model_kwargs.get("past_key_values")) is not None:
            past_length = cache.get_seq_length()
        if past_length != 0 and self.model.rope_deltas is not None:
            return text_positions[None, ...] + self.model.rope_deltas

        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
        if (
            is_input_ids
            and model_kwargs.get("mm_token_type_ids") is not None
            and model_kwargs.get("image_grid_thw") is not None
            and model_kwargs.get("image_metadata") is not None
        ):
            model_kwargs = {k: v for k, v in model_kwargs.items() if k != "input_ids"}
            vision_positions, rope_deltas = self.model.get_rope_index(inputs_tensor, **model_kwargs)
            self.model.rope_deltas = rope_deltas
        else:
            vision_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
            self.model.rope_deltas = torch.zeros(
                inputs_tensor.shape[0], 1, dtype=torch.long, device=inputs_tensor.device
            )

        return torch.cat([text_positions[None, ...], vision_positions], dim=0)

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        position_ids = model_kwargs.pop("position_ids", None)
        if expand_size == 1:
            if position_ids is not None:
                model_kwargs["position_ids"] = position_ids
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "image_metadata"]
        for key in visual_keys:
            value = model_kwargs.get(key)
            if value is not None:
                model_kwargs[key] = value.repeat_interleave(expand_size, dim=0)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        for key, value in list(model_kwargs.items()):
            if key == "position_ids" and value is not None and value.ndim == 3:
                model_kwargs[key] = value.repeat_interleave(expand_size, dim=1)
            elif value is not None and isinstance(value, torch.Tensor) and key not in visual_keys:
                model_kwargs[key] = value.repeat_interleave(expand_size, dim=0)

        if position_ids is not None:
            dim = 1 if position_ids.ndim == 3 else 0
            model_kwargs["position_ids"] = position_ids.repeat_interleave(expand_size, dim=dim)
        return input_ids, model_kwargs


# --------------------------------Isaac Image Processor--------------------------------


class IsaacImageProcessorKwargs(ImagesKwargs, total=False):
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


@auto_docstring
class IsaacImageProcessor(TorchvisionBackend):
    model_input_names = ["pixel_values", "image_grid_thw"]
    valid_kwargs = IsaacImageProcessorKwargs

    resample = PILImageResampling.BILINEAR
    do_resize = True
    do_center_crop = False
    patch_size = 16
    max_num_patches = 256
    min_num_patches = None
    pixel_shuffle_scale = 1
    do_pad = True
    do_rescale = True
    do_normalize = True
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_convert_rgb = True
    disable_grouping = False

    def __init__(self, **kwargs: Unpack[IsaacImageProcessorKwargs]):
        super().__init__(**kwargs)

    def _validate_preprocess_kwargs(self, **kwargs):
        # Allow callers to omit resize-related placeholders that BaseImageProcessorFast checks for.
        kwargs.pop("do_resize", None)
        return super()._validate_preprocess_kwargs(**kwargs)

    def _prepare_images_structure(
        self,
        images: ImageInput,
        expected_ndims: int = 3,
    ) -> ImageInput:
        images = self.fetch_images(images)
        return make_nested_list_of_images(images, expected_ndims=expected_ndims)

    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        **kwargs,
    ) -> torch.Tensor:
        if image.dtype == torch.uint8:
            image = F.interpolate(image.float(), size=(size.height, size.width), mode="bilinear", align_corners=False)
            return image.clamp(0, 255).round().to(torch.uint8)
        return F.interpolate(image, size=(size.height, size.width), mode="bilinear", align_corners=False)

    def pack_images(
        self,
        vision_patches: list[list[torch.Tensor]],
        vision_token_grids: list[list[torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        batch_size = len(vision_patches)
        flat_patches = [patches for sample_patches in vision_patches for patches in sample_patches]

        first_patch = flat_patches[0]
        max_patches = max(patches.shape[0] for patches in flat_patches)
        max_images = max((len(sample_patches) for sample_patches in vision_patches), default=0)

        patch_dim = first_patch.shape[-1]
        tensors = {
            "pixel_values": torch.zeros(
                (batch_size, max_images, max_patches, patch_dim),
                device=first_patch.device,
                dtype=first_patch.dtype,
            ),
            "image_grid_thw": torch.zeros((batch_size, max_images, 3), device=first_patch.device, dtype=torch.long),
        }

        for batch_idx, (sample_patches, sample_token_grids) in enumerate(
            zip(vision_patches, vision_token_grids, strict=True)
        ):
            for image_slot, (patches, token_grid) in enumerate(zip(sample_patches, sample_token_grids, strict=True)):
                patch_count = int(patches.shape[0])
                tensors["pixel_values"][batch_idx, image_slot, :patch_count] = patches
                tensors["image_grid_thw"][batch_idx, image_slot, 0] = 1
                tensors["image_grid_thw"][batch_idx, image_slot, 1:] = token_grid

        return tensors

    def _preprocess(
        self,
        images: list[list[torch.Tensor]],
        do_resize: bool,
        resample: PILImageResampling | tvF.InterpolationMode | int | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool,
        patch_size: int,
        max_num_patches: int,
        min_num_patches: int,
        pixel_shuffle_scale: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping, is_nested=True
        )
        grouped_outputs = {}
        for shape, stacked_images in grouped_images.items():
            grouped_batch_size, channels, original_height, original_width = stacked_images.shape
            if do_resize:
                target_height, target_width = get_image_size_for_max_num_patches(
                    original_height,
                    original_width,
                    patch_size,
                    max_num_patches,
                    min_num_patches=min_num_patches,
                    pixel_shuffle_scale=pixel_shuffle_scale,
                )
                image_batch = self.resize(
                    stacked_images, SizeDict(height=target_height, width=target_width), resample=resample
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
            _, height_tokens, width_tokens, patch_dim = patches.shape

            token_grid = (
                torch.tensor([height_tokens, width_tokens], device=patches.device).long().expand(grouped_batch_size, 2)
            )

            if (height_tokens % pixel_shuffle_scale) or (width_tokens % pixel_shuffle_scale):
                raise ValueError(
                    f"Token grid (h={height_tokens}, w={width_tokens}) must be divisible by pixel_shuffle_scale={pixel_shuffle_scale};"
                    f" adjust resize/patch parameters or disable pixel shuffle."
                )

            grouped_outputs[shape] = (
                patches.reshape(grouped_batch_size, -1, patch_dim),
                token_grid,
            )

        keys = ("vision_patches", "vision_token_grids")
        nested_outputs = {}
        for i, key in enumerate(keys):
            nested_outputs[key] = reorder_images(
                {shape: values[i] for shape, values in grouped_outputs.items()},
                dict(grouped_images_index),
                is_nested=True,
            )

        if not do_pad:
            raise ValueError("IsaacImageProcessor doesn't support `do_pad=False` mode.")

        tensors = self.pack_images(
            vision_patches=nested_outputs["vision_patches"],
            vision_token_grids=nested_outputs["vision_token_grids"],
        )

        return BatchFeature(data=tensors, tensor_type=return_tensors)

    def get_number_of_image_patches(
        self,
        image_height: int,
        image_width: int,
        images_kwargs: dict[str, Any] | None = None,
    ) -> int:
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        max_num_patches = images_kwargs.get("max_num_patches", self.max_num_patches)
        min_num_patches = images_kwargs.get("min_num_patches", self.min_num_patches)
        pixel_shuffle_scale = images_kwargs.get("pixel_shuffle_scale", self.pixel_shuffle_scale)

        target_height, target_width = get_image_size_for_max_num_patches(
            image_height,
            image_width,
            patch_size,
            max_num_patches,
            min_num_patches=min_num_patches,
            pixel_shuffle_scale=pixel_shuffle_scale,
        )
        return (target_height // patch_size) * (target_width // patch_size)


# --------------------------------Isaac Processor--------------------------------


class IsaacProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs = IsaacImageProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "truncation": True,
            "truncation_side": "left",
            "return_attention_mask": True,
            "return_overflowing_tokens": True,
            "return_mm_token_type_ids": True,
            "add_special_tokens": False,
        },
    }


class SinglePoint(NamedTuple):
    x: int
    y: int
    mention: str | None = None
    t: float | None = None


class BoundingBox(NamedTuple):
    top_left: Any
    bottom_right: Any
    mention: str | None = None
    t: float | None = None


class Polygon(NamedTuple):
    points: tuple[Any, ...]
    mention: str | None = None
    t: float | None = None


_point_box_or_polygon_tag = re.compile(
    r"<(?P<tag>point|point_box|polygon)(?P<attrs>[^>]*)>(?P<body>[\s\S]*?)</(?P=tag)>", re.IGNORECASE
)
_attr_re = re.compile(r"(\w+)\s*=\s*(?:\"([^\"]*)\"|([^\s>]+))")
_coord_re = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


@auto_docstring
class IsaacProcessor(ProcessorMixin):
    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template: str | dict[str, str] | None = None,
        max_sequence_length: int = 16384,
    ):
        r"""
        max_sequence_length (`int`, *optional*, defaults to 16384):
            Maximum packed multimodal sequence length produced by the processor.
        """
        if chat_template is None:
            chat_template = getattr(tokenizer, "chat_template", None)

        self.pad_token_id = tokenizer.pad_token_id
        self.image_token = getattr(tokenizer, "image_pad_token", None) or getattr(tokenizer, "image_token", None)
        self.image_token_id = getattr(tokenizer, "image_pad_token_id", None) or getattr(
            tokenizer, "image_token_id", None
        )
        self.max_sequence_length = max_sequence_length
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: str | list[str],
        images: ImageInput,
        **kwargs,
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            IsaacProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # 1. Validate number of that text and images match
        texts = [text] if isinstance(text, str) else text.copy()
        fetched_images = self.image_processor.fetch_images(images)
        batched_images = make_nested_list_of_images(fetched_images)
        if len(batched_images) != len(texts):
            num_images_in_text = [text_value.count(self.image_token) for text_value in texts]
            num_images_in_images = [len(sample_images) for sample_images in batched_images]
            add_message = ""
            if sum(num_images_in_text) == sum(num_images_in_images):
                add_message = " Make sure to pass your images as a nested list, where each sub-list holds images for one text sample."
            raise ValueError(
                f"Received inconsistently sized batches of images ({len(batched_images)}) and text ({len(texts)}).{add_message}"
            )

        # 2. Process images
        image_inputs = self.image_processor(images=batched_images, **output_kwargs["images_kwargs"])
        image_grid_thw = image_inputs["image_grid_thw"]

        # 3. Expand text with image placeholders
        merge_length = self.image_processor.pixel_shuffle_scale**2
        vision_segment_lengths = image_grid_thw.prod(dim=-1) // merge_length
        for batch_idx in range(len(text)):
            image_idx = 0
            while self.image_token in text[batch_idx]:
                num_image_tokens = vision_segment_lengths[batch_idx, image_idx]
                text[batch_idx] = text[batch_idx].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                image_idx += 1
            text[batch_idx] = text[batch_idx].replace("<|placeholder|>", self.image_token)

        # 4. Process text
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids")
        max_length = output_kwargs["text_kwargs"].pop("max_length", None)
        max_length = self.max_sequence_length if max_length is None else max_length
        text_inputs = self.tokenizer(text, max_length=max_length, **output_kwargs["text_kwargs"])

        truncated_input_ids: list[list[int] | None] = [None] * len(texts)
        truncated_attention_mask: list[list[int] | None] = [None] * len(texts)
        overflow_input_ids_per_sample = defaultdict(int)

        # 5. Drop overflowing token ids
        for batch_idx, input_ids, attention_mask in zip(
            text_inputs["overflow_to_sample_mapping"], text_inputs["input_ids"], text_inputs["attention_mask"]
        ):
            if truncated_input_ids[batch_idx] is None:
                truncated_input_ids[batch_idx] = input_ids
                truncated_attention_mask[batch_idx] = attention_mask
            else:
                overflow_input_ids_per_sample[batch_idx] += input_ids.count(self.image_token_id)

        # 6. Do the same for overflowing pixel values. Isaac truncates images based on `max_length`
        # We can't really truncate pixels, so we pass over an image offset mask. Model will crop off
        # truncated image pixels at run-time using this mask
        batch_size, max_images = image_grid_thw.shape[:2]
        image_metadata = torch.zeros((batch_size, max_images, 2), dtype=torch.long)
        for batch_idx, image_lengths in enumerate(vision_segment_lengths):
            remaining_dropped = overflow_input_ids_per_sample[batch_idx]
            for image_idx, length in enumerate(image_lengths):
                offset = 0
                if 0 < remaining_dropped < length:
                    offset = remaining_dropped
                    length -= offset
                    remaining_dropped = 0
                elif remaining_dropped >= length:
                    length = 0
                    remaining_dropped -= length

                # Record which suffix of this image's placeholder span survives left truncation.
                # The model still encodes the full image and uses this window for both feature gathering and vision RoPE.
                image_metadata[batch_idx, image_idx, 0] = offset
                image_metadata[batch_idx, image_idx, 1] = length

        data = {
            "input_ids": torch.tensor(truncated_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(truncated_attention_mask, dtype=torch.long),
            "image_metadata": image_metadata,
            **image_inputs,
        }

        if return_mm_token_type_ids:
            data["mm_token_type_ids"] = self.create_mm_token_type_ids(data["input_ids"])

        return BatchFeature(data=data, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = dict(IsaacProcessorKwargs._defaults.get("images_kwargs", {}))
            images_kwargs.update(kwargs)

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            pixel_shuffle_scale = images_kwargs.get("pixel_shuffle_scale") or self.image_processor.pixel_shuffle_scale
            num_image_tokens = [num_patches // pixel_shuffle_scale**2 for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    @property
    def model_input_names(self):
        return super().model_input_names + ["mm_token_type_ids", "image_metadata"]

    @staticmethod
    def _maybe_float(value: str | None) -> float | None:
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @classmethod
    def _parse_attrs(cls, attr_text: str) -> dict[str, str]:
        attrs = {}
        for match in _attr_re.finditer(attr_text):
            key = match.group(1)
            value = match.group(2) or match.group(3) or ""
            attrs[key] = value
        return attrs

    @classmethod
    def _parse_point_body(
        cls,
        body: str,
        mention: str | None = None,
        t: str | None = None,
    ) -> Any:
        match = _coord_re.search(body)
        if not match:
            raise ValueError(f"Malformed <point> tag: {body!r}")
        x, y = int(match.group(1)), int(match.group(2))
        return SinglePoint(x=x, y=y, mention=mention, t=cls._maybe_float(t))

    @classmethod
    def _parse_box_body(
        cls,
        body: str,
        mention: str | None = None,
        t: str | None = None,
    ) -> Any:
        coords = list(_coord_re.finditer(body))
        if len(coords) < 2:
            raise ValueError(f"Malformed <point_box> tag: {body!r}")

        top_left = SinglePoint(x=int(coords[0].group(1)), y=int(coords[0].group(2)))
        bottom_right = SinglePoint(x=int(coords[1].group(1)), y=int(coords[1].group(2)))
        return BoundingBox(top_left=top_left, bottom_right=bottom_right, mention=mention, t=cls._maybe_float(t))

    @classmethod
    def _parse_polygon_body(
        cls,
        body: str,
        mention: str | None = None,
        t: str | None = None,
    ) -> Any:
        coords = list(_coord_re.finditer(body))
        if len(coords) < 3:
            raise ValueError(f"Malformed <polygon> tag: {body!r}")

        points = tuple(SinglePoint(x=int(coord.group(1)), y=int(coord.group(2))) for coord in coords)
        return Polygon(points=points, mention=mention, t=cls._maybe_float(t))

    @classmethod
    def clean_text_and_extract_points(
        cls,
        text: str,
        expected: str | None = None,
    ) -> tuple[str, list[Any]]:
        results: list[Any] = []
        for match in _point_box_or_polygon_tag.finditer(text):
            tag = match.group("tag").lower()
            attrs = cls._parse_attrs(match.group("attrs"))
            mention = attrs.get("mention")
            t = attrs.get("t")
            if tag == "point":
                if expected not in (None, "point"):
                    continue
                results.append(cls._parse_point_body(match.group("body"), mention=mention, t=t))
            elif tag == "point_box":
                if expected not in (None, "box"):
                    continue
                results.append(cls._parse_box_body(match.group("body"), mention=mention, t=t))
            else:
                if expected not in (None, "polygon"):
                    continue
                results.append(cls._parse_polygon_body(match.group("body"), mention=mention, t=t))

        clean_text = re.sub(r"\s+", " ", _point_box_or_polygon_tag.sub("", text or "")).strip()
        return clean_text, results

    def post_process_generation(
        self,
        text: str,
        expected: str | None = None,
        cleanup_and_extract: bool = True,
    ) -> str | tuple[str, list[Any]]:
        if cleanup_and_extract:
            return self.clean_text_and_extract_points(text, expected=expected)
        return text

    def post_process_image_text_to_text(
        self,
        generated_outputs,
        skip_special_tokens: bool = True,
        cleanup_and_extract: bool = False,
        expected: str | None = None,
        **kwargs,
    ):
        generated_texts = self.batch_decode(generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs)
        return [
            self.post_process_generation(text, expected=expected, cleanup_and_extract=cleanup_and_extract)
            for text in generated_texts
        ]


__all__ = [
    "IsaacConfig",
    "IsaacTextConfig",
    "IsaacTextModel",
    "IsaacVisionConfig",
    "IsaacVisionModel",
    "IsaacModel",
    "IsaacPreTrainedModel",  # noqa: F822
    "IsaacForConditionalGeneration",
    "IsaacImageProcessor",
    "IsaacProcessor",
]
