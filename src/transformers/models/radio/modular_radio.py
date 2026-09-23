# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import get_max_seqlen, is_flash_attention_requested, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..dinov2.modeling_dinov2 import (
    Dinov2Attention,
    Dinov2Layer,
    Dinov2LayerScale,
    Dinov2MLP,
    eager_attention_forward,
)
from .configuration_radio import RadioConfig


logger = logging.get_logger(__name__)

__all__ = ["RadioModel", "RadioPreTrainedModel"]


@auto_docstring(custom_intro="Output of [`RadioModel`].")
@dataclass
class RadioModelOutput(ModelOutput):
    r"""
    summary (`torch.FloatTensor` of shape `(batch_size, num_summary_idxs * hidden_size)`):
        Flattened summary embedding, gathered from the cls tokens selected by `config.summary_idxs`.
    features (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
        Dense spatial patch features. For packed inputs (`image_grid_hw` given), the patch features of all images
        concatenated, of shape `(total_patches, hidden_size)`.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Full token sequence (prefix tokens + patches) from the final encoder layer. For packed inputs, the
        sequences of all images concatenated, of shape `(total_sequence_length, hidden_size)`.
    """

    summary: torch.FloatTensor | None = None
    features: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


class RadioInputConditioner(nn.Module):
    """Normalizes pixel values; arithmetic is done in float32 then cast back."""

    def __init__(self, config: RadioConfig):
        super().__init__()
        self.norm_mean = nn.Buffer(torch.tensor(config.norm_mean).view(-1, 1, 1), persistent=True)
        self.norm_std = nn.Buffer(torch.tensor(config.norm_std).view(-1, 1, 1), persistent=True)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        normalized = (pixel_values.float() - self.norm_mean.float()) / self.norm_std.float()
        return normalized.to(pixel_values.dtype)


class RadioPatchEmbeddings(nn.Module):
    """Cropped Position Embedding (CPE) patch generator.

    Splits the image into patches, projects them, adds a resolution-interpolated
    absolute position embedding, and prepends learned cls + register tokens.
    """

    def __init__(self, config: RadioConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels

        self.max_rows = config.max_img_size // config.patch_size
        self.max_cols = config.max_img_size // config.patch_size
        num_positions = self.max_rows * self.max_cols

        self.patch_projection = nn.Linear(config.num_channels * config.patch_size**2, config.hidden_size, bias=False)
        self.video_patch_projection = (
            # CODEPATH: video-capable checkpoints such as `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`
            # set `video_temporal_patch_size`; image-only ones such as `nvidia/C-RADIOv4-H` leave it unset.
            nn.Linear(
                config.video_temporal_patch_size * config.num_channels * config.patch_size**2,
                config.hidden_size,
                bias=False,
            )
            if config.video_temporal_patch_size is not None
            else None
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, num_positions, config.hidden_size))
        self.cls_register_token = nn.Parameter(
            torch.zeros(config.num_cls_tokens + config.num_registers, config.hidden_size)
        )

    def _image_to_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        ps = self.patch_size
        batch, channels, height, width = pixel_values.shape
        rows, cols = height // ps, width // ps
        patches = pixel_values.reshape(batch, channels, rows, ps, cols, ps)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, rows * cols, channels * ps * ps)
        return patches

    def _interpolate_position_embedding(self, input_dims: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
        pos = self.position_embedding.reshape(1, self.max_rows, self.max_cols, -1).permute(0, 3, 1, 2)
        max_dim = max(input_dims)
        pos = F.interpolate(pos.float(), size=(max_dim, max_dim), mode="bilinear", align_corners=False).to(dtype)
        if input_dims[0] < pos.shape[-2]:
            pos = pos[..., : input_dims[0], :]
        if input_dims[1] < pos.shape[-1]:
            pos = pos[..., :, : input_dims[1]]
        if pos.shape[-2:] != tuple(input_dims):
            pos = F.interpolate(pos.float(), size=tuple(input_dims), mode="bilinear", align_corners=False).to(dtype)
        return pos.flatten(2).permute(0, 2, 1)

    def forward(self, pixel_values: torch.Tensor, image_grid_hw: torch.LongTensor | None = None) -> torch.Tensor:
        if image_grid_hw is not None:
            return self._embed_packed_patches(pixel_values, image_grid_hw)

        # temporally-packed video stacks `video_temporal_patch_size` frames along the channel dim
        is_video = pixel_values.shape[1] != self.num_channels
        if is_video and self.video_patch_projection is None:
            raise ValueError(
                f"Expected {self.num_channels} input channels, got {pixel_values.shape[1]}. Temporally-packed "
                "video input requires `config.video_temporal_patch_size` to be set."
            )
        projection = self.video_patch_projection if is_video else self.patch_projection
        patches = projection(self._image_to_patches(pixel_values))
        input_dims = (pixel_values.shape[-2] // self.patch_size, pixel_values.shape[-1] // self.patch_size)
        patches = patches + self._interpolate_position_embedding(input_dims, patches.dtype)
        prefix = self.cls_register_token.unsqueeze(0).expand(patches.shape[0], -1, -1)
        return torch.cat([prefix, patches], dim=1)

    def _embed_packed_patches(self, pixel_values: torch.Tensor, image_grid_hw: torch.LongTensor) -> torch.Tensor:
        """Embeds the concatenated patches of images with differing grids into one `(1, total_length, hidden)` sequence."""
        patches = self.patch_projection(pixel_values)
        embeddings = []
        for (grid_height, grid_width), image_patches in zip(
            image_grid_hw.tolist(), patches.split(image_grid_hw.prod(-1).tolist())
        ):
            position_embedding = self._interpolate_position_embedding((grid_height, grid_width), patches.dtype)
            embeddings.extend([self.cls_register_token, image_patches + position_embedding[0]])
        return torch.cat(embeddings).unsqueeze(0)


class RadioMLP(Dinov2MLP):
    pass


class RadioLayerScale(Dinov2LayerScale):
    pass


class RadioAttention(Dinov2Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        cu_seqlens (`torch.Tensor` of shape `(num_images + 1,)`, *optional*):
            Boundaries of the image sequences packed into `hidden_states`; each image only attends to itself.
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attention_kwargs = {"dropout": 0.0 if not self.training else self.attention_dropout, "scaling": self.scaling}

        if cu_seqlens is None:
            attn_output, attn_weights = attention_interface(
                self, query_states, key_states, value_states, attention_mask, **attention_kwargs, **kwargs
            )
        elif is_flash_attention_requested(self.config):
            max_seqlen = get_max_seqlen(cu_seqlens, self.config, kwargs)
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                None,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                **attention_kwargs,
                **kwargs,
            )
        else:
            # without a varlen kernel, attend within each image separately
            splits = [
                torch.split(states, (cu_seqlens[1:] - cu_seqlens[:-1]).tolist(), dim=2)
                for states in (query_states, key_states, value_states)
            ]
            outputs = [
                attention_interface(self, query, key, value, None, **attention_kwargs, **kwargs)
                for query, key, value in zip(*splits)
            ]
            attn_output = torch.cat([output[0] for output in outputs], dim=1)
            attn_weights = None
            # eager returns per-image probabilities; lay them out block-diagonally over the packed sequence
            if outputs[0][1] is not None:
                total_length = query_states.shape[2]
                attn_weights = query_states.new_zeros(
                    query_states.shape[0], self.num_attention_heads, total_length, total_length
                )
                start = 0
                for _, weights in outputs:
                    end = start + weights.shape[-1]
                    attn_weights[..., start:end, start:end] = weights
                    start = end

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class RadioLayer(Dinov2Layer):
    pass


@auto_docstring
class RadioPreTrainedModel(PreTrainedModel):
    config_class = RadioConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RadioLayer"]
    _keys_to_ignore_on_load_missing = [r"layer_scale\d+\.lambda1"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _can_record_outputs = {
        "hidden_states": RadioLayer,
        "attentions": RadioAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        # Use `transformers.initialization` (not in-place `.data` ops) so the
        # framework's `_is_hf_initialized` guard skips already-loaded params.
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            init.trunc_normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        elif isinstance(module, RadioPatchEmbeddings):
            init.trunc_normal_(module.position_embedding, mean=0.0, std=std)
            init.trunc_normal_(module.cls_register_token, mean=0.0, std=std)
        elif isinstance(module, RadioLayerScale):
            init.constant_(module.lambda1, self.config.layerscale_value)
        elif isinstance(module, RadioInputConditioner):
            init.copy_(module.norm_mean, torch.tensor(self.config.norm_mean).view(-1, 1, 1))
            init.copy_(module.norm_std, torch.tensor(self.config.norm_std).view(-1, 1, 1))
        elif isinstance(module, RadioModel):
            init.copy_(module.summary_idxs, torch.tensor(self.config.summary_idxs, dtype=torch.long))


class RadioEncoder(RadioPreTrainedModel):
    def __init__(self, config: RadioConfig):
        super().__init__(config)
        self.layer = nn.ModuleList([RadioLayer(config) for _ in range(config.num_hidden_layers)])
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> BaseModelOutput:
        for layer in self.layer:
            hidden_states = layer(hidden_states, **kwargs)
        return BaseModelOutput(last_hidden_state=hidden_states)


@auto_docstring
class RadioModel(RadioPreTrainedModel):
    def __init__(self, config: RadioConfig):
        super().__init__(config)
        self.config = config
        self.input_conditioner = RadioInputConditioner(config)
        self.embeddings = RadioPatchEmbeddings(config)
        self.encoder = RadioEncoder(config)
        self.summary_idxs = nn.Buffer(torch.tensor(config.summary_idxs, dtype=torch.long), persistent=True)
        self.post_init()

    @property
    def patch_size(self) -> int:
        return self.config.patch_size

    def make_preprocessor_external(self):
        """Detach the input conditioner (caller applies normalization itself)."""
        conditioner = self.input_conditioner
        self.input_conditioner = nn.Identity()
        return conditioner

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_hw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> RadioModelOutput:
        r"""
        pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` or `(total_patches, num_channels * patch_size**2)`):
            Images of one size, or, with `image_grid_hw`, the flattened patches of images of different sizes
            concatenated. Each patch is laid out channel-major, i.e. `(num_channels, patch_size, patch_size)`.
        image_grid_hw (`torch.LongTensor` of shape `(num_images, 2)`, *optional*):
            Patch grid `(height, width)` of each image in packed `pixel_values`.
        """
        if image_grid_hw is not None:
            return self._forward_packed(pixel_values, image_grid_hw, **kwargs)

        pixel_values = self.input_conditioner(pixel_values)
        hidden_states = self.embeddings(pixel_values)
        encoder_outputs: BaseModelOutput = self.encoder(hidden_states, **kwargs)
        last_hidden_state = encoder_outputs.last_hidden_state

        num_skip = self.config.num_summary_tokens
        all_summary = last_hidden_state[:, : self.config.num_cls_tokens]
        summary = all_summary[:, self.summary_idxs].flatten(1)
        features = last_hidden_state[:, num_skip:]

        return RadioModelOutput(
            summary=summary,
            features=features,
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _forward_packed(
        self, pixel_values: torch.Tensor, image_grid_hw: torch.LongTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> RadioModelOutput:
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        pixel_values = self.input_conditioner(pixel_values.view(-1, num_channels, patch_size, patch_size))
        hidden_states = self.embeddings(pixel_values.flatten(1), image_grid_hw)

        num_prefix_tokens = self.config.num_summary_tokens
        sequence_lengths = image_grid_hw.prod(-1) + num_prefix_tokens
        cu_seqlens = F.pad(sequence_lengths.cumsum(0), (1, 0)).to(torch.int32)
        # a single image is a plain dense sequence and needs no packed attention
        encoder_outputs: BaseModelOutput = self.encoder(
            hidden_states, cu_seqlens=cu_seqlens if len(sequence_lengths) > 1 else None, **kwargs
        )
        last_hidden_state = encoder_outputs.last_hidden_state[0]

        starts = cu_seqlens[:-1].long()
        summary = last_hidden_state[starts[:, None] + self.summary_idxs[None, :]].flatten(1)
        positions = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
        positions_in_image = positions - starts.repeat_interleave(sequence_lengths)
        features = last_hidden_state[positions_in_image >= num_prefix_tokens]

        return RadioModelOutput(
            summary=summary,
            features=features,
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
