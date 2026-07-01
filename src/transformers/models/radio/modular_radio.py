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

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..dinov2.modeling_dinov2 import (
    Dinov2Attention,
    Dinov2Layer,
    Dinov2LayerScale,
    Dinov2MLP,
    Dinov2SelfAttention,
)
from .configuration_radio import RadioConfig


logger = logging.get_logger(__name__)

__all__ = ["RadioModel", "RadioPreTrainedModel"]


@dataclass
class RadioModelOutput(ModelOutput):
    """Output of [`RadioModel`].

    Args:
        summary (`torch.FloatTensor` of shape `(batch_size, num_summary_idxs * hidden_size)`):
            Flattened summary embedding, gathered from the cls tokens selected by `config.summary_idxs`.
        features (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            Dense spatial patch features.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Full token sequence (prefix tokens + patches) from the final encoder layer.
        hidden_states (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `(batch_size, sequence_length, hidden_size)` tensors, one for the embedding output plus one for
            each encoder layer.
        attentions (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True`):
            Tuple of `(batch_size, num_heads, sequence_length, sequence_length)` attention weights, one per layer.
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
        self.register_buffer("norm_mean", torch.tensor(config.norm_mean).view(-1, 1, 1), persistent=True)
        self.register_buffer("norm_std", torch.tensor(config.norm_std).view(-1, 1, 1), persistent=True)

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
        self.embed_dim = config.hidden_size
        self.num_cls_tokens = config.num_cls_tokens
        self.num_registers = config.num_registers

        self.max_rows = config.max_img_size // config.patch_size
        self.max_cols = config.max_img_size // config.patch_size
        num_positions = self.max_rows * self.max_cols

        self.patch_projection = nn.Linear(config.num_channels * config.patch_size**2, config.hidden_size, bias=False)
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

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = self.patch_projection(self._image_to_patches(pixel_values))
        input_dims = (pixel_values.shape[-2] // self.patch_size, pixel_values.shape[-1] // self.patch_size)
        patches = patches + self._interpolate_position_embedding(input_dims, patches.dtype)
        prefix = self.cls_register_token.unsqueeze(0).expand(patches.shape[0], -1, -1)
        return torch.cat([prefix, patches], dim=1)


class RadioMLP(Dinov2MLP):
    pass


class RadioLayerScale(Dinov2LayerScale):
    pass


class RadioSelfAttention(Dinov2SelfAttention):
    pass


class RadioAttention(Dinov2Attention):
    pass


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
        "attentions": RadioSelfAttention,
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
    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> BaseModelOutput:
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


@auto_docstring
class RadioModel(RadioPreTrainedModel):
    def __init__(self, config: RadioConfig):
        super().__init__(config)
        self.config = config
        self.input_conditioner = RadioInputConditioner(config)
        self.embeddings = RadioPatchEmbeddings(config)
        self.encoder = RadioEncoder(config)
        self.register_buffer("summary_idxs", torch.tensor(config.summary_idxs, dtype=torch.long), persistent=True)
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
    def forward(self, pixel_values: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> RadioModelOutput:
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
