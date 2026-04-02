# Copyright 2026 the HuggingFace Team. All rights reserved.
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
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.generic import TransformersKwargs, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..sam3.configuration_sam3 import (
    Sam3DETRDecoderConfig,
    Sam3DETREncoderConfig,
    Sam3GeometryEncoderConfig,
    Sam3MaskDecoderConfig,
)
from ..sam3.modeling_sam3 import Sam3Model
from ..siglip.modeling_siglip import SiglipAttention, SiglipEncoderLayer, SiglipMLP


@auto_docstring(checkpoint="facebook/sam3_lite_text")
@strict
class Sam3LiteTextGeometryEncoderConfig(Sam3GeometryEncoderConfig):
    pass


@auto_docstring(checkpoint="facebook/sam3_lite_text")
@strict
class Sam3LiteTextDETREncoderConfig(Sam3DETREncoderConfig):
    pass


@auto_docstring(checkpoint="facebook/sam3_lite_text")
@strict
class Sam3LiteTextDETRDecoderConfig(Sam3DETRDecoderConfig):
    pass


@auto_docstring(checkpoint="facebook/sam3_lite_text")
@strict
class Sam3LiteTextMaskDecoderConfig(Sam3MaskDecoderConfig):
    pass


@auto_docstring(checkpoint="yonigozlan/sam3-litetext-s0")
@strict
class Sam3LiteTextTextConfig(PreTrainedConfig):
    r"""
    use_repmixer_blocks (`bool`, *optional*, defaults to `True`):
        Whether to use RepMixer blocks (MobileCLIP-style) for the first and last encoder layers.
        When `False`, all layers are standard Transformer encoder layers.
    """

    model_type = "sam3_lite_text_text_model"

    vocab_size: int = 49408
    hidden_size: int = 512
    intermediate_size: int = 2048
    projection_dim: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 77
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    use_repmixer_blocks: bool = True


@auto_docstring(checkpoint="facebook/sam3_lite_text")
@strict
class Sam3LiteTextConfig(PreTrainedConfig):
    r"""
    geometry_encoder_config (`dict` or `Sam3LiteTextGeometryEncoderConfig`, *optional*):
        Configuration for the geometry encoder.
    detr_encoder_config (`dict` or `Sam3LiteTextDETREncoderConfig`, *optional*):
        Configuration for the DETR encoder.
    detr_decoder_config (`dict` or `Sam3LiteTextDETRDecoderConfig`, *optional*):
        Configuration for the DETR decoder.
    mask_decoder_config (`dict` or `Sam3LiteTextMaskDecoderConfig`, *optional*):
        Configuration for the mask decoder.

    Example:
    ```python
    >>> from transformers import Sam3LiteTextConfig, Sam3LiteTextModel

    >>> # Initializing a SAM3_LITE_TEXT configuration
    >>> configuration = Sam3LiteTextConfig()

    >>> # Initializing a model from the configuration
    >>> model = Sam3LiteTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "sam3_lite_text"
    is_composition = True
    sub_configs = {
        "vision_config": AutoConfig,
        "text_config": Sam3LiteTextTextConfig,
        "geometry_encoder_config": Sam3LiteTextGeometryEncoderConfig,
        "detr_encoder_config": Sam3LiteTextDETREncoderConfig,
        "detr_decoder_config": Sam3LiteTextDETRDecoderConfig,
        "mask_decoder_config": Sam3LiteTextMaskDecoderConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    geometry_encoder_config: dict | PreTrainedConfig | None = None
    detr_encoder_config: dict | PreTrainedConfig | None = None
    detr_decoder_config: dict | PreTrainedConfig | None = None
    mask_decoder_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "sam3_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["sam3_vision_model"]()

        if self.text_config is None:
            self.text_config = Sam3LiteTextTextConfig()
        if isinstance(self.text_config, dict):
            self.text_config = Sam3LiteTextTextConfig(**self.text_config)

        if self.geometry_encoder_config is None:
            self.geometry_encoder_config = Sam3LiteTextGeometryEncoderConfig()
        if isinstance(self.geometry_encoder_config, dict):
            self.geometry_encoder_config = Sam3LiteTextGeometryEncoderConfig(**self.geometry_encoder_config)

        if self.detr_encoder_config is None:
            self.detr_encoder_config = Sam3LiteTextDETREncoderConfig()
        if isinstance(self.detr_encoder_config, dict):
            self.detr_encoder_config = Sam3LiteTextDETREncoderConfig(**self.detr_encoder_config)

        if self.detr_decoder_config is None:
            self.detr_decoder_config = Sam3LiteTextDETRDecoderConfig()
        if isinstance(self.detr_decoder_config, dict):
            self.detr_decoder_config = Sam3LiteTextDETRDecoderConfig(**self.detr_decoder_config)

        if self.mask_decoder_config is None:
            self.mask_decoder_config = Sam3LiteTextMaskDecoderConfig()
        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = Sam3LiteTextMaskDecoderConfig(**self.mask_decoder_config)

        super().__post_init__(**kwargs)

    @property
    def image_size(self):
        """Image size for the SAM3_LITE_TEXT model."""
        return self.vision_config.image_size

    @image_size.setter
    def image_size(self, value):
        """Set the image size and propagate to vision config."""
        self.vision_config.image_size = value


@dataclass
class Sam3LiteTextTextEncoderOutput(BaseModelOutputWithPooling):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Full sequence of hidden states from the text encoder.
    pooler_output (`torch.FloatTensor` of shape `(batch_size, projection_dim)`):
        EOT-pooled output projected to `projection_dim` via the internal CLIP-style projection.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*):
        Tuple of hidden states at each layer, returned when `output_hidden_states=True`.
    attentions (`tuple(torch.FloatTensor)`, *optional*):
        Tuple of attention weights at each transformer layer, returned when `output_attentions=True`.
    """


class Sam3LiteTextTextPositionEmbedding(nn.Module):
    """Learnable positional embedding with bilinear interpolation for variable sequence lengths."""

    def __init__(self, max_position_embeddings: int, hidden_size: int):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.empty(1, 1, max_position_embeddings, hidden_size))

    def forward(self, seq_len: int) -> torch.Tensor:
        position_embedding = self.position_embedding
        if seq_len != position_embedding.shape[2]:
            position_embedding = F.interpolate(
                position_embedding,
                size=(seq_len, position_embedding.shape[-1]),
                mode="bilinear",
            )
        return position_embedding.reshape(1, seq_len, -1)


class Sam3LiteTextRepMixer(nn.Module):
    """Re-parameterisable depthwise-conv token mixer operating on 1D sequence data."""

    def __init__(self, hidden_size: int, kernel_size: int = 11):
        super().__init__()
        self.norm = Sam3LiteTextMobileOneBlock(hidden_size, kernel_size=kernel_size, use_conv_branch=False)
        self.mixer = Sam3LiteTextMobileOneBlock(hidden_size, kernel_size=kernel_size, use_conv_branch=True)
        self.layer_scale = nn.Parameter(1e-5 * torch.ones((hidden_size, 1, 1)), requires_grad=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.layer_scale * (self.mixer(hidden_states) - self.norm(hidden_states))


class Sam3LiteTextMobileOneBlock(nn.Module):
    """Multi-branch depthwise conv block with BN-skip and optional conv branch."""

    def __init__(self, hidden_size: int, kernel_size: int = 3, use_conv_branch: bool = True):
        super().__init__()
        self.rbr_skip = nn.BatchNorm2d(hidden_size)
        self.rbr_conv = nn.ModuleList()
        if use_conv_branch:
            self.rbr_conv.append(
                nn.Sequential(
                    nn.Conv2d(
                        hidden_size,
                        hidden_size,
                        kernel_size=(1, kernel_size),
                        stride=1,
                        padding=(0, kernel_size // 2),
                        groups=hidden_size,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_size),
                )
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.rbr_skip(hidden_states)
        for branch in self.rbr_conv:
            output = output + branch(hidden_states)
        return output


class Sam3LiteTextRepMixerBlock(nn.Module):
    """Full token-mixing block: RepMixer (depthwise conv) + ConvFFN with layer scale."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.layer_scale = nn.Parameter(1e-5 * torch.ones((hidden_size, 1, 1)), requires_grad=True)
        self.token_mixer = Sam3LiteTextRepMixer(hidden_size, kernel_size=11)
        self.convffn = Sam3LiteTextConvFFN(hidden_size, intermediate_size, kernel_size=11)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.token_mixer(hidden_states)
        return hidden_states + self.layer_scale * self.convffn(hidden_states)


class Sam3LiteTextConvFFN(nn.Module):
    """Convolutional feed-forward network: depthwise conv + two pointwise projections."""

    def __init__(self, hidden_size: int, intermediate_size: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                hidden_size,
                hidden_size,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2),
                groups=hidden_size,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_size),
        )
        self.fc1 = nn.Conv2d(hidden_size, intermediate_size, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(intermediate_size, hidden_size, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Sam3LiteTextTextAttention(SiglipAttention):
    pass


class Sam3LiteTextTextMLP(SiglipMLP):
    pass


class Sam3LiteTextTextEncoderLayer(SiglipEncoderLayer):
    def __init__(self, config: Sam3LiteTextTextConfig):
        super().__init__(config)
        self.self_attn = Sam3LiteTextTextAttention(config)
        self.mlp = Sam3LiteTextTextMLP(config)


class Sam3LiteTextTextEmbeddings(nn.Module):
    """Token embedding + interpolatable positional embedding for the text encoder."""

    def __init__(self, config: Sam3LiteTextTextConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = Sam3LiteTextTextPositionEmbedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        hidden_states = self.token_embedding(input_ids)
        hidden_states = hidden_states + self.position_embedding(input_ids.shape[1]).to(hidden_states.dtype)
        return hidden_states


@auto_docstring
class Sam3LiteTextPreTrainedModel(PreTrainedModel):
    config_class = Sam3LiteTextConfig
    base_model_prefix = "sam3_lite_text"
    main_input_name = "pixel_values"
    input_modalities = ["image", "text"]
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Sam3LiteTextTextPositionEmbedding):
            nn.init.normal_(module.position_embedding, std=module.position_embedding.shape[-1] ** -0.5)
        elif isinstance(module, Sam3LiteTextTextModel):
            nn.init.normal_(module.projection, std=module.config.hidden_size**-0.5)


@auto_docstring(
    custom_intro="""
    MobileCLIP MCT text encoder used in EfficientSAM3 LiteText.

    When `config.use_repmixer_blocks` is `True`, the first and last layers are
    `Sam3LiteTextRepMixerBlock` modules; the rest are standard `Sam3LiteTextTextEncoderLayer` layers.
"""
)
class Sam3LiteTextTextModel(Sam3LiteTextPreTrainedModel):
    config_class = Sam3LiteTextTextConfig
    config: Sam3LiteTextTextConfig
    _no_split_modules = ["Sam3LiteTextTextEmbeddings", "Sam3LiteTextTextEncoderLayer", "Sam3LiteTextRepMixerBlock"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Sam3LiteTextTextEncoderLayer,
        "attentions": Sam3LiteTextTextAttention,
    }

    def __init__(self, config: Sam3LiteTextTextConfig):
        super().__init__(config)
        self.embeddings = Sam3LiteTextTextEmbeddings(config)
        if config.use_repmixer_blocks:
            num_transformer_layers = config.num_hidden_layers - 2
            self.layers = nn.ModuleList(
                [Sam3LiteTextRepMixerBlock(config.hidden_size, config.intermediate_size)]
                + [Sam3LiteTextTextEncoderLayer(config) for _ in range(num_transformer_layers)]
                + [Sam3LiteTextRepMixerBlock(config.hidden_size, config.intermediate_size)]
            )
            self.repmixer_layer_indices = frozenset({0, config.num_hidden_layers - 1})
        else:
            self.layers = nn.ModuleList(
                [Sam3LiteTextTextEncoderLayer(config) for _ in range(config.num_hidden_layers)]
            )
            self.repmixer_layer_indices = frozenset()
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.projection = nn.Parameter(torch.empty(config.hidden_size, config.projection_dim))
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Sam3LiteTextTextEncoderOutput:
        hidden_states = self.embeddings(input_ids)

        if attention_mask is not None:
            attention_mask = create_bidirectional_mask(self.config, hidden_states, attention_mask)

        for idx, layer in enumerate(self.layers):
            if idx in self.repmixer_layer_indices:
                hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)
                hidden_states = layer(hidden_states)
                hidden_states = hidden_states.squeeze(2).permute(0, 2, 1)
            else:
                hidden_states = layer(hidden_states, attention_mask=attention_mask, **kwargs)

        hidden_states = self.final_layer_norm(hidden_states)

        pooled = hidden_states[
            torch.arange(hidden_states.shape[0], device=hidden_states.device), input_ids.argmax(dim=-1)
        ]
        pooled = pooled @ self.projection
        return Sam3LiteTextTextEncoderOutput(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
        )


class Sam3LiteTextModel(Sam3Model):
    _supports_flash_attn = False
    _supports_flex_attn = False

    def __init__(self, config: Sam3LiteTextConfig):
        super().__init__(config)
        self.text_encoder = Sam3LiteTextTextModel(config.text_config)
        self.vision_encoder = AutoModel.from_config(config.vision_config)


__all__ = [
    "Sam3LiteTextConfig",
    "Sam3LiteTextTextConfig",
    "Sam3LiteTextGeometryEncoderConfig",
    "Sam3LiteTextDETREncoderConfig",
    "Sam3LiteTextDETRDecoderConfig",
    "Sam3LiteTextMaskDecoderConfig",
    "Sam3LiteTextModel",
    "Sam3LiteTextPreTrainedModel",
    "Sam3LiteTextTextModel",
]
