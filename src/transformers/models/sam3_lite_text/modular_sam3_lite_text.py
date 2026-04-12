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

from ... import initialization as init
from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutputWithPooling
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
from ..sam3.modeling_sam3 import Sam3Model, Sam3PreTrainedModel
from ..siglip.modeling_siglip import SiglipAttention, SiglipEncoderLayer, SiglipMLP


@auto_docstring(checkpoint="facebook/sam3_lite_text")
@strict
class Sam3LiteTextViTConfig(PreTrainedConfig):
    r"""
    rope_theta (`float`, *optional*, defaults to 10000.0):
        Base frequency for RoPE.
    window_size (`int`, *optional*, defaults to 24):
        Window size for windowed attention.
    global_attn_indexes (`list[int]`, *optional*, defaults to `[7, 15, 23, 31]`):
        Indexes of layers with global attention.
    pretrain_image_size (`int`, *optional*, defaults to 336):
        Pretrained model image size for position embedding initialization.
    hidden_dropout (`float`, *optional*, defaults to 0.0):
        Dropout probability for hidden states.
    """

    base_config_key = "backbone_config"
    model_type = "sam3_vit_model"

    hidden_size: int = 1024
    intermediate_size: int = 4736
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 1008
    patch_size: int | list[int] | tuple[int, int] = 14
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    rope_theta: float = 10000.0
    window_size: int = 24
    global_attn_indexes: list[int] | None = None
    layer_scale_init_value: float | None = None
    pretrain_image_size: int | list[int] | tuple[int, int] = 336
    hidden_dropout: float | int = 0.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.global_attn_indexes is None:
            self.global_attn_indexes = [7, 15, 23, 31]


@auto_docstring(checkpoint="facebook/sam3_lite_text")
@strict
class Sam3LiteTextVisionConfig(PreTrainedConfig):
    r"""
    fpn_hidden_size (`int`, *optional*, defaults to 256):
        The hidden dimension of the FPN.
    backbone_feature_sizes (`List[List[int]]`, *optional*, defaults to `[[288, 288], [144, 144], [72, 72]]`):
        The spatial sizes (height, width) of the feature maps from the backbone at different scales.
    scale_factors (`list[float]`, *optional*, defaults to `[4.0, 2.0, 1.0, 0.5]`):
        Scale factors for FPN multi-scale features. List of scaling factors for each FPN level.
    """

    base_config_key = "vision_config"
    model_type = "sam3_vision_model"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    fpn_hidden_size: int = 256
    backbone_feature_sizes: list | None = None
    scale_factors: list[float] | None = None
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.scale_factors = [4.0, 2.0, 1.0, 0.5] if self.scale_factors is None else self.scale_factors
        if self.backbone_feature_sizes is None:
            self.backbone_feature_sizes = [[288, 288], [144, 144], [72, 72]]

        if isinstance(self.backbone_config, dict):
            self.backbone_config["model_type"] = self.backbone_config.get("model_type", "sam3_vit_model")
            self.backbone_config = CONFIG_MAPPING[self.backbone_config["model_type"]](**self.backbone_config)
        elif self.backbone_config is None:
            self.backbone_config = CONFIG_MAPPING["sam3_vit_model"]()

        super().__post_init__(**kwargs)

    @property
    def image_size(self):
        """Image size for the vision encoder."""
        return self.backbone_config.image_size

    @image_size.setter
    def image_size(self, value):
        """Set the image size and propagate to backbone."""
        self.backbone_config.image_size = value


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
    layer_scale_init_value (`float`, *optional*, defaults to `1e-5`):
        Initial value for the learnable layer-scale parameters in RepMixer blocks (residual branches).
    repmixer_kernel_size (`int`, *optional*, defaults to `11`):
        Kernel size for depthwise convolutions in RepMixer blocks (token mixer and convolutional feed-forward path).
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
    layer_scale_init_value: float = 1e-5
    repmixer_kernel_size: int = 11


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


class Sam3LiteTextMobileOneBlock(nn.Module):
    """Depthwise conv branch with batch norm on the skip path and after the conv (MobileOne-style)."""

    def __init__(self, hidden_size: int, kernel_size: int = 3):
        super().__init__()
        self.batchnorm_skip = nn.BatchNorm2d(hidden_size)
        self.conv = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=(0, kernel_size // 2),
            groups=hidden_size,
            bias=False,
        )
        self.batchnorm_conv = nn.BatchNorm2d(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.batchnorm_conv(self.conv(hidden_states))
        hidden_states = hidden_states + self.batchnorm_skip(residual)
        return hidden_states


class Sam3LiteTextConvMLP(SiglipMLP):
    """Pointwise MLP using 1×1 convolutions, compatible with 4-D (B, C, H, W) feature maps."""

    def __init__(self, config: Sam3LiteTextTextConfig):
        nn.Module.__init__(self)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Conv2d(config.hidden_size, config.intermediate_size, kernel_size=1)
        self.fc2 = nn.Conv2d(config.intermediate_size, config.hidden_size, kernel_size=1)


class Sam3LiteTextConvolutionalFeedForward(nn.Module):
    """Convolutional feed-forward network: depthwise conv + two pointwise projections."""

    def __init__(self, config: Sam3LiteTextTextConfig):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=(1, config.repmixer_kernel_size),
            padding=(0, config.repmixer_kernel_size // 2),
            groups=config.hidden_size,
            bias=False,
        )
        self.depthwise_batchnorm = nn.BatchNorm2d(config.hidden_size)
        self.mlp = Sam3LiteTextConvMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.depthwise_batchnorm(self.depthwise_conv(hidden_states))
        return self.mlp(hidden_states)


class Sam3LiteTextLayerScaledResidual(nn.Module):
    """Common layer-scale residual pattern shared by the RepMixer and feed-forward branches."""

    def __init__(self, hidden_size: int, layer_scale_init_value: float):
        super().__init__()
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((hidden_size, 1, 1)), requires_grad=True)

    def layer_scale_residual(self, hidden_states: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.layer_scale * update


class Sam3LiteTextRepMixer(Sam3LiteTextLayerScaledResidual):
    """Re-parameterisable depthwise-conv token mixer operating on 1D sequence data."""

    def __init__(self, config: Sam3LiteTextTextConfig):
        super().__init__(config.hidden_size, config.layer_scale_init_value)
        self.reference_batchnorm = nn.BatchNorm2d(config.hidden_size)
        self.mixer = Sam3LiteTextMobileOneBlock(config.hidden_size, kernel_size=config.repmixer_kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layer_scale_residual(
            hidden_states, self.mixer(hidden_states) - self.reference_batchnorm(hidden_states)
        )


class Sam3LiteTextRepMixerBlock(Sam3LiteTextLayerScaledResidual):
    """Token-mixing RepMixer plus a convolutional feed-forward path, each with layer scale."""

    def __init__(self, config: Sam3LiteTextTextConfig):
        super().__init__(config.hidden_size, config.layer_scale_init_value)
        self.token_mixer = Sam3LiteTextRepMixer(config)
        self.conv_feed_forward = Sam3LiteTextConvolutionalFeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2).unsqueeze(2)
        hidden_states = self.token_mixer(hidden_states)
        hidden_states = self.layer_scale_residual(hidden_states, self.conv_feed_forward(hidden_states))
        return hidden_states.squeeze(2).transpose(1, 2)


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
class Sam3LiteTextPreTrainedModel(Sam3PreTrainedModel):
    config_class = Sam3LiteTextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Sam3LiteTextTextPositionEmbedding):
            init.normal_(module.position_embedding, std=module.position_embedding.shape[-1] ** -0.5)
        elif isinstance(module, Sam3LiteTextTextModel):
            init.normal_(module.projection.weight, std=module.config.hidden_size**-0.5)


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
    _can_record_outputs = {
        "hidden_states": Sam3LiteTextTextEncoderLayer,
        "attentions": Sam3LiteTextTextAttention,
    }

    def __init__(self, config: Sam3LiteTextTextConfig):
        super().__init__(config)
        self.embeddings = Sam3LiteTextTextEmbeddings(config)
        repmixer_positions = {0, config.num_hidden_layers - 1} if config.use_repmixer_blocks else set()
        self.layers = nn.ModuleList(
            [
                Sam3LiteTextRepMixerBlock(config) if i in repmixer_positions else Sam3LiteTextTextEncoderLayer(config)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Sam3LiteTextTextEncoderOutput:
        hidden_states = self.embeddings(input_ids)
        attention_mask = create_bidirectional_mask(self.config, hidden_states, attention_mask)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask, **kwargs)

        hidden_states = self.final_layer_norm(hidden_states)

        pooled = hidden_states[
            torch.arange(hidden_states.shape[0], device=hidden_states.device), input_ids.argmax(dim=-1)
        ]
        pooled = self.projection(pooled)
        return Sam3LiteTextTextEncoderOutput(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
        )


class Sam3LiteTextModel(Sam3Model):
    # DETR components create float masks from features, so flash/flex attention cannot be dispatched safely.
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
