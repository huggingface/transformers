# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""
SAM3-LiteText: A lightweight variant of SAM3 that replaces the CLIP text encoder
with a MobileCLIP-S0 text encoder using RepMixer blocks for efficient token mixing.

Architecture changes from SAM3:
- Text encoder: CLIPTextModelWithProjection -> MobileCLIP-S0 (RepMixer + Transformer)
- Text hidden size: 1024 -> 512
- Text layers: 24 -> 4 transformer + 2 RepMixer blocks
- Context length: 32 -> 16
- Everything else (ViT backbone, FPN, geometry/DETR encoder/decoder, mask decoder) is unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple, logging
from ...utils.generic import TransformersKwargs
from ..sam3.configuration_sam3 import Sam3Config, Sam3ViTConfig
from ..sam3.modeling_sam3 import Sam3Model, Sam3PreTrainedModel, Sam3ViTModel
from ..sam3.modular_sam3 import Sam3ImageProcessor
from ..sam3.processing_sam3 import Sam3Processor


logger = logging.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class Sam3LiteTextViTConfig(Sam3ViTConfig):
    model_type = "sam3_lite_text_vit_model"


@auto_docstring(checkpoint="Simon7108528/EfficientSAM3")
@strict
class Sam3LiteTextMobileCLIPConfig(PreTrainedConfig):
    r"""
    context_length (`int`, *optional*, defaults to 16):
        Maximum sequence length for text input.
    kernel_size (`int`, *optional*, defaults to 11):
        Kernel size for RepMixer depthwise convolutions.
    layer_scale_init_value (`float`, *optional*, defaults to 1e-5):
        Initial value for learnable layer scale parameters.
    norm_type (`str`, *optional*, defaults to `"layer_norm_fp32"`):
        Type of layer normalization. One of `"layer_norm"` or `"layer_norm_fp32"`.
    projection_dim (`int`, *optional*, defaults to 512):
        Dimension of the text projection output.
    """

    base_config_key = "text_config"
    model_type = "sam3_lite_text_mobileclip"

    hidden_size: int = 512
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    hidden_act: str = "gelu"
    vocab_size: int = 49408
    context_length: int = 16
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    kernel_size: int = 11
    layer_scale_init_value: float = 1e-5
    norm_type: str = "layer_norm_fp32"
    projection_dim: int = 512
    initializer_range: float = 0.02


@auto_docstring(checkpoint="Simon7108528/EfficientSAM3")
@strict
class Sam3LiteTextConfig(Sam3Config):
    r"""
    text_config (`dict` or `Sam3LiteTextMobileCLIPConfig`, *optional*):
        Configuration for the MobileCLIP text encoder.
    geometry_encoder_config (`dict` or `Sam3GeometryEncoderConfig`, *optional*):
        Configuration for the geometry encoder.
    detr_encoder_config (`dict` or `Sam3DETREncoderConfig`, *optional*):
        Configuration for the DETR encoder.
    detr_decoder_config (`dict` or `Sam3DETRDecoderConfig`, *optional*):
        Configuration for the DETR decoder.
    mask_decoder_config (`dict` or `Sam3MaskDecoderConfig`, *optional*):
        Configuration for the mask decoder.

    Example:
    ```python
    >>> from transformers import Sam3LiteTextConfig, Sam3LiteTextModel

    >>> # Initializing a SAM3-LiteText configuration
    >>> configuration = Sam3LiteTextConfig()

    >>> # Initializing a model from the configuration
    >>> model = Sam3LiteTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "sam3_lite_text"

    def __post_init__(self, **kwargs):
        # Override text_config to use MobileCLIP instead of CLIP
        if self.text_config is None:
            self.text_config = Sam3LiteTextMobileCLIPConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = Sam3LiteTextMobileCLIPConfig(**self.text_config)
        elif not isinstance(self.text_config, Sam3LiteTextMobileCLIPConfig):
            # Handle case where sub_configs deserialization created a CLIPTextConfig;
            # convert it to Sam3LiteTextMobileCLIPConfig preserving any shared attributes
            self.text_config = Sam3LiteTextMobileCLIPConfig(**self.text_config.to_dict())

        # Let the parent handle all other sub-configs
        super().__post_init__(**kwargs)


# =============================================================================
# MobileCLIP Text Encoder Components
# =============================================================================


class Sam3LiteTextLayerNormFP32(nn.LayerNorm):
    """LayerNorm that casts input to float32 for numerical stability."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_dtype = input.dtype
        return super().forward(input.to(torch.float32)).to(input_dtype)


class Sam3LiteTextLearnablePositionalEmbedding(nn.Module):
    """Learnable positional embeddings with interpolation support for variable sequence lengths."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.empty(1, 1, num_embeddings, embedding_dim))
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

    def forward(self, seq_len: int) -> torch.Tensor:
        pos_embed = self.pos_embed
        if seq_len != self.num_embeddings:
            pos_embed = F.interpolate(
                pos_embed,
                size=(seq_len, self.embedding_dim),
                mode="bilinear",
                align_corners=False,
            )
        return pos_embed.reshape(1, seq_len, self.embedding_dim)


class Sam3LiteTextMobileOneBlock(nn.Module):
    """
    Reparameterizable convolution block with multi-branch training that fuses
    to a single convolution at inference.

    During training, uses parallel branches (conv+BN, scale+BN, skip+BN).
    At inference, all branches are fused into one convolution via `reparameterize()`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int | tuple[int, int] = 0,
        groups: int = 1,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
    ):
        super().__init__()
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.activation = nn.GELU() if use_act else nn.Identity()

        # Skip (identity) branch: only when dimensions match
        self.rbr_skip = (
            nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        )

        # Convolution branches
        if num_conv_branches > 0:
            self.rbr_conv = nn.ModuleList(
                [self._conv_bn(kernel_size=kernel_size, padding=padding) for _ in range(num_conv_branches)]
            )
        else:
            self.rbr_conv = None

        # Scale (1x1) branch
        self.rbr_scale = None
        ks = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        if ks > 1 and use_scale_branch:
            self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        out = scale_out + identity_out
        if self.rbr_conv is not None:
            for conv_branch in self.rbr_conv:
                out = out + conv_branch(x)

        return self.activation(out)

    def reparameterize(self):
        """Fuse all branches into a single convolution for inference."""
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        for para in self.parameters():
            para.detach_()
        if hasattr(self, "rbr_conv"):
            del self.rbr_conv
            self.rbr_conv = None
        if hasattr(self, "rbr_scale"):
            del self.rbr_scale
            self.rbr_scale = None
        if hasattr(self, "rbr_skip"):
            del self.rbr_skip
            self.rbr_skip = None

    def _get_kernel_bias(self):
        kernel_scale, bias_scale = 0, 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            ks = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[1]
            pad = ks // 2
            kernel_scale = F.pad(kernel_scale, [pad, pad, pad, pad])

        kernel_identity, bias_identity = 0, 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        kernel_conv, bias_conv = 0, 0
        if self.rbr_conv is not None:
            for conv_branch in self.rbr_conv:
                k, b = self._fuse_bn_tensor(conv_branch)
                kernel_conv = kernel_conv + k
                bias_conv = bias_conv + b

        return kernel_conv + kernel_scale + kernel_identity, bias_conv + bias_scale + bias_identity

    def _fuse_bn_tensor(self, branch: nn.Sequential | nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            bn = branch.bn
        else:
            # BatchNorm identity branch
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_size = self.kernel_size
                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size, kernel_size)
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, kernel_size[0], kernel_size[1]),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, kernel_size[0] // 2, kernel_size[1] // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            bn = branch
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    def _conv_bn(self, kernel_size, padding):
        mod = nn.Sequential()
        mod.add_module(
            "conv",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod.add_module("bn", nn.BatchNorm2d(num_features=self.out_channels))
        return mod


class Sam3LiteTextRepMixer(nn.Module):
    """
    Token mixing via reparameterizable depthwise convolution.

    During training: computes `x + layer_scale * (mixer(x) - norm(x))`.
    After reparameterization: a single depthwise convolution.
    """

    def __init__(self, dim: int, kernel_size: int = 3, layer_scale_init_value: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.norm = Sam3LiteTextMobileOneBlock(
            dim,
            dim,
            (1, kernel_size),
            padding=(0, kernel_size // 2),
            groups=dim,
            use_act=False,
            use_scale_branch=False,
            num_conv_branches=0,
        )
        self.mixer = Sam3LiteTextMobileOneBlock(
            dim,
            dim,
            (1, kernel_size),
            padding=(0, kernel_size // 2),
            groups=dim,
            use_act=False,
        )
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            return self.reparam_conv(x)
        return x + self.layer_scale * (self.mixer(x) - self.norm(x))

    def reparameterize(self):
        """Fuse mixer, norm, and layer_scale into a single depthwise convolution."""
        self.mixer.reparameterize()
        self.norm.reparameterize()

        w = self.mixer.id_tensor + self.layer_scale.unsqueeze(-1) * (
            self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
        )
        b = torch.squeeze(self.layer_scale) * (self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias)

        self.reparam_conv = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=(1, self.kernel_size),
            stride=1,
            padding=(0, self.kernel_size // 2),
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

        for para in self.parameters():
            para.detach_()
        del self.mixer
        del self.norm
        del self.layer_scale


class Sam3LiteTextConvFFN(nn.Module):
    """Conv-based feed-forward network: depthwise conv + two pointwise convolutions."""

    def __init__(self, in_channels: int, context_size: int, hidden_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, context_size),
                padding=(0, context_size // 2),
                groups=in_channels,
                bias=False,
            ),
        )
        self.conv.add_module("bn", nn.BatchNorm2d(num_features=in_channels))
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Sam3LiteTextRepMixerBlock(GradientCheckpointingLayer):
    """
    RepMixer block: token mixing via RepMixer + ConvFFN.

    Input shape: (batch, seq_len, dim) -> reshapes to (batch, dim, 1, seq_len) for conv ops.
    """

    def __init__(self, config: Sam3LiteTextMobileCLIPConfig):
        super().__init__()
        dim = config.hidden_size
        kernel_size = config.kernel_size
        mlp_hidden_dim = config.intermediate_size

        self.token_mixer = Sam3LiteTextRepMixer(
            dim,
            kernel_size=kernel_size,
            layer_scale_init_value=config.layer_scale_init_value,
        )
        self.convffn = Sam3LiteTextConvFFN(
            in_channels=dim,
            context_size=kernel_size,
            hidden_channels=mlp_hidden_dim,
        )
        self.layer_scale = nn.Parameter(config.layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # (B, seq, dim) -> (B, dim, 1, seq) for conv operations
        x = x.permute(0, 2, 1).unsqueeze(2)
        x = self.token_mixer(x)
        x = x + self.layer_scale * self.convffn(x)
        # (B, dim, 1, seq) -> (B, seq, dim)
        return x.squeeze(2).permute(0, 2, 1)


class Sam3LiteTextAttention(nn.Module):
    """Multi-head self-attention with fused QKV projection."""

    def __init__(self, config: Sam3LiteTextMobileCLIPConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, seq, head_dim)
        query, key, value = qkv.unbind(0)

        query = query * self.scaling
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask.unsqueeze(1)

        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )

        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(hidden_states.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.out_proj(attn_output)


class Sam3LiteTextTransformerLayer(GradientCheckpointingLayer):
    """Pre-norm transformer encoder layer with multi-head attention and FFN."""

    def __init__(self, config: Sam3LiteTextMobileCLIPConfig):
        super().__init__()
        norm_cls = Sam3LiteTextLayerNormFP32 if config.norm_type == "layer_norm_fp32" else nn.LayerNorm

        self.attn_norm = norm_cls(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Sam3LiteTextAttention(config)
        self.attn_dropout = nn.Dropout(config.hidden_dropout)

        self.ffn_norm = norm_cls(config.hidden_size, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()
        self.ffn_dropout = nn.Dropout(config.hidden_dropout)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Pre-norm MHA
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attention(hidden_states, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states

        # Pre-norm FFN
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.ffn_dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Sam3LiteTextMobileCLIPEncoder(PreTrainedModel):
    """
    MobileCLIP-S0 text encoder with RepMixer blocks.

    Architecture: [RepMixerBlock] + N x TransformerLayer + [RepMixerBlock]

    This replaces CLIPTextModelWithProjection in Sam3Model. It accepts `input_ids`
    and `attention_mask` and returns `BaseModelOutputWithPooling` with `last_hidden_state`.
    """

    config_class = Sam3LiteTextMobileCLIPConfig

    def _init_weights(self, module):
        """Initialize MobileCLIP-specific parameters."""
        super()._init_weights(module)
        if isinstance(module, Sam3LiteTextLearnablePositionalEmbedding):
            nn.init.trunc_normal_(module.pos_embed, mean=0, std=module.embedding_dim**-0.5)
        if isinstance(module, (Sam3LiteTextRepMixer, Sam3LiteTextRepMixerBlock)):
            nn.init.constant_(module.layer_scale, 1e-5)

    def __init__(self, config: Sam3LiteTextMobileCLIPConfig):
        super().__init__(config)
        self.config = config

        self.embedding_layer = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = config.hidden_size**-0.5

        self.positional_embedding = Sam3LiteTextLearnablePositionalEmbedding(
            num_embeddings=config.context_length,
            embedding_dim=config.hidden_size,
        )
        self.embedding_dropout = nn.Dropout(config.hidden_dropout)

        # MobileCLIP-S0 ("mct" variant): RepMixerBlock + N TransformerLayers + RepMixerBlock
        self.layers = nn.ModuleList()
        self.layers.append(Sam3LiteTextRepMixerBlock(config))
        for _ in range(config.num_hidden_layers):
            self.layers.append(Sam3LiteTextTransformerLayer(config))
        self.layers.append(Sam3LiteTextRepMixerBlock(config))

        norm_cls = Sam3LiteTextLayerNormFP32 if config.norm_type == "layer_norm_fp32" else nn.LayerNorm
        self.final_layer_norm = norm_cls(config.hidden_size, eps=config.layer_norm_eps)

        self.post_init()

    def resize_positional_embeddings(self, new_length: int):
        """Resize positional embeddings to a new context length (e.g., after loading checkpoint)."""
        pos_embed = self.positional_embedding.pos_embed
        current_length = pos_embed.shape[2]
        if new_length == current_length:
            return
        new_pos_embed = pos_embed[:, :, :new_length, :].clone()
        self.positional_embedding.pos_embed = nn.Parameter(new_pos_embed)
        self.positional_embedding.num_embeddings = new_length

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        # Embed tokens
        hidden_states = self.embedding_layer(input_ids) * self.embed_scale
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states + self.positional_embedding(seq_len).to(hidden_states.dtype)
        hidden_states = self.embedding_dropout(hidden_states)

        # Build key padding mask from attention_mask: True = padding (to mask out)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        # Forward through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, key_padding_mask=key_padding_mask)

        hidden_states = self.final_layer_norm(hidden_states)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=None,
        )


# =============================================================================
# Image Processor and Processor (inherit from SAM3)
# =============================================================================


class Sam3LiteTextImageProcessor(Sam3ImageProcessor):
    pass


class Sam3LiteTextProcessor(Sam3Processor):
    pass


# =============================================================================
# Model
# =============================================================================


class Sam3LiteTextPreTrainedModel(Sam3PreTrainedModel):
    config_class = Sam3LiteTextConfig

    def _init_weights(self, module):
        """Handle MobileCLIP-specific parameters, delegate the rest to parent."""
        super()._init_weights(module)
        if isinstance(module, Sam3LiteTextLearnablePositionalEmbedding):
            nn.init.trunc_normal_(module.pos_embed, mean=0, std=module.embedding_dim**-0.5)
        if isinstance(module, (Sam3LiteTextRepMixer, Sam3LiteTextRepMixerBlock)):
            nn.init.constant_(module.layer_scale, 1e-5)


class Sam3LiteTextViTModel(Sam3ViTModel):
    pass


class Sam3LiteTextModel(Sam3Model):
    config_class = Sam3LiteTextConfig

    def __init__(self, config: Sam3LiteTextConfig):
        # Function-level imports for classes the modular converter doesn't trace
        from ..sam3.modeling_sam3 import (
            Sam3DetrDecoder,
            Sam3DetrEncoder,
            Sam3DotProductScoring,
            Sam3GeometryEncoder,
            Sam3MaskDecoder,
            Sam3VisionModel,
        )

        # Skip Sam3Model.__init__ to replace the text encoder;
        # call the grandparent (Sam3PreTrainedModel -> PreTrainedModel)
        super(Sam3Model, self).__init__(config)

        # loading from a sam3_video config
        if hasattr(config, "detector_config") and config.detector_config is not None:
            detector_config = config.detector_config
            if isinstance(detector_config, dict):
                detector_config = Sam3LiteTextConfig(**detector_config)
            config = detector_config

        self.vision_encoder = Sam3VisionModel(config.vision_config)

        # MobileCLIP text encoder instead of CLIPTextModelWithProjection
        self.text_encoder = Sam3LiteTextMobileCLIPEncoder(config.text_config)
        self.vocab_size = config.text_config.vocab_size

        # Project text features from MobileCLIP hidden size (512) to DETR hidden size (256)
        self.text_projection = nn.Linear(config.text_config.hidden_size, config.detr_encoder_config.hidden_size)

        # Pass _attn_implementation to subconfigs
        config.geometry_encoder_config._attn_implementation = config._attn_implementation
        config.detr_encoder_config._attn_implementation = config._attn_implementation
        config.detr_decoder_config._attn_implementation = config._attn_implementation
        config.mask_decoder_config._attn_implementation = config._attn_implementation

        self.geometry_encoder = Sam3GeometryEncoder(config.geometry_encoder_config)
        self.detr_encoder = Sam3DetrEncoder(config.detr_encoder_config)
        self.detr_decoder = Sam3DetrDecoder(config.detr_decoder_config)
        self.mask_decoder = Sam3MaskDecoder(config.mask_decoder_config)

        self.dot_product_scoring = Sam3DotProductScoring(config)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Example:

        ```python
        >>> from transformers import Sam3LiteTextModel, Sam3LiteTextProcessor

        >>> model = Sam3LiteTextModel.from_pretrained("Simon7108528/EfficientSAM3")
        >>> processor = Sam3LiteTextProcessor.from_pretrained("Simon7108528/EfficientSAM3")

        >>> text_inputs = processor(text="cat", return_tensors="pt")
        >>> text_embeds = model.get_text_features(**text_inputs).pooler_output
        ```
        """
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = text_outputs.last_hidden_state
        text_outputs.pooler_output = self.text_projection(last_hidden_state)
        return text_outputs


__all__ = [
    "Sam3LiteTextConfig",
    "Sam3LiteTextMobileCLIPConfig",
    "Sam3LiteTextViTConfig",
    "Sam3LiteTextViTModel",
    "Sam3LiteTextMobileCLIPEncoder",
    "Sam3LiteTextImageProcessor",
    "Sam3LiteTextProcessor",
    "Sam3LiteTextPreTrainedModel",
    "Sam3LiteTextModel",
]
