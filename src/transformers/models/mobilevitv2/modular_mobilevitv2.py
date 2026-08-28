# Copyright 2023 Apple Inc. and The HuggingFace Inc. team. All rights reserved.
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
#
# Original license: https://github.com/apple/ml-cvnets/blob/main/LICENSE
"""PyTorch MobileViTV2 model."""

import torch
from torch import nn

from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple
from ..mobilevit.modeling_mobilevit import (
    MobileViTASPP,
    MobileViTASPPPooling,
    MobileViTConvLayer,
    MobileViTDeepLabV3,
    MobileViTEncoder,
    MobileViTForSemanticSegmentation,
    MobileViTInvertedResidual,
    MobileViTMobileNetLayer,
    MobileViTPreTrainedModel,
    make_divisible,
)
from .configuration_mobilevitv2 import MobileViTV2Config


logger = logging.get_logger(__name__)


def clip(value: float, min_val: float = float("-inf"), max_val: float = float("inf")) -> float:
    return max(min_val, min(max_val, value))


class MobileViTV2ConvLayer(MobileViTConvLayer):
    pass


class MobileViTV2InvertedResidual(MobileViTInvertedResidual):
    pass


class MobileViTV2MobileNetLayer(MobileViTMobileNetLayer):
    pass


class MobileViTV2LinearSelfAttention(nn.Module):
    """
    This layer applies a self-attention with linear complexity, as described in MobileViTV2 paper:
    https://huggingface.co/papers/2206.02680

    Args:
        config (`MobileVitv2Config`):
             Model configuration object
        embed_dim (`int`):
            `input_channels` from an expected input of size :math:`(batch_size, input_channels, height, width)`
    """

    def __init__(self, config: MobileViTV2Config, embed_dim: int) -> None:
        super().__init__()

        self.qkv_proj = MobileViTV2ConvLayer(
            config=config,
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=True,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
        )

        self.attn_dropout = nn.Dropout(p=config.attn_dropout)
        self.out_proj = MobileViTV2ConvLayer(
            config=config,
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=True,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
        )
        self.embed_dim = embed_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # (batch_size, embed_dim, num_pixels_in_patch, num_patches) --> (batch_size, 1+2*embed_dim, num_pixels_in_patch, num_patches)
        qkv = self.qkv_proj(hidden_states)

        # Project hidden_states into query, key and value
        # Query --> [batch_size, 1, num_pixels_in_patch, num_patches]
        # value, key --> [batch_size, embed_dim, num_pixels_in_patch, num_patches]
        query, key, value = torch.split(qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1)

        # apply softmax along num_patches dimension
        context_scores = torch.nn.functional.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [batch_size, embed_dim, num_pixels_in_patch, num_patches] x [batch_size, 1, num_pixels_in_patch, num_patches] -> [batch_size, embed_dim, num_pixels_in_patch, num_patches]
        context_vector = key * context_scores
        # [batch_size, embed_dim, num_pixels_in_patch, num_patches] --> [batch_size, embed_dim, num_pixels_in_patch, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [batch_size, embed_dim, num_pixels_in_patch, num_patches] * [batch_size, embed_dim, num_pixels_in_patch, 1] --> [batch_size, embed_dim, num_pixels_in_patch, num_patches]
        out = torch.nn.functional.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out


class MobileViTV2FFN(nn.Module):
    def __init__(
        self,
        config: MobileViTV2Config,
        embed_dim: int,
        ffn_latent_dim: int,
        ffn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = MobileViTV2ConvLayer(
            config=config,
            in_channels=embed_dim,
            out_channels=ffn_latent_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            use_normalization=False,
            use_activation=True,
        )
        self.dropout1 = nn.Dropout(ffn_dropout)

        self.conv2 = MobileViTV2ConvLayer(
            config=config,
            in_channels=ffn_latent_dim,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            use_normalization=False,
            use_activation=False,
        )
        self.dropout2 = nn.Dropout(ffn_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.dropout1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        return hidden_states


class MobileViTV2TransformerLayer(nn.Module):
    def __init__(
        self,
        config: MobileViTV2Config,
        embed_dim: int,
        ffn_latent_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layernorm_before = nn.GroupNorm(num_groups=1, num_channels=embed_dim, eps=config.layer_norm_eps)
        self.attention = MobileViTV2LinearSelfAttention(config, embed_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layernorm_after = nn.GroupNorm(num_groups=1, num_channels=embed_dim, eps=config.layer_norm_eps)
        self.ffn = MobileViTV2FFN(config, embed_dim, ffn_latent_dim, config.ffn_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        layernorm_1_out = self.layernorm_before(hidden_states)
        attention_output = self.attention(layernorm_1_out)
        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.ffn(layer_output)

        layer_output = layer_output + hidden_states
        return layer_output


class MobileViTV2Transformer(nn.Module):
    def __init__(self, config: MobileViTV2Config, n_layers: int, d_model: int) -> None:
        super().__init__()

        ffn_multiplier = config.ffn_multiplier

        ffn_dims = [ffn_multiplier * d_model] * n_layers

        # ensure that dims are multiple of 16
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        self.layer = nn.ModuleList()
        for block_idx in range(n_layers):
            transformer_layer = MobileViTV2TransformerLayer(
                config, embed_dim=d_model, ffn_latent_dim=ffn_dims[block_idx]
            )
            self.layer.append(transformer_layer)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class MobileViTV2Layer(GradientCheckpointingLayer):
    """
    MobileViTV2 layer: https://huggingface.co/papers/2206.02680
    """

    def __init__(
        self,
        config: MobileViTV2Config,
        in_channels: int,
        out_channels: int,
        attn_unit_dim: int,
        n_attn_blocks: int = 2,
        dilation: int = 1,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.patch_width = config.patch_size
        self.patch_height = config.patch_size

        cnn_out_dim = attn_unit_dim

        if stride == 2:
            self.downsampling_layer = MobileViTV2InvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if dilation == 1 else 1,
                dilation=dilation // 2 if dilation > 1 else 1,
            )
            in_channels = out_channels
        else:
            self.downsampling_layer = None

        # Local representations
        self.conv_kxk = MobileViTV2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=config.conv_kernel_size,
            groups=in_channels,
        )
        self.conv_1x1 = MobileViTV2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
        )

        # Global representations
        self.transformer = MobileViTV2Transformer(config, d_model=attn_unit_dim, n_layers=n_attn_blocks)

        self.layernorm = nn.GroupNorm(num_groups=1, num_channels=attn_unit_dim, eps=config.layer_norm_eps)

        # Fusion
        self.conv_projection = MobileViTV2ConvLayer(
            config,
            in_channels=cnn_out_dim,
            out_channels=in_channels,
            kernel_size=1,
            use_normalization=True,
            use_activation=False,
        )

    def unfolding(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        batch_size, in_channels, img_height, img_width = feature_map.shape
        patches = nn.functional.unfold(
            feature_map,
            kernel_size=(self.patch_height, self.patch_width),
            stride=(self.patch_height, self.patch_width),
        )
        patches = patches.reshape(batch_size, in_channels, self.patch_height * self.patch_width, -1)

        return patches, (img_height, img_width)

    def folding(self, patches: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        feature_map = nn.functional.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_height, self.patch_width),
            stride=(self.patch_height, self.patch_width),
        )

        return feature_map

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # reduce spatial dimensions if needed
        if self.downsampling_layer:
            features = self.downsampling_layer(features)

        # local representation
        features = self.conv_kxk(features)
        features = self.conv_1x1(features)

        # convert feature map to patches
        patches, output_size = self.unfolding(features)

        # learn global representations
        patches = self.transformer(patches)
        patches = self.layernorm(patches)

        # convert patches back to feature maps
        # [batch_size, patch_height, patch_width, input_dim] --> [batch_size, input_dim, patch_height, patch_width]
        features = self.folding(patches, output_size)

        features = self.conv_projection(features)
        return features


class MobileViTV2Encoder(MobileViTEncoder):
    def __init__(self, config: MobileViTV2Config) -> None:
        nn.Module.__init__(self)
        self.config = config

        self.layer = nn.ModuleList()
        self.gradient_checkpointing = False

        # segmentation architectures like DeepLab and PSPNet modify the strides
        # of the classification backbones
        dilate_layer_4 = dilate_layer_5 = False
        if config.output_stride == 8:
            dilate_layer_4 = True
            dilate_layer_5 = True
        elif config.output_stride == 16:
            dilate_layer_5 = True

        dilation = 1

        layer_0_dim = make_divisible(
            clip(value=32 * config.width_multiplier, min_val=16, max_val=64), divisor=8, min_value=16
        )

        layer_1_dim = make_divisible(64 * config.width_multiplier, divisor=16)
        layer_2_dim = make_divisible(128 * config.width_multiplier, divisor=8)
        layer_3_dim = make_divisible(256 * config.width_multiplier, divisor=8)
        layer_4_dim = make_divisible(384 * config.width_multiplier, divisor=8)
        layer_5_dim = make_divisible(512 * config.width_multiplier, divisor=8)

        layer_1 = MobileViTV2MobileNetLayer(
            config,
            in_channels=layer_0_dim,
            out_channels=layer_1_dim,
            stride=1,
            num_stages=1,
        )
        self.layer.append(layer_1)

        layer_2 = MobileViTV2MobileNetLayer(
            config,
            in_channels=layer_1_dim,
            out_channels=layer_2_dim,
            stride=2,
            num_stages=2,
        )
        self.layer.append(layer_2)

        layer_3 = MobileViTV2Layer(
            config,
            in_channels=layer_2_dim,
            out_channels=layer_3_dim,
            attn_unit_dim=make_divisible(config.base_attn_unit_dims[0] * config.width_multiplier, divisor=8),
            n_attn_blocks=config.n_attn_blocks[0],
        )
        self.layer.append(layer_3)

        if dilate_layer_4:
            dilation *= 2

        layer_4 = MobileViTV2Layer(
            config,
            in_channels=layer_3_dim,
            out_channels=layer_4_dim,
            attn_unit_dim=make_divisible(config.base_attn_unit_dims[1] * config.width_multiplier, divisor=8),
            n_attn_blocks=config.n_attn_blocks[1],
            dilation=dilation,
        )
        self.layer.append(layer_4)

        if dilate_layer_5:
            dilation *= 2

        layer_5 = MobileViTV2Layer(
            config,
            in_channels=layer_4_dim,
            out_channels=layer_5_dim,
            attn_unit_dim=make_divisible(config.base_attn_unit_dims[2] * config.width_multiplier, divisor=8),
            n_attn_blocks=config.n_attn_blocks[2],
            dilation=dilation,
        )
        self.layer.append(layer_5)


@auto_docstring
class MobileViTV2PreTrainedModel(MobileViTPreTrainedModel):
    config: MobileViTV2Config
    base_model_prefix = "mobilevitv2"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    supports_gradient_checkpointing = True
    _no_split_modules = ["MobileViTV2Layer"]


@auto_docstring
class MobileViTV2Model(MobileViTV2PreTrainedModel):
    def __init__(self, config: MobileViTV2Config, expand_output: bool = True):
        r"""
        expand_output (`bool`, *optional*, defaults to `True`):
            Whether to expand the output of the model. If `True`, the model will output pooled features in addition to
            hidden states. If `False`, only the hidden states will be returned.
        """
        super().__init__(config)
        self.config = config
        self.expand_output = expand_output

        layer_0_dim = make_divisible(
            clip(value=32 * config.width_multiplier, min_val=16, max_val=64), divisor=8, min_value=16
        )

        self.conv_stem = MobileViTV2ConvLayer(
            config,
            in_channels=config.num_channels,
            out_channels=layer_0_dim,
            kernel_size=3,
            stride=2,
            use_normalization=True,
            use_activation=True,
        )
        self.encoder = MobileViTV2Encoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.conv_stem(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state

        if self.expand_output:
            # global average pooling: (batch_size, channels, height, width) -> (batch_size, channels)
            pooled_output = torch.mean(last_hidden_state, dim=[-2, -1], keepdim=False)
        else:
            pooled_output = None

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@auto_docstring(
    custom_intro="""
    MobileViTV2 model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """
)
class MobileViTV2ForImageClassification(MobileViTV2PreTrainedModel):
    accepts_loss_kwargs = False

    def __init__(self, config: MobileViTV2Config) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.mobilevitv2 = MobileViTV2Model(config)

        # layer 5 output dimension
        out_channels = make_divisible(512 * config.width_multiplier, divisor=8)
        self.classifier = (
            nn.Linear(in_features=out_channels, out_features=config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutputWithNoAttention:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.mobilevitv2(pixel_values, **kwargs)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(labels=labels, pooled_logits=logits, config=self.config)

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class MobileViTV2ASPPPooling(MobileViTASPPPooling):
    pass


class MobileViTV2ASPP(MobileViTASPP):
    """
    ASPP module defined in DeepLab papers: https://huggingface.co/papers/1606.00915, https://huggingface.co/papers/1706.05587
    """

    def __init__(self, config: MobileViTV2Config) -> None:
        nn.Module.__init__(self)

        encoder_out_channels = make_divisible(512 * config.width_multiplier, divisor=8)
        out_channels = config.aspp_out_channels

        if len(config.atrous_rates) != 3:
            raise ValueError("Expected 3 values for atrous_rates")

        self.convs = nn.ModuleList()

        in_projection = MobileViTV2ConvLayer(
            config,
            in_channels=encoder_out_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation="relu",
        )
        self.convs.append(in_projection)

        self.convs.extend(
            [
                MobileViTV2ConvLayer(
                    config,
                    in_channels=encoder_out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    dilation=rate,
                    use_activation="relu",
                )
                for rate in config.atrous_rates
            ]
        )

        pool_layer = MobileViTV2ASPPPooling(config, encoder_out_channels, out_channels)
        self.convs.append(pool_layer)

        self.project = MobileViTV2ConvLayer(
            config, in_channels=5 * out_channels, out_channels=out_channels, kernel_size=1, use_activation="relu"
        )

        self.dropout = nn.Dropout(p=config.aspp_dropout_prob)


class MobileViTV2DeepLabV3(MobileViTDeepLabV3):
    pass


@auto_docstring(
    custom_intro="""
    MobileViTV2 model with a semantic segmentation head on top, e.g. for Pascal VOC.
    """
)
class MobileViTV2ForSemanticSegmentation(MobileViTForSemanticSegmentation):
    def __init__(self, config: MobileViTV2Config) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.mobilevitv2 = MobileViTV2Model(config, expand_output=False)
        del self.mobilevit
        self.segmentation_head = MobileViTV2DeepLabV3(config)

        # Initialize weights and apply final processing
        self.post_init()


__all__ = [
    "MobileViTV2ForImageClassification",
    "MobileViTV2ForSemanticSegmentation",
    "MobileViTV2Model",
    "MobileViTV2PreTrainedModel",
]
