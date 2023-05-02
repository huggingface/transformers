# coding=utf-8
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
""" PyTorch MobileViTv2 model."""


import math
from typing import Dict, Optional, Set, Tuple, Union, Sequence

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_mobilevitv2 import MobileViTv2Config


logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "MobileViTv2Config"

# Base docstring
_CHECKPOINT_FOR_DOC = "apple/mobilevitv2-1.0-imagenet1k-256"
_EXPECTED_OUTPUT_SHAPE = [1, 512, 8, 8]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "apple/mobilevitv2-1.0-imagenet1k-256"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


MOBILEVITV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "apple/mobilevitv2-1.0-imagenet1k-256"
    # See all MobileViTv2 models at https://huggingface.co/models?filter=mobilevitv2
]


def make_divisible(value: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    Ensure that all layers have a channel count that is divisible by `divisor`. This function is taken from the
    original TensorFlow repo. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

def bound_fn(
    min_val: Union[float, int], max_val: Union[float, int], value: Union[float, int]
) -> Union[float, int]:
    return max(min_val, min(max_val, value))

class MobileViTv2LayerNorm2D(nn.GroupNorm):
    """
    Applies `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ over a 4D input tensor

    Args:
        num_features (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine (bool): If ``True``, use learnable affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`H` is the input height, and :math:`W` is the input width
        - Output: same shape as the input
    """

    def __init__(
        self,
        num_features: int,
        eps: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            num_channels=num_features, eps=eps, affine=elementwise_affine, num_groups=1
        )
        self.num_channels = num_features

    def __repr__(self):
        return "{}(num_channels={}, eps={}, affine={})".format(
            self.__class__.__name__, self.num_channels, self.eps, self.affine
        )

# Copied from transformers.models.mobilevit.modeling_mobilevit.MobileViTConvLayer with MobileViT->MobileViTv2
class MobileViTv2ConvLayer(nn.Module):
    def __init__(
        self,
        config: MobileViTv2Config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        dilation: int = 1,
        use_normalization: bool = True,
        use_activation: Union[bool, str] = True,
    ) -> None:
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation

        if in_channels % groups != 0:
            raise ValueError(f"Input channels ({in_channels}) are not divisible by {groups} groups.")
        if out_channels % groups != 0:
            raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")

        self.convolution = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
        )

        if use_normalization:
            self.normalization = nn.BatchNorm2d(
                num_features=out_channels,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )
        else:
            self.normalization = None

        if use_activation:
            if isinstance(use_activation, str):
                self.activation = ACT2FN[use_activation]
            elif isinstance(config.hidden_act, str):
                self.activation = ACT2FN[config.hidden_act]
            else:
                self.activation = config.hidden_act
        else:
            self.activation = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.convolution(features)
        if self.normalization is not None:
            features = self.normalization(features)
        if self.activation is not None:
            features = self.activation(features)
        return features


# Copied from transformers.models.mobilevit.modeling_mobilevit.MobileViTInvertedResidual with MobileViT->MobileViTv2
class MobileViTv2InvertedResidual(nn.Module):
    """
    Inverted residual block (MobileNetv2): https://arxiv.org/abs/1801.04381
    """

    def __init__(
        self, config: MobileViTv2Config, in_channels: int, out_channels: int, stride: int, dilation: int = 1
    ) -> None:
        super().__init__()
        expanded_channels = make_divisible(int(round(in_channels * config.expand_ratio)), 8)

        if stride not in [1, 2]:
            raise ValueError(f"Invalid stride {stride}.")

        self.use_residual = (stride == 1) and (in_channels == out_channels)

        self.expand_1x1 = MobileViTv2ConvLayer(
            config, in_channels=in_channels, out_channels=expanded_channels, kernel_size=1
        )

        self.conv_3x3 = MobileViTv2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=stride,
            groups=expanded_channels,
            dilation=dilation,
        )

        self.reduce_1x1 = MobileViTv2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features

        features = self.expand_1x1(features)
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)

        return residual + features if self.use_residual else features


# Copied from transformers.models.mobilevit.modeling_mobilevit.MobileViTMobileNetLayer with MobileViT->MobileViTv2
class MobileViTv2MobileNetLayer(nn.Module):
    def __init__(
        self, config: MobileViTv2Config, in_channels: int, out_channels: int, stride: int = 1, num_stages: int = 1
    ) -> None:
        super().__init__()

        self.layer = nn.ModuleList()
        for i in range(num_stages):
            layer = MobileViTv2InvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if i == 0 else 1,
            )
            self.layer.append(layer)
            in_channels = out_channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            features = layer_module(features)
        return features


class MobileViTv2LinearSelfAttention(nn.Module):
    '''LinearSelfAttention'''
    """
    This layer applies a self-attention with linear complexity, as described in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper.
    This layer can be used for self- as well as cross-attention.

    Args:
        config: MobileVitv2Config 
        embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        attn_dropout (Optional[float]): Dropout value for context scores. Default: 0.0
        bias (Optional[bool]): Use bias in learnable layers. Default: True

    Shape:
        - Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input

    .. note::
        For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer.
    """

    def __init__(
        self,
        config:MobileViTv2Config,
        embed_dim: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        self.qkv_proj = MobileViTv2ConvLayer(
            config=config,
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=bias,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = MobileViTv2ConvLayer(
            config=config,
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
        )
        self.embed_dim = embed_dim

    def __repr__(self):
        return "{}(embed_dim={}, attn_dropout={})".format(
            self.__class__.__name__, self.embed_dim, self.attn_dropout.p
        )

    def _forward_self_attn(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        # apply softmax along N dimension
        context_scores = torch.nn.functional.softmax(query, dim=-1)
        # Uncomment below line to visualize context scores
        # self.visualize_context_scores(context_scores=context_scores)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = torch.nn.functional.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def _forward_cross_attn(
        self, x: torch.Tensor, x_prev: Optional[torch.Tensor] = None, *args, **kwargs
    ) -> torch.Tensor:
        # x --> [B, C, P, N]
        # x_prev = [B, C, P, M]

        batch_size, in_dim, kv_patch_area, kv_num_patches = x.shape

        q_patch_area, q_num_patches = x.shape[-2:]

        assert (
            kv_patch_area == q_patch_area
        ), "The number of pixels in a patch for query and key_value should be the same"

        # compute query, key, and value
        # [B, C, P, M] --> [B, 1 + d, P, M]
        qk = torch.nn.functional.conv2d(
            x_prev,
            weight=self.qkv_proj.block.conv.weight[: self.embed_dim + 1, ...],
            bias=self.qkv_proj.block.conv.bias[: self.embed_dim + 1, ...],
        )
        # [B, 1 + d, P, M] --> [B, 1, P, M], [B, d, P, M]
        query, key = torch.split(qk, split_size_or_sections=[1, self.embed_dim], dim=1)
        # [B, C, P, N] --> [B, d, P, N]
        value = torch.nn.functional.conv2d(
            x,
            weight=self.qkv_proj.block.conv.weight[self.embed_dim + 1 :, ...],
            bias=self.qkv_proj.block.conv.bias[self.embed_dim + 1 :, ...],
        )

        # apply softmax along M dimension
        context_scores = torch.nn.functional.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        # compute context vector
        # [B, d, P, M] * [B, 1, P, M] -> [B, d, P, M]
        context_vector = key * context_scores
        # [B, d, P, M] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = torch.nn.functional.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def forward(
        self, x: torch.Tensor, x_prev: Optional[torch.Tensor] = None, *args, **kwargs
    ) -> torch.Tensor:
        if x_prev is None:
            return self._forward_self_attn(x, *args, **kwargs)
        else:
            return self._forward_cross_attn(x, x_prev=x_prev, *args, **kwargs)



class MobileViTv2FFN(nn.Module):
    def __init__(self, 
                    config: MobileViTv2Config, 
                    embed_dim: int,
                    ffn_latent_dim: int,
                    ffn_dropout: float = 0.0,
                    dropout: float = 0.0
                 ) -> None:
        super().__init__()
        self.conv1 = MobileViTv2ConvLayer(
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
        
        self.conv2 = MobileViTv2ConvLayer(
                config=config,
                in_channels=ffn_latent_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                use_normalization=False,
                use_activation=False,
            )
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.dropout1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        return hidden_states


class MobileViTv2TransformerLayer(nn.Module):
    '''LinearAttnFFN'''
    def __init__(self, config: MobileViTv2Config, 
                embed_dim: int,
                ffn_latent_dim: int,
                ffn_dropout: float = 0.0,
                dropout: float = 0.0,
                attn_dropout: float = 0.0,
                ) -> None:
        super().__init__()
        # pre_norm_attn
        self.layernorm_before = MobileViTv2LayerNorm2D(embed_dim, eps=config.layer_norm_eps)
        self.attention = MobileViTv2LinearSelfAttention(config, embed_dim, attn_dropout=attn_dropout, bias=True)
        self.dropout1 = nn.Dropout(p=dropout)
        # pre_norm_ffn
        self.layernorm_after = MobileViTv2LayerNorm2D(embed_dim, eps=config.layer_norm_eps)
        self.ffn = MobileViTv2FFN(config, embed_dim, ffn_latent_dim, ffn_dropout, dropout)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        layernorm_1_out = self.layernorm_before(hidden_states)
        attention_output = self.attention(layernorm_1_out)
        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.ffn(layer_output)
        
        layer_output = layer_output + hidden_states
        return layer_output



class MobileViTv2Transformer(nn.Module):
    def __init__(self, 
            config: MobileViTv2Config, 
            n_layers: int,
            d_model: int,
            ffn_multiplier: Union[Sequence, int, float],
            attn_dropout: float,
            dropout: float,
            ffn_dropout: float,            
            ) -> None:
        super().__init__()
        
        if isinstance(ffn_multiplier, Sequence) and len(ffn_multiplier) == 2:
            ffn_dims = (
                torch.linspace(ffn_multiplier[0], ffn_multiplier[1], n_layers, dtype=float) * d_model
            )
        elif isinstance(ffn_multiplier, Sequence) and len(ffn_multiplier) == 1:
            ffn_dims = [ffn_multiplier[0] * d_model] * n_layers
        elif isinstance(ffn_multiplier, (int, float)):
            ffn_dims = [ffn_multiplier * d_model] * n_layers
        else:
            raise NotImplementedError

        # ensure that dims are multiple of 16
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        self.layer = nn.ModuleList()
        for block_idx in range(n_layers):
            transformer_layer = MobileViTv2TransformerLayer(
                config,
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
            )
            self.layer.append(transformer_layer)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class MobileViTv2Layer(nn.Module):
    """
    MobileViTv2 block: https://arxiv.org/abs/2206.02680 
    """

    def __init__(
        self,
        config: MobileViTv2Config,
        in_channels: int,
        out_channels: int,
        attn_unit_dim: int,
        stride: int,
        n_attn_blocks: Optional[int] = 2,
        ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        dilation: int = 1,
        patch_size: Optional[int] = 2,
        
    ) -> None:
        super().__init__()
        self.patch_width = config.patch_size
        self.patch_height = config.patch_size
        
        cnn_out_dim = attn_unit_dim

        if stride == 2:
            self.downsampling_layer = MobileViTv2InvertedResidual(
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
        self.conv_kxk = MobileViTv2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=config.conv_kernel_size,
            groups=in_channels
        )
        self.conv_1x1 = MobileViTv2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
        )

        # Global representations
        self.transformer = MobileViTv2Transformer(
            config,
            d_model=attn_unit_dim,
            ffn_multiplier = ffn_multiplier,
            n_layers=n_attn_blocks,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
        )
        
        self.layernorm = MobileViTv2LayerNorm2D(attn_unit_dim, eps=config.layer_norm_eps)

        # Fusion
        self.conv_projection = MobileViTv2ConvLayer(
            config, in_channels=cnn_out_dim, out_channels=in_channels, kernel_size=1, 
            use_normalization=True, use_activation=False
        )
        
        self.patch_h = patch_size
        self.patch_w = patch_size

    def unfolding_pytorch(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:

        batch_size, in_channels, img_h, img_w = feature_map.shape

        # [B, C, H, W] --> [B, C, P, N]
        patches = nn.functional.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )

        return patches, (img_h, img_w)

    def folding_pytorch(self, patches: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape

        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        feature_map = nn.functional.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
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
        patches, output_size = self.unfolding_pytorch(features, )
        
        # learn global representations
        patches = self.transformer(patches)
        patches = self.layernorm(patches)
        
        # convert patches back to feature maps 
        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        features = self.folding_pytorch(patches, output_size)

        features = self.conv_projection(features)
        return features


class MobileViTv2Encoder(nn.Module):
    def __init__(self, config: MobileViTv2Config) -> None:
        super().__init__()
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

        layer_0_dim = int(make_divisible(bound_fn(min_val=16, max_val=64, value=32 * config.width_multiplier), divisor=8, min_value=16))
        layer_1_dim = int(make_divisible(64 * config.width_multiplier, divisor=16))
        layer_2_dim = int(make_divisible(128 * config.width_multiplier, divisor=8))
        layer_3_dim = int(make_divisible(256 * config.width_multiplier, divisor=8))
        layer_4_dim = int(make_divisible(384 * config.width_multiplier, divisor=8))
        layer_5_dim = int(make_divisible(512 * config.width_multiplier, divisor=8))
        ffn_multiplier = 2
           
        # 1: layer_1
        layer_1 = MobileViTv2MobileNetLayer(
            config,
            in_channels=layer_0_dim,
            out_channels=layer_1_dim,
            stride=1,
            num_stages=1,
        )
        self.layer.append(layer_1)

        # 2: layer_2
        layer_2 = MobileViTv2MobileNetLayer(
            config,
            in_channels=layer_1_dim,
            out_channels=layer_2_dim,
            stride=2,
            num_stages=2,
        )
        self.layer.append(layer_2)

        # 3: layer_3
        layer_3 = MobileViTv2Layer(
            config,
            in_channels=layer_2_dim,
            out_channels=layer_3_dim,
            attn_unit_dim=int(make_divisible(128 * config.width_multiplier, divisor=8)),
            stride=2,
            ffn_multiplier=ffn_multiplier,
            n_attn_blocks=2,
        )
        self.layer.append(layer_3)

        if dilate_layer_4:
            dilation *= 2

        # 4: layer_4
        layer_4 = MobileViTv2Layer(
            config,
            in_channels=layer_3_dim,
            out_channels=layer_4_dim,
            attn_unit_dim=int(make_divisible(192 * config.width_multiplier, divisor=8)),
            stride=2,
            ffn_multiplier=ffn_multiplier,
            n_attn_blocks=4,
            dilation=dilation,
            patch_size=config.patch_size
        )
        self.layer.append(layer_4)

        if dilate_layer_5:
            dilation *= 2

        # 5: layer_5 
        layer_5 = MobileViTv2Layer(
            config,
            in_channels=layer_4_dim,
            out_channels=layer_5_dim,
            attn_unit_dim=int(make_divisible(256 * config.width_multiplier, divisor=8)),
            stride=2,
            ffn_multiplier=ffn_multiplier,
            n_attn_blocks=3,
            dilation=dilation,
            patch_size=config.patch_size
        )
        self.layer.append(layer_5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                )
            else:
                hidden_states = layer_module(hidden_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


# Copied from transformers.models.mobilevit.modeling_mobilevit.MobileViTPreTrainedModel with MobileViT->MobileViTv2,mobilevit->mobilevitv2
class MobileViTv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MobileViTv2Config
    base_model_prefix = "mobilevitv2"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MobileViTv2Encoder):
            module.gradient_checkpointing = value


MOBILEVITV2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MobileViTv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MOBILEVITV2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`MobileViTv2ImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MobileViTv2 model outputting raw hidden-states without any specific head on top.",
    MOBILEVITV2_START_DOCSTRING,
)
class MobileViTv2Model(MobileViTv2PreTrainedModel):
    def __init__(self, config: MobileViTv2Config, expand_output: bool = True):
        super().__init__(config)
        self.config = config
        self.expand_output = expand_output

        layer_0_dim = int(make_divisible(bound_fn(min_val=16, max_val=64, value=32 * config.width_multiplier), divisor=8, min_value=16))
        
        # 0: conv_1
        self.conv_stem = MobileViTv2ConvLayer(
            config,
            in_channels=config.num_channels,
            out_channels=layer_0_dim,
            kernel_size=3,
            stride=2,
            use_normalization=True,
            use_activation=True
        )

        # 1: layer_1, 2: layer_2, 3: layer_3, 4: layer_4, 5: layer_5 
        self.encoder = MobileViTv2Encoder(config)

        # 6: conv_1x1_exp
        self.conv_1x1_exp = nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base class PreTrainedModel
        """
        for layer_index, heads in heads_to_prune.items():
            mobilevitv2_layer = self.encoder.layer[layer_index]
            if isinstance(mobilevitv2_layer, MobileViTv2Layer):
                for transformer_layer in mobilevitv2_layer.transformer.layer:
                    transformer_layer.attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MOBILEVITV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.conv_stem(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.expand_output:
            last_hidden_state = self.conv_1x1_exp(encoder_outputs[0])

            # global average pooling: (batch_size, channels, height, width) -> (batch_size, channels)
            pooled_output = torch.mean(last_hidden_state, dim=[-2, -1], keepdim=False)
        else:
            last_hidden_state = encoder_outputs[0]
            pooled_output = None

        if not return_dict:
            output = (last_hidden_state, pooled_output) if pooled_output is not None else (last_hidden_state,)
            return output + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    """
    MobileViTv2 model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    MOBILEVITV2_START_DOCSTRING,
)
# Copied from transformers.models.mobilevit.modeling_mobilevit.MobileViTForImageClassification with MOBILEVIT->MOBILEVITV2,MobileViT->MobileViTv2,mobilevit->mobilevitv2
class MobileViTv2ForImageClassification(MobileViTv2PreTrainedModel):
    def __init__(self, config: MobileViTv2Config) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.mobilevitv2 = MobileViTv2Model(config)

        out_channels = int(make_divisible(512 * config.width_multiplier, divisor=8)) # layer 5 output dimension
        # Classifier head
        self.classifier = (
            nn.Linear(in_features = out_channels,
                      out_features = config.num_labels) 
            if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MOBILEVITV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilevitv2(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


# Copied from transformers.models.mobilevit.modeling_mobilevit.MobileViTASPPPooling with MobileViT->MobileViTv2
class MobileViTv2ASPPPooling(nn.Module):
    def __init__(self, config: MobileViTv2Config, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.conv_1x1 = MobileViTv2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_normalization=True,
            use_activation="relu",
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spatial_size = features.shape[-2:]
        features = self.global_pool(features)
        features = self.conv_1x1(features)
        features = nn.functional.interpolate(features, size=spatial_size, mode="bilinear", align_corners=False)
        return features


class MobileViTv2ASPP(nn.Module):
    """
    ASPP module defined in DeepLab papers: https://arxiv.org/abs/1606.00915, https://arxiv.org/abs/1706.05587
    """

    def __init__(self, config: MobileViTv2Config) -> None:
        super().__init__()

        encoder_out_channels = int(make_divisible(512 * config.width_multiplier, divisor=8)) # layer 5 output dimension
        in_channels = encoder_out_channels
        out_channels = config.aspp_out_channels

        if len(config.atrous_rates) != 3:
            raise ValueError("Expected 3 values for atrous_rates")

        self.convs = nn.ModuleList()

        in_projection = MobileViTv2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation="relu",
        )
        self.convs.append(in_projection)

        self.convs.extend(
            [
                MobileViTv2ConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    dilation=rate,
                    use_activation="relu",
                )
                for rate in config.atrous_rates
            ]
        )

        pool_layer = MobileViTv2ASPPPooling(config, in_channels, out_channels)
        self.convs.append(pool_layer)

        self.project = MobileViTv2ConvLayer(
            config, in_channels=5 * out_channels, out_channels=out_channels, kernel_size=1, use_activation="relu"
        )

        self.dropout = nn.Dropout(p=config.aspp_dropout_prob)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pyramid = []
        for conv in self.convs:
            pyramid.append(conv(features))
        pyramid = torch.cat(pyramid, dim=1)

        pooled_features = self.project(pyramid)
        pooled_features = self.dropout(pooled_features)
        return pooled_features


# Copied from transformers.models.mobilevit.modeling_mobilevit.MobileViTDeepLabV3 with MobileViT->MobileViTv2
class MobileViTv2DeepLabV3(nn.Module):
    """
    DeepLabv3 architecture: https://arxiv.org/abs/1706.05587
    """

    def __init__(self, config: MobileViTv2Config) -> None:
        super().__init__()
        self.aspp = MobileViTv2ASPP(config)

        self.dropout = nn.Dropout2d(config.classifier_dropout_prob)

        self.classifier = MobileViTv2ConvLayer(
            config,
            in_channels=config.aspp_out_channels,
            out_channels=config.num_labels,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        features = self.aspp(hidden_states[-1])
        features = self.dropout(features)
        features = self.classifier(features)
        return features


@add_start_docstrings(
    """
    MobileViTv2 model with a semantic segmentation head on top, e.g. for Pascal VOC.
    """,
    MOBILEVITV2_START_DOCSTRING,
)
# Copied from transformers.models.mobilevit.modeling_mobilevit.MobileViTForSemanticSegmentation with MOBILEVIT->MOBILEVITV2,MobileViT->MobileViTv2,mobilevit->mobilevitv2
class MobileViTv2ForSemanticSegmentation(MobileViTv2PreTrainedModel):
    def __init__(self, config: MobileViTv2Config) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.mobilevitv2 = MobileViTv2Model(config, expand_output=False)
        self.segmentation_head = MobileViTv2DeepLabV3(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MOBILEVITV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, MobileViTv2ForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("apple/deeplabv3-mobilevitv2-small")
        >>> model = MobileViTv2ForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevitv2-small")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # logits are of shape (batch_size, num_labels, height, width)
        >>> logits = outputs.logits
        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilevitv2(
            pixel_values,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.segmentation_head(encoder_hidden_states)

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                # upsample logits to the images' original size
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )
