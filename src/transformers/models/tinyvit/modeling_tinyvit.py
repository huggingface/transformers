# coding=utf-8
# Copyright 2023 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch TinyVit Transformer model."""

import collections
import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from ...utils.backbone_utils import BackboneMixin
from .configuration_tinyvit import TinyVitConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "TinyVitConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "microsoft/tinyvit-21m-1k"
_EXPECTED_OUTPUT_SHAPE = [1, 49, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "microsoft/tinyvit-21m-1k"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


TINYVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/tinyvit-21m-1k",
    # See all TinyViT models at https://huggingface.co/models?filter=tinyvit
]


@dataclass
class TinyVitEncoderOutput(ModelOutput):
    """
    TinyViT encoder's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class TinyVitConv2dBatchNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.convolution = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False
        )
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)

    @torch.no_grad()
    def fuse(self):
        convolution, batch_norm = self.convolution, self.batch_norm
        weight = batch_norm.weight / (batch_norm.running_var + batch_norm.eps) ** 0.5
        weight = convolution.weight * weight[:, None, None, None]
        bias = (
            batch_norm.bias
            - batch_norm.running_mean * batch_norm.weight / (batch_norm.running_var + batch_norm.eps) ** 0.5
        )
        module = torch.nn.Conv2d(
            weight.size(1) * self.convolution.groups,
            weight.size(0),
            weight.shape[2:],
            stride=self.cconvolution.stride,
            padding=self.convolution.padding,
            dilation=self.convolution.dilation,
            groups=self.convolution.groups,
        )
        module.weight.data.copy_(weight)
        module.bias.data.copy_(bias)
        return module

    def forward(self, hidden_state):
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.batch_norm(hidden_state)

        return hidden_state


class TinyVitEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_size = config.image_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        self.patches_resolution = (image_size[0] // 4, image_size[1] // 4)

        hidden_size = config.hidden_sizes[0]
        self.seq = nn.Sequential(
            TinyVitConv2dBatchNorm(config.num_channels, hidden_size // 2, 3, 2, 1),
            nn.GELU(),
            TinyVitConv2dBatchNorm(hidden_size // 2, hidden_size, 3, 2, 1),
        )

    def forward(self, pixel_values):
        embeddings = self.seq(pixel_values)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        return embeddings, output_dimensions


class TinyVitMBConv(nn.Module):
    def __init__(self, config, in_channels, out_channels, drop_path):
        super().__init__()
        expand_ratio = config.mbconv_expand_ratio
        self.hidden_chans = int(in_channels * expand_ratio)

        self.conv1 = TinyVitConv2dBatchNorm(in_channels, self.hidden_chans, kernel_size=1)
        self.activation1 = ACT2FN[config.hidden_act]

        self.conv2 = TinyVitConv2dBatchNorm(
            self.hidden_chans, self.hidden_chans, kernel_size=3, stride=1, padding=1, groups=self.hidden_chans
        )
        self.activation2 = ACT2FN[config.hidden_act]

        self.conv3 = TinyVitConv2dBatchNorm(self.hidden_chans, out_channels, kernel_size=1)
        self.activation3 = ACT2FN[config.hidden_act]

        self.drop_path = TinyVitDropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, hidden_state):
        shortcut = hidden_state

        hidden_state = self.conv1(hidden_state)
        hidden_state = self.activation1(hidden_state)

        hidden_state = self.conv2(hidden_state)
        hidden_state = self.activation2(hidden_state)

        hidden_state = self.conv3(hidden_state)

        hidden_state = self.drop_path(hidden_state)

        hidden_state += shortcut
        hidden_state = self.activation3(hidden_state)

        return hidden_state


class TinyVitPatchMerging(nn.Module):
    def __init__(self, config, input_resolution, dim, out_dim):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.activation = ACT2FN[config.hidden_act]
        self.conv1 = TinyVitConv2dBatchNorm(dim, out_dim, 1, 1, 0)
        self.conv2 = TinyVitConv2dBatchNorm(out_dim, out_dim, 3, 2, 1, groups=out_dim)
        self.conv3 = TinyVitConv2dBatchNorm(out_dim, out_dim, 1, 1, 0)

    def forward(self, hidden_state):
        if hidden_state.ndim == 3:
            height, width = self.input_resolution
            batch_size = len(hidden_state)
            # reshape to (batch_size, num_channels, height, width)
            hidden_state = hidden_state.view(batch_size, height, width, -1).permute(0, 3, 1, 2)

        hidden_state = self.conv1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.conv2(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.conv3(hidden_state)

        hidden_state = hidden_state.flatten(2).transpose(1, 2)
        return hidden_state


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->TinyVit
class TinyVitDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class TinyVitConvStage(nn.Module):
    def __init__(self, config, dim, input_resolution, depth, drop_path=0.0, downsample=None, out_dim=None):
        super().__init__()

        self.input_resolution = input_resolution
        self.depth = depth

        # build layers
        self.layers = nn.ModuleList(
            [
                TinyVitMBConv(
                    config,
                    dim,
                    dim,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(config, input_resolution, dim=dim, out_dim=out_dim)
        else:
            self.downsample = None

        self.gradient_checkpointing = False

    def forward(self, hidden_state, input_dimensions, output_attentions=False):
        height, width = input_dimensions
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_state = self._gradient_checkpointing_func(layer.__call__, hidden_state)
            else:
                hidden_state = layer(hidden_state)

        hidden_state_before_downsampling = hidden_state
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_state = self.downsample(hidden_state)
        else:
            output_dimensions = (height, width, height, width)

        return (hidden_state, hidden_state_before_downsampling, output_dimensions)


class TinyVitMlp(nn.Module):
    def __init__(self, config, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_state):
        hidden_state = self.norm(hidden_state)

        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.fc2(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class TinyVitAttention(torch.nn.Module):
    def __init__(
        self,
        dim,
        key_dim,
        num_heads=8,
        attn_ratio=4,
        resolution=(14, 14),
    ):
        super().__init__()
        if not isinstance(resolution, tuple) or len(resolution) != 2:
            raise ValueError("Resolution should be a tuple of (height, width)")
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(N, N), persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, hidden_state, output_attentions):
        batch_size, seq_length, _ = hidden_state.shape

        # Normalization
        hidden_state = self.norm(hidden_state)

        qkv = self.qkv(hidden_state)
        # (batch_size, seq_length, num_heads, dim)
        queries, keys, values = qkv.view(batch_size, seq_length, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.d], dim=3
        )
        # (batch_size, num_heads, seq_length, dim)
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        attention_scores = (queries @ keys.transpose(-2, -1)) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        )
        # Normalize the attention scores to probabilities.
        attention_probs = attention_scores.softmax(dim=-1)
        hidden_state = (attention_probs @ values).transpose(1, 2).reshape(batch_size, seq_length, self.dh)
        hidden_state = self.proj(hidden_state)

        outputs = (hidden_state, attention_probs) if output_attentions else (hidden_state,)

        return outputs


class TinyViTLayer(nn.Module):
    r"""TinyViT layer (block).

    Args:
        config (`TinyVitConfig`):
            Model configuration.
        dim (`int`):
            Number of input channels.
        input_resolution (`Ttuple[int, int]`):
            Input resolution.
        num_heads (`int`):
            Number of attention heads.
        window_size (`int`, *optional*, defaults to 7):
            Window size.
        drop_path (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate.
    """

    def __init__(self, config, dim, input_resolution, num_heads, window_size=7, drop_path=0.0):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        if not window_size > 0:
            raise ValueError("window_size must be greater than 0")
        self.window_size = window_size

        self.drop_path = TinyVitDropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = TinyVitAttention(dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution)

        mlp_hidden_dim = int(dim * config.mlp_ratio)
        self.mlp = TinyVitMlp(config, in_features=dim, hidden_features=mlp_hidden_dim)

        local_conv_size = config.local_conv_size
        pad = local_conv_size // 2
        self.local_conv = TinyVitConv2dBatchNorm(
            dim, dim, kernel_size=local_conv_size, stride=1, padding=pad, groups=dim
        )

    def forward(self, x, output_attentions):
        height, width = self.input_resolution
        batch_size, seq_length, num_channels = x.shape
        if seq_length != height * width:
            raise ValueError("input feature has wrong size")
        res_x = x
        if height == self.window_size and width == self.window_size:
            attention_outputs = self.attn(x, output_attentions)
            x = attention_outputs[0]
        else:
            x = x.view(batch_size, height, width, num_channels)
            pad_b = (self.window_size - height % self.window_size) % self.window_size
            pad_r = (self.window_size - width % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = height + pad_b, width + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # window partition
            x = (
                x.view(batch_size, nH, self.window_size, nW, self.window_size, num_channels)
                .transpose(2, 3)
                .reshape(batch_size * nH * nW, self.window_size * self.window_size, num_channels)
            )
            attention_outputs = self.attn(x, output_attentions)
            x = attention_outputs[0]
            # window reverse
            x = (
                x.view(batch_size, nH, nW, self.window_size, self.window_size, num_channels)
                .transpose(2, 3)
                .reshape(batch_size, pH, pW, num_channels)
            )

            if padding:
                x = x[:, :height, :width].contiguous()

            x = x.view(batch_size, seq_length, num_channels)

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(batch_size, num_channels, height, width)
        x = self.local_conv(x)
        x = x.view(batch_size, num_channels, seq_length).transpose(1, 2)

        x = x + self.drop_path(self.mlp(x))

        layer_outputs = (x, attention_outputs[1]) if output_attentions else (x,)
        return layer_outputs


class TinyVitStage(nn.Module):
    """A basic TinyViT module for one stage.

    Args:
        config (`TinyVitConfig`):
            Model configuration.
        dim (`int`):
            Number of input channels.
        input_resolution (`Tuple[int]`):
            Input resolution.
        depth (`int`):
            Number of layers.
        num_heads (`int`):
            Number of attention heads.
        window_size (`int`):
            Local window size.
        drop_path (`float` or `Tuple[float]`, *optional*, defaults to 0.0):
            Stochastic depth rate.
        downsample (`nn.Module`, *optional*):
            Optional downsample layer at the end of the stage.
        out_dim (`int`, *optional*):
            The output dimension of the stage.
    """

    def __init__(
        self,
        config,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        drop_path=0.0,
        downsample=None,
        out_dim=None,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.depth = depth

        # build layers
        self.layers = nn.ModuleList(
            [
                TinyViTLayer(
                    config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(config, input_resolution, dim=dim, out_dim=out_dim)
        else:
            self.downsample = None

        self.gradient_checkpointing = False

    def forward(self, hidden_state, input_dimensions, output_attentions=False):
        height, width = input_dimensions
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer.__call__, hidden_state, output_attentions)
            else:
                layer_outputs = layer(hidden_state, output_attentions)
                hidden_state = layer_outputs[0]

        hidden_state_before_downsampling = hidden_state
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_state = self.downsample(hidden_state)
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (hidden_state, hidden_state_before_downsampling, output_dimensions)

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


class TinyVitEncoder(nn.Module):
    def __init__(self, config, patches_resolution):
        super().__init__()
        self.num_stages = len(config.depths)
        self.config = config
        hidden_sizes = config.hidden_sizes
        depths = config.depths

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stages):
            kwargs = {
                "dim": hidden_sizes[stage_idx],
                "input_resolution": (
                    patches_resolution[0] // (2**stage_idx),
                    patches_resolution[1] // (2**stage_idx),
                ),
                "depth": depths[stage_idx],
                "drop_path": dpr[sum(depths[:stage_idx]) : sum(depths[: stage_idx + 1])],
                "downsample": TinyVitPatchMerging if (stage_idx < self.num_stages - 1) else None,
                "out_dim": hidden_sizes[min(stage_idx + 1, len(hidden_sizes) - 1)],
            }
            if stage_idx == 0:
                stage = TinyVitConvStage(config, **kwargs)
            else:
                stage = TinyVitStage(
                    config, num_heads=config.num_heads[stage_idx], window_size=config.window_sizes[stage_idx], **kwargs
                )
            self.stages.append(stage)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, TinyVitEncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            # note: this line makes the retain_grad test fail
            all_hidden_states += (hidden_states.flatten(2).transpose(1, 2),)
            all_reshaped_hidden_states += (hidden_states,)

        for i, stage_module in enumerate(self.stages):
            if self.gradient_checkpointing and self.training:
                stage_outputs = self._gradient_checkpointing_func(
                    stage_module.__call__, hidden_states, input_dimensions, output_attentions
                )
            else:
                stage_outputs = stage_module(hidden_states, input_dimensions, output_attentions)

            hidden_states = stage_outputs[0]
            hidden_state_before_downsampling = stage_outputs[1]
            output_dimensions = stage_outputs[2]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_downsampling:
                if i != 0:
                    batch_size, _, hidden_size = hidden_state_before_downsampling.shape
                    # rearrange b (h w) c -> b c h w
                    # here we use the original (not downsampled) height and width
                    reshaped_hidden_state = hidden_state_before_downsampling.view(
                        batch_size, *(output_dimensions[0], output_dimensions[1]), hidden_size
                    )
                    reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                else:
                    reshaped_hidden_state = hidden_state_before_downsampling.flatten(2).transpose(1, 2)

                all_hidden_states += (hidden_state_before_downsampling,)
                all_reshaped_hidden_states += (
                    (hidden_state_before_downsampling,) if i == 0 else (reshaped_hidden_state,)
                )
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)

                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += stage_outputs[3:]

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions, all_reshaped_hidden_states]
                if v is not None
            )

        return TinyVitEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


class TinyVitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TinyVitConfig
    base_model_prefix = "tinyvit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, TinyVitMBConv):
            module.conv3.batch_norm.weight.data.zero_()
            module.conv3.batch_norm.bias.data.zero_()
        elif isinstance(module, TinyVitConv2dBatchNorm):
            torch.nn.init.constant_(module.batch_norm.weight, 1.0)
            torch.nn.init.constant_(module.batch_norm.bias, 0)


TINYVIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`TinyVitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TINYVIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare TinyVit Model transformer outputting raw hidden-states without any specific head on top.",
    TINYVIT_START_DOCSTRING,
)
class TinyVitModel(TinyVitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = TinyVitEmbeddings(config)
        self.encoder = TinyVitEncoder(config, self.embeddings.patches_resolution)

        # Initialize weights and apply final processing
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(TINYVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output, input_dimensions = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]

        pooled_output = sequence_output.mean(1)

        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]

            return output

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """
    TinyVit Model transformer with an image classification head on top (a linear layer on top of the mean pooled final
    hidden states) e.g. for ImageNet.
    """,
    TINYVIT_START_DOCSTRING,
)
class TinyVitForImageClassification(TinyVitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.tinyvit = TinyVitModel(config)

        # Classifier head
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1])
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(TINYVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.tinyvit(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.layernorm(pooled_output)
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

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TinyVitLayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


@add_start_docstrings(
    """
    TinyViT backbone, to be used with frameworks like SAM (segment-anything-model).
    """,
    TINYVIT_START_DOCSTRING,
)
class TinyVitBackbone(TinyVitPreTrainedModel, BackboneMixin):
    def __init__(self, config: TinyVitConfig):
        super().__init__(config)
        super()._init_backbone(config)

        self.num_features = [config.hidden_sizes[0]] + config.hidden_sizes
        self.embeddings = TinyVitEmbeddings(config)
        self.encoder = TinyVitEncoder(config, self.embeddings.patches_resolution)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("microsoft/tinyvit-21m-1k")
        >>> model = AutoBackbone.from_pretrained(
        ...     "microsoft/tinyvit-21m-1k", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 7, 7]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        embedding_output, input_dimensions = self.embeddings(pixel_values)

        outputs = self.encoder(
            embedding_output,
            input_dimensions,
            output_attentions=output_attentions,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True,
            return_dict=return_dict,
        )

        hidden_states = outputs.reshaped_hidden_states if return_dict else outputs[-1]

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                feature_maps += (hidden_state,)

        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs[1],)
            if output_attentions:
                output += (outputs[2],)
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
