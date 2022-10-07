# coding=utf-8
# Copyright 2022 Multimedia Computing Group, Nanjing University and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch TimeSformer model."""


import math
import warnings
from copy import deepcopy
from dataclasses import dataclass
from itertools import repeat
from typing import Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .configuration_timesformer import TimeSformerConfig


TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TimeSformerConfig"
_CHECKPOINT_FOR_DOC = "facebook/timesformer"

TIMESFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/timesformer",
    # See all TimeSformer models at https://huggingface.co/models?filter=timesformer
]


# Adapted from https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py#L155
class TimeSformerPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, config):
        super().__init__()

        img_size = config.image_size
        patch_size = config.patch_size
        in_chans = config.num_channels
        embed_dim = config.hidden_size

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, num_frames, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).view(B * num_frames, C, H, W)

        x = self.projection(x)
        patch_width = x.size(-1)
        embeddings = x.flatten(2).transpose(1, 2)
        return embeddings, num_frames, patch_width


class TimeSformerEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings.

    """

    def __init__(self, config):
        super().__init__()

        embed_dim = config.hidden_size
        num_frames = config.num_frames
        drop_rate = config.hidden_dropout_prob
        attention_type = config.attention_type

        self.attention_type = attention_type
        self.patch_embeddings = TimeSformerPatchEmbed(config)
        self.num_patches = self.patch_embeddings.num_patches

        # Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if attention_type != "space_only":
            self.time_embeddings = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]

        # create patch embeddings
        embeddings, num_frames, patch_width = self.patch_embeddings(pixel_values)

        cls_tokens = self.cls_token.expand(embeddings.size(0), -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # resizing the positional embeddings in case they don't match the input at inference
        if embeddings.size(1) != self.position_embeddings.size(1):
            position_embeddings = self.position_embeddings
            cls_pos_embed = position_embeddings[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = position_embeddings[0, 1:, :].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = embeddings.size(1) // patch_width
            other_pos_embed = other_pos_embed.reshape(1, embeddings.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, patch_width), mode="nearest")
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            embeddings = embeddings + new_pos_embed
        else:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.pos_drop(embeddings)

        # Time Embeddings
        if self.attention_type != "space_only":
            cls_tokens = embeddings[:batch_size, 0, :].unsqueeze(1)
            embeddings = embeddings[:, 1:]
            _, patch_height, patch_width = embeddings.shape
            embeddings = (
                embeddings.view(batch_size, num_frames, patch_height, patch_width)
                .permute(0, 2, 1, 3)
                .view(batch_size * patch_height, num_frames, patch_width)
            )
            # Resizing time embeddings in case they don't match
            if num_frames != self.time_embeddings.size(1):
                time_embeddings = self.time_embeddings.transpose(1, 2)
                new_time_embeddings = F.interpolate(time_embeddings, size=(num_frames), mode="nearest")
                new_time_embeddings = new_time_embeddings.transpose(1, 2)
                embeddings = embeddings + new_time_embeddings
            else:
                embeddings = embeddings + self.time_embeddings
            embeddings = self.time_drop(embeddings)
            # embeddings = rearrange(embeddings, "(b n) t m -> b (n t) m", b=batch_size, t=num_frames)
            embeddings = embeddings.view(batch_size, patch_height, num_frames, patch_width).reshape(
                batch_size, patch_height * num_frames, patch_width
            )
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings


# Copied from https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit_utils.py#L78
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


# Copied from https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit_utils.py#L31
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


# Copied from https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit_utils.py#L64
def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (tensor, float, float, float, float) -> tensor
    r"""Fills the input Tensor with values drawn from a truncated
    Args:
    normal distribution. The values are effectively drawn from the normal distribution :math:`\mathcal{N}(\text{mean},
    \text{std}^2)` with values outside :math:`[a, b]` redrawn until they are within the bounds. The method used for
    generating the random values works best when :math:`a \leq \text{mean} \leq b`.
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5) >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks). Comment by Ross Wightman:
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however, the original name is
    misleading as 'Drop Connect' is a different form of dropout in a separate paper... See discussion:
    https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the layer and
    argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class TimeSformerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, config: TimeSformerConfig):
        super().__init__()
        self.drop_prob = config.drop_path_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# Adapted from https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py#L57
# and transformers.models.videomae.modeling_videomae.VideoMAESelfAttention
class TimeSformerSelfAttention(nn.Module):
    def __init__(self, config: TimeSformerConfig):
        super().__init__()

        dim = config.hidden_size
        num_heads = config.num_attention_heads
        qkv_bias = config.qkv_bias
        attn_drop = config.attention_probs_dropout_prob

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, hidden_states, output_attentions: bool = False):
        B, N, C = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention_probs = (q @ k.transpose(-2, -1)) * self.scale
        attention_probs = attention_probs.softmax(dim=-1)
        attention_probs = self.attn_drop(attention_probs)

        context_layer = (attention_probs @ v).transpose(1, 2).reshape(B, N, C)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# copied from transformers.models.videomae.modeling_videomae.VideoMAESelfOutput
class TimeSformerSelfOutput(nn.Module):
    """
    The residual connection is defined in TimeSformerLayer instead of here (as is the case with other models), due to
    the layernorm applied before each block.
    """

    def __init__(self, config: TimeSformerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# copied from transformers.models.videomae.modeling_videomae.VideoMAEAttention
class TimeSformerAttention(nn.Module):
    def __init__(self, config: TimeSformerConfig) -> None:
        super().__init__()
        self.attention = TimeSformerSelfAttention(config)
        self.output = TimeSformerSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, output_attentions)

        attention_output = self.output(self_outputs[0])

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Adapted from https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py#L39
# and transformers.models.videomae.modeling_videomae.VideoMAEIntermediate
class TimeSformerIntermediate(nn.Module):
    def __init__(self, config: TimeSformerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.VideoMAEOutput
class TimeSformerOutput(nn.Module):
    def __init__(self, config: TimeSformerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Adapted from https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py#L89
class TimeSformerLayer(nn.Module):
    def __init__(self, config: TimeSformerConfig, layer_index: int) -> None:
        super().__init__()

        dim = config.hidden_size
        attention_type = config.attention_type

        dpr = [
            x.item() for x in torch.linspace(0, config.drop_path_prob, config.num_hidden_layers)
        ]  # stochastic depth decay rule
        drop_path_prob = dpr[layer_index]

        self.drop_path = TimeSformerDropPath(config) if drop_path_prob > 0.0 else nn.Identity()
        self.attention = TimeSformerAttention(config)
        self.intermediate = TimeSformerIntermediate(config)
        self.output = TimeSformerOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.config = config
        self.attention_type = attention_type
        assert attention_type in ["divided_space_time", "space_only", "joint_space_time"]

        # Temporal Attention Parameters
        if self.attention_type == "divided_space_time":
            self.temporal_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.temporal_attention = TimeSformerAttention(config)
            self.temporal_dense = nn.Linear(dim, dim)

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False):
        T = self.config.num_frames
        W = self.config.image_size // self.config.patch_size
        B = hidden_states.shape[0]
        num_spatial_tokens = (hidden_states.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ["space_only", "joint_space_time"]:
            self_attention_outputs = self.attention(
                self.layernorm_before(hidden_states), output_attentions=output_attentions
            )
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

            hidden_states = hidden_states + self.drop_path(attention_output)

            layer_output = self.layernorm_after(hidden_states)
            layer_output = self.intermediate(layer_output)
            layer_output = self.output(layer_output)
            layer_output = hidden_states + self.drop_path(layer_output)

            outputs = (layer_output,) + outputs

            return outputs

        elif self.attention_type == "divided_space_time":
            # Temporal
            xt = hidden_states[:, 1:, :]
            xt = xt.reshape(B, H, W, T, xt.shape[2]).reshape(B * H * W, T, xt.shape[2])

            temporal_attention_outputs = self.temporal_attention(
                self.temporal_layernorm(xt), output_attentions=output_attentions
            )
            attention_output = temporal_attention_outputs[0]
            outputs = temporal_attention_outputs[1:]  # add self attentions if we output attention weights

            res_temporal = self.drop_path(attention_output)

            res_temporal = res_temporal.reshape(B, H, W, T, res_temporal.shape[2]).reshape(
                B, H * W * T, res_temporal.shape[2]
            )
            res_temporal = self.temporal_dense(res_temporal)
            xt = hidden_states[:, 1:, :] + res_temporal

            # Spatial
            init_cls_token = hidden_states[:, 0, :].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = cls_token.reshape(B * T, 1, cls_token.shape[2])
            xs = xt
            xs = xs.reshape(B, H, W, T, xs.shape[2]).permute(0, 3, 1, 2, 4).reshape(B * T, H * W, xs.shape[2])
            xs = torch.cat((cls_token, xs), 1)

            spatial_attention_outputs = self.attention(
                self.layernorm_before(xs),
            )
            attention_output = spatial_attention_outputs[0]

            res_spatial = self.drop_path(attention_output)

            # Taking care of CLS token
            cls_token = res_spatial[:, 0, :]
            # cls_token = rearrange(cls_token, "(b t) m -> b t m", b=B, t=T)
            cls_token = cls_token.reshape(B, T, cls_token.shape[1])
            cls_token = torch.mean(cls_token, 1, True)  # averaging for every frame
            res_spatial = res_spatial[:, 1:, :]
            # res_spatial = rearrange(res_spatial, "(b t) (h w) m -> b (h w t) m", b=B, h=H, w=W, t=T)
            res_spatial = (
                res_spatial.reshape(B, T, H, W, res_spatial.shape[2])
                .permute(0, 2, 3, 1, 4)
                .reshape(B, H * W * T, res_spatial.shape[2])
            )
            res = res_spatial
            hidden_states = xt

            # Mlp
            hidden_states = torch.cat((init_cls_token, hidden_states), 1) + torch.cat((cls_token, res), 1)
            layer_output = self.layernorm_after(hidden_states)
            layer_output = self.intermediate(layer_output)
            layer_output = self.output(layer_output)
            layer_output = hidden_states + self.drop_path(layer_output)

            outputs = (layer_output,) + outputs

            return outputs


# Copied from transformers.models.vit.modeling_vit.VideoMAEEncoder
class TimeSformerEncoder(nn.Module):
    def __init__(self, config: TimeSformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([TimeSformerLayer(config, ind) for ind in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                )
            else:
                layer_outputs = layer_module(hidden_states, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# Copied from transformers.models.videomae.modeling_videomae.VideoMAEPreTrainedModel with VideoMAE->TimeSformer,videomae->timesformer
class TimeSformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TimeSformerConfig
    base_model_prefix = "timesformer"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, TimeSformerEncoder):
            module.gradient_checkpointing = value

    config_class = TimeSformerConfig
    base_model_prefix = "timesformer"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True


TIMESFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`TimeSformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TIMESFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`VideoMAEFeatureExtractor`]. See
            [`VideoMAEFeatureExtractor.__call__`] for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

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
    "The bare TimeSformer Model transformer outputting raw hidden-states without any specific head on top.",
    TIMESFORMER_START_DOCSTRING,
)
# Copied from transformers.models.videomae.modeling_videomae.VideoMAEModel with VIDEOMAE->TIMESFORMER,VideoMAE->TimeSformer,MCG-NJU/videomae-base->facebook/timesformer
class TimeSformerModel(TimeSformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = TimeSformerEmbeddings(config)
        self.encoder = TimeSformerEncoder(config)

        if config.use_mean_pooling:
            self.layernorm = None
        else:
            self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(TIMESFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from decord import VideoReader, cpu
        >>> import numpy as np

        >>> from transformers import TimeSformerFeatureExtractor, TimeSformerModel
        >>> from huggingface_hub import hf_hub_download


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...     converted_len = int(clip_len * frame_sample_rate)
        ...     end_idx = np.random.randint(converted_len, seg_len)
        ...     start_idx = end_idx - converted_len
        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...     return indices


        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
        >>> file_path = hf_hub_download(
        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
        ... )
        >>> vr = VideoReader(file_path, num_threads=1, ctx=cpu(0))

        >>> # sample 8 frames
        >>> vr.seek(0)
        >>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=4, seg_len=len(vr))
        >>> buffer = vr.get_batch(indices).asnumpy()

        >>> # create a list of NumPy arrays
        >>> video = [buffer[i] for i in range(buffer.shape[0])]

        >>> feature_extractor = TimeSformerFeatureExtractor.from_pretrained("facebook/timesformer")
        >>> model = TimeSformerModel.from_pretrained("facebook/timesformer")

        >>> # prepare video for the model
        >>> inputs = feature_extractor(video, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 1568, 768]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        if self.layernorm is not None:
            sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """TimeSformer Model transformer with a video classification head on top (a linear layer on top of the final hidden state
of
    the [CLS] token) e.g. for ImageNet.""",
    TIMESFORMER_START_DOCSTRING,
)
# Copied from transformers.models.videomae.modeling_videomae.VideoMAEForVideoClassification with VIDEOMAE->TIMESFORMER,VideoMAE->TimeSformer,videomae->timesformer,MCG-NJU/videomae-base->facebook/timesformer
class TimeSformerForVideoClassification(TimeSformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.timesformer = TimeSformerModel(config)

        # Classifier head
        self.fc_norm = nn.LayerNorm(config.hidden_size) if config.use_mean_pooling else None
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(TIMESFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from decord import VideoReader, cpu
        >>> import torch

        >>> from transformers import TimeSformerFeatureExtractor, TimeSformerForVideoClassification
        >>> from huggingface_hub import hf_hub_download


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...     converted_len = int(clip_len * frame_sample_rate)
        ...     end_idx = np.random.randint(converted_len, seg_len)
        ...     start_idx = end_idx - converted_len
        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...     return indices


        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
        >>> file_path = hf_hub_download(
        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
        ... )
        >>> vr = VideoReader(file_path, num_threads=1, ctx=cpu(0))

        >>> # sample 16 frames
        >>> vr.seek(0)
        >>> indices = sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=len(vr))
        >>> buffer = vr.get_batch(indices).asnumpy()

        >>> # create a list of NumPy arrays
        >>> video = [buffer[i] for i in range(buffer.shape[0])]

        >>> feature_extractor = TimeSformerFeatureExtractor.from_pretrained(
        ...     "MCG-NJU/timesformer-base-finetuned-kinetics"
        ... )
        >>> model = TimeSformerForVideoClassification.from_pretrained("MCG-NJU/timesformer-base-finetuned-kinetics")

        >>> inputs = feature_extractor(video, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ...     logits = outputs.logits

        >>> # model predicts one of the 400 Kinetics-400 classes
        >>> predicted_label = logits.argmax(-1).item()
        >>> print(model.config.id2label[predicted_label])
        eating spaghetti
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)

        outputs = self.timesformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if self.fc_norm is not None:
            sequence_output = self.fc_norm(sequence_output.mean(1))
        else:
            sequence_output = sequence_output[:, 0]

        logits = self.classifier(sequence_output)

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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
