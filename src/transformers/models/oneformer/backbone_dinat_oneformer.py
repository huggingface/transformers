# coding=utf-8
# Copyright 2022 SHI Labs and The HuggingFace Inc. team. All rights reserved.
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
""" DiNAT Backbone model for OneFormer."""
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_utils import ModuleUtilsMixin
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, OptionalDependencyNotAvailable, is_natten_available, logging, requires_backends
from .configuration_oneformer import OneFormerConfig


if is_natten_available():
    from natten.functional import natten2dav, natten2dqkrpb
else:

    def natten2dqkrpb(*args, **kwargs):
        raise OptionalDependencyNotAvailable()

    def natten2dav(*args, **kwargs):
        raise OptionalDependencyNotAvailable()


logger = logging.get_logger(__name__)

# drop_path and DinatDropPath are from the timm library.


@dataclass
# Copied from transformers.models.dinat.modeling_dinat.DinatEncoderOutput with Dinat->OneFormerDinat
class OneFormerDinatEncoderOutput(ModelOutput):
    """
    Dinat encoder's outputs, with potential hidden states and attentions.

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


@dataclass
class OneFormerDinatModelOutput(ModelOutput):
    """
    Dinat model's outputs.

     Args:
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class OneFormerDinatEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings.
    """

    def __init__(self, config: OneFormerConfig):
        super().__init__()

        self.patch_embeddings = OneFormerDinatPatchEmbeddings(config)

        self.norm = nn.LayerNorm(config.backbone_config["embed_dim"])
        self.dropout = nn.Dropout(config.backbone_config["hidden_dropout_prob"])

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor]:
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)

        embeddings = self.dropout(embeddings)

        return embeddings


class OneFormerDinatPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, height, width, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        patch_size = config.backbone_config["patch_size"]
        num_channels, hidden_size = config.backbone_config["num_channels"], config.backbone_config["embed_dim"]
        self.num_channels = num_channels

        if patch_size == 4:
            pass
        else:
            # TODO: Support arbitrary patch sizes.
            raise ValueError("Dinat only supports patch size of 4 at the moment.")

        self.projection = nn.Sequential(
            nn.Conv2d(self.num_channels, hidden_size // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> torch.Tensor:
        _, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = self.projection(pixel_values)
        embeddings = embeddings.permute(0, 2, 3, 1)

        return embeddings


# Copied from transformers.models.dinat.modeling_dinat.DinatDownsampler with Dinat->OneFormerDinat
class OneFormerDinatDownsampler(nn.Module):
    """
    Convolutional Downsampling Layer.

    Args:
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        input_feature = self.reduction(input_feature.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        input_feature = self.norm(input_feature)
        return input_feature


# Copied from transformers.dinat.modeling_dinat.drop_path
def drop_path(input, drop_prob=0.0, training=False, scale_by_keep=True):
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


# Copied from transformers.models.dinat.modeling_dinat.DinatDropPath with Dinat->OneFormerDinat
class OneFormerDinatDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class OneFormerNeighborhoodAttention(nn.Module):
    def __init__(self, config, dim, num_heads, kernel_size, dilation):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.kernel_size = kernel_size
        self.dilation = dilation

        # rpb is learnable relative positional biases; same concept is used Swin.
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * self.kernel_size - 1), (2 * self.kernel_size - 1)))

        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.backbone_config["qkv_bias"])
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.backbone_config["qkv_bias"])
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.backbone_config["qkv_bias"])

        self.dropout = nn.Dropout(config.backbone_config["attention_probs_dropout_prob"])

    # Copied from transformers.models.nat.modeling_nat.NeighborhoodAttention.transpose_for_scores with Dinat-OneFormerDinat
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 3, 1, 2, 4)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Apply the scale factor before computing attention weights. It's usually more efficient because
        # attention weights are typically a bigger tensor compared to query.
        # It gives identical results because scalars are commutable in matrix multiplication.
        query_layer = query_layer / math.sqrt(self.attention_head_size)

        # Compute NA between "query" and "key" to get the raw attention scores, and add relative positional biases.
        attention_scores = natten2dqkrpb(query_layer, key_layer, self.rpb, self.dilation)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = natten2dav(attention_probs, value_layer, self.dilation)
        context_layer = context_layer.permute(0, 2, 3, 1, 4).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class OneFormerNeighborhoodAttentionOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.backbone_config["attention_probs_dropout_prob"])

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.dinat.modeling_dinat.NeighborhoodAttentionModule with Neighborhood->OneFormerNeighborhood
class OneFormerNeighborhoodAttentionModule(nn.Module):
    def __init__(self, config, dim, num_heads, kernel_size, dilation):
        super().__init__()
        self.self = OneFormerNeighborhoodAttention(config, dim, num_heads, kernel_size, dilation)
        self.output = OneFormerNeighborhoodAttentionOutput(config, dim)
        self.pruned_heads = set()

    # Copied from transformers.models.nat.modeling_nat.NeighborhoodAttentionModule.prune_heads
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # Copied from transformers.models.nat.modeling_nat.NeighborhoodAttentionModule.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class OneFormerDinatIntermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, int(config.backbone_config["mlp_ratio"] * dim))
        if isinstance(config.backbone_config["hidden_act"], str):
            self.intermediate_act_fn = ACT2FN[config.backbone_config["hidden_act"]]
        else:
            self.intermediate_act_fn = config.backbone_config["hidden_act"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class OneFormerDinatOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(int(config.backbone_config["mlp_ratio"] * dim), dim)
        self.dropout = nn.Dropout(config.backbone_config["hidden_dropout_prob"])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class OneFormerDinatLayer(nn.Module):
    def __init__(self, config, dim, num_heads, dilation, drop_path_rate=0.0):
        super().__init__()
        self.kernel_size = config.backbone_config["kernel_size"]
        self.dilation = dilation
        self.window_size = self.kernel_size * self.dilation
        self.layernorm_before = nn.LayerNorm(dim, eps=config.general_config["layer_norm_eps"])
        self.attention = OneFormerNeighborhoodAttentionModule(
            config, dim, num_heads, kernel_size=self.kernel_size, dilation=self.dilation
        )
        self.drop_path = OneFormerDinatDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_after = nn.LayerNorm(dim, eps=config.general_config["layer_norm_eps"])
        self.intermediate = OneFormerDinatIntermediate(config, dim)
        self.output = OneFormerDinatOutput(config, dim)
        self.layer_scale_parameters = (
            nn.Parameter(config.backbone_config["layer_scale_init_value"] * torch.ones((2, dim)), requires_grad=True)
            if config.backbone_config["layer_scale_init_value"] > 0
            else None
        )

    def maybe_pad(self, hidden_states, height, width):
        window_size = self.window_size
        pad_values = (0, 0, 0, 0, 0, 0)
        if height < window_size or width < window_size:
            pad_l = pad_t = 0
            pad_r = max(0, window_size - width)
            pad_b = max(0, window_size - height)
            pad_values = (0, 0, pad_l, pad_r, pad_t, pad_b)
            hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, height, width, channels = hidden_states.size()
        shortcut = hidden_states

        hidden_states = self.layernorm_before(hidden_states)
        # pad hidden_states if they are smaller than kernel size x dilation
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        _, height_pad, width_pad, _ = hidden_states.shape

        attention_outputs = self.attention(hidden_states, output_attentions=output_attentions)

        attention_output = attention_outputs[0]

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_output = attention_output[:, :height, :width, :].contiguous()

        if self.layer_scale_parameters is not None:
            attention_output = self.layer_scale_parameters[0] * attention_output

        hidden_states = shortcut + self.drop_path(attention_output)

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.output(self.intermediate(layer_output))

        if self.layer_scale_parameters is not None:
            layer_output = self.layer_scale_parameters[1] * layer_output

        layer_output = hidden_states + self.drop_path(layer_output)

        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs


# Copied from transformers.models.dinat.modeling_dinat.DinatStage with Dinat->OneFormerDinat
class OneFormerDinatStage(nn.Module):
    def __init__(self, config, dim, depth, num_heads, dilations, drop_path_rate, downsample):
        super().__init__()
        self.config = config
        self.dim = dim
        self.layers = nn.ModuleList(
            [
                OneFormerDinatLayer(
                    config=config,
                    dim=dim,
                    num_heads=num_heads,
                    dilation=dilations[i],
                    drop_path_rate=drop_path_rate[i],
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        self.pointing = False

    # Copied from transformers.models.dinat.modeling_dinat.DinatStage.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        _, height, width, _ = hidden_states.size()
        for i, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(hidden_states, output_attentions)
            hidden_states = layer_outputs[0]

        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(layer_outputs[0])
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (hidden_states, layer_outputs[0], output_dimensions)

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


class OneFormerDinatEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_levels = len(config.backbone_config["depths"])
        self.config = config
        dpr = [
            x.item()
            for x in torch.linspace(0, config.backbone_config["drop_path_rate"], sum(config.backbone_config["depths"]))
        ]
        self.levels = nn.ModuleList(
            [
                OneFormerDinatStage(
                    config=config,
                    dim=int(config.backbone_config["embed_dim"] * 2**i_layer),
                    depth=config.backbone_config["depths"][i_layer],
                    num_heads=config.backbone_config["num_heads"][i_layer],
                    dilations=config.backbone_config["dilations"][i_layer],
                    drop_path_rate=dpr[
                        sum(config.backbone_config["depths"][:i_layer]) : sum(
                            config.backbone_config["depths"][: i_layer + 1]
                        )
                    ],
                    downsample=OneFormerDinatDownsampler if (i_layer < self.num_levels - 1) else None,
                )
                for i_layer in range(self.num_levels)
            ]
        )

    # Copied from transformers.models.dinat.modeling_dinat.DinatEncoder.forward with Dinat-OneFormerDinat
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, OneFormerDinatEncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            # rearrange b h w c -> b c h w
            reshaped_hidden_state = hidden_states.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.levels):
            layer_outputs = layer_module(hidden_states, output_attentions)

            hidden_states = layer_outputs[0]
            feature_states = layer_outputs[1]

            if output_hidden_states:
                # rearrange b h w c -> b c h w
                reshaped_hidden_state = feature_states.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[2:]

        return OneFormerDinatEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


class OneFormerDinatModel(nn.Module, ModuleUtilsMixin):
    def __init__(self, config: OneFormerConfig, add_pooling_layer=True):
        super().__init__()

        requires_backends(self, ["natten"])

        self.config = config
        self.num_levels = len(config.backbone_config["depths"])
        self.num_features = int(config.backbone_config["embed_dim"] * 2 ** (self.num_levels - 1))

        self.embeddings = OneFormerDinatEmbeddings(config)
        self.encoder = OneFormerDinatEncoder(config)

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, OneFormerDinatModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        return OneFormerDinatModelOutput(
            hidden_states=encoder_outputs.reshaped_hidden_states,
            attentions=encoder_outputs.attentions,
        )
