# coding=utf-8
# Copyright 2023 The Google Research Team Authors and The HuggingFace Team. All rights reserved.
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
"""PyTorch ALIGN model."""

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndNoAttention,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, auto_docstring, can_return_tuple, logging
from .configuration_align import AlignConfig, AlignTextConfig, AlignVisionConfig


logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.
    """
)
class AlignVisionModelOutput(ModelOutput):
    r"""
    image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
        The image embeddings obtained by applying the projection layer to the pooler_output.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for text model's outputs that also contains a pooling of the last hidden states.
    """
)
class AlignTextModelOutput(ModelOutput):
    r"""
    text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
        The text embeddings obtained by applying the projection layer to the pooler_output.
    """

    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


@dataclass
@auto_docstring
class AlignOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
        Contrastive loss for image-text similarity.
    logits_per_image (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
        The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
        similarity scores.
    logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
        The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
        similarity scores.
    text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The text embeddings obtained by applying the projection layer to the pooled output of [`AlignTextModel`].
    image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The output of [`AlignVisionModel`].
    text_model_output (`BaseModelOutputWithPooling`):
        The output of the [`AlignTextModel`].
    vision_model_output (`BaseModelOutputWithPoolingAndNoAttention`):
        The output of the [`AlignVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: Optional[torch.FloatTensor] = None
    logits_per_text: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPoolingAndNoAttention = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device), label_smoothing=0.1)


def align_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


# Copied from transformers.models.efficientnet.modeling_efficientnet.round_filters with EfficientNet->AlignVision
def round_filters(config: AlignVisionConfig, num_channels: int):
    r"""
    Round number of filters based on depth multiplier.
    """
    divisor = config.depth_divisor
    num_channels *= config.width_coefficient
    new_dim = max(divisor, int(num_channels + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_dim < 0.9 * num_channels:
        new_dim += divisor

    return int(new_dim)


# Copied from transformers.models.efficientnet.modeling_efficientnet.correct_pad
def correct_pad(kernel_size: Union[int, tuple], adjust: bool = True):
    r"""
    Utility function to get the tuple padding value for the depthwise convolution.

    Args:
        kernel_size (`int` or `tuple`):
            Kernel size of the convolution layers.
        adjust (`bool`, *optional*, defaults to `True`):
            Adjusts padding value to apply to right and bottom sides of the input.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    if adjust:
        return (correct[1] - 1, correct[1], correct[0] - 1, correct[0])
    else:
        return (correct[1], correct[1], correct[0], correct[0])


# Copied from transformers.models.efficientnet.modeling_efficientnet.EfficientNetEmbeddings with EfficientNet->AlignVision
class AlignVisionEmbeddings(nn.Module):
    r"""
    A module that corresponds to the stem module of the original work.
    """

    def __init__(self, config: AlignVisionConfig):
        super().__init__()

        self.out_dim = round_filters(config, 32)
        self.padding = nn.ZeroPad2d(padding=(0, 1, 0, 1))
        self.convolution = nn.Conv2d(
            config.num_channels, self.out_dim, kernel_size=3, stride=2, padding="valid", bias=False
        )
        self.batchnorm = nn.BatchNorm2d(self.out_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.padding(pixel_values)
        features = self.convolution(features)
        features = self.batchnorm(features)
        features = self.activation(features)

        return features


# Copied from transformers.models.efficientnet.modeling_efficientnet.EfficientNetDepthwiseConv2d with EfficientNet->AlignVision
class AlignVisionDepthwiseConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        depth_multiplier=1,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
    ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )


# Copied from transformers.models.efficientnet.modeling_efficientnet.EfficientNetExpansionLayer with EfficientNet->AlignVision
class AlignVisionExpansionLayer(nn.Module):
    r"""
    This corresponds to the expansion phase of each block in the original implementation.
    """

    def __init__(self, config: AlignVisionConfig, in_dim: int, out_dim: int, stride: int):
        super().__init__()
        self.expand_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            padding="same",
            bias=False,
        )
        self.expand_bn = nn.BatchNorm2d(num_features=out_dim, eps=config.batch_norm_eps)
        self.expand_act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # Expand phase
        hidden_states = self.expand_conv(hidden_states)
        hidden_states = self.expand_bn(hidden_states)
        hidden_states = self.expand_act(hidden_states)

        return hidden_states


# Copied from transformers.models.efficientnet.modeling_efficientnet.EfficientNetDepthwiseLayer with EfficientNet->AlignVision
class AlignVisionDepthwiseLayer(nn.Module):
    r"""
    This corresponds to the depthwise convolution phase of each block in the original implementation.
    """

    def __init__(
        self,
        config: AlignVisionConfig,
        in_dim: int,
        stride: int,
        kernel_size: int,
        adjust_padding: bool,
    ):
        super().__init__()
        self.stride = stride
        conv_pad = "valid" if self.stride == 2 else "same"
        padding = correct_pad(kernel_size, adjust=adjust_padding)

        self.depthwise_conv_pad = nn.ZeroPad2d(padding=padding)
        self.depthwise_conv = AlignVisionDepthwiseConv2d(
            in_dim, kernel_size=kernel_size, stride=stride, padding=conv_pad, bias=False
        )
        self.depthwise_norm = nn.BatchNorm2d(
            num_features=in_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum
        )
        self.depthwise_act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # Depthwise convolution
        if self.stride == 2:
            hidden_states = self.depthwise_conv_pad(hidden_states)

        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_norm(hidden_states)
        hidden_states = self.depthwise_act(hidden_states)

        return hidden_states


# Copied from transformers.models.efficientnet.modeling_efficientnet.EfficientNetSqueezeExciteLayer with EfficientNet->AlignVision
class AlignVisionSqueezeExciteLayer(nn.Module):
    r"""
    This corresponds to the Squeeze and Excitement phase of each block in the original implementation.
    """

    def __init__(self, config: AlignVisionConfig, in_dim: int, expand_dim: int, expand: bool = False):
        super().__init__()
        self.dim = expand_dim if expand else in_dim
        self.dim_se = max(1, int(in_dim * config.squeeze_expansion_ratio))

        self.squeeze = nn.AdaptiveAvgPool2d(output_size=1)
        self.reduce = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim_se,
            kernel_size=1,
            padding="same",
        )
        self.expand = nn.Conv2d(
            in_channels=self.dim_se,
            out_channels=self.dim,
            kernel_size=1,
            padding="same",
        )
        self.act_reduce = ACT2FN[config.hidden_act]
        self.act_expand = nn.Sigmoid()

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        inputs = hidden_states
        hidden_states = self.squeeze(hidden_states)
        hidden_states = self.reduce(hidden_states)
        hidden_states = self.act_reduce(hidden_states)

        hidden_states = self.expand(hidden_states)
        hidden_states = self.act_expand(hidden_states)
        hidden_states = torch.mul(inputs, hidden_states)

        return hidden_states


class AlignVisionFinalBlockLayer(nn.Module):
    r"""
    This corresponds to the final phase of each block in the original implementation.
    """

    def __init__(
        self, config: AlignVisionConfig, in_dim: int, out_dim: int, stride: int, drop_rate: float, id_skip: bool
    ):
        super().__init__()
        self.apply_dropout = stride == 1 and not id_skip
        self.project_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            padding="same",
            bias=False,
        )
        self.project_bn = nn.BatchNorm2d(
            num_features=out_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum
        )
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, embeddings: torch.FloatTensor, hidden_states: torch.FloatTensor) -> torch.Tensor:
        hidden_states = self.project_conv(hidden_states)
        hidden_states = self.project_bn(hidden_states)

        if self.apply_dropout:
            hidden_states = self.dropout(hidden_states)
            hidden_states = hidden_states + embeddings

        return hidden_states


class AlignVisionBlock(nn.Module):
    r"""
    This corresponds to the block module of original the EfficientNet vision encoder implementation.

    Args:
        config ([`AlignVisionConfig`]):
            Model configuration class.
        in_dim (`int`):
            Number of input channels.
        out_dim (`int`):
            Number of output channels.
        stride (`int`):
            Stride size to be used in convolution layers.
        expand_ratio (`int`):
            Expand ratio to set the output dimensions for the expansion and squeeze-excite layers.
        kernel_size (`int`):
            Kernel size for the depthwise convolution layer.
        drop_rate (`float`):
            Dropout rate to be used in the final phase of each block.
        id_skip (`bool`):
            Whether to apply dropout and sum the final hidden states with the input embeddings during the final phase
            of each block. Set to `True` for the first block of each stage.
        adjust_padding (`bool`):
            Whether to apply padding to only right and bottom side of the input kernel before the depthwise convolution
            operation, set to `True` for inputs with odd input sizes.
    """

    def __init__(
        self,
        config: AlignVisionConfig,
        in_dim: int,
        out_dim: int,
        stride: int,
        expand_ratio: int,
        kernel_size: int,
        drop_rate: float,
        id_skip: bool,
        adjust_padding: bool,
    ):
        super().__init__()
        self.expand_ratio = expand_ratio
        self.expand = self.expand_ratio != 1
        expand_in_dim = in_dim * expand_ratio

        if self.expand:
            self.expansion = AlignVisionExpansionLayer(
                config=config, in_dim=in_dim, out_dim=expand_in_dim, stride=stride
            )

        self.depthwise_conv = AlignVisionDepthwiseLayer(
            config=config,
            in_dim=expand_in_dim if self.expand else in_dim,
            stride=stride,
            kernel_size=kernel_size,
            adjust_padding=adjust_padding,
        )
        self.squeeze_excite = AlignVisionSqueezeExciteLayer(
            config=config, in_dim=in_dim, expand_dim=expand_in_dim, expand=self.expand
        )
        self.projection = AlignVisionFinalBlockLayer(
            config=config,
            in_dim=expand_in_dim if self.expand else in_dim,
            out_dim=out_dim,
            stride=stride,
            drop_rate=drop_rate,
            id_skip=id_skip,
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        embeddings = hidden_states
        # Expansion and depthwise convolution phase
        if self.expand_ratio != 1:
            hidden_states = self.expansion(hidden_states)
        hidden_states = self.depthwise_conv(hidden_states)

        # Squeeze and excite phase
        hidden_states = self.squeeze_excite(hidden_states)
        hidden_states = self.projection(embeddings, hidden_states)
        return hidden_states


class AlignVisionEncoder(nn.Module):
    r"""
    Forward propagates the embeddings through each vision encoder (EfficientNet) block.

    Args:
        config ([`AlignVisionConfig`]):
            Model configuration class.
    """

    def __init__(self, config: AlignVisionConfig):
        super().__init__()
        self.depth_coefficient = config.depth_coefficient

        def round_repeats(repeats):
            # Round number of block repeats based on depth multiplier.
            return int(math.ceil(self.depth_coefficient * repeats))

        num_base_blocks = len(config.in_channels)
        num_blocks = sum(round_repeats(n) for n in config.num_block_repeats)

        curr_block_num = 0
        blocks = []
        for i in range(num_base_blocks):
            in_dim = round_filters(config, config.in_channels[i])
            out_dim = round_filters(config, config.out_channels[i])
            stride = config.strides[i]
            kernel_size = config.kernel_sizes[i]
            expand_ratio = config.expand_ratios[i]

            for j in range(round_repeats(config.num_block_repeats[i])):
                id_skip = j == 0
                stride = 1 if j > 0 else stride
                in_dim = out_dim if j > 0 else in_dim
                adjust_padding = curr_block_num not in config.depthwise_padding
                drop_rate = config.drop_connect_rate * curr_block_num / num_blocks

                block = AlignVisionBlock(
                    config=config,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    stride=stride,
                    kernel_size=kernel_size,
                    expand_ratio=expand_ratio,
                    drop_rate=drop_rate,
                    id_skip=id_skip,
                    adjust_padding=adjust_padding,
                )
                blocks.append(block)
                curr_block_num += 1

        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        for block in self.blocks:
            hidden_states = block(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class AlignTextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask.view(1, -1, 1, 1)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class AlignTextSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.attention_dropout = config.attention_probs_dropout_prob
        self.scaling = self.attention_head_size**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            head_mask=head_mask,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->AlignText
class AlignTextSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AlignTextAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = AlignTextSelfAttention(config)
        self.output = AlignTextSelfOutput(config)
        self.pruned_heads = set()

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->AlignText
class AlignTextIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->AlignText
class AlignTextOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AlignTextLayer(GradientCheckpointingLayer):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = AlignTextAttention(config)
        self.intermediate = AlignTextIntermediate(config)
        self.output = AlignTextOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class AlignTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([AlignTextLayer(config) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    @can_return_tuple
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert -> AlignText
class AlignTextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@auto_docstring
class AlignPreTrainedModel(PreTrainedModel):
    config: AlignConfig
    base_model_prefix = "align"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, AlignModel):
            nn.init.xavier_uniform_(module.text_projection.weight)
            module.text_projection.bias.data.zero_()
            module.temperature.data.fill_(self.config.temperature_init_value)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@auto_docstring(
    custom_intro="""
    The text model from ALIGN without any head or projection on top.
    """
)
class AlignTextModel(AlignPreTrainedModel):
    config: AlignTextConfig
    _no_split_modules = ["AlignTextEmbeddings"]

    def __init__(self, config: AlignTextConfig, add_pooling_layer: bool = True):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
        super().__init__(config)
        self.config = config

        self.embeddings = AlignTextEmbeddings(config)
        self.encoder = AlignTextEncoder(config)

        self.pooler = AlignTextPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPooling]:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, AlignTextModel

        >>> model = AlignTextModel.from_pretrained("kakaobrain/align-base")
        >>> tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The vision model from ALIGN without any head or projection on top.
    """
)
class AlignVisionModel(AlignPreTrainedModel):
    config: AlignVisionConfig
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def __init__(self, config: AlignVisionConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = AlignVisionEmbeddings(config)
        self.encoder = AlignVisionEncoder(config)

        # Final pooling layer
        if config.pooling_type == "mean":
            self.pooler = nn.AvgPool2d(config.hidden_dim, ceil_mode=True)
        elif config.pooling_type == "max":
            self.pooler = nn.MaxPool2d(config.hidden_dim, ceil_mode=True)
        else:
            raise ValueError(f"config.pooling must be one of ['mean', 'max'] got {config.pooling}")

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.convolution

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithPoolingAndNoAttention]:
        r"""
        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AlignVisionModel

        >>> model = AlignVisionModel.from_pretrained("kakaobrain/align-base")
        >>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        # Apply pooling
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state)
        # Reshape (batch_size, projection_dim, 1 , 1) -> (batch_size, projection_dim)
        pooled_output = pooled_output.reshape(pooled_output.shape[:2])

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@auto_docstring
class AlignModel(AlignPreTrainedModel):
    config: AlignConfig

    def __init__(self, config: AlignConfig):
        super().__init__(config)

        if not isinstance(config.text_config, AlignTextConfig):
            raise TypeError(
                "config.text_config is expected to be of type AlignTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, AlignVisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type AlignVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size

        self.text_model = AlignTextModel(text_config)
        self.vision_model = AlignVisionModel(vision_config)

        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim)
        self.temperature = nn.Parameter(torch.tensor(self.config.temperature_init_value))

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`AlignTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, AlignModel

        >>> model = AlignModel.from_pretrained("kakaobrain/align-base")
        >>> tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use ALIGN model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = text_outputs[0][:, 0, :]
        text_features = self.text_projection(last_hidden_state)

        return text_features

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`AlignVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AlignModel

        >>> model = AlignModel.from_pretrained("kakaobrain/align-base")
        >>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use ALIGN model's config for some fields (if specified) instead of those of vision & text components.
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_features = vision_outputs[1]  # pooled_output

        return image_features

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, AlignOutput]:
        r"""
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AlignModel

        >>> model = AlignModel.from_pretrained("kakaobrain/align-base")
        >>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     images=image, text=["a photo of a cat", "a photo of a dog"], return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use ALIGN model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[0][:, 0, :]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) / self.temperature
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = align_loss(logits_per_text)

        return AlignOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


__all__ = ["AlignPreTrainedModel", "AlignTextModel", "AlignVisionModel", "AlignModel"]
