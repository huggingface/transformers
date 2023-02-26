# coding=utf-8
# Copyright 2023 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
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
""" PyTorch ALIGN model."""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPoolingAndNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...models.bert import BertModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_align import ALIGNConfig, ALIGNTextConfig, ALIGNVisionConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "kakaobrain/align-base"
_CONFIG_FOR_DOC = "ALIGNConfig"
_IMAGE_PROCESSOR_FOR_DOC = "EfficientNetImageProcessor"
_TOKENIZER_FOR_DOC = "BertTokenizer"
_PROCESSOR_FOR_DOC = "ALIGNProcessor"

ALIGN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "kakaobrain/align-base",
    # See all ALIGN models at https://huggingface.co/models?filter=align
]


ALIGN_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ALIGNConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ALIGN_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

ALIGN_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

ALIGN_INPUTS_DOCSTRING = r"""
    Args:
       input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@dataclass
class ALIGNVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ALIGNTextModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ALIGNOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`ALIGNTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The output of [`ALIGNVisionModel`].
        text_model_output(`BaseModelOutputWithPoolingAndCrossAttentions`):
            The output of the [`ALIGNTextModel`].
        vision_model_output(`BaseModelOutputWithPoolingAndNoAttention`):
            The output of the [`ALIGNVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPoolingAndCrossAttentions = None
    vision_model_output: BaseModelOutputWithPoolingAndNoAttention = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/ALIGN.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device), label_smoothing=0.1)


def align_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def round_filters(config: ALIGNVisionConfig, num_channels: int):
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


def correct_pad(kernel_size: Union[int, Tuple], adjust: bool = True):
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


class ALIGNVisionEmbeddings(nn.Module):
    r"""
    A module that corresponds to the stem module of the EfficientNet Vision Encoder.
    """

    def __init__(self, config: ALIGNVisionConfig):
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


class ALIGNDepthwiseConv2d(nn.Conv2d):
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


class ALIGNExpansionLayer(nn.Module):
    r"""
    This corresponds to the expansion phase of each block in the original implementation.
    """

    def __init__(
        self,
        config: ALIGNVisionConfig,
        in_dim: int,
        out_dim: int,
        stride: int,
    ):
        super().__init__()
        self.expand_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            padding="same",
            bias=False,
        )
        self.expand_bn = nn.BatchNorm2d(num_features=out_dim)
        self.expand_act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # Expand phase
        hidden_states = self.expand_conv(hidden_states)
        hidden_states = self.expand_bn(hidden_states)
        hidden_states = self.expand_act(hidden_states)

        return hidden_states


class ALIGNDepthwiseLayer(nn.Module):
    r"""
    This corresponds to the depthwise convolution phase of each block in the original implementation.
    """

    def __init__(
        self,
        config: ALIGNVisionConfig,
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
        self.depthwise_conv = ALIGNDepthwiseConv2d(
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


class ALIGNSqueezeExciteLayer(nn.Module):
    r"""
    This corresponds to the Squeeze and Excitement phase of each block in the original implementation.
    """

    def __init__(self, config: ALIGNVisionConfig, in_dim: int, expand_dim: int, expand: bool = False):
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


class ALIGNFinalBlockLayer(nn.Module):
    r"""
    This corresponds to the final phase of each block in the original implementation.
    """

    def __init__(
        self, config: ALIGNVisionConfig, in_dim: int, out_dim: int, stride: int, drop_rate: float, id_skip: bool
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


class ALIGNVisionBlock(nn.Module):
    r"""
    This corresponds to the expansion and depthwise convolution phase of each block in the original implementation.

    Args:
        config ([`ALIGNVisionConfig`]):
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
        config: ALIGNVisionConfig,
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
        self.expand = True if self.expand_ratio != 1 else False
        expand_in_dim = in_dim * expand_ratio

        if self.expand:
            self.expansion = ALIGNExpansionLayer(config=config, in_dim=in_dim, out_dim=expand_in_dim, stride=stride)

        self.depthwise_conv = ALIGNDepthwiseLayer(
            config=config,
            in_dim=expand_in_dim if self.expand else in_dim,
            stride=stride,
            kernel_size=kernel_size,
            adjust_padding=adjust_padding,
        )
        self.squeeze_excite = ALIGNSqueezeExciteLayer(
            config=config, in_dim=in_dim, expand_dim=expand_in_dim, expand=self.expand
        )
        self.projection = ALIGNFinalBlockLayer(
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


class ALIGNVisionEncoder(nn.Module):
    r"""
    Forward propogates the embeddings through each vision encoder (EfficientNet) block.

    Args:
        config ([`ALIGNVisionConfig`]):
            Model configuration class.
    """

    def __init__(self, config: ALIGNVisionConfig):
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
                id_skip = True if j == 0 else False
                stride = 1 if j > 0 else stride
                in_dim = out_dim if j > 0 else in_dim
                adjust_padding = False if curr_block_num in config.depthwise_padding else True
                drop_rate = config.drop_connect_rate * curr_block_num / num_blocks

                block = ALIGNVisionBlock(
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


class ALIGNPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ALIGNConfig
    base_model_prefix = "align"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, ALIGNModel):
            nn.init.xavier_uniform_(module.text_projection.weight)

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (ALIGNTextModel, ALIGNVisionModel)):
            module.gradient_checkpointing = value


@add_start_docstrings(
    """The text model from ALIGN without any head or projection on top.""",
    ALIGN_START_DOCSTRING,
)
class ALIGNTextModel(ALIGNPreTrainedModel):
    config_class = ALIGNTextConfig

    def __init__(self, config: ALIGNTextConfig):
        super().__init__(config)
        self.text_model = BertModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.text_model.embeddings.word_embeddings = value

    @add_start_docstrings_to_model_forward(ALIGN_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=ALIGNTextConfig)
    def forward(
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
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ALIGNTextModel

        >>> model = ALIGNTextModel.from_pretrained("kakaobrain/align-base")
        >>> tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
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


@add_start_docstrings(
    """The vision model from ALIGN without any head or projection on top.""",
    ALIGN_START_DOCSTRING,
)
class ALIGNVisionModel(ALIGNPreTrainedModel):
    config_class = ALIGNVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: ALIGNVisionConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = ALIGNVisionEmbeddings(config)
        self.encoder = ALIGNVisionEncoder(config)

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

    @add_start_docstrings_to_model_forward(ALIGN_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndNoAttention, config_class=ALIGNVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ALIGNVisionModel

        >>> model = ALIGNVisionModel.from_pretrained("kakaobrain/align-base")
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
            return_dict=return_dict,
        )
        # Apply pooling
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state)
        # Reshape (batch_size, projection_dim, 1 , 1) -> (batch_size, projection_dim)
        pooled_output = pooled_output.reshape(pooled_output.shape[:2])

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(ALIGN_START_DOCSTRING)
class ALIGNModel(ALIGNPreTrainedModel):
    config_class = ALIGNConfig

    def __init__(self, config: ALIGNConfig):
        super().__init__(config)

        if not isinstance(config.text_config, ALIGNTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type ALIGNTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, ALIGNVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type ALIGNVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size

        self.text_model = ALIGNTextModel(text_config)
        self.vision_model = ALIGNVisionModel(vision_config)

        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim)
        self.temperature = nn.Parameter(torch.ones([]) * self.config.temperature_init_value)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ALIGN_TEXT_INPUTS_DOCSTRING)
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
            applying the projection layer to the pooled output of [`ALIGNTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ALIGNModel

        >>> model = ALIGNModel.from_pretrained("kakaobrain/align-base")
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

    @add_start_docstrings_to_model_forward(ALIGN_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`ALIGNVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ALIGNModel

        >>> model = ALIGNModel.from_pretrained("kakaobrain/align-base")
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

    @add_start_docstrings_to_model_forward(ALIGN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ALIGNOutput, config_class=ALIGNConfig)
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
    ) -> Union[Tuple, ALIGNOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ALIGNModel

        >>> model = ALIGNModel.from_pretrained("kakaobrain/align-base")
        >>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
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
            return_dict=return_dict,
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
            return_dict=return_dict,
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

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return ALIGNOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
