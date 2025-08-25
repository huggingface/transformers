# coding=utf-8
# Copyright 2024 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch KOSMOS-2.5 model."""

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    CausalLMOutputWithCrossAttentions,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    ModelOutput,
    TransformersKwargs,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
)
from .configuration_kosmos2_5 import (
    Kosmos2_5Config,
    Kosmos2_5TextConfig,
    Kosmos2_5VisionConfig,
)


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from ...integrations.flex_attention import make_flex_block_causal_mask


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = Kosmos2_5Config


# Copied from transformers.models.kosmos2.modeling_kosmos2._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Copied from transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


KOSMOS2_5_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Kosmos2_5Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

KOSMOS2_5_VISION_INPUTS_DOCSTRING = r"""
    Args:
        flattened_patches (`torch.FloatTensor` of shape `(batch_size, max_patches, 2 + patch_height * patch_width * image_channels)`):
            Flattened patches of the images. `flattened_patches` can be obtained using [`AutoImageProcessor`]. See
            [`Kosmos2_5ImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
"""

KOSMOS2_5_TEXT_INPUTS_DOCSTRING = r"""
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
        image_embeds: (`torch.FloatTensor` of shape `(batch_size, latent_query_num, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of `Kosmos2ImageToTextProjection`.
        image_embeds_position_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to indicate the location in a sequence to insert the image features . Mask values selected in `[0,
            1]`:

            - 1 for places where to put the image features,
            - 0 for places that are not for image features (i.e. for text tokens).

        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
"""

KOSMOS2_5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        flattened_patches (`torch.FloatTensor` of shape `(batch_size, max_patches, 2 + patch_height * patch_width * image_channels)`):
            Flattened patches of the images. `flattened_patches` can be obtained using [`AutoImageProcessor`]. See
            [`Kosmos2_5ImageProcessor.__call__`] for details.
        width (`torch.FloatTensor` of shape `(batch_size,)`):
            The original width (before resizing) of each image in the batch. This can be obtained using
            [`AutoImageProcessor`]. See [`Kosmos2_5ImageProcessor.__call__`] for details.
        height (`torch.FloatTensor` of shape `(batch_size,)`):
            The original height (before resizing) of each image in the batch. This can be obtained using
            [`AutoImageProcessor`]. See [`Kosmos2_5ImageProcessor.__call__`] for details.
        image_embeds_position_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to indicate the location in a sequence to insert the image features . Mask values selected in `[0,
            1]`:

            - 1 for places where to put the image features,
            - 0 for places that are not for image features (i.e. for text tokens).

        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        image_embeds: (`torch.FloatTensor` of shape `(batch_size, latent_query_num, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of `Kosmos2ImageToTextProjection`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
"""


@dataclass
class Kosmos2_5ModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
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
        width (`torch.FloatTensor` of shape `(batch_size,)`):
            The original width (before resizing) of each image in the batch.
        height (`torch.FloatTensor` of shape `(batch_size,)`):
            The original height (before resizing) of each image in the batch.
        image_embeds (`torch.FloatTensor` of shape `(batch_size, latent_query_num, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of `Kosmos2ImageToTextProjection`.
        projection_attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights given by `Kosmos2ImageToTextProjection`, after the attention softmax, used to compute
            the weighted average in the self-attention heads.
        vision_model_output(`BaseModelOutputWithPooling`, *optional*):
            The output of the [`Kosmos2VisionModel`].
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    width: Optional[torch.FloatTensor] = None
    height: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    projection_attentions: Optional[tuple[torch.FloatTensor]] = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(
            (self[k] if k not in ["vision_model_output"] else getattr(self, k).to_tuple()) for k in self.keys()
        )


@dataclass
class Kosmos2_5ForConditionalGenerationModelOutput(ModelOutput):
    """
    Model output class for `Kosmos2_5ForConditionalGeneration`.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        width (`torch.FloatTensor` of shape `(batch_size,)`):
            The original width (before resizing) of each image in the batch.
        height (`torch.FloatTensor` of shape `(batch_size,)`):
            The original height (before resizing) of each image in the batch.
        image_embeds (`torch.FloatTensor` of shape `(batch_size, latent_query_num, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of `Kosmos2ImageToTextProjection`.
        projection_attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights given by `Kosmos2ImageToTextProjection`, after the attention softmax, used to compute
            the weighted average in the self-attention heads.
        vision_model_output(`BaseModelOutputWithPooling`, *optional*):
            The output of the [`Kosmos2VisionModel`].
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    width: Optional[torch.FloatTensor] = None
    height: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    projection_attentions: Optional[tuple[torch.FloatTensor]] = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(
            (self[k] if k not in ["vision_model_output"] else getattr(self, k).to_tuple()) for k in self.keys()
        )


# Copied from transformers.models.pix2struct.modeling_pix2struct.Pix2StructLayerNorm with Pix2Struct->Kosmos2_5
class Kosmos2_5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://huggingface.co/papers/1910.07467 thus variance is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


# similar to transformers.models.pix2struct.modeling_pix2struct.Pix2StructVisionEmbeddings but with `inplace=False`
# TODO: check with krip
class Kosmos2_5VisionEmbeddings(nn.Module):
    def __init__(self, config: Kosmos2_5VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_projection = nn.Linear(config.patch_embed_hidden_size, config.hidden_size)

        self.row_embedder = nn.Embedding(config.max_num_patches, config.hidden_size)
        self.column_embedder = nn.Embedding(config.max_num_patches, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout_rate, inplace=False)

    def forward(self, flattened_patches: torch.Tensor) -> torch.Tensor:
        # the row and column indices are stored in the first and second position of the flattened_patches
        # flattened_patches: `batch_size`, `seq_len`, `hidden_size` + 2
        row_indices = flattened_patches[:, :, 0].long()
        col_indices = flattened_patches[:, :, 1].long()

        flattened_patches = flattened_patches[:, :, 2:]

        embeddings = self.patch_projection(flattened_patches)
        row_embeddings = self.row_embedder(row_indices).to(embeddings.device)
        col_embeddings = self.column_embedder(col_indices).to(embeddings.device)

        # sum all embeddings together
        embeddings = embeddings + row_embeddings + col_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


# Copied from transformers.models.t5.modeling_t5.T5DenseGatedActDense with T5DenseGatedActDense->Pix2StructVisionMlp,T5Config->Pix2StructVisionConfig,config.d_model->config.hidden_size,dropout_rate->dropout_rate
class Kosmos2_5VisionMlp(nn.Module):
    def __init__(self, config: Kosmos2_5VisionConfig):
        super().__init__()
        self.wi_0 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.wi_1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

        # Ignore copy
        self.config = config

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # this weight maybe overflow with fp16
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Kosmos2_5VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.n_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.inner_dim = self.n_heads * self.head_dim
        self.is_causal = False
        self.scaling = self.head_dim**-0.5

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.query = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.key = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.value = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.output = nn.Linear(self.inner_dim, self.hidden_size, bias=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        """
        Self-attention block
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.output(attn_output)

        return attn_output, attn_weights


class Kosmos2_5VisionLayer(GradientCheckpointingLayer):
    def __init__(self, config: Kosmos2_5VisionConfig) -> None:
        super().__init__()
        self.config = config

        self.attention = Kosmos2_5VisionAttention(config)
        self.mlp = Kosmos2_5VisionMlp(config)
        self.pre_mlp_layer_norm = Kosmos2_5LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pre_attention_layer_norm = Kosmos2_5LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
        residual = hidden_states

        # in  Kosmos2_5Vision, layernorm is applied before self-attention
        hidden_states = self.pre_attention_layer_norm(hidden_states)

        attention_output, self_attn_weights = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs,
        )

        # first residual connection
        hidden_states = attention_output + residual

        # in  Kosmos2_5Vision, layernorm is also applied after self-attention
        layer_output = self.pre_mlp_layer_norm(hidden_states)
        layer_output = self.mlp(layer_output) + hidden_states  # second residual connection

        outputs = (layer_output,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


# Adapted from transformers.models.pix2struct.modeling_pix2struct.Pix2StructVisionEncoder with Pix2Struct->Kosmos2_5
class Kosmos2_5VisionEncoder(nn.Module):
    def __init__(self, config: Kosmos2_5VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Kosmos2_5VisionLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def _prepare_attention_mask(self, attention_mask, input_shape, inputs_embeds):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
        return expanded_attn_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        attention_mask = self._prepare_attention_mask(attention_mask, hidden_states.shape[:2], hidden_states)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, output_attentions, **kwargs)
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


# Copied from transformers.models.kosmos2.modeling_kosmos2.Kosmos2TextSinusoidalPositionalEmbedding with Kosmos2->Kosmos2_5
class Kosmos2_5TextSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    # Copied from transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding.__init__
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    # Copied from transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding.make_weights
    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype and device of the param
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        self.register_buffer("weights", emb_weights, persistent=False)

    @staticmethod
    # Copied from transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding.get_embedding
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ):
        if input_ids is not None:
            bsz, seq_len = input_ids.size()
            if position_ids is None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                ).to(input_ids.device)
        else:
            bsz, seq_len = inputs_embeds.size()[:-1]
            if position_ids is None:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length)

        # expand embeddings if needed
        max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()

    # Copied from transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding.create_position_ids_from_inputs_embeds
    def create_position_ids_from_inputs_embeds(self, inputs_embeds, past_key_values_length):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape).contiguous() + past_key_values_length


# Copied from transformers.models.kosmos2.modeling_kosmos2.Kosmos2TextFFN with Kosmos2->Kosmos2_5
class Kosmos2_5TextFFN(nn.Module):
    def __init__(self, config: Kosmos2_5TextConfig):
        super().__init__()

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.fc1 = nn.Linear(config.embed_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, config.embed_dim)

        self.ffn_layernorm = nn.LayerNorm(config.ffn_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.ffn_layernorm(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        return hidden_states


class Kosmos2_5TextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal=True,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.is_causal = is_causal

    def forward(
        self,
        hidden_states: torch.Tensor,  # text part
        encoder_hidden_states: Optional[torch.Tensor] = None,  # image part
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # use encoder_hidden_states if cross attention
        is_cross_attention = encoder_hidden_states is not None
        current_states = encoder_hidden_states if is_cross_attention else hidden_states

        current_input_shape = current_states.shape[:-1]
        current_hidden_shape = (*current_input_shape, -1, self.head_dim)

        key_states = self.k_proj(current_states).view(current_hidden_shape).transpose(1, 2)
        value_states = self.v_proj(current_states).view(current_hidden_shape).transpose(1, 2)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply `self.scaling`
        query_states = self.scaling * query_states

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,  # We don't use `self.scaling` as it's already applied to `query_states` above .
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class Kosmos2_5TextBlock(GradientCheckpointingLayer):
    def __init__(self, config: Kosmos2_5TextConfig, layer_idx: int):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.layer_idx = layer_idx
        self.self_attn = Kosmos2_5TextAttention(
            config,
            embed_dim=self.embed_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            layer_idx=layer_idx,
        )
        self.dropout = config.dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.ffn = Kosmos2_5TextFFN(config)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Adapted from transformers.models.kosmos2.modeling_kosmos2.Kosmos2TextBlock.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


# Adapted from transformers.models.kosmos2.modeling_kosmos2.Kosmos2TextTransformer with Kosmos2->Kosmos2_5
class Kosmos2_5TextTransformer(nn.Module):
    """
    Transformer decoder consisting of `config.layers` layers. Each layer is a [`Kosmos2_5TextBlock`].
    Here we doesn't have cross attention.
    Args:
        config: Kosmos2_5TextConfig
    """

    def __init__(self, config: Kosmos2_5TextConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop

        self.embed_scale = math.sqrt(config.embed_dim) if config.scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id)

        self.embed_positions = Kosmos2_5TextSinusoidalPositionalEmbedding(
            num_positions=config.max_position_embeddings,
            embedding_dim=config.embed_dim,
            padding_idx=config.pad_token_id,
        )

        # Ignore copy
        self.segment_emb = nn.Embedding(2, config.embed_dim)
        self.layers = nn.ModuleList([Kosmos2_5TextBlock(config, layer_idx) for layer_idx in range(config.layers)])
        self.layer_norm = nn.LayerNorm(config.embed_dim, config.layer_norm_eps)
        self.gradient_checkpointing = False

    # TODO (ydshieh): Remove this (to match Llama's code)
    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # TODO (ydshieh): Remove this (to match Llama's code)
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        image_embeds_position_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # The argument `inputs_embeds` should be the one without being multiplied by `self.embed_scale`.
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Ignore copy
        if image_embeds is not None:
            inputs_embeds[image_embeds_position_mask == 1] = image_embeds.to(inputs_embeds.device).view(
                -1, image_embeds.size(-1)
            )

        inputs_embeds = inputs_embeds * self.embed_scale

        # embed positions
        positions = self.embed_positions(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=0,
            position_ids=position_ids,
        )
        positions = positions.to(inputs_embeds.device)

        # Ignore copy
        if image_embeds_position_mask is not None:
            # make every not equal 0 be 1
            image_embeds_position_mask = image_embeds_position_mask.ne(0).long()
            segment_embeds = self.segment_emb(image_embeds_position_mask).to(positions.device)
            positions += segment_embeds
        else:
            # add zero embedding for padding tokens
            bsz, seq_len, dim = positions.size()
            zero_emb = self.segment_emb(
                torch.zeros((bsz, 1), dtype=torch.long, device=self.segment_emb.weight.device)
            ).to(positions.device)
            positions += zero_emb

        hidden_states = inputs_embeds + positions

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # add final layer norm
        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output


class Kosmos2_5ImageToTextProjection(nn.Module):
    """The layer that transforms the image model's output to part of the text model's input (namely, image features)"""

    def __init__(self, config: Kosmos2_5Config):
        super().__init__()
        self.dense = nn.Linear(config.vision_config.hidden_size, config.text_config.embed_dim)
        self.latent_query = nn.Parameter(torch.randn(config.latent_query_num, config.text_config.embed_dim))

        # Ignore copy
        self.x_attn = Kosmos2_5TextAttention(
            config.text_config,
            config.text_config.embed_dim,
            config.text_config.attention_heads,
            dropout=config.text_config.attention_dropout,
            is_decoder=False,
            is_causal=False,
        )

    def forward(self, features):
        hidden_states = self.dense(features)

        # shape = [batch, latent_query_num, h_dim]
        latent_query = self.latent_query.unsqueeze(0).expand(hidden_states.size(0), -1, -1)
        key_value_states = torch.cat([hidden_states, latent_query], dim=1)

        hidden_states, attn_weights = self.x_attn(
            hidden_states=latent_query,
            encoder_hidden_states=key_value_states,
            past_key_value=None,
            attention_mask=None,
            output_attentions=None,
            is_causal=False,
        )

        return hidden_states, attn_weights


class Kosmos2_5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Kosmos2_5Config
    supports_gradient_checkpointing = True
    _no_split_modules = ["Kosmos2_5VisionLayer", "Kosmos2_5TextBlock"]
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    _supports_sdpa = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(self, Kosmos2_5VisionModel):
            init_factor = self.config.initializer_factor
            std = self.config.initializer_range * init_factor
        elif isinstance(self, (Kosmos2_5TextModel, Kosmos2_5TextForCausalLM)):
            std = self.config.init_std
        elif isinstance(self, (Kosmos2_5Model, Kosmos2_5ForConditionalGeneration)):
            std = self.config.text_config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, Kosmos2_5LayerNorm)):
            module.weight.data.fill_(1.0)
            if getattr(module, "bias", None) is not None:
                module.bias.data.zero_()
        elif isinstance(module, Kosmos2_5ImageToTextProjection):
            module.latent_query.data.normal_(mean=0.0, std=1.0)


class Kosmos2_5VisionModel(Kosmos2_5PreTrainedModel):
    config_class = Kosmos2_5VisionConfig

    # Copied from transformers.models.pix2struct.modeling_pix2struct.Pix2StructVisionModel.__init__ with Pix2Struct->Kosmos2_5
    def __init__(self, config: Kosmos2_5VisionConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = Kosmos2_5VisionEmbeddings(config)
        self.encoder = Kosmos2_5VisionEncoder(config)

        self.layernorm = Kosmos2_5LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.pix2struct.modeling_pix2struct.Pix2StructVisionModel.get_input_embeddings
    def get_input_embeddings(self):
        return self.embeddings.patch_projection

    # Copied from transformers.models.pix2struct.modeling_pix2struct.Pix2StructVisionModel._prune_heads
    def _prune_heads(self, heads_to_prune: dict[int, list[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Similar to transformers.models.pix2struct.modeling_pix2struct.Pix2StructVisionModel.forward without docstring
    def forward(
        self,
        flattened_patches: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if flattened_patches is None:
            raise ValueError("You have to specify flattened_patches")

        if attention_mask is None:
            # check where `flattened_patches` is not 0
            attention_mask = (flattened_patches.sum(dim=-1) != 0).float()

        embedding_output = self.embeddings(flattened_patches)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# Adapted from transformers.models.kosmos2.modeling_kosmos2.Kosmos2TextModel with KOSMOS2->KOSMOS2_5
class Kosmos2_5TextModel(Kosmos2_5PreTrainedModel):
    config_class = Kosmos2_5TextConfig

    def __init__(self, config: Kosmos2_5TextConfig):
        super().__init__(config)
        self.model = Kosmos2_5TextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(KOSMOS2_5_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=Kosmos2_5TextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        image_embeds_position_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        r"""
        Returns:

        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            image_embeds_position_mask=image_embeds_position_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )


@add_start_docstrings(
    """
    KOSMOS-2.5 Model for generating text and image features. The model consists of a vision encoder and a language model.
    """,
    KOSMOS2_5_START_DOCSTRING,
)
class Kosmos2_5Model(Kosmos2_5PreTrainedModel):
    config_class = Kosmos2_5Config

    def __init__(self, config: Kosmos2_5Config):
        super().__init__(config)

        self.text_model = Kosmos2_5TextModel._from_config(config.text_config)
        self.vision_model = Kosmos2_5VisionModel._from_config(config.vision_config)
        self.image_to_text_projection = Kosmos2_5ImageToTextProjection(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.text_model.model.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(KOSMOS2_5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Kosmos2_5ModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        flattened_patches: Optional[torch.Tensor] = None,
        width: Optional[torch.Tensor] = None,
        height: Optional[torch.Tensor] = None,
        image_embeds_position_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        image_embeds: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Kosmos2_5ModelOutput:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Kosmos2_5Model

        >>> model = Kosmos2_5Model.from_pretrained("microsoft/kosmos2.5")
        >>> processor = AutoProcessor.from_pretrained("microsoft/kosmos2.5")

        >>> url = "https://huggingface.co/microsoft/kosmos2.5/resolve/main/snowman.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = (
        ...     "<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863>"
        ...     "</object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911>"
        ...     "</object>"
        ... )

        >>> inputs = processor(text=text, images=image, return_tensors="pt", add_eos_token=True)

        >>> last_hidden_state = model(
        ...     pixel_values=inputs["pixel_values"],
        ...     input_ids=inputs["input_ids"],
        ...     attention_mask=inputs["attention_mask"],
        ...     image_embeds_position_mask=inputs["image_embeds_position_mask"],
        ... ).last_hidden_state
        >>> list(last_hidden_state.shape)
        [1, 91, 2048]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_model_output = None
        projection_attentions = None
        if image_embeds is None:
            if flattened_patches is not None:
                vision_model_output = self.vision_model(
                    flattened_patches=flattened_patches,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    **kwargs,
                )
                # normalized features
                image_embeds = nn.functional.normalize(vision_model_output.last_hidden_state, dim=-1)
                image_embeds, projection_attentions = self.image_to_text_projection(image_embeds)

        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            image_embeds_position_mask=image_embeds_position_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        return Kosmos2_5ModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            width=width,
            height=height,
            image_embeds=image_embeds,
            projection_attentions=projection_attentions,
            vision_model_output=vision_model_output,
        )


@add_start_docstrings(
    """
    The text model from KOSMOS-2.5 with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    KOSMOS2_5_START_DOCSTRING,
)
class Kosmos2_5TextForCausalLM(Kosmos2_5PreTrainedModel):
    config_class = Kosmos2_5TextConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Kosmos2_5TextConfig):
        super().__init__(config)

        self.model = Kosmos2_5TextTransformer(config)
        self.lm_head = nn.Linear(in_features=config.embed_dim, out_features=config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(KOSMOS2_5_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=Kosmos2_5TextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        image_embeds_position_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithCrossAttentions:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        """
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            image_embeds_position_mask=image_embeds_position_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        lm_logits = self.lm_head(outputs.last_hidden_state)

        lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            lm_loss = self.loss_function(
                lm_logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        image_embeds=None,
        image_embeds_position_mask=None,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        cache_position=None,
        position_ids=None,
        **model_kwargs,
    ):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        position_ids = None

        # cut input_ids if past_key_values is used
        if past_key_values is not None:
            position_ids = create_position_ids_from_input_ids(
                input_ids,
                padding_idx=self.config.pad_token_id,
                past_key_values_length=0,
            )[:, -cache_position.shape[0] :]

            input_ids = input_ids[:, -cache_position.shape[0] :]
            # the image info. is already encoded into the past keys/values
            if past_key_values.get_seq_length() > 0:
                image_embeds = None
                image_embeds_position_mask = None
        elif image_embeds_position_mask is not None:
            # appending `False` to `image_embeds_position_mask` (because `input_ids` grows during generation)
            batch_size, seq_len = input_ids.size()
            mask_len = image_embeds_position_mask.size()[-1]
            image_embeds_position_mask = torch.cat(
                (
                    image_embeds_position_mask,
                    torch.zeros(size=(batch_size, seq_len - mask_len), dtype=torch.bool, device=input_ids.device),
                ),
                dim=1,
            )

        return {
            "input_ids": input_ids,
            "image_embeds": image_embeds,
            "image_embeds_position_mask": image_embeds_position_mask,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": use_cache,
        }


@add_start_docstrings(
    """
    KOSMOS-2.5 Model for generating text and bounding boxes given an image. The model consists of a vision encoder and a
    language model.
    """,
    KOSMOS2_5_START_DOCSTRING,
)
class Kosmos2_5ForConditionalGeneration(Kosmos2_5PreTrainedModel, GenerationMixin):
    config_class = Kosmos2_5Config
    _tied_weights_keys = ["text_model.lm_head.weight"]

    def __init__(self, config: Kosmos2_5Config):
        super().__init__(config)
        self.text_model = Kosmos2_5TextForCausalLM(config.text_config)
        self.vision_model = Kosmos2_5VisionModel(config.vision_config)
        self.image_to_text_projection = Kosmos2_5ImageToTextProjection(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.text_model.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.text_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.text_model.set_output_embeddings(new_embeddings)

    @can_return_tuple
    @add_start_docstrings_to_model_forward(KOSMOS2_5_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=Kosmos2_5ForConditionalGenerationModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        flattened_patches: Optional[torch.Tensor] = None,
        width: Optional[torch.Tensor] = None,
        height: Optional[torch.Tensor] = None,
        image_embeds_position_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        image_embeds: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Kosmos2_5ForConditionalGenerationModelOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> import torch
        >>> from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration

        >>> repo = "ydshieh/kosmos-2.5"
        >>> device = "cuda:0"
        >>> dtype = torch.bfloat16 # torch.float16
        >>> model = Kosmos2_5ForConditionalGeneration.from_pretrained(repo, device_map=device, dtype=dtype)
        >>> processor = AutoProcessor.from_pretrained(repo)

        >>> url = "https://huggingface.co/ydshieh/kosmos-2.5/resolve/main/receipt_00008.png"

        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> prompt = "<ocr>" # <md>

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")
        >>> height, width = inputs.pop("height"), inputs.pop("width")
        >>> inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
        >>> inputs["flattened_patches"] = inputs["flattened_patches"].to(dtype)

        >>> generated_ids = model.generate(**inputs,max_new_tokens=1024)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> generated_text
        '<ocr><bbox><x_53><y_573><x_69><y_606></bbox>1\n<bbox><x_79><y_573><x_464><y_612></bbox>[REG] BLACK SAKURA\n<bbox><x_690><y_569><x_810><y_606></bbox>45,455\n<bbox><x_53><y_614><x_69><y_648></bbox>1\n<bbox><x_79><y_614><x_468><y_650></bbox>COOKIE DOH SAUCES\n<bbox><x_788><y_609><x_812><y_644></bbox>0\n<bbox><x_50><y_658><x_69><y_693></bbox>1\n<bbox><x_79><y_658><x_358><y_693></bbox>NATA DE COCO\n<bbox><x_790><y_652><x_814><y_687></bbox>0\n<bbox><x_31><y_742><x_820><y_781></bbox>Sub Total 45,455\n<bbox><x_27><y_781><x_822><y_827></bbox>PB1 (10%) 4,545\n<bbox><x_27><y_826><x_824><y_872></bbox>Rounding 0\n<bbox><x_24><y_872><x_827><y_921></bbox>Total 50,000\n<bbox><x_17><y_1056><x_836><y_1108></bbox>Card Payment 50,000\n'
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_model_output = None
        projection_attentions = None

        if image_embeds is None:
            if flattened_patches is not None:
                vision_model_output = self.vision_model(
                    flattened_patches=flattened_patches,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    **kwargs,
                )
                image_embeds = nn.functional.normalize(vision_model_output.last_hidden_state, dim=-1)
                image_embeds, projection_attentions = self.image_to_text_projection(image_embeds)

        lm_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            image_embeds_position_mask=image_embeds_position_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        return Kosmos2_5ForConditionalGenerationModelOutput(
            loss=lm_outputs.loss,
            logits=lm_outputs.logits,
            past_key_values=lm_outputs.past_key_values,
            hidden_states=lm_outputs.hidden_states,
            attentions=lm_outputs.attentions,
            width=width,
            height=height,
            image_embeds=image_embeds,
            projection_attentions=projection_attentions,
            vision_model_output=vision_model_output,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        flattened_patches=None,
        image_embeds=None,
        image_embeds_position_mask=None,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        cache_position=None,
        position_ids=None,
        **model_kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = self.text_model.prepare_inputs_for_generation(
            input_ids,
            image_embeds=image_embeds,
            image_embeds_position_mask=image_embeds_position_mask,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            cache_position=cache_position,
            position_ids=position_ids,
            **model_kwargs,
        )

        if cache_position[0] == 0:
            # If we're in cached decoding stage, `flattened_patches` should be `None` because `input_ids` do not contain special image token anymore
            # Otherwise we need `flattened_patches` to be passed to model
            model_inputs["flattened_patches"] = flattened_patches

        return model_inputs


__all__ = [
    "Kosmos2_5ForConditionalGeneration",
    "Kosmos2_5Model",
    "Kosmos2_5PreTrainedModel",
]
