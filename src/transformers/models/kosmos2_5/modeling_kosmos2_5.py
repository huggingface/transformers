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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_flash_attention_utils import _flash_attention_forward
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    CausalLMOutputWithCrossAttentions,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_kosmos2_5 import (
    Kosmos2_5Config,
    Kosmos2_5TextConfig,
    Kosmos2_5VisionConfig,
)


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


# Copied from transformers.models.kosmos2.modeling_kosmos2._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


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
        flattened_patches (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`Kosmos2_5ImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
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

        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

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
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

KOSMOS2_5_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`Kosmos2_5ImageProcessor.__call__`] for details.
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
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
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
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
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
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
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_embeds: Optional[torch.FloatTensor] = None
    projection_attentions: Optional[Tuple[torch.FloatTensor]] = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            (self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple())
            for k in self.keys()
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
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_embeds: Optional[torch.FloatTensor] = None
    projection_attentions: Optional[Tuple[torch.FloatTensor]] = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            (self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple())
            for k in self.keys()
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
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


try:
    from apex.normalization import FusedRMSNorm

    Kosmos2_5LayerNorm = FusedRMSNorm  # noqa

    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of Kosmos2_5LayerNorm")
except ImportError:
    # using the normal Kosmos2_5LayerNorm
    pass
except Exception:
    logger.warning("Discovered apex but it failed to load, falling back to Kosmos2_5LayerNorm")
    pass


# similar to transformers.models.pix2struct.modeling_pix2struct.Pix2StructVisionEmbeddings but with `inplace=False`
# TODO: check with krip
class Kosmos2_5VisionEmbeddings(nn.Module):
    def __init__(self, config: Kosmos2_5VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_projection = nn.Linear(config.patch_embed_hidden_size, config.hidden_size)

        self.row_embedder = nn.Embedding(config.seq_len, config.hidden_size)
        self.column_embedder = nn.Embedding(config.seq_len, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout_rate, inplace=False)

    def forward(self, flattened_patches: torch.Tensor) -> torch.Tensor:
        # the row and column indices are stored in the first and second position of the flattened_patches
        # flattened_patches: `batch_size`, `seq_len`, `hidden_size` + 2
        row_indices = flattened_patches[:, :, 0].long()
        col_indices = flattened_patches[:, :, 1].long()

        flattened_patches = flattened_patches[:, :, 2:]

        embeddings = self.patch_projection(flattened_patches)
        row_embeddings = self.row_embedder(row_indices)
        col_embeddings = self.column_embedder(col_indices)

        # sum all embeddings together
        embeddings = embeddings + row_embeddings + col_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


# Copied from transformers.models.t5.modeling_t5.T5DenseGatedActDense with T5DenseGatedActDense->Pix2StructVisionMlp,T5Config->Pix2StructVisionConfig,config.d_model->config.hidden_size,dropout_rate->dropout_rate
class Kosmos2_5VisionMlp(nn.Module):
    def __init__(self, config: Kosmos2_5VisionConfig):
        super().__init__()
        self.wi_0 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.hidden_size, bias=False)
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


class Kosmos2_5VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.is_causal = False

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
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        """
        Self-attention block
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length, _ = hidden_states.size()

        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)

        # get query states
        # (batch_size, n_heads, seq_length, dim_per_head)
        query_states = query_states.view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.key_value_proj_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, -1)
        attn_output = self.output(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class Kosmos2_5VisionFlashAttention2(Kosmos2_5VisionAttention):
    """
    Kosmos-2.5 vision encoder flash attention module. This module inherits from `Kosmos2_5VisionAttention` as the
    weights of the module stays untouched. The only required change would be on the forward pass where it needs to
    correctly call the public API of flash attention and deal with padding tokens in case the input contains any of
    them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        """
        Flash attn Self-attention block
        """
        output_attentions = False
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        batch_size, seq_length, _ = hidden_states.size()
        # (batch_size, seq_length, inner_dim)
        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)

        # (batch_size, seq_length, self.n_heads , self.key_value_proj_dim)
        query_states = query_states.view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim)
        key_states = key_states.view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim)
        value_states = value_states.view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.query.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            seq_length,
            dropout=self.dropout,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.view(batch_size, -1, self.inner_dim)
        attn_output = self.output(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class Kosmos2_5VisionSdpaAttention(Kosmos2_5VisionAttention):
    """
    Kosmos-2.5 vision encoder attention module using torch.nn.functional.scaled_dot_product_attention. This module
    inherits from` Kosmos2_5VisionAttention` as the weights of the module stays untouched. The only changes are on the
    forward pass to adapt to SDPA API.
    """

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        if output_attentions:
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
        batch_size, seq_length, _ = hidden_states.size()

        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)

        query_states = query_states.view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        causal_mask = attention_mask
        if attention_mask is not None:
            # Slice the causal_mask to match key_states' last dimension
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and seq_length > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, -1)

        attn_output = self.output(attn_output)

        return attn_output, None


KOSMOS2_5_VISION_ATTENTION_CLASSES = {
    "eager": Kosmos2_5VisionAttention,
    "flash_attention_2": Kosmos2_5VisionFlashAttention2,
    "sdpa": Kosmos2_5VisionSdpaAttention,
}


class Kosmos2_5VisionLayer(nn.Module):
    # Copied from transformers.models.pix2struct.modeling_pix2struct.Pix2StructVisionLayer.__init__ with Pix2StructVisionAttention->KOSMOS2_5_VISION_ATTENTION_CLASSES[config._attn_implementation],Pix2Struct->Kosmos2_5
    def __init__(self, config: Kosmos2_5VisionConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        # Ignore copy
        self.config = config

        self.attention = KOSMOS2_5_VISION_ATTENTION_CLASSES[config._attn_implementation](config)
        self.mlp = Kosmos2_5VisionMlp(config)
        self.pre_mlp_layer_norm = Kosmos2_5LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pre_attention_layer_norm = Kosmos2_5LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        residual = hidden_states

        # in  Kosmos2_5Vision, layernorm is applied before self-attention
        hidden_states = self.pre_attention_layer_norm(hidden_states)

        attention_mask = self._prepare_attention_mask(attention_mask, hidden_states.shape[:2], hidden_states)

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + residual

        # in  Kosmos2_5Vision, layernorm is also applied after self-attention
        layer_output = self.pre_mlp_layer_norm(hidden_states)
        layer_output = self.mlp(layer_output) + hidden_states  # second residual connection

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.pix2struct.modeling_pix2struct.Pix2StructVisionEncoder with Pix2Struct->Kosmos2_5
class Kosmos2_5VisionEncoder(nn.Module):
    def __init__(self, config: Kosmos2_5VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Kosmos2_5VisionLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

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
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        past_key_values_length: int = 0,
        position_ids: torch.Tensor = None,
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

    # Similar to ...models.bart.modeling_bart.BartAttention.__init__ except an additional `inner_attn_ln`.
    def __init__(
        self,
        config,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        add_inner_attn_layernorm: bool = False,
        bias: bool = True,
        is_causal=True,
    ):
        super().__init__()
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

        # End opy
        self.inner_attn_ln = None
        if add_inner_attn_layernorm:
            self.inner_attn_ln = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    # Copied from transformers.models.kosmos2.modeling_kosmos2.KosmosTextAttention._shape
    def _shape(self, projection: torch.Tensor) -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.num_heads, self.head_dim)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = encoder_hidden_states is not None
        batch_size, seq_length = hidden_states.shape[:2]

        # use encoder_hidden_states if cross attention
        current_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        # checking that the `sequence_length` of the `past_key_value` is the same as the he provided
        # `encoder_hidden_states` to support prefix tuning
        if is_cross_attention and past_key_value and past_key_value[0].shape[2] == current_states.shape[1]:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        else:
            key_states = self._shape(self.k_proj(current_states))
            value_states = self._shape(self.v_proj(current_states))
            if past_key_value is not None and not is_cross_attention:
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        query_states = self._shape(self.q_proj(hidden_states) * self.scaling)
        # this weight maybe overflow with float16
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_length, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_length, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)

        if self.inner_attn_ln is not None:
            attn_output = self.inner_attn_ln(attn_output)

        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Kosmos2_5TextFlashAttention2(Kosmos2_5TextAttention):
    """
    Kosmos-2.5 text flash attention module. This module inherits from `Kosmos2_5TextAttention` as the weights of the
    module stays untouched. The only required change would be on the forward pass where it needs to correctly call the
    public API of flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False
        is_cross_attention = encoder_hidden_states is not None
        bsz, q_len, _ = hidden_states.size()

        # use encoder_hidden_states if cross attention
        current_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        # checking that the `sequence_length` of the `past_key_value` is the same as the he provided
        # `encoder_hidden_states` to support prefix tuning
        if is_cross_attention and past_key_value and past_key_value[0].shape[2] == current_states.shape[1]:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        else:
            key_states = self._shape(self.k_proj(current_states)).transpose(1, 2)
            value_states = self._shape(self.v_proj(current_states)).transpose(1, 2)
            if past_key_value is not None and not is_cross_attention:
                key_states = torch.cat([past_key_value[0], key_states], dim=1)
                value_states = torch.cat([past_key_value[1], value_states], dim=1)

        query_states = self._shape(self.q_proj(hidden_states)).transpose(1, 2)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        input_dtype = query_states.dtype

        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            None,
            q_len,
            dropout=self.dropout,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.view(bsz, -1, self.embed_dim)

        if self.inner_attn_ln is not None:
            attn_output = self.inner_attn_ln(attn_output)

        attn_output = self.out_proj(attn_output)
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Kosmos2_5TextSdpaAttention(Kosmos2_5TextAttention):
    """
    Kosmos-2.5 text decoder attention module using torch.nn.functional.scaled_dot_product_attention. This module
    inherits from `Kosmos2_5TextAttention` as the weights of the module stays untouched. The only changes are on the
    forward pass to adapt to SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            logger.warning_once(
                "Kosmos2_5TextModel is using Kosmos2_5TextSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )

        is_cross_attention = encoder_hidden_states is not None
        bsz, q_len, _ = hidden_states.size()
        # use encoder_hidden_states if cross attention
        current_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        # checking that the `sequence_length` of the `past_key_value` is the same as the he provided
        # `encoder_hidden_states` to support prefix tuning
        if is_cross_attention and past_key_value and past_key_value[0].shape[2] == current_states.shape[1]:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        else:
            key_states = self._shape(self.k_proj(current_states))
            value_states = self._shape(self.v_proj(current_states))
            if past_key_value is not None and not is_cross_attention:
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        query_states = self._shape(self.q_proj(hidden_states))

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False
        is_causal = is_causal and self.is_causal
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        if self.inner_attn_ln is not None:
            attn_output = self.inner_attn_ln(attn_output)

        attn_output = self.out_proj(attn_output)
        return attn_output, None, past_key_value


KOSMOS2_5_TEXT_ATTENTION_CLASSES = {
    "eager": Kosmos2_5TextAttention,
    "flash_attention_2": Kosmos2_5TextFlashAttention2,
    "sdpa": Kosmos2_5TextSdpaAttention,
}


class Kosmos2_5TextBlock(nn.Module):
    def __init__(self, config: Kosmos2_5TextConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.self_attn = KOSMOS2_5_TEXT_ATTENTION_CLASSES[config._attn_implementation](
            config,
            embed_dim=self.embed_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            add_inner_attn_layernorm=False,
            is_causal=True,
        )
        self.dropout = config.dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        if config.add_cross_attention:
            self.encoder_attn = KOSMOS2_5_TEXT_ATTENTION_CLASSES[config._attn_implementation](
                config,
                embed_dim=self.embed_dim,
                num_heads=config.attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                add_inner_attn_layernorm=False,
                is_causal=True,
            )
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.ffn = Kosmos2_5TextFFN(config)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Copied from transformers.models.kosmos2.modeling_kosmos2.Kosmos2TextBlock.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        hidden_states = self.self_attn_layer_norm(hidden_states)

        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            if not hasattr(self, "encoder_attn"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            residual = hidden_states

            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states

        hidden_states = self.final_layer_norm(hidden_states)

        # FFN
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# Copied from transformers.models.kosmos2.modeling_kosmos2.Kosmos2TextTransformer with Kosmos2->Kosmos2_5
class Kosmos2_5TextTransformer(nn.Module):
    """
    Transformer decoder consisting of `config.layers` layers. Each layer is a [`Kosmos2_5TextBlock`].

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

        self.layers = nn.ModuleList([Kosmos2_5TextBlock(config) for _ in range(config.layers)])
        self.layer_norm = nn.LayerNorm(config.embed_dim, config.layer_norm_eps)

        self.gradient_checkpointing = False

    # Ignore copy
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward_embedding(
        self,
        input_ids,
        inputs_embeds: torch.Tensor = None,
        image_embeds: torch.Tensor = None,
        img_input_mask: torch.Tensor = None,
        past_key_values_length: int = 0,
        position_ids: torch.Tensor = None,
    ):
        # The argument `inputs_embeds` should be the one without being multiplied by `self.embed_scale`.
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Ignore copy
        if image_embeds is not None:
            inputs_embeds[img_input_mask == 1] = image_embeds.to(inputs_embeds.device).view(-1, image_embeds.size(-1))

        inputs_embeds = inputs_embeds * self.embed_scale

        # embed positions
        positions = self.embed_positions(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            position_ids=position_ids,
        )
        positions = positions.to(inputs_embeds.device)

        # Ignore copy
        if img_input_mask is not None:
            # make every not equal 0 be 1
            img_input_mask = img_input_mask.ne(0).long()
            segment_embeds = self.segment_emb(img_input_mask)
            positions += segment_embeds
        else:
            # add zero embedding for padding tokens
            bsz, seq_len, dim = positions.size()
            zero_emb = self.segment_emb(torch.zeros((bsz, 1), dtype=torch.long, device=positions.device))
            positions += zero_emb

        hidden_states = inputs_embeds + positions

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        return hidden_states

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        image_embeds_position_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # We don't need img info. when `past_key_values_length` > 0
        if past_key_values_length > 0:
            image_embeds = None
            image_embeds_position_mask = None

        hidden_states = self.forward_embedding(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            img_input_mask=image_embeds_position_mask,
            past_key_values_length=past_key_values_length,
            position_ids=position_ids,
        )

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, hidden_states, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        present_key_value_states = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                present_key_value_states += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add final layer norm
        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.kosmos2.modeling_kosmos2.Kosmos2ImageToTextProjection with Kosmos2->Kosmos2_5
class Kosmos2_5ImageToTextProjection(nn.Module):
    """The layer that transforms the image model's output to part of the text model's input (namely, image features)"""

    def __init__(self, config: Kosmos2_5Config):
        super().__init__()
        self.dense = nn.Linear(config.vision_config.hidden_size, config.text_config.embed_dim)
        self.latent_query = nn.Parameter(torch.randn(config.latent_query_num, config.text_config.embed_dim))

        # Ignore copy
        self.x_attn = KOSMOS2_5_TEXT_ATTENTION_CLASSES[config._attn_implementation](
            config.text_config,
            config.text_config.embed_dim,
            config.text_config.attention_heads,
            dropout=config.text_config.attention_dropout,
            is_decoder=False,
            add_inner_attn_layernorm=False,
            is_causal=False,
        )

    def forward(self, features):
        hidden_states = self.dense(features)

        # shape = [batch, latent_query_num, h_dim]
        latent_query = self.latent_query.unsqueeze(0).expand(hidden_states.size(0), -1, -1)
        key_value_states = torch.cat([hidden_states, latent_query], dim=1)

        hidden_states, attn_weights, _ = self.x_attn(
            hidden_states=latent_query,
            encoder_hidden_states=key_value_states,
            past_key_value=None,
            attention_mask=None,
            output_attentions=None,
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
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(self, Kosmos2_5VisionModel):
            factor = self.config.initializer_factor
            std = self.config.initializer_range
        elif isinstance(self, (Kosmos2_5TextModel, Kosmos2_5TextForCausalLM)):
            std = self.config.init_std
        elif isinstance(self, (Kosmos2_5Model, Kosmos2_5ForConditionalGeneration)):
            factor = self.config.vision_config.initializer_factor
            std = self.config.text_config.init_std

        if isinstance(module, Kosmos2_5VisionEmbeddings):
            nn.init.normal_(module.column_embedder.weight, std=std)
            nn.init.normal_(module.row_embedder.weight, std=std)
            nn.init.normal_(module.patch_projection.weight, std=std)
            if module.patch_projection.bias is not None:
                module.patch_projection.bias.data.zero_()
        elif isinstance(module, Kosmos2_5VisionAttention):
            in_proj_std = (module.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.hidden_size**-0.5) * factor
            nn.init.normal_(module.query.weight, std=in_proj_std)
            nn.init.normal_(module.key.weight, std=in_proj_std)
            nn.init.normal_(module.value.weight, std=in_proj_std)
            nn.init.normal_(module.output.weight, std=out_proj_std)
            if module.query.bias is not None:
                module.query.bias.data.zero_()
            if module.key.bias is not None:
                module.key.bias.data.zero_()
            if module.value.bias is not None:
                module.value.bias.data.zero_()
            if module.output.bias is not None:
                module.output.bias.data.zero_()
        elif isinstance(module, Kosmos2_5VisionMlp):
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.wi_0.weight, std=fc_std)
            nn.init.normal_(module.wi_1.weight, std=in_proj_std)
            nn.init.normal_(module.wo.weight, std=fc_std)
            if module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            if module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            if module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, Kosmos2_5VisionLayer):
            module.pre_mlp_layer_norm.weight.data.fill_(1.0)
            module.pre_attention_layer_norm.weight.data.fill_(1.0)
        elif isinstance(module, Kosmos2_5TextAttention):
            nn.init.normal_(module.q_proj.weight, std=std)
            nn.init.normal_(module.k_proj.weight, std=std)
            nn.init.normal_(module.v_proj.weight, std=std)
            nn.init.normal_(module.out_proj.weight, std=std)
            if module.q_proj.bias is not None:
                module.q_proj.bias.data.zero_()
            if module.k_proj.bias is not None:
                module.k_proj.bias.data.zero_()
            if module.v_proj.bias is not None:
                module.v_proj.bias.data.zero_()
            if module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
        elif isinstance(module, Kosmos2_5TextFFN):
            nn.init.normal_(module.fc1.weight, std=std)
            nn.init.normal_(module.fc2.weight, std=std)
            if module.fc1.bias is not None:
                module.fc1.bias.data.zero_()
            if module.fc2.bias is not None:
                module.fc2.bias.data.zero_()
        elif isinstance(module, Kosmos2_5TextForCausalLM):
            nn.init.normal_(module.lm_head.weight, std=std)
            if module.lm_head.bias is not None:
                module.lm_head.bias.data.zero_()
        elif isinstance(module, Kosmos2_5ImageToTextProjection):
            nn.init.normal_(module.dense.weight, std=std)
            if module.dense.bias is not None:
                module.dense.bias.data.zero_()
        elif isinstance(module, Kosmos2_5TextTransformer):
            module.embed_tokens.weight.data.normal_(std=std)
            if module.embed_tokens.padding_idx is not None:
                module.embed_tokens.weight.data[module.embed_tokens.padding_idx].zero_()
            module.segment_emb.weight.data.normal_(std=std)


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
    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
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
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if flattened_patches is None:
            raise ValueError("You have to specify flattened_patches")

        if attention_mask is None:
            # check where `flattened_patches` is not 0
            attention_mask = (flattened_patches.sum(dim=-1) != 0).float()

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(flattened_patches)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# Copied from transformers.models.kosmos2.modeling_kosmos2.Kosmos2TextModel with KOSMOS2->KOSMOS2_5,Kosmos2->Kosmos2_5
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
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Returns:

        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            image_embeds_position_mask=image_embeds_position_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


@add_start_docstrings(
    """
    KOSMOS-2.5 Model for generating text and image features. The model consists of a vision encoder and a language model.
    """,
    KOSMOS2_5_START_DOCSTRING,
)
class Kosmos2_5Model(Kosmos2_5PreTrainedModel):
    config_class = Kosmos2_5Config
    main_input_name = "flattened_patches"

    def __init__(self, config: Kosmos2_5Config):
        super().__init__(config)

        self.text_model = Kosmos2_5TextModel._from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vision_model = Kosmos2_5VisionModel._from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        )
        self.image_to_text_projection = Kosmos2_5ImageToTextProjection(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.text_model.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(KOSMOS2_5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Kosmos2_5ModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        flattened_patches: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        image_embeds_position_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        image_embeds: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Kosmos2_5ModelOutput]:
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_model_output = None
        projection_attentions = None
        if image_embeds is None:
            if flattened_patches is None:
                raise ValueError("You have to specify either `flattened_patches` or `image_embeds`.")

            vision_model_output = self.vision_model(
                flattened_patches=flattened_patches,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # normalized features
            image_embeds = nn.functional.normalize(vision_model_output[0], dim=-1)
            image_embeds, projection_attentions = self.image_to_text_projection(image_embeds)

        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            image_embeds_position_mask=image_embeds_position_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            outputs = outputs + (image_embeds, projection_attentions, vision_model_output)
            return tuple(output for output in outputs if output is not None)

        return Kosmos2_5ModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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
# Copied from transformers.models.kosmos2.modeling_kosmos2.Kosmos2TextForCausalLM with KOSMOS-2->KOSMOS-2.5,KOSMOS2->KOSMOS2_5,Kosmos2->Kosmos2_5
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
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            image_embeds_position_mask=image_embeds_position_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        image_embeds=None,
        image_embeds_position_mask=None,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
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
            )[:, -1:]

            input_ids = input_ids[:, -1:]
            # the image info. is already encoded into the past keys/values
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

    @staticmethod
    # Copied from transformers.models.umt5.modeling_umt5.UMT5ForConditionalGeneration._reorder_cache
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    KOSMOS-2.5 Model for generating text and bounding boxes given an image. The model consists of a vision encoder and a
    language model.
    """,
    KOSMOS2_5_START_DOCSTRING,
)
class Kosmos2_5ForConditionalGeneration(Kosmos2_5PreTrainedModel):
    config_class = Kosmos2_5Config
    main_input_name = "flattened_patches"
    _tied_weights_keys = ["text_model.lm_head.weight"]

    def __init__(self, config: Kosmos2_5Config):
        super().__init__(config)
        self.text_model = Kosmos2_5TextForCausalLM._from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vision_model = Kosmos2_5VisionModel._from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        )
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

    @add_start_docstrings_to_model_forward(KOSMOS2_5_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=Kosmos2_5ForConditionalGenerationModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        flattened_patches: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        image_embeds_position_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        image_embeds: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Kosmos2_5ForConditionalGenerationModelOutput]:
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

        >>> repo = "microsoft/kosmos-2.5"
        >>> device = "cuda:0"
        >>> dtype = torch.bfloat16 # torch.float16
        >>> model = Kosmos2_5ForConditionalGeneration.from_pretrained(repo, device_map=device, torch_dtype=dtype)
        >>> processor = AutoProcessor.from_pretrained(repo)

        >>> url = "https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png"

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_model_output = None
        projection_attentions = None
        if image_embeds is None:
            if flattened_patches is None:
                raise ValueError("You have to specify either `flattened_patches` or `image_embeds`.")

            vision_model_output = self.vision_model(
                flattened_patches=flattened_patches,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeds = nn.functional.normalize(vision_model_output[0], dim=-1)
            image_embeds, projection_attentions = self.image_to_text_projection(image_embeds)

        lm_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            image_embeds_position_mask=image_embeds_position_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            outputs = lm_outputs + (image_embeds, projection_attentions, vision_model_output)
            return tuple(output for output in outputs if output is not None)

        return Kosmos2_5ForConditionalGenerationModelOutput(
            loss=lm_outputs.loss,
            logits=lm_outputs.logits,
            past_key_values=lm_outputs.past_key_values,
            hidden_states=lm_outputs.hidden_states,
            attentions=lm_outputs.attentions,
            image_embeds=image_embeds,
            projection_attentions=projection_attentions,
            vision_model_output=vision_model_output,
        )

    @torch.no_grad()
    def generate(
        self,
        flattened_patches: Optional[torch.Tensor] = None,
        image_embeds_position_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # in order to allow `inputs` argument (as in `GenerationMixin`)
        inputs = kwargs.pop("inputs", None)
        if flattened_patches is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs} were passed alongside `flattened_patches` which is not allowed."
                f"Make sure to either pass `inputs` or flattened_patches=..."
            )
        if flattened_patches is None and inputs is not None:
            flattened_patches = inputs

        if image_embeds is None:
            vision_model_output = self.vision_model(
                flattened_patches=flattened_patches,
                attention_mask=image_attention_mask,
                output_hidden_states=True,
            )
            image_embeds = nn.functional.normalize(vision_model_output[0], dim=-1)
            image_embeds, projection_attentions = self.image_to_text_projection(image_embeds)

        output = self.text_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            image_embeds_position_mask=image_embeds_position_mask,
            **kwargs,
        )

        return output
