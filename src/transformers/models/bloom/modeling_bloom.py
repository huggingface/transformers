# coding=utf-8
# Copyright 2022 HuggingFace Inc. team and BigScience workshop.
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
"""PyTorch BLOOM model."""

import math
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, LayerNorm

from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from ...modeling_utils import Conv1D, PreTrainedModel
from ...utils import logging
from .configuration_bloom import BloomConfig
from .fused_bias_gelu import bias_gelu_impl


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bigscience/Bloom"
_CONFIG_FOR_DOC = "BloomConfig"
_TOKENIZER_FOR_DOC = "BloomTokenizer"

BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bigscience/bigscience-small-testing",
    "bigscience/bloom-350m",
    "bigscience/bloom-760m",
    "bigscience/bloom-1b3",
    "bigscience/bloom-2b5",
    "bigscience/bloom-6b3",
    "bigscience/bloom-176b",
]


# Utility functions below:
def divide(numerator, denominator):
    """
    Ensure that numerator is divisible by the denominator and return the division value.


    Args:
        numerator ([`int`, `float`], *required*):
            Numerator to use for the division.
        denominator ([`int`, `float`], *required*):
            Denominator to use for the division."""
    if not (numerator % denominator == 0):
        raise ValueError(f"{numerator} is not divisible by {denominator}")
    return numerator // denominator


def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
    """Split a tensor along its last dimension.

    Args:
        tensor: ([`torch.tensor`], *required*):
            input tensor to split
        num_partitions ([`int`], *required*):
            number of partitions to split the tensor
        contiguous_split_chunks ([`bool`], *optional*, default=`False`)::
            If True, make each chunk contiguous in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def attention_mask_func(attention_scores, attention_mask, causal_mask):
    if attention_mask.dtype == torch.bool:
        attention_mask_bool = ~attention_mask
    else:
        attention_mask_bool = (1 - attention_mask).bool()

    query_length, key_length, n_heads = attention_scores.size(2), attention_scores.size(3), attention_scores.size(1)
    padded_causal_mask = (
        attention_mask_bool[:, None, key_length - query_length : key_length, None]
        + ~causal_mask[:, :, key_length - query_length : key_length, :key_length]
    ).bool()
    # Make use of floats
    if padded_causal_mask.dtype == torch.bool:
        return attention_scores.masked_fill_(padded_causal_mask.expand(-1, n_heads, -1, -1), -10000.0)
    else:
        values_to_attend = 1.0 - padded_causal_mask
        return (values_to_attend * attention_scores) - 10000.0 * padded_causal_mask


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)

    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, False)


class ScaledSoftmax(nn.Module):
    """
    fused operation: scaling + mask + softmax

    Arguments:
        input_in_fp16: flag to indicate if input in fp16 data format.
        input_in_bf16: flag to indicate if input in bf16 data format.
        attn_mask_type: attention mask type (pad or causal)
        scaled_masked_softmax_fusion: flag to indicate user want to use softmax fusion
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(
        self,
        input_in_fp16,
        input_in_bf16,
        scaled_masked_softmax_fusion,
        mask_func,
        softmax_in_fp32,
        scale,
        max_positions,
    ):
        super().__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        if self.input_in_fp16 and self.input_in_bf16:
            raise ValueError("Only one of input_in_fp16 and input_in_bf16 can be True")
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale

        self.causal_mask = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
            1, 1, max_positions, max_positions
        )
        if not (self.scale is None or softmax_in_fp32):
            raise ValueError("softmax should be in fp32 when scaled")

    def forward(self, input, mask):
        if self.input_in_float16 and self.softmax_in_fp32:
            input = input.float()

        if self.scale is not None:
            input = input * self.scale
        mask_output = self.mask_func(input, mask, self.causal_mask) if mask is not None else input
        probs = torch.nn.Softmax(dim=-1)(mask_output)

        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs


class BloomAttention(nn.Module):
    def __init__(self, config, layer_number=None):
        super().__init__()

        dtype = getattr(torch, config.dtype)
        self.pretraining_tp = config.pretraining_tp

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.masked_softmax_fusion = config.masked_softmax_fusion

        self.fp16 = dtype == torch.float16
        self.bf16 = dtype == torch.bfloat16

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.layer_number = max(1, layer_number)
        coeff = self.layer_number
        self.norm_factor = math.sqrt(self.head_dim) * coeff

        # Scaled Softmax
        self.scale_mask_softmax = ScaledSoftmax(
            self.fp16,
            self.bf16,
            self.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff,
            # To avoid cache problems, we use a small offset when creating the alibi positional embeddings
            config.seq_length + config.offset_alibi,
        )

        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, dtype=dtype, bias=True)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size, dtype=dtype)
        self.skip_bias_add = config.skip_bias_add
        self.skip_bias_add_qkv = config.skip_bias_add_qkv
        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        alibi=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]
        # repeat alibi tensor with the batch size
        alibi = alibi.repeat(hidden_states.shape[0], 1, 1).to(hidden_states.device)

        bias = self.query_key_value.bias if not self.skip_bias_add_qkv else None

        output_bias = self.query_key_value.bias if self.skip_bias_add_qkv else None

        mixed_x_layer = F.linear(hidden_states, self.query_key_value.weight, bias)

        # [batch_size, seq_length, 3 x hidden_size] --> [batch_size, seq_length, num_heads, 3 x head_dim]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_heads, 3 * self.head_dim)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [batch_size, seq_length, num_heads, 3 x head_dim] --> 3  [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=1)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=1)

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        # [batch_size, head_dim, q_length, k_length]
        output_size = (query_layer.size(0), query_layer.size(2), query_layer.size(1), key_layer.size(1))

        # [batch_size, q_length, num_heads, head_dim] -> [q_length, batch_size * num_heads, head_dim]
        query_layer = query_layer.contiguous().view(output_size[2], output_size[0] * output_size[1], -1)
        # [batch_size, k_length, num_heads, head_dim] -> [k_length, batch_size * num_heads, head_dim]
        key_layer = key_layer.contiguous().view(output_size[3], output_size[0] * output_size[1], -1)

        # alibi
        matmul_result = alibi[: output_size[0] * output_size[1], :, : output_size[3]]

        # Raw attention scores. [batch_size * num_heads, q_length, k_length]
        beta = 1.0 / self.layer_number

        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(1, 0),
            key_layer.transpose(1, 0).transpose(1, 2),
            beta=beta,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [batch_size, num_heads, q_length, k_length]

        attention_scores = matmul_result.view(*output_size)

        # attention scores and attention mask [b, np, sq, sk]

        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # context layer shape: [batch_size, num_heads, q_length, head_dim]
        output_size = (value_layer.size(0), value_layer.size(2), query_layer.size(0), value_layer.size(3))

        # change view [k_length, batch_size x num_heads, head_dim]
        value_layer = value_layer.contiguous().view(value_layer.size(1), output_size[0] * output_size[1], -1)

        # change view [batch_size x num_heads, q_length, k_length]
        attention_probs_reshaped = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer.transpose(0, 1))
        # context_layer = torch.bmm(attention_probs, value_layer)

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = context_layer.view(*output_size)

        # [batchs_size, num_heads, q_length, head_dim] --> [q_length, batch_size, num_heads, head_dim]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [q_length, batch_size, num_heads, head_dim] --> [q_length, batch_size, hidden_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)

        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [q_length, batch_size, hidden_size]
        # =================

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1:
            slices = context_layer.shape[-1] / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = F.linear(context_layer, self.dense.weight)

        if not self.skip_bias_add:
            output_tensor = output_tensor + self.dense.bias if self.dense.bias is not None else output_tensor
            output_bias = None
        else:
            output_tensor = output_tensor
            output_bias = self.dense.bias
        output = output_tensor.transpose(1, 0)
        outputs = (output, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs, output_bias


class BloomMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        dtype = getattr(torch, config.dtype)
        self.skip_bias_add = config.skip_bias_add
        self.pretraining_tp = config.pretraining_tp
        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size, dtype=dtype)

        self.activation_func = bias_gelu_impl
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size, dtype=dtype)

    def forward(self, hidden_states):
        input_ = hidden_states

        hidden_states = self.activation_func(
            F.linear(hidden_states, self.dense_h_to_4h.weight), self.dense_h_to_4h.bias
        )

        if self.pretraining_tp > 1:
            intermediate_output = torch.zeros_like(input_)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + F.linear(
                    hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            intermediate_output = F.linear(hidden_states, self.dense_4h_to_h.weight)

        if not self.skip_bias_add:
            output = (
                intermediate_output + self.dense_4h_to_h.bias
                if self.dense_4h_to_h.bias is not None
                else intermediate_output
            )
            output_bias = None
        else:
            output = intermediate_output
            output_bias = self.dense_4h_to_h.bias
        return output, output_bias


class BloomBlock(nn.Module):
    def __init__(self, config, layer_number=None):
        super().__init__()
        hidden_size = config.hidden_size
        dtype = getattr(torch, config.dtype)

        self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon).to(dtype)
        self.alibi = self._build_alibi_tensor(config.seq_length + config.offset_alibi, config.n_head, dtype=dtype)
        self.self_attention = BloomAttention(config, layer_number=layer_number)
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon).to(dtype)

        self.mlp = BloomMLP(config)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.bias_dropout_fusion = config.bias_dropout_fusion
        self.hidden_dropout = config.hidden_dropout

    # alibi tensor is not causal as the original paper mentions, it relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value `softmax(l+a) = softmax(l)
    @staticmethod
    def _build_alibi_tensor(max_seq_len, n_head, dtype=torch.bfloat16):
        # Based on https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
        """Returns tensor shaped (n_head, 1, max_seq_len)"""

        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        slopes = torch.Tensor(get_slopes(n_head)).unsqueeze(1).unsqueeze(1)
        arange_tensor = torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0)
        alibi = slopes * arange_tensor.expand(n_head, -1, -1)

        alibi = alibi.to(dtype)

        return alibi

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # hidden_states: [b, s, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attn_outputs, attention_bias = self.self_attention(
            layernorm_output,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=self.alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output, attention_bias.expand_as(residual), residual, self.hidden_dropout
            )

        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.

        mlp_output, mlp_bias = self.mlp(layernorm_output)

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # output = mlp_output + residual
        with torch.enable_grad():
            output = bias_dropout_add_func(mlp_output, mlp_bias.expand_as(residual), residual, self.hidden_dropout)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


class BloomPreTrainedModel(PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BloomConfig
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BloomModel):
            module.gradient_checkpointing = value


BLOOM_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BloomConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BLOOM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`BloomTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
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
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Bloom Model transformer outputting raw hidden-states without any specific head on top.",
    BLOOM_START_DOCSTRING,
)
class BloomModel(BloomPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        dtype = getattr(torch, config.dtype)
        if dtype not in [torch.bfloat16, torch.float32, torch.float, torch.float16]:
            raise ValueError(f"Unsupported dtype {dtype}")

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim, dtype=dtype)
        self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon).to(dtype)

        # Transformer blocks
        self.h = nn.ModuleList([BloomBlock(config, layer_number=i) for i in range(config.num_hidden_layers)])

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon).to(dtype)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_head x N x N
        # head_mask has shape n_layer x batch x n_head x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)
        # hidden_states = hidden_states.transpose(0, 1)
        # hidden_states = hidden_states.transpose(0, 1).contiguous()

        if token_type_ids is not None:
            token_type_embeds = self.word_embeddings(token_type_ids)
            token_type_embeds = token_type_embeds.transpose(0, 1).contiguous()
            hidden_states = hidden_states + token_type_embeds

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                # all_hidden_states = all_hidden_states + (hidden_states.permute(1, 0, 2),)
                # all_hidden_states = all_hidden_states + (hidden_states.view(output_shape),)
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)
        # hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            # all_hidden_states = all_hidden_states + (hidden_states.view(output_shape),)
            # all_hidden_states = all_hidden_states + (hidden_states.permute(1, 0, 2),)

        hidden_states = hidden_states.view(output_shape)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@add_start_docstrings(
    """
    The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    BLOOM_START_DOCSTRING,
)
class BloomForCausalLM(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = BloomModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
