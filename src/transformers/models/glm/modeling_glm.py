# coding=utf-8
# Copyright 2022 shunxing1234 The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch GLM model. """

import math
import os

import torch
import torch.utils.checkpoint
from packaging import version
import torch.nn.functional as F
from torch import nn
from torch.nn import init, LayerNorm, Linear, CrossEntropyLoss
from typing import Optional, Tuple, Union

from ...activations import ACT2FN, gelu
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    SequenceClassifierOutput,
)
from ...modeling_utils import (
    PreTrainedModel,
)
from ...utils import logging
from .configuration_glm import GLMConfig
from torch.nn.parameter import Parameter

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "shunxing1234/GLM"
_CONFIG_FOR_DOC = "GLMConfig"
_TOKENIZER_FOR_DOC = "GLMTokenizer"

GLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "shunxing1234/GLM",
    # See all GLM models at https://huggingface.co/models?filter=glm
]


def unscaled_init_method(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method(mean, std, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = std / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=mean, std=std)

    return init_


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor, num_partitions,
                                contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
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


class MLP(torch.nn.Module):
    """MLP for GPT2.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Arguments:
        hidden_size: The hidden size of the self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layer initialization. If None,
                                  use `init_method`.
    """

    def __init__(self, hidden_size, output_dropout_prob, init_method,
                 output_layer_init_method=None):
        super(MLP, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Project to 4h.
        self.dense_h_to_4h = Linear(hidden_size, 4 * hidden_size)

        # Project back to h.
        self.dense_4h_to_h = Linear(
            4 * hidden_size,
            hidden_size)

        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states):
        # [b, s, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = gelu(intermediate_parallel)

        # [b, s, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output


class VocabEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, config):
        super(VocabEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = config.vocab_size
        self.embedding_dim = config.hidden_size
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None

        self.vocab_start_index = 0
        self.vocab_end_index = self.num_embeddings

        # Allocate weights.
        self.weight = Parameter(torch.Tensor(self.num_embeddings,
                                             self.embedding_dim))
        # And initialize.
        init.xavier_normal_(self.weight)

    def forward(self, input_):
        # Get the embeddings.
        output = F.embedding(input_, self.weight,
                             self.padding_idx, self.max_norm,
                             self.norm_type, self.scale_grad_by_freq,
                             self.sparse)
        return output


class PositionalEmbedding(torch.nn.Module):

    def __init__(self, hidden_size):
        super(PositionalEmbedding, self).__init__()

        self.hidden_size = hidden_size

        inv_freq = 1 / (10000 ** (torch.arange(0.0, hidden_size, 2.0) / hidden_size))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


class SelfAttention(torch.nn.Module):
    """self-attention layer for GLM.

    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence lenght, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size to be divisible by n.
        attention_dropout_prob: dropout probability for the attention scores.
        init_method: weight initialization.
        output_layer_init_method: output layer initialization. If None, use
                                  `init_method`.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
    """

    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob,
                 init_method, output_layer_init_method=None,
                 attention_scale=1.0):
        super(SelfAttention, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Per attention head and per partition values.
        self.hidden_size = hidden_size
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)

        self.num_attention_heads = num_attention_heads
        self.attention_scale = attention_scale
        # Strided linear layer.
        self.query_key_value = Linear(hidden_size, 3 * hidden_size)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = Linear(hidden_size,
                            hidden_size)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    @staticmethod
    def _rel_shift(x, zero_triu=False):
        # ql x kl x bsz x h
        # bsz x h x ql x kl
        zero_pad = torch.zeros((*x.size()[:-2], x.size(-2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:-2], x.size(-1) + 1, x.size(-2))

        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, hidden_states, ltor_mask, position_embeddings=None, r_w_bias=None, r_r_bias=None, mem=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [b,1,s,s]

        # Attention heads. [b, s, hp]
        query_length = hidden_states.size(1)
        # self attention
        if mem is None:
            mixed_x_layer = self.query_key_value(hidden_states)
            (mixed_query_layer,
             mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            cat = torch.cat((mem, hidden_states), 1)
            mixed_x_layer = self.query_key_value(cat)
            (mixed_query_layer,
             mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
            mixed_query_layer = mixed_query_layer[:, -query_length:]

        # Reshape and transpose [b, np, s, hn]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        if self.attention_scale > 1.0:
            # Raw attention scores. [b, np, s, s]
            attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_scale),
                                            key_layer.transpose(-1, -2) / math.sqrt(
                                                self.hidden_size_per_attention_head * self.attention_scale))
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2) / math.sqrt(
                self.hidden_size_per_attention_head))

        # Apply the left to right attention mask.
        attention_scores = torch.mul(attention_scores, ltor_mask)
        if self.attention_scale > 1.0:
            max_attention_scores = attention_scores.max(dim=-1, keepdim=True)[0]
            attention_scores -= max_attention_scores
            attention_scores *= self.attention_scale

        attention_scores = attention_scores + (-65504.0) * (1.0 - ltor_mask)
        # Attention probabilities. [b, np, s, s]
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # with get_cuda_rng_tracker().fork():
        attention_probs = self.attention_dropout(attention_probs)

        # Context layer.
        # [b, np, s, hn]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [b, s, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output


class GLMBlock(torch.nn.Module):
    """A single layer transformer for GLM.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layers (attention output and
                                  mlp output) initialization. If None,
                                  use `init_method`.
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 init_method,
                 output_layer_init_method=None,
                 attention_scale=1.0):
        super(GLMBlock, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method,
            attention_scale=attention_scale)

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(hidden_size,
                                                  eps=layernorm_epsilon)

        # MLP
        self.mlp = MLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method)

    def forward(self, hidden_states, ltor_mask, position_embeddings=None, r_w_bias=None, r_r_bias=None, mem=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [b,1, s,s]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        mem = self.input_layernorm(mem) if mem is not None else None
        # Self attention.
        attention_output = self.attention(layernorm_output, ltor_mask, position_embeddings, r_w_bias, r_r_bias, mem)
        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection.
        output = layernorm_input + mlp_output

        return output


class GLMStack(torch.nn.Module):
    """GLM transformer.

    This module takes input from embedding layer and it's output can
    be used directly by a logit layer. It consists of L (num-layers)
    blocks of:
        layer norm
        self attention
        residual connection
        layer norm
        mlp
        residual connection
    followed by a final layer norm.

    Arguments:
        num_layers: Number of transformer layers.
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        checkpoint_activations: if True, checkpoint activations.
        checkpoint_num_layers: number of layers to checkpoint. This
                               is basically the chunk size in checkpoitning.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method_std: standard deviation of the init method which has
                         the form N(0, std).
        use_scaled_init_for_output_weights: If Ture use 1/sqrt(2*num_layers)
                                            scaling for the output weights (
                                            output of self attention and mlp).
    """

    def __init__(self,
                 num_layers,
                 hidden_size,
                 num_attention_heads,
                 max_sequence_length,
                 max_memory_length,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 layernorm_epsilon=1.0e-5,
                 init_method_std=0.02,
                 use_scaled_init_for_output_weights=True,
                 block_position_encoding=False,
                 attention_scale=1.0,
                 ):
        super(GLMStack, self).__init__()
        self.hidden_size = hidden_size
        # Store activation checkpoiting flag.
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_memory_length = max_memory_length

        output_layer_init_method = None
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method(0.0, init_method_std,
                                                          num_layers)
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)
        self.block_position_encoding = block_position_encoding

        # Position embedding (serial).
        if block_position_encoding:
            self.position_embeddings = torch.nn.Embedding(max_sequence_length + 1, hidden_size)
            self.block_position_embeddings = torch.nn.Embedding(max_sequence_length + 1, hidden_size)
            torch.nn.init.normal_(self.block_position_embeddings.weight, mean=0.0, std=init_method_std)
        else:
            self.position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
        # Initialize the position embeddings.
        torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)

        def get_layer():

            return GLMBlock(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                unscaled_init_method(init_method_std),
                output_layer_init_method=output_layer_init_method,
                attention_scale=attention_scale)

        # Transformer layers.
        self.layers = torch.nn.ModuleList(
            [get_layer() for _ in range(num_layers)])

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

    def forward(self, hidden_states, position_ids, attention_mask, memory_states=None):

        batch_size, query_length = hidden_states.size()[:2]
        memory_length = 0
        key_length = query_length + memory_length
        # attention mask is the beginning postion of B region, \in [0, query_len)
        is_scalar = torch.numel(attention_mask) == 1
        is_sep = is_scalar or torch.numel(attention_mask) == batch_size
        if is_sep:
            sep = attention_mask.item() if is_scalar else attention_mask

            # conventional transformer
            def build_mask_matrix(seq_length, sep, memory_length=0):
                m = hidden_states.new_ones((1, seq_length, seq_length))
                m = torch.tril(m)
                if is_scalar:
                    m[0, :, :int(sep)] = 1
                else:
                    m = m.expand(batch_size, -1, -1)
                    ids = torch.arange(seq_length, device=sep.device, dtype=sep.dtype).view(1, -1)
                    mask = ids < sep.view(-1, 1)
                    m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
                if memory_length > 0:
                    m = m.expand(batch_size, -1, -1)
                    m = torch.cat((hidden_states.new_ones((batch_size, seq_length, memory_length)), m), dim=2)
                m = m.unsqueeze(1)
                return m

            attention_mask = build_mask_matrix(query_length, sep, memory_length=memory_length)
        else:
            # [b,1,s,s]
            attention_mask = attention_mask[:, :, :, -query_length - memory_length:]

        if self.block_position_encoding:
            position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
        position_embeddings = self.position_embeddings(position_ids)

        hidden_states = hidden_states + position_embeddings
        if self.block_position_encoding:
            block_position_embeddings = self.block_position_embeddings(block_position_ids)
            hidden_states = hidden_states + block_position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        def check_detach(_hidden_states):
            return _hidden_states.detach()

        if self.max_memory_length > 0:
            mem_layers = [check_detach(hidden_states)]
        else:
            mem_layers = []

        for i, layer in enumerate(self.layers):

            args = [hidden_states, attention_mask]

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs)

                return custom_forward

            mem_i = None

            if self.checkpoint_activations:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    mem=mem_i,
                )
            else:
                hidden_states = layer(*args, mem=mem_i)
            if self.max_memory_length > 0:
                mem_layers.append(check_detach(hidden_states))

        # Final layer norm.
        output = self.final_layernorm(hidden_states)
        if self.max_memory_length > 0:
            mem_layers = self.update_mems(mem_layers, memory_states)
        return (output, mem_layers)

    def update_mems(self, hiddens, mems):
        memory_length = 0
        query_length = hiddens[0].size(1)
        new_memory_length = memory_length + query_length

        new_memory_length = min(self.max_memory_length, new_memory_length)
        new_mems = []
        # with torch.no_grad():
        for i in range(len(hiddens)):
            if new_memory_length <= query_length:
                new_mems.append(hiddens[i][:, -new_memory_length:])
            else:
                new_mems.append(torch.cat((mems[i][:, -new_memory_length + query_length:], hiddens[i]), dim=1))
        return new_mems


class GLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = GLMConfig
    base_model_prefix = "glm"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, torch.nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GLMModel):
            module.gradient_checkpointing = value


GLM_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config ([`~GLMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GLM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`GLMTokenizer`].
            See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range `[0, config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert *input_ids* indices into associated vectors
            than the model's internal embedding lookup matrix.
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
    "The bare GLM Model transformer outputting raw hidden-states without any specific head on top.",
    GLM_START_DOCSTRING,
)
class GLMModel(GLMPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    `is_decoder` argument of the configuration set to `True`.
    To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder`
    argument and `add_cross_attention` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.output_predict = config.output_predict
        # Word embeddings (parallel).
        self.word_embeddings = VocabEmbedding(config)

        # Transformer
        self.transformer = GLMStack(config.num_layers,
                                    config.hidden_size,
                                    config.num_attention_heads,
                                    config.max_sequence_length,
                                    config.max_memory_length,
                                    config.embedding_dropout_prob,
                                    config.attention_dropout_prob,
                                    config.output_dropout_prob,
                                    config.checkpoint_activations,
                                    config.checkpoint_num_layers,
                                    attention_scale=config.attention_scale,
                                    block_position_encoding=config.block_position_encoding)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            mems=None,
            prompt_pos=None,
            **kwargs
    ):
        batch_size = input_ids.size(0)
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings

        device = input_ids.device
        input_shape = input_ids.size()

        if prompt_pos is not None:
            embeddings = embeddings.clone()
            prompt_embeds = self.prompt_spell()
            batch_index = torch.arange(batch_size, device=input_ids.device).unsqueeze(1)
            embeddings[batch_index, prompt_pos] = prompt_embeds

        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        if attention_mask is None:
            attention_mask = torch.zeros(batch_size)
        # Transformer.
        transformer_output = self.transformer(embeddings, position_ids, attention_mask, mems)
        logits, hidden_layers = transformer_output
        # outputs = hidden_layers
        if self.output_predict:
            # Parallel logits.
            # logits_parallel = mpu.copy_to_model_parallel_region(
            #     logits)
            logits = F.linear(logits, self.word_embeddings.weight)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=logits,
            hidden_states=hidden_layers,
        )


@add_start_docstrings(
    """GLM Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    GLM_START_DOCSTRING,
)
class GLMForSequenceClassification(GLMPreTrainedModel):
    def __init__(self, config, pool_token, hidden_dropout=False, num_class=1):
        super().__init__(config)
        self.pool_token = pool_token
        self.model = GLMModel(config)
        self.model.output_predict = False
        self.num_class = num_class
        # Multi-choice head.
        self.pool_layer = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.multichoice_dropout = torch.nn.Dropout(hidden_dropout)
        self.multichoice_head = torch.nn.Linear(config.hidden_size, num_class)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(self,
                input_ids=None,
                position_ids=None,
                attention_mask=None,
                labels=None):

        num_choices = None

        if len(input_ids.shape) == 3:
            assert self.num_class == 1
            batch_size, num_choices = input_ids.shape[:2]
            input_ids = input_ids.reshape(-1, input_ids.size(-1))
            attention_mask = attention_mask.reshape(-1, *attention_mask.size()[2:])
            position_ids = position_ids.reshape(-1, *position_ids.size()[2:])
        model_out = self.model.forward(input_ids, position_ids, attention_mask)
        outputs, mems = model_out.last_hidden_state, model_out.hidden_states

        if self.pool_token == 'start':
            output = outputs[
                torch.arange(outputs.size(0), dtype=attention_mask.dtype, device=attention_mask.device), attention_mask]
        elif self.pool_token == 'pad':
            output = outputs[torch.arange(outputs.size(0), dtype=attention_mask.dtype,
                                          device=attention_mask.device), attention_mask - 1]
        elif self.pool_token == 'cls':
            output = outputs[:, 0]
        else:
            raise NotImplementedError

        output = torch.tanh(self.pool_layer(output))
        multichoice_output = self.multichoice_dropout(output)
        logits = self.multichoice_head(multichoice_output)
        loss_fct = CrossEntropyLoss()
        if num_choices is not None:
            logits = logits.view(-1, num_choices)
        assert (labels != None, "labels must not None!")
        loss = loss_fct(logits, labels)
        # loss = F.cross_entropy(logits.contiguous().float(), labels.long())
        return SequenceClassifierOutput(loss=loss,
                                        logits=logits,
                                        hidden_states=mems)

