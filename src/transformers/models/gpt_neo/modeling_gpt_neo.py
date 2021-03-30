# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
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
""" PyTorch GPT Neo model. """


import os
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_gpt_neo import GPTNeoConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GPTNeoConfig"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"

GPT_NEO_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EleutherAI/gpt_neo_xl",
    # See all GPTNeo models at https://huggingface.co/models?filter=gpt_neo
]

_CHECKPOINT_FOR_DOC = "EleutherAI/gpt_neo_xl"


def load_tf_weights_in_gpt_neo(model, config, gpt_neo_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re

        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt_neo_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        if "global_step" not in name and "adam" not in name:
            array = tf.train.load_variable(tf_path, name)
            array = tf.dtypes.cast(array.squeeze(), tf.float32).numpy()
            name = name.replace("attn/q", "attn/attention/q_proj/w")
            name = name.replace("attn/k", "attn/attention/k_proj/w")
            name = name.replace("attn/v", "attn/attention/v_proj/w")
            name = name.replace("attn/o", "attn/attention/out_proj/w")
            name = name.replace("norm_1", "ln_1")
            name = name.replace("norm_2", "ln_2")
            name = name.replace("attn/compute_output_bias/o_b", "attn/attention/out_proj/b")
            name = name.replace("conv1d_main/c_fc/kernel", "c_fc/w")
            name = name.replace("conv1d_main/c_fc/bias", "c_fc/b")
            name = name.replace("conv1d_main/c_proj/kernel", "c_proj/w")
            name = name.replace("conv1d_main/c_proj/bias", "c_proj/b")

            names.append(name)
            arrays.append(array)

    for name, array in zip(names, arrays):
        name = name[5:]  # skip "gpt2/"
        name = name.split("/")
        pointer = model.transformer
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        if name[-1] == "w" and name[-2] in ["out_proj", "k_proj", "q_proj", "v_proj", "c_proj", "c_fc"]:
            array = array.transpose()

        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched {name}"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)

    # init the final linear layer using word embeddings
    embs = model.transformer.wte.weight
    lin = nn.Linear(embs.size()[1], embs.size()[0], bias=False)
    lin.weight = embs
    model.set_output_embeddings(lin)
    return model


class GPTNeoSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        # Keep the attention weights computation in fp32 to avoid overflow issues
        q = q.to(torch.float32)
        k = k.to(torch.float32)

        attn_weights = torch.matmul(q, k)
        nd, ns = attn_weights.size(-2), attn_weights.size(-1)

        mask = self.bias[:, :, ns - nd : ns, :ns]
        attn_weights = torch.where(mask.bool(), attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = attn_weights.to(v.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        outputs = (torch.matmul(attn_weights, v),)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.num_heads, x.size(-1) // self.num_heads)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key.transpose(-2, -1), value)  # transpose to have same shapes
        else:
            present = None

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.out_proj(a)
        a = self.resid_dropout(a)

        return (a, present) + attn_outputs[1:]  # a, present, (attentions)


class GPTNeoLocalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.register_buffer("masked_bias", torch.tensor(-1e9))

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.window_size = config.window_size

    def shift(self, x, offset, pad_value=0, dim=2):
        t = x.shape[1]
        dims = (len(x.shape) - dim) * (0, 0)
        padded_x = F.pad(x, (*dims, offset, 0), value=pad_value)
        return padded_x[:, :t, ...]

    def look_around(self, x, block_length, window_size):
        num_complete_blocks = window_size // block_length

        parts = [x]
        for i in range(1, num_complete_blocks + 1):
            parts = [self.shift(x, i)] + parts

        partial_size = window_size % block_length
        if partial_size > 0:
            margin = x[:, :, block_length - partial_size : block_length, ...]
            parts = [self.shift(margin, num_complete_blocks + 1)] + parts
        return torch.cat(parts, dim=2)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.num_heads, x.size(-1) // self.num_heads)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 1, 3, 4, 2)  # (batch, chunks, head, head_features, seq_length)
        else:
            return x.permute(0, 1, 3, 2, 4)  # (batch, chunks, head, seq_length, head_features)

    def merge_heads(self, x):
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def _split_seq_length_dim_to(self, tensors, num_blocks, block_length):
        return tensors.reshape(tensors.size()[0], num_blocks, block_length, -1)

    def create_attention_mask(self, bs, seq_len, windows, block_length, attention_mask):
        ticker = torch.arange(seq_len)[None, :]
        b_t = ticker.reshape(1, windows, block_length)

        bq_t = b_t
        bq_k = self.look_around(b_t, block_length, self.window_size)

        # compute attn mask
        # this matches the original implem in mess-tensorflow
        # https://github.com/tensorflow/mesh/blob/8bd599a21bad01cef1300a8735c17306ce35db6e/mesh_tensorflow/transformer/attention.py#L805
        relative_position = bq_k.unsqueeze(-2) - bq_t.unsqueeze(-1)
        relative_position = relative_position.transpose(-1, -2)

        sequence_id = torch.ones(bs, seq_len)
        q_seq = sequence_id.reshape(-1, windows, block_length)
        m_seq = sequence_id.reshape(-1, windows, block_length)
        m_seq = self.look_around(m_seq, block_length, self.window_size)

        if attention_mask is not None:
            attention_mask = attention_mask.to(m_seq.device)
            attention_mask = attention_mask.reshape(-1, windows, block_length)
            attention_mask = self.look_around(attention_mask, block_length, self.window_size)
            m_seq *= attention_mask

        visible = torch.eq(q_seq.unsqueeze(-1), m_seq.unsqueeze(-2)).transpose(-1, -2)
        visible = torch.logical_and(visible, torch.gt(relative_position, -self.window_size))
        mask = torch.logical_and(visible, torch.less_equal(relative_position, 0)).transpose(-1, -2).unsqueeze(2)
        return mask

    def _attn(self, q, k, v, causal_mask, head_mask=None, output_attentions=False):
        # attn

        # Keep the attention weights computation in fp32 to avoid overflow issues
        q = q.to(torch.float32)
        k = k.to(torch.float32)

        attn_weights = torch.matmul(q, k)
        attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = attn_weights.to(v.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, v)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        query = self.q_proj(hidden_states)

        if layer_past is not None:
            past = layer_past[0]
            key_value_hidden_states = torch.cat([past, hidden_states], dim=1)
            past_length = past.size()[1]
        else:
            key_value_hidden_states = hidden_states
            past_length = 0

        key = self.k_proj(key_value_hidden_states)
        value = self.v_proj(key_value_hidden_states)

        # compute block length and windows
        bs, seq_len = hidden_states.shape[:2]
        full_seq_length = seq_len + past_length
        block_length = self.window_size
        while full_seq_length % block_length != 0:
            block_length -= 1
        num_blocks = full_seq_length // block_length

        # create buckets
        if layer_past is not None:
            # we just need 1 window with block_length 1 when caching is enabled
            query = self._split_seq_length_dim_to(query, 1, 1)
        else:
            query = self._split_seq_length_dim_to(query, num_blocks, block_length)

        key = self._split_seq_length_dim_to(key, num_blocks, block_length)
        value = self._split_seq_length_dim_to(value, num_blocks, block_length)

        key = self.look_around(key, block_length, self.window_size)
        value = self.look_around(value, block_length, self.window_size)

        # select key/value vectors only for the last window
        if layer_past is not None:
            key = key[:, -1:, ...]
            value = value[:, -1:, ...]

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        mask = self.create_attention_mask(bs, full_seq_length, num_blocks, block_length, attention_mask)
        if layer_past is not None:
            mask = mask[:, -1:, :, -1:, :]  # only take the mask for the last window
        mask = mask.to(hidden_states.device)

        # attn
        attn_outputs = self._attn(query, key, value, mask, head_mask, output_attentions)
        attn = attn_outputs[0]

        attn = self.merge_heads(attn)
        attn = attn.reshape(bs, seq_len, self.embed_dim)

        attn = self.out_proj(attn)
        attn = self.resid_dropout(attn)
        return (attn,) + attn_outputs[1:]


class GPTNeoAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attention_layers = config.attention_layers
        self.attention_type = self.attention_layers[layer_id]

        if self.attention_type == "global":
            self.attention = GPTNeoSelfAttention(config)
        elif self.attention_type == "local":
            self.attention = GPTNeoLocalSelfAttention(config)
        else:
            raise NotImplementedError(
                "Only attn layer types 'global' and 'local' exist, but got `config.attention_layers`: {}. Select attn layer types from ['global', 'local'] only.".format(
                    self.attention_layers
                )
            )

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        outputs = self.attention(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        # cache the hidden_states instead of key_value_states
        # for local attention layer
        if self.attention_type == "local":
            if layer_past is None:
                past = hidden_states
            else:
                past = torch.cat([layer_past[0], hidden_states], dim=1)
            outputs = (outputs[0], (past,)) + outputs[1:]
        return outputs


class MLP(nn.Module):
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * hidden_size
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPTNeoAttention(config, layer_id)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + hidden_states

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPTNeoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTNeoConfig
    load_tf_weights = load_tf_weights_in_gpt_neo
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear,)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


GPT_NEO_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.GPTNeoConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

GPT_NEO_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`):
            :obj:`input_ids_length` = ``sequence_length`` if :obj:`past_key_values` is ``None`` else
            ``past_key_values[0][0].shape[-2]`` (``sequence_length`` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If :obj:`past_key_values` is used, only ``input_ids`` that do not have their past calculated should be
            passed as ``input_ids``.

            Indices can be obtained using :class:`~transformers.GPTNeoTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.num_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            :obj:`past_key_values` output below). Can be used to speed up sequential decoding. The ``input_ids`` which
            have their past given to this model should not be passed as ``input_ids`` as they have already been
            computed.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.

            If :obj:`past_key_values` is used, optionally only the last :obj:`inputs_embeds` have to be input (see
            :obj:`past_key_values`).
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare GPTNeo Model transformer outputting raw hidden-states without any specific head on top.",
    GPT_NEO_START_DOCSTRING,
)
class GPTNeoModel(GPTNeoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embed_dropout)
        self.h = nn.ModuleList([Block(config, layer_id=i) for i in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
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
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

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
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            global_attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            global_attention_mask = global_attention_mask[:, None, None, :]

            # Since global_attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            global_attention_mask = global_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            global_attention_mask = (1.0 - global_attention_mask) * -10000.0
        else:
            global_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_headss x N x N
        # head_mask has shape n_layer x batch x num_headss x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            attn_type = self.config.attention_layers[i]
            attn_mask = global_attention_mask if attn_type == "global" else attention_mask

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
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
                    attn_mask,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attn_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@add_start_docstrings(
    """
    The GPT Neo Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT_NEO_START_DOCSTRING,
)
class GPTNeoForCausalLM(GPTNeoPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]
    _keys_to_ignore_on_save = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTNeoModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.init_weights()

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

    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
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
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
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

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Compute loss in fp32 to match with mesh-tf version
            # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PretrainedModel.beam_search` or :meth:`~transformers.PretrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
