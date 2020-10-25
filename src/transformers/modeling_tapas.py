# coding=utf-8
# Copyright (...) and The HuggingFace Inc. team.
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
"""PyTorch TAPAS model. """


import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from .activations import ACT2FN
from .configuration_tapas import TapasConfig
from .file_utils import (ModelOutput, 
                        add_start_docstrings, 
                        add_start_docstrings_to_callable,
                        replace_return_docstrings,
                        is_scatter_available,
                        requires_scatter,
)
from .modeling_outputs import (
    BaseModelOutput, 
    BaseModelOutputWithPooling, 
    MaskedLMOutput,
    SequenceClassifierOutput,
)
from .modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from .utils import logging

# soft dependency
if is_scatter_available():
    from transformers import modeling_tapas_utilities as utils


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TapasConfig"
_TOKENIZER_FOR_DOC = "TapasTokenizer"

TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "tapas-base",
    "tapas-large",
    # See all TAPAS models at https://huggingface.co/models?filter=tapas
]


@dataclass
class TableQuestionAnsweringOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TapasForQuestionAnswering`.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label_ids` and :obj:`answer` (and possibly :obj:`aggregation_labels`, :obj:`numeric_values` and :obj:`numeric_values_scale` are provided):
            Total loss as the sum of the hierarchical cell selection log-likelihood loss, (optionally) supervised cell selection
            loss and (optionally) the semi-supervised regression loss and (optionally) supervised loss for aggregations.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the cell selection head, for every token.
        logits_aggregation (:obj:`torch.FloatTensor`, `optional`, of shape :obj:`(batch_size, num_aggregation_labels)`):
            Prediction scores of the aggregation head, for every aggregation operator.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    logits_aggregation: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def load_tf_weights_in_tapas(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a PyTorch model. This is an adaptation from load_tf_weights_in_bert
    - add cell selection and aggregation heads
    - take into account additional token type embedding layers
    """
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculate m and v
        # which are not required for using pretrained model
        if any(
            n
            in [
                "adam_v",
                "adam_m",
                "AdamWeightDecayOptimizer",
                "AdamWeightDecayOptimizer_1",
                "global_step",
                "seq_relationship",
            ]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        # if first scope name starts with "bert", change it to "tapas"
        if name[0] == "bert":
            name[0] = "tapas"
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            # cell selection heads
            elif scope_names[0] == "output_bias":
                pointer = getattr(pointer, "output_bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "output_weights")
            elif scope_names[0] == "column_output_bias":
                pointer = getattr(pointer, "column_output_bias")
            elif scope_names[0] == "column_output_weights":
                pointer = getattr(pointer, "column_output_weights")
            # aggregation head
            elif scope_names[0] == "output_bias_agg":
                pointer = getattr(pointer, "output_bias_agg")
            elif scope_names[0] == "output_weights_agg":
                pointer = getattr(pointer, "output_weights_agg")
            # classification head
            elif scope_names[0] == "output_bias_cls":
                pointer = getattr(pointer, "output_bias_cls")
            elif scope_names[0] == "output_weights_cls":
                pointer = getattr(pointer, "output_weights_cls")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name[-13:] in [
            "_embeddings_0",
            "_embeddings_1",
            "_embeddings_2",
            "_embeddings_3",
            "_embeddings_4",
            "_embeddings_5",
            "_embeddings_6",
        ]:
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        # added a check whether the array is a scalar (because bias terms are scalar => should first be converted to numpy arrays)
        if np.isscalar(array):
            array = np.array(array)
        pointer.data = torch.from_numpy(array)
    return model


class TapasEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    Same as BertEmbeddings but with a number of additional token type embeddings to encode tabular structure.
    """

    def __init__(self, config):
        super().__init__()
        # we do not include config.disabled_features and config.disable_position_embeddings from the original implementation
        # word embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # position embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # token type embeddings
        token_type_embedding_name = "token_type_embeddings"

        for i, type_vocab_size in enumerate(config.type_vocab_size):
            name = "%s_%d" % (token_type_embedding_name, i)
            setattr(self, name, nn.Embedding(type_vocab_size, config.hidden_size))

        self.number_of_token_type_embeddings = len(config.type_vocab_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.config = config

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            # create absolute position embeddings
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
            # when self.config.reset_position_index_per_cell is set to True, create relative position embeddings
            if self.config.reset_position_index_per_cell:
                col_index = utils.IndexMap(
                    token_type_ids[:, :, 1], self.config.type_vocab_size[1], batch_dims=1
                )  # shape (batch_size, seq_len)
                row_index = utils.IndexMap(
                    token_type_ids[:, :, 2], self.config.type_vocab_size[2], batch_dims=1
                )  # shape (batch_size, seq_len)
                full_index = utils.ProductIndexMap(col_index, row_index)  # shape (batch_size, seq_len)

                first_position_per_segment = utils.reduce_min(position_ids, full_index)[
                    0
                ]  # shape (max_rows * max_columns,). First absolute position for every cell
                first_position = utils.gather(
                    first_position_per_segment, full_index
                )  # ? shape (batch_size, seq_len). First absolute position of the cell for every token
                position = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, seq_len)
                position_ids = torch.min(
                    torch.as_tensor(self.config.max_position_embeddings - 1, device=device), position - first_position
                )

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                (*input_shape, self.number_of_token_type_embeddings), dtype=torch.long, device=device
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings

        token_type_embedding_name = "token_type_embeddings"

        for i in range(self.number_of_token_type_embeddings):
            name = "%s_%d" % (token_type_embedding_name, i)
            embeddings += getattr(self, name)(token_type_ids[:, :, i])

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.modeling_bert.BertSelfAttention with Bert->Tapas
class TapasSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TapasModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


# Copied from transformers.modeling_bert.BertSelfOutput
class TapasSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.modeling_bert.BertAttention with Bert->Tapas
class TapasAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = TapasSelfAttention(config)
        self.output = TapasSelfOutput(config)
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
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.modeling_bert.BertIntermediate
class TapasIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.modeling_bert.BertOutput
class TapasOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.modeling_bert.BertLayer with Bert->Tapas
class TapasLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = TapasAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = TapasAttention(config)
        self.intermediate = TapasIntermediate(config)
        self.output = TapasOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.modeling_bert.BertEncoder with Bert->Tapas
class TapasEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([TapasLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


# Copied from transformers.modeling_bert.BertPooler
class TapasPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class TapasPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = TapasConfig
    base_model_prefix = "tapas"

    # Copied from transformers.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


TAPAS_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its models (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.TapasConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

TAPAS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`~transformers.TapasTokenizer`.
            See :meth:`transformers.PreTrainedTokenizer.encode` and
            :meth:`transformers.PreTrainedTokenizer.__call__` for details.
            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0}, 7)`, `optional`):
            Token indices that encode tabular structure. Indices can be obtained using :class:`~transformers.TapasTokenizer`. See this class for more info. 
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
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
    "The bare Tapas Model transformer outputting raw hidden-states without any specific head on top.",
    TAPAS_START_DOCSTRING,
)
class TapasModel(TapasPreTrainedModel):
    """
    This class is a small change compared to :class:`~transformers.BertModel`, taking into account the additional token type ids.

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    config_class = TapasConfig
    base_model_prefix = "tapas"

    def __init__(self, config):
        requires_scatter(self)
        super().__init__(config)
        self.config = config

        self.embeddings = TapasEmbeddings(config)
        self.encoder = TapasEncoder(config)
        self.pooler = TapasPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Examples::

            >>> from transformers import TapasTokenizer, TapasModel
            >>> import pandas as pd

            >>> tokenizer = TapasTokenizer.from_pretrained('tapas-base-finetuned-wtq')
            >>> model = TapasModel.from_pretrained('tapas-base-finetuned-wtq', return_dict=True)

            >>> data = {'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], 'Age': ["56", "45", "59"], 'Number of movies': ["87", "53", "69"]}
            >>> table = pd.DataFrame.from_dict(data)
            >>> queries = ["How many movies has George Clooney played in?", "How old is he?"]

            >>> inputs = tokenizer(table, queries, return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> last_hidden_states = outputs.last_hidden_state
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                (*input_shape, len(self.config.type_vocab_size)), dtype=torch.long, device=device
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings("""Tapas Model with a `language modeling` head on top. """, TAPAS_START_DOCSTRING)
class TapasForMaskedLM(TapasPreTrainedModel):
    config_class = TapasConfig
    base_model_prefix = "tapas"

    def __init__(self, config):
        super().__init__(config)

        self.tapas = TapasModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    @add_start_docstrings_to_callable(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import TapasTokenizer, TapasForMaskedLM
            >>> import pandas as pd

            >>> tokenizer = TapasTokenizer.from_pretrained('tapas-base')
            >>> model = TapasForMaskedLM.from_pretrained('tapas-base', return_dict=True)

            >>> data = {'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], 'Age': ["56", "45", "59"], 'Number of movies': ["87", "53", "69"]}
            >>> table = pd.DataFrame.from_dict(data)

            >>> inputs = tokenizer(table, "How many [MASK] has George [MASK] played in?", return_tensors="pt")
            >>> labels = tokenizer(table, "How many movies has George Clooney played in?", return_tensors="pt")["input_ids"]
            
            >>> outputs = model(**inputs, labels=labels)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.tapas(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TapasLMHead(nn.Module):
    """Tapas Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


@add_start_docstrings(
    """Tapas Model with a cell selection head and optionally aggregation head on top for question-answering 
    tasks on tables (linear layers on top of the hidden-states output to compute `logits` and optionally `logits_aggregation`), e.g. for SQA, WTQ or WikiSQL tasks. """,
    TAPAS_START_DOCSTRING,
)
class TapasForQuestionAnswering(TapasPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # base model
        self.tapas = TapasModel(config)

        # dropout (only used when training)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # cell selection heads
        if config.init_cell_selection_weights_to_zero:
            # init_cell_selection_weights_to_zero: Whether the initial weights should be
            # set to 0. This ensures that all tokens have the same prior probability.
            self.output_weights = nn.Parameter(torch.zeros(config.hidden_size))
            self.column_output_weights = nn.Parameter(torch.zeros(config.hidden_size))
        else:
            self.output_weights = nn.Parameter(torch.empty(config.hidden_size))
            nn.init.normal_(
                self.output_weights, std=0.02
            )  # here, a truncated normal is used in the original implementation
            self.column_output_weights = nn.Parameter(torch.empty(config.hidden_size))
            nn.init.normal_(
                self.column_output_weights, std=0.02
            )  # here, a truncated normal is used in the original implementation
        self.output_bias = nn.Parameter(torch.zeros([]))
        self.column_output_bias = nn.Parameter(torch.zeros([]))

        # aggregation head
        if config.num_aggregation_labels > 0:
            self.output_weights_agg = nn.Parameter(torch.empty([config.num_aggregation_labels, config.hidden_size]))
            nn.init.normal_(
                self.output_weights_agg, std=0.02
            )  # here, a truncated normal is used in the original implementation
            self.output_bias_agg = nn.Parameter(torch.zeros([config.num_aggregation_labels]))

        self.init_weights()

    @add_start_docstrings_to_callable(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TableQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        table_mask=None,
        label_ids=None,
        aggregation_labels=None,
        answer=None,
        numeric_values=None,
        numeric_values_scale=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        table_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, seq_length)`, `optional`):
            Mask for the table. Indicates which tokens belong to the table (1). Question tokens, table headers and padding are 0.
        label_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, seq_length)`, `optional`):
            Labels per token.
        aggregation_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, )`, `optional`):
            Aggregation function id for every example in the batch for computing the aggregation loss.
            Indices should be in :obj:`[0, ..., config.num_aggregation_labels - 1]`.
            Note: this is called "aggregation_function_id" in the original implementation. 
        answer (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, )`, `optional`):
            Answer for every example in the batch. Nan if there is no scalar answer.
        numeric_values (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_length)`, `optional`):
            Numeric values of every token. Nan for tokens which are not numeric values.
        numeric_values_scale (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_length)`, `optional`):
            Scale of the numeric values of every token.

        Returns:
        
        Examples::

            >>> from transformers import TapasTokenizer, TapasForQuestionAnswering
            >>> import pandas as pd

            >>> tokenizer = TapasTokenizer.from_pretrained('tapas-base-finetuned-wtq')
            >>> model = TapasForQuestionAnswering.from_pretrained('tapas-base-finetuned-wtq', return_dict=True)

            >>> data = {'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], 'Age': ["56", "45", "59"], 'Number of movies': ["87", "53", "69"]}
            >>> table = pd.DataFrame.from_dict(data)
            >>> queries = ["How many movies has George Clooney played in?", "How old is he?"]

            >>> inputs = tokenizer(table, queries, return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> logits = outputs.logits
            >>> logits_aggregation = outputs.logits_aggregation
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.tapas(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # if config.is_training:
        #     sequence_output = self.dropout(sequence_output)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Construct indices for the table.
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                (*input_shape, len(self.config.type_vocab_size)), dtype=torch.long, device=device
            )

        token_types = [
            "segment_ids",
            "column_ids",
            "row_ids",
            "prev_label_ids",
            "column_ranks",
            "inv_column_ranks",
            "numeric_relations",
        ]

        row_ids = token_type_ids[:, :, token_types.index("row_ids")]
        column_ids = token_type_ids[:, :, token_types.index("column_ids")]

        row_index = utils.IndexMap(
            indices=torch.min(row_ids, torch.as_tensor(self.config.max_num_rows - 1, device=row_ids.device)),
            num_segments=self.config.max_num_rows,
            batch_dims=1,
        )
        col_index = utils.IndexMap(
            indices=torch.min(column_ids, torch.as_tensor(self.config.max_num_columns - 1, device=column_ids.device)),
            num_segments=self.config.max_num_columns,
            batch_dims=1,
        )
        cell_index = utils.ProductIndexMap(row_index, col_index)

        # Masks.
        input_shape = input_ids.size() if input_ids is not None else inputs_embeds.size()[:-1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # Table cells only, without question tokens and table headers.
        if table_mask is None:
            table_mask = torch.where(row_ids > 0, torch.ones_like(row_ids), torch.zeros_like(row_ids))
        # torch.FloatTensor[batch_size, seq_length] there's probably a more elegant way to do the 4 lines below
        input_mask_float = attention_mask.type(torch.FloatTensor).to(device)
        table_mask_float = table_mask.type(torch.FloatTensor).to(device)
        # Mask for cells that exist in the table (i.e. that are not padding).
        cell_mask, _ = utils.reduce_mean(input_mask_float, cell_index)

        # Compute logits per token. These are used to select individual cells.
        logits = utils.compute_token_logits(
            sequence_output, self.config.temperature, self.output_weights, self.output_bias
        )

        # Compute logits per column. These are used to select a column.
        column_logits = None
        if self.config.select_one_column:
            column_logits = utils.compute_column_logits(
                sequence_output,
                self.column_output_weights,
                self.column_output_bias,
                cell_index,
                cell_mask,
                self.config.allow_empty_column_selection,
            )

        ########## Aggregation logits ##############
        logits_aggregation = None
        if self.config.num_aggregation_labels > 0:
            logits_aggregation = utils._calculate_aggregation_logits(
                pooled_output, self.output_weights_agg, self.output_bias_agg
            )

        # Total loss calculation
        total_loss = 0.0
        calculate_loss = False
        if label_ids is not None and answer is not None:
            calculate_loss = True
            assert label_ids.shape[0] == answer.shape[0]
            is_supervised = not self.config.num_aggregation_labels > 0 or not self.config.use_answer_as_supervision

            ### Semi-supervised cell selection in case of no aggregation
            #############################################################

            # If the answer (the denotation) appears directly in the table we might
            # select the answer without applying any aggregation function. There are
            # some ambiguous cases, see utils._calculate_aggregate_mask for more info.
            # `aggregate_mask` is 1 for examples where we chose to aggregate and 0
            #  for examples where we chose to select the answer directly.
            # `label_ids` encodes the positions of the answer appearing in the table.
            if is_supervised:
                aggregate_mask = None
            else:
                # <float32>[batch_size]
                aggregate_mask = utils._calculate_aggregate_mask(
                    answer,
                    pooled_output,
                    self.config.cell_select_pref,
                    label_ids,
                    self.output_weights_agg,
                    self.output_bias_agg,
                )

            ### Cell selection log-likelihood
            #################################

            if self.config.average_logits_per_cell:
                logits_per_cell, _ = utils.reduce_mean(logits, cell_index)
                logits = utils.gather(logits_per_cell, cell_index)
            dist_per_token = torch.distributions.Bernoulli(logits=logits)

            # Compute cell selection loss per example.
            selection_loss_per_example = None
            if not self.config.select_one_column:
                weight = torch.where(
                    label_ids == 0,
                    torch.ones_like(label_ids, dtype=torch.float32),
                    self.config.positive_weight * torch.ones_like(label_ids, dtype=torch.float32),
                )
                selection_loss_per_token = -dist_per_token.log_prob(label_ids) * weight
                selection_loss_per_example = torch.sum(selection_loss_per_token * input_mask_float, dim=1) / (
                    torch.sum(input_mask_float, dim=1) + utils.EPSILON_ZERO_DIVISION
                )
            else:
                selection_loss_per_example, logits = utils._single_column_cell_selection_loss(
                    logits, column_logits, label_ids, cell_index, col_index, cell_mask
                )
                dist_per_token = torch.distributions.Bernoulli(logits=logits)

            ### Supervised cell selection
            #############################
            if self.config.span_prediction != "none":
                raise NotImplementedError("Span prediction is not supported right now.")
            elif self.config.disable_per_token_loss:
                pass
            elif is_supervised:
                total_loss += torch.mean(selection_loss_per_example)
            else:
                # For the not supervised case, do not assign loss for cell selection
                total_loss += torch.mean(selection_loss_per_example * (1.0 - aggregate_mask))

            ### Semi-supervised regression loss and supervised loss for aggregations
            ######################f###################################################
            if self.config.num_aggregation_labels > 0:
                # Note that `aggregate_mask` is None if the setting is supervised.
                if aggregation_labels is not None:
                    assert label_ids.shape[0] == aggregation_labels.shape[0]
                    per_example_additional_loss = utils._calculate_aggregation_loss(
                        logits_aggregation, aggregate_mask, aggregation_labels, self.config
                    )
                else:
                    raise ValueError("You have to specify aggregation function ids")

                if self.config.use_answer_as_supervision:
                    if numeric_values is not None and numeric_values_scale is not None:
                        # Add regression loss for numeric answers which require aggregation.
                        answer_loss, large_answer_loss_mask = utils._calculate_regression_loss(
                            answer,
                            aggregate_mask,
                            dist_per_token,
                            numeric_values,
                            numeric_values_scale,
                            table_mask_float,
                            logits_aggregation,
                            self.config,
                        )
                        per_example_additional_loss += answer_loss
                        # Zero loss for examples with answer_loss > cutoff.
                        per_example_additional_loss *= large_answer_loss_mask
                    else:
                        raise ValueError("You have to specify numeric values and numeric values scale")

                total_loss += torch.mean(per_example_additional_loss)

        else:
            # if no label ids provided, set them to zeros in order to properly compute logits
            label_ids = torch.zeros_like(logits)
            _, logits = utils._single_column_cell_selection_loss(
                logits, column_logits, label_ids, cell_index, col_index, cell_mask
            )
        if not return_dict:
            output = (logits, logits_aggregation) + outputs[2:]
            return ((total_loss,) + output) if calculate_loss else output

        return TableQuestionAnsweringOutput(
            loss=total_loss,
            logits=logits,
            logits_aggregation=logits_aggregation,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@add_start_docstrings(
    """Tapas Model with a sequence classification head on top (a linear layer on top of
    the pooled output), e.g. for TabFact (Chen et al., 2020). """,
    TAPAS_START_DOCSTRING,
)
class TapasForSequenceClassification(TapasPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.tapas = TapasModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # classification head
        self.output_weights_cls = nn.Parameter(torch.empty([config.num_labels, config.hidden_size]))
        nn.init.normal_(
            self.output_weights_cls, std=0.02
        )  # here, a truncated normal is used in the original implementation
        self.output_bias_cls = nn.Parameter(torch.zeros([config.num_labels]))

        self.init_weights()

    @add_start_docstrings_to_callable(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            Note: this is called "classification_class_index" in the original implementation. 

        Returns:
        
        Examples::

            >>> from transformers import TapasTokenizer, TapasForSequenceClassification
            >>> import pandas as pd

            >>> tokenizer = TapasTokenizer.from_pretrained('tapas-base-finetuned-tabfact')
            >>> model = TapasForSequenceClassification.from_pretrained('tapas-base-finetuned-tabfact', return_dict=True)

            >>> data = {'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], 'Age': ["56", "45", "59"], 'Number of movies': ["87", "53", "69"]}
            >>> table = pd.DataFrame.from_dict(data)
            >>> queries = ["There is only one actor who is 45 years old", "There are 3 actors having more than 60 movies"]

            >>> inputs = tokenizer(table, queries, return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> logits = outputs.logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.tapas(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        
        ########## Classification logits ###########
        logits_cls = None
        if self.config.num_labels > 0:
            logits_cls = utils.compute_classification_logits(
                pooled_output, self.output_weights_cls, self.output_bias_cls
            )

        ########## Classification loss #############
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits_cls.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits_cls.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits_cls,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits_cls,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
