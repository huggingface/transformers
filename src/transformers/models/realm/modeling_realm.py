# coding=utf-8
# Copyright 2022 The REALM authors and The HuggingFace Inc. team.
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
""" PyTorch REALM model."""

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from packaging import version
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    ModelOutput,
)
from ...modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import logging
from .configuration_realm import RealmConfig


logger = logging.get_logger(__name__)
_EMBEDDER_CHECKPOINT_FOR_DOC = "google/realm-cc-news-pretrained-embedder"
_ENCODER_CHECKPOINT_FOR_DOC = "google/realm-cc-news-pretrained-encoder"
_SCORER_CHECKPOINT_FOR_DOC = "google/realm-cc-news-pretrained-scorer"
_CONFIG_FOR_DOC = "RealmConfig"
_TOKENIZER_FOR_DOC = "RealmTokenizer"

REALM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/realm-cc-news-pretrained-embedder",
    "google/realm-cc-news-pretrained-encoder",
    "google/realm-cc-news-pretrained-scorer",
    "google/realm-cc-news-pretrained-openqa",
    "google/realm-orqa-nq-openqa",
    "google/realm-orqa-nq-reader",
    "google/realm-orqa-wq-openqa",
    "google/realm-orqa-wq-reader",
    # See all REALM models at https://huggingface.co/models?filter=realm
]


def load_tf_weights_in_realm(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
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
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []

    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        if isinstance(model, RealmReader) and "reader" not in name:
            logger.info(f"Skipping {name} as it is not {model.__class__.__name__}'s parameter")
            continue

        # For pretrained openqa reader
        if (name.startswith("bert") or name.startswith("cls")) and isinstance(model, RealmForOpenQA):
            name = name.replace("bert/", "reader/realm/")
            name = name.replace("cls/", "reader/cls/")

        # For pretrained encoder
        if (name.startswith("bert") or name.startswith("cls")) and isinstance(model, RealmKnowledgeAugEncoder):
            name = name.replace("bert/", "realm/")

        # For finetuned reader
        if name.startswith("reader"):
            reader_prefix = "" if isinstance(model, RealmReader) else "reader/"
            name = name.replace("reader/module/bert/", f"{reader_prefix}realm/")
            name = name.replace("reader/module/cls/", f"{reader_prefix}cls/")
            name = name.replace("reader/dense/", f"{reader_prefix}qa_outputs/dense_intermediate/")
            name = name.replace("reader/dense_1/", f"{reader_prefix}qa_outputs/dense_output/")
            name = name.replace("reader/layer_normalization", f"{reader_prefix}qa_outputs/layer_normalization")

        # For embedder and scorer
        if name.startswith("module/module/module/"):  # finetuned
            embedder_prefix = "" if isinstance(model, RealmEmbedder) else "embedder/"
            name = name.replace("module/module/module/module/bert/", f"{embedder_prefix}realm/")
            name = name.replace("module/module/module/LayerNorm/", f"{embedder_prefix}cls/LayerNorm/")
            name = name.replace("module/module/module/dense/", f"{embedder_prefix}cls/dense/")
            name = name.replace("module/module/module/module/cls/predictions/", f"{embedder_prefix}cls/predictions/")
            name = name.replace("module/module/module/bert/", f"{embedder_prefix}realm/")
            name = name.replace("module/module/module/cls/predictions/", f"{embedder_prefix}cls/predictions/")
        elif name.startswith("module/module/"):  # pretrained
            embedder_prefix = "" if isinstance(model, RealmEmbedder) else "embedder/"
            name = name.replace("module/module/LayerNorm/", f"{embedder_prefix}cls/LayerNorm/")
            name = name.replace("module/module/dense/", f"{embedder_prefix}cls/dense/")

        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
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
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


# Copied from transformers.models.bert.modeling_bert.BertEmbeddings with Bert->Realm
class RealmEmbeddings(nn.Module):
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
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

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


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Realm
class RealmSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

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
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RealmModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

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

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->Realm
class RealmSelfOutput(nn.Module):
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


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Realm
class RealmAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = RealmSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = RealmSelfOutput(config)
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
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Realm
class RealmIntermediate(nn.Module):
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


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->Realm
class RealmOutput(nn.Module):
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


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Realm
class RealmLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RealmAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RealmAttention(config, position_embedding_type="absolute")
        self.intermediate = RealmIntermediate(config)
        self.output = RealmOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Realm
class RealmEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RealmLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

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
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->Realm
class RealmPooler(nn.Module):
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


@dataclass
class RealmEmbedderOutput(ModelOutput):
    """
    Outputs of [`RealmEmbedder`] models.

    Args:
        projected_score (`torch.FloatTensor` of shape `(batch_size, config.retriever_proj_size)`):

            Projected score.
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

    projected_score: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class RealmScorerOutput(ModelOutput):
    """
    Outputs of [`RealmScorer`] models.

    Args:
        relevance_score (`torch.FloatTensor` of shape `(batch_size, config.num_candidates)`):
            The relevance score of document candidates (before softmax).
        query_score (`torch.FloatTensor` of shape `(batch_size, config.retriever_proj_size)`):
            Query score derived from the query embedder.
        candidate_score (`torch.FloatTensor` of shape `(batch_size, config.num_candidates, config.retriever_proj_size)`):
            Candidate score derived from the embedder.
    """

    relevance_score: torch.FloatTensor = None
    query_score: torch.FloatTensor = None
    candidate_score: torch.FloatTensor = None


@dataclass
class RealmReaderOutput(ModelOutput):
    """
    Outputs of [`RealmReader`] models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided):
            Total loss.
        retriever_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided):
            Retriever loss.
        reader_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided):
            Reader loss.
        retriever_correct (`torch.BoolTensor` of shape `(config.searcher_beam_size,)`, *optional*):
            Whether or not an evidence block contains answer.
        reader_correct (`torch.BoolTensor` of shape `(config.reader_beam_size, num_candidates)`, *optional*):
            Whether or not a span candidate contains answer.
        block_idx (`torch.LongTensor` of shape `()`):
            The index of the retrieved evidence block in which the predicted answer is most likely.
        candidate (`torch.LongTensor` of shape `()`):
            The index of the retrieved span candidates in which the predicted answer is most likely.
        start_pos (`torch.IntTensor` of shape `()`):
            Predicted answer starting position in *RealmReader*'s inputs.
        end_pos: (`torch.IntTensor` of shape `()`):
            Predicted answer ending position in *RealmReader*'s inputs.
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

    loss: torch.FloatTensor = None
    retriever_loss: torch.FloatTensor = None
    reader_loss: torch.FloatTensor = None
    retriever_correct: torch.BoolTensor = None
    reader_correct: torch.BoolTensor = None
    block_idx: torch.LongTensor = None
    candidate: torch.LongTensor = None
    start_pos: torch.int32 = None
    end_pos: torch.int32 = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class RealmForOpenQAOutput(ModelOutput):
    """

    Outputs of [`RealmForOpenQA`] models.

    Args:
        reader_output (`dict`):
            Reader output.
        predicted_answer_ids (`torch.LongTensor` of shape `(answer_sequence_length)`):
            Predicted answer ids.
    """

    reader_output: dict = None
    predicted_answer_ids: torch.LongTensor = None


class RealmPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class RealmLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = RealmPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class RealmOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = RealmLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class RealmScorerProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = RealmLMPredictionHead(config)
        self.dense = nn.Linear(config.hidden_size, config.retriever_proj_size)
        self.LayerNorm = nn.LayerNorm(config.retriever_proj_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class RealmReaderProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense_intermediate = nn.Linear(config.hidden_size, config.span_hidden_size * 2)
        self.dense_output = nn.Linear(config.span_hidden_size, 1)
        self.layer_normalization = nn.LayerNorm(config.span_hidden_size, eps=config.reader_layer_norm_eps)
        self.relu = nn.ReLU()

    def forward(self, hidden_states, token_type_ids):
        def span_candidates(masks):
            """
            Generate span candidates.

            Args:
                masks: <int32> [num_retrievals, max_sequence_len]

            Returns:
                starts: <int32> [num_spans] ends: <int32> [num_spans] span_masks: <int32> [num_retrievals, num_spans]
                whether spans locate in evidence block.
            """
            _, max_sequence_len = masks.shape

            def _spans_given_width(width):
                current_starts = torch.arange(max_sequence_len - width + 1, device=masks.device)
                current_ends = torch.arange(width - 1, max_sequence_len, device=masks.device)
                return current_starts, current_ends

            starts, ends = zip(*(_spans_given_width(w + 1) for w in range(self.config.max_span_width)))

            # [num_spans]
            starts = torch.cat(starts, 0)
            ends = torch.cat(ends, 0)

            # [num_retrievals, num_spans]
            start_masks = torch.index_select(masks, dim=-1, index=starts)
            end_masks = torch.index_select(masks, dim=-1, index=ends)
            span_masks = start_masks * end_masks

            return starts, ends, span_masks

        def mask_to_score(mask):
            return (1.0 - mask.type(torch.float32)) * -10000.0

        # [reader_beam_size, max_sequence_len, span_hidden_size * 2]
        hidden_states = self.dense_intermediate(hidden_states)
        # [reader_beam_size, max_sequence_len, span_hidden_size]
        start_projection, end_projection = hidden_states.chunk(2, dim=-1)
        block_mask = token_type_ids.detach().clone()
        block_mask[:, -1] = 0
        candidate_starts, candidate_ends, candidate_mask = span_candidates(block_mask)

        candidate_start_projections = torch.index_select(start_projection, dim=1, index=candidate_starts)
        candidate_end_projections = torch.index_select(end_projection, dim=1, index=candidate_ends)
        candidate_hidden = candidate_start_projections + candidate_end_projections

        # [reader_beam_size, num_candidates, span_hidden_size]
        candidate_hidden = self.relu(candidate_hidden)
        # [reader_beam_size, num_candidates, span_hidden_size]
        candidate_hidden = self.layer_normalization(candidate_hidden)
        # [reader_beam_size, num_candidates]
        reader_logits = self.dense_output(candidate_hidden).squeeze(-1)
        # [reader_beam_size, num_candidates]
        reader_logits += mask_to_score(candidate_mask)

        return reader_logits, candidate_starts, candidate_ends


REALM_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`RealmConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

REALM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`RealmTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


class RealmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RealmConfig
    load_tf_weights = load_tf_weights_in_realm
    base_model_prefix = "realm"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
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

    def _flatten_inputs(self, *inputs):
        """Flatten inputs' shape to (-1, input_shape[-1])"""
        flattened_inputs = []
        for tensor in inputs:
            if tensor is None:
                flattened_inputs.append(None)
            else:
                input_shape = tensor.shape
                if len(input_shape) > 2:
                    tensor = tensor.view((-1, input_shape[-1]))
                flattened_inputs.append(tensor)
        return flattened_inputs


class RealmBertModel(RealmPreTrainedModel):
    """
    Same as the original BertModel but remove docstrings.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RealmEmbeddings(config)
        self.encoder = RealmEncoder(config)

        self.pooler = RealmPooler(config) if add_pooling_layer else None

        # Weights initialization is mostly managed by other Realm models,
        # but we also have them initialized here to keep a consistency.
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

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
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
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
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    "The embedder of REALM outputting projected score that will be used to calculate relevance score.",
    REALM_START_DOCSTRING,
)
class RealmEmbedder(RealmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.realm = RealmBertModel(self.config)
        self.cls = RealmScorerProjection(self.config)
        self.post_init()

    def get_input_embeddings(self):
        return self.realm.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.realm.embeddings.word_embeddings = value

    @add_start_docstrings_to_model_forward(REALM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=RealmEmbedderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import RealmTokenizer, RealmEmbedder
        >>> import torch

        >>> tokenizer = RealmTokenizer.from_pretrained("google/realm-cc-news-pretrained-embedder")
        >>> model = RealmEmbedder.from_pretrained("google/realm-cc-news-pretrained-embedder")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> projected_score = outputs.projected_score
        ```
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        realm_outputs = self.realm(
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

        # [batch_size, hidden_size]
        pooler_output = realm_outputs[1]
        # [batch_size, retriever_proj_size]
        projected_score = self.cls(pooler_output)

        if not return_dict:
            return (projected_score,) + realm_outputs[2:4]
        else:
            return RealmEmbedderOutput(
                projected_score=projected_score,
                hidden_states=realm_outputs.hidden_states,
                attentions=realm_outputs.attentions,
            )


@add_start_docstrings(
    "The scorer of REALM outputting relevance scores representing the score of document candidates (before softmax).",
    REALM_START_DOCSTRING,
)
class RealmScorer(RealmPreTrainedModel):
    r"""
    Args:
        query_embedder ([`RealmEmbedder`]):
            Embedder for input sequences. If not specified, it will use the same embedder as candidate sequences.
    """

    def __init__(self, config, query_embedder=None):
        super().__init__(config)

        self.embedder = RealmEmbedder(self.config)

        self.query_embedder = query_embedder if query_embedder is not None else self.embedder

        self.post_init()

    @add_start_docstrings_to_model_forward(REALM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=RealmScorerOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        candidate_input_ids=None,
        candidate_attention_mask=None,
        candidate_token_type_ids=None,
        candidate_inputs_embeds=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        candidate_input_ids (`torch.LongTensor` of shape `(batch_size, num_candidates, sequence_length)`):
            Indices of candidate input sequence tokens in the vocabulary.

            Indices can be obtained using [`RealmTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        candidate_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_candidates, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        candidate_token_type_ids (`torch.LongTensor` of shape `(batch_size, num_candidates, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        candidate_inputs_embeds (`torch.FloatTensor` of shape `(batch_size * num_candidates, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `candidate_input_ids` you can choose to directly pass an embedded
            representation. This is useful if you want more control over how to convert *candidate_input_ids* indices
            into associated vectors than the model's internal embedding lookup matrix.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import RealmTokenizer, RealmScorer

        >>> tokenizer = RealmTokenizer.from_pretrained("google/realm-cc-news-pretrained-scorer")
        >>> model = RealmScorer.from_pretrained("google/realm-cc-news-pretrained-scorer", num_candidates=2)

        >>> # batch_size = 2, num_candidates = 2
        >>> input_texts = ["How are you?", "What is the item in the picture?"]
        >>> candidates_texts = [["Hello world!", "Nice to meet you!"], ["A cute cat.", "An adorable dog."]]

        >>> inputs = tokenizer(input_texts, return_tensors="pt")
        >>> candidates_inputs = tokenizer.batch_encode_candidates(candidates_texts, max_length=10, return_tensors="pt")

        >>> outputs = model(
        ...     **inputs,
        ...     candidate_input_ids=candidates_inputs.input_ids,
        ...     candidate_attention_mask=candidates_inputs.attention_mask,
        ...     candidate_token_type_ids=candidates_inputs.token_type_ids,
        ... )
        >>> relevance_score = outputs.relevance_score
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or input_embeds.")

        if candidate_input_ids is None and candidate_inputs_embeds is None:
            raise ValueError("You have to specify either candidate_input_ids or candidate_inputs_embeds.")

        query_outputs = self.query_embedder(
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

        # [batch_size * num_candidates, candidate_seq_len]
        (flattened_input_ids, flattened_attention_mask, flattened_token_type_ids) = self._flatten_inputs(
            candidate_input_ids, candidate_attention_mask, candidate_token_type_ids
        )

        candidate_outputs = self.embedder(
            flattened_input_ids,
            attention_mask=flattened_attention_mask,
            token_type_ids=flattened_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=candidate_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # [batch_size, retriever_proj_size]
        query_score = query_outputs[0]
        # [batch_size * num_candidates, retriever_proj_size]
        candidate_score = candidate_outputs[0]
        # [batch_size, num_candidates, retriever_proj_size]
        candidate_score = candidate_score.view(-1, self.config.num_candidates, self.config.retriever_proj_size)
        # [batch_size, num_candidates]
        relevance_score = torch.einsum("BD,BND->BN", query_score, candidate_score)

        if not return_dict:
            return relevance_score, query_score, candidate_score

        return RealmScorerOutput(
            relevance_score=relevance_score, query_score=query_score, candidate_score=candidate_score
        )


@add_start_docstrings(
    "The knowledge-augmented encoder of REALM outputting masked language model logits and marginal log-likelihood loss.",
    REALM_START_DOCSTRING,
)
class RealmKnowledgeAugEncoder(RealmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.realm = RealmBertModel(self.config)
        self.cls = RealmOnlyMLMHead(self.config)
        self.post_init()

    def get_input_embeddings(self):
        return self.realm.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.realm.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(
        REALM_INPUTS_DOCSTRING.format("batch_size, num_candidates, sequence_length")
    )
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        relevance_score=None,
        labels=None,
        mlm_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        relevance_score (`torch.FloatTensor` of shape `(batch_size, num_candidates)`, *optional*):
            Relevance score derived from RealmScorer, must be specified if you want to compute the masked language
            modeling loss.

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        mlm_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid calculating joint loss on certain positions. If not specified, the loss will not be masked.
            Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import RealmTokenizer, RealmKnowledgeAugEncoder

        >>> tokenizer = RealmTokenizer.from_pretrained("google/realm-cc-news-pretrained-encoder")
        >>> model = RealmKnowledgeAugEncoder.from_pretrained(
        ...     "google/realm-cc-news-pretrained-encoder", num_candidates=2
        ... )

        >>> # batch_size = 2, num_candidates = 2
        >>> text = [["Hello world!", "Nice to meet you!"], ["The cute cat.", "The adorable dog."]]

        >>> inputs = tokenizer.batch_encode_candidates(text, max_length=10, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        (flattened_input_ids, flattened_attention_mask, flattened_token_type_ids) = self._flatten_inputs(
            input_ids, attention_mask, token_type_ids
        )

        joint_outputs = self.realm(
            flattened_input_ids,
            attention_mask=flattened_attention_mask,
            token_type_ids=flattened_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # [batch_size * num_candidates, joint_seq_len, hidden_size]
        joint_output = joint_outputs[0]
        # [batch_size * num_candidates, joint_seq_len, vocab_size]
        prediction_scores = self.cls(joint_output)
        # [batch_size, num_candidates]
        candidate_score = relevance_score

        masked_lm_loss = None
        if labels is not None:
            if candidate_score is None:
                raise ValueError(
                    "You have to specify `relevance_score` when `labels` is specified in order to compute loss."
                )

            batch_size, seq_length = labels.size()

            if mlm_mask is None:
                mlm_mask = torch.ones_like(labels, dtype=torch.float32)
            else:
                mlm_mask = mlm_mask.type(torch.float32)

            # Compute marginal log-likelihood
            loss_fct = CrossEntropyLoss(reduction="none")  # -100 index = padding token

            # [batch_size * num_candidates * joint_seq_len, vocab_size]
            mlm_logits = prediction_scores.view(-1, self.config.vocab_size)
            # [batch_size * num_candidates * joint_seq_len]
            mlm_targets = labels.tile(1, self.config.num_candidates).view(-1)
            # [batch_size, num_candidates, joint_seq_len]
            masked_lm_log_prob = -loss_fct(mlm_logits, mlm_targets).view(
                batch_size, self.config.num_candidates, seq_length
            )
            # [batch_size, num_candidates, 1]
            candidate_log_prob = candidate_score.log_softmax(-1).unsqueeze(-1)
            # [batch_size, num_candidates, joint_seq_len]
            joint_gold_log_prob = candidate_log_prob + masked_lm_log_prob
            # [batch_size, joint_seq_len]
            marginal_gold_log_probs = joint_gold_log_prob.logsumexp(1)
            # []
            masked_lm_loss = -torch.nansum(torch.sum(marginal_gold_log_probs * mlm_mask) / torch.sum(mlm_mask))

        if not return_dict:
            output = (prediction_scores,) + joint_outputs[2:4]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=joint_outputs.hidden_states,
            attentions=joint_outputs.attentions,
        )


@add_start_docstrings("The reader of REALM.", REALM_START_DOCSTRING)
class RealmReader(RealmPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler", "cls"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.realm = RealmBertModel(config)
        self.cls = RealmOnlyMLMHead(config)
        self.qa_outputs = RealmReaderProjection(config)

        self.post_init()

    @add_start_docstrings_to_model_forward(REALM_INPUTS_DOCSTRING.format("reader_beam_size, sequence_length"))
    @replace_return_docstrings(output_type=RealmReaderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        relevance_score=None,
        start_positions=None,
        end_positions=None,
        has_answers=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        relevance_score (`torch.FloatTensor` of shape `(searcher_beam_size,)`, *optional*):
            Relevance score, which must be specified if you want to compute the marginal log loss.
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        has_answers (`torch.BoolTensor` of shape `(searcher_beam_size,)`, *optional*):
            Whether or not the evidence block has answer(s).

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if relevance_score is None:
            raise ValueError("You have to specify `relevance_score` to calculate logits and loss.")
        if token_type_ids is None:
            raise ValueError("You have to specify `token_type_ids` to separate question block and evidence block.")
        if token_type_ids.size(1) < self.config.max_span_width:
            raise ValueError("The input sequence length must be greater than or equal to config.max_span_width.")
        outputs = self.realm(
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

        # [reader_beam_size, joint_seq_len, hidden_size]
        sequence_output = outputs[0]

        # [reader_beam_size, num_candidates], [num_candidates], [num_candidates]
        reader_logits, candidate_starts, candidate_ends = self.qa_outputs(sequence_output, token_type_ids)
        # [searcher_beam_size, 1]
        retriever_logits = torch.unsqueeze(relevance_score[0 : self.config.reader_beam_size], -1)
        # [reader_beam_size, num_candidates]
        reader_logits += retriever_logits
        # []
        predicted_block_index = torch.argmax(torch.max(reader_logits, dim=1).values)
        # []
        predicted_candidate = torch.argmax(torch.max(reader_logits, dim=0).values)
        # [1]
        predicted_start = torch.index_select(candidate_starts, dim=0, index=predicted_candidate)
        # [1]
        predicted_end = torch.index_select(candidate_ends, dim=0, index=predicted_candidate)

        total_loss = None
        retriever_loss = None
        reader_loss = None
        retriever_correct = None
        reader_correct = None
        if start_positions is not None and end_positions is not None and has_answers is not None:

            def compute_correct_candidates(candidate_starts, candidate_ends, gold_starts, gold_ends):
                """Compute correct span."""
                # [reader_beam_size, num_answers, num_candidates]
                is_gold_start = torch.eq(
                    torch.unsqueeze(torch.unsqueeze(candidate_starts, 0), 0), torch.unsqueeze(gold_starts, -1)
                )
                is_gold_end = torch.eq(
                    torch.unsqueeze(torch.unsqueeze(candidate_ends, 0), 0), torch.unsqueeze(gold_ends, -1)
                )

                # [reader_beam_size, num_candidates]
                return torch.any(torch.logical_and(is_gold_start, is_gold_end), 1)

            def marginal_log_loss(logits, is_correct):
                """Loss based on the negative marginal log-likelihood."""

                def mask_to_score(mask):
                    return (1.0 - mask.type(torch.float32)) * -10000.0

                # []
                log_numerator = torch.logsumexp(logits + mask_to_score(is_correct), dim=-1)
                log_denominator = torch.logsumexp(logits, dim=-1)
                return log_denominator - log_numerator

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            # `-1` is reserved for no answer.
            ignored_index = sequence_output.size(1)
            start_positions = start_positions.clamp(-1, ignored_index)
            end_positions = end_positions.clamp(-1, ignored_index)

            retriever_correct = has_answers
            any_retriever_correct = torch.any(retriever_correct)

            reader_correct = compute_correct_candidates(
                candidate_starts=candidate_starts,
                candidate_ends=candidate_ends,
                gold_starts=start_positions[0 : self.config.reader_beam_size],
                gold_ends=end_positions[0 : self.config.reader_beam_size],
            )
            any_reader_correct = torch.any(reader_correct)

            retriever_loss = marginal_log_loss(relevance_score, retriever_correct)
            reader_loss = marginal_log_loss(reader_logits.view(-1), reader_correct.view(-1))
            retriever_loss *= any_retriever_correct.type(torch.float32)
            reader_loss *= any_reader_correct.type(torch.float32)

            total_loss = (retriever_loss + reader_loss).mean()

        if not return_dict:
            output = (predicted_block_index, predicted_candidate, predicted_start, predicted_end) + outputs[2:]
            return (
                ((total_loss, retriever_loss, reader_loss, retriever_correct, reader_correct) + output)
                if total_loss is not None
                else output
            )

        return RealmReaderOutput(
            loss=total_loss,
            retriever_loss=retriever_loss,
            reader_loss=reader_loss,
            retriever_correct=retriever_correct,
            reader_correct=reader_correct,
            block_idx=predicted_block_index,
            candidate=predicted_candidate,
            start_pos=predicted_start,
            end_pos=predicted_end,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


REALM_FOR_OPEN_QA_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`RealmTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token (should not be used in this model by design).

            [What are token type IDs?](../glossary#token-type-ids)
        answer_ids (`list` of shape `(num_answers, answer_length)`, *optional*):
            Answer ids for computing the marginal log-likelihood loss. Indices should be in `[-1, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-1` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "`RealmForOpenQA` for end-to-end open domain question answering.",
    REALM_START_DOCSTRING,
)
class RealmForOpenQA(RealmPreTrainedModel):
    def __init__(self, config, retriever=None):
        super().__init__(config)
        self.embedder = RealmEmbedder(config)
        self.reader = RealmReader(config)
        self.register_buffer(
            "block_emb",
            torch.zeros(()).new_empty(
                size=(config.num_block_records, config.retriever_proj_size),
                dtype=torch.float32,
                device=torch.device("cpu"),
            ),
        )
        self.retriever = retriever

        self.post_init()

    @property
    def beam_size(self):
        if self.training:
            return self.config.searcher_beam_size
        return self.config.reader_beam_size

    @add_start_docstrings_to_model_forward(REALM_FOR_OPEN_QA_DOCSTRING.format("1, sequence_length"))
    @replace_return_docstrings(output_type=RealmForOpenQAOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        answer_ids=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import RealmForOpenQA, RealmRetriever, RealmTokenizer

        >>> retriever = RealmRetriever.from_pretrained("google/realm-orqa-nq-openqa")
        >>> tokenizer = RealmTokenizer.from_pretrained("google/realm-orqa-nq-openqa")
        >>> model = RealmForOpenQA.from_pretrained("google/realm-orqa-nq-openqa", retriever=retriever)

        >>> question = "Who is the pioneer in modern computer science?"
        >>> question_ids = tokenizer([question], return_tensors="pt")
        >>> answer_ids = tokenizer(
        ...     ["alan mathison turing"],
        ...     add_special_tokens=False,
        ...     return_token_type_ids=False,
        ...     return_attention_mask=False,
        >>> ).input_ids

        >>> reader_output, predicted_answer_ids = model(**question_ids, answer_ids=answer_ids, return_dict=False)
        >>> predicted_answer = tokenizer.decode(predicted_answer_ids)
        >>> loss = reader_output.loss
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and input_ids.shape[0] != 1:
            raise ValueError("The batch_size of the inputs must be 1.")

        question_outputs = self.embedder(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=True
        )

        # [1, projection_size]
        question_projection = question_outputs[0]
        # [1, block_emb_size]
        batch_scores = torch.einsum("BD,QD->QB", self.block_emb, question_projection)
        # [1, searcher_beam_size]
        _, retrieved_block_ids = torch.topk(batch_scores, k=self.beam_size, dim=-1)
        # [searcher_beam_size]
        # Must convert to cpu tensor for subsequent numpy operations
        retrieved_block_ids = retrieved_block_ids.squeeze().cpu()

        # Retrieve possible answers
        has_answers, start_pos, end_pos, concat_inputs = self.retriever(
            retrieved_block_ids, input_ids, answer_ids, max_length=self.config.reader_seq_len
        )

        if has_answers is not None:
            has_answers = torch.tensor(has_answers, dtype=torch.bool, device=self.reader.device)
            start_pos = torch.tensor(start_pos, dtype=torch.long, device=self.reader.device)
            end_pos = torch.tensor(end_pos, dtype=torch.long, device=self.reader.device)

        concat_inputs = concat_inputs.to(self.reader.device)

        # [searcher_beam_size, projection_size]
        retrieved_block_emb = torch.index_select(
            self.block_emb, dim=0, index=retrieved_block_ids.to(self.block_emb.device)
        )
        # [searcher_beam_size]
        retrieved_logits = torch.einsum(
            "D,BD->B", question_projection.squeeze(), retrieved_block_emb.to(question_projection.device)
        )

        reader_output = self.reader(
            input_ids=concat_inputs.input_ids[0 : self.config.reader_beam_size],
            attention_mask=concat_inputs.attention_mask[0 : self.config.reader_beam_size],
            token_type_ids=concat_inputs.token_type_ids[0 : self.config.reader_beam_size],
            relevance_score=retrieved_logits,
            has_answers=has_answers,
            start_positions=start_pos,
            end_positions=end_pos,
            return_dict=True,
        )

        predicted_block = concat_inputs.input_ids[reader_output.block_idx]
        predicted_answer_ids = predicted_block[reader_output.start_pos : reader_output.end_pos + 1]

        if not return_dict:
            return reader_output, predicted_answer_ids

        return RealmForOpenQAOutput(
            reader_output=reader_output,
            predicted_answer_ids=predicted_answer_ids,
        )
