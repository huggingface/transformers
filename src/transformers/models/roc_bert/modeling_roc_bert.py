# coding=utf-8
# Copyright 2022 WeChatAI The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch RoCBert model."""

from collections.abc import Callable
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...generation import GenerationMixin
from ...masking_utils import create_bidirectional_mask, create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple, check_model_inputs
from .configuration_roc_bert import RoCBertConfig


logger = logging.get_logger(__name__)


class RoCBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position, shape, pronunciation and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pronunciation_embed = nn.Embedding(
            config.pronunciation_vocab_size, config.pronunciation_embed_dim, padding_idx=config.pad_token_id
        )
        self.shape_embed = nn.Embedding(
            config.shape_vocab_size, config.shape_embed_dim, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.enable_pronunciation = config.enable_pronunciation
        self.enable_shape = config.enable_shape

        if config.concat_input:
            input_dim = config.hidden_size
            if self.enable_pronunciation:
                pronunciation_dim = config.pronunciation_embed_dim
                input_dim += pronunciation_dim
            if self.enable_shape:
                shape_dim = config.shape_embed_dim
                input_dim += shape_dim
            self.map_inputs_layer = torch.nn.Linear(input_dim, config.hidden_size)
        else:
            self.map_inputs_layer = None

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

    def forward(
        self,
        input_ids=None,
        input_shape_ids=None,
        input_pronunciation_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                # NOTE: We assume either pos ids to have bsz == 1 (broadcastable) or bsz == effective bsz (input_shape[0])
                buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
                buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
                token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if self.map_inputs_layer is None:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = inputs_embeds + token_type_embeddings
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)

            denominator = 1
            embedding_in = torch.clone(embeddings)
            if self.enable_shape and input_shape_ids is not None:
                embedding_shape = self.shape_embed(input_shape_ids)
                embedding_in += embedding_shape
                denominator += 1
            if self.enable_pronunciation and input_pronunciation_ids is not None:
                embedding_pronunciation = self.pronunciation_embed(input_pronunciation_ids)
                embedding_in += embedding_pronunciation
                denominator += 1

            embedding_in /= denominator
            return embedding_in
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)  # embedding_word
            device = inputs_embeds.device

            embedding_in = torch.clone(inputs_embeds)
            if self.enable_shape:
                if input_shape_ids is None:
                    input_shape_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
                embedding_shape = self.shape_embed(input_shape_ids)
                embedding_in = torch.cat((embedding_in, embedding_shape), -1)
            if self.enable_pronunciation:
                if input_pronunciation_ids is None:
                    input_pronunciation_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
                embedding_pronunciation = self.pronunciation_embed(input_pronunciation_ids)
                embedding_in = torch.cat((embedding_in, embedding_pronunciation), -1)

            embedding_in = self.map_inputs_layer(embedding_in)  # batch_size * seq_len * hidden_dim

            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embedding_in += token_type_embeddings
            position_embeddings = self.position_embeddings(position_ids)
            embedding_in += position_embeddings

            embedding_in = self.LayerNorm(embedding_in)
            embedding_in = self.dropout(embedding_in)
            return embedding_in


# Copied from transformers.models.bert.modeling_bert.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->RoCBert
class RoCBertSelfAttention(nn.Module):
    def __init__(self, config, is_causal=False, layer_idx=None):
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
        self.scaling = self.attention_head_size**-0.5

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder
        self.is_causal = is_causal
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        # get all proj
        query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            # decoder-only bert can have a simple dynamic cache for example
            current_past_key_values = past_key_values
            if isinstance(past_key_values, EncoderDecoderCache):
                current_past_key_values = past_key_values.self_attention_cache

            # save all key/value_layer to cache to be re-used for fast auto-regressive generation
            key_layer, value_layer = current_past_key_values.update(
                key_layer,
                value_layer,
                self.layer_idx,
                {"cache_position": cache_position},
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return attn_output, attn_weights


# Copied from transformers.models.bert.modeling_bert.BertCrossAttention with Bert->RoCBert
class RoCBertCrossAttention(nn.Module):
    def __init__(self, config, is_causal=False, layer_idx=None):
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
        self.scaling = self.attention_head_size**-0.5

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.is_causal = is_causal
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        # determine input shapes
        bsz, tgt_len = hidden_states.shape[:-1]
        src_len = encoder_hidden_states.shape[1]

        q_input_shape = (bsz, tgt_len, -1, self.attention_head_size)
        kv_input_shape = (bsz, src_len, -1, self.attention_head_size)

        # get query proj
        query_layer = self.query(hidden_states).view(*q_input_shape).transpose(1, 2)

        is_updated = past_key_values.is_updated.get(self.layer_idx) if past_key_values is not None else False
        if past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_layer = past_key_values.cross_attention_cache.layers[self.layer_idx].keys
            value_layer = past_key_values.cross_attention_cache.layers[self.layer_idx].values
        else:
            key_layer = self.key(encoder_hidden_states).view(*kv_input_shape).transpose(1, 2)
            value_layer = self.value(encoder_hidden_states).view(*kv_input_shape).transpose(1, 2)

            if past_key_values is not None:
                # save all states to the cache
                key_layer, value_layer = past_key_values.cross_attention_cache.update(
                    key_layer, value_layer, self.layer_idx
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                past_key_values.is_updated[self.layer_idx] = True

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        return attn_output, attn_weights


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->RoCBert
class RoCBertSelfOutput(nn.Module):
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


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->RoCBert,BERT->ROC_BERT
class RoCBertAttention(nn.Module):
    def __init__(self, config, is_causal=False, layer_idx=None, is_cross_attention=False):
        super().__init__()
        self.is_cross_attention = is_cross_attention
        attention_class = RoCBertCrossAttention if is_cross_attention else RoCBertSelfAttention
        self.self = attention_class(config, is_causal=is_causal, layer_idx=layer_idx)
        self.output = RoCBertSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        attention_mask = attention_mask if not self.is_cross_attention else encoder_attention_mask
        attention_output, attn_weights = self.self(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        attention_output = self.output(attention_output, hidden_states)
        return attention_output, attn_weights


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->RoCBert
class RoCBertIntermediate(nn.Module):
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


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->RoCBert
class RoCBertOutput(nn.Module):
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


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->RoCBert
class RoCBertLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RoCBertAttention(config, is_causal=config.is_decoder, layer_idx=layer_idx)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RoCBertAttention(
                config,
                is_causal=False,
                layer_idx=layer_idx,
                is_cross_attention=True,
            )
        self.intermediate = RoCBertIntermediate(config)
        self.output = RoCBertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        self_attention_output, _ = self.attention(
            hidden_states,
            attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        attention_output = self_attention_output

        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            cross_attention_output, _ = self.crossattention(
                self_attention_output,
                None,  # attention_mask
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
            attention_output = cross_attention_output

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->RoCBert
class RoCBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RoCBertLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->RoCBert
class RoCBertPooler(nn.Module):
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


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->RoCBert
class RoCBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->RoCBert
class RoCBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = RoCBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->RoCBert
class RoCBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = RoCBertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


@auto_docstring
class RoCBertPreTrainedModel(PreTrainedModel):
    config_class = RoCBertConfig
    base_model_prefix = "roc_bert"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": RoCBertLayer,
        "attentions": RoCBertSelfAttention,
        "cross_attentions": RoCBertCrossAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, RoCBertLMPredictionHead):
            init.zeros_(module.bias)


@auto_docstring(
    custom_intro="""

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://huggingface.co/papers/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to be initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """
)
class RoCBertModel(RoCBertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        self.embeddings = RoCBertEmbeddings(config)
        self.encoder = RoCBertEncoder(config)

        self.pooler = RoCBertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bert.modeling_bert.BertModel.get_input_embeddings
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # Copied from transformers.models.bert.modeling_bert.BertModel.set_input_embeddings
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_pronunciation_embeddings(self):
        return self.embeddings.pronunciation_embed

    def set_pronunciation_embeddings(self, value):
        self.embeddings.pronunciation_embed = value

    def get_shape_embeddings(self):
        return self.embeddings.shape_embed

    def set_shape_embeddings(self, value):
        self.embeddings.shape_embed = value

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_shape_ids: Optional[torch.Tensor] = None,
        input_pronunciation_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        input_shape_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the shape vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input_shape_ids)
        input_pronunciation_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the pronunciation vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input_pronunciation_ids)
        """
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            device = input_ids.device
            input_shape = input_ids.shape
        else:
            device = inputs_embeds.device
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(past_key_values_length, past_key_values_length + seq_length, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        attention_mask, encoder_attention_mask = self._create_attention_masks(
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            embedding_output=embedding_output,
            encoder_hidden_states=encoder_hidden_states,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_ids=position_ids,
            **kwargs,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
        )

    # Copied from transformers.models.bert.modeling_bert.BertModel._create_attention_masks
    def _create_attention_masks(
        self,
        attention_mask,
        encoder_attention_mask,
        embedding_output,
        encoder_hidden_states,
        cache_position,
        past_key_values,
    ):
        if self.config.is_decoder:
            attention_mask = create_causal_mask(
                config=self.config,
                input_embeds=embedding_output,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
            )
        else:
            attention_mask = create_bidirectional_mask(
                config=self.config,
                input_embeds=embedding_output,
                attention_mask=attention_mask,
            )

        if encoder_attention_mask is not None:
            encoder_attention_mask = create_bidirectional_mask(
                config=self.config,
                input_embeds=embedding_output,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        return attention_mask, encoder_attention_mask


@auto_docstring(
    custom_intro="""
    RoCBert Model with contrastive loss and masked_lm_loss during the pretraining.
    """
)
class RoCBertForPreTraining(RoCBertPreTrainedModel):
    _tied_weights_keys = {
        "cls.predictions.decoder.bias": "cls.predictions.bias",
        "cls.predictions.decoder.weight": "roc_bert.embeddings.word_embeddings.weight",
    }

    def __init__(self, config):
        super().__init__(config)

        self.roc_bert = RoCBertModel(config)
        self.cls = RoCBertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bert.modeling_bert.BertForPreTraining.get_output_embeddings
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # Copied from transformers.models.bert.modeling_bert.BertForPreTraining.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_shape_ids: Optional[torch.Tensor] = None,
        input_pronunciation_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attack_input_ids: Optional[torch.Tensor] = None,
        attack_input_shape_ids: Optional[torch.Tensor] = None,
        attack_input_pronunciation_ids: Optional[torch.Tensor] = None,
        attack_attention_mask: Optional[torch.Tensor] = None,
        attack_token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels_input_ids: Optional[torch.Tensor] = None,
        labels_input_shape_ids: Optional[torch.Tensor] = None,
        labels_input_pronunciation_ids: Optional[torch.Tensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        labels_token_type_ids: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        input_shape_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the shape vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input_shape_ids)
        input_pronunciation_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the pronunciation vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input_pronunciation_ids)
        attack_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            attack sample ids for computing the contrastive loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
            the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        attack_input_shape_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            attack sample shape ids for computing the contrastive loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
            the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        attack_input_pronunciation_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            attack sample pronunciation ids for computing the contrastive loss. Indices should be in `[-100, 0,
            ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        attack_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices for the attack sample. Mask values selected in
            `[0, 1]`: `1` for tokens that are NOT MASKED, `0` for MASKED tokens.
        attack_token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate different portions of the attack inputs. Indices are selected in `[0, 1]`:
            `0` corresponds to a sentence A token, `1` corresponds to a sentence B token.
        labels_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            target ids for computing the contrastive loss and masked_lm_loss . Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
            the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        labels_input_shape_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            target shape ids for computing the contrastive loss and masked_lm_loss . Indices should be in `[-100,
            0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        labels_input_pronunciation_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            target pronunciation ids for computing the contrastive loss and masked_lm_loss . Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ...,
            config.vocab_size]`
        labels_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices for the label sample. Mask values selected in
            `[0, 1]`: `1` for tokens that are NOT MASKED, `0` for MASKED tokens.
        labels_token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate different portions of the label inputs. Indices are selected in `[0, 1]`:
            `0` corresponds to a sentence A token, `1` corresponds to a sentence B token.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RoCBertForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("weiweishi/roc-bert-base-zh")
        >>> model = RoCBertForPreTraining.from_pretrained("weiweishi/roc-bert-base-zh")

        >>> inputs = tokenizer("你好，很高兴认识你", return_tensors="pt")
        >>> attack_inputs = {}
        >>> for key in list(inputs.keys()):
        ...     attack_inputs[f"attack_{key}"] = inputs[key]
        >>> label_inputs = {}
        >>> for key in list(inputs.keys()):
        ...     label_inputs[f"labels_{key}"] = inputs[key]

        >>> inputs.update(label_inputs)
        >>> inputs.update(attack_inputs)
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> logits.shape
        torch.Size([1, 11, 21128])
        ```
        """
        outputs = self.roc_bert(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.cls(sequence_output)

        loss = None
        if labels_input_ids is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels_input_ids.view(-1))

            if attack_input_ids is not None:
                batch_size, _ = labels_input_ids.shape
                device = labels_input_ids.device

                target_inputs = torch.clone(labels_input_ids)
                target_inputs[target_inputs == -100] = self.config.pad_token_id

                labels_output = self.roc_bert(
                    target_inputs,
                    input_shape_ids=labels_input_shape_ids,
                    input_pronunciation_ids=labels_input_pronunciation_ids,
                    attention_mask=labels_attention_mask,
                    token_type_ids=labels_token_type_ids,
                )
                attack_output = self.roc_bert(
                    attack_input_ids,
                    input_shape_ids=attack_input_shape_ids,
                    input_pronunciation_ids=attack_input_pronunciation_ids,
                    attention_mask=attack_attention_mask,
                    token_type_ids=attack_token_type_ids,
                )

                labels_pooled_output = labels_output[1]
                attack_pooled_output = attack_output[1]

                pooled_output_norm = torch.nn.functional.normalize(pooled_output, dim=-1)
                labels_pooled_output_norm = torch.nn.functional.normalize(labels_pooled_output, dim=-1)
                attack_pooled_output_norm = torch.nn.functional.normalize(attack_pooled_output, dim=-1)

                sim_matrix = torch.matmul(pooled_output_norm, attack_pooled_output_norm.T)  # batch_size * hidden_dim
                sim_matrix_target = torch.matmul(labels_pooled_output_norm, attack_pooled_output_norm.T)
                batch_labels = torch.tensor(list(range(batch_size)), device=device)
                contrastive_loss = (
                    loss_fct(100 * sim_matrix.view(batch_size, -1), batch_labels.view(-1))
                    + loss_fct(100 * sim_matrix_target.view(batch_size, -1), batch_labels.view(-1))
                ) / 2

                loss = contrastive_loss + masked_lm_loss
            else:
                loss = masked_lm_loss

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class RoCBertForMaskedLM(RoCBertPreTrainedModel):
    _tied_weights_keys = {
        "cls.predictions.decoder.bias": "cls.predictions.bias",
        "cls.predictions.decoder.weight": "roc_bert.embeddings.word_embeddings.weight",
    }

    # Copied from transformers.models.bert.modeling_bert.BertForMaskedLM.__init__ with Bert->RoCBert,bert->roc_bert
    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RoCBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roc_bert = RoCBertModel(config, add_pooling_layer=False)
        self.cls = RoCBertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bert.modeling_bert.BertForMaskedLM.get_output_embeddings
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # Copied from transformers.models.bert.modeling_bert.BertForMaskedLM.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_shape_ids: Optional[torch.Tensor] = None,
        input_pronunciation_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        input_shape_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the shape vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input_shape_ids)
        input_pronunciation_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the pronunciation vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input_pronunciation_ids)
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:
        ```python
        >>> from transformers import AutoTokenizer, RoCBertForMaskedLM
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("weiweishi/roc-bert-base-zh")
        >>> model = RoCBertForMaskedLM.from_pretrained("weiweishi/roc-bert-base-zh")

        >>> inputs = tokenizer("法国是首都[MASK].", return_tensors="pt")

        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits

        >>> # retrieve index of {mask}
        >>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        >>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        >>> tokenizer.decode(predicted_token_id)
        '.'
        ```
        """
        outputs = self.roc_bert(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
            **kwargs,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, input_shape_ids=None, input_pronunciation_ids=None, attention_mask=None, **model_kwargs
    ):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)
        if input_shape_ids is not None:
            input_shape_ids = torch.cat([input_shape_ids, dummy_token], dim=1)
        if input_pronunciation_ids is not None:
            input_pronunciation_ids = torch.cat([input_pronunciation_ids, dummy_token], dim=1)

        return {
            "input_ids": input_ids,
            "input_shape_ids": input_shape_ids,
            "input_pronunciation_ids": input_pronunciation_ids,
            "attention_mask": attention_mask,
        }

    @classmethod
    def can_generate(cls) -> bool:
        """
        Legacy correction: RoCBertForMaskedLM can't call `generate()` from `GenerationMixin`, even though it has a
        `prepare_inputs_for_generation` method.
        """
        return False


@auto_docstring(
    custom_intro="""
    RoCBert Model with a `language modeling` head on top for CLM fine-tuning.
    """
)
class RoCBertForCausalLM(RoCBertPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {
        "cls.predictions.decoder.bias": "cls.predictions.bias",
        "cls.predictions.decoder.weight": "roc_bert.embeddings.word_embeddings.weight",
    }

    # Copied from transformers.models.bert.modeling_bert.BertLMHeadModel.__init__ with BertLMHeadModel->RoCBertForCausalLM,Bert->RoCBert,bert->roc_bert
    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `RoCRoCBertForCausalLM` as a standalone, add `is_decoder=True.`")

        self.roc_bert = RoCBertModel(config, add_pooling_layer=False)
        self.cls = RoCBertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bert.modeling_bert.BertLMHeadModel.get_output_embeddings
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # Copied from transformers.models.bert.modeling_bert.BertLMHeadModel.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_shape_ids: Optional[torch.Tensor] = None,
        input_pronunciation_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        input_shape_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the shape vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input_shape_ids)
        input_pronunciation_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the pronunciation vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input_pronunciation_ids)
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RoCBertForCausalLM, RoCBertConfig
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("weiweishi/roc-bert-base-zh")
        >>> config = RoCBertConfig.from_pretrained("weiweishi/roc-bert-base-zh")
        >>> config.is_decoder = True
        >>> model = RoCBertForCausalLM.from_pretrained("weiweishi/roc-bert-base-zh", config=config)

        >>> inputs = tokenizer("你好，很高兴认识你", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```
        """
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.roc_bert(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.cls(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        input_shape_ids=None,
        input_pronunciation_ids=None,
        past_key_values=None,
        attention_mask=None,
        **model_kwargs,
    ):
        # Overwritten -- `input_pronunciation_ids`

        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **model_kwargs,
        )

        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            if input_shape_ids is not None:
                model_inputs["input_shape_ids"] = input_shape_ids[:, -1:]
            if input_pronunciation_ids is not None:
                model_inputs["input_pronunciation_ids"] = input_pronunciation_ids[:, -1:]

        return model_inputs


@auto_docstring(
    custom_intro="""
    RoCBert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks.
    """
)
class RoCBertForSequenceClassification(RoCBertPreTrainedModel):
    # Copied from transformers.models.bert.modeling_bert.BertForSequenceClassification.__init__ with Bert->RoCBert,bert->roc_bert
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roc_bert = RoCBertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_shape_ids: Optional[torch.Tensor] = None,
        input_pronunciation_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        input_shape_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the shape vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input_shape_ids)
        input_pronunciation_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the pronunciation vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input_pronunciation_ids)
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.roc_bert(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class RoCBertForMultipleChoice(RoCBertPreTrainedModel):
    # Copied from transformers.models.bert.modeling_bert.BertForMultipleChoice.__init__ with Bert->RoCBert,bert->roc_bert
    def __init__(self, config):
        super().__init__(config)

        self.roc_bert = RoCBertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_shape_ids: Optional[torch.Tensor] = None,
        input_pronunciation_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        input_shape_ids (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`):
            Indices of input sequence tokens in the shape vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input_shape_ids)
        input_pronunciation_ids (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`):
            Indices of input sequence tokens in the pronunciation vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input_pronunciation_ids)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        input_shape_ids = input_shape_ids.view(-1, input_shape_ids.size(-1)) if input_shape_ids is not None else None
        input_pronunciation_ids = (
            input_pronunciation_ids.view(-1, input_pronunciation_ids.size(-1))
            if input_pronunciation_ids is not None
            else None
        )
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.roc_bert(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class RoCBertForTokenClassification(RoCBertPreTrainedModel):
    # Copied from transformers.models.bert.modeling_bert.BertForTokenClassification.__init__ with Bert->RoCBert,bert->roc_bert
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roc_bert = RoCBertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_shape_ids: Optional[torch.Tensor] = None,
        input_pronunciation_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, TokenClassifierOutput]:
        r"""
        input_shape_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the shape vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input_shape_ids)
        input_pronunciation_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the pronunciation vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input_pronunciation_ids)
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        outputs = self.roc_bert(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class RoCBertForQuestionAnswering(RoCBertPreTrainedModel):
    # Copied from transformers.models.bert.modeling_bert.BertForQuestionAnswering.__init__ with Bert->RoCBert,bert->roc_bert
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roc_bert = RoCBertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_shape_ids: Optional[torch.Tensor] = None,
        input_pronunciation_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        input_shape_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the shape vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input_shape_ids)
        input_pronunciation_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the pronunciation vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input_pronunciation_ids)
        """
        outputs = self.roc_bert(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "RoCBertForCausalLM",
    "RoCBertForMaskedLM",
    "RoCBertForMultipleChoice",
    "RoCBertForPreTraining",
    "RoCBertForQuestionAnswering",
    "RoCBertForSequenceClassification",
    "RoCBertForTokenClassification",
    "RoCBertLayer",
    "RoCBertModel",
    "RoCBertPreTrainedModel",
]
