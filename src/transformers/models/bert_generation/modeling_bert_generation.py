# Copyright 2020 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""PyTorch BERT model specific for generation."""

from collections.abc import Callable

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...generation import GenerationMixin
from ...masking_utils import create_bidirectional_mask, create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    logging,
)
from ...utils.generic import can_return_tuple, check_model_inputs
from .configuration_bert_generation import BertGenerationConfig


logger = logging.get_logger(__name__)


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->BertGeneration
class BertGenerationSelfOutput(nn.Module):
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


# Copied from transformers.models.bert.modeling_bert.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
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


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->BertGeneration
class BertGenerationSelfAttention(nn.Module):
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
        attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
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


# Copied from transformers.models.bert.modeling_bert.BertCrossAttention with Bert->BertGeneration
class BertGenerationCrossAttention(nn.Module):
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
        encoder_hidden_states: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        past_key_values: EncoderDecoderCache | None = None,
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


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->BertGeneration,BERT->BERT_GENERATION
class BertGenerationAttention(nn.Module):
    def __init__(self, config, is_causal=False, layer_idx=None, is_cross_attention=False):
        super().__init__()
        self.is_cross_attention = is_cross_attention
        attention_class = BertGenerationCrossAttention if is_cross_attention else BertGenerationSelfAttention
        self.self = attention_class(config, is_causal=is_causal, layer_idx=layer_idx)
        self.output = BertGenerationSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
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


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->BertGeneration
class BertGenerationIntermediate(nn.Module):
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


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->BertGeneration
class BertGenerationOutput(nn.Module):
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


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->BertGeneration
class BertGenerationLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertGenerationAttention(config, is_causal=config.is_decoder, layer_idx=layer_idx)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertGenerationAttention(
                config,
                is_causal=False,
                layer_idx=layer_idx,
                is_cross_attention=True,
            )
        self.intermediate = BertGenerationIntermediate(config)
        self.output = BertGenerationOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
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


# Copied from transformers.models.bert.modeling_bert.BertEncoder
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Ignore copy
        self.layer = nn.ModuleList([BertGenerationLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPastAndCrossAttentions:
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


class BertGenerationEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


@auto_docstring
class BertGenerationPreTrainedModel(PreTrainedModel):
    config_class = BertGenerationConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": BertGenerationLayer,
        "attentions": BertGenerationSelfAttention,
        "cross_attentions": BertGenerationCrossAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, BertGenerationOnlyLMHead):
            init.zeros_(module.bias)
        elif isinstance(module, BertGenerationEmbeddings):
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))


@auto_docstring(
    custom_intro="""
    The bare BertGeneration model transformer outputting raw hidden-states without any specific head on top.
    """
)
class BertGenerationEncoder(BertGenerationPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://huggingface.co/papers/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    This model should be used when leveraging Bert or Roberta checkpoints for the [`EncoderDecoderModel`] class as
    described in [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://huggingface.co/papers/1907.12461)
    by Sascha Rothe, Shashi Narayan, and Aliaksei Severyn.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        self.embeddings = BertGenerationEmbeddings(config)
        self.encoder = BertEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPastAndCrossAttentions:
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = (
                EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))
                if encoder_hidden_states is not None or self.config.is_encoder_decoder
                else DynamicCache(config=self.config)
            )

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
            position_ids=position_ids,
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

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=sequence_output,
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


class BertGenerationOnlyLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        logits = self.decoder(hidden_states)
        return logits


@auto_docstring(
    custom_intro="""
    BertGeneration Model with a `language modeling` head on top for CLM fine-tuning.
    """
)
class BertGenerationDecoder(BertGenerationPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {
        "lm_head.decoder.weight": "bert.embeddings.word_embeddings.weight",
        "lm_head.decoder.bias": "lm_head.bias",
    }

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `BertGenerationDecoder` as a standalone, add `is_decoder=True.`")

        self.bert = BertGenerationEncoder(config)
        self.lm_head = BertGenerationOnlyLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings
        self.lm_head.bias = new_embeddings.bias

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.FloatTensor]] | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithCrossAttentions:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BertGenerationDecoder, BertGenerationConfig
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
        >>> config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
        >>> config.is_decoder = True
        >>> model = BertGenerationDecoder.from_pretrained(
        ...     "google/bert_for_seq_generation_L-24_bbc_encoder", config=config
        ... )

        >>> inputs = tokenizer("Hello, my dog is cute", return_token_type_ids=False, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```"""
        if labels is not None:
            use_cache = False

        outputs: BaseModelOutputWithPastAndCrossAttentions = self.bert(
            input_ids,
            attention_mask=attention_mask,
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
        logits = self.lm_head(hidden_states[:, slice_indices, :])

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


__all__ = [
    "BertGenerationDecoder",
    "BertGenerationEncoder",
    "BertGenerationPreTrainedModel",
]
