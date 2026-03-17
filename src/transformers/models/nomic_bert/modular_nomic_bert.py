# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from collections.abc import Callable

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...integrations import use_kernelized_func
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
)
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..bert.modeling_bert import (
    BertEmbeddings,
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForPreTraining,
    BertForPreTrainingOutput,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertModel,
    BertOnlyMLMHead,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
)
from ..gemma.modeling_gemma import GemmaMLP
from ..gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from ..llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..mistral.modeling_mistral import MistralAttention


@auto_docstring(checkpoint="nomic-ai/nomic-embed-text-v1.5")
@strict(accept_kwargs=True)
class NomicBertConfig(PreTrainedConfig):
    model_type = "nomic_bert"

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int | None = None
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    classifier_dropout: float | None = None
    type_vocab_size: int = 2
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int = 0
    tie_word_embeddings = True
    rope_parameters: RopeParameters | dict | None = None
    max_position_embeddings: int = 2048
    head_dim: int | None = None

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.is_decoder = False
        self.add_cross_attention = False
        self.use_cache = False


class NomicBertEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        del self.position_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        batch_size, seq_length = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                # NOTE: We assume either pos ids to have bsz == 1 (broadcastable) or bsz == effective bsz (input_shape[0])
                buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
                buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
                token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
            else:
                input_shape = inputs_embeds.size()[:-1]
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class NomicBertRotaryEmbedding(LlamaRotaryEmbedding):
    pass


@use_kernelized_func(apply_rotary_pos_emb)
class NomicBertAttention(MistralAttention):
    """
    Self-Attention mechanism is essentially Llama attention without caching.
    """

    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)

        del self.layer_idx
        self.attention_dropout = config.attention_probs_dropout_prob
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # get all proj
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply Rotary Position Embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class NomicBertMLP(GemmaMLP):
    pass


class NomicBertLayer(GPTNeoXLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        del self.use_parallel_residual
        del self.input_layernorm
        self.post_attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.post_mlp_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_mlp_dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        # Attention
        residual = hidden_states
        hidden_states, _ = self.attention(
            hidden_states,
            attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = self.post_attention_dropout(hidden_states)
        hidden_states = hidden_states + residual
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_dropout(hidden_states)
        hidden_states = hidden_states + residual
        hidden_states = self.post_mlp_layernorm(hidden_states)

        return hidden_states


class NomicBertPooler(BertPooler):
    pass


class NomicBertPreTrainedModel(BertPreTrainedModel):
    config_class = NomicBertConfig
    base_model_prefix = "nomic_bert"
    _keys_to_ignore_on_load_unexpected = ["inv_freq", "original_inv_freq"]
    _can_record_outputs = {
        "hidden_states": NomicBertLayer,
        "attentions": NomicBertAttention,
    }


class NomicBertForPreTrainingOutput(BertForPreTrainingOutput):
    pass


class NomicBertPredictionHeadTransform(BertPredictionHeadTransform):
    def __init__(self, config):
        super().__init__(config)
        # Use layer_norm rather than LayerNorm to avoid bert legacy mappings weights and bias to gamma and beta
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        del self.LayerNorm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        return hidden_states


@auto_docstring
class NomicBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        """
        Args:
            add_pooling_layer (`bool`, *optional*, defaults to `True`):
                Whether to add a pooling layer.
        """
        super().__init__(config, add_pooling_layer=add_pooling_layer)

        del self.encoder

        self.layers = nn.ModuleList(
            [NomicBertLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.rotary_emb = NomicBertRotaryEmbedding(config)

        self.embeddings_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embeddings_dropout = nn.Dropout(config.hidden_dropout_prob)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPooling:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            seq_length = input_ids.shape[1]
            device = input_ids.device
        else:
            seq_length = inputs_embeds.shape[:-1][1]
            device = inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)[None, :]

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        embedding_output = self.embeddings_layernorm(embedding_output)
        embedding_output = self.embeddings_dropout(embedding_output)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=embedding_output,
            attention_mask=attention_mask,
        )

        hidden_states = embedding_output
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                **kwargs,
            )

        sequence_output = hidden_states
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
        )


class NomicBertForPreTraining(BertForPreTraining):
    pass


class NomicBertForNextSentencePrediction(BertForNextSentencePrediction):
    pass


class NomicBertForSequenceClassification(BertForSequenceClassification):
    pass


class NomicBertOnlyMLMHead(BertOnlyMLMHead):
    pass


class NomicBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        NomicBertPreTrainedModel.__init__(self, config=config)

        self.nomic_bert = NomicBertModel(config, add_pooling_layer=False)
        self.cls = NomicBertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()


class NomicBertForTokenClassification(BertForTokenClassification):
    pass


__all__ = [
    "NomicBertConfig",
    "NomicBertForMaskedLM",
    "NomicBertForNextSentencePrediction",
    "NomicBertForPreTraining",
    "NomicBertForSequenceClassification",
    "NomicBertForTokenClassification",
    "NomicBertLayer",
    "NomicBertModel",
    "NomicBertPreTrainedModel",
]
