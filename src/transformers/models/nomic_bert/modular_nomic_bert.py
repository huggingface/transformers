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

from ...integrations import use_kernelized_func
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
)
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..bert.configuration_bert import BertConfig
from ..bert.modeling_bert import (
    BertEmbeddings,
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForNextSentencePrediction,
    BertForPreTraining,
    BertForPreTrainingOutput,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertModel,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
    BertSelfAttention,
)
from ..llama.modeling_llama import (
    LlamaMLP,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


class NomicBertConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a [`NomicBertModel`]. It is used to instantiate an NomicBERT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the NomicBERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            Number of key-value attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        type_vocab_size (`int`, *optional*, defaults to 2):
            The size of the token type (segment) vocabulary. Used to distinguish different portions of the input,
            such as sentence A and sentence B in pairwise classification tasks.
        bos_token_id (`int`, *optional*):
            The token ID used for the beginning-of-sequence token.
        eos_token_id (`int`, *optional*):
            The token ID used for the end-of-sequence token.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie the input and output word embeddings. If set to `True`, the same embedding matrix
            is used for both input embeddings and output logits.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        pad_token_id (`int`, *optional*, defaults to 0):
            The token ID used for padding.
        head_dim (`int`, *optional*):
            The dimension of the attention heads.

    ```python
    >>> from transformers import NomicBertModel, NomicBertConfig

    >>> # Initializing a NomicBERT 2048 style configuration
    >>> configuration = NomicBertConfig()

    >>> # Initializing a model from the NomicBERT 2048 style configuration
    >>> model = NomicBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "nomic_bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=None,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        classifier_dropout=None,
        type_vocab_size=2,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=True,
        rope_parameters: RopeParameters | None = None,
        max_position_embeddings=2048,
        pad_token_id=0,
        head_dim=None,
        **kwargs,
    ):
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rope_parameters = rope_parameters

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            classifier_dropout=classifier_dropout,
            type_vocab_size=type_vocab_size,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        # is_decoder not deleted as will cause errors if NomicBertForMaskedLM is called
        self.is_decoder = False
        del self.use_cache
        del self.add_cross_attention


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
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
                buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
                token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class NomicBertRotaryEmbedding(LlamaRotaryEmbedding):
    pass


@use_kernelized_func(apply_rotary_pos_emb)
class NomicBertSelfAttention(BertSelfAttention):
    """
    Self-Attention mechanism for NomicBERT is essentially Llama attention without caching logic.
    Key Difference: Replaces standard BERT absolute position embeddings with
    Rotary Positional Embeddings (RoPE) applied directly to Q and K.
    """

    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)

        self.num_kv_heads = (
            config.num_key_value_heads if config.num_key_value_heads is not None else config.num_attention_heads
        )
        self.num_key_value_groups = self.num_attention_heads // self.num_kv_heads
        self.Wqkv = nn.Linear(
            config.hidden_size,
            (self.num_attention_heads + 2 * self.num_kv_heads) * self.attention_head_size,
            bias=False,
        )

        self.o_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

        del self.is_decoder
        del self.is_causal
        del self.query
        del self.key
        del self.value

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        input_shape = hidden_states.shape[:-1]

        # get all proj
        qkv = self.Wqkv(hidden_states)

        q_size = self.num_attention_heads * self.attention_head_size
        kv_size = self.num_kv_heads * self.attention_head_size

        query_states, key_states, value_states = torch.split(
            qkv,
            [q_size, kv_size, kv_size],
            dim=-1,
        )

        query_states = query_states.view(*input_shape, self.num_attention_heads, self.attention_head_size).transpose(
            1, 2
        )

        key_states = key_states.view(*input_shape, self.num_kv_heads, self.attention_head_size).transpose(1, 2)

        value_states = value_states.view(*input_shape, self.num_kv_heads, self.attention_head_size).transpose(1, 2)

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
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class NomicBertIntermediate(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)


class NomicBertLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.attention = NomicBertSelfAttention(config, layer_idx=layer_idx)
        self.intermediate = NomicBertIntermediate(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        position_embeddings: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        residual = hidden_states
        hidden_states, _ = self.attention(
            hidden_states,
            attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = self.norm1(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.intermediate(hidden_states)

        hidden_states = self.norm2(residual + hidden_states)

        return hidden_states


class NomicBertPooler(BertPooler):
    pass


class NomicBertPreTrainedModel(BertPreTrainedModel):
    config_class = NomicBertConfig
    base_model_prefix = "nomic_bert"
    _keys_to_ignore_on_load_unexpected = ["inv_freq", "original_inv_freq"]
    _can_record_outputs = {
        "hidden_states": NomicBertLayer,
        "attentions": NomicBertSelfAttention,
    }


class NomicBertForPreTrainingOutput(BertForPreTrainingOutput):
    pass


class NomicBertPredictionHeadTransform(BertPredictionHeadTransform):
    def __init__(self, config):
        super().__init__(config)
        # Use layer_norm rather than LayerNorm to avoid bert legacy mappings weights and bias to gamma and beta
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        del self.LayerNorm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
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

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPooling:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        else:
            batch_size, seq_length = inputs_embeds.shape[:-1]
            device = inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)[None, :]

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        attention_mask = self.get_extended_attention_mask(attention_mask, (batch_size, seq_length))

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


class NomicBertForMaskedLM(BertForMaskedLM):
    pass


class NomicBertForNextSentencePrediction(BertForNextSentencePrediction):
    pass


class NomicBertForSequenceClassification(BertForSequenceClassification):
    pass


class NomicBertForMultipleChoice(BertForMultipleChoice):
    pass


class NomicBertForTokenClassification(BertForTokenClassification):
    pass


class NomicBertForQuestionAnswering(BertForQuestionAnswering):
    pass


__all__ = [
    "NomicBertConfig",
    "NomicBertForMaskedLM",
    "NomicBertForMultipleChoice",
    "NomicBertForNextSentencePrediction",
    "NomicBertForPreTraining",
    "NomicBertForQuestionAnswering",
    "NomicBertForSequenceClassification",
    "NomicBertForTokenClassification",
    "NomicBertLayer",
    "NomicBertModel",
    "NomicBertPreTrainedModel",
]
