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

from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig
from ...integrations import use_kernelized_func
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
)
from ...modeling_rope_utils import RotaryEmbeddingConfigMixin
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import TransformersKwargs, auto_docstring
from ..bert.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForNextSentencePrediction,
    BertForPreTraining,
    BertForPreTrainingOutput,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertLayer,
    BertLMHeadModel,
    BertLMPredictionHead,
    BertModel,
    BertOnlyMLMHead,
    BertOnlyNSPHead,
    BertOutput,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
    BertPreTrainingHeads,
    BertSelfAttention,
    BertSelfOutput,
)
from ..llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb, eager_attention_forward


class NomicBertConfig(PreTrainedConfig, RotaryEmbeddingConfigMixin):
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
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the attention layer.

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

    model_type = "model"

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
        rope_parameters=None,
        max_position_embeddings=2048,
        pad_token_id=0,
        head_dim=None,
        attention_bias=False,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if rope_parameters is None:
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": 10000.0,
            }

        super().__init__(
            rope_parameters=rope_parameters,
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
            use_cache=False,
            classifier_dropout=classifier_dropout,
            is_decoder=False,
            type_vocab_size=type_vocab_size,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            add_cross_attention=False,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.attention_bias = attention_bias
        self.num_key_value_heads = num_key_value_heads

        self.rope_parameters = rope_parameters


class NomicBertEmbeddings(BertEmbeddings):
    pass


class NomicBertRotaryEmbedding(LlamaRotaryEmbedding):
    pass


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@use_kernelized_func(apply_rotary_pos_emb)
class NomicBertSelfAttention(BertSelfAttention):
    """
    Custom Self-Attention mechanism for NomicBERT.
    Key Difference: Replaces standard BERT absolute position embeddings with
    Rotary Positional Embeddings (RoPE) applied directly to Q and K.
    """

    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.num_kv_heads = (
            config.num_key_value_heads if config.num_key_value_heads is not None else config.num_attention_heads
        )
        self.num_key_value_groups = self.num_attention_heads // self.num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.attention_head_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.attention_head_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.attention_head_size, bias=config.attention_bias)

        self.is_causal = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        batch_size, seq_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        # get all proj
        query_layer = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_layer = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_layer = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply Rotary Position Embeddings
        cos, sin = position_embeddings
        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)

        query_layer = query_layer * self.scaling

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(*input_shape, -1)
        return attn_output, attn_weights


class NomicBertSelfOutput(BertSelfOutput):
    pass


class NomicBertAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.self = NomicBertSelfAttention(config, layer_idx=layer_idx)

        self.output = NomicBertSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # Process context layer (always index 0)
        attention_output = self.output(self_outputs[0], hidden_states)

        return (attention_output,)


class NomicBertIntermediate(nn.Module):
    """
    NomicBERT Intermediate layer.
    Replaces standard BERT GELU with SiLU (Swish).

    Standard BERT: Activation(Linear(x))
    NomicBERT:     SiLU(Gate(x)) * Value(x)
    """

    def __init__(self, config):
        super().__init__()
        # Add the Gate Layer
        # SiLU needs a second parallel layer for the gate.
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)


class NomicBertOutput(BertOutput):
    pass


class NomicBertLayer(BertLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.attention = NomicBertAttention(config, layer_idx=layer_idx)
        self.intermediate = NomicBertIntermediate(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        position_embeddings: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attention_output = self_attention_outputs[0]

        hidden_states = self.attention.output(attention_output, hidden_states)

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, hidden_states
        )

        return (layer_output,)


class NomicBertEncoder(BertEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.gradient_checkpointing = False

        # Re-initialize self.layer with the correct index passed to each layer
        self.layer = nn.ModuleList([NomicBertLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        position_embeddings: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


class NomicBertPooler(BertPooler):
    pass


class NomicBertPredictionHeadTransform(BertPredictionHeadTransform):
    pass


class NomicBertLMPredictionHead(BertLMPredictionHead):
    pass


class NomicBertOnlyMLMHead(BertOnlyMLMHead):
    pass


class NomicBertOnlyNSPHead(BertOnlyNSPHead):
    pass


class NomicBertPreTrainingHeads(BertPreTrainingHeads):
    pass


class NomicBertPreTrainedModel(BertPreTrainedModel):
    pass


class NomicBertForPreTrainingOutput(BertForPreTrainingOutput):
    pass


@auto_docstring
class NomicBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)

        self.encoder = NomicBertEncoder(config, layer_class=NomicBertLayer)

        self.rotary_emb = NomicBertRotaryEmbedding(config)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        else:
            batch_size, seq_length = inputs_embeds.shape[:-1]
            device = inputs_embeds.device

        if attention_mask is None:
            if input_ids is not None:
                # Standard BERT padding mask: 1 for valid, 0 for pad
                attention_mask = (input_ids != self.config.pad_token_id).long()
            else:
                # Cannot infer padding from embeddings alone, defaulting to all ones
                attention_mask = torch.ones((batch_size, seq_length), device=device, dtype=torch.long)

        binary_mask = attention_mask

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        position_embeddings = self.rotary_emb(embedding_output, position_ids)

        extended_attention_mask = self.get_extended_attention_mask(
            binary_mask,
            (batch_size, seq_length)
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        sequence_output = encoder_outputs.last_hidden_state

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class NomicBertForPreTraining(BertForPreTraining):
    config_class = NomicBertConfig
    base_model_prefix = "model"


class NomicBertLMHeadModel(BertLMHeadModel):
    pass


class NomicBertForMaskedLM(BertForMaskedLM):
    config_class = NomicBertConfig
    base_model_prefix = "model"


class NomicBertForNextSentencePrediction(BertForNextSentencePrediction):
    config_class = NomicBertConfig
    base_model_prefix = "model"


class NomicBertForSequenceClassification(BertForSequenceClassification):
    config_class = NomicBertConfig
    base_model_prefix = "model"


class NomicBertForMultipleChoice(BertForMultipleChoice):
    config_class = NomicBertConfig
    base_model_prefix = "model"


class NomicBertForTokenClassification(BertForTokenClassification):
    config_class = NomicBertConfig
    base_model_prefix = "model"


class NomicBertForQuestionAnswering(BertForQuestionAnswering):
    config_class = NomicBertConfig
    base_model_prefix = "model"


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
    "NomicBertLMHeadModel",
    "NomicBertModel",
    "NomicBertPreTrainedModel",
]
