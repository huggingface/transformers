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

from ...configuration_utils import PreTrainedConfig
from ...integrations import use_kernelized_func
from ...masking_utils import create_bidirectional_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutput,
)
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import check_model_inputs
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
    BertLMHeadModel,
    BertModel,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
    BertSelfAttention,
    BertSelfOutput,
)
from ..llama.modeling_llama import (
    LlamaMLP,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


class NomicBertConfig(PreTrainedConfig):
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
        **kwargs: Unpack[TransformersKwargs],
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout = classifier_dropout
        self.type_vocab_size = type_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rope_parameters = rope_parameters

        super().__init__(**kwargs)


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
    Custom Self-Attention mechanism for NomicBERT.
    Key Difference: Replaces standard BERT absolute position embeddings with
    Rotary Positional Embeddings (RoPE) applied directly to Q and K.
    """

    def __init__(self, config, layer_idx=None):
        self.num_kv_heads = (
            config.num_key_value_heads if config.num_key_value_heads is not None else config.num_attention_heads
        )
        super().__init__(config, layer_idx=layer_idx)
        self.num_key_value_groups = self.num_attention_heads // self.num_kv_heads

        self.is_decoder = False
        self.is_causal = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        # get all proj
        query_layer = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply Rotary Position Embeddings
        cos, sin = position_embeddings
        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)

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

        attn_output = attn_output.transpose(1, 2).contiguous().view(*input_shape, -1)
        return attn_output, attn_weights


class NomicBertSelfOutput(BertSelfOutput):
    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)


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
        self_outputs, _ = self.self(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # Process context layer (always index 0)
        output = self.output(self_outputs, hidden_states)

        return output


class NomicBertIntermediate(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)


class NomicBertLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.attention = NomicBertAttention(config, layer_idx=layer_idx)
        self.intermediate = NomicBertIntermediate(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        position_embeddings: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        # Self attention block
        residual = hidden_states
        hidden_states = self.attention(
            hidden_states,
            attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Feed forward block
        residual = hidden_states
        hidden_states = self.intermediate(hidden_states)

        # Residual connection
        hidden_states = residual + hidden_states

        return hidden_states


class NomicBertEncoder(BertEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        position_embeddings: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


class NomicBertPooler(BertPooler):
    pass


class NomicBertPreTrainedModel(BertPreTrainedModel):
    config_class = NomicBertConfig
    base_model_prefix = "nomic_bert"
    supports_gradient_checkpointing = True
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
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        del self.LayerNorm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


@auto_docstring
class NomicBertModel(BertModel):
    """
    Example:
    ```python
    python transformers/src/transformers/models/nomic_bert/convert_nomic_bert_to_hf.py \
        --original_model_id nomic-ai/nomic-embed-text-v1.5 \
        --output_hub_path org/nomic_bert
    ```
    """

    def __init__(self, config, add_pooling_layer=True):
        """
        Args:
            add_pooling_layer (`bool`, *optional*, defaults to `True`):
                Whether to add a pooling layer.
        """
        super().__init__(config, add_pooling_layer=add_pooling_layer)

        self.encoder = NomicBertEncoder(config)

        self.rotary_emb = NomicBertRotaryEmbedding(config)

        self.post_init()

    @check_model_inputs
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
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        attention_mask = create_bidirectional_mask(
            config=self.config,
            input_embeds=embedding_output,
            attention_mask=attention_mask,
        )

        position_embeddings = self.rotary_emb(embedding_output, position_ids)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        sequence_output = encoder_outputs.last_hidden_state

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


class NomicBertForPreTraining(BertForPreTraining):
    pass


class NomicBertLMHeadModel(BertLMHeadModel):
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | CausalLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`
        """

        outputs: BaseModelOutputWithPooling = self.nomic_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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
    "NomicBertLMHeadModel",
    "NomicBertModel",
    "NomicBertPreTrainedModel",
]
