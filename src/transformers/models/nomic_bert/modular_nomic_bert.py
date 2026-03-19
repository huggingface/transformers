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


import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict
from torch.nn import CrossEntropyLoss

from ...configuration_utils import PreTrainedConfig
from ...integrations import use_kernelized_func
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
)
from ...modeling_rope_utils import RopeParameters
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..bert.configuration_bert import BertConfig
from ..bert.modeling_bert import (
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForPreTraining,
    BertForPreTrainingOutput,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertOnlyMLMHead,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
)
from ..gemma.modeling_gemma import GemmaMLP
from ..jina_embeddings_v3.modeling_jina_embeddings_v3 import (
    JinaEmbeddingsV3Attention,
    JinaEmbeddingsV3Embeddings,
    JinaEmbeddingsV3Layer,
    JinaEmbeddingsV3Model,
)
from ..llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)


@auto_docstring(checkpoint="nomic-ai/nomic-embed-text-v1.5")
@strict(accept_kwargs=True)
class NomicBertConfig(BertConfig):
    r"""
    Examples:

    ```python
    >>> from transformers import NomicBertConfig, NomicBertModel

    >>> # Initializing a Nomic BERT nomic-ai/nomic-embed-text-v1.5 style configuration
    >>> configuration = NomicBertConfig()

    >>> # Initializing a model (with random weights) from the nomic-ai/nomic-embed-text-v1.5 style configuration
    >>> model = NomicBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "nomic_bert"

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
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
    is_decoder = AttributeError()
    add_cross_attention = AttributeError()
    use_cache = AttributeError()

    def __post_init__(self, **kwargs):
        PreTrainedConfig.__post_init__(self, **kwargs)
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


class NomicBertEmbeddings(JinaEmbeddingsV3Embeddings):
    pass


class NomicBertRotaryEmbedding(LlamaRotaryEmbedding):
    pass


@use_kernelized_func(apply_rotary_pos_emb)
class NomicBertAttention(JinaEmbeddingsV3Attention):
    def __init__(self, config):
        super().__init__(config)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)


class NomicBertMLP(GemmaMLP):
    pass


class NomicBertLayer(JinaEmbeddingsV3Layer):
    pass


class NomicBertPooler(BertPooler):
    pass


class NomicBertPreTrainedModel(BertPreTrainedModel):
    config_class = NomicBertConfig
    base_model_prefix = "nomic_bert"

    # Are kept as non-persistent buffers to avoid being saved in the state dict
    # and causing mismatch when loading from a checkpoint that doesn't have them
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
class NomicBertModel(JinaEmbeddingsV3Model):
    def __init__(self, config, add_pooling_layer=True):
        """
        Args:
            add_pooling_layer (`bool`, *optional*, defaults to `True`):
                Whether to add a pooling layer.
        """
        super().__init__(config, add_pooling_layer=add_pooling_layer)
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
            seq_length = inputs_embeds.shape[1]
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

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | MaskedLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        outputs = self.nomic_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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
