# Copyright 2025 Nicolas Boizard, Duarte M. Alves, Hippolyte Gisserot-Boukhlef and the EuroBert team. All rights reserved.
#
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
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...configuration_utils import strict
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutput, MaskedLMOutput, SequenceClassifierOutput, TokenClassifierOutput
from ...modeling_rope_utils import RopeParameters
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.generic import TransformersKwargs, can_return_tuple
from ..llama import LlamaConfig
from ..llama.modeling_llama import LlamaAttention, LlamaModel, LlamaPreTrainedModel, LlamaRMSNorm


@auto_docstring(checkpoint="EuroBERT/EuroBERT-210m")
@strict
class EuroBertConfig(LlamaConfig):
    r"""
    mask_token_id (`int`, *optional*, defaults to 128002):
        Mask token id.
    classifier_pooling (`str`, *optional*, defaults to `"late"`):
        The pooling strategy to use for the classifier. Can be one of ['bos', 'mean', 'late'].

    ```python
    >>> from transformers import EuroBertModel, EuroBertConfig

    >>> # Initializing a EuroBert eurobert-base style configuration
    >>> configuration = EuroBertConfig()

    >>> # Initializing a model from the eurobert-base style configuration
    >>> model = EuroBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "eurobert"

    vocab_size: int = 128256
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-05
    bos_token_id: int | None = 128000
    eos_token_id: int | list[int] | None = 128001
    pad_token_id: int | None = 128001
    mask_token_id: int = 128002
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float = 0.0
    mlp_bias: bool = False
    head_dim: int | None = None
    classifier_pooling: str = "late"

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)


class EuroBertRMSNorm(LlamaRMSNorm):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__(hidden_size, eps)


class EuroBertAttention(LlamaAttention):
    def __init__(self, config: EuroBertConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = False


class EuroBertPreTrainedModel(LlamaPreTrainedModel):
    pass


class EuroBertModel(LlamaModel):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        bidirectional_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=bidirectional_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


@auto_docstring
class EuroBertForMaskedLM(EuroBertPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: EuroBertConfig):
        super().__init__(config)
        self.model = EuroBertModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, config.mlp_bias)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | MaskedLMOutput:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, EuroBertForMaskedLM

        >>> model = EuroBertForMaskedLM.from_pretrained("EuroBERT/EuroBERT-210m")
        >>> tokenizer = AutoTokenizer.from_pretrained("EuroBERT/EuroBERT-210m")

        >>> text = "The capital of France is <|mask|>."
        >>> inputs = tokenizer(text, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # To get predictions for the mask:
        >>> masked_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
        >>> predicted_token_id = outputs.logits[0, masked_index].argmax(axis=-1)
        >>> predicted_token = tokenizer.decode(predicted_token_id)
        >>> print("Predicted token:", predicted_token)
        Predicted token:  Paris
        ```"""
        outputs: BaseModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class EuroBertForSequenceClassification(EuroBertPreTrainedModel):
    def __init__(self, config: EuroBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier_pooling = config.classifier_pooling

        self.model = EuroBertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput:
        encoder_output = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        last_hidden_state = encoder_output[0]

        if self.classifier_pooling in ["bos", "mean"]:
            if self.classifier_pooling == "bos":
                pooled_output = last_hidden_state[:, 0]

            elif self.classifier_pooling == "mean":
                if attention_mask is None:
                    pooled_output = last_hidden_state.mean(dim=1)
                else:
                    attention_mask = attention_mask.to(last_hidden_state.device)
                    pooled_output = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1)
                    pooled_output /= attention_mask.sum(dim=1, keepdim=True)

            pooled_output = self.dense(pooled_output)
            pooled_output = self.activation(pooled_output)
            logits = self.classifier(pooled_output)

        elif self.classifier_pooling == "late":
            x = self.dense(last_hidden_state)
            x = self.activation(x)
            logits = self.classifier(x)
            if attention_mask is None:
                logits = logits.mean(dim=1)
            else:
                attention_mask = attention_mask.to(logits.device)
                logits = (logits * attention_mask.unsqueeze(-1)).sum(dim=1)
                logits /= attention_mask.sum(dim=1, keepdim=True)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
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
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
        )


@auto_docstring
class EuroBertForTokenClassification(EuroBertPreTrainedModel):
    def __init__(self, config: EuroBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = EuroBertModel(config)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | TokenClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        sequence_output = outputs[0]
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


__all__ = [
    "EuroBertConfig",
    "EuroBertPreTrainedModel",
    "EuroBertModel",
    "EuroBertForMaskedLM",
    "EuroBertForSequenceClassification",
    "EuroBertForTokenClassification",
]
