# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""PyTorch CharacterBERT model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...masking_utils import create_bidirectional_mask, create_causal_mask
from ...modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..bert.modeling_bert import (
    BertCrossAttention,
    BertEncoder,
    BertLayer,
    BertOnlyMLMHead,
    BertPooler,
    BertPreTrainedModel,
    BertSelfAttention,
)
from .configuration_character_bert import CharacterBertConfig


class CharacterBertSelfAttention(BertSelfAttention):
    pass


class CharacterBertCrossAttention(BertCrossAttention):
    pass


class CharacterBertLayer(BertLayer):
    pass


class CharacterBertEncoder(BertEncoder):
    pass


class CharacterBertPooler(BertPooler):
    pass


class CharacterBertOnlyMLMHead(BertOnlyMLMHead):
    pass


class CharacterBertHighway(nn.Module):
    def __init__(self, input_dim: int, num_layers: int = 1):
        super().__init__()
        self._layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)])
        for layer in self._layers:
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current_input = inputs
        for layer in self._layers:
            projected = layer(current_input)
            nonlinear_part, gate = projected.chunk(2, dim=-1)
            nonlinear_part = nn.functional.relu(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * current_input + (1 - gate) * nonlinear_part
        return current_input


class CharacterBertCharacterCNN(nn.Module):
    def __init__(self, config: CharacterBertConfig):
        super().__init__()
        self.max_characters_per_token = config.max_characters_per_token
        self.hidden_size = config.hidden_size
        self._filters = list(config.character_cnn_filters)

        self._char_embedding_weights = nn.Parameter(
            torch.zeros(config.character_vocab_size + 1, config.character_embedding_dim),
            requires_grad=True,
        )

        convolutions = []
        for index, (width, num_filters) in enumerate(self._filters):
            conv = nn.Conv1d(config.character_embedding_dim, num_filters, kernel_size=width, bias=True)
            self.add_module(f"char_conv_{index}", conv)
            convolutions.append(conv)
        self._convolutions = convolutions

        total_filters = sum(num_filters for _, num_filters in self._filters)
        self._highways = CharacterBertHighway(total_filters, config.num_highway_layers)
        self._projection = nn.Linear(total_filters, config.hidden_size, bias=True)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        if input_ids.ndim < 2:
            raise ValueError(
                "CharacterBERT expects `input_ids` to have shape (*token_dims, max_characters_per_token)."
            )

        *token_dims, max_characters = input_ids.shape
        if max_characters != self.max_characters_per_token:
            raise ValueError(
                f"CharacterBERT expects max_characters_per_token={self.max_characters_per_token} but got "
                f"{max_characters}."
            )

        num_tokens = 1
        for dim in token_dims:
            num_tokens *= dim

        character_embeddings = nn.functional.embedding(
            input_ids.reshape(num_tokens, max_characters),
            self._char_embedding_weights,
        )
        character_embeddings = torch.transpose(character_embeddings, 1, 2)

        conv_outputs = []
        for index in range(len(self._convolutions)):
            conv = getattr(self, f"char_conv_{index}")
            convolved = conv(character_embeddings)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = nn.functional.relu(convolved)
            conv_outputs.append(convolved)

        token_embeddings = torch.cat(conv_outputs, dim=-1)
        token_embeddings = self._highways(token_embeddings)
        token_embeddings = self._projection(token_embeddings)

        return token_embeddings.view(*token_dims, self.hidden_size)


class CharacterBertEmbeddings(nn.Module):
    """Construct embeddings from character CNN, position, and token type embeddings."""

    def __init__(self, config: CharacterBertConfig):
        super().__init__()
        self.word_embeddings = CharacterBertCharacterCNN(config)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()[:-1]
        else:
            input_shape = inputs_embeds.size()[:-1]

        batch_size, sequence_length = input_shape

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : sequence_length + past_key_values_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
                buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
                token_type_ids = buffered_token_type_ids.expand(batch_size, sequence_length)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


@auto_docstring
class CharacterBertPreTrainedModel(BertPreTrainedModel):
    config_class = CharacterBertConfig
    base_model_prefix = "character_bert"
    _can_record_outputs = {
        "hidden_states": CharacterBertLayer,
        "attentions": CharacterBertSelfAttention,
        "cross_attentions": CharacterBertCrossAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, CharacterBertCharacterCNN):
            init.zeros_(module._char_embedding_weights)


@auto_docstring(
    custom_intro="""
    CharacterBERT model with a CharacterCNN token embedder and BERT encoder stack.
    """
)
class CharacterBertModel(CharacterBertPreTrainedModel):
    _no_split_modules = ["CharacterBertEmbeddings", "CharacterBertLayer"]

    def __init__(self, config: CharacterBertConfig, add_pooling_layer: bool = True):
        r"""
        add_pooling_layer (`bool`, *optional*, defaults to `True`):
            Whether to include the pooler over the first token hidden state.
        """
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        self.embeddings = CharacterBertEmbeddings(config)
        self.encoder = CharacterBertEncoder(config)
        self.pooler = CharacterBertPooler(config) if add_pooling_layer else None

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
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
            if input_ids.ndim != 3:
                raise ValueError(
                    "CharacterBERT expects `input_ids` to have shape "
                    "(batch_size, sequence_length, max_characters_per_token)."
                )
            device = input_ids.device
            seq_length = input_ids.shape[1]
        else:
            device = inputs_embeds.device
            seq_length = inputs_embeds.shape[1]

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(past_key_values_length, past_key_values_length + seq_length, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
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
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
        )

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
                inputs_embeds=embedding_output,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
            )
        else:
            attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=embedding_output,
                attention_mask=attention_mask,
            )

        if encoder_attention_mask is not None:
            encoder_attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=embedding_output,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        return attention_mask, encoder_attention_mask


@auto_docstring(
    custom_intro="""
    CharacterBERT model with a `masked language modeling` head on top.
    """
)
class CharacterBertForMaskedLM(CharacterBertPreTrainedModel):
    _tied_weights_keys = {
        "cls.predictions.decoder.bias": "cls.predictions.bias",
    }

    def __init__(self, config: CharacterBertConfig):
        super().__init__(config)

        self.character_bert = CharacterBertModel(config, add_pooling_layer=False)
        self.cls = CharacterBertOnlyMLMHead(config)
        # CharacterBERT keeps untied decoder weights; still share the duplicate MLM bias parameter.
        self.cls.predictions.bias = self.cls.predictions.decoder.bias

        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    def tie_weights(self, missing_keys: set[str] | None = None, recompute_mapping: bool = True):
        super().tie_weights(missing_keys=missing_keys, recompute_mapping=recompute_mapping)
        self.cls.predictions.decoder.bias = self.cls.predictions.bias
        if missing_keys is not None:
            missing_keys.discard("cls.predictions.decoder.bias")

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | MaskedLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        outputs = self.character_bert(
            input_ids,
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


@auto_docstring(
    custom_intro="""
    CharacterBERT model transformer with a sequence classification/regression head on top.
    """
)
class CharacterBertForSequenceClassification(CharacterBertPreTrainedModel):
    def __init__(self, config: CharacterBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.character_bert = CharacterBertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput:
        outputs = self.character_bert(
            input_ids,
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
class CharacterBertForMultipleChoice(CharacterBertPreTrainedModel):
    def __init__(self, config: CharacterBertConfig):
        super().__init__(config)

        self.character_bert = CharacterBertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

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
    ) -> tuple[torch.Tensor] | MultipleChoiceModelOutput:
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.reshape(-1, input_ids.size(-2), input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.reshape(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.character_bert(
            input_ids,
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
class CharacterBertForTokenClassification(CharacterBertPreTrainedModel):
    def __init__(self, config: CharacterBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.character_bert = CharacterBertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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
    ) -> tuple[torch.Tensor] | TokenClassifierOutput:
        outputs = self.character_bert(
            input_ids,
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
class CharacterBertForQuestionAnswering(CharacterBertPreTrainedModel):
    def __init__(self, config: CharacterBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.character_bert = CharacterBertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

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
        start_positions: torch.Tensor | None = None,
        end_positions: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | QuestionAnsweringModelOutput:
        outputs = self.character_bert(
            input_ids,
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
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

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
    "CharacterBertForMaskedLM",
    "CharacterBertForMultipleChoice",
    "CharacterBertForQuestionAnswering",
    "CharacterBertForSequenceClassification",
    "CharacterBertForTokenClassification",
    "CharacterBertModel",
    "CharacterBertPreTrainedModel",
]
