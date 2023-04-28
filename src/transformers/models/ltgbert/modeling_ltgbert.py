# coding=utf-8
# Copyright 2023 Language Technology Group from University of Oslo and The HuggingFace Inc. team.
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

""" PyTorch LTG-BERT model."""


import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

from configuration_ltgbert import LtgBertConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.activations import gelu_new
from transformers.modeling_outputs import (
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    BaseModelOutput
)
from transformers.pytorch_utils import softmax_backward_data
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward


_CHECKPOINT_FOR_DOC = "ltg/bnc-bert-span"
_CONFIG_FOR_DOC = "LtgBertConfig"


LTG_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bnc-bert-span",
    "bnc-bert-span-2x":,
    "bnc-bert-span-0.5x",
    "bnc-bert-span-0.25x",
    "bnc-bert-span-order",
    "bnc-bert-span-document",
    "bnc-bert-span-word",
    "bnc-bert-span-subword",

    "norbert3-xs",
    "norbert3-small",
    "norbert3-base",
    "norbert3-large",

    "norbert3-oversampled-base",
    "norbert3-ncc-base",
    "norbert3-nak-base",
    "norbert3-nb-base",
    "norbert3-wiki-base",
    "norbert3-c4-base"
]


class Encoder(nn.Module):
    def __init__(self, config, activation_checkpointing=False):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

        for i, layer in enumerate(self.layers):
            layer.mlp.mlp[1].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.mlp.mlp[-2].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))

        self.activation_checkpointing = activation_checkpointing
    
    def forward(self, hidden_states, attention_mask, relative_embedding):
        hidden_states, attention_probs = [hidden_states], []

        for layer in self.layers:
            if self.activation_checkpointing:
                hidden_state, attention_p = checkpoint.checkpoint(layer, hidden_states[-1], attention_mask, relative_embedding)
            else:
                hidden_state, attention_p = layer(hidden_states[-1], attention_mask, relative_embedding)

            hidden_states.append(hidden_state)
            attention_probs.append(attention_p)

        return hidden_states, attention_probs


class MaskClassifier(nn.Module):
    def __init__(self, config, subword_embedding):
        super().__init__()
        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(subword_embedding.size(1), subword_embedding.size(0))
        )
        self.initialize(config.hidden_size, subword_embedding)

    def initialize(self, hidden_size, embedding):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.nonlinearity[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.nonlinearity[-1].weight = embedding
        self.nonlinearity[1].bias.data.zero_()
        self.nonlinearity[-1].bias.data.zero_()

    def forward(self, x, masked_lm_labels=None):
        if masked_lm_labels is not None:
            x = torch.index_select(x.flatten(0, 1), 0, torch.nonzero(masked_lm_labels.flatten() != -100).squeeze())
        x = self.nonlinearity(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = FeedForward(config)

    def forward(self, x, padding_mask, relative_embedding):
        attention_output, attention_probs = self.attention(x, padding_mask, relative_embedding)
        x = x + attention_output
        x = x + self.mlp(x)
        return x, attention_probs


class GeGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        x = x * gelu_new(gate)
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, 2*config.intermediate_size, bias=False),
            GeGLU(),
            nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.initialize(config.hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.mlp[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.mlp[-2].weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, x):
        return self.mlp(x)


class MaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(self, x, mask, dim):
        self.dim = dim
        x.masked_fill_(mask, float('-inf'))
        x = torch.softmax(x, self.dim)
        x.masked_fill_(mask, 0.0)
        self.save_for_backward(x)
        return x

    @staticmethod
    def backward(self, grad_output):
        output, = self.saved_tensors
        input_grad = softmax_backward_data(self, grad_output, output, self.dim, output)
        return input_grad, None, None


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads

        self.in_proj_qk = nn.Linear(config.hidden_size, 2*config.hidden_size, bias=True)
        self.in_proj_v = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.pre_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False)
        self.post_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        position_indices = torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(1) \
            - torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(0)
        position_indices = self.make_log_bucket_position(position_indices, config.position_bucket_size, config.max_position_embeddings)
        position_indices = config.position_bucket_size - 1 + position_indices
        self.register_buffer("position_indices", position_indices, persistent=True)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale = 1.0 / math.sqrt(3 * self.head_size)
        self.initialize()

    def make_log_bucket_position(self, relative_pos, bucket_size, max_position):
        sign = torch.sign(relative_pos)
        mid = bucket_size // 2
        abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, torch.abs(relative_pos).clamp(max=max_position - 1))
        log_pos = torch.ceil(torch.log(abs_pos / mid) / math.log((max_position-1) / mid) * (mid - 1)).int() + mid
        bucket_pos = torch.where(abs_pos <= mid, relative_pos, log_pos * sign).long()
        return bucket_pos

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.in_proj_qk.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.in_proj_v.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.in_proj_qk.bias.data.zero_()
        self.in_proj_v.bias.data.zero_()
        self.out_proj.bias.data.zero_()

    def compute_attention_scores(self, hidden_states, relative_embedding):
        key_len, batch_size, _ = hidden_states.size()
        query_len = key_len

        if self.position_indices.size(0) < query_len:
            position_indices = torch.arange(query_len, dtype=torch.long).unsqueeze(1) \
                - torch.arange(query_len, dtype=torch.long).unsqueeze(0)
            position_indices = self.make_log_bucket_position(position_indices, self.position_bucket_size, 512)
            position_indices = self.position_bucket_size - 1 + position_indices
            self.position_indices = position_indices.to(hidden_states.device)

        hidden_states = self.pre_layer_norm(hidden_states)

        query, key = self.in_proj_qk(hidden_states).chunk(2, dim=2)  # shape: [T, B, D]
        value = self.in_proj_v(hidden_states)  # shape: [T, B, D]

        query = query.reshape(query_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        key = key.reshape(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        value = value.view(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)

        attention_scores = torch.bmm(query, key.transpose(1, 2) * self.scale)

        pos = self.in_proj_qk(self.dropout(relative_embedding))  # shape: [2T-1, 2D]
        query_pos, key_pos = pos.view(-1, self.num_heads, 2*self.head_size).chunk(2, dim=2)
        query = query.view(batch_size, self.num_heads, query_len, self.head_size)
        key = key.view(batch_size, self.num_heads, query_len, self.head_size)

        attention_c_p = torch.einsum("bhqd,khd->bhqk", query, key_pos.squeeze(1) * self.scale)
        attention_p_c = torch.einsum("bhkd,qhd->bhqk", key * self.scale, query_pos.squeeze(1))

        position_indices = self.position_indices[:query_len, :key_len].expand(batch_size, self.num_heads, -1, -1)
        attention_c_p = attention_c_p.gather(3, position_indices)
        attention_p_c = attention_p_c.gather(2, position_indices)

        attention_scores = attention_scores.view(batch_size, self.num_heads, query_len, key_len)
        attention_scores.add_(attention_c_p)
        attention_scores.add_(attention_p_c)

        return attention_scores, value

    def compute_output(self, attention_probs, value):
        attention_probs = self.dropout(attention_probs)
        context = torch.bmm(attention_probs.flatten(0, 1), value)  # shape: [B*H, Q, D]
        context = context.transpose(0, 1).reshape(context.size(1), -1, self.hidden_size)  # shape: [Q, B, H*D]
        context = self.out_proj(context)
        context = self.post_layer_norm(context)
        context = self.dropout(context)
        return context

    def forward(self, hidden_states, attention_mask, relative_embedding):
        attention_scores, value = self.compute_attention_scores(hidden_states, relative_embedding)
        attention_probs = MaskedSoftmax.apply(attention_scores, attention_mask, -1)
        return self.compute_output(attention_probs, value), attention_probs.detach()


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.relative_embedding = nn.Parameter(torch.empty(2 * config.position_bucket_size - 1, config.hidden_size))
        self.relative_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.initialize()

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.relative_embedding, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.word_embedding.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, input_ids):
        word_embedding = self.dropout(self.word_layer_norm(self.word_embedding(input_ids)))
        relative_embeddings = self.relative_layer_norm(self.relative_embedding)
        return word_embedding, relative_embeddings


#
# HuggingFace wrappers
#

class LtgBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LtgBertConfig
    base_model_prefix = "bnc-bert"
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Encoder):
            module.activation_checkpointing = value

    def _init_weights(self, _):
        pass  # everything is already initialized


LTG_BERT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`LtgBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

LTG_BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LTG-BERT transformer outputting raw hidden-states without any specific head on top.",
    LTG_BERT_START_DOCSTRING,
)
class LtgBertModel(LtgBertPreTrainedModel):
    def __init__(self, config, add_mlm_layer=False):
        super().__init__(config)
        self.config = config

        self.embedding = Embedding(config)
        self.transformer = Encoder(config, activation_checkpointing=False)
        self.classifier = MaskClassifier(config, self.embedding.word_embedding.weight) if add_mlm_layer else None

    def get_input_embeddings(self):
        return self.embedding.word_embedding

    def set_input_embeddings(self, value):
        self.embedding.word_embedding = value

    def get_contextualized_embeddings(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify input_ids")

        batch_size, seq_length = input_shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
        else:
            attention_mask = ~attention_mask.bool()
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
 
        static_embeddings, relative_embedding = self.embedding(input_ids.t())
        contextualized_embeddings, attention_probs = self.transformer(static_embeddings, attention_mask, relative_embedding)
        contextualized_embeddings = [e.transpose(0, 1) for e in contextualized_embeddings]
        last_layer = contextualized_embeddings[-1]
        contextualized_embeddings = [contextualized_embeddings[0]] + [
            contextualized_embeddings[i] - contextualized_embeddings[i - 1]
            for i in range(1, len(contextualized_embeddings))
        ]
        return last_layer, contextualized_embeddings, attention_probs

    @add_start_docstrings_to_model_forward(LTG_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output, contextualized_embeddings, attention_probs = self.get_contextualized_embeddings(input_ids, attention_mask)

        if not return_dict:
            return (
                sequence_output,
                *([contextualized_embeddings] if output_hidden_states else []),
                *([attention_probs] if output_attentions else [])
            )

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=contextualized_embeddings if output_hidden_states else None,
            attentions=attention_probs if output_attentions else None
        )


@add_start_docstrings("""LTG-BERT model with a `language modeling` head on top.""", LTG_BERT_START_DOCSTRING)
class LtgBertForMaskedLM(LtgBertModel):
    _keys_to_ignore_on_load_unexpected = ["head"]

    def __init__(self, config):
        super().__init__(config, add_mlm_layer=True)

    def get_output_embeddings(self):
        return self.classifier.nonlinearity[-1].weight

    def set_output_embeddings(self, new_embeddings):
        self.classifier.nonlinearity[-1].weight = new_embeddings

    @add_start_docstrings_to_model_forward(LTG_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output, contextualized_embeddings, attention_probs = self.get_contextualized_embeddings(input_ids, attention_mask)
        subword_prediction = self.classifier(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = F.cross_entropy(subword_prediction.flatten(0, 1), labels.flatten())

        if not return_dict:
            output = (
                subword_prediction,
                *([contextualized_embeddings] if output_hidden_states else []),
                *([attention_probs] if output_attentions else [])
            )
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=subword_prediction,
            hidden_states=contextualized_embeddings if output_hidden_states else None,
            attentions=attention_probs if output_attentions else None
        )


class Classifier(nn.Module):
    def __init__(self, config, num_labels: int):
        super().__init__()

        drop_out = getattr(config, "classifier_dropout", config.hidden_dropout_prob)

        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Dropout(drop_out),
            nn.Linear(config.hidden_size, num_labels)
        )
        self.initialize(config.hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.nonlinearity[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.nonlinearity[-1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.nonlinearity[1].bias.data.zero_()
        self.nonlinearity[-1].bias.data.zero_()

    def forward(self, x):
        x = self.nonlinearity(x)
        return x


@add_start_docstrings(
    """
    LTG-BERT model with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    LTG_BERT_START_DOCSTRING,
)
class LtgBertForSequenceClassification(LtgBertModel):
    _keys_to_ignore_on_load_unexpected = ["classifier"]
    _keys_to_ignore_on_load_missing = ["head"]

    def __init__(self, config):
        super().__init__(config, add_mlm_layer=False)

        self.num_labels = config.num_labels
        self.head = Classifier(config, self.num_labels)

    @add_start_docstrings_to_model_forward(LTG_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output, contextualized_embeddings, attention_probs = self.get_contextualized_embeddings(input_ids, attention_mask)
        logits = self.head(sequence_output[:, 0, :])

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
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (
                logits,
                *([contextualized_embeddings] if output_hidden_states else []),
                *([attention_probs] if output_attentions else [])
            )
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=contextualized_embeddings if output_hidden_states else None,
            attentions=attention_probs if output_attentions else None
        )


@add_start_docstrings(
    """
    LTG-BERT model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    LTG_BERT_START_DOCSTRING,
)
class LtgBertForTokenClassification(LtgBertModel):
    _keys_to_ignore_on_load_unexpected = ["classifier"]
    _keys_to_ignore_on_load_missing = ["head"]

    def __init__(self, config):
        super().__init__(config, add_mlm_layer=False)

        self.num_labels = config.num_labels
        self.head = Classifier(config, self.num_labels)

    @add_start_docstrings_to_model_forward(LTG_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output, contextualized_embeddings, attention_probs = self.get_contextualized_embeddings(input_ids, attention_mask)
        logits = self.head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (
                logits,
                *([contextualized_embeddings] if output_hidden_states else []),
                *([attention_probs] if output_attentions else [])
            )
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=contextualized_embeddings if output_hidden_states else None,
            attentions=attention_probs if output_attentions else None
        )


@add_start_docstrings(
    """
    LTG-BERT model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LTG_BERT_START_DOCSTRING,
)
class LtgBertForQuestionAnswering(LtgBertModel):
    _keys_to_ignore_on_load_unexpected = ["classifier"]
    _keys_to_ignore_on_load_missing = ["head"]

    def __init__(self, config):
        super().__init__(config, add_mlm_layer=False)

        self.num_labels = config.num_labels
        self.head = Classifier(config, self.num_labels)

    @add_start_docstrings_to_model_forward(LTG_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output, contextualized_embeddings, attention_probs = self.get_contextualized_embeddings(input_ids, attention_mask)
        logits = self.head(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

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

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (
                start_logits,
                end_logits,
                *([contextualized_embeddings] if output_hidden_states else []),
                *([attention_probs] if output_attentions else [])
            )
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=contextualized_embeddings if output_hidden_states else None,
            attentions=attention_probs if output_attentions else None
        )


@add_start_docstrings(
    """
    LTG-BERT model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    LTG_BERT_START_DOCSTRING,
)
class LtgBertForMultipleChoice(LtgBertModel):
    _keys_to_ignore_on_load_unexpected = ["classifier"]
    _keys_to_ignore_on_load_missing = ["head"]

    def __init__(self, config):
        super().__init__(config, add_mlm_layer=False)

        self.num_labels = getattr(config, "num_labels", 2)
        self.head = Classifier(config, self.num_labels)

    @add_start_docstrings_to_model_forward(LTG_BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        sequence_output, contextualized_embeddings, attention_probs = self.get_contextualized_embeddings(flat_input_ids, flat_attention_mask)
        logits = self.head(sequence_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (
                reshaped_logits,
                *([contextualized_embeddings] if output_hidden_states else []),
                *([attention_probs] if output_attentions else [])
            )
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=contextualized_embeddings if output_hidden_states else None,
            attentions=attention_probs if output_attentions else None
        )
