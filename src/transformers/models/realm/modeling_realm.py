# coding=utf-8
# Copyright 2021 The HuggingFace Team The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch REALM model. """


import math
import os

import torch
import torch.utils.checkpoint
from typing import Optional, Tuple
from dataclasses import dataclass
from packaging import version
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    ModelOutput
)
from ...modeling_utils import (
    PreTrainedModel,
    SequenceSummary,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import logging
from ..bert import BertModel
from .configuration_realm import RealmConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "realm-cc-news-pretrained"
_CONFIG_FOR_DOC = "RealmConfig"
_TOKENIZER_FOR_DOC = "RealmTokenizer"

REALM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "qqaatw/realm-cc-news-pretrained-embedder",
    "qqaatw/realm-cc-news-pretrained-bert"
    # See all REALM models at https://huggingface.co/models?filter=realm
]


def load_tf_weights_in_realm(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        original_name = name

        # embedder
        embedder_prefix = "" if isinstance(model, RealmEmbedder) else "embedder/"
        name = name.replace("module/module/module/bert/", f"{embedder_prefix}bert/")
        name = name.replace("module/module/LayerNorm/", f"{embedder_prefix}cls/LayerNorm/")
        name = name.replace("module/module/dense/", f"{embedder_prefix}cls/dense/")
        name = name.replace("module/module/module/cls/predictions/", f"{embedder_prefix}cls/predictions/")

        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in [
                "adam_v",
                "adam_m",
                "AdamWeightDecayOptimizer",
                "AdamWeightDecayOptimizer_1",
                "global_step"
            ]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            #elif scope_names[0] == "squad":
            #    pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


@dataclass
class BaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class RealmEmbedderOutput(ModelOutput):
    """
    Outputs of embedder models.

    Args:
        projected_score (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.retriever_proj_size)`):
            Projected scores.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    projected_score: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RealmPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class RealmLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = RealmPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class RealmOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = RealmLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class RealmRetrieverProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = RealmLMPredictionHead(config)
        self.dense = nn.Linear(config.hidden_size, config.retriever_proj_size)
        self.LayerNorm = nn.LayerNorm(config.retriever_proj_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        #hidden_states = self.predictions(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class RealmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = RealmConfig
    load_tf_weights = load_tf_weights_in_realm
    base_model_prefix = "realm"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _flatten_inputs(self, *inputs):        
        flattened_inputs = []
        for tensor in inputs:
            input_shape = tensor.shape
            if len(input_shape) > 2:
                tensor = tensor.view((-1, input_shape[-1]))
            flattened_inputs.append(tensor)
        return flattened_inputs
            
            
REALM_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.RealmConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

REALM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.RealmTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.__call__` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The embedder of REALM outputting raw hidden-states without any specific head on top.",
    REALM_START_DOCSTRING,
)
class RealmEmbedder(RealmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(self.config)
        self.cls = RealmRetrieverProjection(self.config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # [batch_size, hidden_size]
        pooler_output = bert_outputs.pooler_output
        # [batch_size, retriever_proj_size]
        projected_score = self.cls(pooler_output)
        
        if not return_dict:
            return (projected_score,) + bert_outputs[1:]
        else:
            return RealmEmbedderOutput(
                projected_score = projected_score,
                hidden_states = bert_outputs.hidden_states,
                attentions = bert_outputs.attentions,
            )


@add_start_docstrings(
    "The retriever of REALM outputting raw hidden-states without any specific head on top.",
    REALM_START_DOCSTRING,
)
class RealmRetriever(RealmPreTrainedModel):
    def __init__(self, config, query_embedder=None):
        super().__init__(config)

        self.embedder = RealmEmbedder(self.config)
        
        if query_embedder:
            self.query_embedder = query_embedder
        else:
            self.query_embedder = self.embedder
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        candidate_input_ids=None,
        candidate_attention_mask=None,
        candidate_token_type_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        query_outputs = self.query_embedder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # [batch_size * num_candidates, candidate_seq_len]
        (
            flattened_input_ids, 
            flattened_attention_mask,
            flattened_token_type_ids
        ) = self._flatten_inputs(
            candidate_input_ids, 
            candidate_attention_mask, 
            candidate_token_type_ids
        )

        candidate_outputs = self.embedder(
            flattened_input_ids,
            attention_mask=flattened_attention_mask,
            token_type_ids=flattened_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # [batch_size, retriever_proj_size]
        query_score = query_outputs[0]
        # [batch_size * num_candidates, retriever_proj_size]
        candidate_score = candidate_outputs[0]
        # [batch_size, num_candidates, retriever_proj_size]
        candidate_score = candidate_score.view(-1, self.config.num_candidates, self.config.retriever_proj_size)
        # [batch_size, num_candidates]
        relevance_score = torch.einsum("BD,BND->BN", query_score, candidate_score)

        return relevance_score, query_score, candidate_score


@add_start_docstrings(
    "The encoder of REALM outputting raw hidden-states without any specific head on top.",
    REALM_START_DOCSTRING,
)
class RealmEncoder(RealmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(self.config)
        self.cls = RealmOnlyMLMHead(self.config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.bert.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        relevance_score=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        mlm_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        (
            flattened_input_ids, 
            flattened_attention_mask, 
            flattened_token_type_ids
        ) = self._flatten_inputs(
            input_ids, 
            attention_mask, 
            token_type_ids
        )

        joint_outputs = self.bert(
            flattened_input_ids,
            attention_mask=flattened_attention_mask,
            token_type_ids=flattened_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # [batch_size * num_candidates, joint_seq_len, hidden_size]
        joint_output = joint_outputs[0]
        # [batch_size * num_candidates, joint_seq_len, vocab_size]
        prediction_scores = self.cls(joint_output)
        # [batch_size, num_candidates]
        candidate_score = relevance_score

        masked_lm_loss = None
        if labels is not None:
            if mlm_mask is None:
                mlm_mask = torch.ones_like(labels, dtype=torch.float32)
            else:
                mlm_mask = mlm_mask.type(torch.float32)

            # Compute marginal log-likelihood
            loss_fct = CrossEntropyLoss(reduction='none')  # -100 index = padding token
            
            # [batch_size * num_candidates * joint_seq_len, vocab_size]
            mlm_logits = prediction_scores.view(-1, self.config.vocab_size)
            # [batch_size * num_candidates * joint_seq_len]
            mlm_targets = labels.tile(1, self.config.num_candidates).view(-1)
            # [batch_size, num_candidates, joint_seq_len]
            masked_lm_log_prob = -loss_fct(mlm_logits, mlm_targets).view_as(input_ids)
            # [batch_size, num_candidates, 1]
            candidate_log_prob = candidate_score.log_softmax(-1).unsqueeze(-1)
            # [batch_size, num_candidates, joint_seq_len]
            joint_gold_log_prob = candidate_log_prob + masked_lm_log_prob
            # [batch_size, joint_seq_len]
            marginal_gold_log_probs = joint_gold_log_prob.logsumexp(1)
            # []
            masked_lm_loss = -torch.nansum(
                torch.sum(marginal_gold_log_probs * mlm_mask) / 
                torch.sum(mlm_mask)
            )

        if not return_dict:
            output = (prediction_scores,) + joint_outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=joint_outputs.hidden_states,
            attentions=joint_outputs.attentions,
        )