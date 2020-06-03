# coding=utf-8
# Copyright 2018 DPR Authors
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
""" PyTorch DPR model. """

####################################################
# In this template, replace all the DPR (various casings) with your model name
####################################################


import collections
import logging
from typing import Tuple

import torch
from torch import Tensor as T
from torch import nn
from torch.serialization import default_restore_location

from .configuration_dpr import DprConfig
from .file_utils import add_start_docstrings
from .modeling_bert import BertConfig, BertModel
from .modeling_utils import PreTrainedModel


logger = logging.getLogger(__name__)

####################################################
# This list contrains shortcut names for some of
# the pretrained weights provided with the models
####################################################
DPR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "dpr-base-uncased",
]

####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (itself a sub-class of torch.nn.Module)
####################################################


#############
# BertEncoder
#############


class DprBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0., **kwargs) -> BertModel:
        cfg = BertConfig.from_pretrained(cfg_name)
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, **kwargs)

    def forward(self, input_ids: T, token_type_ids: T, attention_mask: T) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )
        pooled_output = sequence_output[:, 0, :]
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


###########
# BiEncoder
###########

# Keep this class for now as it can be used to directly load the weights from the official code

BiEncoderBatch = collections.namedtuple(
    "BiENcoderInput",
    ["question_ids", "question_segments", "context_ids", "ctx_segments", "is_positive", "hard_negatives"],
)


class DprBiEncoder(nn.Module):
    """
    Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """

    def __init__(
        self, question_model: nn.Module, ctx_model: nn.Module,
    ):
        super(DprBiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model

    @staticmethod
    def get_representation(
        sub_model: nn.Module, ids: T, segments: T, attn_mask: T, fix_encoder: bool = False
    ) -> (T, T, T):
        sequence_output = None
        pooled_output = None
        hidden_states = None
        if ids is not None:
            with torch.no_grad():
                sequence_output, pooled_output, hidden_states = sub_model(ids, segments, attn_mask)
        return sequence_output, pooled_output, hidden_states


###########
# Reader
###########


ReaderBatch = collections.namedtuple("ReaderBatch", ["input_ids", "start_positions", "end_positions", "answers_mask"])


class Reader(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size):
        super(Reader, self).__init__()
        self.encoder = encoder
        self.qa_outputs = nn.Linear(hidden_size, 2)
        self.qa_classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: T, attention_mask: T, start_positions=None, end_positions=None, answer_mask=None):
        # notations: N - number of questions in a batch, M - number of passages per questions, L - sequence length
        N, M, L = input_ids.size()
        start_logits, end_logits, relevance_logits = self._forward(
            input_ids.view(N * M, L), attention_mask.view(N * M, L)
        )
        return start_logits.view(N, M, L), end_logits.view(N, M, L), relevance_logits.view(N, M)

    def _forward(self, input_ids, attention_mask):
        sequence_output, _pooled_output, _hidden_states = self.encoder(input_ids, None, attention_mask)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        rank_logits = self.qa_classifier(sequence_output[:, 0, :])
        return start_logits, end_logits, rank_logits


##############
# Providers
##############


def get_bert_biencoder_components(config: DprConfig, **kwargs):
    question_encoder = DprBertEncoder.init_encoder(
        config.pretrained_model_cfg, projection_dim=config.projection_dim, **kwargs
    )
    ctx_encoder = DprBertEncoder.init_encoder(
        config.pretrained_model_cfg, projection_dim=config.projection_dim, **kwargs
    )
    biencoder = DprBiEncoder(question_encoder, ctx_encoder)
    return biencoder


def get_bert_reader_components(config: DprConfig, **kwargs):
    encoder = DprBertEncoder.init_encoder(
        config.pretrained_model_cfg, projection_dim=config.projection_dim, **kwargs
    )
    hidden_size = encoder.config.hidden_size
    reader = Reader(encoder, hidden_size)
    return reader


####################################################
# PreTrainedModel is a sub-class of torch.nn.Module
# which take care of loading and saving pretrained weights
# and various common utilities.
#
# Here you just need to specify a few (self-explanatory)
# pointers for your model and the weights initialization
# method if its not fully covered by PreTrainedModel's default method
####################################################


class DprPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = DprConfig
    load_tf_weights = None
    base_model_prefix = "dpr"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


DPR_START_DOCSTRING = r"""    The DPR model was proposed in
    `DPR: Pre-training of Deep Bidirectional Transformers for Language Understanding`_
    by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It's a bidirectional transformer
    pre-trained using a combination of masked language modeling objective and next sentence prediction
    on a large corpus comprising the Toronto Book Corpus and Wikipedia.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`DPR: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1810.04805

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.DprConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

DPR_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, DPR input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``

                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``

                ``token_type_ids:   0   0   0   0  0     0   0``

            Dpr is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            Indices can be obtained using :class:`transformers.DprTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `DPR: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


CheckpointState = collections.namedtuple("CheckpointState",
                                         ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset', 'epoch',
                                          'encoder_params'])


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    return CheckpointState(**state_dict)


@add_start_docstrings(
    "The bare Dpr Model transformer outputting raw hidden-states without any specific head on top.",
    DPR_START_DOCSTRING,
    DPR_INPUTS_DOCSTRING,
)
class DprModel(DprPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Dpr pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = DprTokenizer.from_pretrained('dpr-base-uncased')
        model = DprModel.from_pretrained('dpr-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    def __init__(self, config: DprConfig):
        super().__init__(config)
        self.config = config
        self.biencoder = get_bert_biencoder_components(config)
        self.reader = get_bert_reader_components(config)
        self.init_weights()
    
    def init_weights(self):
        super().init_weights()
        if self.config.biencoder_model_file is not None:
            saved_state = load_states_from_checkpoint(self.config.biencoder_model_file)
            self.biencoder.load_state_dict(saved_state.model_dict)
        if self.config.reader_model_file is not None:
            saved_state = load_states_from_checkpoint(self.config.reader_model_file)
            self.reader.load_state_dict(saved_state.model_dict)

    def forward(self, index, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        ##################################
        # Replace this with your model code
        outputs = self.bidencoder.question_model(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        # _q_seq, q_pooled_out, _q_hidden = self.biencoder.get_representation(self.biencoder.question_model, question_ids, question_segments,
        #                                                           question_attn_mask, self.config.fix_q_encoder)

        # TODO(dpr): query index and get top 100
        # TODO(dpr): build reader input and do passage selection
        # TODO(dpr): return sequence_output, hidden_states, attention, retrieved passages and scores
        return outputs  # sequence_output, (hidden_states), (attentions)
