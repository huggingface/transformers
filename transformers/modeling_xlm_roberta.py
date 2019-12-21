# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch XLM-RoBERTa model. """

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

from .modeling_roberta import RobertaModel, RobertaForMaskedLM, RobertaForSequenceClassification, RobertaForMultipleChoice, RobertaForTokenClassification
from .configuration_xlm_roberta import XLMRobertaConfig
from .file_utils import add_start_docstrings

logger = logging.getLogger(__name__)

XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'xlm-roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-base-pytorch_model.bin",
    'xlm-roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-pytorch_model.bin",
    'xlm-roberta-large-finetuned-conll02-dutch': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-dutch-pytorch_model.bin",
    'xlm-roberta-large-finetuned-conll02-spanish': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-spanish-pytorch_model.bin",
    'xlm-roberta-large-finetuned-conll03-english': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-english-pytorch_model.bin",
    'xlm-roberta-large-finetuned-conll03-german': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-german-pytorch_model.bin",
}


XLM_ROBERTA_START_DOCSTRING = r"""    The XLM-RoBERTa model was proposed in
    `Unsupervised Cross-lingual Representation Learning at Scale`_
    by Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov. It is based on Facebook's RoBERTa model released in 2019.
    
    It is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl data.

    This implementation is the same as RoBERTa.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`Unsupervised Cross-lingual Representation Learning at Scale`:
        https://arxiv.org/abs/1911.02116

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.XLMRobertaConfig`): Model configuration class with all the parameters of the 
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

XLM_ROBERTA_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, XLM-RoBERTa input sequence should be formatted with <s> and </s> tokens as follows:

            (a) For sequence pairs:

                ``tokens:         <s> Is this Jacksonville ? </s> </s> No it is not . </s>``

            (b) For single sequences:

                ``tokens:         <s> the dog is hairy . </s>``

            Fully encoded sequences or sequence pairs can be obtained using the XLMRobertaTokenizer.encode function with 
            the ``add_special_tokens`` parameter set to ``True``.

            XLM-RoBERTa is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional` need to be trained) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Optional segment token indices to indicate first and second portions of the inputs.
            This embedding matrice is not trained (not pretrained during XLM-RoBERTa pretraining), you will have to train it
            during finetuning.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1[``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""

@add_start_docstrings("The bare XLM-RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
                      XLM_ROBERTA_START_DOCSTRING, XLM_ROBERTA_INPUTS_DOCSTRING)
class XLMRobertaModel(RobertaModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            eo match pre-training, XLM-RoBERTa input sequence should be formatted with <s> and </s> tokens as follows:

            (a) For sequence pairs:

                ``tokens:         <s> is this jack ##son ##ville ? </s> </s> no it is not . </s>``

                ``token_type_ids:   0   0  0    0    0     0       0   0   0     1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         <s> the dog is hairy . </s>``

                ``token_type_ids:   0   0   0   0  0     0   0``

            objective during Bert pretraining. This output is usually *not* a good summary
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

        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        model = XLMRobertaModel.from_pretrained('xlm-roberta-large')
        input_ids = torch.tensor(tokenizer.encode("Schloß Nymphenburg ist sehr schön .")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings("""XLM-RoBERTa Model with a `language modeling` head on top. """,
    XLM_ROBERTA_START_DOCSTRING, XLM_ROBERTA_INPUTS_DOCSTRING)
class XLMRobertaForMaskedLM(RobertaForMaskedLM):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-large')
        input_ids = torch.tensor(tokenizer.encode("Schloß Nymphenburg ist sehr schön .")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings("""XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer 
    on top of the pooled output) e.g. for GLUE tasks. """,
    XLM_ROBERTA_START_DOCSTRING, XLM_ROBERTA_INPUTS_DOCSTRING)
class XLMRobertaForSequenceClassification(RobertaForSequenceClassification):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-large')
        input_ids = torch.tensor(tokenizer.encode("Schloß Nymphenburg ist sehr schön .")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings("""XLM-RoBERTa Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    XLM_ROBERTA_START_DOCSTRING, XLM_ROBERTA_INPUTS_DOCSTRING)
class XLMRobertaForMultipleChoice(RobertaForMultipleChoice):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        model = XLMRobertaForMultipleChoice.from_pretrained('xlm-roberta-large')
        choices = ["Schloß Nymphenburg ist sehr schön .", "Der Schloßkanal auch !"]
        input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings("""XLM-RoBERTa Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    XLM_ROBERTA_START_DOCSTRING, XLM_ROBERTA_INPUTS_DOCSTRING)
class XLMRobertaForTokenClassification(RobertaForTokenClassification):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        model = XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-large')
        input_ids = torch.tensor(tokenizer.encode("Schloß Nymphenburg ist sehr schön .", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
