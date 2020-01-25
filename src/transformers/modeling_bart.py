# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""PyTorch BART model, ported from the fairseq repo."""


import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from .configuration_bart import BartConfig
from .file_utils import add_start_docstrings
from .modeling_bert import BertEmbeddings, BertLayerNorm, BertPreTrainedModel, gelu
#import .f
from .fairseq_utils import *


logger = logging.getLogger(__name__)


BART_PRETRAINED_MODEL_ARCHIVE_MAP = {  # TODO(SS): copy to S3
    'bart.large': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz',
    'bart.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz',
    'bart.large.cnn': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz',
}


BART_START_DOCSTRING = r"""  TODOSS: FIXME)  The RoBERTa model was proposed in
    `RoBERTa: A Robustly Optimized BERT Pretraining Approach`_
    by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer,
    Veselin Stoyanov. It is based on Google's BERT model released in 2018.

    It builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining
    objective and training with much larger mini-batches and learning rates.

    This implementation is the same as BertModel with a tiny embeddings tweak as well as a setup for Roberta pretrained
    models.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`RoBERTa: A Robustly Optimized BERT Pretraining Approach`:
        https://arxiv.org/abs/1907.11692

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, RoBERTa input sequence should be formatted with <s> and </s> tokens as follows:

            (a) For sequence pairs:

                ``tokens:         <s> Is this Jacksonville ? </s> </s> No it is not . </s>``

            (b) For single sequences:

                ``tokens:         <s> the dog is hairy . </s>``

            Fully encoded sequences or sequence pairs can be obtained using the RobertaTokenizer.encode function with
            the ``add_special_tokens`` parameter set to ``True``.

            RoBERTa is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional` need to be trained) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Optional segment token indices to indicate first and second portions of the inputs.
            This embedding matrice is not trained (not pretrained during RoBERTa pretraining), you will have to train it
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


class BARTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    # copied from fairseq

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



@add_start_docstrings(
    "The bare BART Model transformer outputting raw hidden-states without any specific head on top.",
    BART_START_DOCSTRING,
    ROBERTA_INPUTS_DOCSTRING,
)
class BARTModel(TransformerModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

    # @staticmethod
    # def add_args(parser):
    #     super(BARTModel, BARTModel).add_args(parser)
    #     parser.add_argument(
    #         '--pooler-dropout', type=float, metavar='D',
    #         help='dropout probability in the masked_lm pooler layers'
    #     )
    #     parser.add_argument(
    #         '--pooler-activation-fn',
    #         choices=utils.get_available_activation_fns(),
    #         help='activation function to use for pooler layer'
    #     )

    def _forward(
        self, src_tokens, src_lengths, prev_output_tokens,
        features_only=False, classification_head_name=None, **kwargs
    ):
        if classification_head_name is not None:
            features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            **kwargs,
        )
        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            **kwargs,
        )

        if classification_head_name is not None:
            sentence_representation = x[
                src_tokens.eq(self.encoder.dictionary.eos()), :
            ].view(x.size(0), -1, x.size(-1))[:, -1, :]
            x = self.classification_heads[classification_head_name](
                sentence_representation
            )
        return x, extra


    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + '.' if name != '' else ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict['encoder.embed_tokens.weight'].size(0)
        if loaded_dict_size == len(self.encoder.dictionary) + 1 and '<mask>' not in self.encoder.dictionary:
            state_dict['encoder.embed_tokens.weight'] = state_dict['encoder.embed_tokens.weight'][:loaded_dict_size-1, :]
            state_dict['decoder.embed_tokens.weight'] = state_dict['decoder.embed_tokens.weight'][:loaded_dict_size-1, :]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    logger.info('Overwriting', prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v



#if __name__ == '__main__':
