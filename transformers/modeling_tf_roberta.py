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
""" TF 2.0 RoBERTa model. """

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import numpy as np
import tensorflow as tf

from .configuration_roberta import RobertaConfig
from .modeling_tf_utils import TFPreTrainedModel, get_initializer
from .file_utils import add_start_docstrings
from .modeling_tf_pytorch_utils import load_pytorch_checkpoint_in_tf2_model

from .modeling_tf_bert import TFBertEmbeddings, TFBertMainLayer, gelu, gelu_new

logger = logging.getLogger(__name__)

TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-tf_model.h5",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-tf_model.h5",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-tf_model.h5",
}

def load_roberta_pt_weights_in_tf2(tf_model, pytorch_checkpoint_path):
    # build the network
    inputs_list = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
    tf_inputs = tf.constant(inputs_list)
    tfo = tf_model(tf_inputs, training=False)
    return load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path, tf_inputs=tf_inputs)


class TFRobertaEmbeddings(TFBertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config, **kwargs):
        super(TFRobertaEmbeddings, self).__init__(config, **kwargs)
        self.padding_idx = 1

    def _embedding(self, inputs, training=False):
        """Applies embedding based on inputs tensor."""
        input_ids, position_ids, token_type_ids = inputs

        seq_length = tf.shape(input_ids)[1]
        if position_ids is None:
            position_ids = tf.range(self.padding_idx+1, seq_length+self.padding_idx+1, dtype=tf.int32)[tf.newaxis, :]

        return super(TFRobertaEmbeddings, self)._embedding([input_ids, position_ids, token_type_ids], training=training)


class TFRobertaMainLayer(TFBertMainLayer):
    """
    Same as TFBertMainLayer but uses TFRobertaEmbeddings.
    """
    def __init__(self, config, **kwargs):
        super(TFRobertaMainLayer, self).__init__(config, **kwargs)
        self.embeddings = TFRobertaEmbeddings(config, name='embeddings')

    def call(self, inputs, **kwargs):
        # Check that input_ids starts with control token
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
        elif isinstance(inputs, dict):
            input_ids = inputs.get('input_ids')
        else:
            input_ids = inputs

        if tf.not_equal(tf.reduce_sum(input_ids[:, 0]), 0):
            logger.warning("A sequence with no special tokens has been passed to the RoBERTa model. "
                           "This model requires special tokens in order to work. "
                           "Please specify add_special_tokens=True in your encoding.")

        return super(TFRobertaMainLayer, self).call(inputs, **kwargs)


class TFRobertaPreTrainedModel(TFPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    load_pt_weights = load_roberta_pt_weights_in_tf2
    base_model_prefix = "roberta"


ROBERTA_START_DOCSTRING = r"""    The RoBERTa model was proposed in
    `RoBERTa: A Robustly Optimized BERT Pretraining Approach`_
    by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer,
    Veselin Stoyanov. It is based on Google's BERT model released in 2018.
    
    It builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining
    objective and training with much larger mini-batches and learning rates.
    
    This implementation is the same as BertModel with a tiny embeddings tweak as well as a setup for Roberta pretrained 
    models.

    This model is a tf.keras.Model `tf.keras.Model`_ sub-class. Use it as a regular TF 2.0 Keras Model and
    refer to the TF 2.0 documentation for all matter related to general usage and behavior.

    .. _`RoBERTa: A Robustly Optimized BERT Pretraining Approach`:
        https://arxiv.org/abs/1907.11692

    .. _`tf.keras.Model`:
        https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model

    Note on the model inputs:
        TF 2.0 models accepts two formats as inputs:

            - having all inputs as keyword arguments (like PyTorch models), or
            - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is usefull when using `tf.keras.Model.fit()` method which currently requires having all the tensors in the first argument of the model call function: `model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the first positional argument :

        - a single Tensor with input_ids only and nothing else: `model(inputs_ids)
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
            `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associaed to the input names given in the docstring:
            `model({'input_ids': input_ids, 'token_type_ids': token_type_ids})`

    Parameters:
        config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the 
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
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
        **attention_mask**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional` need to be trained) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Optional segment token indices to indicate first and second portions of the inputs.
            This embedding matrice is not trained (not pretrained during RoBERTa pretraining), you will have to train it
            during finetuning.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **position_ids**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1[``.
        **head_mask**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""

@add_start_docstrings("The bare RoBERTa Model transformer outputing raw hidden-states without any specific head on top.",
                      ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class TFRobertaModel(TFRobertaPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``tf.Tensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``tf.Tensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import RobertaTokenizer, TFRobertaModel

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = TFRobertaModel.from_pretrained('roberta-base')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config, *inputs, **kwargs):
        super(TFRobertaModel, self).__init__(config, *inputs, **kwargs)
        self.roberta = TFRobertaMainLayer(config, name='roberta')

    def call(self, inputs, **kwargs):
        outputs = self.roberta(inputs, **kwargs)
        return outputs


class TFRobertaLMHead(tf.keras.layers.Layer):
    """Roberta Head for masked language modeling."""
    def __init__(self, config, input_embeddings, **kwargs):
        super(TFRobertaLMHead, self).__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.dense = tf.keras.layers.Dense(config.hidden_size,
                                           kernel_initializer=get_initializer(config.initializer_range),
                                           name='dense')
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        self.act = tf.keras.layers.Activation(gelu)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = input_embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,),
                                    initializer='zeros',
                                    trainable=True,
                                    name='bias')
        super(TFRobertaLMHead, self).build(input_shape)

    def call(self, features):
        x = self.dense(features)
        x = self.act(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x, mode="linear") + self.bias

        return x


@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top. """,
    ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class TFRobertaForMaskedLM(TFRobertaPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``tf.Tensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``tf.Tensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import RobertaTokenizer, TFRobertaForMaskedLM

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = TFRobertaForMaskedLM.from_pretrained('roberta-base')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        prediction_scores = outputs[0]

    """
    def __init__(self, config, *inputs, **kwargs):
        super(TFRobertaForMaskedLM, self).__init__(config, *inputs, **kwargs)

        self.roberta = TFRobertaMainLayer(config, name="roberta")
        self.lm_head = TFRobertaLMHead(config, self.roberta.embeddings, name="lm_head")

    def call(self, inputs, **kwargs):
        outputs = self.roberta(inputs, **kwargs)

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        return outputs  # prediction_scores, (hidden_states), (attentions)


class TFRobertaClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, **kwargs):
        super(TFRobertaClassificationHead, self).__init__(config, **kwargs)
        self.dense = tf.keras.layers.Dense(config.hidden_size,
                                           kernel_initializer=get_initializer(config.initializer_range),
                                           activation='tanh',
                                           name="dense")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.out_proj = tf.keras.layers.Dense(config.num_labels,
                                              kernel_initializer=get_initializer(config.initializer_range),
                                              name="out_proj")

    def call(self, features, training=False):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x, training=training)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        x = self.out_proj(x)
        return x


@add_start_docstrings("""RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer 
    on top of the pooled output) e.g. for GLUE tasks. """,
    ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class TFRobertaForSequenceClassification(TFRobertaPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **logits**: ``tf.Tensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

        tokenizer = RoertaTokenizer.from_pretrained('roberta-base')
        model = TFRobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        labels = tf.constant([1])[None, :]  # Batch size 1
        outputs = model(input_ids)
        logits = outputs[0]

    """
    def __init__(self, config, *inputs, **kwargs):
        super(TFRobertaForSequenceClassification, self).__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.roberta = TFRobertaMainLayer(config, name="roberta")
        self.classifier = TFRobertaClassificationHead(config, name="classifier")
    
    def call(self, inputs, **kwargs):
        outputs = self.roberta(inputs, **kwargs)

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, training=kwargs.get('training', False))

        outputs = (logits,) + outputs[2:]

        return outputs  # logits, (hidden_states), (attentions)
