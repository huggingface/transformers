# coding=utf-8
# Copyright 2018 DPR Authors, The Hugging Face Team.
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

""" TensorFlow DPR model for Open Domain Question Answering."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import tensorflow as tf

from ...modeling_tf_outputs import TFBaseModelOutputWithPooling
from ...modeling_tf_utils import TFPreTrainedModel, get_initializer, shape_list, unpack_inputs
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..bert.modeling_tf_bert import TFBertMainLayer
from .configuration_dpr import DPRConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DPRConfig"

TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-ctx_encoder-single-nq-base",
    "facebook/dpr-ctx_encoder-multiset-base",
]
TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-question_encoder-single-nq-base",
    "facebook/dpr-question_encoder-multiset-base",
]
TF_DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-reader-single-nq-base",
    "facebook/dpr-reader-multiset-base",
]


##########
# Outputs
##########


@dataclass
class TFDPRContextEncoderOutput(ModelOutput):
    r"""
    Class for outputs of [`TFDPRContextEncoder`].

    Args:
        pooler_output (`tf.Tensor` of shape `(batch_size, embeddings_size)`):
            The DPR encoder outputs the *pooler_output* that corresponds to the context representation. Last layer
            hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
            This output is to be used to embed contexts for nearest neighbors queries with questions embeddings.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    pooler_output: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFDPRQuestionEncoderOutput(ModelOutput):
    """
    Class for outputs of [`TFDPRQuestionEncoder`].

    Args:
        pooler_output (`tf.Tensor` of shape `(batch_size, embeddings_size)`):
            The DPR encoder outputs the *pooler_output* that corresponds to the question representation. Last layer
            hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
            This output is to be used to embed questions for nearest neighbors queries with context embeddings.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    pooler_output: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFDPRReaderOutput(ModelOutput):
    """
    Class for outputs of [`TFDPRReaderEncoder`].

    Args:
        start_logits (`tf.Tensor` of shape `(n_passages, sequence_length)`):
            Logits of the start index of the span for each passage.
        end_logits (`tf.Tensor` of shape `(n_passages, sequence_length)`):
            Logits of the end index of the span for each passage.
        relevance_logits (`tf.Tensor` of shape `(n_passages, )`):
            Outputs of the QA classifier of the DPRReader that corresponds to the scores of each passage to answer the
            question, compared to all the other passages.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    start_logits: tf.Tensor = None
    end_logits: tf.Tensor = None
    relevance_logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


class TFDPREncoderLayer(tf.keras.layers.Layer):

    base_model_prefix = "bert_model"

    def __init__(self, config: DPRConfig, **kwargs):
        super().__init__(**kwargs)

        # resolve name conflict with TFBertMainLayer instead of TFBertModel
        self.bert_model = TFBertMainLayer(config, add_pooling_layer=False, name="bert_model")
        self.config = config

        if self.config.hidden_size <= 0:
            raise ValueError("Encoder hidden_size can't be zero")
        self.projection_dim = config.projection_dim
        if self.projection_dim > 0:
            self.encode_proj = tf.keras.layers.Dense(
                config.projection_dim, kernel_initializer=get_initializer(config.initializer_range), name="encode_proj"
            )

    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor, ...]]:
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]
        if self.projection_dim > 0:
            pooled_output = self.encode_proj(pooled_output)

        if not return_dict:
            return (sequence_output, pooled_output) + outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def embeddings_size(self) -> int:
        if self.projection_dim > 0:
            return self.projection_dim
        return self.bert_model.config.hidden_size


class TFDPRSpanPredictorLayer(tf.keras.layers.Layer):

    base_model_prefix = "encoder"

    def __init__(self, config: DPRConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.encoder = TFDPREncoderLayer(config, name="encoder")

        self.qa_outputs = tf.keras.layers.Dense(
            2, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        self.qa_classifier = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="qa_classifier"
        )

    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        training: bool = False,
    ) -> Union[TFDPRReaderOutput, Tuple[tf.Tensor, ...]]:
        # notations: N - number of questions in a batch, M - number of passages per questions, L - sequence length
        n_passages, sequence_length = shape_list(input_ids) if input_ids is not None else shape_list(inputs_embeds)[:2]
        # feed encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = outputs[0]

        # compute logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        relevance_logits = self.qa_classifier(sequence_output[:, 0, :])

        # resize
        start_logits = tf.reshape(start_logits, [n_passages, sequence_length])
        end_logits = tf.reshape(end_logits, [n_passages, sequence_length])
        relevance_logits = tf.reshape(relevance_logits, [n_passages])

        if not return_dict:
            return (start_logits, end_logits, relevance_logits) + outputs[2:]

        return TFDPRReaderOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            relevance_logits=relevance_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TFDPRSpanPredictor(TFPreTrainedModel):

    base_model_prefix = "encoder"

    def __init__(self, config: DPRConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.encoder = TFDPRSpanPredictorLayer(config)

    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        training: bool = False,
    ) -> Union[TFDPRReaderOutput, Tuple[tf.Tensor, ...]]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs


class TFDPREncoder(TFPreTrainedModel):
    base_model_prefix = "encoder"

    def __init__(self, config: DPRConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.encoder = TFDPREncoderLayer(config)

    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        training: bool = False,
    ) -> Union[TFDPRReaderOutput, Tuple[tf.Tensor, ...]]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return outputs


##################
# PreTrainedModel
##################


class TFDPRPretrainedContextEncoder(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DPRConfig
    base_model_prefix = "ctx_encoder"


class TFDPRPretrainedQuestionEncoder(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DPRConfig
    base_model_prefix = "question_encoder"


class TFDPRPretrainedReader(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DPRConfig
    base_model_prefix = "reader"

    @tf.function(
        input_signature=[
            {
                "input_ids": tf.TensorSpec((None, None), tf.int64, name="input_ids"),
                "attention_mask": tf.TensorSpec((None, None), tf.int64, name="attention_mask"),
            }
        ]
    )
    def serving(self, inputs):
        output = self.call(inputs)

        return self.serving_output(output)


###############
# Actual Models
###############


TF_DPR_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Tensorflow [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
    subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to
    general usage and behavior.

    <Tip>

    TensorFlow models and layers in `transformers` accept two formats as input:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional argument.

    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    positional argument:

    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Parameters:
        config ([`DPRConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

TF_DPR_ENCODERS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. To match pretraining, DPR input sequence should be
            formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs (for a pair title+text for example):

            ```
            tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            ```

            (b) For single sequences (for a question for example):

            ```
            tokens:         [CLS] the dog is hairy . [SEP]
            token_type_ids:   0   0   0   0  0     0   0
            ```

            DPR is a model with absolute position embeddings so it's usually advised to pad the inputs on the right
            rather than the left.

            Indices can be obtained using [`DPRTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        inputs_embeds (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""

TF_DPR_READER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Numpy array` or `tf.Tensor` of shapes `(n_passages, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. It has to be a sequence triplet with 1) the question
            and 2) the passages titles and 3) the passages texts To match pretraining, DPR `input_ids` sequence should
            be formatted with [CLS] and [SEP] with the format:

                `[CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>`

            DPR is a model with absolute position embeddings so it's usually advised to pad the inputs on the right
            rather than the left.

            Indices can be obtained using [`DPRReaderTokenizer`]. See this class documentation for more details.
        attention_mask (`Numpy array` or `tf.Tensor` of shape `(n_passages, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        inputs_embeds (`Numpy array` or `tf.Tensor` of shape `(n_passages, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare DPRContextEncoder transformer outputting pooler outputs as context representations.",
    TF_DPR_START_DOCSTRING,
)
class TFDPRContextEncoder(TFDPRPretrainedContextEncoder):
    def __init__(self, config: DPRConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.ctx_encoder = TFDPREncoderLayer(config, name="ctx_encoder")

    def get_input_embeddings(self):
        try:
            return self.ctx_encoder.bert_model.get_input_embeddings()
        except AttributeError:
            self(self.dummy_inputs)
            return self.ctx_encoder.bert_model.get_input_embeddings()

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TF_DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFDPRContextEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids=None,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training: bool = False,
    ) -> Union[TFDPRContextEncoderOutput, Tuple[tf.Tensor, ...]]:
        r"""
        Return:

        Examples:

        ```python
        >>> from transformers import TFDPRContextEncoder, DPRContextEncoderTokenizer

        >>> tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        >>> model = TFDPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", from_pt=True)
        >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="tf")["input_ids"]
        >>> embeddings = model(input_ids).pooler_output
        ```
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = (
                tf.ones(input_shape, dtype=tf.dtypes.int32)
                if input_ids is None
                else (input_ids != self.config.pad_token_id)
            )
        if token_type_ids is None:
            token_type_ids = tf.zeros(input_shape, dtype=tf.dtypes.int32)

        outputs = self.ctx_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        if not return_dict:
            return outputs[1:]

        return TFDPRContextEncoderOutput(
            pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFDPRContextEncoderOutput(pooler_output=output.pooler_output, hidden_states=hs, attentions=attns)


@add_start_docstrings(
    "The bare DPRQuestionEncoder transformer outputting pooler outputs as question representations.",
    TF_DPR_START_DOCSTRING,
)
class TFDPRQuestionEncoder(TFDPRPretrainedQuestionEncoder):
    def __init__(self, config: DPRConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.question_encoder = TFDPREncoderLayer(config, name="question_encoder")

    def get_input_embeddings(self):
        try:
            return self.question_encoder.bert_model.get_input_embeddings()
        except AttributeError:
            self(self.dummy_inputs)
            return self.question_encoder.bert_model.get_input_embeddings()

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TF_DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFDPRQuestionEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids=None,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training: bool = False,
    ) -> Union[TFDPRQuestionEncoderOutput, Tuple[tf.Tensor, ...]]:
        r"""
        Return:

        Examples:

        ```python
        >>> from transformers import TFDPRQuestionEncoder, DPRQuestionEncoderTokenizer

        >>> tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        >>> model = TFDPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base", from_pt=True)
        >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="tf")["input_ids"]
        >>> embeddings = model(input_ids).pooler_output
        ```
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = (
                tf.ones(input_shape, dtype=tf.dtypes.int32)
                if input_ids is None
                else (input_ids != self.config.pad_token_id)
            )
        if token_type_ids is None:
            token_type_ids = tf.zeros(input_shape, dtype=tf.dtypes.int32)

        outputs = self.question_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        if not return_dict:
            return outputs[1:]
        return TFDPRQuestionEncoderOutput(
            pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFDPRQuestionEncoderOutput(pooler_output=output.pooler_output, hidden_states=hs, attentions=attns)


@add_start_docstrings(
    "The bare DPRReader transformer outputting span predictions.",
    TF_DPR_START_DOCSTRING,
)
class TFDPRReader(TFDPRPretrainedReader):
    def __init__(self, config: DPRConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.span_predictor = TFDPRSpanPredictorLayer(config, name="span_predictor")

    def get_input_embeddings(self):
        try:
            return self.span_predictor.encoder.bert_model.get_input_embeddings()
        except AttributeError:
            self(self.dummy_inputs)
            return self.span_predictor.encoder.bert_model.get_input_embeddings()

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TF_DPR_READER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFDPRReaderOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids=None,
        attention_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict=None,
        training: bool = False,
    ) -> Union[TFDPRReaderOutput, Tuple[tf.Tensor, ...]]:
        r"""
        Return:

        Examples:

        ```python
        >>> from transformers import TFDPRReader, DPRReaderTokenizer

        >>> tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
        >>> model = TFDPRReader.from_pretrained("facebook/dpr-reader-single-nq-base", from_pt=True)
        >>> encoded_inputs = tokenizer(
        ...     questions=["What is love ?"],
        ...     titles=["Haddaway"],
        ...     texts=["'What Is Love' is a song recorded by the artist Haddaway"],
        ...     return_tensors="tf",
        ... )
        >>> outputs = model(encoded_inputs)
        >>> start_logits = outputs.start_logits
        >>> end_logits = outputs.end_logits
        >>> relevance_logits = outputs.relevance_logits
        ```
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = tf.ones(input_shape, dtype=tf.dtypes.int32)

        return self.span_predictor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFDPRReaderOutput(
            start_logits=output.start_logits,
            end_logits=output.end_logits,
            relevance_logits=output.relevance_logits,
            hidden_states=hs,
            attentions=attns,
        )
