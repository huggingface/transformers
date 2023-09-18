# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import keras_core as keras

from .utils import ModelOutput


@dataclass
class KerasBaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`keras.KerasVariable` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: keras.KerasVariable = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasBaseModelOutputWithNoAttention(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`keras.KerasVariable` shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, num_channels, height, width)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: keras.KerasVariable = None
    hidden_states: Optional[Tuple[keras.KerasVariable, ...]] = None


@dataclass
class KerasBaseModelOutputWithPooling(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`keras.KerasVariable` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`keras.KerasVariable` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
            prediction (classification) objective during pretraining.

            This output is usually *not* a good summary of the semantic content of the input, you're often better with
            averaging or pooling the sequence of hidden-states for the whole input sequence.
        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: keras.KerasVariable = None
    pooler_output: keras.KerasVariable = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasBaseModelOutputWithPoolingAndNoAttention(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`keras.KerasVariable` of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`keras.KerasVariable` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state after a pooling operation on the spatial dimensions.
        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, num_channels, height, width)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: keras.KerasVariable = None
    pooler_output: keras.KerasVariable = None
    hidden_states: Optional[Tuple[keras.KerasVariable, ...]] = None


@dataclass
class KerasBaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`keras.KerasVariable` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`keras.KerasVariable` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
            prediction (classification) objective during pretraining.

            This output is usually *not* a good summary of the semantic content of the input, you're often better with
            averaging or pooling the sequence of hidden-states for the whole input sequence.
        past_key_values (`List[keras.KerasVariable]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `keras.KerasVariable` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: keras.KerasVariable = None
    pooler_output: keras.KerasVariable = None
    past_key_values: List[keras.KerasVariable] | None = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None
    cross_attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasBaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`keras.KerasVariable` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`List[keras.KerasVariable]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `keras.KerasVariable` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: keras.KerasVariable = None
    past_key_values: List[keras.KerasVariable] | None = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasBaseModelOutputWithCrossAttentions(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`keras.KerasVariable` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: keras.KerasVariable = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None
    cross_attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasBaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`keras.KerasVariable` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`List[keras.KerasVariable]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `keras.KerasVariable` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(tf.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: keras.KerasVariable = None
    past_key_values: List[keras.KerasVariable] | None = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None
    cross_attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasSeq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    Args:
        last_hidden_state (`keras.KerasVariable` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`List[keras.KerasVariable]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `keras.KerasVariable` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`keras.KerasVariable` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    last_hidden_state: keras.KerasVariable = None
    past_key_values: List[keras.KerasVariable] | None = None
    decoder_hidden_states: Tuple[keras.KerasVariable] | None = None
    decoder_attentions: Tuple[keras.KerasVariable] | None = None
    cross_attentions: Tuple[keras.KerasVariable] | None = None
    encoder_last_hidden_state: keras.KerasVariable | None = None
    encoder_hidden_states: Tuple[keras.KerasVariable] | None = None
    encoder_attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`keras.KerasVariable` of shape `(n,)`, *optional*, where n is the number of non-masked labels, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`keras.KerasVariable` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: keras.KerasVariable | None = None
    logits: keras.KerasVariable = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`keras.KerasVariable` of shape `(n,)`, *optional*, where n is the number of non-masked labels, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`keras.KerasVariable` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`List[keras.KerasVariable]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `keras.KerasVariable` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: keras.KerasVariable | None = None
    logits: keras.KerasVariable = None
    past_key_values: List[keras.KerasVariable] | None = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasCausalLMOutputWithCrossAttentions(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`keras.KerasVariable` of shape `(n,)`, *optional*, where n is the number of non-masked labels, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`keras.KerasVariable` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        past_key_values (`List[keras.KerasVariable]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `keras.KerasVariable` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
    """

    loss: keras.KerasVariable | None = None
    logits: keras.KerasVariable = None
    past_key_values: List[keras.KerasVariable] | None = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None
    cross_attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasMaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (`keras.KerasVariable` of shape `(n,)`, *optional*, where n is the number of non-masked labels, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`keras.KerasVariable` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: keras.KerasVariable | None = None
    logits: keras.KerasVariable = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (`keras.KerasVariable` of shape `(n,)`, *optional*, where n is the number of non-masked labels, returned when `labels` is provided):
            Language modeling loss.
        logits (`keras.KerasVariable` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`List[keras.KerasVariable]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `keras.KerasVariable` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`keras.KerasVariable` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: keras.KerasVariable | None = None
    logits: keras.KerasVariable = None
    past_key_values: List[keras.KerasVariable] | None = None
    decoder_hidden_states: Tuple[keras.KerasVariable] | None = None
    decoder_attentions: Tuple[keras.KerasVariable] | None = None
    cross_attentions: Tuple[keras.KerasVariable] | None = None
    encoder_last_hidden_state: keras.KerasVariable | None = None
    encoder_hidden_states: Tuple[keras.KerasVariable] | None = None
    encoder_attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasNextSentencePredictorOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        loss (`keras.KerasVariable` of shape `(n,)`, *optional*, where n is the number of non-masked labels, returned when `next_sentence_label` is provided):
            Next sentence prediction loss.
        logits (`keras.KerasVariable` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: keras.KerasVariable | None = None
    logits: keras.KerasVariable = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`keras.KerasVariable` of shape `(batch_size, )`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`keras.KerasVariable` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: keras.KerasVariable | None = None
    logits: keras.KerasVariable = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasSeq2SeqSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence sentence classification models.

    Args:
        loss (`keras.KerasVariable` of shape `(1,)`, *optional*, returned when `label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`keras.KerasVariable` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        past_key_values (`List[keras.KerasVariable]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `keras.KerasVariable` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`
        encoder_last_hidden_state (`keras.KerasVariable` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: keras.KerasVariable | None = None
    logits: keras.KerasVariable = None
    past_key_values: List[keras.KerasVariable] | None = None
    decoder_hidden_states: Tuple[keras.KerasVariable] | None = None
    decoder_attentions: Tuple[keras.KerasVariable] | None = None
    cross_attentions: Tuple[keras.KerasVariable] | None = None
    encoder_last_hidden_state: keras.KerasVariable | None = None
    encoder_hidden_states: Tuple[keras.KerasVariable] | None = None
    encoder_attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasSemanticSegmenterOutput(ModelOutput):
    """
    Base class for outputs of semantic segmentation models.

    Args:
        loss (`keras.KerasVariable` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`keras.KerasVariable` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            Classification scores for each pixel.

            <Tip warning={true}>

            The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
            to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
            original image size as post-processing. You should always check your logits shape and resize as needed.

            </Tip>

        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, patch_size, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: keras.KerasVariable | None = None
    logits: keras.KerasVariable = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasSemanticSegmenterOutputWithNoAttention(ModelOutput):
    """
    Base class for outputs of semantic segmentation models that do not output attention scores.

    Args:
        loss (`keras.KerasVariable` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`keras.KerasVariable` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            Classification scores for each pixel.

            <Tip warning={true}>

            The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
            to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
            original image size as post-processing. You should always check your logits shape and resize as needed.

            </Tip>

        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, patch_size, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: keras.KerasVariable | None = None
    logits: keras.KerasVariable = None
    hidden_states: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasImageClassifierOutput(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        loss (`keras.KerasVariable` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`keras.KerasVariable` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called
            feature maps) of the model at the output of each stage.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: keras.KerasVariable | None = None
    logits: keras.KerasVariable = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasMultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice models.

    Args:
        loss (`keras.KerasVariable` of shape *(batch_size, )*, *optional*, returned when `labels` is provided):
            Classification loss.
        logits (`keras.KerasVariable` of shape `(batch_size, num_choices)`):
            *num_choices* is the second dimension of the input tensors. (see *input_ids* above).

            Classification scores (before SoftMax).
        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: keras.KerasVariable | None = None
    logits: keras.KerasVariable = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasTokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`keras.KerasVariable` of shape `(n,)`, *optional*, where n is the number of unmasked labels, returned when `labels` is provided) :
            Classification loss.
        logits (`keras.KerasVariable` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: keras.KerasVariable | None = None
    logits: keras.KerasVariable = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        loss (`keras.KerasVariable` of shape `(batch_size, )`, *optional*, returned when `start_positions` and `end_positions` are provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`keras.KerasVariable` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`keras.KerasVariable` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: keras.KerasVariable | None = None
    start_logits: keras.KerasVariable = None
    end_logits: keras.KerasVariable = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasSeq2SeqQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence question answering models.

    Args:
        loss (`keras.KerasVariable` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`keras.KerasVariable` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`keras.KerasVariable` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        past_key_values (`List[keras.KerasVariable]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `keras.KerasVariable` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_last_hidden_state (`keras.KerasVariable` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: keras.KerasVariable | None = None
    start_logits: keras.KerasVariable = None
    end_logits: keras.KerasVariable = None
    past_key_values: List[keras.KerasVariable] | None = None
    decoder_hidden_states: Tuple[keras.KerasVariable] | None = None
    decoder_attentions: Tuple[keras.KerasVariable] | None = None
    encoder_last_hidden_state: keras.KerasVariable | None = None
    encoder_hidden_states: Tuple[keras.KerasVariable] | None = None
    encoder_attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasSequenceClassifierOutputWithPast(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`keras.KerasVariable` of shape `(batch_size, )`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`keras.KerasVariable` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        past_key_values (`List[keras.KerasVariable]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `keras.KerasVariable` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: keras.KerasVariable | None = None
    logits: keras.KerasVariable = None
    past_key_values: List[keras.KerasVariable] | None = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None


@dataclass
class KerasImageClassifierOutputWithNoAttention(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        loss (`keras.KerasVariable` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`keras.KerasVariable` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also called
            feature maps) of the model at the output of each stage.
    """

    loss: keras.KerasVariable | None = None
    logits: keras.KerasVariable = None
    hidden_states: Optional[Tuple[keras.KerasVariable, ...]] = None


@dataclass
class KerasMaskedImageModelingOutput(ModelOutput):
    """
    Base class for outputs of masked image completion / in-painting models.

    Args:
        loss (`keras.KerasVariable` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
            Reconstruction loss.
        reconstruction (`keras.KerasVariable` of shape `(batch_size, num_channels, height, width)`):
           Reconstructed / completed images.
        hidden_states (`tuple(keras.KerasVariable)`, *optional*, returned when `output_hidden_states=True` is passed or when
        `config.output_hidden_states=True`):
            Tuple of `keras.KerasVariable` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called
            feature maps) of the model at the output of each stage.
        attentions (`tuple(keras.KerasVariable)`, *optional*, returned when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `keras.KerasVariable` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: keras.KerasVariable | None = None
    reconstruction: keras.KerasVariable = None
    hidden_states: Tuple[keras.KerasVariable] | None = None
    attentions: Tuple[keras.KerasVariable] | None = None

    @property
    def logits(self):
        warnings.warn(
            "logits attribute is deprecated and will be removed in version 5 of Transformers."
            " Please use the reconstruction attribute to retrieve the final output instead.",
            FutureWarning,
        )
        return self.reconstruction
