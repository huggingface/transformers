# Copyright 2021 The HuggingFace Team. All rights reserved.
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
from typing import Optional

import flax
import jax.numpy as jnp

from .utils import ModelOutput


@flax.struct.dataclass
class FlaxBaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: Optional[jnp.ndarray] = None
    hidden_states: Optional[tuple[jnp.ndarray]] = None
    attentions: Optional[tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxBaseModelOutputWithNoAttention(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings, if the model has an embedding layer, + one
            for the output of each layer) of shape `(batch_size, num_channels, height, width)`. Hidden-states of the
            model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[jnp.ndarray] = None
    hidden_states: Optional[tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxBaseModelOutputWithPoolingAndNoAttention(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state after a pooling operation on the spatial dimensions.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings, if the model has an embedding layer, + one
            for the output of each layer) of shape `(batch_size, num_channels, height, width)`. Hidden-states of the
            model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[jnp.ndarray] = None
    pooler_output: Optional[jnp.ndarray] = None
    hidden_states: Optional[tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxImageClassifierOutputWithNoAttention(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when
        `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings, if the model has an embedding layer, + one
            for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
            called feature maps) of the model at the output of each stage.
    """

    logits: Optional[jnp.ndarray] = None
    hidden_states: Optional[tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxBaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`dict[str, jnp.ndarray]`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: Optional[jnp.ndarray] = None
    past_key_values: Optional[dict[str, jnp.ndarray]] = None
    hidden_states: Optional[tuple[jnp.ndarray]] = None
    attentions: Optional[tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxBaseModelOutputWithPooling(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
            prediction (classification) objective during pretraining.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: Optional[jnp.ndarray] = None
    pooler_output: Optional[jnp.ndarray] = None
    hidden_states: Optional[tuple[jnp.ndarray]] = None
    attentions: Optional[tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxBaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings, if the model has an embedding layer, + one
            for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        past_key_values (`tuple(tuple(jnp.ndarray))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(jnp.ndarray)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
    """

    last_hidden_state: Optional[jnp.ndarray] = None
    pooler_output: Optional[jnp.ndarray] = None
    hidden_states: Optional[tuple[jnp.ndarray]] = None
    past_key_values: Optional[tuple[tuple[jnp.ndarray]]] = None
    attentions: Optional[tuple[jnp.ndarray]] = None
    cross_attentions: Optional[tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxBaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(jnp.ndarray))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(jnp.ndarray)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: Optional[jnp.ndarray] = None
    past_key_values: Optional[tuple[tuple[jnp.ndarray]]] = None
    hidden_states: Optional[tuple[jnp.ndarray]] = None
    attentions: Optional[tuple[jnp.ndarray]] = None
    cross_attentions: Optional[tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxSeq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(jnp.ndarray))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(jnp.ndarray)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    last_hidden_state: Optional[jnp.ndarray] = None
    past_key_values: Optional[tuple[tuple[jnp.ndarray]]] = None
    decoder_hidden_states: Optional[tuple[jnp.ndarray]] = None
    decoder_attentions: Optional[tuple[jnp.ndarray]] = None
    cross_attentions: Optional[tuple[jnp.ndarray]] = None
    encoder_last_hidden_state: Optional[jnp.ndarray] = None
    encoder_hidden_states: Optional[tuple[jnp.ndarray]] = None
    encoder_attentions: Optional[tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxCausalLMOutputWithCrossAttentions(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (`tuple(tuple(jnp.ndarray))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `jnp.ndarray` tuples of length `config.n_layers`, with each tuple containing the cached key, value
            states of the self-attention and the cross-attention layers if model is used in encoder-decoder setting.
            Only relevant if `config.is_decoder = True`.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
    """

    logits: Optional[jnp.ndarray] = None
    past_key_values: Optional[tuple[tuple[jnp.ndarray]]] = None
    hidden_states: Optional[tuple[jnp.ndarray]] = None
    attentions: Optional[tuple[jnp.ndarray]] = None
    cross_attentions: Optional[tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxMaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: Optional[jnp.ndarray] = None
    hidden_states: Optional[tuple[jnp.ndarray]] = None
    attentions: Optional[tuple[jnp.ndarray]] = None


FlaxCausalLMOutput = FlaxMaskedLMOutput


@flax.struct.dataclass
class FlaxSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(jnp.ndarray))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(jnp.ndarray)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    logits: Optional[jnp.ndarray] = None
    past_key_values: Optional[tuple[tuple[jnp.ndarray]]] = None
    decoder_hidden_states: Optional[tuple[jnp.ndarray]] = None
    decoder_attentions: Optional[tuple[jnp.ndarray]] = None
    cross_attentions: Optional[tuple[jnp.ndarray]] = None
    encoder_last_hidden_state: Optional[jnp.ndarray] = None
    encoder_hidden_states: Optional[tuple[jnp.ndarray]] = None
    encoder_attentions: Optional[tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxNextSentencePredictorOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: Optional[jnp.ndarray] = None
    hidden_states: Optional[tuple[jnp.ndarray]] = None
    attentions: Optional[tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: Optional[jnp.ndarray] = None
    hidden_states: Optional[tuple[jnp.ndarray]] = None
    attentions: Optional[tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxSeq2SeqSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence sentence classification models.

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        past_key_values (`tuple(tuple(jnp.ndarray))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(jnp.ndarray)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    logits: Optional[jnp.ndarray] = None
    past_key_values: Optional[tuple[tuple[jnp.ndarray]]] = None
    decoder_hidden_states: Optional[tuple[jnp.ndarray]] = None
    decoder_attentions: Optional[tuple[jnp.ndarray]] = None
    cross_attentions: Optional[tuple[jnp.ndarray]] = None
    encoder_last_hidden_state: Optional[jnp.ndarray] = None
    encoder_hidden_states: Optional[tuple[jnp.ndarray]] = None
    encoder_attentions: Optional[tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxMultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice models.

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, num_choices)`):
            *num_choices* is the second dimension of the input tensors. (see *input_ids* above).

            Classification scores (before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: Optional[jnp.ndarray] = None
    hidden_states: Optional[tuple[jnp.ndarray]] = None
    attentions: Optional[tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxTokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: Optional[jnp.ndarray] = None
    hidden_states: Optional[tuple[jnp.ndarray]] = None
    attentions: Optional[tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        start_logits (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    start_logits: Optional[jnp.ndarray] = None
    end_logits: Optional[jnp.ndarray] = None
    hidden_states: Optional[tuple[jnp.ndarray]] = None
    attentions: Optional[tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxSeq2SeqQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence question answering models.

    Args:
        start_logits (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        past_key_values (`tuple(tuple(jnp.ndarray))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(jnp.ndarray)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    start_logits: Optional[jnp.ndarray] = None
    end_logits: Optional[jnp.ndarray] = None
    past_key_values: Optional[tuple[tuple[jnp.ndarray]]] = None
    decoder_hidden_states: Optional[tuple[jnp.ndarray]] = None
    decoder_attentions: Optional[tuple[jnp.ndarray]] = None
    cross_attentions: Optional[tuple[jnp.ndarray]] = None
    encoder_last_hidden_state: Optional[jnp.ndarray] = None
    encoder_hidden_states: Optional[tuple[jnp.ndarray]] = None
    encoder_attentions: Optional[tuple[jnp.ndarray]] = None
