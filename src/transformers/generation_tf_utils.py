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

import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from .generation_tf_logits_process import (
    TFForcedBOSTokenLogitsProcessor,
    TFForcedEOSTokenLogitsProcessor,
    TFLogitsProcessorList,
    TFMinLengthLogitsProcessor,
    TFNoBadWordsLogitsProcessor,
    TFNoRepeatNGramLogitsProcessor,
    TFRepetitionPenaltyLogitsProcessor,
    TFTemperatureLogitsWarper,
    TFTopKLogitsWarper,
    TFTopPLogitsWarper,
)
from .tf_utils import shape_list, stable_softmax
from .utils import ModelOutput, logging


logger = logging.get_logger(__name__)


@dataclass
class TFGreedySearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. `(max_length-input_ids.shape[-1],)`-shaped tuple of `tf.Tensor` with each tensor
            of shape `(batch_size, config.vocab_size)`).
        attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    scores: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFGreedySearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using greedy search. Hidden states and attention
    weights of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the
    encoder_hidden_states attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)


    Args:
        sequences (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. `(max_length-1,)`-shaped tuple of `tf.Tensor` with each tensor of shape
            `(batch_size, config.vocab_size)`).
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer of the decoder) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        cross_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    scores: Optional[Tuple[tf.Tensor]] = None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFSampleDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using sampling.


    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. `(max_length-input_ids.shape[-1],)`-shaped tuple of `tf.Tensor` with each tensor
            of shape `(batch_size*num_return_sequences, config.vocab_size)`).
        attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(num_return_sequences*batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(num_return_sequences*batch_size, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    scores: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFSampleEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using sampling. Hidden states and attention weights of
    the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)


    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. `(max_length-1,)`-shaped tuple of `tf.Tensor` with each tensor of shape
            `(batch_size*num_return_sequences, config.vocab_size)`).
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer of the decoder) of shape `(batch_size*num_return_sequences,
            num_heads, sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_return_sequences, num_heads, generated_length, sequence_length)`.
        cross_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_return_sequences, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    scores: Optional[Tuple[tf.Tensor]] = None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFBeamSearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using beam search.

    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`tf.Tensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . `(max_length-input_ids.shape[-1],)`-shaped tuple of `tf.Tensor` with each tensor of shape
            `(batch_size*num_beams*num_return_sequences, config.vocab_size)`).
        attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    sequences_scores: Optional[tf.Tensor] = None
    scores: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFBeamSearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using beam search. Hidden states and attention weights
    of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)

    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`tf.Tensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . `(max_length-1,)`-shaped tuple of `tf.Tensor` with each tensor of shape `(batch_size*num_beams,
            config.vocab_size)`).
        attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer of the decoder) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams*num_return_sequences, num_heads, generated_length,
            sequence_length)`.
        cross_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    sequences_scores: Optional[tf.Tensor] = None
    scores: Optional[Tuple[tf.Tensor]] = None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFBeamSampleDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using beam sample.

    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`tf.Tensor` of shape `(batch_size * num_return_sequence)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . `(max_length-input_ids.shape[-1],)`-shaped tuple of `tf.Tensor` with each tensor of shape
            `(batch_size*num_beams*num_return_sequences, config.vocab_size)`).
        attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    sequences_scores: Optional[tf.Tensor] = None
    scores: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFBeamSampleEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using beam sampling. Hidden states and attention
    weights of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the
    encoder_hidden_states attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)

    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_beams, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`tf.Tensor` of shape `(batch_size * num_return_sequence)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . `(max_length-1,)`-shaped tuple of `tf.Tensor` with each tensor of shape `(batch_size*num_beams,
            config.vocab_size)`).
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer of the decoder) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size*num_beams, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
        cross_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    sequences_scores: Optional[tf.Tensor] = None
    scores: Optional[Tuple[tf.Tensor]] = None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


TFGreedySearchOutput = Union[TFGreedySearchEncoderDecoderOutput, TFGreedySearchDecoderOnlyOutput]
TFSampleOutput = Union[TFSampleEncoderDecoderOutput, TFSampleDecoderOnlyOutput]
TFBeamSearchOutput = Union[TFBeamSearchEncoderDecoderOutput, TFBeamSearchDecoderOnlyOutput]
TFBeamSampleOutput = Union[TFBeamSampleEncoderDecoderOutput, TFBeamSampleDecoderOnlyOutput]


class TFGenerationMixin:
    """
    A class containing all of the functions supporting generation, to be used as a mixin in [`TFPreTrainedModel`].
    """

    seed_generator = tf.random.Generator.from_non_deterministic_state()

    def prepare_inputs_for_generation(self, inputs, **kwargs):
        """
        Implement in subclasses of [`TFPreTrainedModel`] for custom behavior to prepare inputs in the generate method.
        """
        return {"input_ids": inputs}

    def _use_cache(self, outputs, use_cache):
        """During generation, decide whether to pass the `past` variable to the next forward pass."""
        use_cache = getattr(self.config, "use_cache", False)
        if len(outputs) <= 1 or use_cache is False:
            return False
        if hasattr(self.config, "mem_len") and self.config.mem_len == 0:
            return False
        return True

    def generate(
        self,
        input_ids=None,
        max_length=None,
        min_length=None,
        do_sample=None,
        early_stopping=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bad_words_ids=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        num_return_sequences=None,
        attention_mask=None,
        decoder_start_token_id=None,
        use_cache=None,
        output_scores=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict_in_generate=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        **model_kwargs,
    ) -> Union[TFGreedySearchOutput, TFSampleOutput, TFBeamSearchOutput, TFBeamSampleOutput, tf.Tensor]:
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

        Adapted in part from [Facebook's XLM beam search
        code](https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529).

        Apart from `input_ids` and `attention_mask`, all the arguments below will default to the value of the attribute
        of the same name inside the [`PretrainedConfig`] of the model. The default values indicated are the default
        values of those config.

        Most of these parameters are explained in more detail in [this blog
        post](https://huggingface.co/blog/how-to-generate).

        Parameters:

            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, `(batch_size, sequence_length,
            feature_dim)` or `(batch_size, num_channels, height, width)`, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
            min_length (`int`, *optional*, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (`bool`, *optional*, defaults to `False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (`bool`, *optional*, defaults to `False`):
                Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (`float`, *optional*, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (`int`, *optional*, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional*, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher
                are kept for generation.
            repetition_penalty (`float`, *optional*, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            bos_token_id (`int`, *optional*):
                The id of the *beginning-of-sequence* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            length_penalty (`float`, *optional*, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty.

                Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
                order to encourage the model to produce longer sequences.
            no_repeat_ngram_size (`int`, *optional*, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            bad_words_ids(`List[int]`, *optional*):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use `tokenizer.encode(bad_word, add_prefix_space=True)`.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            attention_mask (`tf.Tensor` of `dtype=tf.int32` and shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values are in `[0, 1]`, 1 for tokens
                that are not masked, and 0 for masked tokens.

                If not provided, will default to a tensor the same shape as `input_ids` that masks the pad token.

                [What are attention masks?](../glossary#attention-mask)
            decoder_start_token_id (`int`, *optional*):
                If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
            use_cache: (`bool`, *optional*, defaults to `True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            forced_bos_token_id (`int`, *optional*):
                The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful
                for multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be
                the target language token.
            forced_eos_token_id (`int`, *optional*):
                The id of the token to force as the last generated token when `max_length` is reached.
            model_specific_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model.

        Return:
            [`~utils.ModelOutput`] or `tf.Tensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True` or when
            `config.return_dict_in_generate=True`) or a `tf.Tensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation_tf_utils.TFGreedySearchDecoderOnlyOutput`],
                    - [`~generation_tf_utils.TFSampleDecoderOnlyOutput`],
                    - [`~generation_tf_utils.TFBeamSearchDecoderOnlyOutput`],
                    - [`~generation_tf_utils.TFBeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation_tf_utils.TFGreedySearchEncoderDecoderOutput`],
                    - [`~generation_tf_utils.TFSampleEncoderDecoderOutput`],
                    - [`~generation_tf_utils.TFBeamSearchEncoderDecoderOutput`],
                    - [`~generation_tf_utils.TFBeamSampleEncoderDecoderOutput`]

        Examples:

        ```python
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")  # Initialize tokenizer
        model = TFAutoModelWithLMHead.from_pretrained(
            "distilgpt2"
        )  # Download model and configuration from huggingface.co and cache.
        outputs = model.generate(max_length=40)  # do greedy decoding
        print(f"Generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

        tokenizer = AutoTokenizer.from_pretrained("openai-gpt")  # Initialize tokenizer
        model = TFAutoModelWithLMHead.from_pretrained(
            "openai-gpt"
        )  # Download model and configuration from huggingface.co and cache.
        input_context = "The dog"
        input_ids = tokenizer.encode(input_context, return_tensors="tf")  # encode input context
        outputs = model.generate(
            input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5
        )  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
        for i in range(3):  #  3 output sequences were generated
            print(f"Generated {i}: {tokenizer.decode(outputs[i], skip_special_tokens=True)}")

        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")  # Initialize tokenizer
        model = TFAutoModelWithLMHead.from_pretrained(
            "distilgpt2"
        )  # Download model and configuration from huggingface.co and cache.
        input_context = "The dog"
        input_ids = tokenizer.encode(input_context, return_tensors="tf")  # encode input context
        outputs = model.generate(
            input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3, do_sample=True
        )  # generate 3 candidates using sampling
        for i in range(3):  #  3 output sequences were generated
            print(f"Generated {i}: {tokenizer.decode(outputs[i], skip_special_tokens=True)}")

        tokenizer = AutoTokenizer.from_pretrained("ctrl")  # Initialize tokenizer
        model = TFAutoModelWithLMHead.from_pretrained(
            "ctrl"
        )  # Download model and configuration from huggingface.co and cache.
        input_context = "Legal My neighbor is"  # "Legal" is one of the control codes for ctrl
        input_ids = tokenizer.encode(input_context, return_tensors="tf")  # encode input context
        outputs = model.generate(
            input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2
        )  # generate sequences
        print(f"Generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Initialize tokenizer
        model = TFAutoModelWithLMHead.from_pretrained(
            "gpt2"
        )  # Download model and configuration from huggingface.co and cache.
        input_context = "My cute dog"
        bad_words_ids = [
            tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ["idiot", "stupid", "shut up"]
        ]
        input_ids = tokenizer.encode(input_context, return_tensors="tf")  # encode input context
        outputs = model.generate(
            input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids
        )  # generate sequences without allowing bad_words to be generated
        ```"""
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        do_sample = do_sample if do_sample is not None else self.config.do_sample

        if do_sample is False or num_beams == 1:
            return self._generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                num_return_sequences=num_return_sequences,
                attention_mask=attention_mask,
                decoder_start_token_id=decoder_start_token_id,
                use_cache=use_cache,
                seed=model_kwargs.pop("seed", None),
                output_scores=output_scores,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=return_dict_in_generate,
                forced_bos_token_id=forced_bos_token_id,
                forced_eos_token_id=forced_eos_token_id,
            )

        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head. "
                "Please use another model class (e.g. `TFOpenAIGPTLMHeadModel`, `TFXLNetLMHeadModel`, `TFGPT2LMHeadModel`, `TFCTRLLMHeadModel`, `TFT5ForConditionalGeneration`, `TFTransfoXLLMHeadModel`)"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p

        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )
        forced_bos_token_id = (
            forced_bos_token_id if forced_bos_token_id is not None else self.config.forced_bos_token_id
        )
        forced_eos_token_id = (
            forced_eos_token_id if forced_eos_token_id is not None else self.config.forced_eos_token_id
        )

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        model_kwargs["output_scores"] = output_scores
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        if self.config.is_encoder_decoder:
            model_kwargs["encoder_attentions"] = None
            model_kwargs["encoder_hidden_states"] = None

        if input_ids is not None:
            batch_size = shape_list(input_ids)[0]  # overridden by the input batch_size
        else:
            batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        # This block corresponds to the following line in `generation_tf_utils`:
        #   "input_ids = self._prepare_input_ids_for_generation(bos_token_id, model_kwargs.get("encoder_outputs"))"
        # with the following differences:
        #   1. In PT, `generate()`'s `model_kwargs` can accept `encoder_outputs`, but not the case in TF.
        #   2. There is no shape checking in PT.
        # In both PT/TF, if `input_ids` is `None`, we try to create it as it is for a text model.
        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = tf.fill((batch_size, 1), bos_token_id)

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.call).parameters.keys())
        if accepts_attention_mask:
            if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids.numpy()):
                attention_mask = tf.cast(tf.math.not_equal(input_ids, pad_token_id), dtype=tf.int32)
            elif attention_mask is None:
                attention_mask = tf.ones(shape_list(input_ids)[:2], dtype=tf.int32)

        if pad_token_id is None and eos_token_id is not None:
            logger.warning(f"Setting `pad_token_id` to {eos_token_id} (first `eos_token_id`) to generate sequence")
            pad_token_id = eos_token_id

        # current position and vocab size
        cur_len = shape_list(input_ids)[1]  # unused
        vocab_size = getattr(self.config, "vocab_size", None)
        if vocab_size is None and self.config.is_encoder_decoder:
            decoder_config = getattr(self.config, "decoder", None)
            if decoder_config is not None:
                vocab_size = getattr(self.config.decoder, "vocab_size", None)

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                decoder_start_token_id = bos_token_id

            assert (
                decoder_start_token_id is not None
            ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
            assert hasattr(self, "get_encoder"), f"{self} should have a 'get_encoder' function defined"
            assert callable(self.get_encoder), f"{self.get_encoder} should be a method"

            # get encoder and store encoder outputs
            encoder = self.get_encoder()

            encoder_kwargs = {
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict_in_generate,
            }
            if accepts_attention_mask:
                encoder_kwargs["attention_mask"] = attention_mask

            encoder_outputs = encoder(input_ids, **encoder_kwargs)
            if return_dict_in_generate:
                if output_attentions:
                    model_kwargs["encoder_attentions"] = encoder_outputs.attentions
                if output_hidden_states:
                    model_kwargs["encoder_hidden_states"] = encoder_outputs.hidden_states

        expanded_batch_idxs = tf.reshape(
            tf.repeat(tf.expand_dims(tf.range(batch_size), -1), repeats=num_beams * effective_batch_mult, axis=1),
            shape=(-1,),
        )
        # prepares text-based inputs
        if len(shape_list(input_ids)) == 2:
            input_ids = tf.gather(input_ids, expanded_batch_idxs, axis=0)
        if accepts_attention_mask:
            attention_mask = tf.gather(attention_mask, expanded_batch_idxs, axis=0)

        if self.config.is_encoder_decoder:

            # create empty decoder_input_ids
            input_ids = (
                tf.ones(
                    (effective_batch_size * num_beams, 1),
                    dtype=tf.int32,
                )
                * decoder_start_token_id
            )
            cur_len = 1

            assert (
                batch_size == encoder_outputs[0].shape[0]
            ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

            # expand encoder_outputs
            encoder_outputs = (tf.gather(encoder_outputs[0], expanded_batch_idxs, axis=0),)
        else:
            encoder_outputs = None
            cur_len = shape_list(input_ids)[-1]

        assert (
            cur_len < max_length
        ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

        return self._generate_beam_search(
            input_ids,
            cur_len=cur_len,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            early_stopping=early_stopping,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            batch_size=effective_batch_size,
            num_return_sequences=num_return_sequences,
            length_penalty=length_penalty,
            num_beams=num_beams,
            vocab_size=vocab_size,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            use_cache=use_cache,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            return_dict_in_generate=return_dict_in_generate,
            **model_kwargs,
        )

    def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        early_stopping,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        num_return_sequences,
        length_penalty,
        num_beams,
        vocab_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        forced_bos_token_id,
        forced_eos_token_id,
        return_dict_in_generate,
        **kwargs,
    ) -> Union[TFBeamSearchOutput, TFBeamSampleOutput, tf.Tensor]:
        """Generate sequences for each example with beam search."""

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
            for _ in range(batch_size)
        ]

        # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
        if do_sample is False:
            beam_scores_begin = tf.zeros((batch_size, 1), dtype=tf.float32)
            beam_scores_end = tf.ones((batch_size, num_beams - 1), dtype=tf.float32) * (-1e9)
            beam_scores = tf.concat([beam_scores_begin, beam_scores_end], -1)
        else:
            beam_scores = tf.zeros((batch_size, num_beams), dtype=tf.float32)

        beam_scores = tf.reshape(beam_scores, (batch_size * num_beams,))

        # variable to cache compute states
        past = None

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and kwargs["output_scores"]) else None
        decoder_attentions = () if (return_dict_in_generate and kwargs["output_attentions"]) else None
        cross_attentions = () if (return_dict_in_generate and kwargs["output_attentions"]) else None
        decoder_hidden_states = () if (return_dict_in_generate and kwargs["output_hidden_states"]) else None
        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if self.config.is_encoder_decoder:
            encoder_attentions = (
                kwargs["encoder_attentions"] if (return_dict_in_generate and kwargs["encoder_attentions"]) else None
            )
            encoder_hidden_states = (
                kwargs["encoder_hidden_states"]
                if (return_dict_in_generate and kwargs["encoder_hidden_states"])
                else None
            )
            # the refactored generate, without the encoder outputs in `past`, expects the `encoder_outputs`
            # variable to contain all (encoder_outputs, encoder_hidden_states, encoder_attentions) in
            # `prepare_inputs_for_generation`
            if encoder_hidden_states is not None:
                encoder_outputs = (*encoder_outputs, encoder_hidden_states)
            if encoder_attentions is not None:
                encoder_outputs = (*encoder_outputs, encoder_attentions)

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids,
                past=past,
                attention_mask=attention_mask,
                use_cache=use_cache,
                encoder_outputs=encoder_outputs,
                **kwargs,
            )
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=kwargs["output_attentions"],
                output_hidden_states=kwargs["output_hidden_states"],
            )
            next_token_logits = outputs.logits[:, -1, :]  # (batch_size * num_beams, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
                past = outputs[1]

            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                next_token_logits_penalties = _create_next_token_logits_penalties(
                    input_ids, next_token_logits, repetition_penalty
                )
                next_token_logits = tf.math.multiply(next_token_logits, next_token_logits_penalties)

            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if self.config.is_encoder_decoder and do_sample is False:
                next_token_logits = self.adjust_logits_during_generation(
                    next_token_logits,
                    cur_len=cur_len,
                    max_length=max_length,
                    forced_bos_token_id=forced_bos_token_id,
                    forced_eos_token_id=forced_eos_token_id,
                )
            #             calculate log softmax score
            scores = tf.nn.log_softmax(next_token_logits, axis=-1)  # (batch_size * num_beams, vocab_size)

            # set eos token prob to zero if min_length is not reached
            if eos_token_id is not None and cur_len < min_length:
                # create eos_token_id boolean mask
                num_batch_hypotheses = batch_size * num_beams

                is_token_logit_eos_token = tf.convert_to_tensor(
                    [True if token == eos_token_id else False for token in range(vocab_size)], dtype=tf.bool
                )
                eos_token_indices_mask = tf.broadcast_to(is_token_logit_eos_token, [num_batch_hypotheses, vocab_size])
                scores = tf.where(eos_token_indices_mask, -float("inf"), scores)

            if no_repeat_ngram_size > 0:
                # calculate a list of banned tokens to prevent repetitively generating the same ngrams
                # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
                num_batch_hypotheses = batch_size * num_beams
                banned_tokens = calc_banned_ngram_tokens(
                    input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
                )
                # create banned_tokens boolean mask
                banned_tokens_indices_mask = []
                for banned_tokens_slice in banned_tokens:
                    banned_tokens_indices_mask.append(
                        [True if token in banned_tokens_slice else False for token in range(vocab_size)]
                    )

                scores = tf.where(
                    tf.convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool), -float("inf"), scores
                )

            if bad_words_ids is not None:
                # calculate a list of banned tokens according to bad words
                banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

                banned_tokens_indices_mask = []
                for banned_tokens_slice in banned_tokens:
                    banned_tokens_indices_mask.append(
                        [True if token in banned_tokens_slice else False for token in range(vocab_size)]
                    )

                scores = tf.where(
                    tf.convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool), -float("inf"), scores
                )

            assert shape_list(scores) == [batch_size * num_beams, vocab_size]

            if do_sample:
                _scores = scores + tf.broadcast_to(
                    beam_scores[:, None], (batch_size * num_beams, vocab_size)
                )  # (batch_size * num_beams, vocab_size)

                # Top-p/top-k filtering
                _scores = tf_top_k_top_p_filtering(
                    _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                _scores = tf.reshape(_scores, (batch_size, num_beams * vocab_size))

                next_tokens = sample_without_replacement(
                    _scores, num_samples=2 * num_beams
                )  # (batch_size, 2 * num_beams)
                # Compute next scores
                next_scores = tf.gather(_scores, next_tokens, batch_dims=1)  # (batch_size, 2 * num_beams)

                # sort the sampled vector to make sure that the first num_beams samples are the best
                next_scores_indices = tf.argsort(next_scores, direction="DESCENDING", axis=1)
                next_scores = tf.gather(next_scores, next_scores_indices, batch_dims=1)  # (batch_size, num_beams * 2)
                next_tokens = tf.gather(next_tokens, next_scores_indices, batch_dims=1)  # (batch_size, num_beams * 2)
            else:
                # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
                next_scores = scores + tf.broadcast_to(
                    beam_scores[:, None], (batch_size * num_beams, vocab_size)
                )  # (batch_size * num_beams, vocab_size)

                # re-organize to group the beam together (we are keeping top hypothesis across beams)
                next_scores = tf.reshape(
                    next_scores, (batch_size, num_beams * vocab_size)
                )  # (batch_size, num_beams * vocab_size)

                next_scores, next_tokens = tf.math.top_k(next_scores, k=2 * num_beams, sorted=True)

            assert shape_list(next_scores) == shape_list(next_tokens) == [batch_size, 2 * num_beams]

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if kwargs["output_scores"]:
                    scores += (next_token_logits,)
                if kwargs["output_attentions"]:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if kwargs["output_hidden_states"]:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # next batch beam content
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # if we are done with this sentence
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                    ), f"Batch can only be done if at least {num_beams} beams have been generated."
                    assert (
                        eos_token_id is not None and pad_token_id is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next tokens for this sentence
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    # get beam and token IDs
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx * num_beams + beam_id
                    # add to generated hypotheses if end of sentence or last iteration
                    if (eos_token_id is not None) and (token_id.numpy() == eos_token_id):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            tf.identity(input_ids[effective_beam_id]), beam_token_score.numpy()
                        )
                    else:
                        # add next predicted token if it is not eos_token
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # Check if we are done so that we can save a pad step if all(done)
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    tf.reduce_max(next_scores[batch_idx]).numpy(), cur_len
                )

                # update next beam content
                assert len(next_sent_beam) == num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx + 1)

            # stop when we are done with each sentence
            if all(done):
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = tf.convert_to_tensor([x[0] for x in next_batch_beam], dtype=tf.float32)
            beam_tokens = tf.convert_to_tensor([x[1] for x in next_batch_beam], dtype=tf.int32)
            beam_idx = tf.convert_to_tensor([x[2] for x in next_batch_beam], dtype=tf.int32)

            # re-order batch and update current length
            input_ids = tf.stack([tf.identity(input_ids[x, :]) for x in beam_idx])
            input_ids = tf.concat([input_ids, tf.expand_dims(beam_tokens, 1)], axis=-1)
            cur_len = cur_len + 1

            # re-order internal states
            if past is not None:
                past = self._reorder_cache(past, beam_idx)

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = tf.concat(
                    [attention_mask, tf.ones((shape_list(attention_mask)[0], 1), dtype=tf.int32)], axis=-1
                )

        # finalize all open beam hypotheses and end to generated hypotheses
        for batch_idx in range(batch_size):
            # Add all open beam hypothesis to generated_hyps
            if done[batch_idx]:
                continue
            # test that beam scores match previously calculated scores if not eos and batch_idx not done
            if eos_token_id is not None and all(
                (token_id % vocab_size).numpy().item() != eos_token_id for token_id in next_tokens[batch_idx]
            ):
                if not tf.reduce_all(
                    next_scores[batch_idx, :num_beams] == tf.reshape(beam_scores, (batch_size, num_beams))[batch_idx]
                ):
                    raise ValueError(
                        f"If batch_idx is not done, final next scores: {next_scores[:, :num_beams][batch_idx]} have "
                        "to equal to accumulated beam_scores: "
                        f"{tf.reshape(beam_scores, (batch_size, num_beams))[batch_idx]}"
                    )
            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].numpy().item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)

        # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
        output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
        output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

        # select the best hypotheses
        sent_lengths_list = []
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths_list.append(len(best_hyp))
                best.append(best_hyp)
        assert output_batch_size == len(
            best
        ), f"Output batch size {output_batch_size} must match output beam hypotheses {len(best)}"

        sent_lengths = tf.convert_to_tensor(sent_lengths_list, dtype=tf.int32)

        # shorter batches are filled with pad_token
        if tf.reduce_min(sent_lengths).numpy() != tf.reduce_max(sent_lengths).numpy():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined"
            sent_max_len = min(tf.reduce_max(sent_lengths).numpy() + 1, max_length)
            decoded_list = []

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                assert sent_lengths[i] == shape_list(hypo)[0]
                # if sent_length is max_len do not pad
                if sent_lengths[i] == sent_max_len:
                    decoded_slice = hypo
                else:
                    # else pad to sent_max_len
                    num_pad_tokens = sent_max_len - sent_lengths[i]
                    padding = pad_token_id * tf.ones((num_pad_tokens,), dtype=tf.int32)
                    decoded_slice = tf.concat([hypo, padding], axis=-1)

                    # finish sentence with EOS token
                    if sent_lengths[i] < max_length:
                        decoded_slice = tf.where(
                            tf.range(sent_max_len, dtype=tf.int32) == sent_lengths[i],
                            eos_token_id * tf.ones((sent_max_len,), dtype=tf.int32),
                            decoded_slice,
                        )
                # add to list
                decoded_list.append(decoded_slice)

            decoded = tf.stack(decoded_list)
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_length for hypo in best)
            decoded = tf.stack(best)

        if return_dict_in_generate:
            if do_sample and self.config.is_encoder_decoder:
                return TFBeamSampleEncoderDecoderOutput(
                    sequences=decoded,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            elif do_sample and not self.config.is_encoder_decoder:
                return TFBeamSampleDecoderOnlyOutput(
                    sequences=decoded,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
            elif self.config.is_encoder_decoder:
                return TFBeamSearchEncoderDecoderOutput(
                    sequences=decoded,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return TFBeamSearchDecoderOnlyOutput(
                    sequences=decoded,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return decoded

    @staticmethod
    def _reorder_cache(past, beam_idx):
        return tuple(tf.gather(layer_past, beam_idx, axis=1) for layer_past in past)

    def adjust_logits_during_generation(
        self, logits, cur_len, max_length, forced_bos_token_id, forced_eos_token_id, **kwargs
    ):
        """
        Implement in subclasses of [`PreTrainedModel`] for custom behavior to adjust the logits in the generate method.
        """
        vocab_size = getattr(self.config, "vocab_size", None)
        if vocab_size is None and self.config.is_encoder_decoder:
            decoder_config = getattr(self.config, "decoder", None)
            if decoder_config is not None:
                vocab_size = getattr(self.config.decoder, "vocab_size", None)

        if cur_len == 1 and forced_bos_token_id is not None:
            vocab_range = tf.constant(range(vocab_size))
            return tf.where(vocab_range != forced_bos_token_id, -1e8, logits)
        elif cur_len == max_length - 1 and forced_eos_token_id is not None:
            vocab_range = tf.constant(range(vocab_size))
            return tf.where(vocab_range != forced_eos_token_id, -1e8, logits)
        else:
            return logits

    def _generate(
        self,
        input_ids=None,
        max_length=None,
        min_length=None,
        do_sample=None,
        early_stopping=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bad_words_ids=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        num_return_sequences=None,
        attention_mask=None,
        decoder_start_token_id=None,
        use_cache=None,
        seed=None,
        output_scores=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict_in_generate=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        **model_kwargs,
    ) -> Union[TFGreedySearchOutput, TFSampleOutput, TFBeamSearchOutput, TFBeamSampleOutput, tf.Tensor]:
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

        Adapted in part from [Facebook's XLM beam search
        code](https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529).

        Apart from `input_ids` and `attention_mask`, all the arguments below will default to the value of the attribute
        of the same name inside the [`PretrainedConfig`] of the model. The default values indicated are the default
        values of those config.

        Most of these parameters are explained in more detail in [this blog
        post](https://huggingface.co/blog/how-to-generate).

        Parameters:

            input_ids (`tf.Tensor` of `dtype=tf.int32` and shape `(batch_size, sequence_length)`, *optional*):
                The sequence used as a prompt for the generation. If `None` the method initializes it with
                `bos_token_id` and a batch size of 1.
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
            min_length (`int`, *optional*, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (`bool`, *optional*, defaults to `False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (`bool`, *optional*, defaults to `False`):
                Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (`float`, *optional*, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (`int`, *optional*, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional*, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher
                are kept for generation.
            repetition_penalty (`float`, *optional*, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            bos_token_id (`int`, *optional*):
                The id of the *beginning-of-sequence* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            length_penalty (`float`, *optional*, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty.

                Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
                order to encourage the model to produce longer sequences.
            no_repeat_ngram_size (`int`, *optional*, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            bad_words_ids(`List[int]`, *optional*):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use `tokenizer.encode(bad_word, add_prefix_space=True)`.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            attention_mask (`tf.Tensor` of `dtype=tf.int32` and shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values are in `[0, 1]`, 1 for tokens
                that are not masked, and 0 for masked tokens.

                If not provided, will default to a tensor the same shape as `input_ids` that masks the pad token.

                [What are attention masks?](../glossary#attention-mask)
            decoder_start_token_id (`int`, *optional*):
                If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            seed (`List[int]`, *optional*):
                Random seed to control sampling, containing two integers, used when `do_sample` is `True`. See the
                `seed` argument from stateless functions in `tf.random`.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            forced_bos_token_id (`int`, *optional*):
                The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful
                for multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be
                the target language token.
            forced_eos_token_id (`int`, *optional*):
                The id of the token to force as the last generated token when `max_length` is reached.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `call` function of the model.

        Return:
            [`~utils.ModelOutput`] or `tf.Tensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True` or when
            `config.return_dict_in_generate=True`) or a `tf.Tensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation_tf_utils.TFGreedySearchDecoderOnlyOutput`],
                    - [`~generation_tf_utils.TFSampleDecoderOnlyOutput`],
                    - [`~generation_tf_utils.TFBeamSearchDecoderOnlyOutput`],
                    - [`~generation_tf_utils.TFBeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation_tf_utils.TFGreedySearchEncoderDecoderOutput`],
                    - [`~generation_tf_utils.TFSampleEncoderDecoderOutput`],
                    - [`~generation_tf_utils.TFBeamSearchEncoderDecoderOutput`],
                    - [`~generation_tf_utils.TFBeamSampleEncoderDecoderOutput`]

        Examples:

        ```python
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")  # Initialize tokenizer
        model = TFAutoModelWithLMHead.from_pretrained("distilgpt2")
        # Greedy decoding
        outputs = model.generate(max_length=40)
        print(f"Generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

        tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
        model = TFAutoModelWithLMHead.from_pretrained("openai-gpt")
        input_context = "The dog"
        input_ids = tokenizer.encode(input_context, return_tensors="tf")  # encode input context
        # Generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
        outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)
        # 3 output sequences were generated
        for i in range(3):
            print(f"Generated {i}: {tokenizer.decode(outputs[i], skip_special_tokens=True)}")

        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        model = TFAutoModelWithLMHead.from_pretrained("distilgpt2")
        input_context = "The dog"
        input_ids = tokenizer.encode(input_context, return_tensors="tf")
        # Generate 3 candidates using sampling
        outputs = model.generate(
            input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3, do_sample=True
        )
        #  3 output sequences were generated
        for i in range(3):
            print(f"Generated {i}: {tokenizer.decode(outputs[i], skip_special_tokens=True)}")

        tokenizer = AutoTokenizer.from_pretrained("ctrl")
        model = TFAutoModelWithLMHead.from_pretrained("ctrl")
        # "Legal" is one of the control codes for ctrl
        input_context = "Legal My neighbor is"
        input_ids = tokenizer.encode(input_context, return_tensors="tf")
        outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)
        print(f"Generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = TFAutoModelWithLMHead.from_pretrained("gpt2")
        input_context = "My cute dog"
        bad_words_ids = [
            tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ["idiot", "stupid", "shut up"]
        ]
        input_ids = tokenizer.encode(input_context, return_tensors="tf")
        # generate sequences without allowing bad_words to be generated
        outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)
        ```"""
        # 1. Set generation parameters if not already defined
        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        forced_bos_token_id = (
            forced_bos_token_id if forced_bos_token_id is not None else self.config.forced_bos_token_id
        )
        forced_eos_token_id = (
            forced_eos_token_id if forced_eos_token_id is not None else self.config.forced_eos_token_id
        )

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        num_beams = num_beams if num_beams is not None else self.config.num_beams
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        if pad_token_id is None and eos_token_id is not None:
            logger.warning(f"Setting `pad_token_id` to {eos_token_id} (first `eos_token_id`) to generate sequence")
            pad_token_id = eos_token_id
        if min_length is not None and min_length > max_length:
            raise ValueError(
                f"Unfeasable length constraints: the minimum length ({min_length}) is larger than the maximum "
                f"length ({max_length})"
            )

        # 2. Define model inputs
        input_ids = self._prepare_model_inputs(input_ids, bos_token_id)
        # inputs_ids now has to be defined and cannot be None anymore
        batch_size = input_ids.shape[0]

        # 3. Prepare other model kwargs
        if output_attentions is not None:
            model_kwargs["output_attentions"] = output_attentions
        if output_hidden_states is not None:
            model_kwargs["output_hidden_states"] = output_hidden_states
        if use_cache is not None:
            model_kwargs["use_cache"] = use_cache
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.call).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(input_ids, pad_token_id)

        # 4. Prepare model inputs which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            # if encoder-decoder, we create encoder_outputs and add to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)
            # if encoder-decoder then `input_ids` come from `decoder_start_token_id`
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_token_id,
                model_kwargs=model_kwargs,
            )

        if input_ids.shape[-1] >= max_length:
            raise ValueError(
                f"The context has {input_ids.shape[-1]} number of tokens, "
                f"but `max_length` is only {max_length}. "
                "Please make sure that `max_length` is bigger than the number of tokens, "
                "by setting either `generate(max_length=...,...)` or `config.max_length = ...`"
            )

        # 5. determine generation mode
        # TODO(Matt, Joao, Patrick) - add more use cases here
        is_greedy_gen_mode = (num_beams == 1) and do_sample is False
        is_sample_gen_mode = (num_beams == 1) and do_sample is True
        is_beam_gen_mode = (num_beams > 1) and do_sample is False

        # 6. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
        )

        # 7. go into different generation modes
        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )
            # 8. run greedy search
            return self.greedy_search(
                input_ids,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                logits_processor=logits_processor,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )
        elif is_sample_gen_mode:
            # 8. prepare logits warper
            logits_warper = self._get_logits_warper(top_k=top_k, top_p=top_p, temperature=temperature)

            # 9. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 10. run sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                seed=seed,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            if num_beams < num_return_sequences:
                raise ValueError(
                    "Greedy beam search decoding cannot return more sequences than it has beams. Please set "
                    f"num_beams >= num_return_sequences, got {num_beams} and {num_return_sequences} (respectivelly)"
                )

            # 8. broadcast inputs to the desired number of beams
            input_ids = self._expand_to_num_beams(input_ids, num_beams=num_beams)

            if "encoder_outputs" in model_kwargs:
                model_kwargs["encoder_outputs"]["last_hidden_state"] = self._expand_to_num_beams(
                    model_kwargs["encoder_outputs"]["last_hidden_state"], num_beams=num_beams
                )

            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = self._expand_to_num_beams(
                    model_kwargs["attention_mask"], num_beams=num_beams
                )

            # 9. run beam search
            return self.beam_search(
                input_ids,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                logits_processor=logits_processor,
                return_dict_in_generate=return_dict_in_generate,
                num_return_sequences=num_return_sequences,
                **model_kwargs,
            )

        else:
            # TODO(Matt, Joao, Patrick) - add more sub-generation methods here
            raise NotImplementedError("Beam sampling is currently not implemented.")

    @staticmethod
    def _expand_to_num_beams(tensor: tf.Tensor, num_beams: int) -> tf.Tensor:
        return tf.broadcast_to(tensor[:, None], (tensor.shape[0], num_beams) + tensor.shape[1:])

    def _prepare_attention_mask_for_generation(
        self,
        inputs: tf.Tensor,
        pad_token_id: int,
    ) -> tf.Tensor:
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in (tf.int32, tf.int64)
        is_pad_token_in_inputs = (pad_token_id is not None) and tf.math.reduce_any(inputs == pad_token_id)
        # Check if input is input_ids and padded -> only then is attention_mask defined
        if is_input_ids and is_pad_token_in_inputs:
            return tf.cast(tf.math.not_equal(inputs, pad_token_id), dtype=tf.int32)
        else:
            return tf.ones(inputs.shape[:2], dtype=tf.int32)

    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor: tf.Tensor, model_kwargs) -> Dict[str, Any]:
        # get encoder and store encoder outputs
        encoder = self.get_encoder()

        # prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # vision models don't use `attention_mask`.
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[self.main_input_name] = inputs_tensor
        encoder_outputs = encoder(**encoder_kwargs)
        model_kwargs["encoder_outputs"] = encoder_outputs

        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        model_kwargs: Optional[Dict[str, tf.Tensor]] = None,
    ) -> tf.Tensor:

        # prepare `input_ids` for decoder if model is encoder-decoder
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            return model_kwargs.pop("decoder_input_ids")
        else:
            decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
            return tf.ones((batch_size, 1), dtype=tf.int32) * decoder_start_token_id

    def _get_decoder_start_token_id(self, decoder_start_token_id: int = None, bos_token_id: int = None) -> int:
        # retrieve decoder_start_token_id for encoder-decoder models
        # fall back to bos_token_id if necessary
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id

        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif (
            hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "decoder_start_token_id")
            and self.config.decoder.decoder_start_token_id is not None
        ):
            return self.config.decoder.decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        elif (
            hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "bos_token_id")
            and self.config.decoder.bos_token_id is not None
        ):
            return self.config.decoder.bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: tf.Tensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: Optional[tf.Tensor] = None,
        encoder_outputs: Optional[ModelOutput] = None,
        **model_kwargs,
    ) -> Tuple[tf.Tensor, Dict[str, Any]]:
        expanded_return_idx = tf.reshape(
            tf.tile(tf.reshape(tf.range(input_ids.shape[0]), (-1, 1)), (1, expand_size)), (-1,)
        )
        input_ids = tf.gather(input_ids, expanded_return_idx, axis=0)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = tf.gather(token_type_ids, expanded_return_idx, axis=0)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = tf.gather(attention_mask, expanded_return_idx, axis=0)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs["last_hidden_state"] = tf.gather(
                encoder_outputs.last_hidden_state, expanded_return_idx, axis=0
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

    def _prepare_model_inputs(self, inputs: Optional[tf.Tensor] = None, bos_token_id: Optional[int] = None):
        # TODO(Patrick) - adapt this function when making `generate` more flexible
        # for all kinds of input types
        if inputs is None:
            # if no `inputs` are passed create prompt of size (1,1) filled with BOS token
            if not isinstance(bos_token_id, int) or bos_token_id < 0:
                raise ValueError(
                    "you should either supply a context to complete as `input_ids` input "
                    "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
                )
            return tf.cast(tf.fill((1, 1), bos_token_id), dtype=tf.int32)

        return inputs

    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = tf.concat(
                    [attention_mask, tf.ones((shape_list(attention_mask)[0], 1), dtype=tf.int32)], axis=-1
                )

        return model_kwargs

    def _update_model_kwargs_for_xla_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], current_pos: tf.Tensor, max_length: int
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__} is not compileable with XLA at the moment. You should implement a "
            "`_update_model_kwargs_for_xla_generation` in the respective modeling file for XLA-compatible generation."
        )

    def _get_logits_warper(
        self,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> TFLogitsProcessorList:
        """
        This class returns a [`TFLogitsProcessorList`] list object that contains all relevant [`TFLogitsWarper`]
        instances used for multinomial sampling.
        """

        # init warp parameters
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        temperature = temperature if temperature is not None else self.config.temperature
        # instantiate warpers list
        warpers = TFLogitsProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if temperature is not None and temperature != 1.0:
            warpers.append(TFTemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            warpers.append(TFTopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
        if top_p is not None and top_p < 1.0:
            warpers.append(TFTopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
        return warpers

    def _get_logits_processor(
        self,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        bad_words_ids: List[List[int]],
        min_length: int,
        max_length: int,
        eos_token_id: int,
        forced_bos_token_id: int,
        forced_eos_token_id: int,
    ) -> TFLogitsProcessorList:
        """
        This class returns a [`TFLogitsProcessorList`] list object that contains all relevant [`TFLogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        processors = TFLogitsProcessorList()

        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # instantiate processors list
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(TFRepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            processors.append(TFNoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if bad_words_ids is not None:
            processors.append(TFNoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
        if min_length is not None and eos_token_id is not None and min_length > 0:
            processors.append(TFMinLengthLogitsProcessor(min_length, eos_token_id))
        if forced_bos_token_id is not None:
            processors.append(TFForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        if forced_eos_token_id is not None:
            processors.append(TFForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))

        return processors

    def greedy_search(
        self,
        input_ids: tf.Tensor,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        logits_processor: Optional[TFLogitsProcessorList] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[TFGreedySearchOutput, tf.Tensor]:
        r"""
        Generates sequences for models with a language modeling head using greedy decoding.

        Parameters:

            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`TFLogitsProcessorList`, *optional*):
                An instance of [`TFLogitsProcessorList`]. List of instances of class derived from [`TFLogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `call` function of the model. If
                model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation_tf_utils.TFGreedySearchDecoderOnlyOutput`],
            [`~generation_tf_utils.TFGreedySearchEncoderDecoderOutput`] or `tf.Tensor`: A `tf.Tensor` containing the
            generated tokens (default behaviour) or a [`~generation_tf_utils.TFGreedySearchDecoderOnlyOutput`] if
            `model.config.is_encoder_decoder=False` and `return_dict_in_generate=True` or a
            [`~generation_tf_utils.TFGreedySearchEncoderDecoderOutput`] if `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     TFAutoModelForCausalLM,
        ...     TFLogitsProcessorList,
        ...     TFMinLengthLogitsProcessor,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = TFAutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "Today is a beautiful day, and"
        >>> input_ids = tokenizer(input_prompt, return_tensors="tf").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = TFLogitsProcessorList(
        ...     [
        ...         TFMinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model.greedy_search(input_ids, logits_processor=logits_processor)

        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        ```"""

        # 1. init greedy_search values
        logits_processor = logits_processor if logits_processor is not None else TFLogitsProcessorList()

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )
        use_xla = not tf.executing_eagerly()

        # 2. init `attentions`, `hidden_states`, and `scores` tuples
        scores = [] if (return_dict_in_generate and output_scores) else None
        decoder_attentions = [] if (return_dict_in_generate and output_attentions) else None
        cross_attentions = [] if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = [] if (return_dict_in_generate and output_hidden_states) else None

        # 3. init tensors to use for "xla-compileable" generate function
        # define bsz, seq_length
        batch_size, seq_length = input_ids.shape

        # initialize `generated`, `finished_sequences`, and `current_pos`
        generated = tf.TensorArray(
            element_shape=(batch_size,),
            dtype=tf.int32,
            dynamic_size=False,
            size=max_length,
            clear_after_read=False,
        )

        # write prompt to generated
        for i in range(seq_length):
            generated = generated.write(i, input_ids[:, i])

        finished_sequences = tf.zeros((batch_size,), dtype=tf.bool)
        current_pos = tf.ones(shape=(1,), dtype=tf.int32) * seq_length

        # 4. define "xla-compile-able" stop-condition and auto-regressive function
        # define condition fn
        def greedy_search_cond_fn(generated, finished_sequences, next_tokens, current_pos, model_kwargs):
            """state termination condition fn."""
            return ~tf.reduce_all(finished_sequences)

        # define condition fn
        def greedy_search_body_fn(generated, finished_sequences, next_tokens, current_pos, model_kwargs):
            """state update fn."""
            # TODO(pvp, Joao) - `use_xla` can be removed here as soon as `position_ids` are corrected for the non-xla case in gpt2's `prepare_inputs_for_generation`.
            model_inputs = self.prepare_inputs_for_generation(next_tokens, use_xla=use_xla, **model_kwargs)
            # forward pass to get next token logits
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_token_logits = outputs.logits[:, -1]

            # Store scores, attentions and hidden_states when required
            if not use_xla and return_dict_in_generate:
                if output_scores:
                    scores.append(next_token_logits)
                if output_attentions and self.config.is_encoder_decoder:
                    decoder_attentions.append(outputs.decoder_attentions)
                elif output_attentions and not self.config.is_encoder_decoder:
                    decoder_attentions.append(outputs.attentions)
                    if self.config.is_encoder_decoder:
                        cross_attentions.append(outputs.cross_attentions)

                if output_hidden_states and self.config.is_encoder_decoder:
                    decoder_hidden_states.append(outputs.decoder_hidden_states)
                elif output_hidden_states and self.config.is_encoder_decoder:
                    decoder_hidden_states.append(outputs.hidden_states)

            # pre-process distribution
            # TODO(pvp, joao, matt) - all the logits processors need to be adapted
            # to be XLA compatible
            input_ids = None
            if not use_xla:
                input_ids = tf.reshape(generated.concat(), (-1, batch_size))
                input_ids = tf.transpose(input_ids[: current_pos[0]])
            next_tokens_scores = logits_processor(input_ids, next_token_logits, cur_len=current_pos[0])

            # argmax
            next_tokens = tf.argmax(next_tokens_scores, axis=-1, output_type=tf.int32)

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                unfinished_seq = 1 - tf.cast(finished_sequences, tf.int32)
                next_tokens = next_tokens * unfinished_seq + pad_token_id * (1 - unfinished_seq)
            finished_sequences = finished_sequences | (next_tokens == eos_token_id)

            # update `generated` and `current_pos`
            generated = generated.write(current_pos[0], next_tokens)
            next_tokens = next_tokens[:, None]
            current_pos += 1

            # update model_kwargs
            if use_xla:
                model_kwargs = self._update_model_kwargs_for_xla_generation(
                    outputs, model_kwargs, current_pos, max_length
                )
            else:
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                )
                # if we don't cache past key values we need the whole input
                if model_kwargs.get("past", None) is None:
                    # let's throw out `past` since we don't want `None` tensors
                    model_kwargs.pop("past", None)

                    next_tokens = tf.reshape(generated.concat(), (-1, batch_size))
                    next_tokens = tf.transpose(next_tokens[: current_pos[0]])

            return generated, finished_sequences, next_tokens, current_pos, model_kwargs

        # 5. run generation
        # 1st generation step has to be run before to initialize `past`
        generated, finished_sequences, next_tokens, current_pos, model_kwargs = greedy_search_body_fn(
            generated, finished_sequences, input_ids, current_pos, model_kwargs
        )

        # 2-to-n generation steps can then be run in autoregressive fashion
        # only in case 1st generation step does NOT yield EOS token though
        if greedy_search_cond_fn(generated, finished_sequences, next_tokens, current_pos, model_kwargs):
            maximum_iterations = max_length - seq_length - 1
            generated, _, _, current_pos, _ = tf.while_loop(
                greedy_search_cond_fn,
                greedy_search_body_fn,
                (generated, finished_sequences, next_tokens, current_pos, model_kwargs),
                maximum_iterations=maximum_iterations,
            )

        # 6. prepare outputs
        output_ids = tf.transpose(tf.reshape(generated.concat(), (-1, batch_size)))

        if not use_xla:
            # cut for backward compatibility
            output_ids = output_ids[:, : current_pos[0]]

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                # if model is an encoder-decoder, retrieve encoder attention weights
                # and hidden states
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

                scores = tuple(scores) if scores is not None else None
                decoder_attentions = tuple(decoder_attentions) if decoder_attentions is not None else None
                cross_attentions = tuple(cross_attentions) if cross_attentions is not None else None
                decoder_hidden_states = tuple(decoder_hidden_states) if decoder_hidden_states is not None else None

                return TFGreedySearchEncoderDecoderOutput(
                    sequences=output_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return TFGreedySearchDecoderOnlyOutput(
                    sequences=output_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return output_ids

    def sample(
        self,
        input_ids: tf.Tensor,
        logits_processor: Optional[TFLogitsProcessorList] = None,
        logits_warper: Optional[TFLogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        seed: Optional[Tuple[int, int]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[TFSampleOutput, tf.Tensor]:
        r"""
        Generates sequences for models with a language modeling head using multinomial sampling.

        Parameters:

            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`TFLogitsProcessorList`, *optional*):
                An instance of [`TFLogitsProcessorList`]. List of instances of class derived from [`TFLogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`TFLogitsProcessorList`, *optional*):
                An instance of [`TFLogitsProcessorList`]. List of instances of class derived from [`TFLogitsWarper`]
                used to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            seed (`List[int]`, *optional*):
                Random seed to control sampling, containing two integers, used when `do_sample` is `True`. See the
                `seed` argument from stateless functions in `tf.random`.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `call` function of the model. If model is an
                encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation_tf_utils.TFSampleDecoderOnlyOutput`], [`~generation_tf_utils.TFSampleEncoderDecoderOutput`]
            or `tf.Tensor`: A `tf.Tensor` containing the generated tokens (default behaviour) or a
            [`~generation_tf_utils.TFSampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation_tf_utils.TFSampleEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     TFAutoModelForCausalLM,
        ...     TFLogitsProcessorList,
        ...     TFMinLengthLogitsProcessor,
        ...     TFTopKLogitsWarper,
        ...     TFTemperatureLogitsWarper,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = TFAutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "Today is a beautiful day, and"
        >>> input_ids = tokenizer(input_prompt, return_tensors="tf").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = TFLogitsProcessorList(
        ...     [
        ...         TFMinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )
        >>> # instantiate logits processors
        >>> logits_warper = TFLogitsProcessorList(
        ...     [
        ...         TFTopKLogitsWarper(50),
        ...         TFTemperatureLogitsWarper(0.7),
        ...     ]
        ... )

        >>> outputs = model.sample(input_ids, logits_processor=logits_processor, logits_warper=logits_warper)

        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        ```"""

        # 1. init greedy_search values
        logits_processor = logits_processor if logits_processor is not None else TFLogitsProcessorList()
        logits_warper = logits_warper if logits_warper is not None else TFLogitsProcessorList()

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )
        use_xla = not tf.executing_eagerly()

        # 2. init `attentions`, `hidden_states`, and `scores` tuples
        scores = [] if (return_dict_in_generate and output_scores) else None
        decoder_attentions = [] if (return_dict_in_generate and output_attentions) else None
        cross_attentions = [] if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = [] if (return_dict_in_generate and output_hidden_states) else None

        # 3. init tensors to use for "xla-compileable" generate function
        # define bsz, seq_length
        batch_size, cur_len = input_ids.shape

        # initialize `generated`, `finished_sequences`
        generated = tf.TensorArray(
            element_shape=(batch_size,),
            dtype=tf.int32,
            dynamic_size=False,
            size=max_length,
            clear_after_read=False,
        )
        finished_sequences = tf.zeros((batch_size,), dtype=tf.bool)

        # write prompt to generated
        for i in range(cur_len):
            generated = generated.write(i, input_ids[:, i])

        # 4. define "xla-compile-able" stop-condition and auto-regressive function
        def sample_cond_fn(generated, finished_sequences, next_tokens, cur_len, model_kwargs):
            return ~tf.reduce_all(finished_sequences)

        def sample_body_fn(generated, finished_sequences, next_tokens, cur_len, model_kwargs):
            # TODO(pvp, Joao) - `use_xla` can be removed here as soon as `position_ids` are corrected for the non-xla case in gpt2's `prepare_inputs_for_generation`.
            model_inputs = self.prepare_inputs_for_generation(next_tokens, use_xla=use_xla, **model_kwargs)
            # forward pass to get next token logits
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_token_logits = outputs.logits[:, -1]

            # Store scores, attentions and hidden_states when required
            if not use_xla and return_dict_in_generate:
                if output_scores:
                    scores.append(next_token_logits)
                if output_attentions and self.config.is_encoder_decoder:
                    decoder_attentions.append(outputs.decoder_attentions)
                elif output_attentions and not self.config.is_encoder_decoder:
                    decoder_attentions.append(outputs.attentions)
                    if self.config.is_encoder_decoder:
                        cross_attentions.append(outputs.cross_attentions)

                if output_hidden_states and self.config.is_encoder_decoder:
                    decoder_hidden_states.append(outputs.decoder_hidden_states)
                elif output_hidden_states and self.config.is_encoder_decoder:
                    decoder_hidden_states.append(outputs.hidden_states)

            # pre-process distribution
            # TODO(pvp, joao, matt) - all the logits processors/wrappers need to be adapted
            # to be XLA compatible
            input_ids = None
            if not use_xla:
                input_ids = tf.reshape(generated.concat(), (-1, batch_size))
                input_ids = tf.transpose(input_ids[:cur_len])
            next_tokens_scores = logits_processor(input_ids, next_token_logits, cur_len=cur_len)
            next_tokens_scores = logits_warper(input_ids, next_tokens_scores)

            # sample
            if seed is not None:
                sample_seed = seed
            else:
                sample_seed = tf.cast(self.seed_generator.make_seeds(count=1)[:, 0], dtype=tf.int32)
            next_tokens = tf.squeeze(
                tf.random.stateless_categorical(
                    logits=next_tokens_scores, num_samples=1, seed=sample_seed, dtype=tf.int32
                ),
                axis=1,
            )

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                unfinished_seq = 1 - tf.cast(finished_sequences, tf.int32)
                next_tokens = next_tokens * unfinished_seq + pad_token_id * (1 - unfinished_seq)
            finished_sequences = finished_sequences | (next_tokens == eos_token_id)

            # update `generated` and `cur_len`
            generated = generated.write(cur_len, next_tokens)
            next_tokens = next_tokens[:, None]
            cur_len += 1

            # update model_kwargs
            if use_xla:
                model_kwargs = self._update_model_kwargs_for_xla_generation(outputs, model_kwargs, cur_len, max_length)
            else:
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                )
                # if we don't cache past key values we need the whole input
                if model_kwargs.get("past", None) is None:
                    # let's throw out `past` since we don't want `None` tensors
                    model_kwargs.pop("past", None)

                    next_tokens = tf.reshape(generated.concat(), (-1, batch_size))
                    next_tokens = tf.transpose(next_tokens[:cur_len])

            return generated, finished_sequences, next_tokens, cur_len, model_kwargs

        # 5. run generation
        # 1st generation step has to be run before to initialize `past`
        generated, finished_sequences, next_tokens, cur_len, model_kwargs = sample_body_fn(
            generated, finished_sequences, input_ids, cur_len, model_kwargs
        )

        # 2-to-n generation steps can then be run in autoregressive fashion
        # only in case 1st generation step does NOT yield EOS token though
        if sample_cond_fn(generated, finished_sequences, next_tokens, cur_len, model_kwargs):
            maximum_iterations = max_length - cur_len - 1
            generated, _, _, cur_len, _ = tf.while_loop(
                sample_cond_fn,
                sample_body_fn,
                (generated, finished_sequences, next_tokens, cur_len, model_kwargs),
                maximum_iterations=maximum_iterations,
            )

        # 6. prepare outputs
        output_ids = tf.transpose(tf.reshape(generated.concat(), (-1, batch_size)))

        if not use_xla:
            # cut for backward compatibility
            output_ids = output_ids[:, :cur_len]

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                # if model is an encoder-decoder, retrieve encoder attention weights
                # and hidden states
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

                scores = tuple(scores) if scores is not None else None
                decoder_attentions = tuple(decoder_attentions) if decoder_attentions is not None else None
                cross_attentions = tuple(cross_attentions) if cross_attentions is not None else None
                decoder_hidden_states = tuple(decoder_hidden_states) if decoder_hidden_states is not None else None

                return TFSampleEncoderDecoderOutput(
                    sequences=output_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return TFSampleDecoderOnlyOutput(
                    sequences=output_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return output_ids

    def beam_search(
        self,
        input_ids: tf.Tensor,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[bool] = None,
        logits_processor: Optional[TFLogitsProcessorList] = None,
        num_return_sequences: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[TFBeamSearchOutput, tf.Tensor]:
        r"""
        Generates sequences for models with a language modeling head using beam search with multinomial sampling.

        Parameters:

            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            length_penalty (`float`, *optional*, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty.
            early_stopping (`bool`, *optional*, defaults to `False`):
                Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
            logits_processor (`[TFLogitsProcessorList]`, *optional*):
                An instance of [`TFLogitsProcessorList`]. List of instances of class derived from [`TFLogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `call` function of the model. If model is an
                encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation_tf_utils.TFBeamSearchDecoderOnlyOutput`],
            [`~generation_tf_utils.TFBeamSearchEncoderDecoderOutput`] or `tf.Tensor`: A `tf.Tensor` containing the
            generated tokens (default behaviour) or a [`~generation_tf_utils.TFBeamSearchDecoderOnlyOutput`] if
            `model.config.is_encoder_decoder=False` and `return_dict_in_generate=True` or a
            [`~generation_tf_utils.TFBeamSearchEncoderDecoderOutput`] if `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     TFAutoModelForSeq2SeqLM,
        ...     TFLogitsProcessorList,
        ...     TFMinLengthLogitsProcessor,
        ... )
        >>> import tensorflow as tf

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="tf").input_ids

        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = tf.ones((num_beams, 1), dtype=tf.int64)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         tf.repeat(encoder_input_ids, num_beams, axis=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate logits processors
        >>> logits_processor = TFLogitsProcessorList(
        ...     [TFMinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id)]
        ... )

        >>> outputs = model.beam_search(input_ids, logits_processor=logits_processor, **model_kwargs)

        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        ```"""

        def flatten_beam_dim(tensor, batch_axis=0):
            """Flattens the first two dimensions of a non-scalar array."""
            # ignore scalars (e.g. cache index)
            if tf.rank(tensor) == 0:
                return tensor
            return tf.reshape(
                tensor,
                tensor.shape[:batch_axis]
                + [tensor.shape[batch_axis] * tensor.shape[batch_axis + 1]]
                + tensor.shape[batch_axis + 2 :],
            )

        def unflatten_beam_dim(tensor, batch_size, num_beams, batch_axis=0):
            """Unflattens the first, flat batch*beam dimension of a non-scalar array."""
            # ignore scalars (e.g. cache index)
            if tf.rank(tensor) == 0:
                return tensor
            return tf.reshape(
                tensor, tensor.shape[:batch_axis] + [batch_size, num_beams] + tensor.shape[batch_axis + 1 :]
            )

        def gather_beams(nested, beam_indices, batch_axis=0):
            """Gathers the beam slices indexed by beam_indices into new beam array."""

            def gather_fn(tensor):
                # ignore scalars (e.g. cache index)
                if tf.rank(tensor) == 0:
                    return tensor
                else:
                    if batch_axis > 0:
                        # pushes all dimentions before the batch to the end, so we get (batch, beam_id, ...)
                        perm = [axis for axis in range(tf.rank(tensor)) if axis >= batch_axis] + list(
                            range(batch_axis)
                        )
                        tensor = tf.transpose(tensor, perm=perm)

                    gathered_tensor = tf.gather(params=tensor, indices=beam_indices, axis=1, batch_dims=1)
                    if batch_axis > 0:
                        # transposes back to the original dimensions
                        perm = [axis for axis in range(tf.rank(tensor)) if axis >= batch_axis] + list(
                            range(batch_axis)
                        )
                        perm = tf.math.invert_permutation(perm)
                        gathered_tensor = tf.transpose(gathered_tensor, perm=perm)

                    return gathered_tensor

            return tf.nest.map_structure(gather_fn, nested)

        # 1. init beam_search values
        logits_processor = logits_processor if logits_processor is not None else TFLogitsProcessorList()

        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

        use_xla = not tf.executing_eagerly()
        # TODO (Joao): fix cache format or find programatic way to detect cache index
        # GPT2 and other models has a slightly different cache structure, with a different batch axis
        model_name = str(self.decoder) if "EncoderDecoder" in str(self) else str(self)
        cache_batch_axis = 1 if any([model_prefix in model_name for model_prefix in ("TFGPT2", "TFCTRL")]) else 0

        # 2. init `attentions`, `hidden_states`, and `scores` tuples
        scores = [] if (return_dict_in_generate and output_scores) else None
        decoder_attentions = [] if (return_dict_in_generate and output_attentions) else None
        cross_attentions = [] if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = [] if (return_dict_in_generate and output_hidden_states) else None

        # 3. init tensors to use for "xla-compileable" generate function
        batch_size, num_beams, cur_len = input_ids.shape
        input_ids_length = cur_len

        # per batch, beam-item holding current token in loop, pre-populated with `pad_token_id`
        sequences = tf.TensorArray(
            element_shape=(batch_size, num_beams),
            dtype=tf.int32,
            dynamic_size=False,
            size=max_length,
            clear_after_read=False,
        )
        running_sequences = tf.TensorArray(
            element_shape=(batch_size, num_beams),
            dtype=tf.int32,
            dynamic_size=False,
            size=max_length,
            clear_after_read=False,
        )
        intermediary_running_sequences = tf.TensorArray(
            element_shape=(batch_size, num_beams * 2),
            dtype=tf.int32,
            dynamic_size=False,
            size=max_length,
            clear_after_read=False,
        )
        for i in range(max_length):
            sequences = sequences.write(i, tf.broadcast_to(pad_token_id, (batch_size, num_beams)))
            running_sequences = running_sequences.write(i, tf.broadcast_to(pad_token_id, (batch_size, num_beams)))
            intermediary_running_sequences = intermediary_running_sequences.write(
                i, tf.broadcast_to(pad_token_id, (batch_size, num_beams * 2))
            )

        # write prompt to running_sequences
        for i in range(cur_len):
            running_sequences = running_sequences.write(i, input_ids[:, :, i])

        # per batch,beam-item state bit indicating if sentence has finished.
        is_sent_finished = tf.zeros((batch_size, num_beams), dtype=tf.bool)

        # per batch, beam-item score, logprobs
        running_scores = tf.tile(
            tf.expand_dims(tf.convert_to_tensor([0.0] + [-1.0e9] * (num_beams - 1)), axis=0), [batch_size, 1]
        )
        scores = tf.ones((batch_size, num_beams)) * -1.0e9

        # flatten beam dim
        if "encoder_outputs" in model_kwargs:
            model_kwargs["encoder_outputs"]["last_hidden_state"] = flatten_beam_dim(
                model_kwargs["encoder_outputs"]["last_hidden_state"]
            )
        if "attention_mask" in model_kwargs:
            model_kwargs["attention_mask"] = flatten_beam_dim(model_kwargs["attention_mask"])

        # 4. define "xla-compile-able" stop-condition and auto-regressive function
        # define stop-condition and auto-regressive function
        def beam_search_cond_fn(
            cur_len,
            running_sequences,
            running_scores,
            sequences,
            scores,
            is_sent_finished,
            input_ids_length,
            model_kwargs,
        ):
            """
            Beam Search termination condition function -- halts the generation loop if any of these conditions becomes
            False
            """
            # 1. is less than max length?
            not_max_length_yet = cur_len < max_length

            # 2. can the new beams still improve?
            best_running_score = running_scores[:, :1] / (max_length**length_penalty)
            worst_finished_score = tf.where(
                is_sent_finished, tf.math.reduce_min(scores, axis=1, keepdims=True), -1.0e9
            )
            improvement_still_possible = tf.math.reduce_all(worst_finished_score < best_running_score)

            # 3. is there still a beam that has not finished?
            still_open_beam = ~(tf.math.reduce_all(is_sent_finished) & early_stopping)

            return not_max_length_yet & (still_open_beam | improvement_still_possible)

        def beam_search_body_fn(
            cur_len,
            running_sequences,
            running_scores,
            sequences,
            scores,
            is_sent_finished,
            input_ids_length,
            model_kwargs,
            intermediary_running_sequences=None,
        ):
            """
            Beam Search iterative update function -- each iteration adds a new token and updates the best sequences
            seen so far
            """
            # TODO (joao): this loop is probably faster with gather/scatters, instead of using `tf.TensorArray`.
            # Alternativelly, attempt to rewrite function with permuted axis, when enabling XLA.

            # 1. Forward current tokens

            # TF places the dynamic dimension (seq_len) in the first axis, we want it in the last
            running_sequences_seq_last = tf.transpose(running_sequences.stack(), perm=[1, 2, 0])
            input_token = tf.slice(
                running_sequences_seq_last,
                (0, 0, cur_len - input_ids_length),
                (batch_size, num_beams, input_ids_length),
            )
            model_inputs = self.prepare_inputs_for_generation(
                flatten_beam_dim(input_token), use_xla=use_xla, **model_kwargs
            )
            model_outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            logits = unflatten_beam_dim(model_outputs.logits[:, -1], batch_size, num_beams)

            # Store scores, attentions and hidden_states when required
            if not use_xla and return_dict_in_generate:
                if output_scores:
                    scores.append(model_outputs.logits[:, -1])
                if output_attentions and self.config.is_encoder_decoder:
                    decoder_attentions.append(model_outputs.decoder_attentions)
                elif output_attentions and not self.config.is_encoder_decoder:
                    decoder_attentions.append(model_outputs.attentions)
                    if self.config.is_encoder_decoder:
                        cross_attentions.append(model_outputs.cross_attentions)

                if output_hidden_states and self.config.is_encoder_decoder:
                    decoder_hidden_states.append(model_outputs.decoder_hidden_states)
                elif output_hidden_states and self.config.is_encoder_decoder:
                    decoder_hidden_states.append(model_outputs.hidden_states)

            # 2. Compute log probs
            # get log probabilities from logits, process logits with processors (*e.g.* min_length, ...), and
            # add new logprobs to existing running logprobs scores.
            log_probs = tf.nn.log_softmax(logits)
            log_probs = logits_processor(
                flatten_beam_dim(running_sequences_seq_last), flatten_beam_dim(log_probs), cur_len=cur_len
            )
            log_probs = unflatten_beam_dim(log_probs, batch_size, num_beams)
            log_probs = log_probs + tf.expand_dims(running_scores, axis=2)
            vocab_size = log_probs.shape[2]
            log_probs = tf.reshape(log_probs, (batch_size, num_beams * vocab_size))

            # 3. Retrieve top-K
            # Each item in batch has num_beams * vocab_size candidate sequences. For each item, get the top 2*k
            # candidates with the highest log-probabilities. We gather the top 2*K beams here so that even if the
            # best K sequences reach EOS simultaneously, we have another K sequences remaining to continue the live
            # beam search.
            # Gather the top 2*K scores from _all_ beams.
            # Gather 2*k top beams.
            # Recover the beam index by floor division.
            # Recover token id by modulo division and expand Id array for broadcasting.
            # Update sequences for the 2*K top-k new sequences.
            beams_to_keep = 2 * num_beams
            topk_log_probs, topk_indices = tf.math.top_k(log_probs, k=beams_to_keep)
            topk_beam_indices = topk_indices // vocab_size
            topk_running_sequences_seq_last = gather_beams(running_sequences_seq_last, topk_beam_indices)
            topk_ids = topk_indices % vocab_size

            # writes the new token
            intermediary_running_sequences = intermediary_running_sequences.unstack(
                tf.transpose(topk_running_sequences_seq_last, perm=[2, 0, 1])
            )
            topk_sequences = intermediary_running_sequences.write(cur_len, topk_ids)
            topk_sequences_seq_last = tf.transpose(topk_sequences.stack(), perm=[1, 2, 0])

            # 4. Check which sequences have ended
            # Update current sequences: Did the top `num_beams` sequences reach an end marker?
            # To prevent these just finished sequences from being added to the current sequences
            # set of active beam search sequences, set their log probs to a very large negative value.
            eos_in_next_token = topk_sequences_seq_last[:, :, cur_len] == eos_token_id
            if eos_token_id is None:
                eos_in_next_token = tf.broadcast_to(eos_in_next_token, topk_sequences_seq_last[:, :, cur_len].shape)
            did_topk_just_finished = eos_in_next_token & tf.broadcast_to(
                tf.concat((tf.ones((num_beams), dtype=tf.bool), tf.zeros((num_beams), dtype=tf.bool)), axis=0),
                eos_in_next_token.shape,
            )

            # non-top `num_beams` eos tokens can't be used to finish a beam, but the others can't be used in the next
            # running sentences either
            running_topk_log_probs = topk_log_probs + tf.cast(eos_in_next_token, tf.float32) * -1.0e9

            # 5. Get running sequences scores for next
            # Determine the top k beam indices (from top 2*k beams) from log probs and gather top k beams
            # (from top 2*k beams).
            next_topk_indices = tf.math.top_k(running_topk_log_probs, k=num_beams)[1]
            next_running_sequences_seq_last, next_running_scores = gather_beams(
                [topk_sequences_seq_last, running_topk_log_probs], next_topk_indices
            )

            # 6. Process topk logits
            # Further process log probs:
            # - add length penalty
            # - make sure no scores can be added anymore if beam is full
            # - make sure still running sequences cannot be chosen as finalized beam
            topk_log_probs = topk_log_probs / (cur_len**length_penalty)
            beams_in_batch_are_full = (
                tf.broadcast_to(
                    tf.math.reduce_all(is_sent_finished, axis=-1, keepdims=True), did_topk_just_finished.shape
                )
                & early_stopping
            )
            add_penalty = ~did_topk_just_finished | beams_in_batch_are_full
            topk_log_probs += tf.cast(add_penalty, tf.float32) * -1.0e9

            # 7. Get scores, sequences, is sentence finished for next.
            # Combine sequences, scores, and flags along the beam dimension and compare new finished sequence scores
            # to existing finished scores and select the best from the new set of beams
            sequences_seq_last = tf.transpose(sequences.stack(), perm=[1, 2, 0])
            merged_sequences = tf.concat([sequences_seq_last, topk_sequences_seq_last], axis=1)
            merged_scores = tf.concat([scores, topk_log_probs], axis=1)
            merged_is_sent_finished = tf.concat([is_sent_finished, did_topk_just_finished], axis=1)
            topk_merged_indices = tf.math.top_k(merged_scores, k=num_beams)[1]
            next_sequences_seq_last, next_scores, next_is_sent_finished = gather_beams(
                [merged_sequences, merged_scores, merged_is_sent_finished], topk_merged_indices
            )

            # 8. Prepare data for the next iteration
            # Determine the top k beam indices from the original set of all beams. With these, gather the top k
            # beam-associated caches.
            if "past_key_values" in model_outputs:
                cache = tf.nest.map_structure(
                    lambda tensor: unflatten_beam_dim(tensor, batch_size, num_beams, batch_axis=cache_batch_axis),
                    model_outputs.past_key_values,
                )
                next_running_indices = gather_beams(topk_beam_indices, next_topk_indices)
                next_cache = gather_beams(cache, next_running_indices, batch_axis=cache_batch_axis)
                model_outputs["past_key_values"] = tf.nest.map_structure(
                    lambda tensor: flatten_beam_dim(tensor, batch_axis=cache_batch_axis), next_cache
                )

            if use_xla:
                next_model_kwargs = self._update_model_kwargs_for_xla_generation(
                    model_outputs, model_kwargs, cur_len, max_length
                )
            else:
                next_model_kwargs = self._update_model_kwargs_for_generation(
                    model_outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                )

                # if we don't cache past key values we need the whole input
                if model_kwargs.get("past", None) is None:
                    next_input_ids_length = cur_len + 1
                    # let's throw out `past` since we don't want `None` tensors
                    model_kwargs.pop("past", None)
                else:
                    next_input_ids_length = 1

            # 9. Prepare the `tf.TensorArray` for the next iteration
            next_sequences = sequences.unstack(tf.transpose(next_sequences_seq_last, perm=[2, 0, 1]))
            next_running_sequences = running_sequences.unstack(
                tf.transpose(next_running_sequences_seq_last, perm=[2, 0, 1])
            )

            return (
                cur_len + 1,
                next_running_sequences,
                next_running_scores,
                next_sequences,
                next_scores,
                next_is_sent_finished,
                next_input_ids_length,
                next_model_kwargs,
            )

        # 5. run generation
        # Adds the `intermediary_running_sequences` TensorArray into the body, needed as a scratchpad
        beam_search_body_fn = partial(
            beam_search_body_fn, intermediary_running_sequences=intermediary_running_sequences
        )

        # 1st generation step has to be run before to initialize `past` (if active)
        (
            cur_len,
            running_sequences,
            running_scores,
            sequences,
            scores,
            is_sent_finished,
            input_ids_length,
            model_kwargs,
        ) = beam_search_body_fn(
            cur_len,
            running_sequences,
            running_scores,
            sequences,
            scores,
            is_sent_finished,
            input_ids_length,
            model_kwargs,
        )

        # 2-to-n generation steps can then be run in autoregressive fashion (only in case 1st generation step does
        # NOT yield EOS token though)
        if beam_search_cond_fn(
            cur_len,
            running_sequences,
            running_scores,
            sequences,
            scores,
            is_sent_finished,
            input_ids_length,
            model_kwargs,
        ):
            maximum_iterations = max_length - cur_len
            cur_len, running_sequences, running_scores, sequences, scores, is_sent_finished, _, _ = tf.while_loop(
                beam_search_cond_fn,
                beam_search_body_fn,
                (
                    cur_len,
                    running_sequences,
                    running_scores,
                    sequences,
                    scores,
                    is_sent_finished,
                    input_ids_length,
                    model_kwargs,
                ),
                maximum_iterations=maximum_iterations,
            )

        # 6. prepare outputs
        # convert the sequneces to tf.Tensor with shape (batch_size, num_beams, seq_len)
        sequences_seq_last = tf.transpose(sequences.stack(), perm=[1, 2, 0])
        running_sequences_seq_last = tf.transpose(running_sequences.stack(), perm=[1, 2, 0])

        # Account for the edge-case where there are no finished sequences for a particular batch item. If so, return
        # running sequences for that batch item.
        none_finished = tf.math.reduce_any(is_sent_finished, axis=1)
        sequences_seq_last = tf.where(none_finished[:, None, None], sequences_seq_last, running_sequences_seq_last)
        scores = tf.where(none_finished[:, None], scores, running_scores)

        # Take best beams for each batch (the score is sorted in ascending order)
        sequences_seq_last = flatten_beam_dim(sequences_seq_last[:, :num_return_sequences, :])
        scores = flatten_beam_dim(scores[:, :num_return_sequences])

        if not use_xla:
            # Cut for backward compatibility
            sequences_seq_last = sequences_seq_last[:, :cur_len]

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

                return TFBeamSearchEncoderDecoderOutput(
                    sequences=sequences_seq_last,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return TFBeamSearchDecoderOnlyOutput(
                    sequences=sequences_seq_last,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequences_seq_last


def _create_next_token_logits_penalties(input_ids, logits, repetition_penalty):
    # create logit penalties for already seen input_ids
    token_penalties = np.ones(shape_list(logits))
    prev_input_ids = [np.unique(input_id) for input_id in input_ids.numpy()]
    for i, prev_input_id in enumerate(prev_input_ids):
        logit_penalized = logits[i].numpy()[prev_input_id]
        logit_penalties = np.zeros(logit_penalized.shape)
        # if previous logit score is < 0 then multiply repetition penalty else divide
        logit_penalties[logit_penalized < 0] = repetition_penalty
        logit_penalties[logit_penalized > 0] = 1 / repetition_penalty
        np.put(token_penalties[i], prev_input_id, logit_penalties)
    return tf.convert_to_tensor(token_penalties, dtype=tf.float32)


def calc_banned_ngram_tokens(prev_input_ids, num_hypos, no_repeat_ngram_size, cur_len):
    # Copied from fairseq for no_repeat_ngram in beam_search
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].numpy().tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].numpy().tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def calc_banned_bad_words_ids(prev_input_ids, bad_words_ids):
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_tokens):
            # if bad word tokens are longer than prev tokens they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert (
                len(banned_token_seq) > 0
            ), f"Banned words token sequences { bad_words_ids} cannot have an empty list"

            if _tokens_match(prev_input_ids_slice.numpy().tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens


def tf_top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    logits_shape = shape_list(logits)

    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits_shape[-1])  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < tf.math.top_k(logits, k=top_k)[0][..., -1, None]
        logits = tf.where(indices_to_remove, filter_value, logits)
    if top_p < 1.0:
        sorted_indices = tf.argsort(logits, direction="DESCENDING")
        sorted_logits = tf.gather(
            logits, sorted_indices, axis=-1, batch_dims=1
        )  # expects logits to be of dim (batch_size, vocab_size)

        cumulative_probs = tf.math.cumsum(stable_softmax(sorted_logits, axis=-1), axis=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p

        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove = tf.concat(
                [
                    tf.zeros_like(sorted_indices_to_remove[:, :min_tokens_to_keep]),
                    sorted_indices_to_remove[:, min_tokens_to_keep:],
                ],
                -1,
            )

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove = tf.concat(
            [tf.zeros_like(sorted_indices_to_remove[:, :1]), sorted_indices_to_remove[:, :-1]],
            -1,
        )
        # scatter sorted tensors to original indexing
        indices_to_remove = scatter_values_on_batch_indices(sorted_indices_to_remove, sorted_indices)
        logits = tf.where(indices_to_remove, filter_value, logits)
    return logits


def scatter_values_on_batch_indices(values, batch_indices):
    shape = shape_list(batch_indices)
    # broadcast batch dim to shape
    broad_casted_batch_dims = tf.reshape(tf.broadcast_to(tf.expand_dims(tf.range(shape[0]), axis=-1), shape), [1, -1])
    # transform batch_indices to pair_indices
    pair_indices = tf.transpose(tf.concat([broad_casted_batch_dims, tf.reshape(batch_indices, [1, -1])], 0))
    # scatter values to pair indices
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), shape)


def sample_without_replacement(logits, num_samples):
    """
    categorical sampling without replacement is currently not implemented the gumbel-max trick will do for now see
    https://github.com/tensorflow/tensorflow/issues/9260 for more info
    """
    z = -tf.math.log(tf.random.uniform(shape_list(logits), 0, 1))
    _, indices = tf.nn.top_k(logits + z, num_samples)
    return indices


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= cur_score
            return ret
