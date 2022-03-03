# coding=utf-8
# Copyright 2021 The Google AI Flax Team Authors, and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


from functools import partial
from typing import Dict, Optional

import numpy as np

import flax
import jax
import jax.numpy as jnp
from jax import lax

from .file_utils import ModelOutput
from .generation_flax_logits_process import (
    FlaxForcedBOSTokenLogitsProcessor,
    FlaxForcedEOSTokenLogitsProcessor,
    FlaxLogitsProcessorList,
    FlaxMinLengthLogitsProcessor,
    FlaxTemperatureLogitsWarper,
    FlaxTopKLogitsWarper,
    FlaxTopPLogitsWarper,
)
from .utils import logging


logger = logging.get_logger(__name__)


@flax.struct.dataclass
class FlaxGreedySearchOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (`jnp.ndarray` of shape `(batch_size, max_length)`):
            The generated sequences.
    """

    sequences: jnp.ndarray = None


@flax.struct.dataclass
class FlaxSampleOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using sampling.


    Args:
        sequences (`jnp.ndarray` of shape `(batch_size, max_length)`):
            The generated sequences.
    """

    sequences: jnp.ndarray = None


@flax.struct.dataclass
class FlaxBeamSearchOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (`jnp.ndarray` of shape `(batch_size, max_length)`):
            The generated sequences.
        scores (`jnp.ndarray` of shape `(batch_size,)`):
            The scores (log probabilites) of the generated sequences.
    """

    sequences: jnp.ndarray = None
    scores: jnp.ndarray = None


@flax.struct.dataclass
class GreedyState:
    cur_len: jnp.ndarray
    sequences: jnp.ndarray
    running_token: jnp.ndarray
    is_sent_finished: jnp.ndarray
    model_kwargs: Dict[str, jnp.ndarray]


@flax.struct.dataclass
class SampleState:
    cur_len: jnp.ndarray
    sequences: jnp.ndarray
    running_token: jnp.ndarray
    is_sent_finished: jnp.ndarray
    prng_key: jnp.ndarray
    model_kwargs: Dict[str, jnp.ndarray]


@flax.struct.dataclass
class BeamSearchState:
    cur_len: jnp.ndarray
    running_sequences: jnp.ndarray
    running_scores: jnp.ndarray
    sequences: jnp.ndarray
    scores: jnp.ndarray
    is_sent_finished: jnp.ndarray
    model_kwargs: Dict[str, jnp.ndarray]


class FlaxGenerationMixin:
    """
    A class containing all of the functions supporting generation, to be used as a mixin in [`FlaxPreTrainedModel`].
    """

    @staticmethod
    def _run_loop_in_debug(cond_fn, body_fn, init_state):
        """
        Run generation in untraced mode. This should only be used for debugging purposes.
        """
        state = init_state
        while cond_fn(state):
            state = body_fn(state)
        return state

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids, params, model_kwargs):
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
        }
        model_kwargs["encoder_outputs"] = self.encode(input_ids, params=params, return_dict=True, **encoder_kwargs)
        return model_kwargs

    @staticmethod
    def _expand_to_num_beams(tensor, num_beams):
        return jnp.broadcast_to(tensor[:, None], (tensor.shape[0], num_beams) + tensor.shape[1:])

    def _adapt_logits_for_beam_search(self, logits):
        """
        This function can be overwritten in the specific modeling_flax_<model-name>.py classes to allow for custom beam
        search behavior. Note that the only model that overwrites this method is [`~transformes.FlaxMarianMTModel`].
        """
        return logits

    def generate(
        self,
        input_ids: jnp.ndarray,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        do_sample: Optional[bool] = None,
        prng_key: Optional[jnp.ndarray] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        num_beams: Optional[int] = None,
        no_repeat_ngram_size: Optional[int] = None,
        min_length: Optional[int] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[bool] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        **model_kwargs,
    ):
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        and, multinomial sampling.

        Apart from `input_ids`, all the arguments below will default to the value of the attribute of the same name
        inside the [`PretrainedConfig`] of the model. The default values indicated are the default values of those
        config.

        Most of these parameters are explained in more detail in [this blog
        post](https://huggingface.co/blog/how-to-generate).

        Parameters:

            input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
            do_sample (`bool`, *optional*, defaults to `False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            temperature (`float`, *optional*, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (`int`, *optional*, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional*, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher
                are kept for generation.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            bos_token_id (`int`, *optional*):
                The id of the *beginning-of-sequence* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            decoder_start_token_id (`int`, *optional*):
                If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
            trace (`bool`, *optional*, defaults to `True`):
                Whether to trace generation. Setting `trace=False` should only be used for debugging and will lead to a
                considerably slower runtime.
            params (`Dict[str, jnp.ndarray]`, *optional*):
                Optionally the model parameters can be passed. Can be useful for parallelized generation.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If the model
                is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs
                should be prefixed with *decoder_*. Also accepts `encoder_outputs` to skip encoder part.

        Return:
            [`~file_utils.ModelOutput`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

        >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        >>> model = FlaxAutoModelForCausalLM.from_pretrained("distilgpt2")
        >>> input_context = "The dog"
        >>> # encode input context
        >>> input_ids = tokenizer(input_context, return_tensors="np").input_ids
        >>> # generate candidates using sampling
        >>> outputs = model.generate(input_ids=input_ids, max_length=20, top_k=30, do_sample=True)
        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        ```"""
        # set init values
        max_length = max_length if max_length is not None else self.config.max_length
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id else self.config.decoder_start_token_id
        )
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        if decoder_start_token_id is None and self.config.is_encoder_decoder:
            raise ValueError("`decoder_start_token_id` has to be defined for encoder-decoder generation.")

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            if model_kwargs.get("encoder_outputs") is None:
                model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, params, model_kwargs)
            # prepare decoder_input_ids for generation
            input_ids = jnp.ones((input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_beams = num_beams if num_beams is not None else self.config.num_beams

        if not do_sample and num_beams == 1:
            logits_processor = self._get_logits_processor(
                no_repeat_ngram_size, min_length, max_length, eos_token_id, forced_bos_token_id, forced_eos_token_id
            )
            return self._greedy_search(
                input_ids,
                max_length,
                pad_token_id,
                eos_token_id,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        elif do_sample and num_beams == 1:
            logits_warper = self._get_logits_warper(top_k=top_k, top_p=top_p, temperature=temperature)
            logits_processor = self._get_logits_processor(
                no_repeat_ngram_size, min_length, max_length, eos_token_id, forced_bos_token_id, forced_eos_token_id
            )
            return self._sample(
                input_ids,
                max_length,
                pad_token_id,
                eos_token_id,
                prng_key,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        elif not do_sample and num_beams > 1:
            # broadcast input_ids & encoder_outputs
            input_ids = self._expand_to_num_beams(input_ids, num_beams=num_beams)

            if "encoder_outputs" in model_kwargs:
                model_kwargs["encoder_outputs"]["last_hidden_state"] = self._expand_to_num_beams(
                    model_kwargs["encoder_outputs"]["last_hidden_state"], num_beams=num_beams
                )

            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = self._expand_to_num_beams(
                    model_kwargs["attention_mask"], num_beams=num_beams
                )

            logits_processor = self._get_logits_processor(
                no_repeat_ngram_size, min_length, max_length, eos_token_id, forced_bos_token_id, forced_eos_token_id
            )

            return self._beam_search(
                input_ids,
                max_length,
                pad_token_id,
                eos_token_id,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        else:
            raise NotImplementedError("`Beam sampling is currently not implemented.")

    def _get_logits_warper(
        self, top_k: Optional[int] = None, top_p: Optional[float] = None, temperature: Optional[float] = None
    ) -> FlaxLogitsProcessorList:
        """
        This class returns a [`FlaxLogitsProcessorList`] list object that contains all relevant [`FlaxLogitsWarper`]
        instances used for multinomial sampling.
        """

        # init warp parameters
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        temperature = temperature if temperature is not None else self.config.temperature
        # instantiate warpers list
        warpers = FlaxLogitsProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if temperature is not None and temperature != 1.0:
            warpers.append(FlaxTemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            warpers.append(FlaxTopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
        if top_p is not None and top_p < 1.0:
            warpers.append(FlaxTopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))

        return warpers

    def _get_logits_processor(
        self,
        no_repeat_ngram_size: int,
        min_length: int,
        max_length: int,
        eos_token_id: int,
        forced_bos_token_id: int,
        forced_eos_token_id: int,
    ) -> FlaxLogitsProcessorList:
        """
        This class returns a [`FlaxLogitsProcessorList`] list object that contains all relevant [`FlaxLogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        processors = FlaxLogitsProcessorList()

        # init warp parameters
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        min_length = min_length if min_length is not None else self.config.min_length
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        forced_bos_token_id = (
            forced_bos_token_id if forced_bos_token_id is not None else self.config.forced_bos_token_id
        )
        forced_eos_token_id = (
            forced_eos_token_id if forced_eos_token_id is not None else self.config.forced_eos_token_id
        )

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(FlaxMinLengthLogitsProcessor(min_length, eos_token_id))
        if forced_bos_token_id is not None:
            processors.append(FlaxForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        if forced_eos_token_id is not None:
            processors.append(FlaxForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
        return processors

    def _greedy_search(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        # init values
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        batch_size, cur_len = input_ids.shape

        eos_token_id = jnp.array(eos_token_id)
        pad_token_id = jnp.array(pad_token_id)
        cur_len = jnp.array(cur_len)

        # per batch-item holding current token in loop.
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

        # per batch-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
        # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
        model = self.decode if self.config.is_encoder_decoder else self
        # initialize model specific kwargs
        model_kwargs = self.prepare_inputs_for_generation(input_ids, max_length, **model_kwargs)

        # initialize state
        state = GreedyState(
            cur_len=cur_len,
            sequences=sequences,
            running_token=input_ids,
            is_sent_finished=is_sent_finished,
            model_kwargs=model_kwargs,
        )

        def greedy_search_cond_fn(state):
            """state termination condition fn."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            return ~finish_generation

        def greedy_search_body_fn(state):
            """state update fn."""
            model_outputs = model(state.running_token, params=params, **state.model_kwargs)
            logits = model_outputs.logits[:, -1]

            # apply min_length, ...
            logits = logits_processor(state.sequences, logits, state.cur_len)

            next_token = jnp.argmax(logits, axis=-1)

            next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
            next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
            next_token = next_token[:, None]

            next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)
            return GreedyState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                running_token=next_token,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
            )

        # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
        if input_ids.shape[1] > 1:
            state = greedy_search_body_fn(state)

        if not trace:
            state = self._run_loop_in_debug(greedy_search_cond_fn, greedy_search_body_fn, state)
        else:
            state = lax.while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)

        return FlaxGreedySearchOutput(sequences=state.sequences)

    def _sample(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        prng_key: Optional[jnp.ndarray] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        logits_warper: Optional[FlaxLogitsProcessorList] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        # init values
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        batch_size, cur_len = input_ids.shape

        eos_token_id = jnp.array(eos_token_id)
        pad_token_id = jnp.array(pad_token_id)
        cur_len = jnp.array(cur_len)

        # per batch-item holding current token in loop.
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

        # per batch-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
        # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
        model = self.decode if self.config.is_encoder_decoder else self

        # initialize model specific kwargs
        model_kwargs = self.prepare_inputs_for_generation(input_ids, max_length, **model_kwargs)

        # initialize state
        state = SampleState(
            cur_len=cur_len,
            sequences=sequences,
            running_token=input_ids,
            is_sent_finished=is_sent_finished,
            prng_key=prng_key,
            model_kwargs=model_kwargs,
        )

        def sample_search_cond_fn(state):
            """state termination condition fn."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            return ~finish_generation

        def sample_search_body_fn(state):
            """state update fn."""
            prng_key, prng_key_next = jax.random.split(state.prng_key)
            model_outputs = model(state.running_token, params=params, **state.model_kwargs)

            logits = model_outputs.logits[:, -1]

            # apply min_length, ...
            logits = logits_processor(state.sequences, logits, state.cur_len)
            # apply top_k, top_k, temperature
            logits = logits_warper(logits, logits, state.cur_len)

            next_token = jax.random.categorical(prng_key, model_outputs.logits[:, -1], axis=-1)

            next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
            next_token = next_token * ~next_is_sent_finished + pad_token_id * next_is_sent_finished
            next_token = next_token[:, None]

            next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)

            return SampleState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                running_token=next_token,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
                prng_key=prng_key_next,
            )

        # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
        if input_ids.shape[1] > 1:
            state = sample_search_body_fn(state)

        if not trace:
            state = self._run_loop_in_debug(sample_search_cond_fn, sample_search_body_fn, state)
        else:
            state = lax.while_loop(sample_search_cond_fn, sample_search_body_fn, state)

        return FlaxSampleOutput(sequences=state.sequences)

    def _beam_search(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[bool] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        """
        This beam search function is heavily inspired by Flax's official example:
        https://github.com/google/flax/blob/master/examples/wmt/train.py#L254
        """

        def flatten_beam_dim(tensor):
            """Flattens the first two dimensions of a non-scalar array."""
            # ignore scalars (e.g. cache index)
            if tensor.ndim == 0:
                return tensor
            return tensor.reshape((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

        def unflatten_beam_dim(tensor, batch_size, num_beams):
            """Unflattens the first, flat batch*beam dimension of a non-scalar array."""
            # ignore scalars (e.g. cache index)
            if tensor.ndim == 0:
                return tensor
            return tensor.reshape((batch_size, num_beams) + tensor.shape[1:])

        def gather_beams(nested, beam_indices, batch_size, new_num_beams):
            """
            Gathers the beam slices indexed by beam_indices into new beam array.
            """
            batch_indices = jnp.reshape(
                jnp.arange(batch_size * new_num_beams) // new_num_beams, (batch_size, new_num_beams)
            )

            def gather_fn(tensor):
                # ignore scalars (e.g. cache index)
                if tensor.ndim == 0:
                    return tensor
                else:
                    return tensor[batch_indices, beam_indices]

            return jax.tree_map(gather_fn, nested)

        # init values
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

        batch_size, num_beams, cur_len = input_ids.shape

        eos_token_id = jnp.array(eos_token_id)
        pad_token_id = jnp.array(pad_token_id)
        cur_len = jnp.array(cur_len)

        # per batch,beam-item holding current token in loop.
        sequences = jnp.full((batch_size, num_beams, max_length), pad_token_id, dtype=jnp.int32)
        running_sequences = jnp.full((batch_size, num_beams, max_length), pad_token_id, dtype=jnp.int32)
        running_sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0, 0))

        # per batch,beam-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size, num_beams), dtype=jnp.bool_)

        # per batch,beam-item score, logprobs
        running_scores = jnp.tile(jnp.array([0.0] + [np.array(-1.0e7)] * (num_beams - 1)), [batch_size, 1])
        scores = jnp.ones((batch_size, num_beams)) * np.array(-1.0e7)

        # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
        # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
        model = self.decode if self.config.is_encoder_decoder else self

        # flatten beam dim
        if "encoder_outputs" in model_kwargs:
            model_kwargs["encoder_outputs"]["last_hidden_state"] = flatten_beam_dim(
                model_kwargs["encoder_outputs"]["last_hidden_state"]
            )
        if "attention_mask" in model_kwargs:
            model_kwargs["attention_mask"] = flatten_beam_dim(model_kwargs["attention_mask"])

        # initialize model specific kwargs
        model_kwargs = self.prepare_inputs_for_generation(flatten_beam_dim(input_ids), max_length, **model_kwargs)

        # initialize state
        state = BeamSearchState(
            cur_len=cur_len,
            running_sequences=running_sequences,
            running_scores=running_scores,
            sequences=sequences,
            scores=scores,
            is_sent_finished=is_sent_finished,
            model_kwargs=model_kwargs,
        )

        def beam_search_cond_fn(state):
            """beam search state termination condition fn."""

            # 1. is less than max length?
            not_max_length_yet = state.cur_len < max_length

            # 2. can the new beams still improve?
            best_running_score = state.running_scores[:, -1:] / (max_length**length_penalty)
            worst_finished_score = jnp.where(
                state.is_sent_finished, jnp.min(state.scores, axis=1, keepdims=True), np.array(-1.0e7)
            )
            improvement_still_possible = jnp.all(worst_finished_score < best_running_score)

            # 3. is there still a beam that has not finished?
            still_open_beam = ~(jnp.all(state.is_sent_finished) & early_stopping)

            return not_max_length_yet & still_open_beam & improvement_still_possible

        def beam_search_body_fn(state, input_ids_length=1):
            """beam search state update fn."""
            # 1. Forward current tokens
            # Collect the current position slice along length to feed the fast
            # autoregressive decoder model.  Flatten the beam dimension into batch
            # dimension for feeding into the model.
            # unflatten beam dimension
            # Unflatten beam dimension in attention cache arrays
            input_token = flatten_beam_dim(
                lax.dynamic_slice(
                    state.running_sequences,
                    (0, 0, state.cur_len - input_ids_length),
                    (batch_size, num_beams, input_ids_length),
                )
            )
            model_outputs = model(input_token, params=params, **state.model_kwargs)

            logits = unflatten_beam_dim(model_outputs.logits[:, -1], batch_size, num_beams)
            cache = jax.tree_map(
                lambda tensor: unflatten_beam_dim(tensor, batch_size, num_beams), model_outputs.past_key_values
            )

            # adapt logits for FlaxMarianMTModel
            logits = self._adapt_logits_for_beam_search(logits)

            # 2. Compute log probs
            # get log probabilities from logits,
            # process logits with processors (*e.g.* min_length, ...), and
            # add new logprobs to existing running logprobs scores.
            log_probs = jax.nn.log_softmax(logits)
            log_probs = logits_processor(
                flatten_beam_dim(running_sequences), flatten_beam_dim(log_probs), state.cur_len
            )
            log_probs = unflatten_beam_dim(log_probs, batch_size, num_beams)
            log_probs = log_probs + jnp.expand_dims(state.running_scores, axis=2)
            vocab_size = log_probs.shape[2]
            log_probs = log_probs.reshape((batch_size, num_beams * vocab_size))

            # 3. Retrieve top-K
            # Each item in batch has num_beams * vocab_size candidate sequences.
            # For each item, get the top 2*k candidates with the highest log-
            # probabilities. We gather the top 2*K beams here so that even if the best
            # K sequences reach EOS simultaneously, we have another K sequences
            # remaining to continue the live beam search.
            # Gather the top 2*K scores from _all_ beams.
            # Gather 2*k top beams.
            # Recover the beam index by floor division.
            # Recover token id by modulo division and expand Id array for broadcasting.
            # Update sequences for the 2*K top-k new sequences.
            beams_to_keep = 2 * num_beams
            topk_log_probs, topk_indices = lax.top_k(log_probs, k=beams_to_keep)
            topk_beam_indices = topk_indices // vocab_size
            topk_running_sequences = gather_beams(
                state.running_sequences, topk_beam_indices, batch_size, beams_to_keep
            )
            topk_ids = jnp.expand_dims(topk_indices % vocab_size, axis=2)
            topk_sequences = lax.dynamic_update_slice(topk_running_sequences, topk_ids, (0, 0, state.cur_len))

            # 4. Check which sequences have ended
            # Update current sequences:
            # Did any of these sequences reach an end marker?
            # To prevent these just finished sequences from being added to the current sequences
            # set of active beam search sequences, set their log probs to a very large
            # negative value.
            did_topk_just_finished = topk_sequences[:, :, state.cur_len] == eos_token_id
            running_topk_log_probs = topk_log_probs + did_topk_just_finished * np.array(-1.0e7)
            # 5. Get running sequences scores for next
            # Determine the top k beam indices (from top 2*k beams) from log probs
            # and gather top k beams (from top 2*k beams).
            next_topk_indices = jnp.flip(lax.top_k(running_topk_log_probs, k=num_beams)[1], axis=1)
            next_running_sequences, next_running_scores = gather_beams(
                [topk_sequences, running_topk_log_probs], next_topk_indices, batch_size, num_beams
            )

            # 6. Process topk logits
            # Further process log probs:
            # - add length penalty
            # - make sure no scores can be added anymore if beam is full
            # - make sure still running sequences cannot be chosen as finalized beam
            topk_log_probs = topk_log_probs / (state.cur_len**length_penalty)
            beams_in_batch_are_full = (
                jnp.broadcast_to(state.is_sent_finished.all(axis=-1, keepdims=True), did_topk_just_finished.shape)
                & early_stopping
            )
            add_penalty = ~did_topk_just_finished | beams_in_batch_are_full
            topk_log_probs += add_penalty * np.array(-1.0e7)

            # 7. Get scores, sequences, is sentence finished for next.
            # Combine sequences, scores, and flags along the beam dimension and compare
            # new finished sequence scores to existing finished scores and select the
            # best from the new set of beams
            merged_sequences = jnp.concatenate([state.sequences, topk_sequences], axis=1)
            merged_scores = jnp.concatenate([state.scores, topk_log_probs], axis=1)
            merged_is_sent_finished = jnp.concatenate([state.is_sent_finished, did_topk_just_finished], axis=1)
            topk_merged_indices = jnp.flip(lax.top_k(merged_scores, k=num_beams)[1], axis=1)
            next_sequences, next_scores, next_is_sent_finished = gather_beams(
                [merged_sequences, merged_scores, merged_is_sent_finished], topk_merged_indices, batch_size, num_beams
            )

            # 8. Update model kwargs.
            # Determine the top k beam indices from the original set of all beams.
            # With these, gather the top k beam-associated caches.
            next_running_indices = gather_beams(topk_beam_indices, next_topk_indices, batch_size, num_beams)
            next_cache = gather_beams(cache, next_running_indices, batch_size, num_beams)
            model_outputs["past_key_values"] = jax.tree_map(lambda x: flatten_beam_dim(x), next_cache)
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)

            return BeamSearchState(
                cur_len=state.cur_len + 1,
                running_scores=next_running_scores,
                running_sequences=next_running_sequences,
                scores=next_scores,
                sequences=next_sequences,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
            )

        # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
        if input_ids.shape[-1] > 1:
            state = partial(beam_search_body_fn, input_ids_length=input_ids.shape[-1])(state)

        if not trace:
            state = self._run_loop_in_debug(beam_search_cond_fn, beam_search_body_fn, state)
        else:
            state = lax.while_loop(beam_search_cond_fn, beam_search_body_fn, state)

        # Account for the edge-case where there are no finished sequences for a
        # particular batch item. If so, return running sequences for that batch item.
        none_finished = jnp.any(state.is_sent_finished, axis=1)
        sequences = jnp.where(none_finished[:, None, None], state.sequences, state.running_sequences)
        scores = jnp.where(none_finished[:, None], state.scores, state.running_scores)

        # take best beam for each batch
        sequences = sequences[:, -1]
        scores = scores[:, -1]

        return FlaxBeamSearchOutput(sequences=sequences, scores=scores)
