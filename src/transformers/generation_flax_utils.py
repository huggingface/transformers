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


from typing import Dict, Optional

import numpy as np

import flax
import jax
import jax.numpy as jnp
import jaxlib.xla_extension as jax_xla
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
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_length)`):
            The generated sequences.
    """

    sequences: jax_xla.DeviceArray = None


@flax.struct.dataclass
class FlaxSampleOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using sampling.


    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_length)`):
            The generated sequences.
    """

    sequences: jax_xla.DeviceArray = None


@flax.struct.dataclass
class FlaxBeamSearchOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_length)`):
            The generated sequences.
        scores (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`):
            The scores (log probabilites) of the generated sequences.
    """

    sequences: jax_xla.DeviceArray = None
    scores: jax_xla.DeviceArray = None


@flax.struct.dataclass
class GreedyState:
    cur_len: jax_xla.DeviceArray
    sequences: jax_xla.DeviceArray
    current_token: jax_xla.DeviceArray
    is_sent_finished: jax_xla.DeviceArray
    model_kwargs: Dict[str, jax_xla.DeviceArray]


@flax.struct.dataclass
class SampleState:
    cur_len: jax_xla.DeviceArray
    sequences: jax_xla.DeviceArray
    current_token: jax_xla.DeviceArray
    is_sent_finished: jax_xla.DeviceArray
    prng_key: jax_xla.DeviceArray
    model_kwargs: Dict[str, jax_xla.DeviceArray]


@flax.struct.dataclass
class BeamSearchState:
    cur_len: jax_xla.DeviceArray
    current_sequences: jax_xla.DeviceArray
    current_scores: jax_xla.DeviceArray
    sequences: jax_xla.DeviceArray
    scores: jax_xla.DeviceArray
    is_sent_finished: jax_xla.DeviceArray
    model_kwargs: Dict[str, jax_xla.DeviceArray]


class FlaxGenerationMixin:
    """
    A class containing all of the functions supporting generation, to be used as a mixin in
    :class:`~transformers.FlaxPreTrainedModel`.
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

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids, model_kwargs):
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
        }
        model_kwargs["encoder_outputs"] = self.encode(input_ids, return_dict=True, **encoder_kwargs)
        return model_kwargs

    def generate(
        self,
        input_ids: jax_xla.DeviceArray,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        do_sample: Optional[bool] = None,
        prng_key: Optional[jax_xla.DeviceArray] = None,
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
        params: Optional[Dict[str, jax_xla.DeviceArray]] = None,
        **model_kwargs,
    ):
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        and, multinomial sampling.

        Apart from :obj:`input_ids`, all the arguments below will default to the value of the attribute of the same
        name inside the :class:`~transformers.PretrainedConfig` of the model. The default values indicated are the
        default values of those config.

        Most of these parameters are explained in more detail in `this blog post
        <https://huggingface.co/blog/how-to-generate>`__.

        Parameters:

            input_ids (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            temperature (:obj:`float`, `optional`, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (:obj:`int`, `optional`, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (:obj:`float`, `optional`, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or
                higher are kept for generation.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            trace (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether to trace generation. Setting ``trace=False`` should only be used for debugging and will lead to
                a considerably slower runtime.
            params (:obj:`Dict[str, jax_xla.DeviceArray]`, `optional`):
                Optionally the model parameters can be passed. Can be useful for parallelized generation.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model.

        Return:
            :class:`~transformers.file_utils.ModelOutput`.

        Examples::
            >>> from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

            >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            >>> model = FlaxAutoModelForCausalLM.from_pretrained("distilgpt2")
            >>> input_context = "The dog"
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="jax").input_ids
            >>> # generate candidates using sampling
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, top_k=30, do_sample=True)
            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """
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
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)
            # prepare decoder_input_ids for generation
            input_ids = jnp.ones((input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_beams = num_beams if num_beams is not None else self.config.num_beams

        if not do_sample and num_beams == 1:
            return self._greedy_search(
                input_ids,
                max_length,
                pad_token_id,
                eos_token_id,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        elif do_sample and num_beams == 1:
            logits_warper = self._get_logits_warper(top_k=top_k, top_p=top_p, temperature=temperature)
            return self._sample(
                input_ids,
                max_length,
                pad_token_id,
                eos_token_id,
                prng_key,
                logits_warper=logits_warper,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        elif not do_sample and num_beams > 1:
            # broadcast input_ids & encoder_outputs
            batch_size, sequence_length = input_ids.shape

            input_ids = jnp.broadcast_to(input_ids[:, None, :], (batch_size, num_beams, sequence_length))
            last_hidden_state = model_kwargs["encoder_outputs"]["last_hidden_state"]
            model_kwargs["encoder_outputs"]["last_hidden_state"] = jnp.broadcast_to(
                last_hidden_state[:, None, :],
                (
                    batch_size,
                    num_beams,
                )
                + last_hidden_state.shape[1:],
            )

            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = jnp.broadcast_to(
                    model_kwargs["attention_mask"][:, None, :],
                    (batch_size, num_beams, model_kwargs["attention_mask"].shape[-1]),
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
        self, top_k: int = None, top_p: float = None, temperature: float = None
    ) -> FlaxLogitsProcessorList:
        """
        This class returns a :obj:`~transformers.FlaxLogitsProcessorList` list object that contains all relevant
        :obj:`~transformers.FlaxLogitsWarper` instances used for multinomial sampling.
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
        This class returns a :obj:`~transformers.FlaxLogitsProcessorList` list object that contains all relevant
        :obj:`~transformers.FlaxLogitsProcessor` instances used to modify the scores of the language model head.
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
        trace: bool = True,
        params: Optional[Dict[str, jax_xla.DeviceArray]] = None,
        model_kwargs: Optional[Dict[str, jax_xla.DeviceArray]] = None,
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
            current_token=input_ids,
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
            model_outputs = model(state.current_token, params=params, **state.model_kwargs)
            next_token = jnp.argmax(model_outputs.logits[:, -1], axis=-1)

            next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
            next_token = next_token * ~next_is_sent_finished + pad_token_id * next_is_sent_finished
            next_token = next_token[:, None]

            next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)
            return GreedyState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                current_token=next_token,
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
        prng_key: Optional[jax_xla.DeviceArray] = None,
        logits_warper: Optional[FlaxLogitsProcessorList] = None,
        trace: bool = True,
        params: Optional[Dict[str, jax_xla.DeviceArray]] = None,
        model_kwargs: Optional[Dict[str, jax_xla.DeviceArray]] = None,
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
            current_token=input_ids,
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
            model_outputs = model(state.current_token, params=params, **state.model_kwargs)

            logits = model_outputs.logits[:, -1]

            # apply top_k, top_k, temperature
            logits = logits_warper(state.sequences, logits)

            next_token = jax.random.categorical(prng_key, model_outputs.logits[:, -1], axis=-1)

            next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
            next_token = next_token * ~next_is_sent_finished + pad_token_id * next_is_sent_finished
            next_token = next_token[:, None]

            next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)

            return SampleState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                current_token=next_token,
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
        params: Optional[Dict[str, jax_xla.DeviceArray]] = None,
        model_kwargs: Optional[Dict[str, jax_xla.DeviceArray]] = None,
    ):
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
        current_sequences = jnp.full((batch_size, num_beams, max_length), pad_token_id, dtype=jnp.int32)
        current_sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0, 0))

        # per batch,beam-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size, num_beams), dtype=jnp.bool_)

        # per batch,beam-item score, logprobs
        current_scores = jnp.tile(jnp.array([0.0] + [np.array(-1.0e7)] * (num_beams - 1)), [batch_size, 1])
        scores = jnp.ones((batch_size, num_beams)) * np.array(-1.0e7)

        # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
        # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
        model = self.decode if self.config.is_encoder_decoder else self

        # reshape
        reshaped_input_ids = input_ids.reshape((batch_size * num_beams, cur_len))
        last_hidden_state = model_kwargs["encoder_outputs"]["last_hidden_state"]
        model_kwargs["encoder_outputs"]["last_hidden_state"] = last_hidden_state.reshape(
            (batch_size * num_beams,) + last_hidden_state.shape[2:]
        )
        if "attention_mask" in model_kwargs:
            model_kwargs["attention_mask"] = model_kwargs["attention_mask"].reshape(
                (batch_size * num_beams, model_kwargs["attention_mask"].shape[-1])
            )

        # initialize model specific kwargs
        model_kwargs = self.prepare_inputs_for_generation(reshaped_input_ids, max_length, **model_kwargs)

        # initialize state
        state = BeamSearchState(
            cur_len=cur_len,
            current_sequences=current_sequences,
            current_scores=current_scores,
            sequences=sequences,
            scores=scores,
            is_sent_finished=is_sent_finished,
            model_kwargs=model_kwargs,
        )

        def beam_search_cond_fn(state):
            """state termination condition fn."""
            has_reached_max_length = state.cur_len == max_length

            best_curr_score = state.current_scores[:, -1:] / (max_length ** length_penalty)
            worst_finished_score = jnp.min(state.scores, axis=1, keepdims=True)
            worst_finished_score = jnp.where(state.is_sent_finished, worst_finished_score, np.array(-1.0e7))

            no_improvement_possible = jnp.all(worst_finished_score > best_curr_score)

            stop_early = jnp.all(state.is_sent_finished) & early_stopping
            return (~has_reached_max_length) & (~stop_early) & (~no_improvement_possible)

        def flatten_beam_dim(tensor):
            """Flattens the first two dimensions of a non-scalar array."""
            if tensor.ndim == 0:  # ignore scalars (e.g. cache index)
                return tensor
            return tensor.reshape((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

        def unflatten_beam_dim(x, batch_size, num_beams):
            """Unflattens the first, flat batch*beam dimension of a non-scalar array."""
            if x.ndim == 0:  # ignore scalars (e.g. cache index)
                return x
            assert batch_size * num_beams == x.shape[0]
            return x.reshape((batch_size, num_beams) + x.shape[1:])

        def gather_beams(nested, beam_indices, batch_size, new_num_beams):
            """
            Gathers the beam slices indexed by beam_indices into new beam array.

            Args:
                nested: pytree of arrays or scalars (the latter ignored).
                beam_indices: array of beam_indices
                batch_size: int: size of batch.
                new_num_beams: int: size of _new_ beam dimension.

            Returns:
                New pytree with new beam arrays. [batch_size, old_num_beams, ...] --> [batch_size, new_num_beams, ...]
            """
            batch_indices = jnp.reshape(
                jnp.arange(batch_size * new_num_beams) // new_num_beams, (batch_size, new_num_beams)
            )

            def gather_fn(x):
                if x.ndim == 0:  # ignore scalars (e.g. cache index)
                    return x
                else:
                    return x[batch_indices, beam_indices]

            return jax.tree_map(gather_fn, nested)

        def gather_topk_beams(nested, score_or_log_prob, batch_size, new_beam_size):
            _, topk_indices = lax.top_k(score_or_log_prob, k=new_beam_size)
            topk_indices = jnp.flip(topk_indices, axis=1)
            return gather_beams(nested, topk_indices, batch_size, new_beam_size)

        def beam_search_body_fn(state):
            """state update fn."""
            """Beam search loop state update function."""
            # Collect the current position slice along length to feed the fast
            # autoregressive decoder model.  Flatten the beam dimension into batch
            # dimension for feeding into the model.
            # --> [batch * beam, 1]
            input_token = flatten_beam_dim(
                lax.dynamic_slice(state.current_sequences, (0, 0, state.cur_len), (batch_size, num_beams, 1))
            )

            # Call fast-decoder model on current tokens to get next-position logits.
            # --> [batch * beam, vocab]
            model_outputs = model(input_token, **state.model_kwargs)

            # unflatten beam dimension
            # [batch * beam, vocab] --> [batch, beam, vocab]
            new_logits = unflatten_beam_dim(model_outputs.logits, batch_size, num_beams)
            # Unflatten beam dimension in attention cache arrays
            # {[batch * beam, ...], ...} --> {[batch, beam, ...], ...}
            new_cache = jax.tree_map(
                lambda x: unflatten_beam_dim(x, batch_size, num_beams), model_outputs.past_key_values
            )

            # Gather log probabilities from logits
            candidate_log_probs = jax.nn.log_softmax(new_logits[:, :, 0, :])
            candidate_log_probs = logits_processor(
                flatten_beam_dim(current_sequences), flatten_beam_dim(candidate_log_probs), state.cur_len
            )
            candidate_log_probs = unflatten_beam_dim(candidate_log_probs, batch_size, num_beams)

            # Add new logprobs to existing prefix logprobs.
            # --> [batch, beam, vocab]
            log_probs = candidate_log_probs + jnp.expand_dims(state.current_scores, axis=2)
            # We'll need the vocab size, gather it from the log probability dimension.
            vocab_size = log_probs.shape[2]

            # Each item in batch has num_beams * vocab_size candidate sequences.
            # For each item, get the top 2*k candidates with the highest log-
            # probabilities. We gather the top 2*K beams here so that even if the best
            # K sequences reach EOS simultaneously, we have another K sequences
            # remaining to continue the live beam search.
            beams_to_keep = 2 * num_beams
            # Flatten beam and vocab dimensions.
            flat_log_probs = log_probs.reshape((batch_size, num_beams * vocab_size))
            # Gather the top 2*K scores from _all_ beams.
            # --> [batch, 2*beams], [batch, 2*beams]
            topk_log_probs, topk_indices = lax.top_k(flat_log_probs, k=beams_to_keep)
            # Recover the beam index by floor division.
            topk_beam_indices = topk_indices // vocab_size
            # Gather 2*k top beams.
            # --> [batch, 2*beams, length]
            topk_seq = gather_beams(state.current_sequences, topk_beam_indices, batch_size, beams_to_keep)

            # Append the most probable 2*K token IDs to the top 2*K sequences
            # Recover token id by modulo division and expand Id array for broadcasting.
            # --> [batch, 2*beams, 1]
            topk_ids = jnp.expand_dims(topk_indices % vocab_size, axis=2)
            # Update sequences for the 2*K top-k new sequences.
            # --> [batch, 2*beams, length]
            topk_seq = lax.dynamic_update_slice(topk_seq, topk_ids, (0, 0, state.cur_len + 1))

            # Update LIVE (in-progress) sequences:
            # Did any of these sequences reach an end marker?
            # --> [batch, 2*beams]
            newly_finished = topk_seq[:, :, state.cur_len + 1] == eos_token_id
            # To prevent these newly finished sequences from being added to the LIVE
            # set of active beam search sequences, set their log probs to a very large
            # negative value.
            new_log_probs = topk_log_probs + newly_finished * np.array(-1.0e7)
            # Determine the top k beam indices (from top 2*k beams) from log probs.
            # --> [batch, beams]
            _, new_topk_indices = lax.top_k(new_log_probs, k=num_beams)
            new_topk_indices = jnp.flip(new_topk_indices, axis=1)
            # Gather the top k beams (from top 2*k beams).
            # --> [batch, beams, length], [batch, beams]
            top_alive_seq, top_alive_log_probs = gather_beams(
                [topk_seq, new_log_probs], new_topk_indices, batch_size, num_beams
            )

            # Determine the top k beam indices from the original set of all beams.
            # --> [batch, beams]
            top_alive_indices = gather_beams(topk_beam_indices, new_topk_indices, batch_size, num_beams)
            # With these, gather the top k beam-associated caches.
            # --> {[batch, beams, ...], ...}
            top_alive_cache = gather_beams(new_cache, top_alive_indices, batch_size, num_beams)

            # Update FINISHED (reached end of sentence) sequences:
            # Calculate new seq scores from log probabilities.
            new_scores = topk_log_probs / ((state.cur_len + 1) ** length_penalty)
            # Mask out the still unfinished sequences by adding large negative value.
            # --> [batch, 2*beams]
            new_scores += (~newly_finished) * np.array(-1.0e7)

            # Combine sequences, scores, and flags along the beam dimension and compare
            # new finished sequence scores to existing finished scores and select the
            # best from the new set of beams.

            #TODO(Patrick - add early stopping here)

            finished_seqs = jnp.concatenate([state.sequences, topk_seq], axis=1)  # --> [batch, 3*beams, length]
            finished_scores = jnp.concatenate([state.scores, new_scores], axis=1)  # --> [batch, 3*beams]
            finished_flags = jnp.concatenate([state.is_sent_finished, newly_finished], axis=1)  # --> [batch, 3*beams]
            # --> [batch, beams, length], [batch, beams], [batch, beams]
            top_finished_seq, top_finished_scores, top_finished_flags = gather_topk_beams(
                [finished_seqs, finished_scores, finished_flags], finished_scores, batch_size, num_beams
            )

            model_outputs["past_key_values"] = jax.tree_map(lambda x: flatten_beam_dim(x), top_alive_cache)
            new_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)

            return BeamSearchState(
                cur_len=state.cur_len + 1,
                current_scores=top_alive_log_probs,
                current_sequences=top_alive_seq,
                scores=top_finished_scores,
                sequences=top_finished_seq,
                is_sent_finished=top_finished_flags,
                model_kwargs=new_model_kwargs,
            )

            # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU

        state = beam_search_body_fn(state)

        if not trace:
            state = self._run_loop_in_debug(beam_search_cond_fn, beam_search_body_fn, state)
        else:
            state = lax.while_loop(beam_search_cond_fn, beam_search_body_fn, state)

        # Account for the edge-case where there are no finished sequences for a
        # particular batch item. If so, return live sequences for that batch item.
        # --> [batch]
        none_finished = jnp.any(state.is_sent_finished, axis=1)
        # --> [batch, beams, length]
        sequences = jnp.where(none_finished[:, None, None], state.sequences, state.current_sequences)

        # --> [batch, beams]
        scores = jnp.where(none_finished[:, None], state.scores, state.current_scores)

        # take best beam
        sequences = sequences[:, -1]
        scores = scores[:, -1]

        return FlaxBeamSearchOutput(sequences=sequences, scores=scores)
