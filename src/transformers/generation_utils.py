# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
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

from typing import Iterable, Optional, Tuple

import torch
from torch.nn import functional as F

from .file_utils import ModelOutput
from .generation_utils_beam_search import BeamScorer, BeamSearchBase
from .generation_utils_samplers import (
    MinLengthSampler,
    NoBadWordsSampler,
    NoRepeatNGramSampler,
    RepetitionPenaltySampler,
    TemperatureSampler,
    TopKSampler,
    TopPSampler,
)
from .utils import logging


logger = logging.get_logger(__name__)


class ProcessorList(list):

    """
    This class inherits from list and adds a special `__call__`
    method that call each distribution processing function one by one
    and returns the processed scores
    """

    def __call__(self, input_ids, scores):
        for processor in self:
            scores = processor(input_ids, scores)
        return scores


class GenerationMixin:
    """
    A class contraining all of the functions supporting generation, to be used as a mixin in
    :class:`~transfomers.PreTrainedModel`.
    """

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        Implement in subclasses of :class:`~transfomers.PreTrainedModel` for custom behavior to prepare inputs in the
        generate method.
        """
        return {"input_ids": input_ids}

    def adjust_logits_during_generation(self, logits, **kwargs):
        """
        Implement in subclasses of :class:`~transfomers.PreTrainedModel` for custom behavior to adjust the logits in
        the generate method.
        """
        return logits

    @staticmethod
    def _reorder_cache(past: Tuple, beam_idx: torch.Tensor) -> Tuple[torch.Tensor]:
        return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)

    def get_dist_warpper(self, top_k=None, top_p=None, temperature=None, num_beams=None):
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        temperature = temperature if temperature is not None else self.config.temperature
        """
            This class returns a `ProcessorList` object, that contains all distribution pre processing functions
            that are ONLY related to sampling
        """
        # check that args are set if none, fall back to self.config.top_k

        # instantiate warpers list
        warpers = ProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if top_k is not None and top_k != 0:
            warpers.append(TopKSampler(k=top_k, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        if top_p is not None:
            warpers.append(TopPSampler(probability=top_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        if temperature is not None:
            warpers.append(TemperatureSampler(temperature))
        return warpers

    def get_dist_pre_processor(
        self, repetition_penalty, no_repeat_ngram_size, bad_words_ids, min_length, eos_token_id
    ):

        # verify pre-prossed tokens
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        min_length = min_length if min_length is not None else self.config.min_length
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # instantiate processors list
        processors = ProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if repetition_penalty is not None:
            processors.append(RepetitionPenaltySampler(penalty=repetition_penalty))
        if no_repeat_ngram_size is not None:
            processors.append(NoRepeatNGramSampler(no_repeat_ngram_size))
        if bad_words_ids is not None:
            processors.append(NoBadWordsSampler(bad_words_ids))
        if min_length is not None and eos_token_id is not None:
            processors.append(MinLengthSampler(min_length, eos_token_id))
        return processors

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs
    ) -> torch.LongTensor:

        # set init values
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        max_length = max_length if max_length is not None else self.config.max_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # TODO(PVP): check if the following ifs should stay here or whether they should be moved into a new function
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
            )
            pad_token_id = eos_token_id

        if input_ids is None:
            if pad_token_id is None:
                raise ValueError("`pad_token_id` has to be defined when no `input_ids` are provided.")
            input_ids = torch.ones((1, 1), dtype=torch.long, device=next(self.parameters()).device) * pad_token_id

        if self.config.is_encoder_decoder:
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
            }
            model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)
            batch_size = input_ids.shape[0]

            if "decoder_input_ids" in model_kwargs:
                input_ids = model_kwargs["decoder_input_ids"]
            else:
                decoder_start_token_id = self.get_decoder_start_token_id(bos_token_id)
                input_ids = (
                    torch.ones((batch_size, 1), dtype=input_ids.dtype, device=input_ids.device)
                    * decoder_start_token_id
                )

        # determine generation model
        is_greedy_gen_mode = (num_beams == 1) and do_sample is False
        is_sample_gen_mode = (num_beams == 1) and do_sample is True
        is_beam_gen_mode = (num_beams > 1) and do_sample is False
        is_beam_sample_gen_mode = (num_beams > 1) and do_sample is True

        # get distribution pre_processing samplers
        pre_processor = self.get_dist_pre_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            eos_token_id=eos_token_id,
        )

        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # greedy search
            return self.greedy_search(
                input_ids,
                pre_processor=pre_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # get probability distribution warper
            dist_warper = self.get_dist_warpper(top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams)

            # expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)

            # interleave with `num_return_sequences`
            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = model_kwargs["attention_mask"].repeat_interleave(
                    num_return_sequences, dim=0
                )

            if self.config.is_encoder_decoder:
                assert "encoder_outputs" in model_kwargs
                encoder_outputs = model_kwargs["encoder_outputs"]
                encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.repeat_interleave(
                    num_return_sequences, dim=0
                )

            # sample
            return self.sample(
                input_ids,
                pre_processor=pre_processor,
                dist_warper=dist_warper,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            batch_size = input_ids.shape[0]

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

            beam_scorer = BeamSearchBase(
                batch_size,
                max_length,
                num_beams,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_return_sequences=num_return_sequences,
            )

            # interleave with `num_beams`
            input_ids = input_ids.repeat_interleave(num_beams, dim=0)

            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = model_kwargs["attention_mask"].repeat_interleave(num_beams, dim=0)

            if self.config.is_encoder_decoder:
                assert "encoder_outputs" in model_kwargs
                encoder_outputs = model_kwargs["encoder_outputs"]
                encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.repeat_interleave(
                    num_beams, dim=0
                )

            return self.beam_search(
                input_ids,
                beam_scorer,
                pre_processor=pre_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            # DELETE ME LATER WHEN IMPLEMENTED
            return torch.tensor([8 * [0]])

    @torch.no_grad()
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        pre_processor: Optional[ProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **model_kwargs
    ):
        # init values
        pre_processor = pre_processor if pre_processor is not None else ProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # init sequence length tensors
        sequence_lengths, unfinished_sequences, cur_len = self.init_sequence_length(input_ids, max_length)

        while cur_len < max_length:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            scores = pre_processor(input_ids, next_token_logits)

            # argmax
            next_tokens = torch.argmax(scores)

            # add code that transfomers next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # update sequence length
            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = self.update_sequence_lengths(
                    sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
                )

            # update model kwargs
            model_kwargs = self.update_model_kwargs(outputs, model_kwargs)

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break

            # increase cur_len
            cur_len = cur_len + 1

        return input_ids

    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.LongTensor,
        pre_processor: Optional[ProcessorList] = None,
        dist_warper: Optional[ProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **model_kwargs
    ):
        # init values
        pre_processor = pre_processor if pre_processor is not None else ProcessorList()
        dist_warper = dist_warper if dist_warper is not None else ProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # init sequence length tensors
        sequence_lengths, unfinished_sequences, cur_len = self.init_sequence_length(input_ids, max_length)

        # auto-regressive generation
        while cur_len < max_length:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            scores = pre_processor(input_ids, next_token_logits)
            scores = dist_warper(input_ids, next_token_logits)

            # sample
            probs = F.softmax(scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # add code that transfomers next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            cur_len = cur_len + 1

            # update sequence length
            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = self.update_sequence_lengths(
                    sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
                )

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break

            # update model kwargs
            model_kwargs = self.update_model_kwargs(outputs, model_kwargs)

        return input_ids

    @staticmethod
    def init_sequence_length(input_ids, max_length):
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        sequence_lengths = input_ids.new(input_ids.shape[0]).fill_(max_length)

        cur_len = input_ids.shape[-1]
        return sequence_lengths, unfinished_sequences, cur_len

    @staticmethod
    def update_sequence_lengths(sequence_lengths, unfinished_sequences, cur_len, is_eos_in_next_token):
        # check if sentence is not finished yet
        is_sent_unfinished = unfinished_sequences.mul(is_eos_in_next_token.long()).bool()

        # update sentence length
        sequence_lengths = sequence_lengths.masked_fill(is_sent_unfinished, cur_len)
        unfinished_sequences = unfinished_sequences.mul((~is_eos_in_next_token).long())
        return sequence_lengths, unfinished_sequences

    @staticmethod
    def update_model_kwargs(outputs, model_kwargs):
        # update past
        model_kwargs["past"] = outputs.past_key_values if "past_key_values" in outputs else None
        model_kwargs["past"] = outputs.mems if "mems" in outputs else None

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return model_kwargs

    @torch.no_grad()
    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        pre_processor: Optional[ProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **model_kwargs
    ):

        # init values
        pre_processor = pre_processor if pre_processor is not None else ProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        assert num_beams * batch_size == batch_beam_size, "TODO ..."

        if "attention_mask" in model_kwargs:
            model_kwargs["attention_mask"] = model_kwargs["attention_mask"].repeat_interleave(num_beams, dim=0)

        if self.config.is_encoder_decoder:
            assert "encoder_outputs" in model_kwargs
            encoder_outputs = model_kwargs["encoder_outputs"]
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.repeat_interleave(
                num_beams, dim=0
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            # adjust tokens for Bart, *e.g.*
            next_token_scores = self.adjust_logits_during_generation(
                next_token_scores, cur_len=cur_len, max_length=max_length
            )

            next_token_scores = pre_processor(input_ids, next_token_scores)

            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // self.config.vocab_size
            next_tokens = next_tokens // self.config.vocab_size

            # stateless
            beam_scores, beam_next_tokens, beam_idx = beam_scorer.update_beams(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            model_kwargs = self.update_model_kwargs(outputs, model_kwargs)
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if beam_scorer.is_done():
                break

        decoded = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        return decoded

    @torch.no_grad()
    def beam_sample(self, input_ids, pre_processor, dist_wrapper, beam_scorer, post_processor, kwargs):
        raise NotImplementedError()


#        next_beam_scores, unfinished_sequences, sequence_lengths = init(...)

# add necessary encoder decoder code
# ...

#        while cur_len < max_length:
#            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
#
#            outputs = self(**model_inputs, return_dict=True)
#            next_token_logits = outputs.logits[:, -1, :]
#            scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
#            scores = pre_processor(input_ids, next_token_logits)
#
# special for beam_decode
# IMPORTANT: Note how the following line has to be between `pre_precossor` and `dist_warper`
#            scores = scores + next_beam_scores
#            scores = dist_warper(input_ids, scores)
#            scores, next_tokens = torch.topk(scores)
#
# this function will factor out all of the beam search code that is currently in `_no_beam_search_...`
#            beam_scorer.update(scores, next_tokens)
#
#            next_beam_scores = beam_scorer.get_scores()
#            next_beam_tokens = beam_scorer.get_next_tokens()
#            next_beam_idx = beam_scorer.get_next_beam_idx()
#
#            input_ids = torch.cat([input_ids[next_beam_idx, :], next_beam_tokens.unsqueeze(-1)], dim=-1)
#            cur_len = cur_len + 1
#
#            past = self._reorder_cache(next_beam_idx)
#
#            if beam_scorer.is_done():
#                break
#
# add all post processing functions
# ...
#        return input_ids


def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        logits = TopKSampler(k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(None, logits)

    if 0 <= top_p <= 1.0:
        logits = TopPSampler(probability=top_p, min_tokens_to_keep=min_tokens_to_keep)(None, logits)

    return logits
