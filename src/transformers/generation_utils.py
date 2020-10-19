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

from typing import Optional, Iterable

import torch
from torch.nn import functional as F

from .utils import logging
from .generation_utils_samplers import TopPSampler, TopKSampler, NoBadWordsSampler, NoRepeatNGramSampler, TemperatureSampler, MinLengthSampler, RepetitionPenaltySampler


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
            warpers.append(
                TopKSampler(k=top_k, min_tokens_to_keep=(2 if num_beams > 1 else 1))
            )
        if top_p is not None:
            warpers.append(
                TopPSampler(p=top_p, min_tokens_to_keep=(2 if num_beams > 1 else 1))
            )
        if temperature is not None:
            warpers.append(TemperatureSampler(temperature))
        return warpers

    def get_dist_pre_processor(self, repetition_penalty, no_repeat_ngram_size, bad_words_ids, min_length, eos_token_id):

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
        decoder_input_ids: Optional[torch.LongTensor] = None,
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
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs
    ) -> torch.LongTensor:

        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_return_sequences = num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        is_beam_search = (num_beams > 1)

        if self.is_encoder_decoder:
            raise NotImplementedError()

        pre_processor = self.get_dist_pre_processor(repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size, bad_words_ids=bad_words_ids, min_length=min_length, eos_token_id=eos_token_id)

        if not is_beam_search:
            # single beam generation
            if do_sample is False:
                if num_return_sequences > 1:
                    raise ValueError(f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search.")
                return self.greedy_search(input_ids, pre_processor=pre_processor, max_length=max_length, pad_token_id=pad_token_id, eos_token_id=eos_token_id, **model_kwargs)
            else:
                dist_warper = self.get_dist_warpper(top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams)
                input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
                return self.sample(input_ids, pre_processor=pre_processor, dist_warper=dist_warper, **model_kwargs)
        else:
            raise NotImplementedError()
#            beam_scorer = generation_beam_search.BeamScorer(num_beams)  # refactor all important beam search functions out into BeamScorer class
#            if do_sample is False:
#                return self.beam_search(input_ids, pre_processor, beam_scorer, model_kwargs)
#            else:
#                dist_warper = self.get_dist_warpper(top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams)
#                return self.beam_sample(input_ids, pre_processor, dist_warper, beam_scorer, model_kwargs)

    @torch.no_grad()
    def greedy_search(self, input_ids: torch.LongTensor, pre_processor: Optional[ProcessorList] = None, max_length: Optional[int] = None, pad_token_id: Optional[int] = None, eos_token_id: Optional[int] = None, **model_kwargs):
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
                sequence_lengths, unfinished_sequences = self.update_sequence_lengths(sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id)

            # update model kwargs
            model_kwargs = self.update_model_kwargs(outputs, model_kwargs)

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break

            # increase cur_len
            cur_len = cur_len + 1

        return input_ids

    @torch.no_grad()
    def sample(self, input_ids: torch.LongTensor, pre_processor: Optional[ProcessorList] = None, dist_warper: Optional[ProcessorList] = None, max_length: Optional[int] = None, pad_token_id: Optional[int] = None, eos_token_id: Optional[int] = None, **model_kwargs):
        # init values
        pre_processor = pre_processor if pre_processor is not None else ProcessorList()
        dist_warper = dist_warper if dist_warper is not None else ProcessorList()
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

            # update sequence length
            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = self.update_sequence_lengths(sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id)

            # update model kwargs
            model_kwargs = self.update_model_kwargs(outputs, model_kwargs)

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break

            # increase cur_len
            cur_len = cur_len + 1

        return input_ids

    @staticmethod
    def init_sequence_length(input_ids, max_length):
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        sequence_lengths = input_ids.new(input_ids.shape[-1]).fill_(max_length)

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
            model_kwargs["attention_mask"] = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

        return model_kwargs

    @torch.no_grad()
    def beam_search(self, input_ids, pre_processor, beam_scorer, post_processor, kwargs):
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
#            scores = scores + next_beam_scores
#            next_scores, next_tokens = torch.topk(scores)
#
            # this function will factor out all of the beam search code that is currently in `_no_beam_search_...`
#            beam_scorer.update(next_scores, next_tokens)
#
#            next_beam_scores = beam_scorer.get_next_scores()
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

        # add all post processing functions
        # ...
#        return input_ids

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
#            next_scores, next_tokens = torch.topk(scores)
#
            # this function will factor out all of the beam search code that is currently in `_no_beam_search_...`
#            beam_scorer.update(next_scores, next_tokens)
#
#            next_beam_scores = beam_scorer.get_next_scores()
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
