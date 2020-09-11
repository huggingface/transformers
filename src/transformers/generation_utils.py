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

from typing import Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from .file_utils import ModelOutput
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

    def get_dist_warpper(self, top_k=None, top_p=None, temperature=None, num_beams=None):
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
                generation_utils_samplers.TopKSampler(k=top_k, min_tokens_to_keep=(2 if num_beams > 1 else 1))
            )
        if top_p is not None:
            warpers.append(
                generation_utils_samplers.TopPSampler(p=top_p, min_tokens_to_keep=(2 if num_beams > 1 else 1))
            )
        if temperature is not None:
            warpers.append(generation_utils_samplers.TemperatureSampler(temperature))
        return warpers

    def get_dist_pre_processor(self, repetition_penalty, no_repeat_ngram_size, bad_words_ids, min_length, eos_token_id):
        """
            This class returns a `ProcessorList` object, that contains all distribution pre processing functions
            that are NOT related to sampling
        """
        # check that args are set if none, fall back to self.config.top_k
        ...

        # instantiate processors list
        processors = ProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if repetition_penalty is not None:
            processors.append(generation_utils_samplers.RepetitionPenaltySampler(penalty=repetition_penalty))
        if no_repeat_ngram_size is not None:
            processors.append(generation_utils_samplers.NoRepeatNGramSampler(no_repeat_ngram_size))
        if bad_words_ids is not None:
            processors.append(generation_utils_samplers.NoBadWordsSampler(bad_words_ids))
        if min_length is not None and eos_token_id is not None:
            processors.append(generation_utils_samplers.MinLengthSampler(min_length, eos_token_id))
        return processors

    def generate(
        self,
        input_ids,
        # ... all sampler arguments
        model_kwargs
    ):

        if self.is_encoder_decoder:
            ...
            # encoder outputs does not have to be touched anymore
            model_kwargs["encoder_outputs"] = encoder_outputs

        pre_processor = self.get_dist_pre_processor(...)

        if do_sample is True:
            dist_warper = self.get_dist_warpper(...)

        if num_beams > 1:
            beam_scorer = generation_beam_search.BeamScorer()  # refactor all important beam search functions out into BeamScorer class
                                                               # could also use different beam scorer classes here

        if do_sample is False and num_beams == 1:
            return self.greedy_search(input_ids, pre_processor, model_kwargs)

        elif do_sample is True and num_beams == 1:
            return self.sample(input_ids, pre_processor, dist_warper, model_kwargs)

        elif do_sample is False and num_beams > 1:
            return self.beam_search(input_ids, pre_processor, beam_scorer, model_kwargs)

        elif do_sample is True and num_beams > 1:
            return self.beam_sample(input_ids, pre_processor, dist_warper, beam_scorer, model_kwargs)

    @torch.no_grad()
    def greedy_search(self, input_ids, pre_processor, model_kwargs):
        unfinished_sents, sent_lengths = init(...)

        # add necessary encoder decoder code
        # ...

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            scores = pre_processor(input_ids, next_token_logits)
            next_token = torch.argmax(scores)

            # add code that transfomers next_token to tokens_to_add
            ...

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            past, attention_mask, unfished_sents, sent_length = update(...)

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # add all post processing functions
        # ...

        return input_ids

    @torch.no_grad()
    def sample(self, input_ids, pre_processor, dist_wrapper, kwargs):
        unfinished_sents, sent_lengths = init(...)

        # add necessary encoder decoder code
        # ...

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            scores = pre_processor(input_ids, next_token_logits)
            scores = dist_warper(input_ids, next_token_logits)
            probs = F.softmax(scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            # add code that transfomers next_token to tokens_to_add
            ...

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            past, attention_mask, unfished_sents, sent_length = update(...)

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # add all post processing functions
        # ...
        return input_ids

    @torch.no_grad()
    def beam_search(self, input_ids, pre_processor, beam_scorer, post_processor, kwargs):
        next_beam_scores, unfinished_sents, sent_lengths = init(...)

        # add necessary encoder decoder code
        # ...

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
            scores = pre_processor(input_ids, next_token_logits)

            # special for beam_decode
            scores = scores + next_beam_scores
            next_scores, next_tokens = torch.topk(scores)

            # this function will factor out all of the beam search code that is currently in `_no_beam_search_...`
            beam_scorer.update(next_scores, next_tokens)

            next_beam_scores = beam_scorer.get_next_scores()
            next_beam_tokens = beam_scorer.get_next_tokens()
            next_beam_idx = beam_scorer.get_next_beam_idx()

            input_ids = torch.cat([input_ids[next_beam_idx, :], next_beam_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            past = self._reorder_cache(next_beam_idx)

            if beam_scorer.is_done():
                break

        # add all post processing functions
        # ...
        return input_ids

    @torch.no_grad()
    def beam_sample(self, input_ids, pre_processor, dist_wrapper, beam_scorer, post_processor, kwargs):
        next_beam_scores, unfinished_sents, sent_lengths = init(...)

        # add necessary encoder decoder code
        # ...

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
            scores = pre_processor(input_ids, next_token_logits)

            # special for beam_decode
            # IMPORTANT: Note how the following line has to be between `pre_precossor` and `dist_warper`
            scores = scores + next_beam_scores
            scores = dist_warper(input_ids, scores)
            next_scores, next_tokens = torch.topk(scores)

            # this function will factor out all of the beam search code that is currently in `_no_beam_search_...`
            beam_scorer.update(next_scores, next_tokens)

            next_beam_scores = beam_scorer.get_next_scores()
            next_beam_tokens = beam_scorer.get_next_tokens()
            next_beam_idx = beam_scorer.get_next_beam_idx()

            input_ids = torch.cat([input_ids[next_beam_idx, :], next_beam_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            past = self._reorder_cache(next_beam_idx)

            if beam_scorer.is_done():
                break

        # add all post processing functions
        # ...
        return input_ids
