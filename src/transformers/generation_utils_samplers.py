# coding=utf-8
# Copyright 2020 HuggingFace Inc.
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

from abc import ABC
from typing import Iterable, List

import torch
from torch.nn import functional as F


class Sampler(ABC):
    """Abstract base class for all samplers which are probability distribution warps performed during generation."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Torch method for warping a distribution, defaults to `warp`'s implementation."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class MinLengthSampler(Sampler):
    """Sampler enforcing a min-length by setting EOS probability to 0."""

    def __init__(self, min_length: int, eos_token_id: int):
        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            scores[:, self.eos_token_id] = -float("inf")
        return scores


class TemperatureSampler(Sampler):
    """Sampler for temperature (exponential scaling output probability distribution)."""

    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return scores / self.temperature


class RepetitionPenaltySampler(Sampler):
    """Sampler enforcing an exponential penalty on repeated sequences."""

    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for i in range(scores.shape[0]):
            for previous_token in set(input_ids[i].tolist()):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if scores[i, previous_token] < 0:
                    scores[i, previous_token] *= self.penalty
                else:
                    scores[i, previous_token] /= self.penalty
        return scores


class TopPSampler(Sampler):
    """Sampler that performs top-p, i.e. restricting to top tokens summing to probability <= probability."""

    def __init__(self, probability: float = 1.0, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        assert 0 <= probability <= 1.0, "P must be a probability between 0 and 1"
        self.probability = probability
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.probability
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores[indices_to_remove] = self.filter_value
        return scores


class TopKSampler(Sampler):
    """Sampler that performs top-k, i.e. restricting to the k highest probability elements."""

    def __init__(self, k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        assert k > 0, "Must specify a positive Top-K value"

        self.k = k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(max(self.k, self.min_tokens_to_keep), scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores[indices_to_remove] = self.filter_value
        return scores


class NoRepeatNGramSampler(Sampler):
    """Sampler that enforces no repetition of n-grams.
    See Fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345."""

    def __init__(self, ngram_size: int):
        self.ngram_size = ngram_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = self._calc_banned_ngram_tokens(input_ids, num_batch_hypotheses, cur_len)

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores

    def _calc_banned_ngram_tokens(
        self, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int
    ) -> List[Iterable[int]]:
        """Copied from fairseq for no_repeat_ngram in beam_search"""
        if cur_len + 1 < self.ngram_size:
            # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            return [[] for _ in range(num_hypos)]
        generated_ngrams = [{} for _ in range(num_hypos)]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(*[gen_tokens[i:] for i in range(self.ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

        def _get_generated_ngrams(hypo_idx):
            # Before decoding the next token, prevent decoding of ngrams that have already appeared
            start_idx = cur_len + 1 - self.ngram_size
            ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
            return generated_ngrams[hypo_idx].get(ngram_idx, [])

        banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
        return banned_tokens


class NoBadWordsSampler(Sampler):
    """Sampler that enforces that specified sequences will never be sampled."""

    def __init__(self, bad_words_ids: Iterable[Iterable[int]], eos_token_id: int):
        self.bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [eos_token_id], bad_words_ids))
        for banned_token_seq in self.bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        banned_tokens = self._calc_banned_bad_words_ids(input_ids)
        scores = self._set_scores_to_inf_for_banned_tokens(scores, banned_tokens)

        return scores

    def _tokens_match(self, prev_tokens, tokens) -> bool:
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_tokens):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    def _calc_banned_bad_words_ids(self, prev_input_ids: Iterable[int]) -> Iterable[int]:
        banned_tokens = []
        for prev_input_ids_slice in prev_input_ids:
            banned_tokens_slice = []
            for banned_token_seq in self.bad_words_ids:
                if self._tokens_match(prev_input_ids_slice, banned_token_seq[:-1]) is False:
                    # if tokens do not match continue
                    continue

                banned_tokens_slice.append(banned_token_seq[-1])

            banned_tokens.append(banned_tokens_slice)

        return banned_tokens

    def _set_scores_to_inf_for_banned_tokens(self, scores: torch.Tensor, banned_tokens: List[List[int]]) -> None:
        """Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be
        a list of list of banned tokens to ban in the format [[batch index, vocabulary position],...]
            Args:
                scores: logits distribution of shape (batch size, vocabulary size)
                banned_tokens: list of list of tokens to ban of length (batch_size)
        """
        banned_mask_list = []
        for idx, batch_banned_tokens in enumerate(banned_tokens):
            for token in batch_banned_tokens:
                banned_mask_list.append([idx, token])
        if not banned_mask_list:
            return scores
        banned_mask = torch.LongTensor(banned_mask_list)
        indices = torch.ones(len(banned_mask))
        # A sparse tensor is generated from a list of coordinates: [[0, 1], [0, 2], [2, 0]]. A conversion to dense tensor generates:
        # [ 0  1  1 ]
        # [ 0  0  0 ]
        # [ 1  0  0 ]

        banned_mask = (
            torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
        )
        scores.masked_fill_(banned_mask, -float("inf"))
        return scores
