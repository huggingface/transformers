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
import numpy as np
from typing import Iterable, Union, TypeVar, List

from .file_utils import is_tf_available, is_torch_available

TorchTensor = TypeVar('TorchTensor')
TFTensor = TypeVar('TFTensor')

if is_torch_available():
    import torch
    from torch.nn import functional as F
    TorchTensor = torch.Tensor

if is_tf_available():
    import tensorflow as tf
    TFTensor = tf.Tensor

AnyTensor = Union[TorchTensor, TFTensor]

def shape_list(x: TFTensor) -> List[int]:
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def set_tensor_by_indices_to_value(tensor, indices, value):
    # create value_tensor since tensor value assignment is not possible in TF
    value_tensor = tf.zeros_like(tensor) + value
    return tf.where(indices, value_tensor, tensor)

def scatter_values_on_batch_indices(values, batch_indices):
    shape = shape_list(batch_indices)
    # broadcast batch dim to shape
    broad_casted_batch_dims = tf.reshape(tf.broadcast_to(tf.expand_dims(tf.range(shape[0]), axis=-1), shape), [1, -1])
    # transform batch_indices to pair_indices
    pair_indices = tf.transpose(tf.concat([broad_casted_batch_dims, tf.reshape(batch_indices, [1, -1])], 0))
    # scatter values to pair indices
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), shape)

class GenerationSampler(ABC):
    def warp(self, input_ids: AnyTensor, next_token_logits: AnyTensor) -> AnyTensor:
        raise NotImplementedError("Warp called on base class")

    def warp_torch(self, input_ids: TorchTensor, next_token_logits: TorchTensor) -> TorchTensor:
        return self.warp(input_ids, next_token_logits)

    def warp_tf(self, input_ids: TFTensor, next_token_logits: TFTensor) -> TFTensor:
        return self.warp(input_ids, next_token_logits)
    

class IdentitySampler(GenerationSampler):
    def warp(self, input_ids: AnyTensor, next_token_logits: AnyTensor) -> AnyTensor:
        return next_token_logits


class CompositionSampler(GenerationSampler):
    def __init__(self, samplers: Iterable[GenerationSampler]):
        self.samplers = list(samplers)

    def warp_torch(self, input_ids: TorchTensor, next_token_logits: TorchTensor) -> TorchTensor:
        for warp in self.samplers:
            next_token_logits = warp.warp_torch(input_ids, next_token_logits)
        return next_token_logits

    def warp_tf(self, input_ids: TFTensor, next_token_logits: TFTensor) -> TFTensor:
        for warp in self.samplers:
            next_token_logits = warp.warp_tf(input_ids, next_token_logits)
        return next_token_logits

class MinLengthSampler(GenerationSampler):
    def __init__(self, min_length: int, eos_token_id: int):
        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def warp_torch(self, input_ids: TorchTensor, next_token_logits: TorchTensor) -> TorchTensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            next_token_logits[:, self.eos_token_id] = -float("inf")
        return next_token_logits

    def warp_tf(self, input_ids: TFTensor, next_token_logits: TFTensor) -> TFTensor:
        cur_len = shape_list(input_ids)[-1]
        vocab_size = shape_list(next_token_logits)[-1]
        batch_size = shape_list(next_token_logits)[0]
        if cur_len < self.min_length:
            # create eos_token_id boolean mask
            is_token_logit_eos_token = tf.convert_to_tensor(
                [True if token is self.eos_token_id else False for token in range(vocab_size)], dtype=tf.bool
            )
            eos_token_indices_mask = tf.broadcast_to(is_token_logit_eos_token, [batch_size, vocab_size])

            next_token_logits = set_tensor_by_indices_to_value(
                next_token_logits, eos_token_indices_mask, -float("inf")
            )
        return next_token_logits

class TemperatureSampler(GenerationSampler):
    def __init__(self, temperature: float):
        self.temperature = temperature

    def warp(self, input_ids: AnyTensor, next_token_logits: AnyTensor) -> AnyTensor:
        return next_token_logits / self.temperature
    

class RepetitionPenaltySampler(GenerationSampler):
    def __init__(self, penalty: float):
        self.penalty = penalty

    def warp_torch(self, input_ids: TorchTensor, next_token_logits: TorchTensor) -> TorchTensor:
        # TODO: verify this works for beam search
        for i in range(next_token_logits.shape[0]):
            for previous_token in set(input_ids[i].tolist()):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if next_token_logits[i, previous_token] < 0:
                    next_token_logits[i, previous_token] *= self.penalty
                else:
                    next_token_logits[i, previous_token] /= self.penalty
        return next_token_logits

    def warp_tf(self, input_ids: TFTensor, next_token_logits: TFTensor) -> TFTensor:
        next_token_logits_penalties = self._tf_create_next_token_logits_penalties(
            input_ids, next_token_logits
        )
        next_token_logits = tf.math.multiply(next_token_logits, next_token_logits_penalties)
        return next_token_logits

    def _tf_create_next_token_logits_penalties(self, input_ids: TFTensor, logits: TFTensor) -> TFTensor:
        # create logit penalties for already seen input_ids
        token_penalties = np.ones(shape_list(logits))
        prev_input_ids = [np.unique(input_id) for input_id in input_ids.numpy()]
        for i, prev_input_id in enumerate(prev_input_ids):
            logit_penalized = logits[i].numpy()[prev_input_id]
            logit_penalties = np.zeros(logit_penalized.shape)
            # if previous logit score is < 0 then multiply repetition penalty else divide
            logit_penalties[logit_penalized < 0] = self.penalty
            logit_penalties[logit_penalized > 0] = 1 / self.penalty
            np.put(token_penalties[i], prev_input_id, logit_penalties)
        return tf.convert_to_tensor(token_penalties, dtype=tf.float32)


class TopPSampler(GenerationSampler):
    def __init__(self, p: float = 1.0, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        assert 0 <= p <= 1.0, "P must be a probability between 0 and 1"
        self.p = p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def warp_torch(self, input_ids: TorchTensor, next_token_logits: TorchTensor) -> TorchTensor:
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.p
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :self.min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        next_token_logits[indices_to_remove] = self.filter_value
        return next_token_logits

    def warp_tf(self, input_ids: TFTensor, next_token_logits: TFTensor) -> TFTensor:
        sorted_indices = tf.argsort(next_token_logits, direction="DESCENDING")
        sorted_logits = tf.gather(
            next_token_logits, sorted_indices, axis=-1, batch_dims=1
        )  # expects logits to be of dim (batch_size, vocab_size)

        cumulative_probs = tf.math.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.p

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove = tf.concat(
                [
                    tf.zeros_like(sorted_indices_to_remove[:, :self.min_tokens_to_keep]),
                    sorted_indices_to_remove[:, self.min_tokens_to_keep:],
                ],
                -1,
            )

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove = tf.roll(sorted_indices_to_remove, 1, axis=-1)
        sorted_indices_to_remove = tf.concat(
            [tf.zeros_like(sorted_indices_to_remove[:, :1]), sorted_indices_to_remove[:, 1:]], -1,
        )
        # scatter sorted tensors to original indexing
        indices_to_remove = scatter_values_on_batch_indices(sorted_indices_to_remove, sorted_indices)
        next_token_logits = set_tensor_by_indices_to_value(next_token_logits, indices_to_remove, self.filter_value)

        return next_token_logits


class TopKSampler(GenerationSampler):
    def __init__(self, k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        assert k > 0, "Must specify a positive Top-K value"

        self.k = k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def warp_torch(self, input_ids: TorchTensor, next_token_logits: TorchTensor) -> TorchTensor:
        top_k = min(max(self.k, self.min_tokens_to_keep), next_token_logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
        next_token_logits[indices_to_remove] = self.filter_value
        return next_token_logits
        
    def warp_tf(self, input_ids: TFTensor, next_token_logits: TFTensor) -> TFTensor:
        logits_shape = shape_list(next_token_logits)
        top_k = min(max(self.k, self.min_tokens_to_keep), logits_shape[-1])  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = next_token_logits < tf.math.top_k(next_token_logits, k=top_k)[0][..., -1, None]
        next_token_logits = set_tensor_by_indices_to_value(next_token_logits, indices_to_remove, self.filter_value)
        return next_token_logits


class NoRepeatNGramSampler(GenerationSampler):
    # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
    def __init__(self, ngram_size: int):
        self.ngram_size = ngram_size
    
    def warp_torch(self, input_ids: TorchTensor, next_token_logits: TorchTensor) -> TorchTensor:
        num_batch_hypotheses = next_token_logits.shape[0]
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = self._torch_calc_banned_ngram_tokens(input_ids, num_batch_hypotheses, cur_len)

        for i, banned_tokens in enumerate(banned_batch_tokens):
            next_token_logits[i, banned_tokens] = -float("inf")
        
        return next_token_logits

    def warp_tf(self, input_ids: TFTensor, next_token_logits: TFTensor) -> TFTensor:
        num_batch_hypotheses = shape_list(next_token_logits)[0]
        vocab_size = shape_list(next_token_logits)[-1]
        cur_len = shape_list(input_ids)[-1]
        banned_tokens = self._tf_calc_banned_ngram_tokens(input_ids, num_batch_hypotheses, cur_len)
        banned_tokens_indices_mask = []
        for banned_tokens_slice in banned_tokens:
            banned_tokens_indices_mask.append(
                [True if token in banned_tokens_slice else False for token in range(vocab_size)]
            )

        next_token_logits = set_tensor_by_indices_to_value(
            next_token_logits, tf.convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool), -float("inf")
        )
        return next_token_logits

    def _torch_calc_banned_ngram_tokens(self, prev_input_ids: TorchTensor, num_hypos: int, cur_len: int) -> List[Iterable[int]]:
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

    def _tf_calc_banned_ngram_tokens(self, prev_input_ids: TFTensor, num_hypos, cur_len) -> List[Iterable[int]]:
        # Copied from fairseq for no_repeat_ngram in beam_search"""
        if cur_len + 1 < self.ngram_size:
            # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            return [[] for _ in range(num_hypos)]
        generated_ngrams = [{} for _ in range(num_hypos)]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].numpy().tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(*[gen_tokens[i:] for i in range(self.ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

        def _get_generated_ngrams(hypo_idx):
            # Before decoding the next token, prevent decoding of ngrams that have already appeared
            start_idx = cur_len + 1 - self.ngram_size 
            ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].numpy().tolist())
            return generated_ngrams[hypo_idx].get(ngram_idx, [])

        banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
        return banned_tokens


class NoBadWordsSampler(GenerationSampler):
    def __init__(self, bad_words_ids: Iterable[int]):
        self.bad_words_ids = list(bad_words_ids)
        for banned_token_seq in self.bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )
    
    def warp_torch(self, input_ids: TorchTensor, next_token_logits: TorchTensor) -> TorchTensor:
        banned_tokens = self._torch_calc_banned_bad_words_ids(input_ids)
        for i, banned_tokens in enumerate(banned_tokens):
            next_token_logits[i, banned_tokens] = -float("inf")
        return next_token_logits
    
    def warp_tf(self, input_ids: TFTensor, next_token_logits: TFTensor) -> TFTensor:
        vocab_size = shape_list(next_token_logits)[-1]
        banned_tokens = self._tf_calc_banned_bad_words_ids(input_ids)

        banned_tokens_indices_mask = []
        for banned_tokens_slice in banned_tokens:
            banned_tokens_indices_mask.append(
                [True if token in banned_tokens_slice else False for token in range(vocab_size)]
            )

        next_token_logits = set_tensor_by_indices_to_value(
            next_token_logits, tf.convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool), -float("inf")
        )
        return next_token_logits

    def _tokens_match(self, prev_input_ids, prev_tokens, tokens) -> bool:
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    def _torch_calc_banned_bad_words_ids(self, prev_input_ids: Iterable[int]) -> Iterable[int]:
        banned_tokens = []
        for prev_input_ids_slice in prev_input_ids:
            banned_tokens_slice = []
            for banned_token_seq in self.bad_words_ids:
                if self._tokens_match(prev_input_ids, prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                    # if tokens do not match continue
                    continue

                banned_tokens_slice.append(banned_token_seq[-1])

            banned_tokens.append(banned_tokens_slice)

        return banned_tokens

    def _tf_calc_banned_bad_words_ids(self, prev_input_ids: Iterable[int]) -> Iterable[int]:
        banned_tokens = []
        for prev_input_ids_slice in prev_input_ids:
            banned_tokens_slice = []
            for banned_token_seq in self.bad_words_ids:
                if self._tokens_match(prev_input_ids, prev_input_ids_slice.numpy().tolist(), banned_token_seq[:-1]) is False:
                    # if tokens do not match continue
                    continue

                banned_tokens_slice.append(banned_token_seq[-1])

            banned_tokens.append(banned_tokens_slice)

        return banned_tokens