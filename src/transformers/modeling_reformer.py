# coding=utf-8
# Copyright 2020 The Trax Authors and The HuggingFace Inc. team.
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
"""PyTorch REFORMER model. """

import logging
import sys
from collections import namedtuple
from functools import reduce
from operator import mul

import numpy as np
import torch
from torch import nn
from torch.autograd.function import Function
from torch.nn import CrossEntropyLoss

from .activations import gelu, gelu_fast, gelu_new, swish
from .configuration_reformer import ReformerConfig
from .file_utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
)
from .modeling_utils import PreTrainedModel, apply_chunking_to_forward


logger = logging.getLogger(__name__)

_TOKENIZER_FOR_DOC = "ReformerTokenizer"

REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/reformer-crime-and-punishment",
    "google/reformer-enwik8",
    # See all Reformer models at https://huggingface.co/models?filter=reformer
]


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {
    "gelu": gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "mish": mish,
}


# Define named tuples for nn.Modules here
LSHSelfAttentionOutput = namedtuple("LSHSelfAttentionOutput", ["hidden_states", "attention_probs", "buckets"])
LocalSelfAttentionOutput = namedtuple("LocalSelfAttentionOutput", ["hidden_states", "attention_probs"])
AttentionOutput = namedtuple("AttentionOutput", ["hidden_states", "attention_probs", "buckets"])
ReformerOutput = namedtuple("ReformerOutput", ["hidden_states", "attn_output", "attention_probs", "buckets"])
ReformerBackwardOutput = namedtuple(
    "ReformerBackwardOutput", ["attn_output", "hidden_states", "grad_attn_output", "grad_hidden_states"]
)
ReformerEncoderOutput = namedtuple("ReformerEncoderOutput", ["hidden_states", "all_hidden_states", "all_attentions"])


def _get_least_common_mult_chunk_len(config):
    attn_types = config.attn_layers
    attn_types_set = set(attn_types)
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":
        return config.lsh_attn_chunk_length
    elif len(attn_types_set) == 1 and attn_types[0] == "local":
        return config.local_attn_chunk_length
    elif len(attn_types_set) == 2 and attn_types_set == set(["lsh", "local"]):
        return np.lcm(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    else:
        raise NotImplementedError(
            "Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {}. Select attn layer types from ['lsh', 'local'] only.".format(
                config.attn_layers
            )
        )


class AxialPositionEmbeddings(nn.Module):
    """Constructs axial position embeddings. Useful for very long input
    sequences to save memory and time.
    """

    def __init__(self, config):
        super().__init__()
        self.axial_pos_shape = config.axial_pos_shape
        self.axial_pos_embds_dim = config.axial_pos_embds_dim
        self.dropout = config.hidden_dropout_prob

        self.least_common_mult_chunk_length = _get_least_common_mult_chunk_len(config)
        self.weights = nn.ParameterList()

        assert (
            sum(self.axial_pos_embds_dim) == config.hidden_size
        ), "Make sure that config.axial_pos_embds factors: {} sum to config.hidden_size: {}".format(
            self.axial_pos_embds_dim, config.hidden_size
        )

        # create weights
        for axis, axial_pos_embd_dim in enumerate(self.axial_pos_embds_dim):
            # create expanded shapes
            ax_shape = [1] * len(self.axial_pos_shape)
            ax_shape[axis] = self.axial_pos_shape[axis]
            ax_shape = tuple(ax_shape) + (axial_pos_embd_dim,)

            # create tensor and init
            self.weights.append(nn.Parameter(torch.ones(ax_shape, dtype=torch.float32)))

    def forward(self, position_ids):
        # broadcast weights to correct shape
        batch_size = position_ids.shape[0]
        sequence_length = position_ids.shape[1]

        broadcasted_weights = [
            weight.expand((batch_size,) + self.axial_pos_shape + weight.shape[-1:]) for weight in self.weights
        ]

        if self.training is True:
            assert (
                reduce(mul, self.axial_pos_shape) == sequence_length
            ), "If training, make sure that config.axial_pos_shape factors: {} multiply to sequence length. Got prod({}) != sequence_length: {}. You might want to consider padding your sequence length to {} or changing config.axial_pos_shape.".format(
                self.axial_pos_shape, self.axial_pos_shape, sequence_length, reduce(mul, self.axial_pos_shape)
            )
            if self.dropout > 0:
                weights = torch.cat(broadcasted_weights, dim=-1)
                # permute weights so that 2D correctly drops dims 1 and 2
                transposed_weights = weights.transpose(2, 1)
                # drop entire matrix of last two dims (prev dims 1 and 2)
                dropped_transposed_weights = nn.functional.dropout2d(
                    transposed_weights, p=self.dropout, training=self.training
                )
                dropped_weights = dropped_transposed_weights.transpose(2, 1)

                position_encodings = torch.reshape(dropped_weights, (batch_size, sequence_length, -1))

            else:
                position_encodings = torch.cat(
                    [torch.reshape(weight, (batch_size, sequence_length, -1)) for weight in broadcasted_weights],
                    dim=-1,
                )

        else:
            assert (
                reduce(mul, self.axial_pos_shape) >= sequence_length
            ), "Make sure that config.axial_pos_shape factors: {} multiply at least to max(sequence_length, least_common_mult_chunk_length): max({}, {})".format(
                self.axial_pos_shape, sequence_length, self.least_common_mult_chunk_length,
            )

            # compute how many columns are needed
            required_pos_encodings_columns = -(-sequence_length // self.axial_pos_shape[1])

            # cut to columns that are needed
            position_encodings = torch.cat(
                [weight[:, :required_pos_encodings_columns] for weight in broadcasted_weights], dim=-1
            )
            position_encodings = torch.reshape(position_encodings, (batch_size, -1, position_encodings.shape[-1]))[
                :, :sequence_length
            ]

        return position_encodings


class PositionEmbeddings(nn.Module):
    """Constructs conventional position embeddings of shape `[max_pos_embeddings, hidden_size]`.
    """

    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob
        self.embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, position_ids):
        position_embeddings = self.embedding(position_ids)
        position_embeddings = nn.functional.dropout(position_embeddings, p=self.dropout, training=self.training)
        return position_embeddings


class ReformerEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.max_position_embeddings = config.max_position_embeddings
        self.dropout = config.hidden_dropout_prob

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = (
            AxialPositionEmbeddings(config) if config.axial_pos_embds else PositionEmbeddings(config)
        )

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        assert (
            position_ids.shape[-1] <= self.max_position_embeddings
        ), "Sequence Length: {} has to be larger equal than config.max_position_embeddings: {}".format(
            position_ids.shape[-1], self.max_position_embeddings
        )

        # dropout
        embeddings = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)

        # add positional embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        return embeddings


class EfficientAttentionMixin:
    """
    A few utilities for nn.Modules in Reformer, to be used as a mixin.
    """

    def _look_adjacent(self, vectors, num_chunks_before, num_chunks_after):
        """ Used to implement attention between consecutive chunks.

            Args:
                vectors: array of shape [batch_size, num_attention_heads, n_chunks, chunk_len, ...]
                num_chunks_before: chunks before current chunk to include in attention
                num_chunks_after: chunks after current chunk to include in attention

            Returns:
                tensor of shape [num_chunks, N * chunk_length, ...], where
                N = (1 + num_chunks_before + num_chunks_after).
        """
        if num_chunks_before == 0 and num_chunks_after == 0:
            return vectors

        slices = []
        for i in range(-num_chunks_before, num_chunks_after + 1):
            if i == 0:
                slices.append(vectors)
            else:
                slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        return torch.cat(slices, dim=3)

    def _split_hidden_size_dim(self, x, num_attn_heads, attn_head_size):
        """
            splits hidden_size dim into attn_head_size and num_attn_heads
        """
        new_x_shape = x.size()[:-1] + (num_attn_heads, attn_head_size)
        x = x.view(*new_x_shape)
        return x.transpose(2, 1)

    def _merge_hidden_size_dims(self, x, num_attn_heads, attn_head_size):
        """
            merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        x = x.permute(0, 2, 1, 3)
        return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))

    def _split_seq_length_dim_to(self, vectors, dim_factor_1, dim_factor_2, num_attn_heads, attn_head_size=None):
        """
            splits sequence length dim of vectors into `dim_factor_1` and `dim_factor_2` dims
        """
        batch_size = vectors.shape[0]
        split_dim_shape = (batch_size, num_attn_heads, dim_factor_1, dim_factor_2)

        if len(vectors.shape) == 4:
            return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        elif len(vectors.shape) == 3:
            return torch.reshape(vectors, split_dim_shape)
        else:
            raise ValueError("Input vector rank should be one of [3, 4], but is: {}".format(len(vectors.shape)))


class LSHSelfAttention(nn.Module, EfficientAttentionMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.chunk_length = config.lsh_attn_chunk_length
        self.num_hashes = config.num_hashes
        self.num_buckets = config.num_buckets
        self.num_chunks_before = config.lsh_num_chunks_before
        self.num_chunks_after = config.lsh_num_chunks_after
        self.hash_seed = config.hash_seed
        self.is_decoder = config.is_decoder
        self.max_position_embeddings = config.max_position_embeddings

        self.dropout = config.lsh_attention_probs_dropout_prob

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size

        # projection matrices
        self.query_key = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=False)

        # save mask value here. Need fp32 and fp16 mask values
        self.register_buffer("self_mask_value_float16", torch.tensor(-1e3))
        self.register_buffer("self_mask_value_float32", torch.tensor(-1e5))
        self.register_buffer("mask_value_float16", torch.tensor(-1e4))
        self.register_buffer("mask_value_float32", torch.tensor(-1e9))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        output_attentions=False,
        buckets=None,
        **kwargs
    ):
        sequence_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]

        # num hashes can optionally be overwritten by user
        num_hashes = num_hashes if num_hashes is not None else self.num_hashes

        # project hidden_states to query_key and value
        query_key_vectors = self.query_key(hidden_states)
        value_vectors = self.value(hidden_states)

        # free memory
        del hidden_states

        query_key_vectors = self._split_hidden_size_dim(
            query_key_vectors, self.num_attention_heads, self.attention_head_size
        )
        value_vectors = self._split_hidden_size_dim(value_vectors, self.num_attention_heads, self.attention_head_size)

        assert (
            query_key_vectors.shape[-1] == self.attention_head_size
        ), "last dim of query_key_vectors is {} but should be {}.".format(
            query_key_vectors.shape[-1], self.attention_head_size
        )
        assert (
            value_vectors.shape[-1] == self.attention_head_size
        ), "last dim of value_vectors is {} but should be {}.".format(
            value_vectors.shape[-1], self.attention_head_size
        )

        # LSH attention only makes sense if chunked attention should be performed
        if self.chunk_length < sequence_length:
            # set `num_buckets` on the fly, recommended way to do it
            if self.num_buckets is None:
                self._set_num_buckets(sequence_length)

            # use cached buckets for backprop only
            if buckets is None:
                # hash query key vectors into buckets
                buckets = self._hash_vectors(query_key_vectors, num_hashes)

            assert (
                int(buckets.shape[-1]) == num_hashes * sequence_length
            ), "last dim of buckets is {}, but should be {}".format(buckets.shape[-1], num_hashes * sequence_length)

            sorted_bucket_idx, undo_sorted_bucket_idx = self._get_sorted_bucket_idx_and_undo_sorted_bucket_idx(
                sequence_length, buckets, num_hashes
            )

            # make sure bucket idx is not longer then sequence length
            sorted_bucket_idx_per_hash = sorted_bucket_idx % sequence_length

            # cluster query key value vectors according to hashed buckets
            query_key_vectors = self._gather_by_expansion(query_key_vectors, sorted_bucket_idx_per_hash, num_hashes)
            value_vectors = self._gather_by_expansion(value_vectors, sorted_bucket_idx_per_hash, num_hashes)

            query_key_vectors = self._split_seq_length_dim_to(
                query_key_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size,
            )
            value_vectors = self._split_seq_length_dim_to(
                value_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size,
            )

            if self.chunk_length is None:
                assert (
                    self.num_chunks_before == 0 and self.num_chunks_after == 0
                ), "If `config.chunk_length` is `None`, make sure `config.num_chunks_after` and `config.num_chunks_before` are set to 0."
        else:
            # get sequence length indices
            sorted_bucket_idx_per_hash = torch.arange(sequence_length, device=query_key_vectors.device).repeat(
                batch_size, self.num_attention_heads, 1
            )

        # scale key vectors
        key_vectors = self._len_and_dim_norm(query_key_vectors)

        # get attention probs
        out_vectors, logits, attention_probs = self._attend(
            query_vectors=query_key_vectors,
            key_vectors=key_vectors,
            value_vectors=value_vectors,
            sorted_bucket_idx_per_hash=sorted_bucket_idx_per_hash,
            attention_mask=attention_mask,
            head_mask=head_mask,
            sequence_length=sequence_length,
        )

        # free memory
        del query_key_vectors, key_vectors, value_vectors

        # re-order out_vectors and logits
        if self.chunk_length < sequence_length:
            # sort clusters back to correct ordering
            out_vectors, logits = ReverseSort.apply(out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx)

            # sum up all hash rounds
            if num_hashes > 1:
                out_vectors = self._split_seq_length_dim_to(
                    out_vectors, num_hashes, sequence_length, self.num_attention_heads, self.attention_head_size,
                )
                logits = self._split_seq_length_dim_to(
                    logits, num_hashes, sequence_length, self.num_attention_heads, self.attention_head_size,
                ).unsqueeze(-1)

                probs_vectors = torch.exp(logits - torch.logsumexp(logits, dim=2, keepdim=True))
                out_vectors = torch.sum(out_vectors * probs_vectors, dim=2)
                # free memory
                del probs_vectors

            # free memory
            del logits

        assert out_vectors.shape == (
            batch_size,
            self.num_attention_heads,
            sequence_length,
            self.attention_head_size,
        ), "out_vectors have be of shape `[batch_size, config.num_attention_heads, sequence_length, config.attention_head_size]`."

        out_vectors = self._merge_hidden_size_dims(out_vectors, self.num_attention_heads, self.attention_head_size)

        if output_attentions is False:
            attention_probs = ()

        return LSHSelfAttentionOutput(hidden_states=out_vectors, attention_probs=attention_probs, buckets=buckets)

    def _hash_vectors(self, vectors, num_hashes):
        batch_size = vectors.shape[0]

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        if isinstance(self.num_buckets, int):
            assert (
                self.num_buckets % 2 == 0
            ), "There should be an even number of bucktes, but `self.num_bucktes`: {}".format(self.num_buckets)
            rotation_size = self.num_buckets
            num_buckets = self.num_buckets
        else:
            # Factorize the hash if self.num_buckets is a list or tuple
            rotation_size, num_buckets = 0, 1
            for bucket_factor in self.num_buckets:
                assert bucket_factor % 2 == 0, "The number of buckets should be even, but `num_bucket`: {}".format(
                    bucket_factor
                )
                rotation_size = rotation_size + bucket_factor
                num_buckets = num_buckets * bucket_factor

        # remove gradient
        vectors = vectors.detach()

        if self.hash_seed is not None:
            # for determinism
            torch.manual_seed(self.hash_seed)

        rotations_shape = (self.num_attention_heads, vectors.shape[-1], num_hashes, rotation_size // 2)
        # create a random self.attention_head_size x num_hashes x num_buckets/2
        random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)

        # Output dim: Batch_Size x Num_Attn_Heads x Num_Hashes x Seq_Len x Num_Buckets/2
        rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)

        if isinstance(self.num_buckets, int) or len(self.num_buckets) == 1:
            rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
            buckets = torch.argmax(rotated_vectors, dim=-1)
        else:
            # Get the buckets for them and combine.
            buckets, cur_sum, cur_product = None, 0, 1
            for bucket_factor in self.num_buckets:
                rotated_vectors_factor = rotated_vectors[..., cur_sum : cur_sum + (bucket_factor // 2)]
                cur_sum = cur_sum + bucket_factor // 2
                rotated_vectors_factor = torch.cat([rotated_vectors_factor, -rotated_vectors_factor], dim=-1)

                if buckets is None:
                    buckets = torch.argmax(rotated_vectors_factor, dim=-1)
                else:
                    buckets = buckets + (cur_product * torch.argmax(rotated_vectors_factor, dim=-1))

                cur_product = cur_product * bucket_factor

        # buckets is now (Batch_size x Num_Attn_Heads x Num_Hashes x Seq_Len).
        # Next we add offsets so that bucket numbers from different hashing rounds don't overlap.
        offsets = torch.arange(num_hashes, device=vectors.device)
        offsets = (offsets * num_buckets).view((1, 1, -1, 1))

        # expand to batch size and num attention heads
        offsets = offsets.expand((batch_size, self.num_attention_heads) + offsets.shape[-2:])
        offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)

        return offset_buckets

    def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(self, sequence_length, buckets, num_hashes):
        # no gradients are needed
        with torch.no_grad():
            batch_size = buckets.shape[0]

            # arange and expand
            orig_indices = torch.arange(num_hashes * sequence_length, device=buckets.device).view(1, 1, -1)
            orig_indices = orig_indices.expand(batch_size, self.num_attention_heads, orig_indices.shape[-1])

            # scale buckets
            scaled_buckets = sequence_length * buckets + (orig_indices % sequence_length)

            # remove gradient
            scaled_buckets = scaled_buckets.detach()

            # Hash-based sort
            sorted_bucket_idx = torch.argsort(scaled_buckets, dim=-1)

            # create simple indices to scatter to, to have undo sort
            indices = (
                torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
                .view(1, 1, -1)
                .expand(sorted_bucket_idx.shape)
            )

            # get undo sort
            undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
            undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)

        return sorted_bucket_idx, undo_sorted_bucket_idx

    def _set_num_buckets(self, sequence_length):
        # `num_buckets` should be set to 2 * sequence_length // chunk_length as recommended in paper
        num_buckets_pow_2 = (2 * (sequence_length // self.chunk_length)).bit_length() - 1
        # make sure buckets are power of 2
        num_buckets = 2 ** num_buckets_pow_2

        # factorize `num_buckets` if `num_buckets` becomes too large
        num_buckets_limit = 2 * max(
            int((self.max_position_embeddings // self.chunk_length) ** (0.5)), self.chunk_length,
        )
        if num_buckets > num_buckets_limit:
            num_buckets = [2 ** (num_buckets_pow_2 // 2), 2 ** (num_buckets_pow_2 - num_buckets_pow_2 // 2)]

        logger.warning("config.num_buckets is not set. Setting config.num_buckets to {}...".format(num_buckets))

        # set num buckets in config to be properly saved
        self.config.num_buckets = num_buckets
        self.num_buckets = num_buckets

    def _attend(
        self,
        query_vectors,
        key_vectors,
        value_vectors,
        sorted_bucket_idx_per_hash,
        attention_mask,
        head_mask,
        sequence_length,
    ):

        # look at previous and following chunks if chunked attention
        if self.chunk_length < sequence_length:
            key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
            value_vectors = self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)

        # get logits and dots
        query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))

        # free memory
        del query_vectors, key_vectors

        # if chunked attention split bucket idxs to query and key
        if self.chunk_length < sequence_length:
            query_bucket_idx = self._split_seq_length_dim_to(
                sorted_bucket_idx_per_hash, -1, self.chunk_length, self.num_attention_heads
            )
            key_value_bucket_idx = self._look_adjacent(query_bucket_idx, self.num_chunks_before, self.num_chunks_after)
        else:
            query_bucket_idx = key_value_bucket_idx = sorted_bucket_idx_per_hash

        # get correct mask values depending on precision
        if query_key_dots.dtype == torch.float16:
            self_mask_value = self.self_mask_value_float16.half()
            mask_value = self.mask_value_float16.half()
        else:
            self_mask_value = self.self_mask_value_float32
            mask_value = self.mask_value_float32

        mask = self._compute_attn_mask(query_bucket_idx, key_value_bucket_idx, attention_mask, sequence_length)

        if mask is not None:
            query_key_dots = torch.where(mask, query_key_dots, mask_value)

        # free memory
        del mask

        # Self mask is ALWAYS applied.
        # From the reformer paper (https://arxiv.org/pdf/2001.04451.pdf):
        # " While attention to the future is not allowed, typical implementations of the
        # Transformer do allow a position to attend to itself.
        # Such behavior is undesirable in a shared-QK formulation because the dot-product
        # of a query vector with itself will almost always be greater than the dot product of a
        # query vector with a vector at another position. We therefore modify the masking
        # to forbid a token from attending to itself, except in situations
        # where a token has no other valid attention targets (e.g. the first token in a sequence) "

        self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
            query_bucket_idx.device
        )

        # apply self_mask
        query_key_dots = torch.where(self_mask, query_key_dots, self_mask_value)

        # free memory
        del self_mask

        logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        # dots shape is `[batch_size, num_attn_heads, num_hashes * seq_len // chunk_length, chunk_length, chunk_length * (1 + num_chunks_before + num_chunks_after)]`
        attention_probs = torch.exp(query_key_dots - logits)

        # free memory
        del query_key_dots

        # dropout
        attention_probs = nn.functional.dropout(attention_probs, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # attend values
        out_vectors = torch.matmul(attention_probs, value_vectors)

        # free memory
        del value_vectors

        # merge chunk length
        if self.chunk_length < sequence_length:
            logits = logits.flatten(start_dim=2, end_dim=3).squeeze(-1)
            out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)

        return out_vectors, logits, attention_probs

    def _compute_attn_mask(self, query_indices, key_indices, attention_mask, sequence_length):
        mask = None

        # Causal mask
        if self.is_decoder:
            mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2)).to(query_indices.device)

        # Attention mask: chunk, look up correct mask value from key_value_bucket_idx
        # IMPORTANT: official trax code does not use a mask for LSH Atttention. Not sure why.
        if attention_mask is not None:
            # if chunked attention, the attention mask has to correspond to LSH order
            if sequence_length > self.chunk_length:
                attention_mask = attention_mask.to(torch.uint8)[:, None, None, :]
                # expand attn_mask to fit with key_value_bucket_idx shape
                attention_mask = attention_mask.expand(query_indices.shape[:-1] + (-1,))
                key_attn_mask = torch.gather(attention_mask, -1, key_indices)
                query_attn_mask = torch.gather(attention_mask, -1, query_indices)
                # expand to query_key_dots shape: duplicate along query axis since key sorting is the same for each query position in chunk
                attn_mask = query_attn_mask.unsqueeze(-1) * key_attn_mask.unsqueeze(-2)

                # free memory
                del query_attn_mask, key_attn_mask
            else:
                # usual attention mask creation
                attention_mask = attention_mask.to(torch.uint8)[:, None, :]
                attn_mask = (attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(-2)).expand(
                    query_indices.shape + attention_mask.shape[-1:]
                )

            # free memory
            del attention_mask

            # multiply by casaul mask if necessary
            if mask is not None:
                mask = mask * attn_mask
            else:
                mask = attn_mask

        return mask

    def _len_and_dim_norm(self, vectors):
        """
            length and attention head size dim normalization
        """
        vectors = self._len_norm(vectors)
        vectors = vectors * torch.rsqrt(
            torch.tensor(self.attention_head_size, device=vectors.device, dtype=vectors.dtype)
        )
        return vectors

    def _len_norm(self, x, epsilon=1e-6):
        """
            length normalization
        """
        variance = torch.mean(x ** 2, -1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + epsilon)
        return norm_x

    def _gather_by_expansion(self, vectors, idxs, num_hashes):
        """
            expand dims of idxs and vectors for all hashes and gather
        """
        expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        vectors = vectors.repeat(1, 1, num_hashes, 1)
        return torch.gather(vectors, 2, expanded_idxs)


class ReverseSort(Function):
    """
        After chunked attention is applied which sorted clusters,
        original ordering has to be restored.
        Since customized backward function is used for Reformer,
        the gradients of the output vectors have to be explicitely
        sorted here.
    """

    @staticmethod
    def forward(ctx, out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx):
        # save sorted_bucket_idx for backprop
        with torch.no_grad():
            ctx.sorted_bucket_idx = sorted_bucket_idx

            # undo sort to have correct order for next layer
            expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(-1).expand(out_vectors.shape)
            out_vectors = torch.gather(out_vectors, 2, expanded_undo_sort_indices)
            logits = torch.gather(logits, 2, undo_sorted_bucket_idx)
        return out_vectors, logits

    @staticmethod
    def backward(ctx, grad_out_vectors, grad_logits):
        # get parameters saved in ctx
        sorted_bucket_idx = ctx.sorted_bucket_idx

        expanded_sort_indices = sorted_bucket_idx.unsqueeze(-1).expand(grad_out_vectors.shape)
        # reverse sort of forward
        grad_out_vectors = torch.gather(grad_out_vectors, 2, expanded_sort_indices)
        grad_logits = torch.gather(grad_logits, 2, sorted_bucket_idx)

        # return grad and `None` fillers for last 2 forward args
        return grad_out_vectors, grad_logits, None, None


class LocalSelfAttention(nn.Module, EfficientAttentionMixin):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.chunk_length = config.local_attn_chunk_length
        self.num_chunks_before = config.local_num_chunks_before
        self.num_chunks_after = config.local_num_chunks_after
        self.is_decoder = config.is_decoder
        self.pad_token_id = config.pad_token_id

        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size

        # projection matrices
        self.query = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=False)

        self.dropout = config.local_attention_probs_dropout_prob

        # save mask value here
        self.register_buffer("mask_value_float16", torch.tensor(-1e4))
        self.register_buffer("mask_value_float32", torch.tensor(-1e9))

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs):
        sequence_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]

        # project hidden_states to query, key and value
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        # split last dim into `config.num_attention_heads` and `config.attention_head_size`
        query_vectors = self._split_hidden_size_dim(query_vectors, self.num_attention_heads, self.attention_head_size)
        key_vectors = self._split_hidden_size_dim(key_vectors, self.num_attention_heads, self.attention_head_size)
        value_vectors = self._split_hidden_size_dim(value_vectors, self.num_attention_heads, self.attention_head_size)

        assert (
            query_vectors.shape[-1] == self.attention_head_size
        ), "last dim of query_key_vectors is {} but should be {}.".format(
            query_vectors.shape[-1], self.attention_head_size
        )
        assert (
            key_vectors.shape[-1] == self.attention_head_size
        ), "last dim of query_key_vectors is {} but should be {}.".format(
            key_vectors.shape[-1], self.attention_head_size
        )
        assert (
            value_vectors.shape[-1] == self.attention_head_size
        ), "last dim of query_key_vectors is {} but should be {}.".format(
            value_vectors.shape[-1], self.attention_head_size
        )

        if self.chunk_length is None:
            assert (
                self.num_chunks_before == 0 and self.num_chunks_after == 0
            ), "If `config.chunk_length` is `None`, make sure `config.num_chunks_after` and `config.num_chunks_before` are set to 0."

        # normalize key vectors
        key_vectors = key_vectors / torch.sqrt(
            torch.tensor(self.attention_head_size, device=key_vectors.device, dtype=key_vectors.dtype)
        )

        # get sequence length indices
        indices = torch.arange(sequence_length, device=query_vectors.device).repeat(
            batch_size, self.num_attention_heads, 1
        )

        # if input should be chunked
        if self.chunk_length < sequence_length:
            # chunk vectors
            # B x Num_Attn_Head x Seq_Len // chunk_len x chunk_len  x  attn_head_size
            query_vectors = self._split_seq_length_dim_to(
                query_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size,
            )
            key_vectors = self._split_seq_length_dim_to(
                key_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size,
            )
            value_vectors = self._split_seq_length_dim_to(
                value_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size,
            )

            # chunk indices
            query_indices = self._split_seq_length_dim_to(indices, -1, self.chunk_length, self.num_attention_heads)
            key_indices = self._split_seq_length_dim_to(indices, -1, self.chunk_length, self.num_attention_heads)

            # append chunks before and after
            key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
            value_vectors = self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)
            key_indices = self._look_adjacent(key_indices, self.num_chunks_before, self.num_chunks_after)
        else:
            query_indices = key_indices = indices

        # query-key matmul: QK^T
        query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))

        # free memory
        del query_vectors, key_vectors

        mask = self._compute_attn_mask(
            query_indices, key_indices, attention_mask, query_key_dots.shape, sequence_length
        )

        if mask is not None:
            # get mask tensor depending on half precision or not
            if query_key_dots.dtype == torch.float16:
                mask_value = self.mask_value_float16.half()
            else:
                mask_value = self.mask_value_float32

            query_key_dots = torch.where(mask, query_key_dots, mask_value)

        # free memory
        del mask

        # softmax
        logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        attention_probs = torch.exp(query_key_dots - logits)

        # free memory
        del logits

        # dropout
        attention_probs = nn.functional.dropout(attention_probs, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # attend values
        out_vectors = torch.matmul(attention_probs, value_vectors)

        # free memory
        del value_vectors

        # merge chunk length
        if self.chunk_length < sequence_length:
            out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)

        assert out_vectors.shape == (batch_size, self.num_attention_heads, sequence_length, self.attention_head_size,)

        out_vectors = self._merge_hidden_size_dims(out_vectors, self.num_attention_heads, self.attention_head_size)

        if output_attentions is False:
            attention_probs = ()

        return LocalSelfAttentionOutput(hidden_states=out_vectors, attention_probs=attention_probs)

    def _compute_attn_mask(self, query_indices, key_indices, attention_mask, query_key_dots_shape, sequence_length):
        mask = None

        # chunk attention mask and look before and after
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.uint8)[:, None, :]

            if self.chunk_length < sequence_length:
                attention_mask = self._split_seq_length_dim_to(attention_mask, -1, self.chunk_length, 1)
                attention_mask_key = self._look_adjacent(attention_mask, self.num_chunks_before, self.num_chunks_after)
            else:
                attention_mask_key = attention_mask

        # Causal mask
        if self.is_decoder is True:
            mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2)).to(query_indices.device)

        # Attention mask
        if attention_mask is not None:
            # create attn_mask
            attn_mask = (attention_mask.unsqueeze(-1) * attention_mask_key.unsqueeze(-2)).expand(query_key_dots_shape)
            # multiply by casaul mask if necessary
            if mask is not None:
                mask = mask * attn_mask
            else:
                mask = attn_mask
        return mask


class ReformerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        all_head_size = config.num_attention_heads * config.attention_head_size
        self.dropout = config.hidden_dropout_prob

        self.dense = nn.Linear(all_head_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class ReformerAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attn_layers = config.attn_layers

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if len(set(self.attn_layers)) == 1 and self.attn_layers[0] == "lsh":
            self.self_attention = LSHSelfAttention(config)
        elif len(set(self.attn_layers)) == 1 and self.attn_layers[0] == "local":
            self.self_attention = LocalSelfAttention(config)
        elif len(set(self.attn_layers)) == 2 and set(self.attn_layers) == set(["lsh", "local"]):
            # get correct attn layers
            if self.attn_layers[self.layer_id] == "lsh":
                self.self_attention = LSHSelfAttention(config)
            else:
                self.self_attention = LocalSelfAttention(config)
        else:
            raise NotImplementedError(
                "Only attn layer types 'lsh' and 'local' exist, but got `config.attn_layers`: {}. Select attn layer types from ['lsh', 'local'] only.".format(
                    self.attn_layers
                )
            )
        self.output = ReformerSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        output_attentions=False,
        buckets=None,
    ):
        hidden_states = self.layer_norm(hidden_states)

        # use cached buckets for backprob if buckets not None for LSHSelfAttention
        self_attention_outputs = self.self_attention(
            hidden_states=hidden_states,
            head_mask=head_mask,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            output_attentions=output_attentions,
            buckets=buckets,
        )
        attention_output = self.output(self_attention_outputs.hidden_states)

        # add buckets if necessary
        if hasattr(self_attention_outputs, "buckets"):
            buckets = self_attention_outputs.buckets
        else:
            buckets = None

        return AttentionOutput(
            hidden_states=attention_output, attention_probs=self_attention_outputs.attention_probs, buckets=buckets,
        )


class ReformerFeedForwardDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob

        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

        self.dense = nn.Linear(config.hidden_size, config.feed_forward_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states


class ReformerFeedForwardOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob

        self.dense = nn.Linear(config.feed_forward_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class ChunkReformerFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = ReformerFeedForwardDense(config)
        self.output = ReformerFeedForwardOutput(config)

    def forward(self, attention_output):
        return apply_chunking_to_forward(
            self.chunk_size_feed_forward, self.seq_len_dim, self.forward_chunk, attention_output,
        )

    def forward_chunk(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        return self.output(hidden_states)


class ReformerLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.attention = ReformerAttention(config, layer_id)
        # dropout requires to have the same
        # seed for forward and backward pass
        self.attention_seed = None
        self.feed_forward_seed = None

        self.feed_forward = ChunkReformerFeedForward(config)

    def _init_attention_seed(self):
        """
            This function sets a new seed for the
            attention layer to make dropout deterministic
            for both forward calls: 1 normal forward
            call and 1 forward call in backward
            to recalculate activations.
        """

        # randomize seeds
        if next(self.parameters()).device.type == "cuda":
            # GPU
            device_idx = torch.cuda.current_device()
            self.attention_seed = torch.cuda.default_generators[device_idx].seed()
            torch.cuda.manual_seed(self.attention_seed)
        else:
            # CPU
            self.attention_seed = int(torch.seed() % sys.maxsize)
            torch.manual_seed(self.attention_seed)

    def _init_feed_forward_seed(self):
        """
            This function sets a new seed for the
            feed forward layer to make dropout deterministic
            for both forward calls: 1 normal forward
            call and 1 forward call in backward
            to recalculate activations.
        """

        # randomize seeds
        if next(self.parameters()).device.type == "cuda":
            # GPU
            device_idx = torch.cuda.current_device()
            self.feed_forward_seed = torch.cuda.default_generators[device_idx].seed()
            torch.cuda.manual_seed(self.feed_forward_seed)
        else:
            # CPU
            self.feed_forward_seed = int(torch.seed() % sys.maxsize)
            torch.manual_seed(self.feed_forward_seed)

    def forward(
        self,
        prev_attn_output,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        output_attentions=False,
    ):
        with torch.no_grad():
            # every forward pass we sample a different seed
            # for dropout and save for forward fn in backward pass
            # to have correct dropout
            self._init_attention_seed()
            attn_outputs = self.attention(
                hidden_states=hidden_states,
                head_mask=head_mask,
                attention_mask=attention_mask,
                num_hashes=num_hashes,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs.hidden_states

            # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
            # Y_1 = X_1 + f(X_2)
            attn_output = prev_attn_output + attn_output

            # free memory
            del prev_attn_output

            # every forward pass we sample a different seed
            # for dropout and save seed for forward fn in backward
            # to have correct dropout
            self._init_feed_forward_seed()
            # Y_2 = X_2 + g(Y_1)
            hidden_states = hidden_states + self.feed_forward(attn_output)

        return ReformerOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            attention_probs=attn_outputs.attention_probs,
            buckets=attn_outputs.buckets,
        )

    def backward_pass(
        self,
        next_attn_output,
        hidden_states,
        grad_attn_output,
        grad_hidden_states,
        attention_mask=None,
        head_mask=None,
        buckets=None,
    ):
        # Implements the backward pass for reversible ResNets.
        # A good blog post on how this works can be found here:
        # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
        # This code is heavily inspired by https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py

        with torch.enable_grad():
            next_attn_output.requires_grad = True

            # set seed to have correct dropout
            torch.manual_seed(self.feed_forward_seed)
            # g(Y_1)
            res_hidden_states = self.feed_forward(next_attn_output)
            res_hidden_states.backward(grad_hidden_states, retain_graph=True)

        with torch.no_grad():
            # X_2 = Y_2 - g(Y_1)
            hidden_states = hidden_states - res_hidden_states
            del res_hidden_states

            grad_attn_output = grad_attn_output + next_attn_output.grad
            next_attn_output.grad = None

        with torch.enable_grad():
            hidden_states.requires_grad = True

            # set seed to have correct dropout
            torch.manual_seed(self.attention_seed)
            # f(X_2)
            # use cached buckets for backprob if buckets not None for LSHSelfAttention
            output = self.attention(
                hidden_states=hidden_states, head_mask=head_mask, attention_mask=attention_mask, buckets=buckets,
            ).hidden_states
            output.backward(grad_attn_output, retain_graph=True)

        with torch.no_grad():
            # X_1 = Y_1 - f(X_2)
            attn_output = next_attn_output - output
            del output, next_attn_output

            grad_hidden_states = grad_hidden_states + hidden_states.grad
            hidden_states.grad = None
            hidden_states = hidden_states.detach()

        return ReformerBackwardOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            grad_attn_output=grad_attn_output,
            grad_hidden_states=grad_hidden_states,
        )


class _ReversibleFunction(Function):
    """
    To prevent PyTorch from performing the usual backpropagation,
    a customized backward function is implemented here. This way
    it is made sure that no memory expensive activations are
    saved during the forward pass.
    This function is heavily inspired by https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        layers,
        attention_mask,
        head_mask,
        num_hashes,
        all_hidden_states,
        all_attentions,
        output_hidden_states,
        output_attentions,
    ):
        all_buckets = ()

        # split duplicated tensor
        hidden_states, attn_output = torch.chunk(hidden_states, 2, dim=-1)

        for layer, layer_head_mask in zip(layers, head_mask):
            if output_hidden_states is True:
                all_hidden_states.append(hidden_states)

            layer_outputs = layer(
                prev_attn_output=attn_output,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                num_hashes=num_hashes,
                output_attentions=output_attentions,
            )
            attn_output = layer_outputs.attn_output
            hidden_states = layer_outputs.hidden_states
            all_buckets = all_buckets + (layer_outputs.buckets,)

            if output_attentions:
                all_attentions.append(layer_outputs.attention_probs)

        # Add last layer
        if output_hidden_states is True:
            all_hidden_states.append(hidden_states)

        # attach params to ctx for backward
        ctx.save_for_backward(attn_output.detach(), hidden_states.detach())
        ctx.layers = layers
        ctx.all_buckets = all_buckets
        ctx.head_mask = head_mask
        ctx.attention_mask = attention_mask

        # Concatenate 2 RevNet outputs
        return torch.cat([attn_output, hidden_states], dim=-1)

    @staticmethod
    def backward(ctx, grad_hidden_states):
        grad_attn_output, grad_hidden_states = torch.chunk(grad_hidden_states, 2, dim=-1)

        # retrieve params from ctx for backward
        attn_output, hidden_states = ctx.saved_tensors

        # create tuple
        output = ReformerBackwardOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            grad_attn_output=grad_attn_output,
            grad_hidden_states=grad_hidden_states,
        )

        # free memory
        del grad_attn_output, grad_hidden_states, attn_output, hidden_states

        layers = ctx.layers
        all_buckets = ctx.all_buckets
        head_mask = ctx.head_mask
        attention_mask = ctx.attention_mask

        for idx, layer in enumerate(layers[::-1]):
            # pop last buckets from stack
            buckets = all_buckets[-1]
            all_buckets = all_buckets[:-1]

            # backprop
            output = layer.backward_pass(
                next_attn_output=output.attn_output,
                hidden_states=output.hidden_states,
                grad_attn_output=output.grad_attn_output,
                grad_hidden_states=output.grad_hidden_states,
                head_mask=head_mask[len(layers) - idx - 1],
                attention_mask=attention_mask,
                buckets=buckets,
            )

        assert all_buckets == (), "buckets have to be empty after backpropagation"
        grad_hidden_states = torch.cat([output.grad_attn_output, output.grad_hidden_states], dim=-1)

        # num of return vars has to match num of forward() args
        # return gradient for hidden_states arg and None for other args
        return grad_hidden_states, None, None, None, None, None, None, None, None


class ReformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob

        self.layers = nn.ModuleList([ReformerLayer(config, i) for i in range(config.num_hidden_layers)])
        # Reformer is using Rev Nets, thus last layer outputs are concatenated and
        # Layer Norm is done over 2 * hidden_size
        self.layer_norm = nn.LayerNorm(2 * config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        output_hidden_states=False,
        output_attentions=False,
    ):
        # hidden_states and attention lists to be filled if wished
        all_hidden_states = []
        all_attentions = []

        # concat same tensor for reversible ResNet
        hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        hidden_states = _ReversibleFunction.apply(
            hidden_states,
            self.layers,
            attention_mask,
            head_mask,
            num_hashes,
            all_hidden_states,
            all_attentions,
            output_hidden_states,
            output_attentions,
        )

        # Apply layer norm to concatenated hidden states
        hidden_states = self.layer_norm(hidden_states)

        # Apply dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        return ReformerEncoderOutput(
            hidden_states=hidden_states, all_hidden_states=all_hidden_states, all_attentions=all_attentions
        )


class ReformerOnlyLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Reformer is using Rev Nets, thus last layer outputs are concatenated and
        # Layer Norm is done over 2 * hidden_size
        self.seq_len_dim = 1
        self.chunk_size_lm_head = config.chunk_size_lm_head
        self.decoder = nn.Linear(2 * config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        return apply_chunking_to_forward(self.chunk_size_lm_head, self.seq_len_dim, self.forward_chunk, hidden_states)

    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class ReformerPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = ReformerConfig
    base_model_prefix = "reformer"

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, AxialPositionEmbeddings):
            for weight in module.weights:
                torch.nn.init.normal_(weight, std=self.config.axial_norm_std)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


REFORMER_START_DOCSTRING = r"""
    Reformer was proposed in `Reformer: The Efficient Transformer <https://arxiv.org/abs/2001.0445>`__
    by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.ReformerConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

REFORMER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            During training the input_ids sequence_length has to be a multiple of the relevant model's
            chunk lengths (lsh's, local's or both). During evaluation, the indices are automatically
            padded to be a multiple of the chunk length.

            Indices can be obtained using :class:`transformers.ReformerTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.__call__` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        num_hashes (:obj:`int`, `optional`, defaults to :obj:`None`):
            `num_hashes` is the number of hashing rounds that should be performed during
            bucketing. Setting `num_hashes` overwrites the default `num_hashes` defined
            in `config.num_hashes`.
            For more information, see `num_hashes` in :class:`transformers.ReformerConfig`.
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
"""


@add_start_docstrings(
    "The bare Reformer Model transformer outputting raw hidden-states" "without any specific head on top.",
    REFORMER_START_DOCSTRING,
)
class ReformerModel(ReformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        assert (
            self.config.num_hidden_layers > 0
        ), "`config.attn_layers` is empty. Select at least one attn layer form ['lsh', 'local']"

        self.embeddings = ReformerEmbeddings(config)
        self.encoder = ReformerEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(REFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="google/reformer-crime-and-punishment")
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        num_hashes=None,
        output_hidden_states=None,
        output_attentions=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        all_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        all_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()  # noqa: F841
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]  # noqa: F841
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        assert (
            len(input_shape) == 2
        ), "`input_ids` have be of shape `[batch_size, sequence_length]`, but got shape: {}".format(input_shape)

        # prepare head mask
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers, is_attention_chunked=True)

        # original sequence length for padding
        orig_sequence_length = input_shape[-1]

        # if needs padding
        least_common_mult_chunk_length = _get_least_common_mult_chunk_len(self.config)
        must_pad_to_match_chunk_length = (
            input_shape[-1] % least_common_mult_chunk_length != 0 and input_shape[-1] > least_common_mult_chunk_length
        )

        if must_pad_to_match_chunk_length:
            padding_length = least_common_mult_chunk_length - input_shape[-1] % least_common_mult_chunk_length

            if self.training is True:
                raise ValueError(
                    "If training, sequence Length {} has to be a multiple of least common multiple chunk_length {}. Please consider padding the input to a length of {}.".format(
                        input_shape[-1], least_common_mult_chunk_length, input_shape[-1] + padding_length
                    )
                )

            # pad input
            input_ids, inputs_embeds, attention_mask, position_ids, input_shape = self._pad_to_mult_of_chunk_length(
                input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                input_shape=input_shape,
                padding_length=padding_length,
                padded_seq_length=least_common_mult_chunk_length,
                device=device,
            )

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            head_mask=head_mask,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        sequence_output = encoder_outputs.hidden_states

        # if padding was applied
        if must_pad_to_match_chunk_length:
            sequence_output = sequence_output[:, :orig_sequence_length]

        outputs = (sequence_output,)
        # TODO(PVP): Replace by named tuple after namedtuples are introduced in the library.
        if output_hidden_states is True:
            outputs = outputs + (encoder_outputs.all_hidden_states,)
        if output_attentions is True:
            outputs = outputs + (encoder_outputs.all_attentions,)
        return outputs

    def _pad_to_mult_of_chunk_length(
        self,
        input_ids,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        input_shape=None,
        padding_length=None,
        padded_seq_length=None,
        device=None,
    ):
        logger.info(
            "Input ids are automatically padded from {} to {} to be a multiple of `config.chunk_length`: {}".format(
                input_shape[-1], input_shape[-1] + padding_length, padded_seq_length
            )
        )

        padded_input_ids = torch.full(
            (input_shape[0], padding_length), self.config.pad_token_id, device=device, dtype=torch.long,
        )

        # Extend `attention_mask`
        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.zeros(input_shape[0], padding_length, device=device, dtype=attention_mask.dtype,),
                ],
                dim=-1,
            )
        else:
            attention_mask = torch.cat(
                [
                    torch.ones(input_shape, device=device, dtype=torch.uint8),
                    torch.zeros((input_shape[0], padding_length), device=device, dtype=torch.uint8),
                ],
                dim=-1,
            )

        # Extend `input_ids` with padding to match least common multiple chunk_length
        if input_ids is not None:
            input_ids = torch.cat([input_ids, padded_input_ids], dim=-1)
            input_shape = input_ids.size()

            # Pad position ids if given
            if position_ids is not None:
                padded_position_ids = torch.arange(input_shape[-1], padded_seq_length, dtype=torch.long, device=device)
                padded_position_ids = position_ids.unsqueeze(0).expand(input_shape[0], padding_length)
                position_ids = torch.cat([position_ids, padded_position_ids], dim=-1)

        # Extend `inputs_embeds` with padding to match least common multiple chunk_length
        if inputs_embeds is not None:
            padded_inputs_embeds = self.embeddings(padded_input_ids, position_ids)
            inputs_embeds = torch.cat([inputs_embeds, padded_inputs_embeds], dim=-2)
            input_shape = inputs_embeds.size()
        return input_ids, inputs_embeds, attention_mask, position_ids, input_shape


@add_start_docstrings("""Reformer Model with a `language modeling` head on top. """, REFORMER_START_DOCSTRING)
class ReformerModelWithLMHead(ReformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        assert config.is_decoder, "If you want to use `ReformerLMHeadModel` make sure that `is_decoder=True`."
        self.reformer = ReformerModel(config)
        self.lm_head = ReformerOnlyLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def tie_weights(self):
        # word embeddings are not tied in Reformer
        pass

    @add_start_docstrings_to_callable(REFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="google/reformer-crime-and-punishment")
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        num_hashes=None,
        labels=None,
        output_hidden_states=None,
        output_attentions=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
                All labels set to ``-100`` are ignored (masked), the loss is only
                computed for labels in ``[0, ..., config.vocab_size]``

    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss (cross entropy).
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        all_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        all_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        reformer_outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            num_hashes=num_hashes,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        sequence_output = reformer_outputs[0]
        logits = self.lm_head(sequence_output)
        outputs = (logits,) + reformer_outputs[1:]

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (lm_loss), lm_logits, (hidden_states), (attentions)

    def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
        # TODO(PVP): Add smart caching
        inputs_dict = {"input_ids": input_ids}

        if "num_hashes" in kwargs:
            inputs_dict["num_hashes"] = kwargs["num_hashes"]

        return inputs_dict


@add_start_docstrings("""Reformer Model with a `language modeling` head on top. """, REFORMER_START_DOCSTRING)
class ReformerForMaskedLM(ReformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        assert (
            not config.is_decoder
        ), "If you want to use `ReformerForMaskedLM` make sure `config.is_decoder=False` for bi-directional self-attention."
        self.reformer = ReformerModel(config)
        self.lm_head = ReformerOnlyLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def tie_weights(self):
        # word embeddings are not tied in Reformer
        pass

    @add_start_docstrings_to_callable(REFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="google/reformer-crime-and-punishment")
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        num_hashes=None,
        labels=None,
        output_hidden_states=None,
        output_attentions=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Labels for computing the masked language modeling loss.
                Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels

    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss (cross entropy).
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        all_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        all_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        reformer_outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            num_hashes=num_hashes,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        sequence_output = reformer_outputs[0]
        logits = self.lm_head(sequence_output)
        outputs = (logits,) + reformer_outputs[1:]

        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (mlm_loss), lm_logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Reformer Model with a span classification head on top for
    extractive question-answering tasks like SQuAD / TriviaQA ( a linear layer on
    top of hidden-states output to compute `span start logits` and `span end logits`. """,
    REFORMER_START_DOCSTRING,
)
class ReformerForQuestionAnswering(ReformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.reformer = ReformerModel(config)
        # 2 * config.hidden_size because we use reversible residual layers
        self.qa_outputs = nn.Linear(2 * config.hidden_size, config.num_labels)

        self.init_weights()

    def tie_weights(self):
        # word embeddings are not tied in Reformer
        pass

    @add_start_docstrings_to_callable(REFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="google/reformer-crime-and-punishment")
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        num_hashes=None,
        start_positions=None,
        end_positions=None,
        output_hidden_states=None,
        output_attentions=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.ReformerConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        all_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        all_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        reformer_outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            num_hashes=num_hashes,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        sequence_output = reformer_outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + reformer_outputs[1:]

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
