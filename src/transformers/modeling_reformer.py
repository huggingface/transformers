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

import inspect
import logging
import sys
from collections import namedtuple
from functools import reduce
from operator import mul
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.autograd.function import Function
from torch.nn import CrossEntropyLoss

from .activations import gelu, gelu_fast, gelu_new, swish
from .configuration_reformer import ReformerConfig
from .modeling_utils import PreTrainedModel


logger = logging.getLogger(__name__)

REFORMER_PRETRAINED_MODEL_ARCHIVE_MAP = {}
# TODO: fill with pretrained model weights
#    "reformer-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/reformer/reformer-base-uncased-pytorch_model.bin",
#    "reformer-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/reformer/reformer-large-uncased-pytorch_model.bin",
#    "reformer-base-cased": "https://s3.amazonaws.com/models.huggingface.co/reformer/reformer-base-cased-pytorch_model.bin",
#    "reformer-large-cased": "https://s3.amazonaws.com/models.huggingface.co/reformer/reformer-large-cased-pytorch_model.bin",


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
ReformerLayerNorm = torch.nn.LayerNorm


LSHSelfAttentionOutput = namedtuple("LSHSelfAttentionOutput", ["hidden_states", "attention_probs", "buckets"])
LocalSelfAttentionOutput = namedtuple("LocalSelfAttentionOutput", ["hidden_states", "attention_probs"])
AttentionOutput = namedtuple("AttentionOutput", ["hidden_states", "attention_probs", "buckets"])
ReformerOutput = namedtuple("ReformerOutput", ["hidden_states", "attn_output", "attention_probs", "buckets"])
ReformerBackwardOutput = namedtuple(
    "ReformerBackwardOutput", ["attn_output", "hidden_states", "grad_attn_output", "grad_hidden_states"]
)
ReformerEncoderOutput = namedtuple("ReformerEncoderOutput", ["hidden_states", "all_hidden_states", "all_attentions"])


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


def _get_least_common_mult_chunk_len(config):
    attn_types = config.attn_layers
    if len(set(attn_types)) == 1 and attn_types[0] == "lsh":
        return config.lsh_attn_chunk_length
    elif len(set(attn_types)) == 1 and attn_types[0] == "local":
        return config.local_attn_chunk_length
    elif len(set(attn_types)) == 2 and set(attn_types) == set(["lsh", "local"]):
        return np.lcm(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    else:
        raise NotImplementedError(
            "Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {}. Select attn layer types from ['lsh', 'local'] only.".format(
                config.attn_layers
            )
        )


def apply_chunking_to_forward(
    chunk_size: int, chunk_dim: int, forward_fn: Callable[..., torch.Tensor], *input_tensors
) -> torch.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension `chunk_dim`.
    It then applies a layer `forward_fn` to each chunk independently to save memory.
    If the `forward_fn` is independent across the `chunk_dim` this function will yield the
    same result as not applying it.

    Args:
        chunk_size: int - the chunk size of a chunked tensor. `num_chunks` = `len(input_tensors[0]) / chunk_size`
        chunk_dim: int - the dimension over which the input_tensors should be chunked
        forward_fn: fn - the forward fn of the model
        input_tensors: tuple(torch.Tensor) - the input tensors of `forward_fn` which are chunked
    Returns:
        a Tensor with the same shape the foward_fn would have given if applied
    """

    assert len(input_tensors) > 0, "{} has to be a tuple/list of tensors".format(input_tensors)
    tensor_shape = input_tensors[0].shape
    assert all(
        input_tensor.shape == tensor_shape for input_tensor in input_tensors
    ), "All input tenors have to be of the same shape"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compability
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    assert num_args_in_forward_chunk_fn == len(
        input_tensors
    ), "forward_chunk_fn expects {} arguments, but only {} input tensors are given".format(
        num_args_in_forward_chunk_fn, len(input_tensors)
    )

    if chunk_size > 0:
        assert (
            input_tensors[0].shape[chunk_dim] % chunk_size == 0
        ), "The dimension to be chunked {} has to be a multiple of the chunk size {}".format(
            input_tensors[0][chunk_dim], chunk_size
        )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)


class AxialPositionEmbeddings(nn.Module):
    """Constructs axial position embeddings. Useful for very long input
    sequences to stay memory and computational efficient
    """

    def __init__(self, config):
        super().__init__()
        self.axial_pos_shape = config.axial_pos_shape
        self.axial_pos_embds_dim = config.axial_pos_embds_dim
        self.least_common_mult_chunk_length = _get_least_common_mult_chunk_len(config)
        self.weights = nn.ParameterList()
        self.dropout = config.hidden_dropout_prob

        assert (
            sum(self.axial_pos_embds_dim) == config.hidden_size
        ), "Make sure that config.axial_pos_embds factors: {} sum to config.hidden_size: {}".format(
            self.axial_pos_embds_dim, config.hidden_size
        )

        # create weights
        for axis, axial_pos_embd_dim in enumerate(self.axial_pos_embds_dim):
            # create shape
            ax_shape = [1] * len(self.axial_pos_shape)
            ax_shape[axis] = self.axial_pos_shape[axis]
            ax_shape = tuple(ax_shape) + (axial_pos_embd_dim,)

            # create tenser and init
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
            ), "Make sure that config.axial_pos_shape factors: {} multiply to sequence length: {}".format(
                self.axial_pos_shape, sequence_length
            )
            if self.dropout > 0:
                weights = torch.cat(broadcasted_weights, dim=-1)
                # permute weights so that 2D correctly drops dims 1 and 2
                perm_weigthts = weights.permute(0, 3, 2, 1)
                # drop entire matrix of last two dims (prev dims 1 and 2)
                drop_perm_weights = nn.functional.dropout2d(perm_weigthts, self.dropout, training=self.training)
                drop_weights = drop_perm_weights.permute(0, 3, 2, 1)
                position_encodings = torch.reshape(drop_weights, (batch_size, sequence_length, -1))

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
            # reshape axial encodings and use only until sequence_length
            position_encodings = torch.cat(broadcasted_weights, dim=-1)
            position_encodings = position_encodings.view(batch_size, -1, position_encodings.shape[-1])[
                :, :sequence_length
            ]

        return position_encodings


class PositionEmbeddings(nn.Module):
    """Constructs conventional position embeddings of shape `[max_pos_embeddings, hidden_size]`.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = config.hidden_dropout_prob
        self.embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        if self.config.sinusoidal_pos_embds is True:
            create_sinusoidal_embeddings(
                n_pos=config.max_position_embeddings, dim=config.hidden_size, out=self.embedding.weight,
            )

    def forward(self, position_ids):
        position_embeddings = self.embedding(position_ids)
        if self.config.sinusoidal_pos_embds is False:
            position_embeddings = nn.functional.dropout(position_embeddings, self.dropout, self.training)
        return position_embeddings


class ReformerEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = (
            AxialPositionEmbeddings(config) if config.axial_pos_embds else PositionEmbeddings(config)
        )
        assert not (
            config.sinusoidal_pos_embds and config.axial_pos_embds
        ), "Select either config.sinusoidal_pos_embds or config.axial_pos_embds"
        self.max_position_embeddings = config.max_position_embeddings
        self.dropout = config.hidden_dropout_prob

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
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
        embeddings = nn.functional.dropout(inputs_embeds, self.dropout, self.training)

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = embeddings + position_embeddings
        embeddings = nn.functional.dropout(embeddings, self.dropout, self.training)
        return embeddings


class EfficientAttentionUtils(object):
    """
    A few utilities for nn.Modules in Reformer, to be used as a mixin.
    """

    def _look_adjacent(self, vectors, num_chunks_before, num_chunks_after):
        """ Used to implement attention between consecutive chunks.

            Args:
                vectors: array of shape [batch_size, num_attention_heads, n_chunks, chunk_len, ...]
            Returns:
                array of shape [n_chunks, N * chunk_len, ...], where
                N = (1 + n_chunks_before + n_chunks_after).
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

    def _transpose_for_scores(self, x, num_attn_heads, attn_head_size):
        new_x_shape = x.size()[:-1] + (num_attn_heads, attn_head_size)
        x = x.view(*new_x_shape)
        return x.transpose(2, 1)

    def _transpose_for_output(self, x, num_attn_heads, attn_head_size):
        x = x.permute(0, 2, 1, 3)
        return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))

    def _split_dim_by(self, vectors, dim_factor_1, dim_factor_2, num_attn_heads, attn_head_size=None):
        batch_size = vectors.shape[0]
        split_dim_shape = (batch_size, num_attn_heads, dim_factor_1, dim_factor_2)

        if len(vectors.shape) == 4:
            return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        elif len(vectors.shape) == 3:
            return torch.reshape(vectors, split_dim_shape)
        else:
            raise ValueError("Input vector rank should be one of [3, 4], but is: {}".format(len(vectors.shape)))

    def _merge_by_middle_dim(self, vectors, num_attn_heads):
        batch_size = vectors.shape[0]
        new_dim_shape = (batch_size, num_attn_heads, -1)

        if len(vectors.shape) == 5:
            return torch.reshape(vectors, new_dim_shape + (vectors.shape[-1],))
        elif len(vectors.shape) == 4:
            return torch.reshape(vectors, new_dim_shape)
        else:
            raise ValueError("Input vector rank should be one of [4, 5], but is: {}".format(len(vectors.shape)))


class LSHSelfAttention(nn.Module, EfficientAttentionUtils):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.hash_seed = config.hash_seed
        self.num_hashes = config.num_hashes
        self.num_buckets = config.num_buckets
        self.chunk_length = config.lsh_attn_chunk_length
        self.num_chunks_before = config.lsh_num_chunks_before
        self.num_chunks_after = config.lsh_num_chunks_after
        self.is_decoder = config.is_decoder
        self.max_position_embeddings = config.max_position_embeddings

        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size

        # projection matrices
        self.query_key = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=False)

        self.dropout = config.lsh_attention_probs_dropout_prob

        # save mask value here
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
        do_output_attentions=False,
        buckets=None,
        **kwargs
    ):
        # get SeqLen and BatchSize
        sequence_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]
        # num hashes can optionally be overwritten by user
        num_hashes = num_hashes if num_hashes is not None else self.num_hashes

        query_key_vectors = self.query_key(hidden_states)
        value_vectors = self.value(hidden_states)

        # free memory
        del hidden_states

        query_key_vectors = self._transpose_for_scores(
            query_key_vectors, self.num_attention_heads, self.attention_head_size
        )
        value_vectors = self._transpose_for_scores(value_vectors, self.num_attention_heads, self.attention_head_size)

        assert query_key_vectors.shape[-1] == self.attention_head_size
        assert value_vectors.shape[-1] == self.attention_head_size

        if self.num_buckets is None:
            # set `num_buckets` on the fly
            self._set_num_buckets_on_the_fly(sequence_length)

        # use cached buckets for backprop only
        if buckets is None:
            # hash query key vectors into buckets
            buckets = self._hash_vectors(query_key_vectors, num_hashes)

        assert int(buckets.shape[-1]) == num_hashes * sequence_length

        sorted_bucket_idx, undo_sorted_bucket_idx = self._get_sorted_bucket_idx_and_undo_sorted_bucket_idx(
            sequence_length, buckets, num_hashes
        )

        # make sure bucket idx is not longer then sequence length
        sorted_bucket_idx = sorted_bucket_idx % sequence_length

        query_key_vectors = self._gather_by_expansion(query_key_vectors, sorted_bucket_idx, num_hashes)
        value_vectors = self._gather_by_expansion(value_vectors, sorted_bucket_idx, num_hashes)

        query_key_vectors = self._split_dim_by(
            query_key_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size,
        )
        value_vectors = self._split_dim_by(
            value_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size,
        )

        if self.chunk_length is None:
            assert self.num_chunks_before == 0 and self.num_chunks_after == 0

        key_vectors = self._len_and_dim_norm(query_key_vectors)

        out_vectors, logits, attention_probs = self._attend(
            query_vectors=query_key_vectors,
            key_vectors=key_vectors,
            value_vectors=value_vectors,
            sorted_bucket_idx=sorted_bucket_idx,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        # free memory
        del query_key_vectors, key_vectors, value_vectors

        # calculate total concatenad chunks to split gradients correctly
        out_vectors, logits = GatherSorted.apply(
            out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx, self.num_hashes
        )

        if num_hashes > 1:
            out_vectors = self._split_dim_by(
                out_vectors, num_hashes, sequence_length, self.num_attention_heads, self.attention_head_size,
            )
            logits = self._split_dim_by(
                logits, num_hashes, sequence_length, self.num_attention_heads, self.attention_head_size,
            ).unsqueeze(-1)

            probs_vectors = torch.exp(logits - torch.logsumexp(logits, dim=2, keepdim=True))
            out_vectors = torch.sum(out_vectors * probs_vectors, dim=2)
            # free memory
            del probs_vectors

        # free memory
        del logits

        assert out_vectors.shape == (batch_size, self.num_attention_heads, sequence_length, self.attention_head_size,)

        out_vectors = self._transpose_for_output(out_vectors, self.num_attention_heads, self.attention_head_size)

        if do_output_attentions is False:
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
        random_rotations = torch.randn(rotations_shape, device=vectors.device).to(vectors.dtype)

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
        offsets = torch.reshape(offsets * num_buckets, (-1, 1))

        # repeat same values for Batch_size and Num_Attn_Heads
        offsets = offsets.repeat(batch_size, self.num_attention_heads, 1, 1)
        offset_buckets = self._merge_by_middle_dim(buckets + offsets, self.num_attention_heads)

        return offset_buckets

    def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(self, sequence_length, buckets, num_hashes):
        batch_size = buckets.shape[0]

        orig_indices = torch.arange(num_hashes * sequence_length, device=buckets.device)
        orig_indices = orig_indices.repeat(batch_size, self.num_attention_heads, 1)

        # scale buckets
        scaled_buckets = sequence_length * buckets + (orig_indices % sequence_length)

        # remove gradient
        scaled_buckets = scaled_buckets.detach()

        # Hash-based sort
        sorted_bucket_idx = torch.argsort(scaled_buckets, dim=-1)
        undo_sorted_bucket_idx = torch.argsort(sorted_bucket_idx, dim=-1)

        # remove gradient
        sorted_bucket_idx = sorted_bucket_idx.detach()
        undo_sorted_bucket_idx = undo_sorted_bucket_idx.detach()

        return sorted_bucket_idx, undo_sorted_bucket_idx

    def _set_num_buckets_on_the_fly(self, sequence_length):
        # recommended `num_buckets` from paper
        num_buckets = 2 * sequence_length // self.chunk_length

        # factorize `num_buckets` if `num_buckets` becomes too large
        num_buckets_limit = max(int((self.max_position_embeddings // self.chunk_length) ** (0.5)), self.chunk_length,)
        if num_buckets > 2 * num_buckets_limit:
            num_buckets = [num_buckets_limit, num_buckets // num_buckets_limit + 1]

        logger.warning("config.num_buckets is not set. Setting config.num_buckets to {}...".format(num_buckets))
        self.num_buckets = num_buckets

    def _attend(
        self, query_vectors, key_vectors, value_vectors, sorted_bucket_idx, attention_mask, head_mask,
    ):
        key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
        value_vectors = self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)

        # get logits and dots
        query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        # free memory
        del query_vectors, key_vectors

        query_bucket_idx = self._split_dim_by(sorted_bucket_idx, -1, self.chunk_length, self.num_attention_heads)
        key_value_bucket_idx = self._look_adjacent(query_bucket_idx, self.num_chunks_before, self.num_chunks_after)

        mask = None
        # Causal mask
        if self.is_decoder:
            mask = torch.ge(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
                query_bucket_idx.device
            )

        # Attention mask: chunk, look up correct mask value from key_value_bucket_idx
        # IMPORTANT: official trax code does not use a mask for LSH Atttention. Not sure why.
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.uint8)[:, None, None, :]
            # expand attn_mask to fit with key_value_bucket_idx shape
            attention_mask = attention_mask.expand(query_bucket_idx.shape[:-1] + (-1,))
            key_attn_mask = torch.gather(attention_mask, -1, key_value_bucket_idx)
            query_attn_mask = torch.gather(attention_mask, -1, query_bucket_idx)
            # expand to query_key_dots shape: duplicate along query axis since key sorting is the same for each query position in chunk
            attn_mask = query_attn_mask.unsqueeze(-1) * key_attn_mask.unsqueeze(-2)
            # free memory
            del query_attn_mask, key_attn_mask, attention_mask

            # multiply by casaul mask if necessary
            if mask is not None:
                mask = mask * attn_mask
            else:
                mask = attn_mask
            # free memory
            del attn_mask

        # get correct mask values depending on precision
        if query_key_dots.dtype == torch.float16:
            self_mask_value = self.self_mask_value_float16
            mask_value = self.mask_value_float16
        else:
            self_mask_value = self.self_mask_value_float32
            mask_value = self.mask_value_float32

        # if attention_mask and/or casaul mask apply here
        if mask is not None:
            query_key_dots = torch.where(mask, query_key_dots, mask_value)

        # Self mask is ALWAYS applied.
        # From the reformer paper (https://arxiv.org/pdf/2001.04451.pdf):
        # " While attention to the future is not allowed, typical implementations of the
        # Transformer do allow a position to attend to itself.
        # Such behavior is undesirable in a shared-QK formulation because the dot-product
        # of a query vector with itself will almost always be greater than the dot product of a
        # query vector with a vector at another position. We therefore modify the masking
        # to forbid a token from attending to itself, except in situations
        # where a token has no other valid attention targets (e.g. the first token in a sequence) "
        mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(query_bucket_idx.device)
        query_key_dots = torch.where(mask, query_key_dots, self_mask_value)

        # free memory
        del mask

        logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        # dots shape is `[batch_size, num_attn_heads, num_hashes * seq_len // chunk_length, chunk_length, chunk_length * (1 + num_chunks_before + num_chunks_after)]`
        attention_probs = torch.exp(query_key_dots - logits)

        # free memory
        del query_key_dots

        # dropout
        attention_probs = nn.functional.dropout(attention_probs, self.dropout, self.training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # attend values
        out_vectors = torch.matmul(attention_probs, value_vectors)

        # free memory
        del value_vectors

        # merge chunk length
        logits = self._merge_by_middle_dim(logits, self.num_attention_heads).squeeze(-1)
        out_vectors = self._merge_by_middle_dim(out_vectors, self.num_attention_heads)

        return out_vectors, logits, attention_probs

    def _len_and_dim_norm(self, vectors):
        vectors = self._len_norm(vectors)
        vectors = vectors / torch.sqrt(
            torch.tensor(self.attention_head_size, device=vectors.device, dtype=vectors.dtype)
        )
        return vectors

    def _len_norm(self, x, epsilon=1e-6):
        variance = torch.mean(x ** 2, -1, keepdim=True)
        norm_x = x / torch.sqrt(variance + epsilon)
        return norm_x

    def _gather_by_expansion(self, vectors, idxs, num_hashes):
        expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        vectors = vectors.repeat(1, 1, num_hashes, 1)
        return torch.gather(vectors, 2, expanded_idxs)


class GatherSorted(Function):
    @staticmethod
    def forward(ctx, out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx, num_hashes):
        # save sorted_bucket_idx for backprop
        ctx.sorted_bucket_idx = sorted_bucket_idx
        ctx.num_hashes = num_hashes

        out_vectors = out_vectors.detach()
        logits = logits.detach()
        # undo sort to have correct order for next layer
        expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(-1).expand(out_vectors.shape)
        out_vectors = torch.gather(out_vectors, 2, expanded_undo_sort_indices)
        logits = torch.gather(logits, 2, undo_sorted_bucket_idx)
        return out_vectors, logits

    @staticmethod
    def backward(ctx, grad_out_vectors, grad_logits):
        # get parameters saved in ctx
        sorted_bucket_idx = ctx.sorted_bucket_idx
        num_hashes = ctx.num_hashes

        # get real gradient shape
        # shape is BatchSize x NumAttnHeads x ChunkLen * NumHashes
        grad_logits_shape = grad_logits.shape
        # shape is BatchSize x NumAttnHeads x ChunkLen * NumHashes x ChunkLen
        grad_out_vectors_shape = grad_out_vectors.shape

        # split gradient vectors and sorted bucket idxs by concatenated chunk dimension to gather correct indices
        # shape is BatchSize x NumAttnHeads x NumHashes x ChunkLen
        grad_logits = torch.reshape(grad_logits, (grad_logits_shape[:2] + (num_hashes, -1)))
        # shape is BatchSize x NumAttnHeads x NumHashes x ChunkLen x ChunkLen
        grad_out_vectors = torch.reshape(
            grad_out_vectors, (grad_out_vectors_shape[:2] + (num_hashes, -1) + grad_out_vectors_shape[-1:])
        )

        sorted_bucket_idx = torch.reshape(sorted_bucket_idx, (sorted_bucket_idx.shape[:2] + (num_hashes, -1)))

        # reverse sort of forward
        expanded_sort_indices = sorted_bucket_idx.unsqueeze(-1).expand(grad_out_vectors.shape)
        grad_out_vectors = torch.gather(grad_out_vectors, 3, expanded_sort_indices)
        grad_logits = torch.gather(grad_logits, 3, sorted_bucket_idx)

        # reshape into correct shape
        grad_logits = torch.reshape(grad_logits, grad_logits_shape)
        grad_out_vectors = torch.reshape(grad_out_vectors, grad_out_vectors_shape)

        # return grad and `None` fillers for last 3 forward args
        return grad_out_vectors, grad_logits, None, None, None


class LocalSelfAttention(nn.Module, EfficientAttentionUtils):
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

    def forward(self, hidden_states, attention_mask=None, head_mask=None, do_output_attentions=False, **kwargs):

        # get SeqLen and BatchSize
        sequence_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]

        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        query_vectors = self._transpose_for_scores(query_vectors, self.num_attention_heads, self.attention_head_size)
        key_vectors = self._transpose_for_scores(key_vectors, self.num_attention_heads, self.attention_head_size)
        value_vectors = self._transpose_for_scores(value_vectors, self.num_attention_heads, self.attention_head_size)

        assert query_vectors.shape[-1] == self.attention_head_size
        assert key_vectors.shape[-1] == self.attention_head_size
        assert value_vectors.shape[-1] == self.attention_head_size

        if self.chunk_length is None:
            assert self.num_chunks_before == 0 and self.num_chunks_after == 0

        key_vectors = key_vectors / torch.sqrt(
            torch.tensor(self.attention_head_size, device=key_vectors.device, dtype=key_vectors.dtype)
        )

        # chunk vectors
        query_vectors = self._split_dim_by(
            query_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size,
        )  # B x Num_Attn_Head x Seq_Len // chunk_len x chunk_len  x  attn_head_size
        key_vectors = self._split_dim_by(
            key_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size,
        )  # B x Num_Attn_Head x Seq_Len // chunk_len x chunk_len  x  attn_head_size
        value_vectors = self._split_dim_by(
            value_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size,
        )  # B x Num_Attn_Head x Seq_Len // chunk_len x chunk_len  x  attn_head_size

        # chunk indices
        indices = torch.arange(sequence_length, device=query_vectors.device).repeat(
            batch_size, self.num_attention_heads, 1
        )
        query_indices = self._split_dim_by(indices, -1, self.chunk_length, self.num_attention_heads)
        key_value_indices = self._split_dim_by(indices, -1, self.chunk_length, self.num_attention_heads)

        # append chunks before and after
        key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
        value_vectors = self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)
        key_value_indices = self._look_adjacent(key_value_indices, self.num_chunks_before, self.num_chunks_after)

        # chunk attention mask and look before and after
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.uint8)[:, None, :]
            attention_mask = self._split_dim_by(attention_mask, -1, self.chunk_length, 1)
            attention_mask_key = self._look_adjacent(attention_mask, self.num_chunks_before, self.num_chunks_after)

        # get logits and dots
        # query_key_dots shape is `[batch_size, num_attn_heads, seq_len // chunk_length, chunk_length, chunk_length * (1 + num_chunks_before + num_chunks_after)]`
        query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))

        # free memory
        del query_vectors, key_vectors

        mask = None
        # Causal mask
        if self.is_decoder is True:
            mask = torch.ge(query_indices.unsqueeze(-1), key_value_indices.unsqueeze(-2)).to(query_indices.device)

        # Attention mask
        if attention_mask is not None:
            # create attn_mask
            attn_mask = (attention_mask.unsqueeze(-1) * attention_mask_key.unsqueeze(-2)).expand(query_key_dots.shape)
            # multiply by casaul mask if necessary
            if mask is not None:
                mask = mask * attn_mask
            else:
                mask = attn_mask

            # free memory
            del attn_mask, attention_mask, attention_mask_key

        if mask is not None:
            # get mask tensor depending on half precision or not
            if query_key_dots.dtype == torch.float16:
                mask_value = self.mask_value_float16
            else:
                mask_value = self.mask_value_float32

            query_key_dots = torch.where(mask, query_key_dots, mask_value)

        # free memory
        del mask

        logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        attention_probs = torch.exp(query_key_dots - logits)

        # free memory
        del logits

        # dropout
        attention_probs = nn.functional.dropout(attention_probs, self.dropout, self.training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # attend values
        out_vectors = torch.matmul(attention_probs, value_vectors)

        # free memory
        del value_vectors

        # merge chunk length
        out_vectors = self._merge_by_middle_dim(out_vectors, self.num_attention_heads)

        assert out_vectors.shape == (batch_size, self.num_attention_heads, sequence_length, self.attention_head_size,)

        out_vectors = self._transpose_for_output(out_vectors, self.num_attention_heads, self.attention_head_size)

        if do_output_attentions is False:
            attention_probs = ()

        return LocalSelfAttentionOutput(hidden_states=out_vectors, attention_probs=attention_probs)


class ReformerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        all_head_size = config.num_attention_heads * config.attention_head_size
        self.dense = nn.Linear(all_head_size, config.hidden_size, bias=False)
        self.dropout = config.hidden_dropout_prob

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, self.dropout, self.training)
        return hidden_states


class ReformerAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.layer_norm = ReformerLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn_layers = config.attn_layers

        if len(set(self.attn_layers)) == 1 and self.attn_layers[0] == "lsh":
            self.self_attention = LSHSelfAttention(config)
        elif len(set(self.attn_layers)) == 1 and self.attn_layers[0] == "local":
            self.self_attention = LocalSelfAttention(config)
        elif len(set(self.attn_layers)) == 2 and set(self.attn_layers) == set(["lsh", "local"]):
            # get correct attn_layers
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
        do_output_attentions=False,
        buckets=None,
    ):
        hidden_states = self.layer_norm(hidden_states)

        # use cached buckets for backprob if buckets not None for LSHSelfAttention
        self_attention_outputs = self.self_attention(
            hidden_states=hidden_states,
            head_mask=head_mask,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            do_output_attentions=do_output_attentions,
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
        self.dense = nn.Linear(config.hidden_size, config.feed_forward_size)
        self.dropout = config.hidden_dropout_prob
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, self.dropout, self.training)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states


class ReformerFeedForwardOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.feed_forward_size, config.hidden_size)
        self.dropout = config.hidden_dropout_prob

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, self.dropout, self.training)
        return hidden_states


class ChunkReformerFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len_dim = 1
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.layer_norm = ReformerLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
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
        self.feed_forward = ChunkReformerFeedForward(config)
        # dropout requires to have the same
        # seed for forward and backward pass
        self.attention_seed = None
        self.feed_forward_seed = None

    def _init_attention_seed(self):
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
        do_output_attentions=False,
    ):
        with torch.no_grad():
            # every forward pass we sample a different seed
            # for dropout and save seed for forward fn in backward
            # to have correct dropout
            self._init_attention_seed()
            attn_outputs = self.attention(
                hidden_states=hidden_states,
                head_mask=head_mask,
                attention_mask=attention_mask,
                num_hashes=num_hashes,
                do_output_attentions=do_output_attentions,
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
        # This code is heavily inspired by https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py

        with torch.enable_grad():
            next_attn_output.requires_grad = True

            # set seed to have correct dropout
            torch.manual_seed(self.feed_forward_seed)
            # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
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
        do_output_hidden_states,
        do_output_attentions,
    ):

        all_buckets = ()
        hidden_states, attn_output = torch.chunk(hidden_states, 2, dim=-1)
        for layer, layer_head_mask in zip(layers, head_mask):
            if do_output_hidden_states is True:
                all_hidden_states.append(hidden_states)

            layer_outputs = layer(
                prev_attn_output=attn_output,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                num_hashes=num_hashes,
                do_output_attentions=do_output_attentions,
            )
            attn_output = layer_outputs.attn_output
            hidden_states = layer_outputs.hidden_states
            all_buckets = all_buckets + (layer_outputs.buckets,)

            if do_output_attentions:
                all_attentions.append(layer_outputs.attention_probs)

        # Add last layer
        if do_output_hidden_states is True:
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
        self.layers = nn.ModuleList([ReformerLayer(config, i) for i in range(config.num_hidden_layers)])
        # Reformer is using Rev Nets, thus last layer outputs are concatenated and
        # Layer Norm is done over 2 * hidden_size
        self.layer_norm = ReformerLayerNorm(2 * config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = config.hidden_dropout_prob

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        do_output_hidden_states=False,
        do_output_attentions=False,
    ):
        # hidden_states and attention lists to be filled if wished
        all_hidden_states = []
        all_attentions = []

        # Make this work
        hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        hidden_states = _ReversibleFunction.apply(
            hidden_states,
            self.layers,
            attention_mask,
            head_mask,
            num_hashes,
            all_hidden_states,
            all_attentions,
            do_output_hidden_states,
            do_output_attentions,
        )

        # Apply layer norm to concatenated hidden states
        hidden_states = self.layer_norm(hidden_states)

        # Apply dropout
        hidden_states = nn.functional.dropout(hidden_states, self.dropout, self.training)

        return ReformerEncoderOutput(
            hidden_states=hidden_states, all_hidden_states=all_hidden_states, all_attentions=all_attentions
        )


class ReformerPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = ReformerConfig
    pretrained_model_archive_map = REFORMER_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "reformer"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, AxialPositionEmbeddings):
            for weight in module.weights:
                torch.nn.init.normal_(weight, std=self.config.axial_norm_std)
        elif isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                # TODO(PVP): discuss with Thom if necessary here to use different init
        #                torch.nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, ReformerLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            # TODO(PVP): discuss with Thom if necessary here to use different init


#            module.bias.data.normal_(mean=0.0, std=1e-6)


class ReformerModel(ReformerPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        num_hashes=None,
        do_output_hidden_states=False,
        do_output_attentions=False,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``do_output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        """
        # TODO(PVP): delete when PR to change output_attentions is made
        do_output_attentions = self.config.output_attentions
        do_output_hidden_states = self.config.output_hidden_states

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
        has_to_pad_to_match_chunk_length = input_shape[-1] % least_common_mult_chunk_length != 0
        if has_to_pad_to_match_chunk_length is True:
            # pad input
            input_ids, inputs_embeds, attention_mask, position_ids, input_shape = self._pad_to_mult_of_chunk_length(
                input_ids,
                inputs_embeds,
                attention_mask,
                position_ids,
                input_shape,
                least_common_mult_chunk_length,
                device,
            )

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            head_mask=head_mask,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            do_output_hidden_states=do_output_hidden_states,
            do_output_attentions=do_output_attentions,
        )
        sequence_output = encoder_outputs.hidden_states

        # if padding was applied
        if has_to_pad_to_match_chunk_length is True:
            sequence_output = sequence_output[:, :orig_sequence_length]

        outputs = (sequence_output,)

        # TODO(PVP): Replace by named tuple after namedtuples are introduced in the library.
        if do_output_hidden_states is True:
            outputs = outputs + (encoder_outputs.all_hidden_states,)
        if do_output_attentions is True:
            outputs = outputs + (encoder_outputs.all_attentions,)
        return outputs

    def _pad_to_mult_of_chunk_length(
        self, input_ids, inputs_embeds, attention_mask, position_ids, input_shape, padded_seq_length, device
    ):
        padding_length = padded_seq_length - input_shape[-1] % padded_seq_length
        assert (
            self.training is False
        ), "Sequence Length {} has to be a multiple of least common multiple chunk_length {} if training. Please consider padding the input to a length of {}.".format(
            input_shape[-2], padded_seq_length, input_shape[-2] + padding_length
        )

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

        # Extend `input_embeds` with padding to match least common multiple chunk_length
        if inputs_embeds is not None:
            padded_inputs_embeds = self.embeddings(padded_input_ids, position_ids)
            inputs_embeds = torch.cat([inputs_embeds, padded_inputs_embeds], dim=-2)
            input_shape = inputs_embeds.size()

        return input_ids, inputs_embeds, attention_mask, position_ids, input_shape


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


class ReformerModelWithLMHead(ReformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.reformer = ReformerModel(config)
        self.lm_head = ReformerOnlyLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def tie_weights(self):
        pass

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        num_hashes=None,
        labels=None,
        do_output_hidden_states=False,
        do_output_attentions=False,
    ):

        reformer_outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            num_hashes=num_hashes,
            do_output_hidden_states=do_output_hidden_states,
            do_output_attentions=do_output_attentions,
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
