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
"""PyTorch REFORMER model. """

import logging

import torch
from torch import nn

from .activations import gelu, gelu_new, swish

# DELETE later
import numpy


logger = logging.getLogger(__name__)

REFORMER_PRETRAINED_MODEL_ARCHIVE_MAP = {}
# TODO: fill with pretrained model weights
#    "reformer-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/reformer/reformer-base-uncased-pytorch_model.bin",
#    "reformer-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/reformer/reformer-large-uncased-pytorch_model.bin",
#    "reformer-base-cased": "https://s3.amazonaws.com/models.huggingface.co/reformer/reformer-base-cased-pytorch_model.bin",
#    "reformer-large-cased": "https://s3.amazonaws.com/models.huggingface.co/reformer/reformer-large-cased-pytorch_model.bin",


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}
ReformerLayerNorm = torch.nn.LayerNorm


class LSHSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.hash_seed = config.seed
        self.num_hashes = config.num_hashes
        self.num_buckets = config.num_buckets
        self.chunk_length = config.chunk_length
        self.num_chunks_before = config.num_chunks_before
        self.num_chunks_after = config.num_chunks_after
        self.output_attentions = config.output_attentions
        self.is_decoder = config.is_decoder

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.query_key = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=False)

        self.dropout = config.attention_probs_dropout_prob

    def forward(
        self,
        hidden_states,
        head_mask=None,
    ):

        # get SeqLen and BatchSize
        sequence_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]

        mixed_query_key_vectors = self.query_key(hidden_states)
        mixed_value_vectors = self.value(hidden_states)

        query_key_vectors = self._transpose_for_scores(mixed_query_key_vectors)
        value_vectors = self._transpose_for_scores(mixed_value_vectors)

        assert query_key_vectors.shape[-1] == self.attention_head_size
        assert value_vectors.shape[-1] == self.attention_head_size

        # hash query key vectors into buckets
        buckets = self._hash_vectors(query_key_vectors)
        assert int(buckets.shape[-1]) == self.num_hashes * sequence_length

        ticker, undo_ticker = self._get_ticker_and_undo_ticker(sequence_length, buckets)

        query_key_vectors = self._gather_by_expansion(query_key_vectors, ticker)
        value_vectors = self._gather_by_expansion(value_vectors, ticker)

        # q_info = ticker

        query_key_vectors = self._split_dim_by(query_key_vectors, -1, self.chunk_length)
        value_vectors = self._split_dim_by(value_vectors, -1, self.chunk_length)
        ticker = self._split_dim_by(ticker, -1, self.chunk_length)

        # Optionally include adjacent chunks.
        if self.chunk_length is None:
            assert self.num_chunks_before == 0 and self.num_chunks_after == 0

        key_vectors = self._len_and_dim_norm(query_key_vectors)

        out_vectors, logits, attention_probs = self._attend(query_key_vectors, key_vectors, value_vectors, ticker, undo_ticker, head_mask)

        if self.num_hashes > 1:
            out_vectors = self._split_dim_by(out_vectors, self.num_hashes, sequence_length)
            logits = self._split_dim_by(logits, self.num_hashes, sequence_length).unsqueeze(-1)

            probs_vectors = torch.exp(logits - torch.logsumexp(logits, dim=2, keepdim=True))
            out_vectors = torch.sum(out_vectors * probs_vectors, dim=2)

        assert out_vectors.shape == (batch_size, self.num_attention_heads, sequence_length, self.attention_head_size)

        out_vectors = self._transpose_for_output(out_vectors)
        outputs = (out_vectors, attention_probs) if self.output_attentions else (out_vectors,)
        return outputs

    def _hash_vectors(self, vectors):
        batch_size = vectors.shape[0]

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        if isinstance(self.num_buckets, int):
            assert self.num_buckets % 2 == 0, 'There should be an even number of bucktes, but `self.num_bucktes`: {}'.format(self.num_buckets)
            rotation_size = self.num_buckets
            num_buckets = self.num_buckets
        else:
            # Factorize the hash if self.num_buckets is a list or tuple
            rotation_size, num_buckets = 0, 1
            for num_bucket in self.num_buckets:
                assert num_bucket % 2 == 0, 'The number of buckets should be even, but `num_bucket`: {}'.format(num_bucket)
                rotation_size += num_bucket
                num_buckets *= num_bucket

        rotations_shape = (vectors.shape[-1], self.num_hashes, rotation_size // 2)

        # TODO: delete later when integration tests are ok
        # create a random self.attention_head_size x self.num_hashes x self.num_buckets/2
#        random_rotations = torch.randn(rotations_shape, device=vectors.device)
        numpy.random.seed(self.hash_seed)
        random_rotations = torch.tensor(numpy.random.normal(size=rotations_shape), dtype=torch.float32, device=vectors.device)
        # rotated_vectors has dim:
        # Output dim: Batch_Size x Num_Attn_Heads x Num_Hashes x Seq_Len x Num_Buckets/2
        # TODO: IMPORTANT: At the moment we use the same random rotation over all batches
        # and heads -> is that bad? It seems like in original reformer a different random
        # rotation is used over heads and batches
        rotated_vectors = torch.einsum('bmtd,dhr->bmhtr', vectors, random_rotations)

        if isinstance(self.num_buckets, int) or len(self.num_buckets) == 1:
            rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
            buckets = torch.argmax(rotated_vectors, dim=-1)
        else:
            # Get the buckets for them and combine.
            buckets, cur_sum, cur_product = None, 0, 1
            for num_bucket in self.num_buckets:
                rotated_vectors = rotated_vectors[..., cur_sum:cur_sum + (num_bucket // 2)]
                cur_sum += num_bucket // 2
                rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)

                if buckets is None:
                    buckets = torch.argmax(rotated_vectors, dim=-1)
                else:
                    buckets += cur_product * torch.argmax(rotated_vectors, dim=-1)

                cur_product *= num_bucket

        # buckets is now (Batch_size x Num_Attn_Heads x Num_Hashes x Seq_Len).
        # Next we add offsets so that bucket numbers from different hashing rounds don't overlap.
        offsets = torch.arange(self.num_hashes, device=vectors.device)
        offsets = torch.reshape(offsets * num_buckets, (-1, 1))

        # repeat same values for Batch_size and Num_Attn_Heads
        offsets = offsets.repeat(batch_size, self.num_attention_heads, 1, 1)
        offset_buckets = self._merge_by_middle_dim(buckets + offsets)

        return offset_buckets

    def _get_ticker_and_undo_ticker(self, sequence_length, buckets):
        batch_size = buckets.shape[0]

        # TODO: what is ticker? Is ticker something like indices?? Ask authors
        ticker = torch.arange(self.num_hashes * sequence_length, device=buckets.device)
        ticker = ticker.repeat(batch_size, self.num_attention_heads, 1)

        buckets_and_t = sequence_length * buckets + (ticker % sequence_length)

        # Hash-based sort
        sorted_ticker = torch.argsort(buckets_and_t, dim=-1)
        undo_sorted_ticker = torch.argsort(sorted_ticker, dim=-1)

        sorted_ticker = (sorted_ticker % sequence_length)
        return sorted_ticker, undo_sorted_ticker

    def _attend(self, query_vectors, key_vectors, value_vectors, ticker, undo_ticker, head_mask):
        key_vectors = self._look_adjacent(key_vectors)
        value_vectors = self._look_adjacent(value_vectors)

        # get logits and dots
        query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))

        # TODO(PVP): better naming
        query_info = ticker
        key_value_info = self._look_adjacent(ticker)

        # Causal mask
        if self.is_decoder:
            # TODO (PVP): This line can be improved in terms of memory. Mask should be inserted and does not have to be created each time layer is called
            mask = torch.lt(query_info.unsqueeze(-1), key_value_info.unsqueeze(-2)).long().to(query_info.device)
            query_key_dots = query_key_dots - mask * 1e9

        # Self mask
        # TODO (PVP): This line can be improved in terms of memory. Mask should be inserted and does not have to be created each time layer is called
        mask = torch.eq(query_info.unsqueeze(-1), key_value_info.unsqueeze(-2)).long().to(query_info.device)
        query_key_dots = query_key_dots - mask * 1e5

        # Note: Causal mask probably uses higher mask value (-1e9) than Self mask (-1e5) so that token is able to attend to itself when it has no other valid attention targets.

        # Note: Self mask is used because Q and K projection weights are shared.
        # From the reformer paper (https://arxiv.org/pdf/2001.04451.pdf):

        # " While attention to the future is not allowed, typical implementations of the
        # Transformer do allow a position to attend to itself.
        # Such behavior is undesirable in a shared-QK formulation because the dot-product
        # of a query vector with itself will almost always be greater than the dot product of a
        # query vector with a vector at another position. We therefore modify the masking
        # to forbid a token from attending to itself, except in situations
        # where a token has no other valid attention targets (e.g. the first token in a sequence) "

        logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        dots = torch.exp(query_key_dots - logits)

        # dropout
        dots = nn.functional.dropout(dots, self.dropout, self.training)

        # Mask heads if we want to
        if head_mask is not None:
            dots = dots * head_mask

        # attend values
        out_vectors = torch.matmul(dots, value_vectors)

        # merge chunk length
        logits = self._merge_by_middle_dim(logits).squeeze(-1)
        out_vectors = self._merge_by_middle_dim(out_vectors)

        expanded_undo_sort_indices = undo_ticker.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        out_vectors = torch.gather(out_vectors, 2, expanded_undo_sort_indices)
        logits = torch.gather(logits, 2, undo_ticker)

        return out_vectors, logits, dots

    def _look_adjacent(self, vectors):
        """ Used to implement attention between consecutive chunks.

            Args:
                vectors: array of shape [batch_size, num_attention_heads, n_chunks, chunk_len, ...]
            Returns:
                array of shape [n_chunks, N * chunk_len, ...], where
                N = (1 + n_chunks_before + n_chunks_after).
        """
        if self.num_chunks_before == 0 and self.num_chunks_after == 0:
            return vectors

        slices = []
        for i in range(-self.num_chunks_before, self.num_chunks_after + 1):
            if i == 0:
                slices.append(vectors)
            else:
                slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        return torch.cat(slices, dim=3)

    def _len_and_dim_norm(self, vectors):
        vectors = self._len_norm(vectors)
        vectors = vectors / torch.sqrt(torch.tensor(self.attention_head_size, device=vectors.device, dtype=torch.float32))
        return vectors

    def _len_norm(self, x, epsilon=1e-6):
        variance = torch.mean(x**2, -1, keepdim=True)
        norm_x = x / torch.sqrt(variance + epsilon)
        return norm_x

    def _gather_by_expansion(self, vectors, idxs):
        expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        vectors = vectors.repeat(1, 1, self.num_hashes, 1)
        return torch.gather(vectors, 2, expanded_idxs)

    def _transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.transpose(2, 1)

    def _transpose_for_output(self, x):
        x = x.permute(0, 2, 1, 3)
        return torch.reshape(x, (x.size()[0], -1, self.num_attention_heads * self.attention_head_size))

    def _split_dim_by(self, vectors, dim_factor_1, dim_factor_2):
        batch_size = vectors.shape[0]
        split_dim_shape = (batch_size, self.num_attention_heads, dim_factor_1, dim_factor_2)

        if len(vectors.shape) == 4:
            return torch.reshape(vectors, split_dim_shape + (self.attention_head_size,))
        elif len(vectors.shape) == 3:
            return torch.reshape(vectors, split_dim_shape)
        else:
            raise ValueError("Input vector rank should be one of [3, 4], but is: {}".format(len(vectors.shape)))

    def _merge_by_middle_dim(self, vectors):
        batch_size = vectors.shape[0]
        new_dim_shape = (batch_size, self.num_attention_heads, -1)

        if len(vectors.shape) == 5:
            return torch.reshape(vectors, new_dim_shape + (vectors.shape[-1],))
        elif len(vectors.shape) == 4:
            return torch.reshape(vectors, new_dim_shape)
        else:
            raise ValueError("Input vector rank should be one of [4, 5], but is: {}".format(len(vectors.shape)))


class ReformerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = config.hidden_dropout_prob

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, self.dropout, self.training)
        # residual connection
        output = (hidden_states + input_tensor)
        return output


class ReformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = ReformerLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attention = LSHSelfAttention(config)
        self.output = ReformerSelfOutput(config)

    def forward(
        self,
        prev_attention_output,
        hidden_states,
        head_mask=None,
    ):
        norm_hidden_states = self.layer_norm(hidden_states)
        self_attention_outputs = self.self_attention(
            norm_hidden_states, head_mask
        )

        # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
        # X_1 = prev_attention_output; X_2 = hidden_states
        attention_output = self.output(self_attention_outputs[0], prev_attention_output)
        outputs = (attention_output,) + self_attention_outputs[1:]  # add attentions if we output them
        return outputs


class ReformerFeedForwardDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
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
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = config.hidden_dropout_prob

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, self.dropout, self.training)
        output = (hidden_states + input_tensor)
        return output


class ReformerFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = ReformerLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = ReformerFeedForwardDense(config)
        self.output = ReformerFeedForwardOutput(config)

    def forward(self, attention_output, prev_attention_output):
        norm_attention_output = self.layer_norm(attention_output)
        dense_output = self.dense(norm_attention_output)
        output = self.output(dense_output, prev_attention_output)
        return output


class ReformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ReformerAttention(config)
        self.feed_forward = ReformerFeedForward(config)

    def forward(
        self,
        prev_attention_output,
        hidden_states,
        head_mask=None,
    ):

        # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
        # X_1 = prev_attention_output; X_2 = hidden_states
        # Y_1 = attention_output; Y_2 = output
        attention_outputs = self.attention(prev_attention_output, hidden_states, head_mask)
        attention_output = attention_outputs[0]
        output = self.feed_forward(attention_output, prev_attention_output)

        outputs = (attention_output, output) + attention_outputs[1:]
        return outputs
