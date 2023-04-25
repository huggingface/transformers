# coding=utf-8
# Copyright 2022 University of Wisconsin-Madison and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch MRA model."""


import math
import os
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.cpp_extension import load

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_ninja_available,
    is_torch_cuda_available,
    logging,
)
from .configuration_mra import MRAConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "uw-madison/mra-base-512-4"
_CONFIG_FOR_DOC = "MRAConfig"
_TOKENIZER_FOR_DOC = "AutoTokenizer"

MRA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "uw-madison/mra-base-512-4",
    # See all MRA models at https://huggingface.co/models?filter=mra
]


def load_cuda_kernels():
    global cuda_kernel
    curr_path = os.path.dirname(os.path.realpath(__file__))
    src_files = ["cuda_kernel.cu", "cuda_launch.cu", "torch_extension.cpp"]
    src_files = [os.path.join(curr_path, file) for file in src_files]
    cuda_kernel = load("cuda_kernel", src_files, verbose=True)

    import cuda_kernel


if is_torch_cuda_available() and is_ninja_available():
    logger.info("Loading custom CUDA kernels...")

    try:
        load_cuda_kernels()
    except Exception as e:
        logger.warning(
            "Failed to load CUDA kernels. MRA requires custom CUDA kernels. Please verify that compatible versions of"
            f" PyTorch and CUDA Toolkit are installed: {e}"
        )
        cuda_kernel = None
else:
    cuda_kernel = None


def sparse_max(sparse_C, indices, A_num_block, B_num_block):
    """
    Computes maximum values for softmax stability.
    """
    assert len(sparse_C.size()) == 4
    assert len(indices.size()) == 2
    assert sparse_C.size(2) == 32
    assert sparse_C.size(3) == 32

    index_vals = sparse_C.max(dim=-2).values.transpose(-1, -2)
    index_vals = index_vals.contiguous()

    indices = indices.int()
    indices = indices.contiguous()

    max_vals, max_vals_scatter = cuda_kernel.index_max(index_vals, indices, A_num_block, B_num_block)
    max_vals_scatter = max_vals_scatter.transpose(-1, -2)[:, :, None, :]

    return max_vals, max_vals_scatter


def sparse_mask_B(mask, indices, block_size=32):
    """
    Converts attention mask to a sparse mask for high resolution logits.
    """
    assert len(mask.size()) == 2
    assert len(indices.size()) == 2
    assert mask.shape[0] == indices.shape[0]

    batch_size, seq_len = mask.shape
    num_block = seq_len // block_size

    batch_idx = torch.arange(indices.size(0), dtype=torch.long, device=indices.device)
    mask = mask.reshape(batch_size, num_block, block_size)
    mask = mask[batch_idx[:, None], (indices % num_block).long(), :]

    return mask


def mm_to_sparse(dense_A, dense_B, indices, block_size=32):
    """
    Performs Sampled Dense Matrix Multiplication.
    """
    batch_size, A_size, dim = dense_A.size()
    _, B_size, dim = dense_B.size()
    assert A_size % block_size == 0
    assert B_size % block_size == 0

    dense_A = dense_A.reshape(batch_size, A_size // block_size, block_size, dim).transpose(-1, -2)
    dense_B = dense_B.reshape(batch_size, B_size // block_size, block_size, dim).transpose(-1, -2)

    assert len(dense_A.size()) == 4
    assert len(dense_B.size()) == 4
    assert len(indices.size()) == 2
    assert dense_A.size(3) == 32
    assert dense_B.size(3) == 32

    dense_A = dense_A.contiguous()
    dense_B = dense_B.contiguous()

    indices = indices.int()
    indices = indices.contiguous()

    assert dense_A.is_contiguous()
    assert dense_B.is_contiguous()
    assert indices.is_contiguous()

    return cuda_kernel.mm_to_sparse(dense_A, dense_B, indices.int())


def sparse_dense_mm(sparse_A, indices, dense_B, A_num_block, block_size=32):
    """
    Performs matrix multiplication of a sparse matrix with a dense matrix.
    """
    batch_size, B_size, dim = dense_B.size()

    assert B_size % block_size == 0
    assert sparse_A.size(2) == block_size
    assert sparse_A.size(3) == block_size

    dense_B = dense_B.reshape(batch_size, B_size // block_size, block_size, dim).transpose(-1, -2)

    assert len(sparse_A.size()) == 4
    assert len(dense_B.size()) == 4
    assert len(indices.size()) == 2
    assert sparse_A.size(2) == 32
    assert sparse_A.size(3) == 32
    assert dense_B.size(3) == 32

    sparse_A = sparse_A.contiguous()

    indices = indices.int()
    indices = indices.contiguous()
    dense_B = dense_B.contiguous()

    assert sparse_A.is_contiguous()
    assert indices.is_contiguous()
    assert dense_B.is_contiguous()

    dense_C = cuda_kernel.sparse_dense_mm(sparse_A, indices, dense_B, A_num_block)
    dense_C = dense_C.transpose(-1, -2).reshape(batch_size, A_num_block * block_size, dim)
    return dense_C


def transpose_indices(indices, dim_1_block, dim_2_block):
    return ((indices % dim_2_block) * dim_1_block + torch.div(indices, dim_2_block, rounding_mode="floor")).long()


class SampledDenseMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dense_A, dense_B, indices, block_size):
        sparse_AB = mm_to_sparse(dense_A, dense_B, indices, block_size)
        ctx.save_for_backward(dense_A, dense_B, indices)
        ctx.block_size = block_size
        return sparse_AB

    @staticmethod
    def backward(ctx, grad):
        dense_A, dense_B, indices = ctx.saved_tensors
        block_size = ctx.block_size
        A_num_block = dense_A.size(1) // block_size
        B_num_block = dense_B.size(1) // block_size
        indices_T = transpose_indices(indices, A_num_block, B_num_block)
        grad_B = sparse_dense_mm(grad.transpose(-1, -2), indices_T, dense_A, B_num_block)
        grad_A = sparse_dense_mm(grad, indices, dense_B, A_num_block)
        return grad_A, grad_B, None, None

    @staticmethod
    def operator_call(dense_A, dense_B, indices, block_size=32):
        return SampledDenseMM.apply(dense_A, dense_B, indices, block_size)


class SparseDenseMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse_A, indices, dense_B, A_num_block):
        sparse_AB = sparse_dense_mm(sparse_A, indices, dense_B, A_num_block)
        ctx.save_for_backward(sparse_A, indices, dense_B)
        ctx.A_num_block = A_num_block
        return sparse_AB

    @staticmethod
    def backward(ctx, grad):
        sparse_A, indices, dense_B = ctx.saved_tensors
        A_num_block = ctx.A_num_block
        B_num_block = dense_B.size(1) // sparse_A.size(-1)
        indices_T = transpose_indices(indices, A_num_block, B_num_block)
        grad_B = sparse_dense_mm(sparse_A.transpose(-1, -2), indices_T, grad, B_num_block)
        grad_A = mm_to_sparse(grad, dense_B, indices)
        return grad_A, None, grad_B, None

    @staticmethod
    def operator_call(sparse_A, indices, dense_B, A_num_block):
        return SparseDenseMM.apply(sparse_A, indices, dense_B, A_num_block)


class ReduceSum:
    @staticmethod
    def operator_call(sparse_A, indices, A_num_block, B_num_block):
        batch_size, num_block, block_size, _ = sparse_A.size()

        assert len(sparse_A.size()) == 4
        assert len(indices.size()) == 2

        _, _, block_size, _ = sparse_A.size()
        batch_size, num_block = indices.size()

        sparse_A = sparse_A.sum(dim=2).reshape(batch_size * num_block, block_size)

        batch_idx = torch.arange(indices.size(0), dtype=torch.long, device=indices.device)
        global_idxes = (
            torch.div(indices, B_num_block, rounding_mode="floor").long() + batch_idx[:, None] * A_num_block
        ).reshape(batch_size * num_block)
        temp = torch.zeros((batch_size * A_num_block, block_size), dtype=sparse_A.dtype, device=sparse_A.device)
        output = temp.index_add(0, global_idxes, sparse_A).reshape(batch_size, A_num_block, block_size)

        output = output.reshape(batch_size, A_num_block * block_size)
        return output


def get_low_resolution_logit(Q, K, block_size, mask=None, V=None):
    """
    Compute low resolution approximation.
    """
    batch_size, seq_len, head_dim = Q.size()

    num_block_per_row = seq_len // block_size

    V_hat = None
    if mask is not None:
        token_count = mask.reshape(batch_size, num_block_per_row, block_size).sum(dim=-1)
        Q_hat = Q.reshape(batch_size, num_block_per_row, block_size, head_dim).sum(dim=-2) / (
            token_count[:, :, None] + 1e-6
        )
        K_hat = K.reshape(batch_size, num_block_per_row, block_size, head_dim).sum(dim=-2) / (
            token_count[:, :, None] + 1e-6
        )
        if V is not None:
            V_hat = V.reshape(batch_size, num_block_per_row, block_size, head_dim).sum(dim=-2) / (
                token_count[:, :, None] + 1e-6
            )
    else:
        token_count = block_size * torch.ones(batch_size, num_block_per_row, dtype=torch.float, device=Q.device)
        Q_hat = Q.reshape(batch_size, num_block_per_row, block_size, head_dim).mean(dim=-2)
        K_hat = K.reshape(batch_size, num_block_per_row, block_size, head_dim).mean(dim=-2)
        if V is not None:
            V_hat = V.reshape(batch_size, num_block_per_row, block_size, head_dim).mean(dim=-2)

    low_resolution_logit = torch.matmul(Q_hat, K_hat.transpose(-1, -2)) / math.sqrt(head_dim)

    low_resolution_logit_row_max = low_resolution_logit.max(dim=-1, keepdims=True).values

    if mask is not None:
        low_resolution_logit = (
            low_resolution_logit - 1e4 * ((token_count[:, None, :] * token_count[:, :, None]) < 0.5).float()
        )

    return low_resolution_logit, token_count, low_resolution_logit_row_max, V_hat


def get_block_idxes(
    low_resolution_logit, num_blocks, approx_mode, initial_prior_first_n_blocks, initial_prior_diagonal_n_blocks
):
    """
    Compute the indices of the subset of components to be used in the approximation.
    """
    batch_size, total_blocks_per_row, _ = low_resolution_logit.shape

    if initial_prior_diagonal_n_blocks > 0:
        offset = initial_prior_diagonal_n_blocks // 2
        temp_mask = torch.ones(total_blocks_per_row, total_blocks_per_row, device=low_resolution_logit.device)
        diagonal_mask = torch.tril(torch.triu(temp_mask, diagonal=-offset), diagonal=offset)
        low_resolution_logit = low_resolution_logit + diagonal_mask[None, :, :] * 5e3

    if initial_prior_first_n_blocks > 0:
        low_resolution_logit[:, :initial_prior_first_n_blocks, :] = (
            low_resolution_logit[:, :initial_prior_first_n_blocks, :] + 5e3
        )
        low_resolution_logit[:, :, :initial_prior_first_n_blocks] = (
            low_resolution_logit[:, :, :initial_prior_first_n_blocks] + 5e3
        )

    top_k_vals = torch.topk(
        low_resolution_logit.reshape(batch_size, -1), num_blocks, dim=-1, largest=True, sorted=False
    )
    indices = top_k_vals.indices

    if approx_mode == "full":
        threshold = top_k_vals.values.min(dim=-1).values
        high_resolution_mask = (low_resolution_logit >= threshold[:, None, None]).float()
    elif approx_mode == "sparse":
        high_resolution_mask = None
    else:
        raise ValueError(f"{approx_mode} is not a valid approx_model value.")

    return indices, high_resolution_mask


def mra2_attention(
    Q,
    K,
    V,
    mask,
    num_blocks,
    approx_mode,
    block_size=32,
    initial_prior_first_n_blocks=0,
    initial_prior_diagonal_n_blocks=0,
):
    """
    Use MRA to approximate self-attention.
    """
    if cuda_kernel is None:
        return torch.zeros_like(Q).requires_grad_()

    batch_size, num_head, seq_len, head_dim = Q.size()
    meta_batch = batch_size * num_head

    assert seq_len % block_size == 0
    num_block_per_row = seq_len // block_size

    Q = Q.reshape(meta_batch, seq_len, head_dim)
    K = K.reshape(meta_batch, seq_len, head_dim)
    V = V.reshape(meta_batch, seq_len, head_dim)

    """
    mask = (
        None if torch.all(mask == 1).item() else mask[:, None, :].repeat(1, num_head, 1).reshape(meta_batch, seq_len)
    )
    """

    if mask is not None:
        Q = Q * mask[:, :, None]
        K = K * mask[:, :, None]
        V = V * mask[:, :, None]

    if approx_mode == "full":
        low_resolution_logit, token_count, low_resolution_logit_row_max, V_hat = get_low_resolution_logit(
            Q, K, block_size, mask, V
        )
    elif approx_mode == "sparse":
        with torch.no_grad():
            low_resolution_logit, token_count, low_resolution_logit_row_max, _ = get_low_resolution_logit(
                Q, K, block_size, mask
            )
    else:
        raise Exception('approx_mode must be "full" or "sparse"')

    with torch.no_grad():
        low_resolution_logit_normalized = low_resolution_logit - low_resolution_logit_row_max
        indices, high_resolution_mask = get_block_idxes(
            low_resolution_logit_normalized,
            num_blocks,
            approx_mode,
            initial_prior_first_n_blocks,
            initial_prior_diagonal_n_blocks,
        )

    high_resolution_logit = SampledDenseMM.operator_call(Q, K, indices, block_size=block_size) / math.sqrt(head_dim)
    max_vals, max_vals_scatter = sparse_max(high_resolution_logit, indices, num_block_per_row, num_block_per_row)
    high_resolution_logit = high_resolution_logit - max_vals_scatter
    if mask is not None:
        high_resolution_logit = high_resolution_logit - 1e4 * (1 - sparse_mask_B(mask, indices)[:, :, :, None])
    high_resolution_attn = torch.exp(high_resolution_logit)
    high_resolution_attn_out = SparseDenseMM.operator_call(high_resolution_attn, indices, V, num_block_per_row)
    high_resolution_normalizer = ReduceSum.operator_call(
        high_resolution_attn, indices, num_block_per_row, num_block_per_row
    )

    if approx_mode == "full":
        low_resolution_attn = (
            torch.exp(low_resolution_logit - low_resolution_logit_row_max - 1e4 * high_resolution_mask)
            * token_count[:, None, :]
        )

        low_resolution_attn_out = (
            torch.matmul(low_resolution_attn, V_hat)[:, :, None, :]
            .repeat(1, 1, block_size, 1)
            .reshape(meta_batch, seq_len, head_dim)
        )
        low_resolution_normalizer = (
            low_resolution_attn.sum(dim=-1)[:, :, None].repeat(1, 1, block_size).reshape(meta_batch, seq_len)
        )

        log_correction = low_resolution_logit_row_max.repeat(1, 1, block_size).reshape(meta_batch, seq_len) - max_vals
        if mask is not None:
            log_correction = log_correction * mask

        low_resolution_corr = torch.exp(log_correction * (log_correction <= 0).float())
        low_resolution_attn_out = low_resolution_attn_out * low_resolution_corr[:, :, None]
        low_resolution_normalizer = low_resolution_normalizer * low_resolution_corr

        high_resolution_corr = torch.exp(-log_correction * (log_correction > 0).float())
        high_resolution_attn_out = high_resolution_attn_out * high_resolution_corr[:, :, None]
        high_resolution_normalizer = high_resolution_normalizer * high_resolution_corr

        attn = (high_resolution_attn_out + low_resolution_attn_out) / (
            high_resolution_normalizer[:, :, None] + low_resolution_normalizer[:, :, None] + 1e-6
        )

    elif approx_mode == "sparse":
        attn = high_resolution_attn_out / (high_resolution_normalizer[:, :, None] + 1e-6)
    else:
        raise Exception('config.approx_mode must be "full" or "sparse"')

    if mask is not None:
        attn = attn * mask[:, :, None]

    attn = attn.reshape(batch_size, num_head, seq_len, head_dim)

    return attn


class MRAEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings + 2, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)) + 2)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MRASelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = (
            position_embedding_type if position_embedding_type is not None else config.position_embedding_type
        )

        self.num_block = (config.max_position_embeddings // 32) * config.block_per_row
        self.num_block = min(self.num_block, int((config.max_position_embeddings // 32) ** 2))

        self.approx_mode = config.approx_mode
        self.initial_prior_first_n_blocks = config.initial_prior_first_n_blocks
        self.initial_prior_diagonal_n_blocks = config.initial_prior_diagonal_n_blocks

    def transpose_for_scores(self, layer):
        new_layer_shape = layer.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        layer = layer.view(*new_layer_shape)
        return layer.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        batch_size, num_heads, seq_len, head_dim = query_layer.size()

        # revert changes made by get_extended_attention_mask
        attention_mask = 1.0 + attention_mask / 10000.0
        attention_mask = (
            attention_mask.squeeze().repeat(1, num_heads, 1).reshape(batch_size * num_heads, seq_len).int()
        )

        # The CUDA kernels are most efficient with inputs whose size is a multiple of a GPU's warp size (32). Inputs
        # smaller than this are padded with zeros.
        gpu_warp_size = 32

        if head_dim < gpu_warp_size:
            pad_size = batch_size, num_heads, seq_len, gpu_warp_size - head_dim

            query_layer = torch.cat([query_layer, torch.zeros(pad_size, device=query_layer.device)], dim=-1)
            key_layer = torch.cat([key_layer, torch.zeros(pad_size, device=key_layer.device)], dim=-1)
            value_layer = torch.cat([value_layer, torch.zeros(pad_size, device=value_layer.device)], dim=-1)

        context_layer = mra2_attention(
            query_layer.float(),
            key_layer.float(),
            value_layer.float(),
            attention_mask.float(),
            self.num_block,
            approx_mode=self.approx_mode,
            initial_prior_first_n_blocks=self.initial_prior_first_n_blocks,
            initial_prior_diagonal_n_blocks=self.initial_prior_diagonal_n_blocks,
        )

        if head_dim < gpu_warp_size:
            context_layer = context_layer[:, :, :, :head_dim]

        context_layer = context_layer.reshape(batch_size, num_heads, seq_len, head_dim)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, context_layer) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class MRASelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.yoso.modeling_yoso.YosoAttention with Yoso->MRA
class MRAAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = MRASelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = MRASelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        self_outputs = self.self(hidden_states, attention_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class MRAIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class MRAOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.yoso.modeling_yoso.YosoLayer with Yoso->MRA
class MRALayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MRAAttention(config)
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = MRAIntermediate(config)
        self.output = MRAOutput(config)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.yoso.modeling_yoso.YosoEncoder with Yoso->MRA
class MRAEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([MRALayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, output_attentions)

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform
class MRAPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->MRA
class MRALMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = MRAPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->MRA
class MRAOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = MRALMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# Copied from transformers.models.yoso.modeling_yoso.YosoPreTrainedModel with Yoso->MRA,yoso->mra
class MRAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MRAConfig
    base_model_prefix = "mra"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MRAEncoder):
            module.gradient_checkpointing = value


MRA_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MRAConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MRA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MRA Model transformer outputting raw hidden-states without any specific head on top.",
    MRA_START_DOCSTRING,
)
# Copied from transformers.models.yoso.modeling_yoso.YosoModel with YOSO->MRA,Yoso->MRA
class MRAModel(MRAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = MRAEmbeddings(config)
        self.encoder = MRAEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings("""MRA Model with a `language modeling` head on top.""", MRA_START_DOCSTRING)
# Copied from transformers.models.yoso.modeling_yoso.YosoForMaskedLM with YOSO->MRA,Yoso->MRA,yoso->mra
class MRAForMaskedLM(MRAPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        "cls.predictions.decoder.bias",
        "cls.predictions.decoder.weight",
        "embeddings.position_ids",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.mra = MRAModel(config)
        self.cls = MRAOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Copied from transformers.models.yoso.modeling_yoso.YosoClassificationHead with Yoso->MRA
class MRAClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.config = config

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """MRA Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks.""",
    MRA_START_DOCSTRING,
)
# Copied from transformers.models.yoso.modeling_yoso.YosoForSequenceClassification with YOSO->MRA,Yoso->MRA,yoso->mra
class MRAForSequenceClassification(MRAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.mra = MRAModel(config)
        self.classifier = MRAClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """MRA Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks.""",
    MRA_START_DOCSTRING,
)
# Copied from transformers.models.yoso.modeling_yoso.YosoForMultipleChoice with YOSO->MRA,Yoso->MRA,yoso->mra
class MRAForMultipleChoice(MRAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.mra = MRAModel(config)
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.mra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = outputs[0]  # (bs * num_choices, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs * num_choices, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        logits = self.classifier(pooled_output)

        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """MRA Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.""",
    MRA_START_DOCSTRING,
)
# Copied from transformers.models.yoso.modeling_yoso.YosoForTokenClassification with YOSO->MRA,Yoso->MRA,yoso->mra
class MRAForTokenClassification(MRAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.mra = MRAModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """MRA Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).""",
    MRA_START_DOCSTRING,
)
# Copied from transformers.models.yoso.modeling_yoso.YosoForQuestionAnswering with YOSO->MRA,Yoso->MRA,yoso->mra
class MRAForQuestionAnswering(MRAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.mra = MRAModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
