# coding=utf-8
# Copyright 2022 The TorchAudio authors and The HuggingFace Inc. team,
# All rights reserved.
# Copyright 2017 Facebook Inc. (Soumith Chintala), All rights reserved.
#
# Licensed under BSD 2-Clause License (the "License");
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
""" PyTorch Emformer model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_emformer import EmformerConfig


logger = logging.get_logger(__name__)


_HIDDEN_STATES_START_POSITION = 1

# General docstring
_CONFIG_FOR_DOC = "EmformerConfig"
_PROCESSOR_FOR_DOC = "EmformerProcessor"

# Base docstring
_CHECKPOINT_FOR_DOC = "anton-l/emformer-base-librispeech"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 768]

# CTC docstring
_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 53.48


EMFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "anton-l/emformer-base-librispeech",
    # See all Emformer models at https://huggingface.co/models?filter=emformer
]


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype)
    padding_mask = padding_mask.expand(batch_size, max_length)
    padding_mask = padding_mask >= lengths.unsqueeze(1)
    return padding_mask


def _gen_padding_mask(
    utterance: torch.Tensor,
    right_context: torch.Tensor,
    summary: torch.Tensor,
    lengths: torch.Tensor,
    mems: torch.Tensor,
    left_context_key: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    time_length = right_context.size(0) + utterance.size(0) + summary.size(0)
    batch_size = right_context.size(1)
    if batch_size == 1:
        padding_mask = None
    else:
        right_context_blocks_length = time_length - torch.max(lengths).int() - summary.size(0)
        left_context_blocks_length = left_context_key.size(0) if left_context_key is not None else 0
        klengths = lengths + mems.size(0) + right_context_blocks_length + left_context_blocks_length
        padding_mask = _lengths_to_padding_mask(lengths=klengths)
    return padding_mask


def _gen_attention_mask_block(
    col_widths: List[int], col_mask: List[bool], num_rows: int, device: torch.device
) -> torch.Tensor:
    mask_block = [
        torch.ones(num_rows, col_width, device=device)
        if is_ones_col
        else torch.zeros(num_rows, col_width, device=device)
        for col_width, is_ones_col in zip(col_widths, col_mask)
    ]
    return torch.cat(mask_block, dim=1)


@dataclass
class EmformerModelOutput(BaseModelOutput):
    """
    Class for Emformer's outputs, with optional hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        left_context_states (`list(list(torch.FloatTensor))`, *optional*, returned when `is_streaming=True` is passed.
            Emformer internal state representation required for conditioning on the previously processed audio chunk
            during streaming inference.
    """

    left_context_states: Optional[List[List[torch.Tensor]]] = None


@dataclass
class EmformerRNNTOutput(BaseModelOutput):
    """
    Class for Emformer RNN-Transducer outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        left_context_states (`list(list(torch.FloatTensor))`, *optional*, returned when `is_streaming=True` is passed.
            Emformer internal state representation required for conditioning on the previously processed audio chunk
            during streaming inference.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    left_context_states: Optional[List[List[torch.Tensor]]] = None


class EmformerTimeReduction(torch.nn.Module):
    """
    Coalesces frames along the time dimension into a fewer number of frames with higher feature dimensionality.
    """

    def __init__(self, stride: int) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, input: torch.Tensor):
        batch_size, time_size, feature_size = input.shape
        num_frames = time_size - (time_size % self.stride)
        input = input[:, :num_frames, :]
        max_time_size = num_frames // self.stride

        output = input.reshape(batch_size, max_time_size, feature_size * self.stride)
        return output


class EmformerAttention(torch.nn.Module):
    r"""Emformer layer attention module.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Emformer layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        negative_inf (float, optional): value to use for negative infinity in attention weights. (Default: -1e8)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        negative_inf: float = -1e8,
    ):
        super().__init__()

        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim ({input_dim}) is not a multiple of num_heads ({num_heads}).")

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.negative_inf = negative_inf

        self.scaling = (self.input_dim // self.num_heads) ** -0.5

        self.emb_to_key_value = torch.nn.Linear(input_dim, 2 * input_dim, bias=True)
        self.emb_to_query = torch.nn.Linear(input_dim, input_dim, bias=True)
        self.out_proj = torch.nn.Linear(input_dim, input_dim, bias=True)

    def _gen_key_value(self, input: torch.Tensor, mems: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        time_length = input.size(0)
        summary_length = mems.size(0) + 1
        right_ctx_utterance_block = input[: time_length - summary_length]
        mems_right_ctx_utterance_block = torch.cat([mems, right_ctx_utterance_block])
        key, value = self.emb_to_key_value(mems_right_ctx_utterance_block).chunk(chunks=2, dim=2)
        return key, value

    def _gen_attention_probs(
        self,
        attention_weights: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attention_weights_float = attention_weights.float()
        attention_weights_float = attention_weights_float.masked_fill(attention_mask.unsqueeze(0), self.negative_inf)
        time_length = attention_weights.size(1)
        batch_size = attention_weights.size(0) // self.num_heads
        if padding_mask is not None:
            attention_weights_float = attention_weights_float.view(batch_size, self.num_heads, time_length, -1)
            attention_weights_float = attention_weights_float.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), self.negative_inf
            )
            attention_weights_float = attention_weights_float.view(batch_size * self.num_heads, time_length, -1)
        attention_probs = torch.nn.functional.softmax(attention_weights_float, dim=-1).type_as(attention_weights)
        return torch.nn.functional.dropout(attention_probs, p=float(self.dropout), training=self.training)

    def _forward_impl(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        summary: torch.Tensor,
        mems: torch.Tensor,
        attention_mask: torch.Tensor,
        left_context_key: Optional[torch.Tensor] = None,
        left_context_val: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = utterance.size(1)
        time_length = right_context.size(0) + utterance.size(0) + summary.size(0)

        # Compute query with [right context, utterance, summary].
        query = self.emb_to_query(torch.cat([right_context, utterance, summary]))

        # Compute key and value with [mems, right context, utterance].
        key, value = self.emb_to_key_value(torch.cat([mems, right_context, utterance])).chunk(chunks=2, dim=2)

        if left_context_key is not None and left_context_val is not None:
            right_context_blocks_length = time_length - torch.max(lengths).int() - summary.size(0)
            key = torch.cat(
                [
                    key[: mems.size(0) + right_context_blocks_length],
                    left_context_key,
                    key[mems.size(0) + right_context_blocks_length :],
                ],
            )
            value = torch.cat(
                [
                    value[: mems.size(0) + right_context_blocks_length],
                    left_context_val,
                    value[mems.size(0) + right_context_blocks_length :],
                ],
            )

        # Compute attention weights from query, key, and value.
        reshaped_query, reshaped_key, reshaped_value = [
            tensor.contiguous().view(-1, batch_size * self.num_heads, self.input_dim // self.num_heads).transpose(0, 1)
            for tensor in [query, key, value]
        ]
        attention_weights = torch.bmm(reshaped_query * self.scaling, reshaped_key.transpose(1, 2))

        # Compute padding mask.
        padding_mask = _gen_padding_mask(utterance, right_context, summary, lengths, mems, left_context_key)

        # Compute attention probabilities.
        attention_probs = self._gen_attention_probs(attention_weights, attention_mask, padding_mask)
        if output_attentions:
            attention_outputs = attention_probs.reshape(batch_size, self.num_heads, time_length, time_length)
            utterance_start = right_context.size(0)
            utterance_end = right_context.size(0) + utterance.size(0)
            attention_outputs = attention_outputs[:, :, utterance_start:utterance_end, utterance_start:utterance_end]
        else:
            attention_outputs = None

        # Compute attention.
        attention = torch.bmm(attention_probs, reshaped_value)
        # attention.shape == (batch_size * self.num_heads, time_length, self.input_dim // self.num_heads)
        attention = attention.transpose(0, 1).reshape(time_length, batch_size, self.input_dim)

        # Apply output projection.
        output_right_context_mems = self.out_proj(attention)

        summary_length = summary.size(0)
        output_right_context = output_right_context_mems[: time_length - summary_length]
        output_mems = output_right_context_mems[time_length - summary_length :]
        output_mems = torch.tanh(output_mems)

        return output_right_context, output_mems, key, value, attention_outputs

    def forward(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        summary: torch.Tensor,
        mems: torch.Tensor,
        attention_mask: torch.Tensor,
        left_context_key: torch.Tensor = None,
        left_context_val: torch.Tensor = None,
        output_attentions: bool = False,
    ):
        r"""Forward pass for training.

        B: batch size; D: feature dimension of each frame; T: number of utterance frames; R: number of right context
        frames; S: number of summary elements; M: number of memory elements.

        Args:
            utterance (torch.Tensor): utterance frames, with shape *(T, B, D)*.
            lengths (torch.Tensor): with shape *(B,)* and i-th element representing
                number of valid frames for i-th batch element in `utterance`.
            right_context (torch.Tensor): right context frames, with shape *(R, B, D)*.
            summary (torch.Tensor): summary elements, with shape *(S, B, D)*.
            mems (torch.Tensor): memory elements, with shape *(M, B, D)*.
            attention_mask (torch.Tensor): attention mask for underlying attention module.
            left_context_key (torch.Tensor): left context attention key computed from preceding invocation.
            left_context_val (torch.Tensor): left context attention value computed from preceding invocation.
        """
        is_streaming = left_context_key is not None and left_context_val is not None

        if is_streaming:
            query_dim = right_context.size(0) + utterance.size(0) + summary.size(0)
            key_dim = right_context.size(0) + utterance.size(0) + mems.size(0) + left_context_key.size(0)
            attention_mask = torch.zeros(query_dim, key_dim).to(dtype=torch.bool, device=utterance.device)
            attention_mask[-1, : mems.size(0)] = True

        output, output_mems, key, value, attention_probs = self._forward_impl(
            utterance, lengths, right_context, summary, mems, attention_mask,
            left_context_key,
            left_context_val,
            output_attentions
        )

        if is_streaming:
            output_mems = output_mems[:-1]

        return (
            output,
            output_mems,
            key[mems.size(0) + right_context.size(0) :],
            value[mems.size(0) + right_context.size(0) :],
            attention_probs
        )


class EmformerLayer(torch.nn.Module):
    r"""Emformer layer that constitutes Emformer.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads.
        ffn_dim: (int): hidden layer dimension of feedforward network.
        segment_length (int): length of each input segment.
        dropout (float, optional): dropout probability. (Default: 0.0)
        activation (str, optional): activation function to use in feedforward network.
            Must be one of ("relu", "gelu", "silu"). (Default: "relu")
        left_context_length (int, optional): length of left context. (Default: 0)
        max_memory_size (int, optional): maximum number of memory elements to use. (Default: 0)
        negative_inf (float, optional): value to use for negative infinity in attention weights. (Default: -1e8)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        segment_length: int,
        dropout: float = 0.0,
        activation: str = "relu",
        left_context_length: int = 0,
        max_memory_size: int = 0,
        negative_inf: float = -1e8,
    ):
        super().__init__()

        self.attention = EmformerAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            negative_inf=negative_inf,
        )
        self.dropout = torch.nn.Dropout(dropout)

        activation_module = ACT2FN[activation]
        self.pos_ff = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, ffn_dim),
            activation_module,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ffn_dim, input_dim),
            torch.nn.Dropout(dropout),
        )
        self.layer_norm_input = torch.nn.LayerNorm(input_dim)
        self.layer_norm_output = torch.nn.LayerNorm(input_dim)

        self.left_context_length = left_context_length
        self.segment_length = segment_length
        self.max_memory_size = max_memory_size
        self.input_dim = input_dim

    def _process_attention_output(
        self,
        right_context_output: torch.Tensor,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
    ) -> torch.Tensor:
        result = self.dropout(right_context_output) + torch.cat([right_context, utterance])
        result = self.pos_ff(result) + result
        result = self.layer_norm_output(result)
        return result

    def _apply_pre_attention_layer_norm(
        self, utterance: torch.Tensor, right_context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        layer_norm_input = self.layer_norm_input(torch.cat([right_context, utterance]))
        return (
            layer_norm_input[right_context.size(0) :],
            layer_norm_input[: right_context.size(0)],
        )

    def _apply_post_attention_ffn(
        self, right_context_output: torch.Tensor, utterance: torch.Tensor, right_context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        right_context_output = self._process_attention_output(right_context_output, utterance, right_context)
        return right_context_output[right_context.size(0):], right_context_output[: right_context.size(0)]

    def _apply_attention_forward(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        mems: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            raise ValueError("attention_mask must be not None when for_inference is False")

        empty_summary = torch.empty(0).to(dtype=utterance.dtype, device=utterance.device)
        right_context_output, next_mem, _, _, attention_probs = self.attention(
            utterance=utterance,
            lengths=lengths,
            right_context=right_context,
            summary=empty_summary,
            mems=mems,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        return right_context_output, next_mem, attention_probs

    def _apply_attention_streaming(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        mems: torch.Tensor,
        state: Optional[List[torch.Tensor]],
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        if state is None:
            state = self._init_state(utterance.size(1), device=utterance.device)
        pre_mems, lc_key, lc_val = self._unpack_state(state)
        if self.use_mem:
            summary = self.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)
            summary = summary[:1]
        else:
            summary = torch.empty(0).to(dtype=utterance.dtype, device=utterance.device)
        rc_output, next_m, next_k, next_v, attention_probs = self.attention.infer(
            utterance=utterance,
            lengths=lengths,
            right_context=right_context,
            summary=summary,
            mems=pre_mems,
            left_context_key=lc_key,
            left_context_val=lc_val,
            output_attentions=output_attentions,
        )
        state = self._pack_state(next_k, next_v, utterance.size(0), mems, state)
        return rc_output, next_m, state, attention_probs

    def forward(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        mems: torch.Tensor,
        state: Optional[List[torch.Tensor]],
        attention_mask: torch.Tensor,
        output_attentions: bool,
        is_streaming: bool = False
    ):
        r"""Forward pass for training.

        utterance (torch.Tensor): utterance frames, with shape *(T, B, D)*.
        lengths (torch.Tensor): with shape *(B,)* and i-th element representing
            number of valid frames for i-th batch element in `utterance`.
        right_context (torch.Tensor): right context frames, with shape *(R, B, D)*.
        state (List[torch.Tensor] or None): list of tensors representing layer internal state
            generated in preceding invocation of ``infer``.
        mems (torch.Tensor): memory elements, with shape *(M, B, D)*.
        attention_mask (torch.Tensor): attention mask for underlying attention module.
        """
        (
            layer_norm_utterance,
            layer_norm_right_context,
        ) = self._apply_pre_attention_layer_norm(utterance, right_context)
        if not is_streaming:
            right_context_output, output_mems, attention_probs = self._apply_attention_forward(
                layer_norm_utterance, lengths, layer_norm_right_context, mems, attention_mask, output_attentions
            )
            output_state = None
        else:
            rc_output, output_mems, output_state, attention_probs = self._apply_attention_streaming(
                layer_norm_utterance, lengths, layer_norm_right_context, mems, state
            )
        output_utterance, output_right_context = self._apply_post_attention_ffn(right_context_output, utterance, right_context)
        return output_utterance, output_right_context, output_state, output_mems, attention_probs


class Emformer(torch.nn.Module):
    r"""Implements the Emformer architecture introduced in [Emformer: Efficient Memory Transformer Based Acoustic
    Model For Low Latency Streaming Speech Recognition](https://arxiv.org/abs/2010.10759)

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Emformer layer.
        ffn_dim (int): hidden layer dimension of each Emformer layer's feedforward network.
        num_layers (int): number of Emformer layers to instantiate.
        segment_length (int): length of each input segment.
        dropout (float, optional): dropout probability. (Default: 0.0)
        activation (str, optional): activation function to use in each Emformer layer's
            feedforward network. Must be one of ("relu", "gelu", "silu"). (Default: "relu")
        left_context_length (int, optional): length of left context. (Default: 0)
        right_context_length (int, optional): length of right context. (Default: 0)
        max_memory_size (int, optional): maximum number of memory elements to use. (Default: 0)
        negative_inf (float, optional): value to use for negative infinity in attention weights. (Default: -1e8)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        segment_length: int,
        dropout: float = 0.0,
        activation: str = "relu",
        left_context_length: int = 0,
        right_context_length: int = 0,
        max_memory_size: int = 0,
        negative_inf: float = -1e8,
    ):
        super().__init__()

        self.emformer_layers = torch.nn.ModuleList(
            [
                EmformerLayer(
                    input_dim,
                    num_heads,
                    ffn_dim,
                    segment_length,
                    dropout=dropout,
                    activation=activation,
                    left_context_length=left_context_length,
                    max_memory_size=max_memory_size,
                    negative_inf=negative_inf,
                )
                for layer_idx in range(num_layers)
            ]
        )

        self.left_context_length = left_context_length
        self.right_context_length = right_context_length
        self.segment_length = segment_length
        self.max_memory_size = max_memory_size

    def _gen_right_context(self, input: torch.Tensor) -> torch.Tensor:
        time_length = input.shape[0]
        num_segs = math.ceil((time_length - self.right_context_length) / self.segment_length)
        right_context_blocks = []
        for seg_idx in range(num_segs - 1):
            start = (seg_idx + 1) * self.segment_length
            end = start + self.right_context_length
            right_context_blocks.append(input[start:end])
        right_context_blocks.append(input[time_length - self.right_context_length :])
        return torch.cat(right_context_blocks)

    def _gen_attention_mask_column_widths(self, segment_idx: int, utterance_length: int) -> List[int]:
        num_segments = math.ceil(utterance_length / self.segment_length)
        right_context_start = segment_idx * self.right_context_length
        right_context_end = right_context_start + self.right_context_length
        segment_start = max(segment_idx * self.segment_length - self.left_context_length, 0)
        segment_end = min((segment_idx + 1) * self.segment_length, utterance_length)
        right_context_length = self.right_context_length * num_segments

        column_widths = [
            right_context_start,  # before right context
            self.right_context_length,  # right context
            right_context_length - right_context_end,  # after right context
            segment_start,  # before query segment
            segment_end - segment_start,  # query segment
            utterance_length - segment_end,  # after query segment
        ]

        return column_widths

    def _gen_attention_mask(self, input: torch.Tensor) -> torch.Tensor:
        utterance_length = input.size(0)
        num_segs = math.ceil(utterance_length / self.segment_length)

        right_context_mask = []
        query_mask = []
        summary_mask = []

        num_cols = 6
        # right context, query segment
        right_context_q_columns_mask = [idx in [1, 4] for idx in range(num_cols)]
        summary_columns_mask = None
        masks_to_concat = [right_context_mask, query_mask]

        for seg_idx in range(num_segs):
            col_widths = self._gen_attention_mask_column_widths(seg_idx, utterance_length)

            right_context_mask_block = _gen_attention_mask_block(
                col_widths, right_context_q_columns_mask, self.right_context_length, input.device
            )
            right_context_mask.append(right_context_mask_block)

            query_mask_block = _gen_attention_mask_block(
                col_widths,
                right_context_q_columns_mask,
                min(
                    self.segment_length,
                    utterance_length - seg_idx * self.segment_length,
                ),
                input.device,
            )
            query_mask.append(query_mask_block)

            if summary_columns_mask is not None:
                summary_mask_block = _gen_attention_mask_block(col_widths, summary_columns_mask, 1, input.device)
                summary_mask.append(summary_mask_block)

        attention_mask = (1 - torch.cat([torch.cat(mask) for mask in masks_to_concat])).to(torch.bool)
        return attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        left_context_states: Optional[List[List[torch.Tensor]]] = None,
        is_streaming=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        lengths = attention_mask.sum(-1)

        input = hidden_states.permute(1, 0, 2)

        if is_streaming:
            right_context_start_idx = input.size(0) - self.right_context_length
            right_context = input[right_context_start_idx:]
            utterance = input[:right_context_start_idx]
            lengths = torch.clamp(lengths - self.right_context_length, min=0)
        else:
            right_context = self._gen_right_context(input)
            utterance = input[: input.size(0) - self.right_context_length]
            attention_mask = self._gen_attention_mask(utterance)

        mems = torch.empty(0).to(dtype=input.dtype, device=input.device)
        hidden_states = utterance
        output_states: List[List[torch.Tensor]] = []

        for layer_idx, layer in enumerate(self.emformer_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.permute(1, 0, 2),)

            if is_streaming:
                hidden_states, right_context, output_state, mems, attention_probs = layer(
                    utterance=hidden_states,
                    lengths=lengths,
                    right_context=right_context,
                    mems=mems,
                    state=None if left_context_states is None else left_context_states[layer_idx],
                    attention_mask=attention_mask,
                    is_streaming=is_streaming,
                    output_attentions=output_attentions
                )
                output_states.append(output_state)
            else:
                hidden_states, right_context, output_state, mems, attention_probs = layer(
                    utterance=hidden_states,
                    lengths=lengths,
                    right_context=right_context,
                    state=None,
                    mems=mems,
                    attention_mask=attention_mask,
                    is_streaming=is_streaming,
                    output_attentions=output_attentions
                )

            if output_attentions:
                all_self_attentions = all_self_attentions + (attention_probs,)

        hidden_states = hidden_states.permute(1, 0, 2)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return EmformerModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            left_context_states=output_states,
        )


class EmformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EmformerConfig
    base_model_prefix = "emformer"
    main_input_name = "input_features"
    _keys_to_ignore_on_load_missing = []
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # TODO (Anton): implement weight_init_gain for key-value linear layers

    def _get_time_reduced_output_lengths(self, input_lengths):
        output_lengths = input_lengths.div(self.config.time_reduction_stride, rounding_mode="trunc")
        return output_lengths

    def _get_time_reduced_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # in inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_time_reduced_output_lengths(non_padded_lengths)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations make sure that all values before the output lengths indices are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask


EMFORMER_START_DOCSTRING = r"""
    Emformer was proposed in [Emformer: Efficient Memory Transformer Based Acoustic Model For Low Latency Streaming
    Speech Recognition](https://arxiv.org/abs/2010.10759) by Yangyang Shi, Yongqiang Wang, Chunyang Wu, Ching-Feng Yeh,
    Julian Chan, Frank Zhang, Duc Le, Mike Seltzer.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving etc.).

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`EmformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


EMFORMER_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_size)`, *optional*):
            Float values of fbank features extracted from the raw speech waveform. To convert the waveform array into
            `input_features`, the [`EmformerFeatureExtractor`] should be used for extracting the Mel spectrogram
            features, padding and conversion into a tensor of type `torch.FloatTensor`. See
            [`~EmformerFeatureExtractor.__call__`]
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            <Tip warning={true}>

            `attention_mask` should only be passed if the corresponding processor has `config.return_attention_mask ==
            True`. For all models whose processor has `config.return_attention_mask == False`, such as
            [emformer-base](https://huggingface.co/anton-l/emformer-base-librispeech), `attention_mask` should **not**
            be passed to avoid degraded performance when doing batched inference. For such models `input_values` should
            simply be padded with 0 and passed without `attention_mask`. Be aware that these models also yield slightly
            different results depending on whether `input_values` is padded or not.

            </Tip>

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.EmformerModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Emformer Model transformer outputting raw hidden-states without any specific head on top.",
    EMFORMER_START_DOCSTRING,
)
class EmformerModel(EmformerPreTrainedModel):
    def __init__(self, config: EmformerConfig):
        super().__init__(config)
        self.config = config

        self.input_linear = nn.Linear(
            config.input_dim,
            config.time_reduction_input_dim,
            bias=False,
        )
        self.time_reduction = EmformerTimeReduction(config.time_reduction_stride)
        transformer_input_dim = config.time_reduction_input_dim * config.time_reduction_stride

        self.encoder = Emformer(
            input_dim=transformer_input_dim,
            num_heads=config.num_attention_heads,
            ffn_dim=config.intermediate_size,
            num_layers=config.num_hidden_layers,
            segment_length=config.segment_length // config.time_reduction_stride,
            dropout=config.hidden_dropout,
            activation=config.hidden_act,
            left_context_length=config.left_context_length,
            right_context_length=config.right_context_length // config.time_reduction_stride,
            max_memory_size=0,
        )

        self.output_linear = nn.Linear(transformer_input_dim, config.output_dim)
        self.layer_norm = nn.LayerNorm(config.output_dim)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(EMFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_PROCESSOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=EmformerModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        left_context_states: Optional[List[List[torch.Tensor]]] = None,
        is_streaming: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, EmformerModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.input_linear(input_features)
        hidden_states = self.time_reduction(hidden_states)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_time_reduced_attention_mask(hidden_states.shape[1], attention_mask)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            left_context_states=left_context_states,
            is_streaming=is_streaming,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        hidden_states = self.output_linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return EmformerModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            left_context_states=encoder_outputs.left_context_states,
        )


class RNNTCustomLSTM(torch.nn.Module):
    r"""Custom long-short-term memory (LSTM) block that applies layer normalization
    to internal nodes.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        layer_norm (bool, optional): if `True`, enables layer normalization. (Default: `False`)
        layer_norm_epsilon (float, optional):  value of epsilon to use in
            layer normalization layers (Default: 1e-5)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_norm: bool = False,
        layer_norm_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.x2g = torch.nn.Linear(input_dim, 4 * hidden_dim, bias=(not layer_norm))
        self.p2g = torch.nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        self.c_norm = torch.nn.LayerNorm(hidden_dim, eps=layer_norm_epsilon)
        self.g_norm = torch.nn.LayerNorm(4 * hidden_dim, eps=layer_norm_epsilon)

        self.hidden_dim = hidden_dim

    def forward(
        self, input: torch.Tensor, state: Optional[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        r"""Forward pass.

        input (torch.Tensor): with shape *(T, B, D)*.
        state (List[torch.Tensor] or None): list of tensors representing internal state generated in 
            preceding invocation of `forward`.
        """
        if state is None:
            batch_size = input.size(1)
            cell_output = torch.zeros(batch_size, self.hidden_dim, device=input.device, dtype=input.dtype)
            cell_state = torch.zeros(batch_size, self.hidden_dim, device=input.device, dtype=input.dtype)
        else:
            cell_output, cell_state = state

        gated_input = self.x2g(input)
        outputs = []
        for gates in gated_input.unbind(0):
            gates = gates + self.p2g(cell_output)
            gates = self.g_norm(gates)
            input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
            input_gate = input_gate.sigmoid()
            forget_gate = forget_gate.sigmoid()
            cell_gate = cell_gate.tanh()
            output_gate = output_gate.sigmoid()
            cell_state = forget_gate * cell_state + input_gate * cell_gate
            cell_state = self.c_norm(cell_state)
            cell_output = output_gate * cell_state.tanh()
            outputs.append(cell_output)

        output = torch.stack(outputs, dim=0)
        state = [cell_output, cell_state]

        return output, state


class RNNTPredictor(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) prediction network.

    Args:
        num_symbols (int): size of target token lexicon.
        output_dim (int): feature dimension of each output sequence element.
        symbol_embedding_dim (int): dimension of each target token embedding.
        num_lstm_layers (int): number of LSTM layers to instantiate.
        lstm_hidden_dim (int): output dimension of each LSTM layer.
        lstm_layer_norm_epsilon (float, optional): value of epsilon to use in
            LSTM layer normalization layers. (Default: 1e-5)
        lstm_dropout (float, optional): LSTM dropout probability. (Default: 0.0)

    """

    def __init__(self, config: EmformerConfig) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(config.vocab_size, config.token_embedding_dim)
        self.input_layer_norm = torch.nn.LayerNorm(config.token_embedding_dim)
        self.lstm_layers = torch.nn.ModuleList(
            [
                RNNTCustomLSTM(
                    config.token_embedding_dim if idx == 0 else config.lstm_hidden_dim,
                    config.lstm_hidden_dim,
                    layer_norm=True,
                    layer_norm_epsilon=config.lstm_layer_norm_epsilon,
                )
                for idx in range(config.num_lstm_layers)
            ]
        )
        self.dropout = torch.nn.Dropout(p=config.lstm_dropout)
        self.linear = torch.nn.Linear(config.lstm_hidden_dim, config.output_dim)
        self.output_layer_norm = torch.nn.LayerNorm(config.output_dim)

        self.lstm_dropout = config.lstm_dropout

    def forward(
        self,
        input: torch.Tensor,
        state: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass.

        B: batch size; U: maximum sequence length in batch; D: feature dimension of each input sequence element.

        Args:
            input (torch.Tensor): target sequences, with shape *(B, U)* and each element
                mapping to a target symbol, i.e. in range *[0, num_symbols)*.
            lengths (torch.Tensor): with shape *(B,)* and i-th element representing
                number of valid frames for i-th batch element in `input`.
            state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing internal state generated in preceding invocation of `forward`. (Default: `None`)

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output encoding sequences, with shape *(B, U, output_dim)*
                torch.Tensor
                    output lengths, with shape *(B,)* and i-th element representing number of valid elements for i-th
                    batch element in output encoding sequences.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors representing internal state generated in current invocation
                    of `forward`.
        """
        input_tb = input.permute(1, 0)
        embedding_out = self.embedding(input_tb)
        input_layer_norm_out = self.input_layer_norm(embedding_out)

        lstm_out = input_layer_norm_out
        state_out: List[List[torch.Tensor]] = []
        for layer_idx, lstm in enumerate(self.lstm_layers):
            lstm_out, lstm_state_out = lstm(lstm_out, None if state is None else state[layer_idx])
            lstm_out = self.dropout(lstm_out)
            state_out.append(lstm_state_out)

        linear_out = self.linear(lstm_out)
        output_layer_norm_out = self.output_layer_norm(linear_out)
        return output_layer_norm_out.permute(1, 0, 2), state_out


class RNNTJoiner(torch.nn.Module):
    def __init__(self, config: EmformerConfig) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(config.output_dim, config.vocab_size, bias=True)
        self.activation = ACT2FN[config.joiner_activation]

    def forward(self, source_encodings, target_encodings):
        joint_encodings = source_encodings.unsqueeze(2).contiguous() + target_encodings.unsqueeze(1).contiguous()
        joint_encodings = self.activation(joint_encodings)
        output = self.linear(joint_encodings)
        return output


@add_start_docstrings(
    """Emformer Model with an RNN-Transducer decoder.""",
    EMFORMER_START_DOCSTRING,
)
class EmformerForRNNT(EmformerPreTrainedModel):
    def __init__(self, config: EmformerConfig):
        super().__init__(config)

        self.emformer = EmformerModel(config)
        self.predictor = RNNTPredictor(config)
        self.joiner = RNNTJoiner(config)

        self.blank_token_id = config.blank_token_id
        self.pad_token_id = config.pad_token_id
        self.max_output_length = config.max_output_length

        # Initialize weights and apply final processing
        self.post_init()

    def _gen_next_token_probs(self, encoder_out, predictor_out):
        joined_out = self.joiner(encoder_out, predictor_out)
        joined_out = torch.nn.functional.log_softmax(joined_out, dim=3)
        return joined_out[:, 0, 0]

    def _greedy_decode(self, encoder_out, encoder_lengths):
        past_rnn_state = None
        batch_size = encoder_out.shape[0]
        device = encoder_out.device

        logits = [[] for _ in range(batch_size)]

        last_label = torch.full([batch_size, 1], fill_value=self.blank_token_id, dtype=torch.long, device=device)
        blank_mask = torch.full([batch_size], fill_value=0, dtype=torch.bool, device=device)

        max_time_len = encoder_out.shape[1]
        for time_idx in range(max_time_len):
            encoder_slice = encoder_out.narrow(dim=1, start=time_idx, length=1)

            not_blank = True
            n_decoded_tokens = 0

            blank_mask = blank_mask.fill_(False)
            time_mask = time_idx >= encoder_lengths
            blank_mask = torch.bitwise_or(blank_mask, time_mask)

            while not_blank and (self.max_output_length is None or n_decoded_tokens < self.max_output_length):
                rnn_out, current_rnn_state = self.predictor(last_label, past_rnn_state)
                next_token_logits = self._gen_next_token_probs(encoder_slice, rnn_out)
                next_token_id = next_token_logits.argmax(-1)

                token_is_blank = next_token_id == self.blank_token_id
                blank_mask = torch.bitwise_or(blank_mask, token_is_blank)

                if blank_mask.all():
                    not_blank = False
                else:
                    blank_indices = []
                    if past_rnn_state is not None:
                        blank_indices = (blank_mask == 1).nonzero(as_tuple=False)

                    if past_rnn_state is not None:
                        # 3 LSTM layers
                        for layer_id in range(len(past_rnn_state)):
                            # 2 states
                            for state_id in range(len(past_rnn_state[layer_id])):
                                current_rnn_state[layer_id][state_id][blank_indices, :] = past_rnn_state[layer_id][
                                    state_id
                                ][blank_indices, :]

                    next_token_id[blank_indices] = last_label[blank_indices, 0]

                    last_label = next_token_id.clone().view(-1, 1)
                    past_rnn_state = current_rnn_state

                    next_token_id.masked_fill_(blank_mask, self.blank_token_id)
                    for batch_idx, (token_id, token_logits) in enumerate(zip(next_token_id, next_token_logits)):
                        if time_mask[batch_idx] == 0 and token_id != self.blank_token_id:
                            logits[batch_idx].append(token_logits)

                    n_decoded_tokens += 1
        for batch_idx, seq_logits in enumerate(logits):
            if len(seq_logits) > 0:
                logits[batch_idx] = torch.stack(seq_logits, dim=-2)
            else:
                logits[batch_idx] = torch.tensor([], device=device)

        return logits

    @add_start_docstrings_to_model_forward(EMFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_PROCESSOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=EmformerRNNTOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        left_context_states: Optional[List[List[torch.Tensor]]] = None,
        is_streaming: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, EmformerRNNTOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        attention_mask = (
            attention_mask
            if attention_mask is not None
            else torch.ones(input_features.shape[:-1], dtype=torch.long, device=self.device)
        )

        encoder_outputs = self.emformer(
            input_features,
            attention_mask=attention_mask,
            left_context_states=left_context_states,
            is_streaming=is_streaming,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        encoded_frames = encoder_outputs[0]

        attention_mask = self._get_time_reduced_attention_mask(input_features.shape[1], attention_mask)

        logits = self._greedy_decode(encoded_frames, attention_mask.sum(-1))
        logits = nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=self.pad_token_id)

        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        # TODO (Anton): support RNN-T loss
        loss = None
        if not return_dict:
            output = (logits,) + encoder_outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return EmformerRNNTOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            left_context_states=left_context_states,
        )
