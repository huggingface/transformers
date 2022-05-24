# coding=utf-8
# Copyright 2022 The TorchAudio Authors and the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Emformer model."""

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import torch_int_div
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_emformer import EmformerConfig


logger = logging.get_logger(__name__)


_HIDDEN_STATES_START_POSITION = 2

# General docstring
_CONFIG_FOR_DOC = "EmformerConfig"
_PROCESSOR_FOR_DOC = "Wav2Vec2Processor"

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
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


def _gen_padding_mask(
    utterance: torch.Tensor,
    right_context: torch.Tensor,
    summary: torch.Tensor,
    lengths: torch.Tensor,
    mems: torch.Tensor,
    left_context_key: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    T = right_context.size(0) + utterance.size(0) + summary.size(0)
    B = right_context.size(1)
    if B == 1:
        padding_mask = None
    else:
        right_context_blocks_length = T - torch.max(lengths).int() - summary.size(0)
        left_context_blocks_length = left_context_key.size(0) if left_context_key is not None else 0
        klengths = lengths + mems.size(0) + right_context_blocks_length + left_context_blocks_length
        padding_mask = _lengths_to_padding_mask(lengths=klengths)
    return padding_mask


def _get_activation_module(activation: str) -> torch.nn.Module:
    if activation == "relu":
        return torch.nn.ReLU()
    elif activation == "gelu":
        return torch.nn.GELU()
    elif activation == "silu":
        return torch.nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation {activation}")


def _get_weight_init_gains(weight_init_scale_strategy: Optional[str], num_layers: int) -> List[Optional[float]]:
    if weight_init_scale_strategy is None:
        return [None for _ in range(num_layers)]
    elif weight_init_scale_strategy == "depthwise":
        return [1.0 / math.sqrt(layer_idx + 1) for layer_idx in range(num_layers)]
    elif weight_init_scale_strategy == "constant":
        return [1.0 / math.sqrt(2) for layer_idx in range(num_layers)]
    else:
        raise ValueError(f"Unsupported weight_init_scale_strategy value {weight_init_scale_strategy}")


def _gen_attention_mask_block(
    col_widths: List[int], col_mask: List[bool], num_rows: int, device: torch.device
) -> torch.Tensor:
    # assert len(col_widths) == len(col_mask), "Length of col_widths must match that of col_mask"

    mask_block = [
        torch.ones(num_rows, col_width, device=device)
        if is_ones_col
        else torch.zeros(num_rows, col_width, device=device)
        for col_width, is_ones_col in zip(col_widths, col_mask)
    ]
    return torch.cat(mask_block, dim=1)


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
        output = output.contiguous()
        return output


class EmformerAttention(torch.nn.Module):
    r"""Emformer layer attention module.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Emformer layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        weight_init_gain (float or None, optional): scale factor to apply when initializing
            attention module parameters. (Default: ``None``)
        tanh_on_mem (bool, optional): if ``True``, applies tanh to memory elements. (Default: ``False``)
        negative_inf (float, optional): value to use for negative infinity in attention weights. (Default: -1e8)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        weight_init_gain: Optional[float] = None,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
    ):
        super().__init__()

        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim ({input_dim}) is not a multiple of num_heads ({num_heads}).")

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.tanh_on_mem = tanh_on_mem
        self.negative_inf = negative_inf

        self.scaling = (self.input_dim // self.num_heads) ** -0.5

        self.emb_to_key_value = torch.nn.Linear(input_dim, 2 * input_dim, bias=True)
        self.emb_to_query = torch.nn.Linear(input_dim, input_dim, bias=True)
        self.out_proj = torch.nn.Linear(input_dim, input_dim, bias=True)

        if weight_init_gain:
            torch.nn.init.xavier_uniform_(self.emb_to_key_value.weight, gain=weight_init_gain)
            torch.nn.init.xavier_uniform_(self.emb_to_query.weight, gain=weight_init_gain)

    def _gen_key_value(self, input: torch.Tensor, mems: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T, _, _ = input.shape
        summary_length = mems.size(0) + 1
        right_ctx_utterance_block = input[: T - summary_length]
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
        T = attention_weights.size(1)
        B = attention_weights.size(0) // self.num_heads
        if padding_mask is not None:
            attention_weights_float = attention_weights_float.view(B, self.num_heads, T, -1)
            attention_weights_float = attention_weights_float.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), self.negative_inf
            )
            attention_weights_float = attention_weights_float.view(B * self.num_heads, T, -1)
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
        output_attentions: bool = False,
        left_context_key: Optional[torch.Tensor] = None,
        left_context_val: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attention_probs_reshaped = attention_probs.view(batch_size, self.num_heads, time_length, time_length)
            attention_probs = attention_probs_reshaped.view(batch_size * self.num_heads, time_length, time_length)
        else:
            attention_probs_reshaped = None

        # Compute attention.
        attention = torch.bmm(attention_probs, reshaped_value)
        # attention.shape == (batch_size * self.num_heads, time_length, self.input_dim // self.num_heads)
        attention = attention.transpose(0, 1).contiguous().view(time_length, batch_size, self.input_dim)

        # Apply output projection.
        output_right_context_mems = self.out_proj(attention)

        summary_length = summary.size(0)
        output_right_context = output_right_context_mems[: time_length - summary_length]
        output_mems = output_right_context_mems[time_length - summary_length :]
        if self.tanh_on_mem:
            output_mems = torch.tanh(output_mems)
        else:
            output_mems = torch.clamp(output_mems, min=-10, max=10)

        return output_right_context, output_mems, key, value, attention_probs_reshaped

    def forward(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        summary: torch.Tensor,
        mems: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass for training.

        B: batch size; D: feature dimension of each frame; T: number of utterance frames; R: number of right context
        frames; S: number of summary elements; M: number of memory elements.

        Args:
            utterance (torch.Tensor): utterance frames, with shape `(T, B, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``utterance``.
            right_context (torch.Tensor): right context frames, with shape `(R, B, D)`.
            summary (torch.Tensor): summary elements, with shape `(S, B, D)`.
            mems (torch.Tensor): memory elements, with shape `(M, B, D)`.
            attention_mask (torch.Tensor): attention mask for underlying attention module.

        Returns:
            (Tensor, Tensor):
                Tensor
                    output frames corresponding to utterance and right_context, with shape `(T + R, B, D)`.
                Tensor
                    updated memory elements, with shape `(M, B, D)`.
        """
        output, output_mems, _, _, attention_probs = self._forward_impl(
            utterance, lengths, right_context, summary, mems, attention_mask, output_attentions
        )
        return output, output_mems[:-1], attention_probs


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
        weight_init_gain (float or None, optional): scale factor to apply when initializing
            attention module parameters. (Default: ``None``)
        tanh_on_mem (bool, optional): if ``True``, applies tanh to memory elements. (Default: ``False``)
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
        weight_init_gain: Optional[float] = None,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
    ):
        super().__init__()

        self.attention = EmformerAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            weight_init_gain=weight_init_gain,
            tanh_on_mem=tanh_on_mem,
            negative_inf=negative_inf,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.memory_op = torch.nn.AvgPool1d(kernel_size=segment_length, stride=segment_length, ceil_mode=True)

        activation_module = _get_activation_module(activation)
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

        self.use_mem = max_memory_size > 0

    def _init_state(self, batch_size: int, device: Optional[torch.device]) -> List[torch.Tensor]:
        empty_memory = torch.zeros(self.max_memory_size, batch_size, self.input_dim, device=device)
        left_context_key = torch.zeros(self.left_context_length, batch_size, self.input_dim, device=device)
        left_context_val = torch.zeros(self.left_context_length, batch_size, self.input_dim, device=device)
        past_length = torch.zeros(1, batch_size, dtype=torch.int32, device=device)
        return [empty_memory, left_context_key, left_context_val, past_length]

    def _unpack_state(self, state: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        past_length = state[3][0][0].item()
        past_left_context_length = min(self.left_context_length, past_length)
        past_mem_length = min(self.max_memory_size, math.ceil(past_length / self.segment_length))
        pre_mems = state[0][self.max_memory_size - past_mem_length :]
        lc_key = state[1][self.left_context_length - past_left_context_length :]
        lc_val = state[2][self.left_context_length - past_left_context_length :]
        return pre_mems, lc_key, lc_val

    def _pack_state(
        self,
        next_k: torch.Tensor,
        next_v: torch.Tensor,
        update_length: int,
        mems: torch.Tensor,
        state: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        new_k = torch.cat([state[1], next_k])
        new_v = torch.cat([state[2], next_v])
        state[0] = torch.cat([state[0], mems])[-self.max_memory_size :]
        state[1] = new_k[new_k.shape[0] - self.left_context_length :]
        state[2] = new_v[new_v.shape[0] - self.left_context_length :]
        state[3] = state[3] + update_length
        return state

    def _process_attention_output(
        self,
        rc_output: torch.Tensor,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
    ) -> torch.Tensor:
        result = self.dropout(rc_output) + torch.cat([right_context, utterance])
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
        self, rc_output: torch.Tensor, utterance: torch.Tensor, right_context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rc_output = self._process_attention_output(rc_output, utterance, right_context)
        return rc_output[right_context.size(0) :], rc_output[: right_context.size(0)]

    def _apply_attention_forward(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        mems: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            raise ValueError("attention_mask must be not None when for_inference is False")

        if self.use_mem:
            summary = self.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)
        else:
            summary = torch.empty(0).to(dtype=utterance.dtype, device=utterance.device)
        rc_output, next_m, attention_probs = self.attention(
            utterance=utterance,
            lengths=lengths,
            right_context=right_context,
            summary=summary,
            mems=mems,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        return rc_output, next_m, attention_probs

    def forward(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        mems: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Forward pass for training.

        B: batch size; D: feature dimension of each frame; T: number of utterance frames; R: number of right context
        frames; M: number of memory elements.

        Args:
            utterance (torch.Tensor): utterance frames, with shape `(T, B, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``utterance``.
            right_context (torch.Tensor): right context frames, with shape `(R, B, D)`.
            mems (torch.Tensor): memory elements, with shape `(M, B, D)`.
            attention_mask (torch.Tensor): attention mask for underlying attention module.

        Returns:
            (Tensor, Tensor, Tensor):
                Tensor
                    encoded utterance frames, with shape `(T, B, D)`.
                Tensor
                    updated right context frames, with shape `(R, B, D)`.
                Tensor
                    updated memory elements, with shape `(M, B, D)`.
        """
        (
            layer_norm_utterance,
            layer_norm_right_context,
        ) = self._apply_pre_attention_layer_norm(utterance, right_context)

        rc_output, output_mems, attention_probs = self._apply_attention_forward(
            layer_norm_utterance,
            lengths,
            layer_norm_right_context,
            mems,
            attention_mask,
            output_attentions
        )
        output_utterance, output_right_context = self._apply_post_attention_ffn(rc_output, utterance, right_context)
        return output_utterance, output_right_context, output_mems, attention_probs


class Emformer(torch.nn.Module):
    r"""Implements the Emformer architecture introduced in
    *Emformer: Efficient Memory Transformer Based Acoustic Model for Low Latency Streaming Speech Recognition*
    [:footcite:`shi2021emformer`].

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
        weight_init_scale_strategy (str, optional): per-layer weight initialization scaling
            strategy. Must be one of ("depthwise", "constant", ``None``). (Default: "depthwise")
        tanh_on_mem (bool, optional): if ``True``, applies tanh to memory elements. (Default: ``False``)
        negative_inf (float, optional): value to use for negative infinity in attention weights. (Default: -1e8)

    Examples:
        >>> emformer = Emformer(512, 8, 2048, 20, 4, right_context_length=1) >>> input = torch.rand(128, 400, 512) #
        batch, num_frames, feature_dim >>> lengths = torch.randint(1, 200, (128,)) # batch >>> output = emformer(input,
        lengths) >>> input = torch.rand(128, 5, 512) >>> lengths = torch.ones(128) * 5 >>> output, lengths, states =
        emformer.infer(input, lengths, None)
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
        weight_init_scale_strategy: str = "depthwise",
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
    ):
        super().__init__()

        self.use_mem = max_memory_size > 0
        self.memory_op = torch.nn.AvgPool1d(
            kernel_size=segment_length,
            stride=segment_length,
            ceil_mode=True,
        )

        weight_init_gains = _get_weight_init_gains(weight_init_scale_strategy, num_layers)
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
                    weight_init_gain=weight_init_gains[layer_idx],
                    tanh_on_mem=tanh_on_mem,
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
        T = input.shape[0]
        num_segs = math.ceil((T - self.right_context_length) / self.segment_length)
        right_context_blocks = []
        for seg_idx in range(num_segs - 1):
            start = (seg_idx + 1) * self.segment_length
            end = start + self.right_context_length
            right_context_blocks.append(input[start:end])
        right_context_blocks.append(input[T - self.right_context_length :])
        return torch.cat(right_context_blocks)

    def _gen_attention_mask_col_widths(self, seg_idx: int, utterance_length: int) -> List[int]:
        num_segs = math.ceil(utterance_length / self.segment_length)
        rc = self.right_context_length
        lc = self.left_context_length
        rc_start = seg_idx * rc
        rc_end = rc_start + rc
        seg_start = max(seg_idx * self.segment_length - lc, 0)
        seg_end = min((seg_idx + 1) * self.segment_length, utterance_length)
        rc_length = self.right_context_length * num_segs

        if self.use_mem:
            m_start = max(seg_idx - self.max_memory_size, 0)
            mem_length = num_segs - 1
            col_widths = [
                m_start,  # before memory
                seg_idx - m_start,  # memory
                mem_length - seg_idx,  # after memory
                rc_start,  # before right context
                rc,  # right context
                rc_length - rc_end,  # after right context
                seg_start,  # before query segment
                seg_end - seg_start,  # query segment
                utterance_length - seg_end,  # after query segment
            ]
        else:
            col_widths = [
                rc_start,  # before right context
                rc,  # right context
                rc_length - rc_end,  # after right context
                seg_start,  # before query segment
                seg_end - seg_start,  # query segment
                utterance_length - seg_end,  # after query segment
            ]

        return col_widths

    def _gen_attention_mask(self, input: torch.Tensor) -> torch.Tensor:
        utterance_length = input.size(0)
        num_segs = math.ceil(utterance_length / self.segment_length)

        rc_mask = []
        query_mask = []
        summary_mask = []

        if self.use_mem:
            num_cols = 9
            # memory, right context, query segment
            rc_q_cols_mask = [idx in [1, 4, 7] for idx in range(num_cols)]
            # right context, query segment
            s_cols_mask = [idx in [4, 7] for idx in range(num_cols)]
            masks_to_concat = [rc_mask, query_mask, summary_mask]
        else:
            num_cols = 6
            # right context, query segment
            rc_q_cols_mask = [idx in [1, 4] for idx in range(num_cols)]
            s_cols_mask = None
            masks_to_concat = [rc_mask, query_mask]

        for seg_idx in range(num_segs):
            col_widths = self._gen_attention_mask_col_widths(seg_idx, utterance_length)

            rc_mask_block = _gen_attention_mask_block(
                col_widths, rc_q_cols_mask, self.right_context_length, input.device
            )
            rc_mask.append(rc_mask_block)

            query_mask_block = _gen_attention_mask_block(
                col_widths,
                rc_q_cols_mask,
                min(
                    self.segment_length,
                    utterance_length - seg_idx * self.segment_length,
                ),
                input.device,
            )
            query_mask.append(query_mask_block)

            if s_cols_mask is not None:
                summary_mask_block = _gen_attention_mask_block(col_widths, s_cols_mask, 1, input.device)
                summary_mask.append(summary_mask_block)

        attention_mask = (1 - torch.cat([torch.cat(mask) for mask in masks_to_concat])).to(torch.bool)
        return attention_mask

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # TODO: replace with attention_mask
        lengths = attention_mask.sum(-1)

        input = hidden_states.permute(1, 0, 2)
        right_context = self._gen_right_context(input)
        utterance = input[: input.size(0) - self.right_context_length]
        attention_mask = self._gen_attention_mask(utterance)
        mems = (
            self.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)[:-1]
            if self.use_mem
            else torch.empty(0).to(dtype=input.dtype, device=input.device)
        )
        hidden_states = utterance
        for layer in self.emformer_layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, right_context, mems, attention_probs = layer(hidden_states, lengths, right_context, mems, attention_mask, output_attentions)

            if output_attentions:
                all_self_attentions = all_self_attentions + (attention_probs,)

        hidden_states = hidden_states.permute(1, 0, 2)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class EmformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EmformerConfig
    base_model_prefix = "emformer"
    main_input_name = "input_values"
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # gumbel softmax requires special init
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_time_reduced_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # in inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = non_padded_lengths.div(self.config.time_reduction_stride, rounding_mode="trunc")

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations make sure that all values before the output lengths indices are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask


EMFORMER_START_DOCSTRING = r"""
    Emformer was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
    Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
    Auli.

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
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
            into an array of type *List[float]* or a *numpy.ndarray*, *e.g.* via the soundfile library (*pip install
            soundfile*). To prepare the array into *input_values*, the [`Wav2Vec2Processor`] should be used for padding
            and conversion into a tensor of type *torch.FloatTensor*. See [`Wav2Vec2Processor.__call__`] for details.
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
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
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
            weight_init_scale_strategy="depthwise",
            tanh_on_mem=True,
            negative_inf=-1e8,
        )

        self.output_linear = nn.Linear(transformer_input_dim, config.output_dim)
        self.layer_norm = nn.LayerNorm(config.output_dim)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(EMFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_PROCESSOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        hidden_states = self.output_linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class RNNTCustomLSTM(torch.nn.Module):
    r"""Custom long-short-term memory (LSTM) block that applies layer normalization
    to internal nodes.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        layer_norm (bool, optional): if ``True``, enables layer normalization. (Default: ``False``)
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
        if layer_norm:
            self.c_norm = torch.nn.LayerNorm(hidden_dim, eps=layer_norm_epsilon)
            self.g_norm = torch.nn.LayerNorm(4 * hidden_dim, eps=layer_norm_epsilon)
        else:
            self.c_norm = torch.nn.Identity()
            self.g_norm = torch.nn.Identity()

        self.hidden_dim = hidden_dim

    def forward(
        self, input: torch.Tensor, state: Optional[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        r"""Forward pass.

        B: batch size; T: maximum sequence length in batch; D: feature dimension of each input sequence element.

        Args:
            input (torch.Tensor): with shape `(T, B, D)`.
            state (List[torch.Tensor] or None): list of tensors
                representing internal state generated in preceding invocation of ``forward``.

        Returns:
            (torch.Tensor, List[torch.Tensor]):
                torch.Tensor
                    output, with shape `(T, B, hidden_dim)`.
                List[torch.Tensor]
                    list of tensors representing internal state generated in current invocation of ``forward``.
        """
        if state is None:
            B = input.size(1)
            h = torch.zeros(B, self.hidden_dim, device=input.device, dtype=input.dtype)
            c = torch.zeros(B, self.hidden_dim, device=input.device, dtype=input.dtype)
        else:
            h, c = state

        gated_input = self.x2g(input)
        outputs = []
        for gates in gated_input.unbind(0):
            gates = gates + self.p2g(h)
            gates = self.g_norm(gates)
            input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
            input_gate = input_gate.sigmoid()
            forget_gate = forget_gate.sigmoid()
            cell_gate = cell_gate.tanh()
            output_gate = output_gate.sigmoid()
            c = forget_gate * c + input_gate * cell_gate
            c = self.c_norm(c)
            h = output_gate * c.tanh()
            outputs.append(h)

        output = torch.stack(outputs, dim=0)
        state = [h, c]

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
        self.embedding = torch.nn.Embedding(config.vocab_size, config.symbol_embedding_dim)
        self.input_layer_norm = torch.nn.LayerNorm(config.symbol_embedding_dim)
        self.lstm_layers = torch.nn.ModuleList(
            [
                RNNTCustomLSTM(
                    config.symbol_embedding_dim if idx == 0 else config.lstm_hidden_dim,
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
            input (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol, i.e. in range `[0, num_symbols)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
            state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing internal state generated in preceding invocation of ``forward``. (Default: ``None``)

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output encoding sequences, with shape `(B, U, output_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing number of valid elements for i-th
                    batch element in output encoding sequences.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors representing internal state generated in current invocation
                    of ``forward``.
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
        # TODO: parameterize
        self.max_output_length = 256

        # Initialize weights and apply final processing
        self.post_init()

    def _gen_next_token_probs(self, encoder_out, predictor_out):
        joined_out = self.joiner(encoder_out, predictor_out)
        joined_out = torch.nn.functional.log_softmax(joined_out, dim=3)
        return joined_out[:, 0, 0]

    def _greedy_decode(self, encoder_out, encoder_lengths, device):
        past_rnn_state = None
        batch_size = encoder_out.shape[0]

        logits = [[] for _ in range(batch_size)]

        last_label = torch.full([batch_size, 1], fill_value=self.blank_token_id, dtype=torch.long, device=device)
        blank_mask = torch.full([batch_size], fill_value=0, dtype=torch.bool, device=device)

        max_time_len = encoder_out.shape[1]
        for time_idx in range(max_time_len):
            encoder_slice = encoder_out.narrow(dim=1, start=time_idx, length=1)

            not_blank = True
            n_decoded_tokens = 0

            blank_mask *= False
            time_mask = time_idx >= encoder_lengths
            blank_mask.bitwise_or_(time_mask)

            while not_blank and (self.max_output_length is None or n_decoded_tokens < self.max_output_length):
                rnn_out, current_rnn_state = self.predictor(last_label, past_rnn_state)
                next_token_logits = self._gen_next_token_probs(encoder_slice, rnn_out)
                next_token_id = next_token_logits.argmax(-1)

                token_is_blank = next_token_id == self.blank_token_id
                blank_mask.bitwise_or_(token_is_blank)

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
            logits[batch_idx] = torch.stack(seq_logits, dim=-2)

        return logits

    @add_start_docstrings_to_model_forward(EMFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_PROCESSOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        encoded_frames = encoder_outputs[0]

        attention_mask = self._get_time_reduced_attention_mask(input_features.shape[1], attention_mask)

        logits = self._greedy_decode(encoded_frames, attention_mask.sum(-1), encoded_frames.device)
        logits = nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=self.pad_token_id)

        # TODO: support RNN-T loss
        loss = None
        if not return_dict:
            output = (logits,) + encoder_outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
