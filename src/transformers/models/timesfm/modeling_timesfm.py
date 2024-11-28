# coding=utf-8
# Copyright 2024 Google LLC and HuggingFace Inc. team.
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
"""PyTorch TimesFM model."""


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of nn.Module)
####################################################

import logging
import math
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from .configuration_timesfm import TimesFMConfig


@dataclass
class TimesFMOutput(BaseModelOutput):
    mean_predictions: np.ndarray = None
    full_predictions: np.ndarray = None


class TimesFMTransformerMLP(nn.Module):
    """Pax transformer MLP in pytorch."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size, eps=1e-6)

    def forward(self, x, paddings=None):
        gate_inp = self.layer_norm(x)
        gate = self.gate_proj(gate_inp)
        gate = F.relu(gate)
        outputs = self.down_proj(gate)
        if paddings is not None:
            outputs = outputs * (1.0 - paddings[:, :, None])
        return outputs + x


class TimesFMResidualBlock(nn.Module):
    """TimesFM residual block."""

    def __init__(
        self,
        input_dims,
        hidden_dims,
        output_dims,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        # Hidden Layer
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.SiLU(),
        )

        # Output Layer
        self.output_layer = nn.Linear(hidden_dims, output_dims)
        # Residual Layer
        self.residual_layer = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        hidden = self.hidden_layer(x)
        output = self.output_layer(hidden)
        residual = self.residual_layer(x)
        return output + residual


class TimesFMRMSNorm(torch.nn.Module):
    """Pax rms norm in pytorch."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)


class TimesFMPositionalEmbedding(nn.Module):
    """Generates position embedding for a given 1-d sequence.

    Attributes:
        min_timescale: Start of the geometric index. Determines the periodicity of
          the added signal.
        max_timescale: End of the geometric index. Determines the frequency of the
          added signal.
        embedding_dims: Dimension of the embedding to be generated.
    """

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: int = 1,
        max_timescale: int = 10_000,
    ) -> None:
        super().__init__()
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.embedding_dims = embedding_dims

    def forward(self, seq_length=None, position=None):
        """Generates a Tensor of sinusoids with different frequencies.

        Args:
            seq_length: an optional Python int defining the output sequence length.
              if the `position` argument is specified.
            position:   [B, seq_length], optional position for each token in the
              sequence, only required when the sequence is packed.

        Returns:
            [B, seqlen, D] if `position` is specified, else [1, seqlen, D]
        """
        if position is None:
            assert seq_length is not None
            # [1, seqlen]
            position = torch.arange(seq_length, dtype=torch.float32).unsqueeze(0)
        else:
            assert position.ndim == 2, position.shape

        num_timescales = self.embedding_dims // 2
        log_timescale_increment = math.log(float(self.max_timescale) / float(self.min_timescale)) / max(
            num_timescales - 1, 1
        )
        inv_timescales = self.min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
        )
        scaled_time = position.unsqueeze(2) * inv_timescales.unsqueeze(0).unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        # Padding to ensure correct embedding dimension
        signal = F.pad(signal, (0, 0, 0, self.embedding_dims % 2))
        return signal


class TimesFMAttention(nn.Module):
    """Implements the attention used in TimesFM."""

    def __init__(self, config: TimesFMConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = config.model_dim
        self.head_dim = config.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = nn.Parameter(
            torch.empty((self.head_dim,), dtype=torch.float32),
        )

        self.qkv_proj = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

    def _per_dim_scaling(self, query: torch.Tensor) -> torch.Tensor:
        # [batch_size, n_local_heads, input_len, head_dim]
        r_softplus_0 = 1.442695041
        softplus_func = torch.nn.Softplus()
        scale = r_softplus_0 / math.sqrt(self.head_dim)
        scale = scale * softplus_func(self.scaling)
        return query * scale[None, None, None, :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_write_indices: torch.Tensor | None = None,
        kv_cache: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xq = self._per_dim_scaling(xq)

        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        if kv_cache is not None and kv_write_indices is not None:
            k_cache, v_cache = kv_cache
            k_cache.index_copy_(1, kv_write_indices, xk)
            v_cache.index_copy_(1, kv_write_indices, xv)

            key = k_cache
            value = v_cache
        else:
            key = xk
            value = xv
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # [batch_size, n_local_heads, input_len, max_seq_len]
        scores = torch.matmul(q, k.transpose(2, 3))

        if attention_mask is not None:
            scores = scores + attention_mask

        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = torch.matmul(scores, v)
        # return scores, output.transpose(1, 2).contiguous()

        # [batch_size, input_len, hidden_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)
        output = self.o_proj(output)
        return output, scores


class TimesFMDecoderLayer(nn.Module):
    """Transformer layer."""

    def __init__(self, config: TimesFMConfig):
        super().__init__()
        self.self_attn = TimesFMAttention(config)
        self.mlp = TimesFMTransformerMLP(config.model_dim, config.intermediate_size)
        self.input_layernorm = TimesFMRMSNorm(config.model_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        paddings: torch.Tensor,
        kv_write_indices: torch.Tensor | None = None,
        kv_cache: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, scores = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        # MLP
        hidden_states = self.mlp(hidden_states, paddings=paddings)

        return scores, hidden_states


class TimesFMStackedDecoder(nn.Module):
    """Stacked transformer layer."""

    def __init__(self, config: TimesFMConfig):
        super().__init__()

        self.layers = nn.ModuleList([TimesFMDecoderLayer(config) for _ in range(config.num_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        paddings: torch.Tensor,
        kv_write_indices: torch.Tensor | None = None,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> torch.Tensor:
        padding_mask = timesfm_convert_paddings_to_mask(paddings, hidden_states.dtype)
        atten_mask = timesfm_causal_mask(hidden_states)
        mask = timesfm_merge_masks(padding_mask, atten_mask)
        all_attentions = []
        all_hidden_states = []

        for i in range(len(self.layers)):
            layer = self.layers[i]
            kv_cache = kv_caches[i] if kv_caches is not None else None
            scores, hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=mask,
                paddings=paddings,
                kv_write_indices=kv_write_indices,
                kv_cache=kv_cache,
            )
            if output_attentions:
                all_attentions.append(scores)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        return hidden_states, all_attentions, all_hidden_states


# Move utility functions here
def timesfm_masked_mean_std(inputs: torch.Tensor, padding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculates mean and standard deviation of `inputs` across axis 1.

    It excludes values where `padding` is 1.

    Args:
        inputs: A PyTorch tensor of shape [b, n, p].
        padding: A PyTorch tensor of shape [b, n, p] with values 0 or 1.

    Returns:
        A tuple containing the mean and standard deviation.
        We return the statistics of the first patch with more than three non-padded values.
    """

    # Selecting the first patch with more than 3 unpadded values.
    def _get_patch_index(arr: torch.Tensor):
        indices = torch.argmax((arr >= 3).to(torch.int32), dim=1)
        row_sum = (arr >= 3).to(torch.int32).sum(dim=1)
        return torch.where(row_sum == 0, arr.shape[1] - 1, indices)

    pad_sum = torch.sum(1 - padding, dim=2)
    patch_indices = _get_patch_index(pad_sum)
    bidxs = torch.arange(inputs.shape[0])

    arr = inputs[bidxs, patch_indices, :]
    pad = padding[bidxs, patch_indices, :]

    # Create a mask where padding is 0
    mask = 1 - pad

    # Calculate the number of valid elements
    num_valid_elements = torch.sum(mask, dim=1)
    num_valid_elements = torch.where(
        num_valid_elements == 0,
        torch.tensor(1, dtype=num_valid_elements.dtype, device=num_valid_elements.device),
        num_valid_elements,
    )

    # Calculate the masked sum and squared sum
    masked_sum = torch.sum(arr * mask, dim=1)
    masked_squared_sum = torch.sum((arr * mask) ** 2, dim=1)

    # Calculate the masked mean and standard deviation
    masked_mean = masked_sum / num_valid_elements
    masked_var = masked_squared_sum / num_valid_elements - masked_mean**2
    masked_var = torch.where(
        masked_var < 0.0,
        torch.tensor(0.0, dtype=masked_var.dtype, device=masked_var.device),
        masked_var,
    )
    masked_std = torch.sqrt(masked_var)

    return masked_mean, masked_std


def timesfm_shift_padded_seq(mask: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    """Shifts rows of seq based on the first 0 in each row of the mask.

    Args:
        mask: mask tensor of shape [B, N]
        seq: seq tensor of shape [B, N, P]

    Returns:
        The shifted sequence.
    """
    batch_size, num_seq, feature_dim = seq.shape

    new_mask: torch.BoolTensor = mask == 0

    # Use argmax to find the first True value in each row
    indices = new_mask.to(torch.int32).argmax(dim=1)

    # Handle rows with all zeros
    indices[~new_mask.any(dim=1)] = -1

    # Create index ranges for each sequence in the batch
    idx_range = torch.arange(num_seq).to(seq.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, feature_dim)

    # Calculate shifted indices for each element in each sequence
    shifted_idx = (idx_range - indices[:, None, None]) % num_seq

    # Gather values from seq using shifted indices
    shifted_seq = seq.gather(1, shifted_idx)

    return shifted_seq


def timesfm_moving_average(arr: torch.Tensor, window_size: int) -> list[torch.Tensor]:
    """Calculates the moving average using NumPy's convolution function."""
    # Pad with zeros to handle initial window positions
    arr_padded = np.pad(arr, (window_size - 1, 0), "constant")
    smoothed_arr = np.convolve(arr_padded, np.ones(window_size), "valid") / window_size
    return [smoothed_arr, arr - smoothed_arr]


def timesfm_get_large_negative_number(dtype: torch.dtype) -> torch.Tensor:
    """Returns a large negative value for the given dtype."""
    if dtype.is_floating_point:
        dtype_max = torch.finfo(dtype).max
    else:
        dtype_max = torch.iinfo(dtype).max
    return torch.tensor(-0.7 * dtype_max, dtype=dtype)


def timesfm_causal_mask(input_t: torch.Tensor) -> torch.Tensor:
    """Computes and returns causal mask.

    Args:
        input_t: A torch.Tensor of shape [B, T, D].

    Returns:
        An attention_mask torch.Tensor of shape [1, 1, T, T]. Attention mask has
        already been converted to large negative values.
    """
    assert input_t.dtype.is_floating_point, input_t.dtype
    large_negative_number = timesfm_get_large_negative_number(input_t.dtype)
    t = input_t.shape[1]
    col_idx = torch.arange(t).unsqueeze(0).repeat(t, 1)
    row_idx = torch.arange(t).unsqueeze(1).repeat(1, t)
    mask = (row_idx < col_idx).to(input_t.dtype) * large_negative_number
    return mask.unsqueeze(0).unsqueeze(0).to(input_t.device)  # Equivalent to jnp.newaxis


def timesfm_convert_paddings_to_mask(paddings: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Converts binary paddings to a logit mask ready to add to attention matrix.

    Args:
        paddings: binary torch.Tensor of shape [B, T], with 1 denoting padding
          token.
        dtype: data type of the input.

    Returns:
        A torch.Tensor of shape [B, 1, 1, T] ready to add to attention logits.
    """
    attention_mask = paddings.detach().clone()
    attention_mask = attention_mask[:, None, None, :]  # Equivalent to jnp.newaxis
    attention_mask *= timesfm_get_large_negative_number(dtype)
    return attention_mask


def timesfm_merge_masks(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Merges 2 masks.

    logscale mask is expected but 0/1 mask is also fine.

    Args:
        a: torch.Tensor of shape [1|B, 1, 1|T, S].
        b: torch.Tensor of shape [1|B, 1, 1|T, S].

    Returns:
        torch.Tensor of shape [1|B, 1, 1|T, S].
    """

    def expand_t(key_mask):
        query_mask = key_mask.transpose(-1, -2)  # Equivalent of jnp.transpose
        return torch.minimum(query_mask, key_mask)

    if a.shape[2] != b.shape[2]:
        if a.shape[2] == 1:
            a = expand_t(a)
        else:
            assert b.shape[2] == 1
            b = expand_t(b)

    assert a.shape[1:] == b.shape[1:], f"a.shape={a.shape}, b.shape={b.shape}."
    return torch.minimum(a, b)  # Element-wise minimum, similar to jnp.minimum


class TimesFMPreTrainedModel(PreTrainedModel):
    """handles the loading for all models."""

    config_class = TimesFMConfig
    base_model_prefix = "timesfm"
    main_input_name = "inputs"

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0, std=self.config.initializer_factor)

        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0, std=self.config.initializer_factor)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, TimesFMRMSNorm):
            nn.init.zeros_(module.weight)

        elif isinstance(module, TimesFMPositionalEmbedding):
            pass


class PatchedTimeSeriesDecoder(TimesFMPreTrainedModel):
    """Patched time-series decoder."""

    def __init__(self, config: TimesFMConfig):
        super().__init__(config)

        self.config = config
        self.input_ff_layer = TimesFMResidualBlock(
            input_dims=2 * config.patch_len,
            output_dims=config.model_dim,
            hidden_dims=config.model_dim,
        )
        self.freq_emb = nn.Embedding(num_embeddings=config.freq_size, embedding_dim=config.model_dim)
        self.horizon_ff_layer = TimesFMResidualBlock(
            input_dims=config.model_dim,
            output_dims=config.horizon_len * (1 + len(config.quantiles)),
            hidden_dims=config.model_dim,
        )
        self.stacked_transformer = TimesFMStackedDecoder(config=config)
        if self.config.use_positional_embedding:
            self.position_emb = TimesFMPositionalEmbedding(
                embedding_dims=self.config.model_dim,
            )

        # Initialize weights and apply final processing
        self.post_init()

    def _forward_transform(
        self, inputs: torch.Tensor, patched_pads: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Input is of shape [B, N, P]."""
        mu, sigma = timesfm_masked_mean_std(inputs, patched_pads)
        sigma = torch.where(
            sigma < self.config.tolerance,
            torch.tensor(1.0, dtype=sigma.dtype, device=sigma.device),
            sigma,
        )

        # Normalize each patch
        outputs = (inputs - mu[:, None, None]) / sigma[:, None, None]
        outputs = torch.where(
            torch.abs(inputs - self.config.pad_val) < self.config.tolerance,
            torch.tensor(self.config.pad_val, dtype=outputs.dtype, device=outputs.device),
            outputs,
        )
        return outputs, (mu, sigma)

    def _reverse_transform(self, outputs: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Output is of shape [B, N, P, Q]."""
        mu, sigma = stats
        return outputs * sigma[:, None, None, None] + mu[:, None, None, None]

    def _preprocess_input(
        self,
        input_ts: torch.Tensor,
        input_padding: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | None,
        torch.Tensor,
    ]:
        """Preprocess input for stacked transformer."""

        # Reshape into patches (using view for efficiency)
        bsize = input_ts.shape[0]
        patched_inputs = input_ts.view(bsize, -1, self.config.patch_len)
        patched_pads = input_padding.view(bsize, -1, self.config.patch_len)

        patched_inputs = torch.where(
            torch.abs(patched_pads - 1.0) < self.config.tolerance,
            torch.tensor(0.0, dtype=patched_inputs.dtype, device=patched_inputs.device),
            patched_inputs,
        )
        patched_pads = torch.where(
            torch.abs(patched_inputs - self.config.pad_val) < self.config.tolerance,
            torch.tensor(1.0, dtype=patched_pads.dtype, device=patched_pads.device),
            patched_pads,
        )
        patched_inputs, stats = self._forward_transform(patched_inputs, patched_pads)

        # B x N x D
        patched_inputs = patched_inputs * (1.0 - patched_pads)
        concat_inputs = torch.cat([patched_inputs, patched_pads], dim=-1)
        model_input = self.input_ff_layer(concat_inputs)

        # A patch should not be padded even if there is at least one zero.
        patched_padding = torch.min(patched_pads, dim=-1)[0]  # Get the values from the min result
        if self.config.use_positional_embedding:
            pos_emb = self.position_emb(model_input.shape[1]).to(model_input.device)
            pos_emb = torch.concat([pos_emb] * model_input.shape[0], dim=0)
            pos_emb = timesfm_shift_padded_seq(patched_padding, pos_emb)
            model_input += pos_emb

        return model_input, patched_padding, stats, patched_inputs

    def _postprocess_output(
        self,
        model_output: torch.Tensor,
        num_outputs: int,
        stats: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Postprocess output of stacked transformer."""

        # B x N x (H.Q)
        output_ts = self.horizon_ff_layer(model_output)

        # Reshape using view
        b, n, _ = output_ts.shape
        output_ts = output_ts.view(b, n, self.config.horizon_len, num_outputs)

        return self._reverse_transform(output_ts, stats)

    def forward(
        self,
        input_ts: torch.Tensor,
        input_padding: torch.LongTensor,
        freq: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> torch.Tensor:
        num_outputs = len(self.config.quantiles) + 1
        model_input, patched_padding, stats, _ = self._preprocess_input(
            input_ts=input_ts,
            input_padding=input_padding,
        )
        f_emb = self.freq_emb(freq)  # B x 1 x D
        model_input += f_emb

        model_output, all_attentions, all_hidden_states = self.stacked_transformer(
            model_input,
            patched_padding,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        if output_hidden_states:
            all_hidden_states = [model_input] + all_hidden_states

        output_ts = self._postprocess_output(model_output, num_outputs, stats)
        return output_ts, all_attentions, all_hidden_states

    def decode(
        self,
        input_ts: torch.Tensor,
        paddings: torch.Tensor,
        freq: torch.LongTensor,
        horizon_len: int,
        output_patch_len: int | None = None,
        max_len: int = 512,
        return_forecast_on_context: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Auto-regressive decoding without caching.

        Args:
          input_ts: input time-series and paddings. Time-series shape B x C.
          paddings: padding shape B x (C + H) where H is the prediction length.
          freq: frequency shape B x 1
          horizon_len: prediction length.
          output_patch_len: output length to be fetched from one step of
            auto-regressive decoding.
          max_len: maximum training context length.
          return_forecast_on_context: whether to return the model forecast on the
            context except the first input patch.

        Returns:
          Tuple of two forecasting results:
          - Point (mean) output predictions as a tensor with shape B x H'.
          - Full predictions (mean and quantiles) as a tensor with shape
            B x H' x (1 + # quantiles).
          In particular, if return_forecast_on_context is True, H' is H plus
          the forecastable context length, i.e. context_len - (first) patch_len.
        """
        final_out = input_ts
        context_len = final_out.shape[1]
        full_outputs = []
        if paddings.shape[1] != final_out.shape[1] + horizon_len:
            raise ValueError(
                "Length of paddings must match length of input + horizon_len:"
                f" {paddings.shape[1]} != {final_out.shape[1]} + {horizon_len}"
            )
        if output_patch_len is None:
            output_patch_len = self.config.horizon_len
        num_decode_patches = (horizon_len + output_patch_len - 1) // output_patch_len
        for step_index in range(num_decode_patches):
            current_padding = paddings[:, 0 : final_out.shape[1]]
            input_ts = final_out[:, -max_len:]
            input_padding = current_padding[:, -max_len:]
            fprop_outputs, all_attentions, all_hidden_states = self.forward(
                input_ts,
                input_padding,
                freq,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            if return_forecast_on_context and step_index == 0:
                # For the first decodings step, collect the model forecast on the
                # context except the unavailable first input batch forecast.
                new_full_ts = fprop_outputs[:, :-1, : self.config.patch_len, :]
                new_full_ts = fprop_outputs.view(new_full_ts.size(0), -1, new_full_ts.size(3))

                full_outputs.append(new_full_ts)

            # (full batch, last patch, output_patch_len, index of mean forecast = 0)
            new_ts = fprop_outputs[:, -1, :output_patch_len, 0]
            new_full_ts = fprop_outputs[:, -1, :output_patch_len, :]
            # (full batch, last patch, output_patch_len, all output indices)
            full_outputs.append(new_full_ts)
            final_out = torch.concatenate([final_out, new_ts], axis=-1)

        if return_forecast_on_context:
            # `full_outputs` indexing starts at after the first input patch.
            full_outputs = torch.concatenate(full_outputs, axis=1)[
                :, : (context_len - self.config.patch_len + horizon_len), :
            ]
        else:
            # `full_outputs` indexing starts at the forecast horizon.
            full_outputs = torch.concatenate(full_outputs, axis=1)[:, 0:horizon_len, :]

        return full_outputs[:, :, 0], full_outputs, fprop_outputs, all_attentions, all_hidden_states


class TimesFMModel(TimesFMPreTrainedModel):
    def __init__(self, config: TimesFMConfig):
        super().__init__(config)

        self.config = config

        self.decoder = PatchedTimeSeriesDecoder(config)

        self.context_len = config.context_len
        self.horizon_len = config.horizon_len
        self.input_patch_len = config.patch_len
        self.output_patch_len = config.horizon_len
        self.num_layers = config.num_layers
        self.model_dims = config.model_dim
        self.quantiles = config.quantiles
        self.num_heads = config.num_heads
        self.batch_size = config.batch_size
        self._horizon_start = self.context_len - self.input_patch_len

        # Initialize weights and apply final processing
        self.post_init()

    def _preprocess(self, inputs: Sequence[np.array], freq: Sequence[int]) -> tuple[np.array, np.array, int]:
        """Formats and pads raw inputs to feed into the model.

        This function both pads each time series to match the context length, and
        pads the inputs to meet the SPMD shape requirement.

        Args:
          inputs: A list of 1d JTensors. Each JTensor is the context time series of
            a single forecast task.
          freq: list of frequencies

        Returns:
        A tuple of:
        - the padded input time series to meet the model required context.
        - the padding indicator.
        - the number of padded examples for SPMD so that each core has the same
            number (a multiple of `batch_size`) of examples.
        """
        input_ts, input_padding, inp_freq = [], [], []

        for i, ts in enumerate(inputs):
            input_len = ts.shape[0]
            padding = np.zeros(shape=(input_len + self.horizon_len,), dtype=float)
            if input_len < self.context_len:
                num_front_pad = self.context_len - input_len
                ts = np.concatenate([np.zeros(shape=(num_front_pad,), dtype=float), ts], axis=0)
                padding = np.concatenate([np.ones(shape=(num_front_pad,), dtype=float), padding], axis=0)
            elif input_len > self.context_len:
                ts = ts[-self.context_len :]
                padding = padding[-(self.context_len + self.horizon_len) :]

            input_ts.append(ts)
            input_padding.append(padding)
            inp_freq.append(freq[i])

        return (
            np.stack(input_ts, axis=0),
            np.stack(input_padding, axis=0),
            np.array(inp_freq).astype(np.int32).reshape(-1, 1),
        )

    def forward(
        self,
        inputs: Sequence[Any],
        freq: Sequence[int] | None = None,
        window_size: int | None = None,
        forecast_context_len: int | None = None,
        return_forecast_on_context: bool = False,
        truncate_negative: bool = False,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forecasts on a list of time series.

        Args:
          inputs: list of time series forecast contexts. Each context time series
            should be in a format convertible to JTensor by `jnp.array`.
          freq: frequency of each context time series. 0 for high frequency
            (default), 1 for medium, and 2 for low. Notice this is different from
            the `freq` required by `forecast_on_df`.
          window_size: window size of trend + residual decomposition. If None then
            we do not do decomposition.
          forecast_context_len: optional max context length.
          return_forecast_on_context: True to return the forecast on the context
            when available, i.e. after the first input patch.
          truncate_negative: truncate to only non-negative values if all the contexts
            have non-negative values.

        Returns:
        A tuple for JTensors:
        - the mean forecast of size (# inputs, # forecast horizon),
        - the full forecast (mean + quantiles) of size
            (# inputs,  # forecast horizon, 1 + # quantiles).

        Raises:
        ValueError: If the checkpoint is not properly loaded.
        """
        if return_dict is None:
            return_dict = self.config.use_return_dict

        if forecast_context_len is None:
            fcontext_len = self.context_len
        else:
            fcontext_len = forecast_context_len
        inputs = [np.array(ts)[-fcontext_len:] for ts in inputs]
        inp_min = np.min([np.min(ts) for ts in inputs])

        if window_size is not None:
            new_inputs = []
            for ts in inputs:
                new_inputs.extend(timesfm_moving_average(ts, window_size))
            inputs = new_inputs

        if freq is None:
            logging.info("No frequency provided via `freq`. Default to high (0).")
            freq = [0] * len(inputs)

        if output_attentions is None:
            output_attentions = self.config.output_attentions
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        input_ts, input_padding, inp_freq = self._preprocess(inputs, freq)

        input_ts_in = torch.from_numpy(
            np.array(
                input_ts,
                dtype=np.float32,
            )
        )
        input_padding_in = torch.from_numpy(
            np.array(
                input_padding,
                dtype=np.float32,
            )
        )
        inp_freq_in = torch.from_numpy(
            np.array(
                inp_freq,
                dtype=np.int32,
            )
        ).long()
        mean_outputs, full_outputs, last_hidden_state, all_attentions, all_hidden_states = self.decoder.decode(
            input_ts=input_ts_in,
            paddings=input_padding_in,
            freq=inp_freq_in,
            horizon_len=self.horizon_len,
            return_forecast_on_context=return_forecast_on_context,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if window_size is not None:
            mean_outputs = mean_outputs[0::2, ...] + mean_outputs[1::2, ...]
            full_outputs = full_outputs[0::2, ...] + full_outputs[1::2, ...]
        if inp_min >= 0 and truncate_negative:
            mean_outputs = torch.maximum(mean_outputs, 0.0)
            full_outputs = torch.maximum(full_outputs, 0.0)

        if return_dict:
            return TimesFMOutput(
                last_hidden_state=last_hidden_state,
                attentions=all_attentions if output_attentions else None,
                hidden_states=all_hidden_states if output_hidden_states else None,
                mean_predictions=mean_outputs,
                full_predictions=full_outputs,
            )
        else:
            return_tuple = [last_hidden_state]
            if output_hidden_states:
                return_tuple.append(all_hidden_states)
            if output_attentions:
                return_tuple.append(all_attentions)
            return_tuple += [mean_outputs, full_outputs]
            return tuple(return_tuple)
