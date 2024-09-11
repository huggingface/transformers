# coding=utf-8
# Copyright 2024 Mesh TensorFlow authors, TimesFM Authors and HuggingFace Inc. team.
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

import copy
import math
import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_timesfm import TimesFMConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TimesFMConfig"
_CHECKPOINT_FOR_DOC = "google/timesfm-1.0-200m"


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of nn.Module)
####################################################
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`Dict[int, list]`, *optional*):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the timesfm models have the
            following number of attention modules:

                - google/timesfm-1.0-200m: 6
                - google-timesfm/timesfm-base: 12
                - google-timesfm/timesfm-large: 24
                - google-timesfm/timesfm-3b: 24
                - google-timesfm/timesfm-11b: 24

    Example:

    ```python
    # Here is an example of a device map on a machine with 4 GPUs using google-timesfm/timesfm-3b, which has a total of 24 attention modules:
    model = TimesFMForConditionalGeneration.from_pretrained("google-timesfm/timesfm-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```python
    # On a 4 GPU machine with google-timesfm/timesfm-3b:
    model = TimesFMForConditionalGeneration.from_pretrained("google-timesfm/timesfm-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""


# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->TimesFM
class TimesFMLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the TimesFM style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # TimesFM uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


try:
    from apex.normalization import FusedRMSNorm

    TimesFMLayerNorm = FusedRMSNorm  # noqa

    logger.info(
        "Discovered apex.normalization.FusedRMSNorm - will use it instead of TimesFMLayerNorm"
    )
except ImportError:
    # using the normal TimesFMLayerNorm
    pass
except Exception:
    logger.warning(
        "discovered apex but it failed to load, falling back to TimesFMLayerNorm"
    )
    pass

ALL_LAYERNORM_LAYERS.append(TimesFMLayerNorm)


class TimesFMResidualBlock(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, dropout=0.1):
        super().__init__()

        self.hidden_layer = nn.Sequential(nn.Linear(input_dims, hidden_dims), nn.SiLU())
        self.output_layer = nn.Linear(hidden_dims, output_dims)
        self.residual_layer = nn.Linear(input_dims, output_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        hidden = self.hidden_layer(inputs)
        output = self.output_layer(hidden)
        output = self.dropout(output)
        residual = self.residual_layer(inputs)

        return output + residual


class TimesFMPositionalEmbedding(nn.Module):
    """Generates position embedding for a given 1-d sequence.

    Attributes:
      min_timescale: Start of the geometric index. Determines the periodicity of
        the added signal.
      max_timescale: End of the geometric index. Determines the frequency of the
        added signal.
      embedding_dims: Dimension of the embedding to be generated.
    """

    def __init__(self, min_timescale=1, max_timescale=10000, embedding_dims=0):
        super().__init__()

        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.embedding_dims = embedding_dims

    def forward(self, seq_length=None, position=None):
        """Generates a tensor of sinusoids with different frequencies.

        Args:
          seq_length: an optional Python int defining the output sequence length.
            if the `position` argument is specified.
          position:   [B, seq_length], optional position for each token in the
            sequence, only required when the sequence is packed.

        Returns:
          [B, seqlen, D] if `position` is specified, else [1, seqlen, D]
        """
        if position is None:
            if seq_length is None:
                raise ValueError("If position is None, seq_length should be specified.")
            # [1, seqlen]
            position = torch.arange(seq_length, dtype=torch.float32).unsqueeze(0)
        else:
            if position.ndim != 2:
                raise ValueError(
                    f"position should have 2 dimensions, got {position.ndim}"
                )

        num_timescales = self.embedding_dims // 2
        log_timescale_increment = math.log(
            float(self.max_timescale) / float(self.min_timescale)
        ) / max(torch.tensor(num_timescales, dtype=torch.float32) - 1, 1)
        inv_timescales = self.min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
        )
        scaled_time = position.unsqueeze(2) * inv_timescales.unsqueeze(0).unsqueeze(0)
        signal = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=2
        ).type(torch.float32)

        signal = F.pad(signal, (0, 0, 0, self.embedding_dims % 2))
        return signal


class TimesFMDenseActDense(nn.Module):
    def __init__(self, config: TimesFMConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=True)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=True)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class TimesFMLayerFF(nn.Module):
    def __init__(self, config: TimesFMConfig):
        super().__init__()

        self.DenseReluDense = TimesFMDenseActDense(config)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class TimesFMPerHeadDimScale(nn.Module):
    def __init__(self, config: TimesFMConfig):
        super().__init__()
        dim = config.d_model // config.num_heads
        r_softplus_0 = 1.442695041
        self.scale_factor = r_softplus_0 / math.sqrt(dim)
        self.scale = nn.Parameter(torch.empty(self.dim))

    def forward(self, hidden_states):
        scale = self.scale_factor * F.softplus(self.scale)
        return hidden_states * scale


class TimesFMAttention(nn.Module):
    def __init__(self, config: TimesFMConfig, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=True)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=True)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=True)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=True)
        self.per_head_dim_scale = TimesFMPerHeadDimScale(config)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        key_length = (
            real_seq_length if key_value_states is None else key_value_states.shape[1]
        )

        def shape(states):
            """projection"""
            return states.view(
                batch_size, -1, self.n_heads, self.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return (
                states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
            )

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        unscaled_query_states = shape(
            self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)
        query_states = self.per_head_dim_scale(unscaled_query_states)

        # get key/value states
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device
                )

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = (
                    position_bias + mask
                )  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(
            torch.matmul(attn_weights, value_states)
        )  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (
            (key_states, value_states) if (self.is_decoder and use_cache) else None
        )
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class TimesFMTransformerLayer(nn.Module):
    def __init__(self, config: TimesFMConfig):
        super().__init__()
        self.attention = TimesFMAttention(config)
        self.ff = TimesFMLayerFF(config)
        self.layer_norm = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, inputs, mask=None):
        x = self.layer_norm(inputs)
        x = self.attention(x, mask=mask)
        x = self.dropout(x)
        x = x + inputs
        x = self.ff(x)
        return x


class TimesFMTransformerStack(nn.Module):
    def __init__(self, config: TimesFMConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [TimesFMTransformerLayer(config) for _ in range(config.num_layers)]
        )

    def forward(self, hidden_states, mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=mask)
        return hidden_states


class TimesFMModel(PreTrainedModel):
    def __init__(self, config: TimesFMConfig):
        super().__init__(config)

        self.freq_emb = nn.Embedding(
            num_embeddings=config.freq_size,
            embedding_dim=config.d_model,
        )
        self.position_emb = TimesFMPositionalEmbedding(
            embedding_dims=config.d_model,
        )

        self.input_ff_layer = TimesFMResidualBlock(
            input_dims=config.patch_len * 2,
            output_dims=config.d_model,
            hidden_dims=config.d_ff,
            dropout=config.dropout_rate,
        )

        self.stacked_transformer_layer = TimesFMTransformerStack(config)

    def preprocess_inputs(self, inputs):
        assert len(inputs.shape) == 3  # (batch_size, num_patches, patch_len)
        inputs_mean = inputs.mean(dim=(1, 2))
        inputs_std = inputs.std(dim=(1, 2))
        processed_input = (inputs - inputs_mean[:, None, None]) / inputs_std[
            :, None, None
        ]
        return processed_input, (inputs_mean, inputs_std)

    def create_causal_mask(batch_size, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(1)
        mask = mask.expand(batch_size, 1, seq_len, seq_len)
        mask = mask.float().masked_fill(mask, -2.3819763e38).masked_fill(~mask, 0.0)
        return mask

    def forward(
        self,
        input_ts,
    ):
        batch_size = input_ts.shape[0]
        patched_inputs = input_ts.reshape(batch_size, -1, self.config.patch_len)
        patched_pads = torch.zeros_like(patched_inputs)
        patched_inputs, input_stats = self.preprocess_inputs(patched_inputs)
        concat_inputs = torch.concat([patched_inputs, patched_pads], dim=-1)

        model_input = self.input_ff_layer(concat_inputs)
        position_emb = self.position_emb(seq_length=model_input.shape[1]).expand(
            model_input.shape[0], -1, -1
        )
        model_input = model_input + position_emb
        f_emb = self.freq_emb(
            torch.zeros((batch_size, 1), dtype=torch.long)
        )  # freq set to zero, change if needed
        model_input = model_input + f_emb
        mask = self.create_causal_mask(model_input.shape[0], model_input.shape[1])
        model_output = self.stacked_transformer_layer(model_input, mask=mask)
        return model_output, input_stats


class TimesFMPredictionHead(nn.Module):
    def __init__(self, config: TimesFMConfig):
        super().__init__()
        self.config = config
        self.horizon_ff_layer = TimesFMResidualBlock(
            input_dims=config.d_model,
            output_dims=config.horizon_len,
            hidden_dims=config.d_ff,
            dropout=config.dropout_rate,
        )

    def postprocess_outputs(self, outputs, stats):
        mean, std = stats
        return outputs * std[:, None, None, None] + mean[:, None, None, None]

    def forward(self, model_output, input_stats):
        batch_size = model_output.shape[0]
        output_ts = self.horizon_ff_layer(model_output)

        assert self.config.d_model % self.config.horizon_len == 0
        num_outputs = self.config.d_model // self.config.horizon_len

        output_ts = output_ts.reshape(
            batch_size, -1, self.config.horizon_len, num_outputs
        )
        output_ts = self.postprocess_outputs(output_ts, input_stats)
        return output_ts


class TimesFMForPrediction(PreTrainedModel):
    def __init__(self, config: TimesFMConfig):
        super().__init__(config)
        self.timesfm = TimesFMModel(config)
        self.prediction_head = TimesFMPredictionHead(config)

    def forward(
        self,
        input_ts,
    ):
        model_output, input_stats = self.timesfm(input_ts)
        output_ts = self.prediction_head(model_output, input_stats)
        return output_ts


TIMESFM_START_DOCSTRING = r"""

    The TIMESFM model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
    Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan
    Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a
    text-to-text denoising generative setting.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TimesFMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TIMESFM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. TIMESFM is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            [What are input IDs?](../glossary#input-ids)

            To know more on how to prepare `input_ids` for pretraining take a look a [TIMESFM Training](./timesfm#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            TIMESFM uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [TIMESFM
            Training](./timesfm#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at
            the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
            of `inputs_embeds`.

        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
