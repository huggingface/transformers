# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...activations import ACT2FN
from ...masking_utils import create_causal_mask
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..apertus.modeling_apertus import ApertusAttention
from ..clip.modeling_clip import CLIPMLP
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..timesfm.configuration_timesfm import TimesFmConfig
from ..timesfm.modeling_timesfm import (
    TimesFmModelForPrediction,
    TimesFmOutput,
    TimesFmOutputForPrediction,
    TimesFmPreTrainedModel,
    TimesFmResidualBlock,
)


logger = logging.get_logger(__name__)


class TimesFm2_5Config(TimesFmConfig):
    r"""
    This is the configuration class to store the configuration of a [`TimesFm2_5ModelForPrediction`]. It is used to
    instantiate a TimesFM 2.5 model according to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the TimesFM 2.5
    [google/timesfm-2.5-200m-transformers](https://huggingface.co/google/timesfm-2.5-200m-transformers) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        patch_length (`int`, *optional*, defaults to 32):
            The length of one patch in the input sequence.
        context_length (`int`, *optional*, defaults to 16384):
            The length of the input context.
        horizon_length (`int`, *optional*, defaults to 128):
            The length of the prediction horizon.
        quantiles (`list[float]`, *optional*, defaults to `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`):
            The quantiles to predict.
        hidden_size (`int`, *optional*, defaults to 1280):
            Size of the hidden layers.
        intermediate_size (`int`, *optional*, defaults to 1280):
            Dimension of the MLP representations.
        head_dim (`int`, *optional*, defaults to 80):
            Size of the key, query, value projections per attention head.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of key-value heads. Set equal to `num_attention_heads` for full (non-grouped) attention.
        num_hidden_layers (`int`, *optional*, defaults to 20):
            Number of Transformer layers.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the RMS normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention scores.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the attention linear projections.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        output_quantile_len (`int`, *optional*, defaults to 1024):
            Length of the quantile output projection dimension.
        decode_index (`int`, *optional*, defaults to 5):
            Index into the quantile dimension used to extract the point (median) forecast.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in MLP and transformer linear layers.
        activation (`str`, *optional*, defaults to `"swish"`):
            Activation function used in MLP and residual block layers (any key from `ACT2FN`).
        use_continuous_quantile_head (`bool`, *optional*, defaults to `True`):
            Whether to use the continuous quantile head for non-median quantile predictions.
        force_flip_invariance (`bool`, *optional*, defaults to `True`):
            Whether to apply flip-invariance averaging during forecasting.
        infer_is_positive (`bool`, *optional*, defaults to `True`):
            Whether to clamp forecasts to non-negative values when the input minimum is non-negative.
        max_position_embeddings (`int`, *optional*, defaults to 16384):
            Maximum sequence length supported by the rotary position encoding.
        rope_parameters (`RopeParameters` or `dict[str, RopeParameters]`, *optional*):
            Dictionary containing the RoPE configuration. Uses default rope type with theta=10000.0 when not set.

    Example:

    ```python
    >>> from transformers import TimesFm2_5Config, TimesFm2_5ModelForPrediction

    >>> configuration = TimesFm2_5Config()
    >>> model = TimesFm2_5ModelForPrediction(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "timesfm2_5"

    def __init__(
        self,
        patch_length: int = 32,
        context_length: int = 16384,
        horizon_length: int = 128,
        quantiles: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        hidden_size: int = 1280,
        intermediate_size: int = 1280,
        head_dim: int = 80,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        num_hidden_layers: int = 20,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        initializer_range: float = 0.02,
        output_quantile_len: int = 1024,
        decode_index: int = 5,
        use_bias: bool = False,
        activation: str = "swish",
        use_continuous_quantile_head: bool = True,
        force_flip_invariance: bool = True,
        infer_is_positive: bool = True,
        max_position_embeddings: int = 16384,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        **kwargs,
    ):
        self.num_key_value_heads = num_key_value_heads
        self.attention_bias = attention_bias
        self.output_quantile_len = output_quantile_len
        self.decode_index = decode_index
        self.use_bias = use_bias
        self.activation = activation
        self.use_continuous_quantile_head = use_continuous_quantile_head
        self.force_flip_invariance = force_flip_invariance
        self.infer_is_positive = infer_is_positive
        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = rope_parameters

        super().__init__(
            patch_length=patch_length,
            context_length=context_length,
            horizon_length=horizon_length,
            quantiles=quantiles,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rms_norm_eps=rms_norm_eps,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            initializer_range=initializer_range,
            num_hidden_layers=num_hidden_layers,
            use_positional_embedding=False,
            **kwargs,
        )
        # Delete inherited attributes that TimesFM 2.5 does not use
        del self.freq_size
        del self.pad_val
        del self.tolerance
        del self.normalize_inputs
        del self.use_positional_embedding
        del self.use_rotary_embeddings
        del self.min_timescale
        del self.max_timescale


@dataclass
@auto_docstring
class TimesFm2_5Output(TimesFmOutput):
    r"""
    context_mu (`torch.Tensor` of shape `(batch_size, num_patches)`):
        Running means computed per input patch during normalization.
    context_sigma (`torch.Tensor` of shape `(batch_size, num_patches)`):
        Running standard deviations computed per input patch during normalization.
    """

    context_mu: torch.Tensor | None = None
    context_sigma: torch.Tensor | None = None


@dataclass
@auto_docstring
class TimesFm2_5OutputForPrediction(TimesFmOutputForPrediction):
    r"""
    mean_predictions (`torch.Tensor` of shape `(batch_size, horizon_length)`):
        Deterministic forecasts after denormalization.
    full_predictions (`torch.Tensor` of shape `(batch_size, horizon_length, quantiles)`):
        Quantile forecasts including the median after denormalization.
    loss (`torch.Tensor` of shape `(1,)`, *optional*, returned when `future_values` is provided):
        Training loss combining MSE and quantile losses when targets are supplied.
    """

    pass


class TimesFm2_5MLP(CLIPMLP):
    def __init__(self, config: TimesFm2_5Config):
        super().__init__()
        self.activation_fn = ACT2FN[config.activation]


class TimesFm2_5ResidualBlock(TimesFmResidualBlock):
    """[`TimesFmResidualBlock`] variant with configurable `use_bias` and `activation`."""

    def __init__(self, config, input_dims: int, hidden_dims: int, output_dims: int, use_bias: bool | None = None):
        super().__init__(input_dims, hidden_dims, output_dims)
        use_bias = use_bias if use_bias is not None else config.use_bias
        self.input_layer = nn.Linear(input_dims, hidden_dims, bias=use_bias)
        self.output_layer = nn.Linear(hidden_dims, output_dims, bias=use_bias)
        self.residual_layer = nn.Linear(input_dims, output_dims, bias=use_bias)
        self.activation = ACT2FN[config.activation]


class TimesFm2_5RMSNorm(LlamaRMSNorm):
    pass


class TimesFm2_5RotaryEmbedding(LlamaRotaryEmbedding):
    pass


class TimesFm2_5Attention(ApertusAttention):
    """TimesFM 2.5 attention with learnable per-dimension query scaling."""

    def __init__(self, config: TimesFm2_5Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.scaling = nn.Parameter(torch.empty((self.head_dim,)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        scale = F.softplus(self.scaling).mul(1.442695041 / math.sqrt(self.head_dim))
        query_states = query_states * scale[None, None, None, :]

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            # scaling=1.0 because per-dimension learnable scaling is already applied to query_states above
            scaling=1.0,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class TimesFm2_5DecoderLayer(LlamaDecoderLayer):
    """TimesFM 2.5 Transformer decoder layer with pre/post RMS normalization and no KV cache."""

    def __init__(self, config: TimesFm2_5Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.pre_feedforward_layernorm = TimesFm2_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = TimesFm2_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states) + residual

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states) + residual

        return hidden_states


@auto_docstring
class TimesFm2_5PreTrainedModel(TimesFmPreTrainedModel):
    config_class = TimesFm2_5Config
    base_model_prefix = "model"
    _no_split_modules = ["TimesFm2_5DecoderLayer"]
    _supports_flash_attn = True
    _supports_flex_attn = True
    _can_record_outputs = {
        "hidden_states": TimesFm2_5DecoderLayer,
        "attentions": TimesFm2_5Attention,
    }


class TimesFm2_5Model(TimesFm2_5PreTrainedModel):
    def __init__(self, config: TimesFm2_5Config):
        super().__init__(config)
        self.config = config
        self.tolerance = 1e-6

        self.input_ff_layer = TimesFm2_5ResidualBlock(
            config,
            input_dims=2 * config.patch_length,
            hidden_dims=config.hidden_size,
            output_dims=config.hidden_size,
            use_bias=True,
        )

        self.layers = nn.ModuleList(
            [TimesFm2_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = TimesFm2_5RotaryEmbedding(config)
        self.gradient_checkpointing = False

        self.post_init()

    def _revin(
        self,
        hidden_states: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
        reverse: bool = False,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reversible instance normalization (RevIN).

        Normalizes or denormalizes `hidden_states` using the provided location and scale statistics.
        When `mask` is provided during normalization (reverse=False), masked positions are zeroed out.
        """
        if len(loc.shape) == len(hidden_states.shape) - 1:
            loc = loc[..., None]
            scale = scale[..., None]
        elif len(loc.shape) == len(hidden_states.shape) - 2:
            loc = loc[..., None, None]
            scale = scale[..., None, None]

        loc = loc.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        safe_scale = torch.where(scale < self.tolerance, torch.ones_like(scale), scale)

        if reverse:
            return hidden_states * scale + loc

        normed = (hidden_states - loc) / safe_scale
        if mask is not None:
            normed = torch.where(mask, torch.zeros_like(normed), normed)
        return normed

    @staticmethod
    def _update_running_stats(
        count: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        new_values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update running mean/std using Welford's online algorithm.

        Combines existing statistics (`count`, `mean`, `std`) with a new batch of values,
        respecting the boolean `mask` (True = masked/invalid).
        """
        is_valid = (~mask).to(new_values.dtype)
        inc_count = is_valid.sum(dim=-1)

        inc_count_safe = torch.where(inc_count == 0, torch.ones_like(inc_count), inc_count)
        inc_mean = (new_values * is_valid).sum(dim=-1) / inc_count_safe
        inc_mean = torch.where(inc_count == 0, torch.zeros_like(inc_mean), inc_mean)

        centered = new_values - inc_mean.unsqueeze(-1)
        inc_var = ((centered * is_valid) ** 2).sum(dim=-1) / inc_count_safe
        inc_var = torch.where(inc_count == 0, torch.zeros_like(inc_var), inc_var)
        inc_std = torch.sqrt(torch.clamp(inc_var, min=0.0))

        new_count = count + inc_count
        new_count_safe = torch.where(new_count == 0, torch.ones_like(new_count), new_count)

        new_mean = (count * mean + inc_mean * inc_count) / new_count_safe
        new_mean = torch.where(new_count == 0, torch.zeros_like(new_mean), new_mean)

        term1 = count * std.pow(2)
        term2 = inc_count * inc_std.pow(2)
        term3 = count * (mean - new_mean).pow(2)
        term4 = inc_count * (inc_mean - new_mean).pow(2)

        new_var = (term1 + term2 + term3 + term4) / new_count_safe
        new_var = torch.where(new_count == 0, torch.zeros_like(new_var), new_var)
        new_std = torch.sqrt(torch.clamp(new_var, min=0.0))

        return new_count, new_mean, new_std

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        past_values: torch.Tensor,
        past_values_padding: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TimesFm2_5Output:
        r"""
        past_values (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            Past values of the time series used as input to the model.
        past_values_padding (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Padding mask for the input. `1` indicates padded (masked) time steps, `0` indicates valid values.
        """
        batch_size, seq_len = past_values.shape
        patch_len = self.config.patch_length

        if past_values_padding is None:
            past_values_padding = torch.zeros_like(past_values, dtype=torch.long)

        patched_inputs = past_values.view(batch_size, -1, patch_len)
        patched_masks = past_values_padding[:, :seq_len].view(batch_size, -1, patch_len)
        patched_masks_bool = patched_masks >= 0.5

        count = past_values.new_zeros(batch_size)
        mean = past_values.new_zeros(batch_size)
        std = past_values.new_zeros(batch_size)
        mean_history: list[torch.Tensor] = []
        std_history: list[torch.Tensor] = []

        for i in range(patched_inputs.shape[1]):
            count, mean, std = self._update_running_stats(
                count, mean, std, patched_inputs[:, i, :], patched_masks_bool[:, i, :]
            )
            mean_history.append(mean)
            std_history.append(std)

        if mean_history:
            context_mu = torch.stack(mean_history, dim=1)
            context_sigma = torch.stack(std_history, dim=1)
        else:
            context_mu = mean.unsqueeze(1)
            context_sigma = std.unsqueeze(1)

        normed_inputs = self._revin(patched_inputs, context_mu, context_sigma, reverse=False, mask=patched_masks_bool)

        tokenizer_inputs = torch.cat(
            [normed_inputs, patched_masks_bool.to(dtype=normed_inputs.dtype)],
            dim=-1,
        )
        input_embeddings = self.input_ff_layer(tokenizer_inputs)

        patch_padding = patched_masks_bool[..., -1]

        sequence_length = input_embeddings.shape[1]
        num_masked = patch_padding.to(torch.int32).sum(dim=-1, keepdim=True)
        position_ids = torch.arange(sequence_length, device=input_embeddings.device).unsqueeze(0) - num_masked

        padding_mask = (~patch_padding).to(torch.int64)
        cache_position = torch.arange(sequence_length, device=input_embeddings.device)
        attention_mask = create_causal_mask(
            self.config, input_embeddings, padding_mask, cache_position, past_key_values=None
        )
        position_embeddings = self.rotary_emb(input_embeddings, position_ids)

        hidden_states = input_embeddings

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )

        loc = context_mu[:, -1]
        scale = torch.clamp(context_sigma[:, -1], min=self.tolerance)

        return TimesFm2_5Output(
            last_hidden_state=hidden_states,
            loc=loc,
            scale=scale,
            context_mu=context_mu,
            context_sigma=context_sigma,
        )


class TimesFm2_5ModelForPrediction(TimesFmModelForPrediction):
    def __init__(self, config: TimesFm2_5Config):
        super().__init__(config)
        self.config = config
        self.context_len = config.context_length
        self.horizon_len = config.horizon_length

        # Remove inherited attributes from parent TimesFmModelForPrediction
        del self.decoder
        del self.horizon_ff_layer

        self.model = TimesFm2_5Model(config)

        num_quantiles = len(config.quantiles) + 1
        self.output_projection_point = TimesFm2_5ResidualBlock(
            config,
            input_dims=config.hidden_size,
            hidden_dims=config.hidden_size,
            output_dims=config.horizon_length * num_quantiles,
        )
        self.output_projection_quantiles = TimesFm2_5ResidualBlock(
            config,
            input_dims=config.hidden_size,
            hidden_dims=config.hidden_size,
            output_dims=config.output_quantile_len * num_quantiles,
        )

        self.post_init()

    def _decode_and_project(
        self,
        normalized_ts: torch.Tensor,
        input_padding: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the decoder and project to point/quantile outputs.

        Returns:
            Tuple of (point_forecast, quantile_spreads), each of shape `(batch, length, num_quantiles)`.
        """
        model_outputs = self.model(
            past_values=normalized_ts,
            past_values_padding=input_padding,
            **kwargs,
        )

        hidden_states = model_outputs.last_hidden_state
        context_mu = model_outputs.context_mu
        context_sigma = model_outputs.context_sigma

        point_output = self.model._revin(
            self.output_projection_point(hidden_states), context_mu, context_sigma, reverse=True
        )
        quantile_output = self.model._revin(
            self.output_projection_quantiles(hidden_states), context_mu, context_sigma, reverse=True
        )

        batch_size, num_patches = point_output.shape[:2]
        num_quantiles = len(self.config.quantiles) + 1

        point_forecast = point_output.view(batch_size, num_patches, self.config.horizon_length, num_quantiles)[
            :, -1, :, :
        ]
        quantile_spreads = quantile_output.view(
            batch_size, num_patches, self.config.output_quantile_len, num_quantiles
        )[:, -1, :, :]

        # Ensure both outputs are on the same device for model parallelism
        quantile_spreads = quantile_spreads.to(point_forecast.device)

        return point_forecast, quantile_spreads, model_outputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        past_values: Sequence[torch.Tensor],
        window_size: int | None = None,
        future_values: torch.Tensor | None = None,
        forecast_context_len: int | None = None,
        truncate_negative: bool | None = None,
        force_flip_invariance: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TimesFm2_5OutputForPrediction:
        r"""
        past_values (`Sequence[torch.Tensor]`):
            Past values of the time series that serves as input to the model. Each tensor is a 1D time series.
        window_size (`int`, *optional*):
            Window size of trend + residual decomposition. If `None`, decomposition is not applied.
        future_values (`torch.Tensor`, *optional*):
            Optional future values used to compute the loss.
        forecast_context_len (`int`, *optional*):
            Optional context length override used during forecasting.
        truncate_negative (`bool`, *optional*):
            Whether to clamp outputs to non-negative values. If `None`, defaults to `config.infer_is_positive`.
        force_flip_invariance (`bool`, *optional*):
            Whether to apply the flip-invariance combination. If `None`, defaults to
            `config.force_flip_invariance`.
        """
        forecast_context_len = forecast_context_len or self.context_len
        device = past_values[0].device

        inputs = [ts[-forecast_context_len:] for ts in past_values]
        input_min = torch.min(torch.stack([torch.min(ts) for ts in inputs]))

        if window_size is not None:
            new_inputs: list[torch.Tensor] = []
            for ts in inputs:
                new_inputs.extend(self._timesfm_moving_average(ts, window_size))
            inputs = new_inputs

        if truncate_negative is None:
            truncate_negative = self.config.infer_is_positive
        if force_flip_invariance is None:
            force_flip_invariance = self.config.force_flip_invariance

        input_ts, input_padding = self._preprocess(inputs, context_len=forecast_context_len)
        input_ts = input_ts.to(device)
        input_padding = input_padding.to(device)

        mu_global = input_ts.mean(dim=1, keepdim=True)
        sigma_global = input_ts.std(dim=1, keepdim=True)

        normalized_ts = self.model._revin(input_ts, mu_global, sigma_global, reverse=False)

        pf_outputs, quantile_spreads, model_outputs = self._decode_and_project(normalized_ts, input_padding, **kwargs)

        if force_flip_invariance:
            flipped_pf, flipped_qs, _ = self._decode_and_project(-normalized_ts, input_padding, **kwargs)

            def _flip_quantiles(x: torch.Tensor) -> torch.Tensor:
                return torch.cat([x[..., :1], torch.flip(x[..., 1:], dims=(-1,))], dim=-1)

            pf_outputs = (pf_outputs - _flip_quantiles(flipped_pf)) / 2
            quantile_spreads = (quantile_spreads - _flip_quantiles(flipped_qs)) / 2

        horizon = min(self.horizon_len, pf_outputs.shape[1])
        full_forecast = pf_outputs[:, :horizon, :].clone()

        median_index = min(self.config.decode_index, full_forecast.shape[-1] - 1)
        if self.config.use_continuous_quantile_head:
            max_quantile_horizon = min(horizon, quantile_spreads.shape[1])
            for idx, _ in enumerate(self.config.quantiles, start=1):
                if idx == median_index or idx >= full_forecast.shape[-1]:
                    continue
                full_forecast[:, :max_quantile_horizon, idx] = (
                    quantile_spreads[:, :max_quantile_horizon, idx]
                    - quantile_spreads[:, :max_quantile_horizon, median_index]
                    + full_forecast[:, :max_quantile_horizon, median_index]
                )

        full_predictions = self.model._revin(full_forecast, mu_global, sigma_global, reverse=True)
        decode_index = min(self.config.decode_index, full_predictions.shape[-1] - 1)
        mean_predictions = full_predictions[:, :, decode_index]

        if window_size is not None:
            mean_predictions = mean_predictions[0::2, ...] + mean_predictions[1::2, ...]
            full_predictions = full_predictions[0::2, ...] + full_predictions[1::2, ...]

        if truncate_negative:
            zero = torch.zeros(1, device=mean_predictions.device, dtype=mean_predictions.dtype)
            clamped_mean = torch.maximum(mean_predictions, zero)
            clamped_full = torch.maximum(full_predictions, zero)
            should_clamp = (input_min >= 0).to(mean_predictions.device)
            mean_predictions = torch.where(should_clamp, clamped_mean, mean_predictions)
            full_predictions = torch.where(should_clamp, clamped_full, full_predictions)

        loss = None
        if future_values is not None:
            mse_loss = F.mse_loss(mean_predictions, future_values)
            quantile_indices = [i for i in range(full_predictions.shape[-1]) if i != decode_index]
            if quantile_indices:
                index_tensor = torch.tensor(quantile_indices, device=full_predictions.device, dtype=torch.long)
                quantile_tensor = torch.index_select(full_predictions, dim=-1, index=index_tensor)
                quantile_loss = self._quantile_loss(quantile_tensor, future_values)
                loss = mse_loss + quantile_loss
            else:
                loss = mse_loss

        return TimesFm2_5OutputForPrediction(
            last_hidden_state=model_outputs.last_hidden_state,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
            mean_predictions=mean_predictions,
            full_predictions=full_predictions,
            loss=loss,
        )


__all__ = [
    "TimesFm2_5Config",
    "TimesFm2_5ModelForPrediction",
    "TimesFm2_5PreTrainedModel",
    "TimesFm2_5Model",
]
