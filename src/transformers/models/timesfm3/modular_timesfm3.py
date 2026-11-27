# Copyright 2026 Google LLC and The HuggingFace Team. All rights reserved.
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
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...masking_utils import create_bidirectional_mask, create_causal_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward
from ..timesfm2_5.configuration_timesfm2_5 import TimesFm2_5Config
from ..timesfm2_5.modeling_timesfm2_5 import (
    TimesFm2_5Attention,
    TimesFm2_5DecoderLayer,
    TimesFm2_5MLP,
    TimesFm2_5PreTrainedModel,
    TimesFm2_5ResidualBlock,
    TimesFm2_5RMSNorm,
    TimesFm2_5RotaryEmbedding,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/timesfm-3.0-pytorch")
@strict
class TimesFm3Config(TimesFm2_5Config):
    r"""
    patch_length (`int`, *optional*, defaults to 32):
        The length of one patch in the input sequence.
    context_length (`int`, *optional*, defaults to 15360):
        The maximum number of context time steps the model reads. Longer inputs keep their most recent steps.
    horizon_length (`int`, *optional*, defaults to 64):
        The number of future time steps predicted from each patch. It must be a multiple of `patch_length` and
        larger than it. Forecasts of any length are assembled from the predictions of several patches.
    quantiles (`list[float]`, *optional*, defaults to `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`):
        The quantiles to predict.
    decode_index (`int`, *optional*, defaults to 4):
        Index into the quantile dimension used to extract the point (median) forecast.
    use_bias (`bool`, *optional*, defaults to `False`):
        Whether to use bias in MLP and transformer linear layers.
    force_flip_invariance (`bool`, *optional*, defaults to `False`):
        Whether to also forecast the negated inputs and average the two forecasts.
    infer_is_positive (`bool`, *optional*, defaults to `False`):
        Whether to clamp to non-negative values the forecasts of the series that have no negative value.
    value_clip (`float`, *optional*, defaults to 1e+20):
        Inputs and forecasts are clipped to `[-value_clip, value_clip]`.
    linear_detrending_threshold (`float`, *optional*, defaults to 0.5):
        A linear trend is removed from a series, and added back to its forecast, when this brings the standard
        deviation of the series below `linear_detrending_threshold` times its original value.

    Example:

    ```python
    >>> from transformers import TimesFm3Config, TimesFm3ModelForPrediction

    >>> configuration = TimesFm3Config()
    >>> model = TimesFm3ModelForPrediction(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "timesfm3"

    context_length: int = 15360
    horizon_length: int = 64
    # `torch.nn.RMSNorm`, which the original implementation uses, defaults to the machine epsilon of float32.
    rms_norm_eps: float = 1.1920929e-07
    decode_index: int = 4
    activation: str = "relu"
    force_flip_invariance: bool = False
    infer_is_positive: bool = False
    value_clip: float = 1e20
    linear_detrending_threshold: float = 0.5

    output_quantile_len = AttributeError()
    use_continuous_quantile_head = AttributeError()

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.horizon_length % self.patch_length != 0 or self.horizon_length <= self.patch_length:
            raise ValueError(
                f"`horizon_length` ({self.horizon_length}) must be a multiple of `patch_length` "
                f"({self.patch_length}) and larger than it."
            )
        if not 0 <= self.decode_index < len(self.quantiles):
            raise ValueError(
                f"`decode_index` ({self.decode_index}) must index one of the {len(self.quantiles)} quantiles."
            )


@auto_docstring
@dataclass
class TimesFm3Output(BaseModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_variates, num_patches, hidden_size)`):
        Hidden states at the output of the last layer of the model.
    context_count (`torch.Tensor` of shape `(batch_size, num_variates, num_patches)`):
        Number of observed values of each series up to and including each patch.
    context_mu (`torch.Tensor` of shape `(batch_size, num_variates, num_patches)`):
        Running means of each series, computed over the observed values up to and including each patch.
    context_sigma (`torch.Tensor` of shape `(batch_size, num_variates, num_patches)`):
        Running standard deviations of each series, computed like `context_mu`.
    """

    context_count: torch.Tensor | None = None
    context_mu: torch.Tensor | None = None
    context_sigma: torch.Tensor | None = None


@auto_docstring
@dataclass
class TimesFm3OutputForPrediction(BaseModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_variates, num_patches, hidden_size)`):
        Hidden states at the output of the last layer of the model. The variates are the targets followed by the
        covariates, and the patches cover the context followed by the horizon.
    mean_predictions (`torch.Tensor` of shape `(batch_size, num_targets, prediction_length)`):
        Point forecasts of the target series: the predicted median.
    full_predictions (`torch.Tensor` of shape `(batch_size, num_targets, prediction_length, num_quantiles)`):
        Quantile forecasts of the target series, in the order of `config.quantiles`.
    """

    mean_predictions: torch.Tensor | None = None
    full_predictions: torch.Tensor | None = None


def _update_running_statistics(
    count: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, values: torch.Tensor, padding: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Updates the count, mean and standard deviation of each series, of shape `(batch_size, num_variates)`, with one
    more patch of `values`, of shape `(batch_size, num_variates, patch_length)`. `padding` marks the values to skip.
    """
    patch_count = (~padding).to(values.dtype).sum(dim=-1)
    safe_patch_count = patch_count.clamp_min(1.0)
    values = values.masked_fill(padding, 0.0)
    patch_sum = values.sum(dim=-1)
    patch_mean = patch_sum / safe_patch_count
    patch_variance = (values - patch_mean.unsqueeze(-1)).square().masked_fill(padding, 0.0).sum(dim=-1)
    patch_variance = patch_variance / safe_patch_count

    new_count = count + patch_count
    safe_new_count = new_count.clamp_min(1.0)
    new_mean = (count * mean + patch_sum) / safe_new_count
    new_std = torch.sqrt(
        (
            count * std.square()
            + patch_count * patch_variance
            + count * (mean - new_mean).square()
            + patch_count * (patch_mean - new_mean).square()
        )
        / safe_new_count
    )
    return new_count, new_mean, new_std


def _revin(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, reverse: bool = False) -> torch.Tensor:
    """
    Reversible instance normalization of `values` with statistics that have one or two fewer trailing dimensions.
    A standard deviation close to zero, as for a constant series, leaves the scale of the values unchanged.
    """
    while mean.ndim < values.ndim:
        mean, std = mean.unsqueeze(-1), std.unsqueeze(-1)
    std = torch.where(std < 1e-6, 1.0, std)
    if reverse:
        return values * std + mean
    return (values - mean) / std


def _interpolate_missing(values: torch.Tensor) -> torch.Tensor:
    """
    Fills the NaNs of each series, along the last dimension: linearly between the two surrounding observations,
    with the nearest observation before the first and after the last one, and with zeros without any observation.
    """
    observed = ~torch.isnan(values)
    if observed.all():
        return values

    length = values.shape[-1]
    steps = torch.arange(length, device=values.device)
    previous = torch.where(observed, steps, -1).cummax(dim=-1).values
    following = torch.where(observed, steps, length).flip(-1).cummin(dim=-1).values.flip(-1)
    has_previous, has_following = previous >= 0, following < length

    values = torch.where(observed, values, 0.0)
    previous_values = values.gather(-1, previous.clamp(min=0))
    following_values = values.gather(-1, following.clamp(max=length - 1))
    slope = (following_values - previous_values) / (following - previous).clamp(min=1)
    filled = previous_values + slope * (steps - previous)
    filled = torch.where(has_previous & has_following, filled, torch.where(has_previous, previous_values, 0.0))
    filled = torch.where(~has_previous & has_following, following_values, filled)
    return torch.where(observed, values, filled)


def _linear_trend(
    values: torch.Tensor, padding: torch.Tensor, steps: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Least-squares line of each series over its observed values, as a function of `steps`.

    Returns the slope and the intercept of the line, and whether to remove it: it is removed when this brings the
    standard deviation of the series below `threshold` times its original value. The three tensors have shape
    `(batch_size, num_variates, 1)`.
    """
    observed = ~padding
    count = observed.to(values.dtype).sum(dim=-1, keepdim=True)
    safe_count = count.clamp_min(1.0)
    sum_steps = torch.where(observed, steps, 0.0).sum(dim=-1, keepdim=True)
    sum_squared_steps = torch.where(observed, steps**2, 0.0).sum(dim=-1, keepdim=True)
    sum_values = torch.where(observed, values, 0.0).sum(dim=-1, keepdim=True)
    sum_products = torch.where(observed, steps * values, 0.0).sum(dim=-1, keepdim=True)

    determinant = count * sum_squared_steps - sum_steps**2
    is_singular = determinant == 0.0
    safe_determinant = torch.where(is_singular, 1.0, determinant)
    slope = torch.where(is_singular, 0.0, (count * sum_products - sum_steps * sum_values) / safe_determinant)
    intercept = torch.where(
        is_singular,
        torch.where(count > 0, sum_values / safe_count, 0.0),
        (sum_values - slope * sum_steps) / safe_count,
    )
    detrended = values - (slope * steps + intercept)

    def _std(series: torch.Tensor) -> torch.Tensor:
        mean = torch.where(observed, series, 0.0).sum(dim=-1, keepdim=True) / safe_count
        mean_of_squares = torch.where(observed, series**2, 0.0).sum(dim=-1, keepdim=True) / safe_count
        return torch.sqrt((mean_of_squares - mean**2).clamp_min(0.0))

    return slope, intercept, _std(detrended) < threshold * _std(values)


def _stitch_patches(patch_predictions: torch.Tensor, patch_length: int) -> torch.Tensor:
    """
    Joins the predictions of consecutive patches into one forecast.

    Each patch predicts `patch_length + overlap` steps, the last `overlap` of which are also the first steps
    predicted by the next patch. Over these shared steps, the forecast moves linearly from the prediction of the
    earlier patch to the prediction of the later one.

    Args:
        patch_predictions (`torch.Tensor` of shape `(batch_size, num_variates, num_patches, patch_length + overlap, num_quantiles)`):
            Predictions of each patch.
        patch_length (`int`):
            Number of steps between the starts of two consecutive predictions.

    Returns:
        `torch.Tensor` of shape `(batch_size, num_variates, num_patches * patch_length + overlap, num_quantiles)`.
    """
    num_patches, overlap = patch_predictions.shape[2], patch_predictions.shape[3] - patch_length
    if num_patches == 1:
        return patch_predictions[:, :, 0]

    weights = torch.linspace(1.0, 0.0, overlap, device=patch_predictions.device, dtype=patch_predictions.dtype)
    weights = weights[:, None]
    earlier = patch_predictions[:, :, :-1, patch_length:]
    later = patch_predictions[:, :, 1:, :overlap]
    stitched = patch_predictions[:, :, :, :patch_length].clone()
    stitched[:, :, 1:, :overlap] = weights * earlier + (1.0 - weights) * later
    return torch.cat([stitched.flatten(2, 3), patch_predictions[:, :, -1, patch_length:]], dim=2)


class TimesFm3MLP(TimesFm2_5MLP):
    pass


class TimesFm3ResidualBlock(TimesFm2_5ResidualBlock):
    pass


class TimesFm3RMSNorm(TimesFm2_5RMSNorm):
    pass


class TimesFm3RotaryEmbedding(TimesFm2_5RotaryEmbedding):
    pass


class TimesFm3Attention(TimesFm2_5Attention):
    """
    TimesFM 3 attention over one axis of the grid of patches: causal with rotary positions over time, bidirectional
    and without positions over the variates.
    """

    def __init__(self, config: TimesFm3Config, layer_idx: int, is_causal: bool = True):
        super().__init__(config, layer_idx)
        self.is_causal = is_causal

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # The learnable per-dimension scale replaces the usual division by `sqrt(head_dim)`.
        scale = F.softplus(self.scaling).mul(1.442695041)
        query_states = query_states * scale[None, None, None, :]

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


class TimesFm3Layer(TimesFm2_5DecoderLayer):
    """
    TimesFM 3 layer: attention over the patches of each series, then attention over the series at each patch
    position, then a feed-forward block. Each of the three is wrapped in a pre- and a post-normalization.
    """

    def __init__(self, config: TimesFm3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.variate_attn = TimesFm3Attention(config=config, layer_idx=layer_idx, is_causal=False)
        self.pre_variate_layernorm = TimesFm3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_variate_layernorm = TimesFm3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        variate_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        batch_size, num_variates, num_patches, hidden_size = hidden_states.shape

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states.reshape(batch_size * num_variates, num_patches, hidden_size)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states.view(batch_size, num_variates, num_patches, hidden_size)
        hidden_states = self.post_attention_layernorm(hidden_states) + residual

        residual = hidden_states
        hidden_states = self.pre_variate_layernorm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size * num_patches, num_variates, hidden_size)
        hidden_states, _ = self.variate_attn(
            hidden_states=hidden_states,
            attention_mask=variate_attention_mask,
            **kwargs,
        )
        hidden_states = hidden_states.view(batch_size, num_patches, num_variates, hidden_size).transpose(1, 2)
        hidden_states = self.post_variate_layernorm(hidden_states) + residual

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states) + residual

        return hidden_states


@auto_docstring
class TimesFm3PreTrainedModel(TimesFm2_5PreTrainedModel):
    config: TimesFm3Config
    base_model_prefix = "model"
    _no_split_modules = ["TimesFm3Layer"]
    _supports_flash_attn = False
    _supports_flex_attn = False
    _can_record_outputs = {
        "hidden_states": TimesFm3Layer,
        "attentions": TimesFm3Attention,
    }

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, TimesFm3Attention):
            init.ones_(module.scaling)


@auto_docstring
class TimesFm3Model(TimesFm3PreTrainedModel):
    def __init__(self, config: TimesFm3Config):
        super().__init__(config)

        # A token holds the values of its patch, the values of the `horizon_length` steps that follow it when they
        # are known, and the padding of both.
        self.input_ff_layer = TimesFm3ResidualBlock(
            config,
            input_dims=2 * (config.patch_length + config.horizon_length),
            hidden_dims=config.hidden_size,
            output_dims=config.hidden_size,
            use_bias=config.use_bias,
        )
        self.layers = nn.ModuleList(
            [TimesFm3Layer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = TimesFm3RotaryEmbedding(config)
        self.gradient_checkpointing = False

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        past_values: torch.Tensor,
        past_values_padding: torch.Tensor | None = None,
        future_known_mask: torch.BoolTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TimesFm3Output:
        r"""
        past_values (`torch.Tensor` of shape `(batch_size, num_variates, sequence_length)`):
            Values of the series: the context, followed by the horizon when there is one to predict.
            `sequence_length` must be a multiple of `config.patch_length`.
        past_values_padding (`torch.Tensor` of shape `(batch_size, num_variates, sequence_length)`, *optional*):
            Padding mask for the input. `1` indicates padded (masked) time steps, `0` indicates valid values.
            The values to predict are masked. Every value is valid by default.
        future_known_mask (`torch.BoolTensor` of shape `(batch_size, num_variates)`, *optional*):
            Marks the series whose values are known in advance, like a calendar of promotions. Each patch of
            these series also carries the values of the `config.horizon_length` steps that follow it. No series
            is known in advance by default.
        """
        patch_length = self.config.patch_length
        batch_size, num_variates, sequence_length = past_values.shape
        if sequence_length == 0 or sequence_length % patch_length != 0:
            raise ValueError(
                f"The length of `past_values` ({sequence_length}) must be a positive multiple of the patch "
                f"length ({patch_length})."
            )
        num_patches = sequence_length // patch_length
        num_future_patches = self.config.horizon_length // patch_length

        if past_values_padding is None:
            past_values_padding = torch.zeros_like(past_values, dtype=torch.bool)
        if future_known_mask is None:
            future_known_mask = torch.zeros((batch_size, num_variates), dtype=torch.bool, device=past_values.device)
        padding = past_values_padding.to(torch.bool).reshape(batch_size, num_variates, num_patches, patch_length)

        values = torch.nan_to_num(past_values, nan=0.0)
        value_clip = min(self.config.value_clip, torch.finfo(values.dtype).max)
        values = values.clamp(-value_clip, value_clip)
        values = values.reshape(batch_size, num_variates, num_patches, patch_length)

        # Each patch is normalized with the statistics of its series up to and including the patch.
        count = values.new_zeros((batch_size, num_variates))
        mean, std = count, count
        counts, means, stds = [], [], []
        for patch_index in range(num_patches):
            count, mean, std = _update_running_statistics(
                count, mean, std, values[:, :, patch_index], padding[:, :, patch_index]
            )
            counts.append(count)
            means.append(mean)
            stds.append(std)
        context_count = torch.stack(counts, dim=2)
        context_mu = torch.stack(means, dim=2)
        context_sigma = torch.stack(stds, dim=2)

        patches = _revin(values, context_mu, context_sigma).masked_fill(padding, 0.0)

        # The patches that follow each patch, masked past the end of the sequence and for the series that are not
        # known in advance.
        future_index = torch.arange(num_patches, device=values.device)[:, None] + torch.arange(
            1, num_future_patches + 1, device=values.device
        )
        past_the_end = (future_index >= num_patches).repeat_interleave(patch_length, dim=-1)
        future_index = future_index.clamp(max=num_patches - 1)
        future_padding = padding[:, :, future_index].flatten(3)
        future_padding = future_padding | past_the_end | ~future_known_mask[:, :, None, None]
        future_patches = _revin(values[:, :, future_index].flatten(3), context_mu, context_sigma)
        future_patches = future_patches.masked_fill(future_padding, 0.0)

        tokens = torch.cat(
            [patches, future_patches, padding.to(patches.dtype), future_padding.to(patches.dtype)], dim=-1
        )
        hidden_states = self.input_ff_layer(tokens)

        # Only the fully padded patches at the start of a series are hidden from the attention: the patches of
        # the horizon are fully padded as well, and are the ones to predict.
        is_padded_patch = padding.all(dim=-1) & future_padding.all(dim=-1)
        is_visible = ~is_padded_patch.cumprod(dim=-1).to(torch.bool)

        # Only the shape and the dtype of the embeddings are used to create the masks.
        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_states.view(batch_size * num_variates, num_patches, -1),
            attention_mask=is_visible.reshape(batch_size * num_variates, num_patches),
            cache=None,
        )
        variate_attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states.view(batch_size * num_patches, num_variates, -1),
            attention_mask=is_visible.transpose(1, 2).reshape(batch_size * num_patches, num_variates),
        )
        position_ids = torch.arange(num_patches, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                variate_attention_mask=variate_attention_mask,
                **kwargs,
            )

        return TimesFm3Output(
            last_hidden_state=hidden_states,
            context_count=context_count,
            context_mu=context_mu,
            context_sigma=context_sigma,
        )


@auto_docstring(
    custom_intro="""
    TimesFM 3 model with a quantile head on top, to forecast one or several target series, with or without
    covariates, in a single forward pass.
    """
)
class TimesFm3ModelForPrediction(TimesFm3PreTrainedModel):
    def __init__(self, config: TimesFm3Config):
        super().__init__(config)

        self.model = TimesFm3Model(config)
        self.output_projection = nn.Linear(config.hidden_size, config.horizon_length * len(config.quantiles))

        self.post_init()

    def _refine_statistics(
        self, logits: torch.Tensor, count: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Statistics used to denormalize the predictions of the last patch of the context and of the patches of the
        horizon.

        The statistics of a series stop at the end of its context. For the patches of the horizon, they are
        extended with the median predicted for the patches before them: the model predicts `horizon_length` steps
        from the last patch of the context, the statistics grow with these steps one patch at a time, and once
        they are used up, the patch reached predicts the next `horizon_length` steps.

        Args:
            logits (`torch.Tensor` of shape `(batch_size, num_variates, 1 + num_horizon_patches, horizon_length * num_quantiles)`):
                Normalized predictions of the last patch of the context and of the patches of the horizon.
            count, mean, std (`torch.Tensor` of shape `(batch_size, num_variates)`):
                Statistics of each series at the last patch of its context.

        Returns:
            The means and the standard deviations, of shape `(batch_size, num_variates, 1 + num_horizon_patches)`.
        """
        patch_length = self.config.patch_length
        num_future_patches = self.config.horizon_length // patch_length
        num_quantiles = len(self.config.quantiles)
        value_clip = min(self.config.value_clip, torch.finfo(logits.dtype).max)
        medians = logits.unflatten(-1, (num_future_patches, patch_length, num_quantiles))[..., num_quantiles // 2]

        means, stds = [mean], [std]
        predicted = _revin(medians[:, :, 0], mean, std, reverse=True).clamp(-value_clip, value_clip)
        offset = 0
        no_padding = torch.zeros_like(predicted[:, :, 0], dtype=torch.bool)
        for patch_index in range(1, logits.shape[2]):
            count, mean, std = _update_running_statistics(count, mean, std, predicted[:, :, offset], no_padding)
            means.append(mean)
            stds.append(std)
            offset = (offset + 1) % num_future_patches
            if offset == 0:
                predicted = _revin(medians[:, :, patch_index], mean, std, reverse=True)
                predicted = predicted.clamp(-value_clip, value_clip)
        return torch.stack(means, dim=2), torch.stack(stds, dim=2)

    def _forecast(
        self,
        past_values: torch.Tensor,
        past_values_padding: torch.Tensor,
        prediction_length: int,
        past_covariates: torch.Tensor | None = None,
        known_covariates: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, TimesFm3Output]:
        """
        Forecasts `prediction_length` steps of the targets in one pass over the context followed by the horizon.

        Args:
            past_values (`torch.Tensor` of shape `(batch_size, num_targets, context_length)`):
                Context of the target series, without NaN.
            past_values_padding (`torch.BoolTensor` of shape `(batch_size, context_length)`):
                Marks the time steps to ignore for every series.
            prediction_length (`int`):
                Number of time steps to forecast.
            past_covariates (`torch.Tensor` of shape `(batch_size, num_past_covariates, context_length)`, *optional*):
                Covariates over the context, without NaN.
            known_covariates (`torch.Tensor` of shape `(batch_size, num_known_covariates, context_length + prediction_length)`, *optional*):
                Covariates over the context and the horizon, without NaN.

        Returns:
            The sorted quantile forecasts of the targets, of shape `(batch_size, num_targets, prediction_length,
            num_quantiles)`, and the outputs of the base model.
        """
        patch_length, horizon_length = self.config.patch_length, self.config.horizon_length
        value_clip = min(self.config.value_clip, torch.finfo(past_values.dtype).max)
        batch_size, num_targets, context_length = past_values.shape

        # As in the original implementation, the horizon is decoded exactly with known covariates, since they stop
        # there, and rounded up to a multiple of `horizon_length` without them.
        decoded_length = prediction_length
        if known_covariates is None:
            decoded_length = math.ceil(prediction_length / horizon_length) * horizon_length

        # The series are the targets, then the covariates known over the context, then those known in advance.
        series = [past_values]
        if past_covariates is not None:
            series.append(past_covariates)
        num_unknown = sum(variates.shape[1] for variates in series)
        if known_covariates is not None:
            series.append(known_covariates[..., :context_length])
        values = torch.cat(series, dim=1)
        num_variates = values.shape[1]
        padding = past_values_padding[:, None, :].expand(batch_size, num_variates, context_length)

        # Pad the context on the left to a whole number of patches.
        num_padded = -context_length % patch_length
        values = F.pad(values, (num_padded, 0))
        padding = F.pad(padding, (num_padded, 0), value=True)
        padded_context_length = context_length + num_padded
        num_context_patches = padded_context_length // patch_length

        # Remove the linear trend of the series it explains well. Time ends at 0 on the last step of the context.
        steps = torch.arange(-(padded_context_length - 1), 1, device=values.device, dtype=values.dtype)
        steps = steps / padded_context_length
        slope, intercept, is_detrended = _linear_trend(values, padding, steps, self.config.linear_detrending_threshold)
        values = torch.where(is_detrended, values - (slope * steps + intercept), values)
        values = values.masked_fill(padding, 0.0)
        horizon_steps = torch.arange(1, decoded_length + 1, device=values.device, dtype=values.dtype)
        horizon_steps = horizon_steps / padded_context_length
        horizon_trend = torch.where(is_detrended, slope * horizon_steps + intercept, 0.0)

        # A patch predicts `patch_length + overlap` steps, and the predictions of consecutive patches are stitched
        # over their overlap. The horizon is padded so that each of these patches reads the patches it predicts.
        overlap = min(2 * patch_length, horizon_length) - patch_length
        num_forecast_patches = max(math.ceil((decoded_length - overlap) / patch_length), 1)
        num_horizon_patches = num_forecast_patches + horizon_length // patch_length - 1
        horizon_values = values.new_zeros((batch_size, num_variates, num_horizon_patches * patch_length))
        horizon_padding = torch.ones_like(horizon_values, dtype=torch.bool)
        if known_covariates is not None:
            future_values = known_covariates[..., context_length:] - horizon_trend[:, num_unknown:]
            horizon_values[:, num_unknown:, :decoded_length] = future_values
            horizon_padding[:, num_unknown:, :decoded_length] = False
        future_known_mask = torch.zeros((batch_size, num_variates), dtype=torch.bool, device=values.device)
        future_known_mask[:, num_unknown:] = True

        outputs: TimesFm3Output = self.model(
            past_values=torch.cat([values, horizon_values], dim=-1),
            past_values_padding=torch.cat([padding, horizon_padding], dim=-1),
            future_known_mask=future_known_mask,
            **kwargs,
        )

        # The forecast starts with the prediction of the last patch of the context.
        last_context_patch = num_context_patches - 1
        hidden_states = outputs.last_hidden_state[:, :, last_context_patch:]
        logits = self.output_projection(hidden_states.to(self.output_projection.weight.device))
        mean, std = self._refine_statistics(
            logits,
            outputs.context_count[:, :, last_context_patch].to(logits.device),
            outputs.context_mu[:, :, last_context_patch].to(logits.device),
            outputs.context_sigma[:, :, last_context_patch].to(logits.device),
        )
        predictions = _revin(logits, mean, std, reverse=True).clamp(-value_clip, value_clip)
        predictions = predictions.unflatten(-1, (horizon_length, len(self.config.quantiles)))

        predictions = predictions[:, :, :num_forecast_patches, : patch_length + overlap]
        predictions = _stitch_patches(predictions, patch_length)[:, :, :decoded_length]
        predictions = predictions + horizon_trend.to(predictions.device).unsqueeze(-1)

        predictions = predictions[:, :num_targets, :prediction_length]
        return predictions.sort(dim=-1).values, outputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        past_values: torch.Tensor,
        prediction_length: int | None = None,
        past_covariates: torch.Tensor | None = None,
        known_covariates: torch.Tensor | None = None,
        truncate_negative: bool | None = None,
        force_flip_invariance: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TimesFm3OutputForPrediction:
        r"""
        past_values (`torch.Tensor` of shape `(batch_size, num_targets, sequence_length)`):
            Past values of the target series, which are forecast jointly. A tensor of shape `(batch_size,
            sequence_length)` or `(sequence_length,)` is read as a single target series. Missing values are NaN:
            the time steps before the first observation of a batch element are ignored, and the other missing
            values are interpolated linearly.
        prediction_length (`int`, *optional*):
            Number of time steps to forecast. Defaults to `config.horizon_length`.
        past_covariates (`torch.Tensor` of shape `(batch_size, num_past_covariates, sequence_length)`, *optional*):
            Covariates observed over the context only (`past_only_covariates` in the original implementation).
            Missing values are NaN.
        known_covariates (`torch.Tensor` of shape `(batch_size, num_known_covariates, sequence_length + prediction_length)`, *optional*):
            Covariates known over the context and the horizon (`past_future_covariates` in the original
            implementation). Missing values are NaN.
        truncate_negative (`bool`, *optional*):
            Whether to clamp to non-negative values the forecasts of the target series that have no negative
            value. If `None`, defaults to `config.infer_is_positive`.
        force_flip_invariance (`bool`, *optional*):
            Whether to also forecast the negated inputs and average the two forecasts. If `None`, defaults to
            `config.force_flip_invariance`.

        Example:

        ```python
        >>> import torch
        >>> from transformers import TimesFm3ModelForPrediction

        >>> model = TimesFm3ModelForPrediction.from_pretrained("google/timesfm-3.0-pytorch")

        >>> # two target series over 512 time steps, forecast jointly
        >>> past_values = torch.sin(torch.arange(512) / 10).repeat(1, 2, 1)
        >>> with torch.no_grad():
        ...     outputs = model(past_values=past_values, prediction_length=96)
        >>> outputs.mean_predictions.shape
        torch.Size([1, 2, 96])
        ```
        """
        if past_values.ndim == 1:
            past_values = past_values[None, None, :]
        elif past_values.ndim == 2:
            past_values = past_values[:, None, :]
        if past_values.ndim != 3:
            raise ValueError(
                f"`past_values` must have shape (batch_size, num_targets, sequence_length), "
                f"got {tuple(past_values.shape)}."
            )
        batch_size, _, sequence_length = past_values.shape
        if sequence_length == 0:
            raise ValueError("`past_values` must have at least one time step.")

        prediction_length = prediction_length if prediction_length is not None else self.config.horizon_length
        if prediction_length <= 0:
            raise ValueError(f"`prediction_length` must be positive, got {prediction_length}.")
        for name, covariates, expected_length in (
            ("past_covariates", past_covariates, sequence_length),
            ("known_covariates", known_covariates, sequence_length + prediction_length),
        ):
            if covariates is not None and covariates.shape[::2] != (batch_size, expected_length):
                raise ValueError(
                    f"`{name}` must have shape (batch_size, num_covariates, {expected_length}), "
                    f"got {tuple(covariates.shape)}."
                )
        if truncate_negative is None:
            truncate_negative = self.config.infer_is_positive
        if force_flip_invariance is None:
            force_flip_invariance = self.config.force_flip_invariance

        dtype = self.output_projection.weight.dtype
        past_values = past_values.to(dtype)
        observed = ~torch.isnan(past_values)
        is_nonnegative = observed.any(dim=-1) & ((past_values >= 0) | ~observed).all(dim=-1)

        # Ignore the time steps before the first observation of a target, for the covariates too, and fill the
        # other missing values.
        is_leading = observed.any(dim=1).cumsum(dim=-1) == 0
        is_leading = is_leading & ~is_leading.all(dim=-1, keepdim=True)
        past_values = _interpolate_missing(past_values)
        if past_covariates is not None:
            past_covariates = past_covariates.to(dtype).masked_fill(is_leading[:, None], float("nan"))
            past_covariates = _interpolate_missing(past_covariates)
        if known_covariates is not None:
            is_leading_known = F.pad(is_leading, (0, prediction_length))
            known_covariates = known_covariates.to(dtype).masked_fill(is_leading_known[:, None], float("nan"))
            known_covariates = _interpolate_missing(known_covariates)

        # Drop the time steps without any observation in the batch, then keep the most recent steps of the
        # context, up to a whole number of patches. The positions of the patches count from the first one kept.
        patch_length = self.config.patch_length
        context_length = sequence_length - int(is_leading.all(dim=0).sum())
        context_length = min(context_length, math.ceil(self.config.context_length / patch_length) * patch_length)
        past_values, is_leading = past_values[..., -context_length:], is_leading[..., -context_length:]
        if past_covariates is not None:
            past_covariates = past_covariates[..., -context_length:]
        if known_covariates is not None:
            known_covariates = known_covariates[..., -(context_length + prediction_length) :]

        predictions, outputs = self._forecast(
            past_values, is_leading, prediction_length, past_covariates, known_covariates, **kwargs
        )
        if force_flip_invariance:
            flipped_predictions, _ = self._forecast(
                -past_values,
                is_leading,
                prediction_length,
                None if past_covariates is None else -past_covariates,
                None if known_covariates is None else -known_covariates,
                **kwargs,
            )
            predictions = (predictions - flipped_predictions.flip(-1)) / 2
        if truncate_negative:
            is_nonnegative = is_nonnegative[:, :, None, None].to(predictions.device)
            predictions = torch.where(is_nonnegative, predictions.clamp(min=0.0), predictions)

        return TimesFm3OutputForPrediction(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mean_predictions=predictions[..., self.config.decode_index],
            full_predictions=predictions,
        )


__all__ = [
    "TimesFm3Config",
    "TimesFm3ModelForPrediction",
    "TimesFm3PreTrainedModel",
    "TimesFm3Model",
]
