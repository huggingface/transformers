# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""PyTorch Cisco Time Series Model (CTSM)."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...masking_utils import create_causal_mask
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..phi4_multimodal.modeling_phi4_multimodal import simple_eager_attention_forward
from ..timesfm.configuration_timesfm import TimesFmConfig
from ..timesfm.modeling_timesfm import (
    TimesFmAttention,
    TimesFmDecoderLayer,
    TimesFmModel,
    TimesFmModelForPrediction,
    TimesFmOutput,
    TimesFmOutputForPrediction,
    TimesFmPreTrainedModel,
    TimesFmResidualBlock,  # re-exported as CtsmResidualBlock in the generated file
)
from ..timesfm2_5.modeling_timesfm2_5 import (
    TimesFm2_5RotaryEmbedding,
    apply_rotary_pos_emb,
)


logger = logging.get_logger(__name__)

_STREAM_NORM_EPS = 1e-8


@auto_docstring(checkpoint="cisco-ai/cisco-time-series-model-1.0")
@strict
class CtsmConfig(TimesFmConfig):
    r"""
    patch_length (`int`, *optional*, defaults to 32):
        Length of one patch in the input sequence for each resolution stream.
    context_length (`int`, *optional*, defaults to 512):
        Length of the input context for each resolution stream.
    horizon_length (`int`, *optional*, defaults to 128):
        Length of the prediction horizon produced per autoregressive step.
    freq_size (`int`, *optional*, defaults to 3):
        Number of frequency embeddings.
    tolerance (`float`, *optional*, defaults to 1e-06):
        Numerical tolerance used in normalization.
    pad_val (`float`, *optional*, defaults to 1123581321.0):
        Sentinel value marking padded positions in the input series.
    use_positional_embedding (`bool`, *optional*, defaults to `False`):
        CTSM uses rotary position embeddings and does not add sinusoidal positional embeddings.
    num_hidden_layers (`int`, *optional*, defaults to 25):
        Number of decoder layers (CTSM 1.0 uses 25 instead of TimesFM 2.0's 50).
    quantiles (`list[float]`, *optional*, defaults to 15 values between 0.01 and 0.99):
        Quantile levels predicted by the model (CTSM widens TimesFM's 9 quantiles to 15).
    use_resolution_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to add a learned embedding per resolution bucket (coarse / special / fine).
    use_special_token (`bool`, *optional*, defaults to `True`):
        Whether to insert a learned special token between the coarse and fine streams.
    num_resolutions (`int`, *optional*, defaults to 3):
        Number of resolution embeddings (coarse, special token, fine).
    aggregation_factor (`int`, *optional*, defaults to 60):
        Aggregation factor between fine and coarse resolutions (e.g. 60 minutes -> 1 hour).
    max_position_embeddings (`int`, *optional*, defaults to 1025):
        Maximum number of patches in the concatenated `[coarse, special, fine]` sequence; bounds the RoPE table.
    rope_parameters (`dict`, *optional*):
        Rotary position embedding parameters. Defaults to `{"rope_type": "default", "rope_theta": 10000.0}`.

    Example:

    ```python
    >>> from transformers import CtsmConfig, CtsmModelForPrediction

    >>> configuration = CtsmConfig()
    >>> model = CtsmModelForPrediction(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "ctsm"

    num_hidden_layers: int = 25
    quantiles: list[float] | tuple[float, ...] = (
        0.01,
        0.05,
        0.1,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.75,
        0.8,
        0.9,
        0.95,
        0.99,
    )
    use_resolution_embeddings: bool = True
    use_special_token: bool = True
    num_resolutions: int = 3
    aggregation_factor: int = 60
    max_position_embeddings: int = 1025
    rope_parameters: RopeParameters | dict | None = None

    min_timescale = AttributeError()
    max_timescale = AttributeError()


@auto_docstring
@dataclass
class CtsmOutput(TimesFmOutput):
    r"""
    loc (`torch.Tensor` of shape `(batch_size,)`):
        Stream-level mean used to normalize the fine-resolution context, reused to rescale the final forecast.
    scale (`torch.Tensor` of shape `(batch_size,)`):
        Stream-level standard deviation of the fine-resolution context.
    loc_coarse (`torch.Tensor` of shape `(batch_size,)`):
        Stream-level mean used to normalize the coarse-resolution context.
    scale_coarse (`torch.Tensor` of shape `(batch_size,)`):
        Stream-level standard deviation of the coarse-resolution context.
    loc_fine_patch (`torch.Tensor` of shape `(batch_size,)`):
        Per-first-patch mean of the (already stream-normalized) fine context, used as the second
        denormalization stage when projecting the horizon head's output back to the original scale.
    scale_fine_patch (`torch.Tensor` of shape `(batch_size,)`):
        Per-first-patch standard deviation matching `loc_fine_patch`.
    num_coarse_patches (`int`):
        Number of patches (including the optional special token) preceding the fine-resolution block.
    num_fine_patches (`int`):
        Number of patches in the fine-resolution block of the concatenated sequence.
    """

    loc_coarse: torch.Tensor | None = None
    scale_coarse: torch.Tensor | None = None
    loc_fine_patch: torch.Tensor | None = None
    scale_fine_patch: torch.Tensor | None = None
    num_coarse_patches: int | None = None
    num_fine_patches: int | None = None


@auto_docstring
@dataclass
class CtsmOutputForPrediction(TimesFmOutputForPrediction):
    r"""
    mean_predictions (`torch.Tensor` of shape `(batch_size, horizon_length)`):
        Point forecasts over the fine-resolution horizon.
    full_predictions (`torch.Tensor` of shape `(batch_size, horizon_length, 1 + num_quantiles)`):
        Concatenation of the mean prediction and the quantile predictions along the last axis.
    loss (`torch.Tensor` of shape `(1,)`, *optional*, returned when `future_values` is provided):
        Training loss combining MSE of the mean forecast and quantile loss when fine-resolution targets are supplied.
    """

    pass


class CtsmResidualBlock(TimesFmResidualBlock):
    pass


class CtsmRotaryEmbedding(TimesFm2_5RotaryEmbedding):
    pass


class CtsmAttention(TimesFmAttention):
    """TimesFM 2.0 style attention with learnable per-dimension Q scaling and rotary position embeddings."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Reference applies per-dim Q scaling *before* RoPE. The two operations don't commute when the
        # learned scaling differs across the two halves of head_dim that RoPE pairs up, so order matters.
        query_states = self._scale_query(query_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, simple_eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            # `scaling=1.0` because per-dim Q scaling is applied above (before RoPE) via `_scale_query`.
            scaling=1.0,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class CtsmDecoderLayer(TimesFmDecoderLayer):
    """CTSM transformer block: attention with RoPE followed by TimesFM 2.0 MLP with padding masking."""

    def __init__(self, config: CtsmConfig, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        # TimesFM wires `TimesFmAttention`; CTSM needs RoPE-aware attention.
        self.self_attn = CtsmAttention(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        paddings: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP
        hidden_states = self.mlp(hidden_states, paddings=paddings)
        return hidden_states


@auto_docstring
class CtsmPreTrainedModel(TimesFmPreTrainedModel):
    config: CtsmConfig
    base_model_prefix = "model"
    _supports_flash_attn = False

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: str | None, is_init_check: bool = False, allow_all_kernels: bool = False
    ) -> str:
        if attn_implementation is None:
            on_accelerator = torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available())
            attn_implementation = "flex_attention" if on_accelerator else "sdpa"
        return PreTrainedModel._check_and_adjust_attn_implementation(
            self, attn_implementation, is_init_check, allow_all_kernels
        )

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, CtsmModel) and module.special_token is not None:
            init.normal_(module.special_token, mean=0.0, std=self.config.initializer_range)


class CtsmModel(TimesFmModel):
    r"""
    The multi-resolution CTSM encoder. The forward pass consumes two aligned streams (a coarse low-frequency
    context and a fine high-frequency context), concatenates them along the sequence dimension with an
    optional learned special token, and runs a stack of rotary-attention transformer layers. Attention is
    bidirectional within the coarse block and causal elsewhere.
    """

    def __init__(self, config: CtsmConfig):
        super().__init__(config)

        # The parent only instantiates `position_emb` when `use_positional_embedding=True`, which is
        # `False` by default on `CtsmConfig` because CTSM uses rotary position embeddings instead.
        self.rotary_emb = CtsmRotaryEmbedding(config)

        self.multi_resolution = (
            nn.Embedding(config.num_resolutions, config.hidden_size) if config.use_resolution_embeddings else None
        )
        self.special_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if config.use_special_token else None

        self.post_init()

    @staticmethod
    def _left_pad_to_patch_boundary(
        values: torch.Tensor, paddings: torch.Tensor, patch_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-pad ``values`` (and the matching ``paddings`` mask) so that the time dimension is a
        multiple of ``patch_length``. Padding values are zeros; padded positions are flagged with ``1``
        in ``paddings`` so downstream patchifying / normalization can ignore them."""
        remaining = values.shape[1] % patch_length
        if remaining == 0:
            return values, paddings
        pad_len = patch_length - remaining
        values_pad = torch.zeros((values.shape[0], pad_len), device=values.device, dtype=values.dtype)
        paddings_pad = torch.ones((paddings.shape[0], pad_len), device=paddings.device, dtype=paddings.dtype)
        return torch.cat([values_pad, values], dim=1), torch.cat([paddings_pad, paddings], dim=1)

    @staticmethod
    def _normalize_with_pad(
        context: torch.Tensor, padding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stream-level normalization that matches the original CTSM reference.

        Normalizes ``context`` using the mean and standard deviation computed over the
        non-padded positions (``padding == 0``) across the whole context, rather than
        TimesFM's per-first-patch statistics. The normalized tensor has padded positions
        zeroed out and is clamped to a safe range. Falls back to fp64 when the fp32 stats
        are non-finite.
        """
        valid = 1.0 - padding
        count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        mu = (context * valid).sum(dim=1, keepdim=True) / count

        seq_len_float = context.new_tensor(float(context.shape[1]))
        pad_bool = padding.to(dtype=torch.bool)
        filled = torch.where(pad_bool, mu, context)
        sigma = filled.std(dim=1, keepdim=True, unbiased=False) * torch.sqrt(seq_len_float / count)

        if not torch.isfinite(mu).all() or not torch.isfinite(sigma).all():
            context64 = context.to(dtype=torch.float64)
            valid64 = valid.to(dtype=torch.float64)
            count64 = valid64.sum(dim=1, keepdim=True).clamp_min(1.0)
            mu64 = (context64 * valid64).sum(dim=1, keepdim=True) / count64
            filled64 = torch.where(pad_bool, mu64, context64)
            sigma64 = filled64.std(dim=1, keepdim=True, unbiased=False) * torch.sqrt(
                seq_len_float.to(torch.float64) / count64
            )
            mu = mu64.to(dtype=context.dtype)
            sigma = sigma64.to(dtype=context.dtype)

        sigma = sigma.clamp_min(1e-2)

        normalized = (context - mu) / (sigma + _STREAM_NORM_EPS)
        normalized = normalized * valid
        normalized = normalized.clamp(-1000.0, 1000.0)
        return normalized, mu.squeeze(-1), sigma.squeeze(-1)

    def _patchify(
        self, past_values: torch.Tensor, past_values_padding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Patchify a (stream-normalized) stream, apply TimesFM's per-first-patch normalization,
        and project through the input tokenizer. The per-first-patch ``(mu, sigma)`` are returned
        so the horizon head's output can be denormalized through both levels later."""
        bsz = past_values.shape[0]
        patched_inputs = past_values.view(bsz, -1, self.config.patch_length)
        patched_pads = past_values_padding.view(bsz, -1, self.config.patch_length)

        # CTSM's stream-normalized input is the analogue of TimesFM's raw input here, so we reuse the
        # parent's per-first-patch transform to match the reference's two-stage normalization.
        patched_inputs = patched_inputs * (1.0 - patched_pads)
        patched_inputs, patch_stats = self._forward_transform(patched_inputs, patched_pads)
        patched_inputs = patched_inputs * (1.0 - patched_pads)
        concat_inputs = torch.cat([patched_inputs, patched_pads], dim=-1)
        embeddings = self.input_ff_layer(concat_inputs)
        patch_padding = torch.min(patched_pads, dim=-1)[0]
        return embeddings, patch_padding, patch_stats

    @staticmethod
    def _block_sequence_ids_for_full_forward(
        bsz: int, num_coarse_patches: int, num_special: int, num_fine_patches: int, device: torch.device
    ) -> torch.Tensor:
        """Block ids for `create_causal_mask`: coarse positions share group ``0`` (bidirectional among themselves),
        the special token and fine positions are ``-1`` (strictly causal)."""
        block_ids = torch.cat(
            [
                torch.zeros(num_coarse_patches, dtype=torch.long, device=device),
                torch.full((num_special + num_fine_patches,), -1, dtype=torch.long, device=device),
            ]
        )
        return block_ids.unsqueeze(0).expand(bsz, -1)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        past_values_coarse: torch.Tensor | None = None,
        past_values_fine: torch.Tensor | None = None,
        past_values_coarse_padding: torch.LongTensor | None = None,
        past_values_fine_padding: torch.LongTensor | None = None,
        freq: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CtsmOutput:
        r"""
        past_values_coarse (`torch.FloatTensor` of shape `(batch_size, coarse_length)`):
            Coarse-resolution context (e.g. hourly aggregates). Length must be a multiple of `patch_length`
            or will be left-padded to one.
        past_values_fine (`torch.FloatTensor` of shape `(batch_size, fine_length)`):
            Fine-resolution context (e.g. minute-level).
        past_values_coarse_padding (`torch.LongTensor`, *optional*):
            Padding mask for the coarse stream, `1.0` for padded positions and `0.0` for real values.
        past_values_fine_padding (`torch.LongTensor`, *optional*):
            Padding mask for the fine stream.
        freq (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*):
            Frequency indices. Defaults to all zeros.
        """
        if past_values_coarse_padding is None:
            past_values_coarse_padding = torch.zeros_like(past_values_coarse)
        if past_values_fine_padding is None:
            past_values_fine_padding = torch.zeros_like(past_values_fine)
        past_values_coarse_padding = past_values_coarse_padding.to(past_values_coarse.dtype)
        past_values_fine_padding = past_values_fine_padding.to(past_values_fine.dtype)

        patch_length = self.config.patch_length
        past_values_coarse, past_values_coarse_padding = self._left_pad_to_patch_boundary(
            past_values_coarse, past_values_coarse_padding, patch_length
        )
        past_values_fine, past_values_fine_padding = self._left_pad_to_patch_boundary(
            past_values_fine, past_values_fine_padding, patch_length
        )

        coarse_normalized, loc_coarse, scale_coarse = self._normalize_with_pad(
            past_values_coarse, past_values_coarse_padding
        )
        fine_normalized, loc_fine, scale_fine = self._normalize_with_pad(past_values_fine, past_values_fine_padding)

        coarse_embeddings, coarse_patch_padding, _ = self._patchify(coarse_normalized, past_values_coarse_padding)
        fine_embeddings, fine_patch_padding, (fine_patch_mu, fine_patch_sigma) = self._patchify(
            fine_normalized, past_values_fine_padding
        )

        bsz, num_coarse_patches, hidden_size = coarse_embeddings.shape
        num_fine_patches = fine_embeddings.shape[1]
        device = coarse_embeddings.device
        dtype = coarse_embeddings.dtype

        if self.config.use_special_token:
            special = self.special_token.to(device=device, dtype=dtype).expand(bsz, 1, hidden_size)
            special_padding = torch.zeros(bsz, 1, device=device, dtype=coarse_patch_padding.dtype)
            model_input = torch.cat([coarse_embeddings, special, fine_embeddings], dim=1)
            patch_padding = torch.cat([coarse_patch_padding, special_padding, fine_patch_padding], dim=1)
            num_special = 1
        else:
            model_input = torch.cat([coarse_embeddings, fine_embeddings], dim=1)
            patch_padding = torch.cat([coarse_patch_padding, fine_patch_padding], dim=1)
            num_special = 0

        if self.config.use_resolution_embeddings:
            mr_coarse = torch.zeros(num_coarse_patches, dtype=torch.long, device=device)
            mr_special = torch.full((num_special,), 1, dtype=torch.long, device=device)
            mr_fine = torch.full((num_fine_patches,), 2, dtype=torch.long, device=device)
            mr_idx = torch.cat([mr_coarse, mr_special, mr_fine], dim=0).unsqueeze(0).expand(bsz, -1)
            model_input = model_input + self.multi_resolution(mr_idx)

        if freq is None:
            freq = torch.zeros((bsz, 1), dtype=torch.long, device=device)
        else:
            freq = freq.to(device=device, dtype=torch.long)
        model_input = model_input + self.freq_emb(freq)

        position_ids = torch.arange(model_input.shape[1], device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        position_embeddings = self.rotary_emb(model_input, position_ids)

        # Causal everywhere, with a bidirectional carve-out for the coarse block. The 2D `attention_mask`
        # follows the HF convention (1 = real, 0 = padded) and is the inverse of our internal `patch_padding`.
        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=model_input,
            attention_mask=(patch_padding == 0).to(torch.long),
            past_key_values=None,
            position_ids=position_ids,
            block_sequence_ids=self._block_sequence_ids_for_full_forward(
                bsz, num_coarse_patches, num_special, num_fine_patches, device
            ),
        )

        hidden_states = model_input
        for layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                paddings=patch_padding,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return CtsmOutput(
            last_hidden_state=hidden_states,
            loc=loc_fine,
            scale=scale_fine,
            loc_coarse=loc_coarse,
            scale_coarse=scale_coarse,
            loc_fine_patch=fine_patch_mu,
            scale_fine_patch=fine_patch_sigma,
            num_coarse_patches=num_coarse_patches + num_special,
            num_fine_patches=num_fine_patches,
        )


class CtsmModelForPrediction(TimesFmModelForPrediction):
    """CTSM model with a multi-resolution prediction head and autoregressive multi-resolution decoding.

    For horizons that require autoregressive decoding (``horizon_len > config.horizon_length``), the
    full multi-resolution context is rebuilt at every step: predictions are appended to the raw fine
    context, new coarse patches are aggregated every ``aggregation_factor`` fine values, and stream +
    per-first-patch normalization statistics are recomputed on the current window. There is no KV
    cache — the reference implementation also full-recomputes every step because the input
    normalization is global over the current window, which makes any cached K/V from a prior step
    stale by construction.
    """

    def __init__(self, config: CtsmConfig):
        super().__init__(config)
        del self.decoder
        del self.horizon_ff_layer

        self.model = CtsmModel(config)
        num_outputs = 1 + len(config.quantiles)
        self.horizon_ff_layer = CtsmResidualBlock(
            input_dims=config.hidden_size,
            output_dims=config.horizon_length * num_outputs,
            hidden_dims=config.intermediate_size,
        )
        self.post_init()

    @staticmethod
    def _build_multi_resolution(
        series: torch.Tensor, aggregation_factor: int, coarse_len: int, fine_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (coarse, fine) contexts from a 1-D fine-resolution series.

        Coarse is the mean of the last `coarse_len * aggregation_factor` fine samples, aligned to block boundaries.
        Fine is the last `fine_len` samples.
        """
        series = series.to(torch.float32).reshape(-1)
        needed = coarse_len * aggregation_factor
        raw = series[-needed:]
        remainder = raw.shape[0] % aggregation_factor
        if remainder:
            raw = raw[remainder:]
        if raw.numel() == 0:
            coarse = series.new_empty((0,), dtype=torch.float32)
        else:
            coarse = raw.reshape(-1, aggregation_factor).mean(dim=1)
            if coarse.shape[0] > coarse_len:
                coarse = coarse[-coarse_len:]
        fine = series[-fine_len:].to(torch.float32)
        return coarse, fine

    def _prepare_context(
        self,
        past_values: Sequence[torch.Tensor] | Sequence[tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coarse_len = self.config.context_length
        fine_len = self.config.context_length
        aggregation_factor = self.config.aggregation_factor

        coarse_batch = torch.zeros((len(past_values), coarse_len), dtype=torch.float32, device=device)
        coarse_pad = torch.zeros_like(coarse_batch)
        fine_batch = torch.zeros((len(past_values), fine_len), dtype=torch.float32, device=device)
        fine_pad = torch.zeros_like(fine_batch)

        for i, item in enumerate(past_values):
            if isinstance(item, (tuple, list)) and len(item) == 2:
                coarse, fine = item
                coarse = torch.as_tensor(coarse, dtype=torch.float32, device=device).reshape(-1)
                fine = torch.as_tensor(fine, dtype=torch.float32, device=device).reshape(-1)
            else:
                series = torch.as_tensor(item, dtype=torch.float32, device=device).reshape(-1)
                coarse, fine = self._build_multi_resolution(series, aggregation_factor, coarse_len, fine_len)

            num_coarse_values = coarse.shape[0]
            if num_coarse_values >= coarse_len:
                coarse_batch[i] = coarse[-coarse_len:]
            elif num_coarse_values > 0:
                coarse_batch[i, coarse_len - num_coarse_values :] = coarse
                coarse_pad[i, : coarse_len - num_coarse_values] = 1.0
            else:
                coarse_pad[i] = 1.0

            num_fine_values = fine.shape[0]
            if num_fine_values >= fine_len:
                fine_batch[i] = fine[-fine_len:]
            elif num_fine_values > 0:
                fine_batch[i, fine_len - num_fine_values :] = fine
                fine_pad[i, : fine_len - num_fine_values] = 1.0
            else:
                fine_pad[i] = 1.0

        return coarse_batch, coarse_pad, fine_batch, fine_pad

    def _project_last_fine(self, outputs: CtsmOutput, last_position: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the hidden state at `last_position` through the horizon head and denormalize."""
        last_hidden = outputs.last_hidden_state[:, last_position : last_position + 1, :]
        head = self.horizon_ff_layer(last_hidden)
        bsz = head.shape[0]
        num_outputs = 1 + len(self.config.quantiles)
        head = head.view(bsz, self.config.horizon_length, num_outputs)

        # Two-stage denormalization, mirroring the official reference: the head's output is in
        # (per-first-patch ∘ stream)-normalized space because `_patchify` applies TimesFM's
        # `_forward_transform` on top of the stream-level `_normalize_with_pad`. Undo per-first-patch
        # first, then undo the stream normalization.
        if outputs.loc_fine_patch is not None and outputs.scale_fine_patch is not None:
            mu_fp = outputs.loc_fine_patch[:, None, None]
            sigma_fp = outputs.scale_fine_patch[:, None, None]
            head = head * sigma_fp + mu_fp

        loc = outputs.loc[:, None, None]
        scale = outputs.scale[:, None, None] + _STREAM_NORM_EPS
        mean_patch = head[..., 0] * scale[..., 0] + loc[..., 0]
        quant_patch = head[..., 1:] * scale + loc
        mean_patch = torch.nan_to_num(mean_patch, nan=0.0, posinf=0.0, neginf=0.0)
        quant_patch = torch.nan_to_num(quant_patch, nan=0.0, posinf=0.0, neginf=0.0)
        return mean_patch, quant_patch

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        past_values: Sequence[torch.Tensor] | Sequence[tuple[torch.Tensor, torch.Tensor]],
        future_values: torch.Tensor | None = None,
        horizon_len: int | None = None,
        freq: Sequence[int] | torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CtsmOutputForPrediction:
        r"""
        past_values (`Sequence[torch.Tensor]`):
            Either a list of 1-D fine-resolution tensors (the coarse stream is derived by mean-aggregating over
            `aggregation_factor` consecutive points) or a list of `(coarse, fine)` pairs if both streams are provided.
        future_values (`torch.Tensor`, *optional*):
            Optional fine-resolution ground truth used to compute the loss.
        horizon_len (`int`, *optional*):
            Number of fine-resolution steps to forecast. Defaults to `config.horizon_length`. Values larger than
            `config.horizon_length` trigger autoregressive decoding (the full multi-resolution context is
            rebuilt at every step, matching the reference).
        freq (`Sequence[int]` or `torch.Tensor`, *optional*):
            Frequency indices. Defaults to zeros.
        """
        device = self.horizon_ff_layer.input_layer.weight.device
        horizon_len = horizon_len or self.config.horizon_length
        if horizon_len <= 0:
            raise ValueError("horizon_len must be positive")

        output_patch_len = self.config.horizon_length
        num_decode_patches = (horizon_len + output_patch_len - 1) // output_patch_len

        coarse, coarse_pad, fine, fine_pad = self._prepare_context(past_values, device=device)
        bsz = coarse.shape[0]

        if freq is None:
            freq_tensor = torch.zeros((bsz, 1), dtype=torch.long, device=device)
        else:
            freq_tensor = torch.as_tensor(
                list(freq) if not isinstance(freq, torch.Tensor) else freq, dtype=torch.long, device=device
            ).view(bsz, 1)

        mean_chunks: list[torch.Tensor] = []
        quant_chunks: list[torch.Tensor] = []
        remaining = horizon_len
        last_outputs: CtsmOutput | None = None
        max_fine = self.config.context_length
        max_coarse = self.config.context_length
        aggregation_factor = self.config.aggregation_factor
        coarse_buffer = torch.zeros((bsz, 0), dtype=torch.float32, device=device)

        for _ in range(num_decode_patches):
            # Full recompute at every AR step: stream stats, per-first-patch stats, and the
            # coarse/fine windows are all refreshed by `CtsmModel.forward` from the current
            # raw context. This matches the reference's `decode` loop exactly.
            last_outputs: CtsmOutput = self.model(
                past_values_coarse=coarse,
                past_values_fine=fine,
                past_values_coarse_padding=coarse_pad,
                past_values_fine_padding=fine_pad,
                freq=freq_tensor,
                **kwargs,
            )
            mean_patch, quant_patch = self._project_last_fine(
                last_outputs, last_outputs.last_hidden_state.shape[1] - 1
            )

            take = min(remaining, output_patch_len)
            mean_chunks.append(mean_patch[:, :take])
            quant_chunks.append(quant_patch[:, :take, :])
            remaining -= take
            if remaining <= 0:
                break

            new_fine = mean_patch[:, :output_patch_len]

            # Slide the fine window: append the new predictions, trim to the most recent
            # `max_fine` values. The trimmed values may shift the "first patch" used by the
            # per-first-patch normalization, which is intentional.
            fine = torch.cat([fine, new_fine], dim=1)
            fine_pad = torch.cat(
                [fine_pad, torch.zeros((bsz, output_patch_len), device=device, dtype=fine_pad.dtype)], dim=1
            )
            if fine.shape[1] > max_fine:
                fine = fine[:, -max_fine:]
                fine_pad = fine_pad[:, -max_fine:]

            # Aggregate new fine predictions into the coarse stream as full blocks of length
            # `aggregation_factor` become available; carry the remainder in `coarse_buffer`.
            coarse_buffer = torch.cat([coarse_buffer, new_fine], dim=1)
            full_blocks = coarse_buffer.shape[1] // aggregation_factor
            if full_blocks > 0:
                blocks = (
                    coarse_buffer[:, : full_blocks * aggregation_factor]
                    .view(bsz, full_blocks, aggregation_factor)
                    .mean(dim=2)
                )
                coarse_buffer = coarse_buffer[:, full_blocks * aggregation_factor :]
                coarse = torch.cat([coarse, blocks], dim=1)
                coarse_pad = torch.cat(
                    [coarse_pad, torch.zeros((bsz, full_blocks), device=device, dtype=coarse_pad.dtype)], dim=1
                )
                if coarse.shape[1] > max_coarse:
                    coarse = coarse[:, -max_coarse:]
                    coarse_pad = coarse_pad[:, -max_coarse:]

        mean_predictions = torch.cat(mean_chunks, dim=1)[:, :horizon_len]
        full_predictions = torch.cat(
            [torch.cat(mean_chunks, dim=1)[:, :horizon_len, None], torch.cat(quant_chunks, dim=1)[:, :horizon_len, :]],
            dim=-1,
        )

        loss = None
        if future_values is not None:
            target_len = min(future_values.shape[1], mean_predictions.shape[1])
            mse_loss = F.mse_loss(mean_predictions[:, :target_len], future_values[:, :target_len])
            quantile_loss = self._quantile_loss(full_predictions[:, :target_len, 1:], future_values[:, :target_len])
            loss = mse_loss + quantile_loss

        return CtsmOutputForPrediction(
            last_hidden_state=last_outputs.last_hidden_state if last_outputs is not None else None,
            hidden_states=last_outputs.hidden_states if last_outputs is not None else None,
            attentions=last_outputs.attentions if last_outputs is not None else None,
            mean_predictions=mean_predictions,
            full_predictions=full_predictions,
            loss=loss,
        )


__all__ = [
    "CtsmConfig",
    "CtsmModel",
    "CtsmModelForPrediction",
    "CtsmPreTrainedModel",
]
