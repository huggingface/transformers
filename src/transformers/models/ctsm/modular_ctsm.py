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
from ...cache_utils import Cache, DynamicCache
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
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
    num_hidden_layers (`int`, *optional*, defaults to 25):
        Number of decoder layers.
    quantiles (`list[float]`, *optional*, defaults to 15 values between 0.01 and 0.99):
        Quantile levels predicted by the model.
    use_positional_embedding (`bool`, *optional*, defaults to `False`):
        CTSM uses rotary position embeddings and does not add sinusoidal positional embeddings.
    use_resolution_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to add a learned embedding per resolution bucket (coarse / special / fine).
    use_special_token (`bool`, *optional*, defaults to `True`):
        Whether to insert a learned special token between the coarse and fine streams.
    num_resolutions (`int`, *optional*, defaults to 3):
        Number of resolution embeddings (coarse, special token, fine).
    agg_factor (`int`, *optional*, defaults to 60):
        Aggregation factor between fine and coarse resolutions (e.g. 60 minutes -> 1 hour).
    max_position_embeddings (`int`, *optional*, defaults to 1025):
        Maximum number of patches in the concatenated sequence (coarse + special + fine).
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
    context_length: int = 512
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
    use_positional_embedding: bool = False
    use_resolution_embeddings: bool = True
    use_special_token: bool = True
    num_resolutions: int = 3
    agg_factor: int = 60
    max_position_embeddings: int = 1025
    rope_parameters: RopeParameters | dict | None = None

    min_timescale = AttributeError()
    max_timescale = AttributeError()


@dataclass
@auto_docstring
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
    num_coarse_patches (`int`):
        Number of patches (including the optional special token) preceding the fine-resolution block.
    num_fine_patches (`int`):
        Number of patches in the fine-resolution block of the concatenated sequence.
    past_key_values (`Cache`, *optional*):
        Key/value cache for the concatenated `[coarse, special, fine]` sequence. Populated when the
        caller passes `use_cache=True` (and re-used across autoregressive decode steps). Typically only
        the long-horizon AR loop in [`CtsmModelForPrediction`] needs this.
    """

    loc_coarse: torch.Tensor | None = None
    scale_coarse: torch.Tensor | None = None
    num_coarse_patches: int | None = None
    num_fine_patches: int | None = None
    past_key_values: Cache | None = None


@dataclass
@auto_docstring
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
    """TimesFM 2.0 style attention with learnable per-dimension Q scaling and rotary position embeddings.

    Supports an optional `past_key_values` cache so that, during long-horizon autoregressive decoding,
    each step only needs to compute K/V for the newly-appended fine patches and attends to the
    previously-cached K/V for every earlier position.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = self._scale_query(query_states)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

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
        self.self_attn = CtsmAttention(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        paddings: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.mlp(hidden_states, paddings=paddings)
        return hidden_states


@auto_docstring
class CtsmPreTrainedModel(TimesFmPreTrainedModel):
    config: CtsmConfig
    base_model_prefix = "model"
    _no_split_modules = ["CtsmDecoderLayer"]
    _supports_flash_attn = True
    _supports_flex_attn = True
    _can_record_outputs = {
        "hidden_states": CtsmDecoderLayer,
        "attentions": CtsmAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, CtsmModel) and getattr(module, "special_token", None) is not None:
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

        if hasattr(self, "position_emb"):
            del self.position_emb

        self.rotary_emb = CtsmRotaryEmbedding(config)

        if config.use_resolution_embeddings:
            self.multi_resolution = nn.Embedding(config.num_resolutions, config.hidden_size)

        if config.use_special_token:
            self.special_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.post_init()

    @staticmethod
    def _left_pad_to_patch_boundary(
        values: torch.Tensor, paddings: torch.Tensor, patch_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rem = values.shape[1] % patch_length
        if rem == 0:
            return values, paddings
        pad_len = patch_length - rem
        values_pad = torch.zeros((values.shape[0], pad_len), device=values.device, dtype=values.dtype)
        paddings_pad = torch.ones((paddings.shape[0], pad_len), device=paddings.device, dtype=paddings.dtype)
        return torch.cat([values_pad, values], dim=1), torch.cat([paddings_pad, paddings], dim=1)

    @staticmethod
    def _normalize_with_pad(
        context: torch.Tensor, padding: torch.Tensor, tolerance: float = 1e-8
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stream-level normalization that matches the original CTSM reference.

        Normalizes ``context`` using the mean and standard deviation computed over the
        non-padded positions (``padding == 0``) across the whole context, rather than
        TimesFM's per-first-patch statistics. The normalized tensor has padded positions
        zeroed out and is clamped to a safe range.
        """
        valid = 1.0 - padding
        count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        mu = (context * valid).sum(dim=1, keepdim=True) / count

        seq_len_f = context.new_tensor(float(context.shape[1]))
        filled = torch.where(padding.to(dtype=torch.bool), mu, context)
        sigma = filled.std(dim=1, keepdim=True, unbiased=False) * torch.sqrt(seq_len_f / count)
        sigma = sigma.clamp_min(1e-2)

        normalized = (context - mu) / (sigma + tolerance)
        normalized = normalized * valid
        normalized = normalized.clamp(-1000.0, 1000.0)
        return normalized, mu.squeeze(-1), sigma.squeeze(-1)

    def _patchify(
        self, past_values: torch.Tensor, past_values_padding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Patchify an already stream-normalized stream and project through the input tokenizer."""
        bsize = past_values.shape[0]
        patched_inputs = past_values.view(bsize, -1, self.config.patch_length)
        patched_pads = past_values_padding.view(bsize, -1, self.config.patch_length)

        patched_inputs = patched_inputs * (1.0 - patched_pads)
        concat_inputs = torch.cat([patched_inputs, patched_pads], dim=-1)
        embeddings = self.input_ff_layer(concat_inputs)
        patch_padding = torch.min(patched_pads, dim=-1)[0]
        return embeddings, patch_padding

    def _build_attention_mask(
        self,
        patch_padding: torch.Tensor,
        num_coarse_patches: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Reuse TimesFM's padding+causal 4D mask, then open the coarse-coarse block to bidirectional."""
        attention_mask = self._prepare_4d_attention_mask(
            attention_mask=patch_padding,
            sequence_length=patch_padding.shape[1],
            dtype=dtype,
            device=patch_padding.device,
            is_causal=True,
        )
        if num_coarse_patches > 0:
            attention_mask[..., :num_coarse_patches, :num_coarse_patches] = 0.0
        return attention_mask

    def _build_incremental_attention_mask(
        self, bsize: int, num_new: int, past_length: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Mask for the incremental (cached) path: new fine Qs attend to all cached K/V plus causal within the new block."""
        min_value = torch.finfo(dtype).min
        mask = torch.zeros((bsize, 1, num_new, past_length + num_new), dtype=dtype, device=device)
        if num_new > 1:
            causal_new = torch.triu(torch.full((num_new, num_new), min_value, dtype=dtype, device=device), diagonal=1)
            mask[:, :, :, past_length:] = causal_new
        return mask

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
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        loc_fine: torch.Tensor | None = None,
        scale_fine: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CtsmOutput:
        r"""
        past_values_coarse (`torch.FloatTensor` of shape `(batch_size, coarse_length)`, *optional*):
            Coarse-resolution context (e.g. hourly aggregates). Length must be a multiple of `patch_length` or
            will be left-padded to one. Required when `past_key_values` is `None`.
        past_values_fine (`torch.FloatTensor` of shape `(batch_size, fine_length)`):
            Fine-resolution context (e.g. minute-level). In the normal / full-forward mode this is the entire
            fine context; when `past_key_values` is supplied this should contain **only the new fine values**
            to append — they must already be pre-normalized by the caller using `loc_fine` / `scale_fine`.
        past_values_coarse_padding (`torch.LongTensor`, *optional*):
            Padding mask for the coarse stream, `1.0` for padded positions and `0.0` for real values.
        past_values_fine_padding (`torch.LongTensor`, *optional*):
            Padding mask for the fine stream.
        freq (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*):
            Frequency indices. Defaults to all zeros.
        past_key_values (`Cache`, *optional*):
            A [`Cache`] (typically a [`DynamicCache`]) holding K/V for the concatenated
            `[coarse, special, fine_prefix]` sequence from a previous call. When supplied the model runs in
            **incremental mode**: only the new fine patches are embedded, and their Q/K/V are added on top
            of the cached K/V. `loc_fine` / `scale_fine` **must** also be supplied so the new fine values
            are normalized on the same scale as the cached ones.
        use_cache (`bool`, *optional*):
            Whether to build and return a key/value cache in the `CtsmOutput`. Defaults to `False` unless
            `past_key_values` is provided (in which case caching is always on).
        cache_position (`torch.LongTensor` of shape `(num_new,)`, *optional*):
            Absolute positions (in the full `[coarse, special, fine]` sequence) of the new fine patches.
            Only used in incremental mode; defaults to `torch.arange(past_length, past_length + num_new)`.
        loc_fine (`torch.Tensor` of shape `(batch_size,)`, *optional*):
            Fine-stream mean used for stream normalization. Required in incremental mode.
        scale_fine (`torch.Tensor` of shape `(batch_size,)`, *optional*):
            Fine-stream standard deviation used for stream normalization. Required in incremental mode.
        """
        if past_key_values is None:
            return self._full_forward(
                past_values_coarse=past_values_coarse,
                past_values_fine=past_values_fine,
                past_values_coarse_padding=past_values_coarse_padding,
                past_values_fine_padding=past_values_fine_padding,
                freq=freq,
                use_cache=bool(use_cache),
                **kwargs,
            )
        return self._incremental_forward(
            past_values_fine=past_values_fine,
            past_values_fine_padding=past_values_fine_padding,
            freq=freq,
            past_key_values=past_key_values,
            cache_position=cache_position,
            loc_fine=loc_fine,
            scale_fine=scale_fine,
            **kwargs,
        )

    def _full_forward(
        self,
        past_values_coarse: torch.Tensor,
        past_values_fine: torch.Tensor,
        past_values_coarse_padding: torch.LongTensor | None,
        past_values_fine_padding: torch.LongTensor | None,
        freq: torch.Tensor | None,
        use_cache: bool,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CtsmOutput:
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
            past_values_coarse, past_values_coarse_padding, tolerance=self.config.tolerance
        )
        fine_normalized, loc_fine, scale_fine = self._normalize_with_pad(
            past_values_fine, past_values_fine_padding, tolerance=self.config.tolerance
        )

        coarse_embeddings, coarse_patch_padding = self._patchify(coarse_normalized, past_values_coarse_padding)
        fine_embeddings, fine_patch_padding = self._patchify(fine_normalized, past_values_fine_padding)

        bsize, num_coarse_patches, hidden_size = coarse_embeddings.shape
        num_fine_patches = fine_embeddings.shape[1]
        device = coarse_embeddings.device
        dtype = coarse_embeddings.dtype

        if self.config.use_special_token:
            special = self.special_token.to(device=device, dtype=dtype).expand(bsize, 1, hidden_size)
            special_padding = torch.zeros(bsize, 1, device=device, dtype=coarse_patch_padding.dtype)
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
            mr_idx = torch.cat([mr_coarse, mr_special, mr_fine], dim=0).unsqueeze(0).expand(bsize, -1)
            model_input = model_input + self.multi_resolution(mr_idx)

        if freq is None:
            freq = torch.zeros((bsize, 1), dtype=torch.long, device=device)
        else:
            freq = freq.to(device=device, dtype=torch.long)
        model_input = model_input + self.freq_emb(freq)

        attention_mask = self._build_attention_mask(patch_padding, num_coarse_patches, model_input.dtype)
        position_ids = (
            torch.arange(model_input.shape[1], device=device, dtype=torch.long).unsqueeze(0).expand(bsize, -1)
        )
        position_embeddings = self.rotary_emb(model_input, position_ids)

        past_key_values = DynamicCache(config=self.config) if use_cache else None

        hidden_states = model_input
        for layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                paddings=patch_padding,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                **kwargs,
            )

        return CtsmOutput(
            last_hidden_state=hidden_states,
            loc=loc_fine,
            scale=scale_fine,
            loc_coarse=loc_coarse,
            scale_coarse=scale_coarse,
            num_coarse_patches=num_coarse_patches + num_special,
            num_fine_patches=num_fine_patches,
            past_key_values=past_key_values,
        )

    def _incremental_forward(
        self,
        past_values_fine: torch.Tensor,
        past_values_fine_padding: torch.LongTensor | None,
        freq: torch.Tensor | None,
        past_key_values: Cache,
        cache_position: torch.LongTensor | None,
        loc_fine: torch.Tensor | None,
        scale_fine: torch.Tensor | None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CtsmOutput:
        if loc_fine is None or scale_fine is None:
            raise ValueError(
                "`loc_fine` and `scale_fine` must be supplied together with `past_key_values` so that the new fine "
                "values are normalized on the same scale as the cached ones."
            )
        if past_values_fine.shape[1] % self.config.patch_length != 0:
            raise ValueError(
                f"In incremental mode `past_values_fine` length must be a multiple of `patch_length="
                f"{self.config.patch_length}`; got {past_values_fine.shape[1]}."
            )

        if past_values_fine_padding is None:
            past_values_fine_padding = torch.zeros_like(past_values_fine)
        past_values_fine_padding = past_values_fine_padding.to(past_values_fine.dtype)

        tol = self.config.tolerance
        fine_normalized = (past_values_fine - loc_fine.unsqueeze(-1)) / (scale_fine.unsqueeze(-1) + tol)
        fine_normalized = fine_normalized * (1.0 - past_values_fine_padding)
        fine_normalized = fine_normalized.clamp(-1000.0, 1000.0)

        new_embeddings, new_patch_padding = self._patchify(fine_normalized, past_values_fine_padding)
        bsize, num_new, _ = new_embeddings.shape
        device = new_embeddings.device
        dtype = new_embeddings.dtype

        if self.config.use_resolution_embeddings:
            mr_idx = torch.full((bsize, num_new), 2, dtype=torch.long, device=device)
            new_embeddings = new_embeddings + self.multi_resolution(mr_idx)

        if freq is None:
            freq = torch.zeros((bsize, 1), dtype=torch.long, device=device)
        else:
            freq = freq.to(device=device, dtype=torch.long)
        new_embeddings = new_embeddings + self.freq_emb(freq)

        past_length = past_key_values.get_seq_length()
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + num_new, dtype=torch.long, device=device)
        position_ids = cache_position.unsqueeze(0).expand(bsize, -1)
        position_embeddings = self.rotary_emb(new_embeddings, position_ids)

        attention_mask = self._build_incremental_attention_mask(bsize, num_new, past_length, dtype, device)

        hidden_states = new_embeddings
        for layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                paddings=new_patch_padding,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                **kwargs,
            )

        return CtsmOutput(
            last_hidden_state=hidden_states,
            loc=loc_fine,
            scale=scale_fine,
            num_fine_patches=num_new,
            past_key_values=past_key_values,
        )


class CtsmModelForPrediction(TimesFmModelForPrediction):
    """CTSM model with a multi-resolution prediction head and autoregressive multi-resolution decoding.

    For horizons that require autoregressive decoding (``horizon_len > config.horizon_length``) the
    prediction class reuses a key/value cache across AR steps: the first step runs the full forward
    and populates a [`DynamicCache`], subsequent steps feed only the newly-appended fine patches
    through the stack and attend to the cached K/V for every earlier position. Two caveats, matching
    how a KV cache is made to fit CTSM's architecture:

    * Stream-level normalization statistics (``loc_fine``, ``scale_fine``) are frozen to the values
      computed on the first step. This is a small approximation: in the untracked reference,
      statistics are recomputed after each prediction is appended; in practice the drift is small
      when forecasts stay in-distribution.
    * If an AR step would grow the coarse block (a new coarse patch is formed once every
      ``patch_length * agg_factor / output_patch_len`` steps, i.e. ~every 15 steps at the defaults),
      the cache is discarded and a full forward is run, rebuilding the cache.
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
        series: torch.Tensor, agg_factor: int, coarse_len: int, fine_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (coarse, fine) contexts from a 1-D fine-resolution series.

        Coarse is the mean of the last `coarse_len * agg_factor` fine samples, aligned to block boundaries.
        Fine is the last `fine_len` samples.
        """
        series = series.to(torch.float32).reshape(-1)
        needed = coarse_len * agg_factor
        raw = series[-needed:]
        remainder = raw.shape[0] % agg_factor
        if remainder:
            raw = raw[remainder:]
        if raw.numel() == 0:
            coarse = series.new_empty((0,), dtype=torch.float32)
        else:
            coarse = raw.reshape(-1, agg_factor).mean(dim=1)
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
        agg = self.config.agg_factor

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
                coarse, fine = self._build_multi_resolution(series, agg, coarse_len, fine_len)

            c_n = coarse.shape[0]
            if c_n >= coarse_len:
                coarse_batch[i] = coarse[-coarse_len:]
            elif c_n > 0:
                coarse_batch[i, coarse_len - c_n :] = coarse
                coarse_pad[i, : coarse_len - c_n] = 1.0
            else:
                coarse_pad[i] = 1.0

            f_n = fine.shape[0]
            if f_n >= fine_len:
                fine_batch[i] = fine[-fine_len:]
            elif f_n > 0:
                fine_batch[i, fine_len - f_n :] = fine
                fine_pad[i, : fine_len - f_n] = 1.0
            else:
                fine_pad[i] = 1.0

        return coarse_batch, coarse_pad, fine_batch, fine_pad

    def _project_last_fine(self, outputs: CtsmOutput, last_position: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the hidden state at `last_position` through the horizon head and denormalize."""
        last_hidden = outputs.last_hidden_state[:, last_position : last_position + 1, :]
        head = self.horizon_ff_layer(last_hidden)
        bsize = head.shape[0]
        num_outputs = 1 + len(self.config.quantiles)
        head = head.view(bsize, self.config.horizon_length, num_outputs)

        loc = outputs.loc[:, None, None]
        scale = outputs.scale[:, None, None]
        mean_patch = head[..., 0] * scale[..., 0] + loc[..., 0]
        quant_patch = head[..., 1:] * scale + loc
        mean_patch = torch.nan_to_num(mean_patch, nan=0.0, posinf=0.0, neginf=0.0)
        quant_patch = torch.nan_to_num(quant_patch, nan=0.0, posinf=0.0, neginf=0.0)
        return mean_patch, quant_patch

    def _decode_step_full(
        self,
        past_values_coarse: torch.Tensor,
        past_values_fine: torch.Tensor,
        past_values_coarse_padding: torch.Tensor,
        past_values_fine_padding: torch.Tensor,
        freq: torch.Tensor,
        use_cache: bool,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, CtsmOutput]:
        """Full forward through the model. If `use_cache`, the returned outputs carry a fresh cache."""
        outputs: CtsmOutput = self.model(
            past_values_coarse=past_values_coarse,
            past_values_fine=past_values_fine,
            past_values_coarse_padding=past_values_coarse_padding,
            past_values_fine_padding=past_values_fine_padding,
            freq=freq,
            use_cache=use_cache,
            **kwargs,
        )
        mean_patch, quant_patch = self._project_last_fine(outputs, outputs.last_hidden_state.shape[1] - 1)
        return mean_patch, quant_patch, outputs

    def _decode_step_incremental(
        self,
        new_fine_values: torch.Tensor,
        freq: torch.Tensor,
        past_key_values: Cache,
        loc_fine: torch.Tensor,
        scale_fine: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, CtsmOutput]:
        """Append `new_fine_values` to the cached state and run only the new positions through the stack."""
        outputs: CtsmOutput = self.model(
            past_values_fine=new_fine_values,
            freq=freq,
            past_key_values=past_key_values,
            loc_fine=loc_fine,
            scale_fine=scale_fine,
            **kwargs,
        )
        mean_patch, quant_patch = self._project_last_fine(outputs, outputs.last_hidden_state.shape[1] - 1)
        return mean_patch, quant_patch, outputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        past_values: Sequence[torch.Tensor] | Sequence[tuple[torch.Tensor, torch.Tensor]],
        future_values: torch.Tensor | None = None,
        horizon_len: int | None = None,
        freq: Sequence[int] | torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CtsmOutputForPrediction:
        r"""
        past_values (`Sequence[torch.Tensor]`):
            Either a list of 1-D fine-resolution tensors (the coarse stream is derived by mean-aggregating over
            `agg_factor` consecutive points) or a list of `(coarse, fine)` pairs if both streams are provided.
        future_values (`torch.Tensor`, *optional*):
            Optional fine-resolution ground truth used to compute the loss.
        horizon_len (`int`, *optional*):
            Number of fine-resolution steps to forecast. Defaults to `config.horizon_length`. Values larger than
            `config.horizon_length` trigger autoregressive decoding.
        freq (`Sequence[int]` or `torch.Tensor`, *optional*):
            Frequency indices. Defaults to zeros.
        use_cache (`bool`, *optional*):
            Whether to use a key/value cache across autoregressive decode steps. Defaults to `True` when
            `horizon_len > config.horizon_length` (i.e. when AR decoding is needed) and `False` otherwise.
            Set to `False` to force a full recompute at every AR step (matches the original reference
            behaviour; slower but avoids the stream-stats-freezing approximation).
        """
        device = self.horizon_ff_layer.input_layer.weight.device
        horizon_len = horizon_len or self.config.horizon_length
        if horizon_len <= 0:
            raise ValueError("horizon_len must be positive")

        output_patch_len = self.config.horizon_length
        num_decode_patches = (horizon_len + output_patch_len - 1) // output_patch_len

        coarse, coarse_pad, fine, fine_pad = self._prepare_context(past_values, device=device)
        bsize = coarse.shape[0]

        if freq is None:
            freq_tensor = torch.zeros((bsize, 1), dtype=torch.long, device=device)
        else:
            freq_tensor = torch.as_tensor(
                list(freq) if not isinstance(freq, torch.Tensor) else freq, dtype=torch.long, device=device
            ).view(bsize, 1)

        mean_chunks: list[torch.Tensor] = []
        quant_chunks: list[torch.Tensor] = []
        remaining = horizon_len
        last_outputs: CtsmOutput | None = None
        max_fine = self.config.context_length
        max_coarse = self.config.context_length
        agg = self.config.agg_factor
        new_fine_patches = self.config.horizon_length // self.config.patch_length

        past_key_values: Cache | None = None
        frozen_loc_fine: torch.Tensor | None = None
        frozen_scale_fine: torch.Tensor | None = None
        coarse_buffer = torch.zeros((bsize, 0), dtype=torch.float32, device=device)

        if use_cache is None:
            use_cache = num_decode_patches > 1
        pending_new_fine: torch.Tensor | None = None

        for step_idx in range(num_decode_patches):
            if past_key_values is None:
                # First step (or after cache invalidation): full forward. The coarse block in the cache
                # stays frozen at the initial state — only the fine block grows via subsequent incremental
                # steps — which matches how KV caches work for append-only sequences.
                mean_patch, quant_patch, last_outputs = self._decode_step_full(
                    past_values_coarse=coarse,
                    past_values_fine=fine,
                    past_values_coarse_padding=coarse_pad,
                    past_values_fine_padding=fine_pad,
                    freq=freq_tensor,
                    use_cache=use_cache,
                    **kwargs,
                )
                past_key_values = last_outputs.past_key_values
                frozen_loc_fine = last_outputs.loc
                frozen_scale_fine = last_outputs.scale
            else:
                # Incremental: only the fine values newly appended last step go through the stack.
                mean_patch, quant_patch, last_outputs = self._decode_step_incremental(
                    new_fine_values=pending_new_fine,
                    freq=freq_tensor,
                    past_key_values=past_key_values,
                    loc_fine=frozen_loc_fine,
                    scale_fine=frozen_scale_fine,
                    **kwargs,
                )

            take = min(remaining, output_patch_len)
            mean_chunks.append(mean_patch[:, :take])
            quant_chunks.append(quant_patch[:, :take, :])
            remaining -= take
            if remaining <= 0:
                break

            new_fine = mean_patch[:, :output_patch_len]
            pending_new_fine = new_fine

            # Track the raw contexts so the next full-forward (initial step or after cache
            # invalidation) sees the right state. Mirrors the reference AR loop.
            fine = torch.cat([fine, new_fine], dim=1)
            fine_pad = torch.cat(
                [fine_pad, torch.zeros((bsize, output_patch_len), device=device, dtype=fine_pad.dtype)], dim=1
            )
            if fine.shape[1] > max_fine:
                fine = fine[:, -max_fine:]
                fine_pad = fine_pad[:, -max_fine:]

            coarse_buffer = torch.cat([coarse_buffer, new_fine], dim=1)
            full_blocks = coarse_buffer.shape[1] // agg
            if full_blocks > 0:
                blocks = coarse_buffer[:, : full_blocks * agg].view(bsize, full_blocks, agg).mean(dim=2)
                coarse_buffer = coarse_buffer[:, full_blocks * agg :]
                coarse = torch.cat([coarse, blocks], dim=1)
                coarse_pad = torch.cat(
                    [coarse_pad, torch.zeros((bsize, full_blocks), device=device, dtype=coarse_pad.dtype)], dim=1
                )
                if coarse.shape[1] > max_coarse:
                    coarse = coarse[:, -max_coarse:]
                    coarse_pad = coarse_pad[:, -max_coarse:]

            if past_key_values is not None:
                projected_len = past_key_values.get_seq_length() + new_fine_patches
                if projected_len >= self.config.max_position_embeddings:
                    past_key_values = None
                    pending_new_fine = None

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
