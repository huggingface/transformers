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
from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_masks_for_generate
from ...modeling_outputs import BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...vision_utils import get_vision_position_ids
from ..deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave
from ..gemma2.configuration_gemma2 import Gemma2Config
from ..gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2DecoderLayer,
    Gemma2ForCausalLM,
    Gemma2MLP,
    Gemma2Model,
    Gemma2PreTrainedModel,
    Gemma2RMSNorm,
    Gemma2RotaryEmbedding,
    eager_attention_forward,
)
from ..gemma3.modeling_gemma3 import (
    Gemma3CausalLMOutputWithPast,
    Gemma3ForConditionalGeneration,
    Gemma3Model,
    Gemma3ModelOutputWithPast,
)
from ..gemma4.modeling_gemma4 import Gemma4VisionRotaryEmbedding, apply_multidimensional_rope
from ..llama4.modeling_llama4 import Llama4TextL2Norm


logger = logging.get_logger(__name__)


class OnyxModelOutputWithPast(Gemma3ModelOutputWithPast):
    pass


class OnyxCausalLMOutputWithPast(Gemma3CausalLMOutputWithPast):
    pass


@auto_docstring
@strict
class OnyxVisionConfig(PreTrainedConfig):
    r"""
    TODO
    """

    model_type = "onyx_vision"
    attribute_map = {
        "hidden_size": "hidden_size",
        "vision_heads": "num_attention_heads",
        "vision_layers": "num_hidden_layers",
    }

    hidden_size: int = 1536
    output_dim: int = 6144
    num_hidden_layers: int = 50
    num_attention_heads: int = 16
    mlp_ratio: float = 8960 / 1536
    patch_size: int = 14
    patch_temporal: int = 2
    downsample_factor: int = 2
    sparse_attention_factor: int = 4
    pos_emb_grid_h: int = 32
    pos_emb_grid_w: int = 32
    adapter_dim: int = 4096
    video_num_frames: int = 96
    video_sampling_fps: float = 2.0


@auto_docstring
@strict
class OnyxTextConfig(Gemma2Config, PreTrainedConfig):
    r"""
    qk_scale_factor (`float`, *optional*, defaults to 43.7840518911):
        Multiplier applied to Q after QK-norm, before the standard `1/sqrt(head_dim)` attention scaling.
    use_qk_norm (`bool`, *optional*, defaults to `True`):
        Whether to apply a scaleless RMSNorm to Q and K before rotary.
    use_attn_output_gate (`bool`, *optional*, defaults to `True`):
        Whether to gate the per-head attention output with `sigmoid(output_gate_proj(hidden))`.
    output_multiplier (`float`, *optional*, defaults to 0.19611613513818404):
        Scale applied to logits before the final tanh softcap.
    normalize_tok_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to apply a scaleless RMSNorm to the token embeddings before the decoder stack.
    post_norm_eps (`float`, *optional*, defaults to 1e-8):
        Epsilon used for the post-attention and post-FFN norms (which sit between the sub-layer output and the residual).
    every_n_layers_nope (`int`, *optional*, defaults to 4):
        iRoPE stride. NoPE (no rotary) is applied every N layers, counting backward from the last layer.
    no_rope_layers (`list[int]`, *optional*):
        Explicit per-layer rotary mask: 1 = apply rotary, 0 = NoPE. Derived from `every_n_layers_nope` if unset.
    """

    model_type = "onyx_text"

    vocab_size: int = 202_048
    hidden_size: int = 6656
    intermediate_size: int = 19968
    num_hidden_layers: int = 52
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    hidden_activation: str = "silu"
    max_position_embeddings: int = 16_384
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    bos_token_id: int | None = 200_000
    eos_token_id: int | list[int] | None = 200_001
    pad_token_id: int | None = None
    sliding_window: int | None = 2048
    final_logit_softcapping: float | None = 20.0
    attn_logit_softcapping: float | None = None
    layer_types: list[str] | None = None

    # Onyx-specific fields
    qk_scale_factor: float = 43.7840518911
    use_qk_norm: bool = True
    use_attn_output_gate: bool = True
    output_multiplier: float = 0.19611613513818404
    normalize_tok_embeddings: bool = True
    post_norm_eps: float = 1e-8
    every_n_layers_nope: int = 4
    no_rope_layers: list[int] | None = None

    def __post_init__(self, **kwargs):
        # Accept the legacy `hidden_act` alias from checkpoints saved with the trust_remote_code impl.
        if (legacy_act := kwargs.pop("hidden_act", None)) is not None:
            self.hidden_activation = legacy_act

        # iRoPE mask: NoPE layers counted backward from the last layer.
        if self.no_rope_layers is None:
            self.no_rope_layers = [
                0 if (self.num_hidden_layers - 1 - i) % self.every_n_layers_nope == 0 else 1
                for i in range(self.num_hidden_layers)
            ]

        # Full attention for NoPE layers, sliding otherwise (Onyx's default layout matches
        # the sliding_window_pattern [w, w, w, 0] used in the reference config).
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if self.no_rope_layers[i] == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]

        PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring
@strict
class OnyxConfig(PreTrainedConfig):
    r"""
    TODO
    """

    model_type = "onyx"
    sub_configs = {
        "text_config": OnyxTextConfig,
        "vision_config": OnyxVisionConfig,
    }

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 200092
    video_token_id: int = 200091
    video_start_id: int = 200082
    video_end_id: int = 200083
    video_frame_sep_id: int = 200087

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = OnyxTextConfig()
            logger.info("text_config is None, using default OnyxTextConfig text config.")
        elif isinstance(self.text_config, dict):
            self.text_config = OnyxTextConfig(**self.text_config)

        if isinstance(self.vision_config, dict):
            self.vision_config = OnyxVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = OnyxVisionConfig()
            logger.info("vision_config is None, using default OnyxVisionConfig vision config.")

        super().__post_init__(**kwargs)


class OnyxRMSNorm(Gemma2RMSNorm):
    pass


class OnyxScalelessRMSNorm(Llama4TextL2Norm):
    pass


# Weight-as-scale, no offset — used only for the final norm.
class OnyxFinalRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


class OnyxMLP(Gemma2MLP):
    pass


class OnyxRotaryEmbedding(Gemma2RotaryEmbedding):
    pass


class OnyxAttention(Gemma2Attention):
    def __init__(self, config: OnyxTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.scaling = self.head_dim**-0.5
        self.attn_logit_softcapping = None

        self.use_rope = config.no_rope_layers[layer_idx] == 1

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.qk_norm = OnyxScalelessRMSNorm(eps=config.rms_norm_eps)
            self.scale_query_by = config.qk_scale_factor / (config.head_dim**0.5)

        self.use_output_gate = config.use_attn_output_gate
        if self.use_output_gate:
            self.output_gate_proj = nn.Linear(
                config.hidden_size, config.num_attention_heads * config.head_dim, bias=False
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.use_qk_norm:
            query_states = self.qk_norm(query_states) * self.scale_query_by
            key_states = self.qk_norm(key_states)

        if self.use_rope:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb_interleave(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

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
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        # attn_output shape here: (batch, seq, num_heads, head_dim)

        if self.use_output_gate:
            gate = torch.sigmoid(self.output_gate_proj(hidden_states).view(*attn_output.shape))
            attn_output = gate * attn_output

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class OnyxDecoderLayer(Gemma2DecoderLayer):
    pass


class OnyxVisionRotaryEmbedding(Gemma4VisionRotaryEmbedding):
    pass


class OnyxVisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]

        hidden_shape = (*input_shape, -1, self.head_dim)
        queries = self.q_proj(hidden_states).view(hidden_shape)
        keys = self.k_proj(hidden_states).view(hidden_shape)
        values = self.v_proj(hidden_states).view(hidden_shape)

        cos, sin = position_embeddings
        queries = apply_multidimensional_rope(queries, cos, sin, position_ids)
        keys = apply_multidimensional_rope(keys, cos, sin, position_ids)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class OnyxVisionMLP(nn.Module):
    def __init__(self, dim: int, hidden_size: int):
        super().__init__()
        self.c_fc = nn.Linear(dim, hidden_size, bias=True)
        self.c_proj = nn.Linear(hidden_size, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class OnyxVisionEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.attn = OnyxVisionAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size)
        self.mlp = OnyxVisionMLP(config.hidden_size, config.hidden_size * config.mlp_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, s, d = hidden_states.shape

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states.view(bs * s, d)).reshape(bs, s, d)
        hidden_states = self.attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states.view(bs * s, d)).reshape(bs, s, d)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class OnyxVisionAdapter(nn.Module):
    def __init__(self, config: OnyxConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.vision_output_dim, config.vision_adapter_dim, bias=False)
        self.c_proj = nn.Linear(config.vision_adapter_dim, config.vision_adapter_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.c_proj(F.gelu(self.c_fc(hidden_states))))


class OnyxVisionPatchEmbedder(nn.Module):
    def __init__(self, config: OnyxVisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_temporal * 3 * config.patch_size**2

        self.input_proj = nn.Linear(self.patch_size, self.hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(
            torch.zeros(config.pos_emb_grid_h * config.pos_emb_grid_w, self.hidden_size)
        )

    def _position_embeddings(self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor) -> torch.Tensor:
        """Compute 2-D patch position embeddings via embedding lookup.

        ``pixel_position_ids`` has shape ``(batch, num_patches, 2)`` where the
        last dimension holds (x, y) indices into ``position_embedding_table``
        (shape ``(2, position_embedding_size, hidden_size)``).  The result is the
        sum of the x- and y-embeddings for each patch.
        """
        clamped_positions = pixel_position_ids.clamp(min=0)
        x_emb = F.embedding(clamped_positions[..., 0], self.position_embedding_table[0])
        y_emb = F.embedding(clamped_positions[..., 1], self.position_embedding_table[1])
        position_embeddings = x_emb + y_emb
        position_embeddings = torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)
        return position_embeddings

    def forward(
        self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor
    ) -> torch.Tensor:
        # Gemma4 applies no normalization and instead scales in model code
        if (target_dtype := self.input_proj.weight.dtype).is_floating_point:
            pixel_values = pixel_values.to(target_dtype)
        hidden_states = self.input_proj(pixel_values)
        position_embeddings = self._position_embeddings(pixel_position_ids, padding_positions)
        return hidden_states + position_embeddings


class OnyxPreTrainedModel(Gemma2PreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        # Gemma2's init assumes every RMSNorm has a `weight`, but OnyxScalelessRMSNorm doesn't.
        # Route it to the base PreTrainedModel init (which handles Linear/Embedding via initializer_range)
        # and skip the RMSNorm branch for the scaleless variant.
        if isinstance(module, OnyxScalelessRMSNorm):
            return
        super()._init_weights(module)


class OnyxVisionModel(OnyxPreTrainedModel):
    def __init__(self, config: OnyxVisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.patch_temporal = config.patch_temporal
        self.downsample_factor = config.downsample_factor
        self.sparse_attention_factor = config.sparse_attention_factor
        self.pos_emb_grid_h = config.pos_emb_grid_h
        self.pos_emb_grid_w = config.pos_emb_grid_w
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.patch_embedder = OnyxVisionPatchEmbedder(config)
        self.rotary_emb = OnyxVisionRotaryEmbedding(config)
        self.ln_pre = nn.LayerNorm(config.hidden_size)
        self.layers = nn.ModuleList([OnyxVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.ln_post = nn.LayerNorm(config.hidden_size)

    def _get_pos_emb(self, grid_h: int, grid_w: int, device: torch.device) -> torch.Tensor:
        # Interpolate positional embeddings in the encoder's native dtype (bf16);
        # using float32 here causes a small numerical drift in both image and
        # video outputs.
        dtype = self.positional_embedding_vlm.dtype
        pos_emb = (
            self.positional_embedding_vlm.view(self.pos_emb_grid_h, self.pos_emb_grid_w, self.hidden_size)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        inv_h = 1.0 / grid_h
        inv_w = 1.0 / grid_w
        ys = torch.linspace(-1 + inv_h, 1 - inv_h, grid_h, device=device, dtype=dtype)
        xs = torch.linspace(-1 + inv_w, 1 - inv_w, grid_w, device=device, dtype=dtype)
        pos_xy = torch.stack(torch.meshgrid(ys, xs, indexing="xy"), dim=-1).reshape(-1, 2)[None, None]
        sampled = F.grid_sample(pos_emb, pos_xy, mode="bilinear", align_corners=False)
        return sampled[0, :, 0, :].T

    def _pixel_shuffle_downsample(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        f = self.downsample_factor
        n_out = (grid_h // f) * (grid_w // f)
        device = x.device
        ds_perm = torch.arange(grid_h * grid_w, device=device)
        ds_perm = ds_perm.view(grid_h // f, f, grid_w // f, f)
        ds_perm = ds_perm.permute(0, 2, 1, 3).reshape(-1)
        x_flat = x.squeeze(0)
        ds_x = x_flat[ds_perm]
        d = x_flat.shape[-1]
        ds_x = ds_x.view(n_out, f * f, d).permute(0, 2, 1).contiguous().view(n_out, d * f * f)
        return ds_x.unsqueeze(0)

    def _get_sparse_perm_and_mask(self, grid_h: int, grid_w: int, device: torch.device):
        gh, gw = self.pos_emb_grid_h, self.pos_emb_grid_w
        pad_h = math.ceil(grid_h / gh) * gh
        pad_w = math.ceil(grid_w / gw) * gw
        idx = torch.arange(grid_h * grid_w, device=device).view(grid_h, grid_w)
        idx = F.pad(idx, (0, pad_w - grid_w, 0, pad_h - grid_h), value=-1).flatten()
        idx = idx.view(pad_h // gh, gh, pad_w // gw, gw)
        idx = idx.permute(0, 2, 1, 3).reshape(-1)
        sp_perm = idx[idx != -1]
        valid = (idx != -1).view(-1, gh * gw)
        sp_slens = valid.sum(dim=1).tolist()
        return sp_perm, sp_slens

    def _make_block_diag_mask(self, slens: list[int], device: torch.device) -> torch.Tensor:
        total = sum(slens)
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        offset = 0
        for s in slens:
            mask[offset : offset + s, offset : offset + s] = True
            offset += s
        return mask

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        position_ids = get_vision_position_ids(grid_thw, spatial_merge_size=1)
        position_ids = position_ids.flip(0)  # seq-len, 2, should we flip?

        padding_positions = (position_ids == -1).all(dim=-1)
        inputs_embeds = self.patch_embedder(pixel_values, position_ids, padding_positions)
        hidden_states = self.ln_pre(inputs_embeds.view(-1, self.hidden_size))
        hidden_states = hidden_states.view(len(pixel_values), -1, self.hidden_size)

        sp_perm = None
        if self.sparse_attention_factor > 1:
            sp_perm, sp_slens = self._get_sparse_perm_and_mask(grid_thw)
            hidden_states = hidden_states[:, sp_perm]
            position_ids = position_ids[sp_perm]

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        global_mask = self._make_block_diag_mask() if len(pixel_values) > 1 else None
        sp_mask = self._make_block_diag_mask() if self.sparse_attention_factor else None

        for i, block in enumerate(self.layers):
            is_global = (i == len(self.layers) - 1) or ((i + 1) % self.sparse_attention_factor == 0)
            hidden_states = block(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                attention_mask=global_mask if is_global or not self.sparse_attention_factor else sp_mask,
            )

        all_features = []
        offset = 0
        for _, grid_h, grid_w in grid_thw:
            n_tokens = grid_h * grid_w
            img_x = hidden_states[:, offset : offset + n_tokens]
            offset += n_tokens

            if sp_perm is not None:
                inv_perm = torch.empty_like(sp_perm)
                inv_perm[sp_perm] = torch.arange(len(sp_perm), device=hidden_states.device)
                img_x = img_x[:, inv_perm]

            img_x = self.ln_post(img_x.view(-1, self.hidden_size)).view(1, -1, self.hidden_size)
            img_x = self._pixel_shuffle_downsample(img_x, grid_h, grid_w)
            all_features.append(img_x.squeeze(0))

        hidden_states = torch.cat(all_features, dim=0)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


class OnyxTextModel(Gemma2Model):
    config: OnyxTextConfig

    def __init__(self, config: OnyxTextConfig):
        super().__init__(config)
        # Replace Gemma2's sqrt(hidden_size)-scaled embedding with a plain one — Onyx uses norm, not scale.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # Final norm uses weight-as-scale (no offset), unlike the per-layer OnyxRMSNorm.
        self.norm = OnyxFinalRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_norm = OnyxScalelessRMSNorm(eps=config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.embed_norm(self.embed_tokens(input_ids))
            input_ids = None  # avoid double-embedding in the super().forward call

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )


class OnyxModel(Gemma3Model):
    def __init__(self, config: OnyxConfig):
        super().__init__(config)
        self.vision_adapter = OnyxVisionAdapter(config)
        self.vision_projection = nn.Linear(config.vision_adapter_dim, config.hidden_size, bias=False)
        self.perception_emb_norm = OnyxScalelessRMSNorm(config.hidden_size, config.rms_norm_eps)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values,
            **kwargs,
        )
        vision_features = self.vision_adapter(vision_outputs.last_hidden_state)
        vision_features = self.vision_projection(vision_features)
        vision_outputs.pooler_output = self.perception_emb_norm(vision_features)
        return vision_outputs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **lm_kwargs: Unpack[TransformersKwargs],
    ) -> tuple | OnyxModelOutputWithPast:
        r"""
        Example:

        ```python
        TODO
        ```"""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, return_dict=True).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config.get_text_config(),
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }

            causal_mask_mapping = create_masks_for_generate(**mask_kwargs)

        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **lm_kwargs,
        )

        return OnyxModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class OnyxForCausalLM(Gemma2ForCausalLM):
    config: OnyxTextConfig

    def __init__(self, config: OnyxTextConfig):
        super().__init__(config)
        self.model = OnyxTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # Onyx pre-scales logits by `output_multiplier` before the Gemma-style tanh softcap.
        # Together with `final_logit_softcapping = T`, this gives `T * tanh(logits * mult / T)`.
        logits = logits * self.config.output_multiplier
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class OnyxForConditionalGeneration(Gemma3ForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # Onyx pre-scales logits by `output_multiplier` before the Gemma-style tanh softcap.
        # Together with `final_logit_softcapping = T`, this gives `T * tanh(logits * mult / T)`.
        logits = logits * self.config.output_multiplier
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return OnyxCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def create_masks_for_generate(**super_kwargs):
        raise NotImplementedError("custom mask not needed for Onyx")


__all__ = [
    "OnyxTextConfig",
    "OnyxVisionConfig",
    "OnyxConfig",
    "OnyxPreTrainedModel",
    "OnyxTextModel",
    "OnyxVisionModel",
    "OnyxModel",
    "OnyxForCausalLM",
    "OnyxForConditionalGeneration",
]
