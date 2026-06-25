# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
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
import copy
from collections.abc import Callable
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging, no_inherit_decorator, torch_compilable_check, torch_int
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from .configuration_step3p7 import Step3p7Config, Step3p7TextConfig, Step3p7VisionConfig


from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts, DeepseekV4MLP
from ..siglip.modeling_siglip import SiglipVisionEmbeddings
from ..mimi.modeling_mimi import MimiLayerScale
from ..deepseek_vl.modeling_deepseek_vl import DeepseekVLModel
from ..minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration, MiniMaxM3VLAttention, MiniMaxM3VLDecoderLayer, MiniMaxM3VLRotaryEmbedding, MiniMaxM3VLDenseMLP, MiniMaxM3VLRMSNorm, MiniMaxM3VLSparseMoeBlock, MiniMaxM3VLTextModel, MiniMaxM3VLTopKRouter, MiniMaxM3VLVisionMLP, MiniMaxM3VLVisionAttention, MiniMaxM3VLVisionEncoderLayer, rotate_half, apply_rotary_pos_emb, repeat_kv, eager_attention_forward


logger = logging.get_logger(__name__)

__all__ = [
    "Step3p7Model",
]

#  Vision encoder
class Step3p7VisionRope2D(nn.Module):
    """Cacheable 2D rotary positional embedding for the vision encoder.

    Uses an interleaved-pairs frequency layout (repeat_interleave) which requires
    a pair-wise rotate_half, distinct from the block-split convention used in the
    text decoder.
    """

    def __init__(
        self,
        dim: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        theta: int | float = 10000,
        max_freq: int = 10,
        num_freqs: int = 1,
        theta_rescale_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_grid_height = max_grid_height
        self.max_grid_width = max_grid_width
        self.use_cls_token = use_cls_token
        self.theta = theta * theta_rescale_factor ** (dim / (dim - 2))
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        cache = self._compute_2d_freqs()
        self.register_buffer("freqs_cache", cache, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Pair-wise rotate: [a1,b1, a2,b2,...] → [-b1,a1, -b2,a2,...]."""
        x = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x.unbind(dim=-1)
        return torch.stack((-x2, x1), dim=-1).reshape(*x.shape[:-2], -1)

    def _compute_inv_freq(self, base: int | float, dim: int) -> torch.Tensor:
        return 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    def _compute_freqs(self, t: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
        freqs = torch.einsum("..., f -> ... f", t.type(inv_freq.dtype), inv_freq)
        return freqs.repeat_interleave(2, dim=-1)

    def _compute_2d_freqs(self) -> torch.Tensor:
        grid_h_range = torch.arange(self.max_grid_height, dtype=torch.float)
        grid_w_range = torch.arange(self.max_grid_width, dtype=torch.float)
        if self.use_cls_token:
            grid_h_range += 1
            grid_w_range += 1
        inv_freq = self._compute_inv_freq(self.theta, self.dim // 2)
        freqs_h = self._compute_freqs(grid_h_range, inv_freq)[:, None].expand(
            self.max_grid_height, self.max_grid_width, -1
        )
        freqs_w = self._compute_freqs(grid_w_range, inv_freq)[None, :].expand(
            self.max_grid_height, self.max_grid_width, -1
        )
        freqs = torch.cat([freqs_w, freqs_h], dim=-1).reshape(self.max_grid_height * self.max_grid_width, -1)
        if self.use_cls_token:
            freqs = torch.cat([torch.zeros(1, freqs.shape[-1]), freqs], dim=0)
        return freqs[None, None, ...]

    def forward(self, q: torch.Tensor, k: torch.Tensor, grid_hw: tuple[int, int]):
        if grid_hw[0] != self.max_grid_height or grid_hw[1] != self.max_grid_width:
            rows = torch.arange(grid_hw[0], device=q.device).view(-1, 1)
            cols = torch.arange(grid_hw[1], device=q.device).view(1, -1)
            positions = (rows * self.max_grid_width + cols).reshape(-1).to(torch.long)
            if self.use_cls_token:
                positions = torch.cat([torch.zeros(1, device=q.device), positions + 1], dim=0)
            freqs = self.freqs_cache.index_select(2, positions)
        else:
            freqs = self.freqs_cache
        dtype = q.dtype
        q = (q * freqs.cos() + self._rotate_half(q) * freqs.sin()).to(dtype)
        k = (k * freqs.cos() + self._rotate_half(k) * freqs.sin()).to(dtype)
        return q, k

    def get_cos_sin(
        self, grid_hw: tuple[int, int], device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) shaped (seq_len, head_dim) for the given grid, for use as position_embeddings."""
        if grid_hw[0] != self.max_grid_height or grid_hw[1] != self.max_grid_width:
            rows = torch.arange(grid_hw[0], device=device or self.freqs_cache.device).view(-1, 1)
            cols = torch.arange(grid_hw[1], device=device or self.freqs_cache.device).view(1, -1)
            positions = (rows * self.max_grid_width + cols).reshape(-1).to(torch.long)
            freqs = self.freqs_cache.index_select(2, positions)
        else:
            freqs = self.freqs_cache
        freqs = freqs.squeeze(0).squeeze(0)  # (seq_len, head_dim)
        if device is not None:
            freqs = freqs.to(device=device)
        if dtype is not None:
            freqs = freqs.to(dtype=dtype)
        return freqs.cos(), freqs.sin()


class Step3p7VisionLayerScale(MimiLayerScale):
    """Per-channel residual scaling used when ls_init_value is set."""

    def __init__(self, dim: int, init_values: float):
        nn.Module.__init__(self)
        self.scale = nn.Parameter(torch.full((dim,), init_values))


class Step3p7VisionMLP(MiniMaxM3VLVisionMLP):
    pass


class Step3p7VisionAttention(MiniMaxM3VLVisionAttention):
    def __init__(self, config: Step3p7VisionConfig):
        super().__init__(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        queries = self.q_proj(hidden_states).view(hidden_shape)
        keys = self.k_proj(hidden_states).view(hidden_shape)
        values = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        # expand (seq_len, head_dim) → (1, seq_len, 1, head_dim) for broadcasting
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        # pair-wise (interleaved) rotation — distinct from the block-split used in the text decoder
        queries = (queries * cos + Step3p7VisionRope2D._rotate_half(queries) * sin).to(queries.dtype)
        keys = (keys * cos + Step3p7VisionRope2D._rotate_half(keys) * sin).to(keys.dtype)
        queries, keys = queries.transpose(1, 2), keys.transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(queries, keys, values, scale=self.scale)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.out_proj(attn_output), None


class Step3p7VisionBlock(MiniMaxM3VLVisionEncoderLayer):
    def __init__(self, config: Step3p7VisionConfig):
        super().__init__(config)
        self.self_attn = Step3p7VisionAttention(config)
        self.mlp = Step3p7VisionMLP(config)
        self.ls_1 = Step3p7VisionLayerScale(config.hidden_size, config.ls_init_value)
        self.ls_2 = Step3p7VisionLayerScale(config.hidden_size, config.ls_init_value)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, **kwargs)
        hidden_states = residual + self.ls_1(hidden_states)
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.ls_2(hidden_states)
        return hidden_states


class Step3p7VisionEncoder(nn.Module):
    """Stack of vision encoder blocks; holds the shared 2-D RoPE."""

    def __init__(self, config: Step3p7VisionConfig):
        super().__init__()
        self.layers = nn.ModuleList([Step3p7VisionBlock(config) for _ in range(config.num_hidden_layers)])
        head_dim = config.hidden_size // config.num_attention_heads
        grid = config.image_size // config.patch_size
        self.rotary_emb = Step3p7VisionRope2D(
            dim=head_dim,
            max_grid_height=grid,
            max_grid_width=grid,
            theta=getattr(config, "rope_theta", 10000),
            max_freq=getattr(config, "rope_max_freq", 10),
            num_freqs=getattr(config, "rope_num_freqs", 1),
            theta_rescale_factor=getattr(config, "rope_theta_rescale_factor", 1.0),
        )

    def forward(self, hidden_states: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        cos, sin = self.rotary_emb.get_cos_sin(grid_hw, device=hidden_states.device, dtype=hidden_states.dtype)
        for block in self.layers:
            hidden_states = block(hidden_states, attention_mask=None, position_embeddings=(cos, sin))
        return hidden_states


class Step3p7VisionEmbeddings(SiglipVisionEmbeddings):
    def __init__(self, config: Step3p7VisionConfig):
        super().__init__(config)
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_positions = self.position_embedding.weight.shape[0]
        new_height = height // self.patch_size
        new_width = width // self.patch_size
        sqrt_num_positions = torch_int(num_positions**0.5)
        if not torch.jit.is_tracing() and new_height == sqrt_num_positions and new_width == sqrt_num_positions:
            return self.position_embedding.weight.unsqueeze(0)
        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)
        dim = embeddings.shape[-1]
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed


class Step3p7VisionModel(nn.Module):
    def __init__(self, config: Step3p7VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = Step3p7VisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = Step3p7VisionEncoder(config)
        self.downsampler = nn.Sequential(
            nn.Conv2d(config.hidden_size, config.hidden_size * 2, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(config.hidden_size * 2, config.hidden_size * 4, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        grid_h, grid_w = height // self.embeddings.patch_size, width // self.embeddings.patch_size
        hidden_state = self.embeddings(pixel_values, interpolate_pos_encoding=True)
        hidden_state = self.pre_layernorm(hidden_state)
        hidden_state = self.encoder(hidden_state, grid_hw=(grid_h, grid_w))
        batch_size, num_patches, channels = hidden_state.shape
        grid_size = int(num_patches**0.5)
        hidden_state = hidden_state.permute(0, 2, 1).view(batch_size, channels, grid_size, grid_size)
        hidden_state = self.downsampler(hidden_state)
        return hidden_state.flatten(2).permute(0, 2, 1)


# Text model
class Step3p7PreTrainedModel(PreTrainedModel):
    config_class = Step3p7Config
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _keys_to_ignore_on_load_unexpected = [
        r"model\.layers\.45\.*",
        r"model\.layers\.46\.*",
        r"model\.layers\.47\.*",
    ]
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        super()._init_weights(module)
        # issues from _move_missing_keys_from_meta_to_device, this runs after that migration step, so the correct values are restored.
        if hasattr(module, "_compute_2d_freqs") and hasattr(module, "freqs_cache"):
            cache = module._compute_2d_freqs()
            module.register_buffer("freqs_cache", cache, persistent=False)
        if isinstance(module, Step3p7VisionEmbeddings):
            module.register_buffer("position_ids", torch.arange(module.num_positions).expand((1, -1)), persistent=False)


class Step3p7RotaryEmbedding(MiniMaxM3VLRotaryEmbedding):
    def __init__(self, config: Step3p7TextConfig, device=None):
        super().__init__(config, device=device)

    @staticmethod
    def compute_default_rope_parameters(
        config: "Step3p7TextConfig | None" = None,
        device=None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor



class Step3p7RMSNorm(MiniMaxM3VLRMSNorm):
    pass


class Step3p7MLP(DeepseekV4MLP):
    def __init__(self, config, swiglu_limit=None):
        config = copy.copy(config)
        config.swiglu_limit = float("inf") if swiglu_limit is None else swiglu_limit
        super().__init__(config)

    def forward(self, x):
        # silu-then-clamp (Step3p7 order); DeepseekV4MLP clamps before act_fn
        gate = self.act_fn(self.gate_proj(x)).clamp(max=self.limit)
        up = self.up_proj(x).clamp(min=-self.limit, max=self.limit)
        return self.down_proj(gate * up)


class Step3p7SharedExpert(Step3p7MLP):
    def __init__(self, config, swiglu_limit=None):
        super().__init__(config, swiglu_limit=swiglu_limit)
        self.intermediate_size = config.share_expert_dim
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)


class Step3p7TopKRouter(MiniMaxM3VLTopKRouter):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts, dtype=torch.float32))


@no_inherit_decorator
class Step3p7Experts(DeepseekV4Experts):
    def __init__(self, config, swiglu_limit=None):
        config = copy.copy(config)
        config.intermediate_size = config.moe_intermediate_size
        config.swiglu_limit = float("inf") if swiglu_limit is None else swiglu_limit
        super().__init__(config)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        # silu-then-clamp (Step3p7 order); DeepseekV4 clamps before act_fn
        gate, up = gate_up.chunk(2, dim=-1)
        gate = self.act_fn(gate).clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return gate * up


class Step3p7SparseMoeBlock(MiniMaxM3VLSparseMoeBlock):
    def __init__(self, config, layer_idx):
        nn.Module.__init__(self)
        swiglu_limit = (config.swiglu_limits[layer_idx] or None) if config.swiglu_limits else None
        swiglu_limit_shared = (config.swiglu_limits_shared[layer_idx] or None) if config.swiglu_limits_shared else None
        self.gate = Step3p7TopKRouter(config)
        self.experts = Step3p7Experts(config, swiglu_limit=swiglu_limit)
        self.shared_experts = Step3p7SharedExpert(config, swiglu_limit=swiglu_limit_shared)
        self.routed_scaling_factor = getattr(config, "moe_router_scaling_factor", 1.0)


class Step3p7Attention(MiniMaxM3VLAttention):
    def __init__(self, config: Step3p7TextConfig, layer_idx):
        self.layer_type = config.layer_types[layer_idx]
        config = copy.copy(config)
        if self.layer_type == "sliding_attention":
            config.num_attention_heads = config.num_sliding_attention_heads
            config.rope_parameters = {"rope_type": "default", "rope_theta": config.rope_theta}
        else:
            config.rope_parameters = config.rope_scaling or {"rope_type": "default", "rope_theta": config.rope_theta}
        super().__init__(config, layer_idx)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        self.q_norm = Step3p7RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Step3p7RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.g_proj = nn.Linear(config.hidden_size, self.num_attention_heads, bias=False)
        self.indexer = None  # Step3p7 has no minimax_m3_sparse layers; prevents converter from copying MiniMaxM3VLIndexer

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        gate_states = self.g_proj(hidden_states)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = (
            attn_output.view(*attn_output.shape[:-1], self.num_attention_heads, self.head_dim)
            * gate_states.unsqueeze(-1).sigmoid()
        ).view(*attn_output.shape)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Step3p7DecoderLayer(MiniMaxM3VLDecoderLayer): #TODO: switch to llama
    def __init__(self, config, layer_idx):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Step3p7Attention(config, layer_idx)
        self.attention_type = config.layer_types[layer_idx]

        swiglu_limit_shared = (config.swiglu_limits_shared[layer_idx] or None) if config.swiglu_limits_shared else None
        self.mlp = (
            Step3p7SparseMoeBlock(config, layer_idx)
            if config.mlp_layer_types[layer_idx] == "sparse"
            else Step3p7MLP(config, swiglu_limit=swiglu_limit_shared)
        )

        self.input_layernorm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Step3p7TextModel(MiniMaxM3VLTextModel):
    _no_split_modules = ["Step3p7DecoderLayer"]
    config_class = Step3p7TextConfig
    config: Step3p7TextConfig
    _can_record_outputs = {"hidden_states": Step3p7DecoderLayer}

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Step3p7Model(DeepseekVLModel):
    config: Step3p7Config

    def __init__(self, config: Step3p7Config):
        Step3p7PreTrainedModel.__init__(self, config)
        self.vision_model = Step3p7VisionModel(config.vision_config)
        self.language_model = Step3p7TextModel(config.text_config)
        self.vocab_size = config.text_config.vocab_size
        self.multi_modal_projector = nn.Linear(
            config.vision_config.hidden_size * 4, config.text_config.hidden_size, bias=config.projector_bias
        )
        self.image_placeholder_token_id = config.image_token_id
        self.post_init()

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        patch_pixel_values: torch.Tensor | None = None,
        num_patches: list[int] | None = None,
    ) -> list[torch.Tensor]:
        if pixel_values.dim() >= 3:
            pixel_values = pixel_values.view(-1, *pixel_values.shape[-3:])
        if patch_pixel_values is not None:
            patch_pixel_values = patch_pixel_values.view(-1, *patch_pixel_values.shape[-3:])
            if patch_pixel_values.shape[0] == 0:
                patch_pixel_values = None

        image_features = self.multi_modal_projector(
            self.vision_model(pixel_values.to(self.dtype).to(self.device))
        )
        patch_image_features = (
            self.multi_modal_projector(
                self.vision_model(patch_pixel_values.to(self.dtype).to(self.device))
            )
            if patch_pixel_values is not None
            else None
        )

        if num_patches is None:
            num_patches = [0] * image_features.shape[0]

        merged = []
        cur_patch_idx = 0
        for i, num_patch in enumerate(num_patches):
            cur_feature = []
            if num_patch > 0:
                patch_slice = patch_image_features[cur_patch_idx : cur_patch_idx + num_patch]
                cur_feature.append(patch_slice.view(-1, patch_slice.shape[-1]))
            cur_feature.append(image_features[i].view(-1, image_features.shape[-1]))
            cur_patch_idx += num_patch
            merged.append(torch.cat(cur_feature) if len(cur_feature) > 1 else cur_feature[0])
        return merged

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        patch_pixel_values: torch.Tensor | None = None,
        num_patches=None,
        image_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                image_features = torch.cat(
                    self.get_image_features(pixel_values, patch_pixel_values, num_patches), dim=0
                ).to(inputs_embeds)
                image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
            elif image_embeds is not None:
                image_features = self.multi_modal_projector(image_embeds.to(self.dtype).to(self.device))
                image_features = image_features.view(-1, image_features.shape[-1]).to(inputs_embeds)
                image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
            input_ids = None

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Step3p7ForConditionalGeneration(MiniMaxM3SparseForConditionalGeneration):
    base_model_prefix = "model"
    config: Step3p7Config

    def get_image_features(self, pixel_values, patch_pixel_values=None, num_patches=None):
        return self.model.get_image_features(pixel_values, patch_pixel_values, num_patches)

    def get_video_features(self, *args, **kwargs):
        raise NotImplementedError("Step3p7 does not support video.")

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.Tensor | None = None,
        patch_pixel_values=None,
        num_patches=None,
        image_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            patch_pixel_values=patch_pixel_values,
            num_patches=num_patches,
            image_embeds=image_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        patch_pixel_values=None,
        num_patches=None,
        image_embeds=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if is_first_iteration or not kwargs.get("use_cache", True):
            model_inputs["pixel_values"] = pixel_values
            model_inputs["patch_pixel_values"] = patch_pixel_values
            model_inputs["num_patches"] = num_patches
            model_inputs["image_embeds"] = image_embeds

        return model_inputs

