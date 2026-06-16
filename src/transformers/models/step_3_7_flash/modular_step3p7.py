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
import inspect
from collections.abc import Callable
from dataclasses import dataclass
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
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple, logging

from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3MLP, DeepseekV3MoE
from .configuration_step3p7 import Step3p7Config, Step3p7TextConfig, StepRoboticsVisionEncoderConfig
from ..mimi.modeling_mimi import MimiLayerScale, MimiMLP
from ..minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLAttention, MiniMaxM3VLModel, MiniMaxM3VLTextModel, MiniMaxM3VLDecoderLayer, MiniMaxM3VLRotaryEmbedding, MiniMaxM3VLCausalLMOutputWithPast, MiniMaxM3VLDenseMLP, MiniMaxM3VLRMSNorm, rotate_half, apply_rotary_pos_emb, repeat_kv, eager_attention_forward


logger = logging.get_logger(__name__)
_MASK_INPUT_EMBEDS_ARG = (
    "inputs_embeds" if "inputs_embeds" in inspect.signature(create_causal_mask).parameters else "input_embeds"
)

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


class Step3p7VisionLayerScale(MimiLayerScale):
    """Per-channel residual scaling used when ls_init_value is set."""

    def __init__(self, dim: int, init_values: float):
        nn.Module.__init__(self)
        self.scale = nn.Parameter(torch.full((dim,), init_values))


class Step3p7VisionMLP(MimiMLP):
    """Feed-forward network used inside each vision transformer block."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        nn.Module.__init__(self)
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)


class Step3p7VisionAttention(nn.Module):
    """Multi-head self attention with optional 2D RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
        rope_theta: int | float = 10000,
        rope_max_freq: int = 10,
        rope_num_freqs: int = 1,
        rope_theta_rescale_factor: float = 1.0,
        rope_freqs_for: Literal["lang", "pixel", "constant"] = "lang",
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.in_proj_weight = nn.Parameter(torch.zeros(hidden_size * 3, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.zeros(hidden_size * 3))
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.rope = None
        if use_rope2d:
            self.rope = Step3p7VisionRope2D(
                dim=self.head_dim,
                max_grid_height=max_grid_height,
                max_grid_width=max_grid_width,
                use_cls_token=use_cls_token,
                theta=rope_theta,
                max_freq=rope_max_freq,
                num_freqs=rope_num_freqs,
                theta_rescale_factor=rope_theta_rescale_factor,
            )

    def forward(self, hidden_states: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        qkv = F.linear(hidden_states, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, k, grid_hw=grid_hw)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.scale)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(attn_output)


class Step3p7VisionBlock(nn.Module):
    """A single Vision Transformer block (self-attention + MLP)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        hidden_act: str,
        layer_norm_eps: float,
        ls_init_value: float | None = None,
        max_grid_height: int | None = None,
        max_grid_width: int | None = None,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
        rope_kwargs: dict | None = None,
    ):
        super().__init__()
        rope_kwargs = rope_kwargs or {}
        self.attn = Step3p7VisionAttention(
            hidden_size,
            num_heads,
            max_grid_height=max_grid_height,
            max_grid_width=max_grid_width,
            use_cls_token=use_cls_token,
            use_rope2d=use_rope2d,
            **rope_kwargs,
        )
        self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        intermediate = int(hidden_size * mlp_ratio)
        self.mlp = Step3p7VisionMLP(hidden_size, intermediate, hidden_act)
        self.ls_1 = Step3p7VisionLayerScale(hidden_size, ls_init_value)
        self.ls_2 = Step3p7VisionLayerScale(hidden_size, ls_init_value)

    def forward(self, hidden_states: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(hidden_states, grid_hw=grid_hw)
        hidden_states = residual + self.ls_1(hidden_states)
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.ls_2(hidden_states)
        return hidden_states


class EncoderVisionTransformer(nn.Module):
    """Stack of vision encoder blocks."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        hidden_act: str,
        layer_norm_eps: float,
        ls_init_value: float | None = None,
        max_grid_height: int | None = None,
        max_grid_width: int | None = None,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
        rope_kwargs: dict | None = None,
    ):
        super().__init__()
        self.layers = depth
        rope_kwargs = rope_kwargs or {}
        self.resblocks = nn.ModuleList(
            [
                Step3p7VisionBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    hidden_act,
                    layer_norm_eps,
                    max_grid_height=max_grid_height,
                    max_grid_width=max_grid_width,
                    use_cls_token=use_cls_token,
                    use_rope2d=use_rope2d,
                    ls_init_value=ls_init_value,
                    rope_kwargs=rope_kwargs,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, hidden_states: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        for block in self.resblocks:
            hidden_states = block(hidden_states, grid_hw=grid_hw)
        return hidden_states


class StepRoboticsVisionEncoder(nn.Module):
    def __init__(self, config: StepRoboticsVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.width
        self.num_heads = config.heads
        self.num_hidden_layers = config.layers
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.use_cls_token = getattr(config, "use_cls_token", False)
        self.use_rope2d = getattr(config, "use_rope2d", True)
        self.use_abs_posemb = getattr(config, "use_abs_posemb", True)
        self.layer_norm_eps = config.layer_norm_eps
        self.mlp_ratio = getattr(config, "mlp_ratio", 8960 / 1536)
        self.ls_init_value = getattr(config, "ls_init_value", None)
        self.hidden_act = config.hidden_act
        self.use_ln_pre = getattr(config, "use_ln_pre", False)
        self.use_ln_post = getattr(config, "use_ln_post", True)

        self.conv1 = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.ln_pre = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps) if self.use_ln_pre else nn.Identity()
        self.ln_post = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps) if self.use_ln_post else nn.Identity()

        grid_size = self.image_size // self.patch_size
        self.base_grid = (grid_size, grid_size)

        if self.use_cls_token:
            self.class_embedding = nn.Parameter(torch.randn(self.hidden_size) * (self.hidden_size**-0.5))
        else:
            self.class_embedding = None

        if self.use_abs_posemb:
            self.posemb_grid_size = self.image_size // self.patch_size
            self.positional_embedding = nn.Parameter(
                (self.hidden_size**-0.5)
                * torch.randn(
                    int(self.use_cls_token) + self.posemb_grid_size**2,
                    self.hidden_size,
                )
            )

        self.transformer = EncoderVisionTransformer(
            embed_dim=self.hidden_size,
            depth=self.num_hidden_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            ls_init_value=self.ls_init_value,
            max_grid_height=self.base_grid[0],
            max_grid_width=self.base_grid[1],
            use_cls_token=self.use_cls_token,
            use_rope2d=self.use_rope2d,
            rope_kwargs={
                "rope_theta": getattr(config, "rope_theta", 10000),
                "rope_max_freq": getattr(config, "rope_max_freq", 10),
                "rope_num_freqs": getattr(config, "rope_num_freqs", 1),
                "rope_theta_rescale_factor": getattr(config, "rope_theta_rescale_factor", 1.0),
                "rope_freqs_for": getattr(config, "rope_freqs_for", "lang"),
            },
        )
        self.vit_downsampler1 = nn.Conv2d(self.hidden_size, self.hidden_size * 2, kernel_size=3, stride=2, padding=1)
        self.vit_downsampler2 = nn.Conv2d(
            self.hidden_size * 2, self.hidden_size * 4, kernel_size=3, stride=2, padding=1
        )

    def sample_abs_posemb(self, grid_h: int, grid_w: int):
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]
        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]
        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1).permute(0, 3, 1, 2).contiguous()
        )
        pos_embed = F.interpolate(pos_embed, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.hidden_size)
        if self.use_cls_token:
            pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)
        return pos_embed[None, ...]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = pixel_values.shape
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        hidden_state = self.conv1(pixel_values)
        hidden_state = hidden_state.flatten(2).transpose(1, 2)
        if self.use_cls_token:
            cls_token = self.class_embedding.view(1, 1, -1).expand(bsz, -1, -1)
            hidden_state = torch.cat([cls_token, hidden_state], dim=1)
        if self.use_abs_posemb:
            pos_emb = self.sample_abs_posemb(grid_h, grid_w)
            hidden_state = hidden_state + pos_emb
        hidden_state = self.ln_pre(hidden_state)
        hidden_state = self.transformer(hidden_state, grid_hw=(grid_h, grid_w))
        if self.use_ln_post:
            hidden_state = self.ln_post(hidden_state)
        if self.use_cls_token:
            hidden_state = hidden_state[:, 1:, :]
        return hidden_state


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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        key_mapping = getattr(cls, "_checkpoint_conversion_mapping", None)
        if key_mapping is not None and kwargs.get("key_mapping") is None:
            kwargs["key_mapping"] = copy.deepcopy(key_mapping)
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class Step3p7RotaryEmbedding(MiniMaxM3VLRotaryEmbedding):
    def __init__(self, config: Step3p7TextConfig, device=None, layer_idx=None):
        # Always copy to prevent shared-config mutations from later layers corrupting this
        # layer's stored rope_parameters (Step3p7Attention mutates config.rope_parameters in-place).
        config = copy.copy(config)
        if getattr(config, "rope_parameters", None) is None:
            config.rope_parameters = {"rope_type": "default", "rope_theta": config.rope_theta}
        super().__init__(config, device=device)



@dataclass
class Step3p7CausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    loss: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


class Step3p7RMSNorm(MiniMaxM3VLRMSNorm):
    pass


class Step3p7MLP(DeepseekV3MLP):
    def __init__(self, config, intermediate_size=None, swiglu_limit=None):
        super().__init__(config, intermediate_size=intermediate_size)
        self.act_fn = ACT2FN["silu"]
        self.limit = swiglu_limit

    def forward(self, x):
        up = self.up_proj(x)
        gate = self.act_fn(self.gate_proj(x))
        if self.limit is not None:
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)

        return self.down_proj(gate * up)


def sigmoid_routing_function(gating_output: torch.Tensor, topk: int, renormalize: bool):
    gating_output = gating_output.float()
    gate_prob = torch.sigmoid(gating_output)
    gate_prob = gate_prob / gate_prob.sum(dim=-1, keepdim=True)
    topk_prob, indices = torch.topk(gate_prob, k=topk, dim=1)
    expert_topk_weight = topk_prob
    if renormalize:
        expert_topk_weight = expert_topk_weight / torch.sum(expert_topk_weight, dim=-1, keepdim=True)
    return expert_topk_weight, indices


class MoELinear(nn.Module):
    def __init__(self, num_experts, in_features, out_features):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))

    def forward(self, x, expert_id):
        x = F.linear(x.float(), self.weight[expert_id].float())
        return x


class Step3p7MoEMLP(DeepseekV3MoE):
    def __init__(self, config, swiglu_limit=None):
        nn.Module.__init__(self)
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        self.use_moe_router_bias = config.use_moe_router_bias
        if self.use_moe_router_bias:
            self.router_bias = nn.Parameter(
                torch.zeros(config.moe_num_experts, dtype=torch.float32), requires_grad=False
            )
            self.custom_routing_function = self.router_bias_func
        elif config.moe_router_activation == "sigmoid":
            self.custom_routing_function = sigmoid_routing_function
        else:
            self.custom_routing_function = None
        self.need_fp32_gate = config.need_fp32_gate
        self.routed_scaling_factor = getattr(config, "moe_router_scaling_factor", 1.0)

        # gating
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        self.act_fn = ACT2FN["silu"]
        self.limit = swiglu_limit

        self.up_proj = MoELinear(self.num_experts, self.hidden_size, self.moe_intermediate_size)
        self.gate_proj = MoELinear(self.num_experts, self.hidden_size, self.moe_intermediate_size)
        self.down_proj = MoELinear(self.num_experts, self.moe_intermediate_size, self.hidden_size)

    def router_bias_func(self, gating_output: torch.Tensor, topk: int, renormalize: bool):
        gate_prob = torch.sigmoid(gating_output.float())
        gate_prob_with_bias = gate_prob + self.router_bias.unsqueeze(0)
        _, indices = torch.topk(gate_prob_with_bias, k=topk, dim=1)
        topk_prob = torch.gather(gate_prob, 1, indices)
        expert_topk_weight = topk_prob
        if renormalize:
            expert_topk_weight = expert_topk_weight / (torch.sum(expert_topk_weight, dim=-1, keepdim=True) + 1e-20)
        return expert_topk_weight, indices

    def get_expert_output(self, inputs: torch.Tensor, expert_id):
        up = self.up_proj(inputs, expert_id)
        gate = self.act_fn(self.gate_proj(inputs, expert_id))
        if self.limit is not None:
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)

        return self.down_proj(gate * up, expert_id)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.need_fp32_gate:
            router_logits = torch.matmul(
                hidden_states.to(torch.float32),
                self.gate.weight.t().to(torch.float32),
            )
        else:
            router_logits = self.gate(hidden_states)

        if self.custom_routing_function:
            routing_weights, selected_experts = self.custom_routing_function(
                router_logits, self.top_k, renormalize=True
            )
        else:
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        routing_weights = routing_weights * self.routed_scaling_factor

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                self.get_expert_output(current_state, expert_idx) * routing_weights[top_x, idx, None]
            )

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class Step3p7Attention(MiniMaxM3VLAttention):
    def __init__(self, config: Step3p7TextConfig, layer_idx):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_groups

        layer_types = getattr(config, "layer_types", [])
        if layer_types:
            enable_sliding_window = layer_types[self.layer_idx] == "sliding_attention"
        else:
            enable_sliding_window = self.layer_idx % 2 == 0

        yarn_only_types = getattr(config, "yarn_only_types", None)
        if yarn_only_types and layer_types[self.layer_idx] not in yarn_only_types:
            config.rope_parameters = None
        else:
            config.rope_parameters = getattr(config, "rope_scaling", None)

        self.sliding_window = config.sliding_window
        if enable_sliding_window:
            self.num_attention_heads = config.attention_other_setting["num_attention_heads"]
            self.num_key_value_heads = config.attention_other_setting["num_attention_groups"]

        if self.sliding_window is not None and enable_sliding_window:
            self.sliding_window = self.sliding_window
        else:
            self.sliding_window = None
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.rotary_emb = Step3p7RotaryEmbedding(config, layer_idx=layer_idx)

        self.q_size = self.num_attention_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.q_norm = Step3p7RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Step3p7RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.use_head_wise_attn_gate = config.use_head_wise_attn_gate
        if self.use_head_wise_attn_gate:
            self.g_proj = nn.Linear(config.hidden_size, self.num_attention_heads, bias=False)

        self.use_rope = True
        use_rope_layers = getattr(config, "use_rope_layers", None)
        if use_rope_layers:
            self.use_rope = use_rope_layers[self.layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_value: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        if self.use_head_wise_attn_gate:
            gate_states = self.g_proj(hidden_states)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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
        if self.use_head_wise_attn_gate:
            output = (
                attn_output.view(*attn_output.shape[:-1], self.num_attention_heads, self.head_dim)
                * gate_states.unsqueeze(-1).sigmoid()
            )
            attn_output = output.view(*attn_output.shape)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Step3p7DecoderLayer(MiniMaxM3VLDecoderLayer):
    def __init__(self, config, layer_idx):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Step3p7Attention(config, layer_idx)
        layer_types = getattr(config, "layer_types", None) or []
        if layer_types:
            self.attention_type = layer_types[layer_idx]
        else:
            self.attention_type = "sliding_attention" if layer_idx % 2 == 0 else "full_attention"

        moe_layers_enum = getattr(config, "moe_layers_enum", None)
        if moe_layers_enum is not None:
            if isinstance(moe_layers_enum, str):
                moe_layers_idx = [int(i) for i in moe_layers_enum.split(",") if i.strip()]
            else:
                moe_layers_idx = [int(i) for i in moe_layers_enum]
        else:
            moe_layers_idx = [i for i in range(1, config.num_hidden_layers)]
        self.is_moe_layer = layer_idx in moe_layers_idx
        self.use_moe = False

        if (
            config.swiglu_limits_shared
            and config.swiglu_limits_shared[layer_idx] is not None
            and config.swiglu_limits_shared[layer_idx] != 0
        ):
            swiglu_limit_shared = config.swiglu_limits_shared[layer_idx]
        else:
            swiglu_limit_shared = None
        if (
            config.swiglu_limits
            and config.swiglu_limits[layer_idx] is not None
            and config.swiglu_limits[layer_idx] != 0
        ):
            swiglu_limit = config.swiglu_limits[layer_idx]
        else:
            swiglu_limit = None
        if self.is_moe_layer:
            self.moe = Step3p7MoEMLP(config, swiglu_limit=swiglu_limit)
            self.share_expert = Step3p7MLP(
                config, intermediate_size=config.share_expert_dim, swiglu_limit=swiglu_limit_shared
            )
            self.use_moe = True
        else:
            self.mlp = Step3p7MLP(config, intermediate_size=config.intermediate_size, swiglu_limit=swiglu_limit_shared)

        self.input_layernorm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.use_moe:
            share_output = self.share_expert(hidden_states)
            moe_output = self.moe(hidden_states)
            ffn_output = moe_output + share_output
        else:
            ffn_output = self.mlp(hidden_states)
        if isinstance(ffn_output, tuple):
            hidden_states, _ = ffn_output
        else:
            hidden_states = ffn_output

        hidden_states = residual + hidden_states
        return hidden_states


class Step3p7TextPreTrainedModel(Step3p7PreTrainedModel):
    config_class = Step3p7TextConfig


class Step3p7TextModel(Step3p7TextPreTrainedModel):
    _no_split_modules = ["Step3p7DecoderLayer"]
    base_model_prefix = "model"
    config: Step3p7TextConfig

    def __init__(self, config: Step3p7TextConfig):
        Step3p7TextPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Step3p7DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        layer_types = self.config.layer_types or []
        self.has_sliding_layers = not layer_types or "sliding_attention" in layer_types

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device))

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            mask_kwargs[_MASK_INPUT_EMBEDS_ARG] = inputs_embeds
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }

            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Step3p7Model(Step3p7PreTrainedModel):
    config: Step3p7Config
    base_model_prefix = ""

    def __init__(self, config: Step3p7Config):
        Step3p7PreTrainedModel.__init__(self, config)
        self.vision_model = StepRoboticsVisionEncoder(config.vision_config)
        self.language_model = Step3p7TextModel(config.text_config)
        self.vocab_size = config.text_config.vocab_size
        self.vit_large_projector = nn.Linear(
            config.vision_config.width * 4, config.text_config.hidden_size, bias=config.projector_bias
        )
        self.image_placeholder_token_id = config.image_token_id

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def _compute_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        input_ids = input_ids.squeeze(0)
        embed = self.language_model.get_input_embeddings()
        if multimodal_embeddings is None:
            inputs_embeds = embed(input_ids)
        else:
            is_text = input_ids != self.config.image_token_id
            text_embeds = embed(input_ids[is_text])
            inputs_embeds = torch.empty(
                input_ids.shape[0], text_embeds.shape[-1], dtype=text_embeds.dtype, device=text_embeds.device
            )
            inputs_embeds[is_text] = text_embeds
            image_mask = ~is_text
            inputs_embeds[image_mask] = torch.cat(
                [e.reshape(-1, e.shape[-1]) for e in multimodal_embeddings]
            ).to(text_embeds)
        inputs_embeds = inputs_embeds.unsqueeze(0)
        return inputs_embeds

    def _process_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        B, P = image_features.shape[:2]
        HW = int(P**0.5)
        image_features = image_features.permute(0, 2, 1).view(B, -1, HW, HW)
        image_features = self.vision_model.vit_downsampler1(image_features)
        image_features = self.vision_model.vit_downsampler2(image_features)

        B, C, HW, HW = image_features.shape
        image_features = image_features.view(B, -1, HW * HW).permute(0, 2, 1)
        image_features = self.vit_large_projector(image_features)
        return image_features

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

        image_features = self._process_image_features(
            self.vision_model(pixel_values.to(self.dtype).to(self.device))
        )
        patch_image_features = (
            self._process_image_features(
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
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Step3p7CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            if pixel_values is not None:
                image_features = self.get_image_features(pixel_values, patch_pixel_values, num_patches)
            elif image_embeds is not None:
                if image_embeds.dim() < 2:
                    raise ValueError(f"Unexpected shape for image_embeds: {image_embeds.shape}")
                processed = self._process_image_features(image_embeds.to(self.dtype).to(self.device))
                image_features = [processed[i].view(-1, processed.shape[-1]) for i in range(processed.shape[0])]
            else:
                image_features = None
            inputs_embeds = self._compute_inputs_embeds(input_ids, image_features)
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

        output = Step3p7CausalLMOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            attentions=outputs.attentions,
        )
        return output if return_dict else output.to_tuple()


class Step3p7ForConditionalGeneration(Step3p7PreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^vision_model": "model.vision_model",
        r"^model(?!\.(language_model|vision_model))": "model.language_model",
        "^vit_large_projector": "model.vit_large_projector",
        ".ls_1.gamma": ".ls_1.scale",
        ".ls_2.gamma": ".ls_2.scale",
        ".mlp.c_fc.": ".mlp.fc1.",
        ".mlp.c_proj.": ".mlp.fc2.",
    }
    _tied_weights_keys = ["lm_head.weight"]
    config: Step3p7Config

    def __init__(self, config: Step3p7Config):
        Step3p7PreTrainedModel.__init__(self, config)
        self.model = Step3p7Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.vision_model

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
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Step3p7CausalLMOutputWithPast:
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
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return Step3p7CausalLMOutputWithPast(
            logits=logits,
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

        generation_cache_position = model_inputs.get("cache_position", cache_position)
        is_prefill = past_key_values is None
        if generation_cache_position is not None and generation_cache_position.numel() > 0:
            is_prefill = generation_cache_position[0].item() == 0

        if is_prefill:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["patch_pixel_values"] = patch_pixel_values
            model_inputs["num_patches"] = num_patches
            model_inputs["image_embeds"] = image_embeds

        return model_inputs

    def _fix_state_dict_key_on_load(self, key: str) -> tuple[str, bool]:
        if key.startswith("language_model."):
            return key[len("language_model.") :], True

        return key, False
