# Copyright 2026 Upstage AI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch SolarOpen2 model."""

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.import_utils import is_flash_linear_attention_available
from ..glm4_moe.modeling_glm4_moe import (
    Glm4MoeAttention,
    Glm4MoeDecoderLayer,
    Glm4MoeMLP,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..qwen3_next.modeling_qwen3_next import (
    Qwen3NextModel,
    apply_mask_to_padding_states,
    causal_conv1d_update,
    l2norm,
)
from ..solar_open.modeling_solar_open import (
    SolarOpenForCausalLM,
    SolarOpenMoE,
    SolarOpenPreTrainedModel,
    SolarOpenRMSNorm,
    SolarOpenRotaryEmbedding,
)


logger = logging.get_logger(__name__)


# the KDA kernels and the bounded-gate signature require fla-core >= 0.5.0
if is_flash_linear_attention_available("0.5.0"):
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
else:
    chunk_kda, fused_recurrent_kda, fused_kda_gate = None, None, None


@auto_docstring(checkpoint="upstage/Solar-Open2-250B")
@strict
class SolarOpen2Config(PreTrainedConfig):
    r"""
    routed_scaling_factor (`float`, *optional*, defaults to 1.0):
        Scaling factor applied to the routed-expert contribution.
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts (grouped top-k routing).
    topk_group (`int`, *optional*, defaults to 1):
        Number of groups selected per token for grouped top-k routing.
    norm_topk_prob (`bool`, *optional*, defaults to `True`):
        Whether to normalize the top-k routing probabilities.
    first_k_dense_replace (`int`, *optional*, defaults to 0):
        Number of leading dense (non-MoE) decoder layers.
    layer_types (`list[str]`, *optional*):
        Per-layer attention type (`"full_attention"` / `"linear_attention"`). Derived from `gqa_layers` /
        `gqa_interval` when not given, and takes priority over both when given.
    use_rope (`bool`, *optional*, defaults to `False`):
        Whether the grouped-query full-attention layers apply rotary position embeddings. When `False` the model
        is NoPE (positional information is carried by the linear-attention layers).
    use_qk_norm (`bool`, *optional*, defaults to `False`):
        Whether to apply per-head RMS normalization to the query and key projections of the full-attention layers.
    use_gqa_gate (`bool`, *optional*, defaults to `True`):
        Whether the full-attention layers apply an output sigmoid gate (`attn_output * sigmoid(g_proj(x))`).
    use_gqa_gate_bias (`bool`, *optional*, defaults to `False`):
        Whether the output gate projection uses a bias.
    gqa_interval (`int`, *optional*, defaults to 3):
        Number of Kimi-Delta linear-attention layers between two consecutive full-attention layers, used when
        `gqa_layers` is `None`: layer `i` is a full-attention layer when `i % (gqa_interval + 1) == 0` (one
        full-attention layer every `gqa_interval + 1` layers, starting at layer 0). The default reproduces the
        [upstage/Solar-Open2-250B](https://huggingface.co/upstage/Solar-Open2-250B) layout
        (`range(0, 48, 4)`).
    gqa_layers (`list[int]`, *optional*):
        Explicit list of layer indices that use full attention; all other layers use Kimi-Delta linear
        attention. Takes priority over `gqa_interval` when given.
    linear_attn_config (`dict`, *optional*):
        Configuration of the Kimi-Delta-Attention layers, with keys `short_conv_kernel_size`, `head_dim`,
        `num_heads` and (optionally) `num_kv_heads`.
    kda_use_full_proj (`bool`, *optional*, defaults to `False`):
        When `True`, the KDA gate/output-gate use single full-rank projections (`f_proj`, `g_proj`) instead of the
        factored low-rank `f_a_proj`/`f_b_proj` and `g_a_proj`/`g_b_proj`.
    kda_allow_neg_eigval (`bool`, *optional*, defaults to `True`):
        When `True`, the KDA `beta` gate is scaled by 2 (allowing negative eigenvalues of the state transition).
    """

    model_type = "solar_open2"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Attention is left unsharded: KDA and GQA layers share the `self_attn.` path, and the KDA conv/state ops
    # assume unsharded heads.
    base_model_tp_plan = {
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    attribute_map = {"num_local_experts": "n_routed_experts"}

    vocab_size: int = 196608
    hidden_size: int = 4096
    num_hidden_layers: int = 48
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 1_048_576
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    moe_intermediate_size: int = 1280
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    n_routed_experts: int = 320
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    pad_token_id: int | None = 2
    default_theta = 10_000.0  # rope_theta fallback when user-provided `rope_parameters` omits it
    head_dim: int = 128
    intermediate_size: int = 10240
    first_k_dense_replace: int = 0
    layer_types: list[str] | None = None
    use_rope: bool = False
    use_qk_norm: bool = False
    use_gqa_gate: bool = True
    use_gqa_gate_bias: bool = False
    gqa_interval: int = 3
    gqa_layers: tuple[int, ...] | list[int] | None = None
    linear_attn_config: dict | None = None
    kda_use_full_proj: bool = False
    kda_allow_neg_eigval: bool = True

    def __post_init__(self, **kwargs):
        # Fold the legacy `rope_theta` / `rope_scaling` checkpoint keys into `rope_parameters`.
        rope_theta = kwargs.pop("rope_theta", None)
        rope_scaling = kwargs.pop("rope_scaling", None)
        if self.rope_parameters is None:
            rope_parameters = dict(rope_scaling) if rope_scaling else {"rope_type": "default"}
            if "type" in rope_parameters and "rope_type" not in rope_parameters:
                rope_parameters["rope_type"] = rope_parameters.pop("type")
            rope_parameters.setdefault("rope_type", "default")
            rope_parameters.setdefault("rope_theta", rope_theta if rope_theta is not None else 10000.0)
            self.rope_parameters = rope_parameters

        # NoPE by default; keep the full rotary dim available when `use_rope=True`.
        kwargs.setdefault("partial_rotary_factor", 1.0)

        if self.linear_attn_config is None:
            self.linear_attn_config = {
                "short_conv_kernel_size": 4,
                "head_dim": self.head_dim,
                "num_heads": self.num_attention_heads,
                "num_kv_heads": None,
            }

        num_kv_heads = self.linear_attn_config.get("num_kv_heads") or self.linear_attn_config["num_heads"]
        if self.linear_attn_config["num_heads"] % num_kv_heads != 0:
            raise ValueError(
                f"`linear_attn_config['num_heads']` ({self.linear_attn_config['num_heads']}) must be divisible by "
                f"`linear_attn_config['num_kv_heads']` ({num_kv_heads})."
            )

        # Derive the per-layer attention pattern consumed by the model and the hybrid cache.
        if self.gqa_interval < 1:
            raise ValueError(f"`gqa_interval` must be a positive integer, got {self.gqa_interval}.")
        if self.gqa_layers is not None:
            self.gqa_layers = list(self.gqa_layers)
            if any(not 0 <= i < self.num_hidden_layers for i in self.gqa_layers):
                raise ValueError(
                    f"`gqa_layers` entries must be valid layer indices in [0, {self.num_hidden_layers}), "
                    f"got {self.gqa_layers}."
                )
        if self.layer_types is None:
            if self.gqa_layers is not None:
                full = set(self.gqa_layers)
                self.layer_types = [
                    "full_attention" if i in full else "linear_attention" for i in range(self.num_hidden_layers)
                ]
            else:
                period = self.gqa_interval + 1
                self.layer_types = [
                    "full_attention" if i % period == 0 else "linear_attention" for i in range(self.num_hidden_layers)
                ]
        if "full_attention" not in self.layer_types:
            raise ValueError(
                "SolarOpen2 requires at least one full-attention layer; check `gqa_layers` / `gqa_interval`."
            )

        super().__post_init__(**kwargs)
        self.validate_layer_type()


class SolarOpen2RMSNorm(SolarOpenRMSNorm):
    pass


class SolarOpen2RMSNormGated(nn.Module):
    """KDA output norm, ``RMSNorm(x) * sigmoid(gate)``: `fla.modules.FusedRMSNormGated` with sigmoid activation."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states * self.weight.to(torch.float32)
        hidden_states = hidden_states * torch.sigmoid(gate.to(torch.float32))
        return hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def torch_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch reference for `fla.ops.kda.gate.fused_kda_gate`."""
    num_heads = A_log.numel()
    head_dim = dt_bias.numel() // num_heads
    g = g.to(torch.float32)
    g = g + dt_bias.to(torch.float32).reshape(num_heads, head_dim)
    A = A_log.to(torch.float32).reshape(num_heads, 1).exp()
    return -A * F.softplus(g)


def torch_recurrent_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Pure-PyTorch reference for the KDA delta rule: per-(head, dim) vector decay, unlike Qwen3-Next's per-head."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, g = (x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, g))
    beta = beta.transpose(1, 2).contiguous().to(torch.float32)

    batch_size, num_heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (k_head_dim**0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, seq_len, v_head_dim, dtype=value.dtype, device=value.device)
    state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(seq_len):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        # vector gate: each key-dim row of the state decays by exp(g[k])
        g_t = g[:, :, i].exp().unsqueeze(-1)  # (batch, num_heads, k_head_dim, 1)
        beta_t = beta[:, :, i].unsqueeze(-1)  # (batch, num_heads, 1)

        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # (batch, num_heads, v_head_dim)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, state


class SolarOpen2LinearAttention(nn.Module):
    """Kimi-Delta-Attention linear-attention layer (gated delta rule with a per-(head, dim) decay gate)."""

    def __init__(self, config: SolarOpen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.num_kv_heads = config.linear_attn_config.get("num_kv_heads", None) or self.num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.use_full_proj = config.kda_use_full_proj
        self.allow_neg_eigval = config.kda_allow_neg_eigval

        projection_size = self.head_dim * self.num_heads
        kv_projection_size = self.head_dim * self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, kv_projection_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, kv_projection_size, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.q_conv1d = nn.Conv1d(
            projection_size,
            projection_size,
            self.conv_size,
            groups=projection_size,
            padding=self.conv_size - 1,
            bias=False,
        )
        self.k_conv1d = nn.Conv1d(
            kv_projection_size,
            kv_projection_size,
            self.conv_size,
            groups=kv_projection_size,
            padding=self.conv_size - 1,
            bias=False,
        )
        self.v_conv1d = nn.Conv1d(
            kv_projection_size,
            kv_projection_size,
            self.conv_size,
            groups=kv_projection_size,
            padding=self.conv_size - 1,
            bias=False,
        )

        if self.use_full_proj:
            self.f_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
            self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        else:
            self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
            self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.dt_bias = nn.Parameter(torch.empty(projection_size))
        self.A_log = nn.Parameter(torch.empty(1, 1, self.num_heads, 1))

        self.o_norm = SolarOpen2RMSNormGated(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def _conv_weight(self) -> torch.Tensor:
        # concatenated so q/k/v share a single grouped conv and conv-state
        return torch.cat([self.q_conv1d.weight, self.k_conv1d.weight, self.v_conv1d.weight], dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, None]:
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = past_key_values is not None and past_key_values.has_previous_state(self.layer_idx)
        conv_state = recurrent_state = None
        if use_precomputed_states:
            conv_state = past_key_values.layers[self.layer_idx].conv_states[0]
            recurrent_state = past_key_values.layers[self.layer_idx].recurrent_states[0]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        mixed = torch.cat((q, k, v), dim=-1).transpose(1, 2)  # (batch, q_dim + 2 * kv_dim, seq_len)
        conv_weight = self._conv_weight()

        if use_precomputed_states and seq_len == 1:
            mixed = causal_conv1d_update(mixed, conv_state, conv_weight.squeeze(1), None, "silu")
        else:
            if use_precomputed_states:
                mixed = torch.cat([conv_state, mixed], dim=-1)
            if past_key_values is not None:
                new_conv_state = F.pad(mixed, (self.conv_size - mixed.shape[-1], 0))
                past_key_values.update_conv_state(new_conv_state, self.layer_idx)
            mixed = F.silu(
                F.conv1d(mixed, conv_weight, bias=None, padding=self.conv_size - 1, groups=conv_weight.shape[0])[
                    ..., : mixed.shape[-1]
                ]
            )
            if use_precomputed_states:
                mixed = mixed[..., -seq_len:]

        mixed = mixed.transpose(1, 2)
        q, k, v = torch.split(
            mixed, [self.q_proj.out_features, self.k_proj.out_features, self.v_proj.out_features], dim=-1
        )
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, -1, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, -1, self.num_kv_heads, self.head_dim)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        g_raw = self.f_proj(hidden_states) if self.use_full_proj else self.f_b_proj(self.f_a_proj(hidden_states))
        g_raw = g_raw.reshape(batch_size, -1, self.num_heads, self.head_dim)
        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        if chunk_kda is not None:
            g = fused_kda_gate(g_raw, self.A_log, dt_bias=self.dt_bias)
            kda_kernel = fused_recurrent_kda if (use_precomputed_states and seq_len == 1) else chunk_kda
            core_attn_out, recurrent_state = kda_kernel(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=past_key_values is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            g = torch_kda_gate(g_raw, self.A_log, self.dt_bias)
            core_attn_out, recurrent_state = torch_recurrent_kda(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=past_key_values is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if past_key_values is not None:
            past_key_values.update_recurrent_state(recurrent_state, self.layer_idx)

        g_out = self.g_proj(hidden_states) if self.use_full_proj else self.g_b_proj(self.g_a_proj(hidden_states))
        g_out = g_out.reshape(batch_size, -1, self.num_heads, self.head_dim)
        core_attn_out = self.o_norm(core_attn_out, g_out)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.o_proj(core_attn_out), None


class SolarOpen2Attention(Glm4MoeAttention):
    """Grouped-query full attention with optional RoPE (NoPE by default), qk-norm, and an output sigmoid gate."""

    def __init__(self, config: SolarOpen2Config, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.use_rope = config.use_rope
        self.use_gqa_gate = config.use_gqa_gate
        if self.use_gqa_gate:
            self.g_proj = nn.Linear(
                config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.use_gqa_gate_bias
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.use_rope:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward
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
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if self.use_gqa_gate:
            attn_output = attn_output * torch.sigmoid(self.g_proj(hidden_states))
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class SolarOpen2MLP(Glm4MoeMLP):
    pass


class SolarOpen2MoE(SolarOpenMoE):
    pass


class SolarOpen2RotaryEmbedding(SolarOpenRotaryEmbedding):
    pass


class SolarOpen2DecoderLayer(Glm4MoeDecoderLayer):
    def __init__(self, config: SolarOpen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        if config.layer_types[layer_idx] == "linear_attention":
            self.self_attn = SolarOpen2LinearAttention(config, layer_idx)


@auto_docstring
class SolarOpen2PreTrainedModel(SolarOpenPreTrainedModel):
    _no_split_modules = ["SolarOpen2DecoderLayer"]
    _is_stateful = True
    _can_compile_fullgraph = False
    _can_record_outputs = {
        "hidden_states": SolarOpen2DecoderLayer,
        "attentions": SolarOpen2Attention,
    }

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, SolarOpen2LinearAttention):
            init.ones_(module.dt_bias)
            init.copy_(module.A_log, torch.empty_like(module.A_log).uniform_(1, 16).log_())
        elif isinstance(module, SolarOpen2RMSNormGated):
            init.ones_(module.weight)


@auto_docstring
class SolarOpen2Model(Qwen3NextModel):
    def __init__(self, config: SolarOpen2Config):
        if getattr(config, "hc_rate", 0) > 0:
            raise NotImplementedError("Hyper-Connections (hc_rate > 0) are not supported by this implementation.")
        super().__init__(config)


@auto_docstring(checkpoint="upstage/Solar-Open2-250B")
class SolarOpen2ForCausalLM(SolarOpenForCausalLM):
    pass


__all__ = [
    "SolarOpen2Config",
    "SolarOpen2PreTrainedModel",
    "SolarOpen2Model",
    "SolarOpen2ForCausalLM",
]
