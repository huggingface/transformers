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
from collections.abc import Callable

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...integrations import use_kernelized_func
from ...integrations.accelerate import force_accelerate_hooks
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.output_capturing import OutputRecorder
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3Experts,
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3MoE,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    DeepseekV3TopkRouter,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..qwen3_next.modeling_qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextModel,
    Qwen3NextRMSNormGated,
    apply_mask_to_padding_states,
    causal_conv1d_fn,
    causal_conv1d_update,
    l2norm,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Goekdeniz-Guelmez/Omoikane-50M-A27M-Base")
@strict
class OmoikaneConfig(PreTrainedConfig):
    r"""
    layer_types (`list[str]`, *optional*):
        The token mixer of each layer: `"linear_attention"` (Gated DeltaNet) or `"full_attention"` (multi-head latent
        attention). Defaults to Gated DeltaNet everywhere except layers 3 and 7.
    mlp_layer_types (`list[str]`, *optional*):
        The feed-forward block of each layer: `"dense"` (SwiGLU MLP) or `"sparse"` (mixture of experts). Defaults to
        dense in the first and last layers and sparse elsewhere.
    linear_num_heads (`int`, *optional*, defaults to 6):
        Number of heads in the Gated DeltaNet layers (the same for keys and values).
    linear_key_head_dim (`int`, *optional*, defaults to 64):
        Dimension of each key head in the Gated DeltaNet layers.
    linear_value_head_dim (`int`, *optional*, defaults to 64):
        Dimension of each value head in the Gated DeltaNet layers.
    linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
        Kernel size of the short causal convolution applied to queries, keys and values in the Gated DeltaNet layers.
    n_group (`int`, *optional*, defaults to 4):
        Number of groups of routed experts.
    topk_group (`int`, *optional*, defaults to 2):
        Number of groups that each token may route to.

    The nested format of the training checkpoints (`gdn_config`, `mla_config`, `moe_config`, `mtp_config`,
    `dense_layer_indices` and `mla_layer_indices`) is also accepted and converted to the fields above.

    Example:

    ```python
    >>> from transformers import OmoikaneConfig, OmoikaneModel

    >>> # Initializing an Omoikane-50M-A27M style configuration
    >>> configuration = OmoikaneConfig()

    >>> # Initializing a model from the configuration
    >>> model = OmoikaneModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "omoikane"
    keys_to_ignore_at_inference = ["cache"]
    attribute_map = {"num_local_experts": "n_routed_experts"}
    base_model_tp_plan = {
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
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "ep_dispatch_experts",
    }

    vocab_size: int = 16384
    hidden_size: int = 384
    intermediate_size: int = 1152
    num_hidden_layers: int = 10
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 2
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    tie_word_embeddings: bool = True
    layer_types: list[str] | None = None
    mlp_layer_types: list[str] | None = None
    linear_num_heads: int = 6
    linear_key_head_dim: int = 64
    linear_value_head_dim: int = 64
    linear_conv_kernel_dim: int = 4
    num_attention_heads: int = 6
    num_key_value_heads: int | None = None
    q_lora_rank: int = 192
    kv_lora_rank: int = 96
    qk_nope_head_dim: int = 48
    qk_rope_head_dim: int = 16
    v_head_dim: int = 64
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    moe_intermediate_size: int = 192
    n_routed_experts: int = 16
    n_shared_experts: int = 1
    num_experts_per_tok: int = 4
    n_group: int = 4
    topk_group: int = 2
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0

    def __post_init__(self, **kwargs):
        self._from_nested_format(kwargs)
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i in (3, 7) else "linear_attention" for i in range(self.num_hidden_layers)
            ]
        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "dense" if i in (0, self.num_hidden_layers - 1) else "sparse" for i in range(self.num_hidden_layers)
            ]
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 100000.0}
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim
        super().__post_init__(**kwargs)

    def _from_nested_format(self, kwargs):
        """Read the nested format of the training checkpoints, and remove its keys from `kwargs`."""
        gdn = kwargs.pop("gdn_config", None)
        mla = kwargs.pop("mla_config", None)
        moe = kwargs.pop("moe_config", None)
        kwargs.pop("mtp_config", None)  # training only, see `OmoikaneModel`
        dense_layers = kwargs.pop("dense_layer_indices", None)
        mla_layers = kwargs.pop("mla_layer_indices", None)
        if gdn:
            if gdn.get("conv_bias", False) or not gdn.get("qk_l2norm", True):
                raise ValueError("Omoikane supports only `conv_bias=False` and `qk_l2norm=True`.")
            self.linear_num_heads = gdn.get("num_heads", self.linear_num_heads)
            self.linear_key_head_dim = gdn.get("head_dim_k", self.linear_key_head_dim)
            self.linear_value_head_dim = gdn.get("head_dim_v", self.linear_value_head_dim)
            self.linear_conv_kernel_dim = gdn.get("short_conv_kernel", self.linear_conv_kernel_dim)
        if mla:
            self.num_attention_heads = mla.get("num_heads", self.num_attention_heads)
            self.num_key_value_heads = self.num_attention_heads
            for name in ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
                setattr(self, name, mla.get(name, getattr(self, name)))
            self.rope_parameters = {"rope_type": "default", "rope_theta": float(mla.get("rope_theta", 100000.0))}
        if moe is not None:
            groups = moe.get("group_limited_routing", {})
            self.n_group = groups.get("n_group", 1)
            self.topk_group = groups.get("topk_group", 1)
            for name in (
                "moe_intermediate_size",
                "n_routed_experts",
                "n_shared_experts",
                "num_experts_per_tok",
                "norm_topk_prob",
                "routed_scaling_factor",
            ):
                setattr(self, name, moe.get(name, getattr(self, name)))
        if mla_layers is not None:
            self.layer_types = [
                "full_attention" if i in mla_layers else "linear_attention" for i in range(self.num_hidden_layers)
            ]
        if dense_layers is not None or moe is not None:
            dense_layers = dense_layers or []
            self.mlp_layer_types = [
                "dense" if (i in dense_layers or not moe) else "sparse" for i in range(self.num_hidden_layers)
            ]

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.n_routed_experts % self.n_group != 0:
            raise ValueError(
                f"`n_routed_experts` ({self.n_routed_experts}) must be divisible by `n_group` ({self.n_group})."
            )


class OmoikaneRMSNorm(DeepseekV3RMSNorm):
    pass


class OmoikaneRMSNormGated(Qwen3NextRMSNormGated):
    pass


class OmoikaneRotaryEmbedding(DeepseekV3RotaryEmbedding):
    pass


class OmoikaneMLP(DeepseekV3MLP):
    pass


class OmoikaneTopkRouter(DeepseekV3TopkRouter):
    pass


class OmoikaneExperts(DeepseekV3Experts):
    pass


class OmoikaneMoE(DeepseekV3MoE):
    pass


@use_kernelized_func(
    [torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule, causal_conv1d_fn, causal_conv1d_update]
)
class OmoikaneGatedDeltaNet(nn.Module):
    """
    Gated DeltaNet token mixer of Omoikane.

    Differences from `Qwen3NextGatedDeltaNet`: separate q/k/v/a/b/g projections, the same number of key and value
    heads, and a readout scale of 1: queries and keys are L2-normalized (with an epsilon of 1e-12 inside the square
    root), but queries are not scaled by `head_k_dim**-0.5`. The delta rule runs in float32.
    """

    def __init__(self, config: OmoikaneConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_heads
        self.value_dim = self.head_v_dim * self.num_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        # The short convolution always uses SiLU, whatever `hidden_act` is
        self.activation = "silu"

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # The checkpoints store one convolution per projection; they are concatenated on load.
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.A_log = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.o_norm = OmoikaneRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)

    @force_accelerate_hooks("conv1d")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Requires LEFT padding to work correctly
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_cache = cache is not None
        if use_cache:
            conv_state = cache.get_conv_state(self.layer_idx, state_idx=0)
            recurrent_state = cache.get_recurrent_state(self.layer_idx, state_idx=0)
        else:
            conv_state, recurrent_state = None, None

        mixed_qkv = torch.cat(
            [self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)], dim=-1
        ).transpose(1, 2)

        if conv_state is not None and seq_len == 1 and not cache.is_recording_past(self.layer_idx):
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        else:
            if cache is not None:
                mixed_qkv = cache.update_conv_state(mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size)
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )
            mixed_qkv = mixed_qkv[:, :, -seq_len:]

        query, key, value = torch.split(
            mixed_qkv.transpose(1, 2), [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim).float()
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim).float()
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim).float()

        # The delta-rule kernels always scale queries by `head_k_dim**-0.5`; multiply it back for a readout scale of 1.
        query = l2norm(query, eps=1e-12) * self.head_k_dim**0.5
        key = l2norm(key, eps=1e-12)
        beta = self.b_proj(hidden_states).float().sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        if recurrent_state is not None and seq_len == 1:
            output, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=False,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
                **kwargs,
            )
        else:
            output, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=False,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
                **kwargs,
            )

        if cache is not None:
            cache.update_recurrent_state(last_recurrent_state, self.layer_idx)

        gate = self.g_proj(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        output = self.o_norm(output, gate).reshape(batch_size, seq_len, -1)
        return self.o_proj(output.to(hidden_states.dtype))


class OmoikaneAttention(DeepseekV3Attention):
    """
    Multi-head latent attention of Omoikane. Differences from `DeepseekV3Attention`: the rotary dimensions are rotated
    in halves (not interleaved pairs), and the attention output is multiplied by `sigmoid(o_gate_proj(x))` before
    `o_proj`.
    """

    def __init__(self, config: OmoikaneConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_a_layernorm = OmoikaneRMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.kv_a_layernorm = OmoikaneRMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.o_gate_proj = nn.Linear(config.hidden_size, self.num_heads * self.v_head_dim, bias=config.attention_bias)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        cache: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_nope, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_nope = self.kv_a_layernorm(kv_nope)
        # Both latents are viewed as single-head, 4D tensors so all cache layers handle them correctly
        kv_nope = kv_nope.view(batch_size, 1, seq_length, self.kv_lora_rank)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        # Cache read / write is performed while latent KV is still compressed
        if cache is not None:
            kv_nope, k_rot = cache.update(kv_nope, k_rot, self.layer_idx, kwargs=kwargs)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states, value_states = self.expand_kv(kv_nope, k_rot)

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
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(self.o_gate_proj(hidden_states))
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class OmoikaneDecoderLayer(Qwen3NextDecoderLayer):
    def __init__(self, config: OmoikaneConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        # The checkpoints name both token mixers `self_attn`
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.self_attn = OmoikaneGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = OmoikaneAttention(config, layer_idx)

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = OmoikaneMoE(config)
        else:
            self.mlp = OmoikaneMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = OmoikaneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OmoikaneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                cache=cache,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class OmoikanePreTrainedModel(PreTrainedModel):
    config: OmoikaneConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OmoikaneDecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(OmoikaneTopkRouter, index=0),
        "hidden_states": OmoikaneDecoderLayer,
        "attentions": OmoikaneAttention,
    }
    _keep_in_fp32_modules_strict = ["e_score_correction_bias", "gate.weight", "A_log", "dt_bias"]
    _is_stateful = True

    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, OmoikaneGatedDeltaNet):
            init.copy_(module.A_log, torch.empty_like(module.A_log).uniform_(1, 16).log_())
            dt = torch.exp(torch.rand_like(module.dt_bias) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            init.copy_(module.dt_bias, dt + torch.log(-torch.expm1(-dt)))
            init.normal_(module.conv1d.weight, mean=0.0, std=module.conv_kernel_size**-0.5)
        elif isinstance(module, OmoikaneTopkRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, OmoikaneExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)


@auto_docstring
class OmoikaneModel(Qwen3NextModel):
    def __init__(self, config: OmoikaneConfig):
        OmoikanePreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [OmoikaneDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = OmoikaneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = OmoikaneRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # The checkpoints store the multi-token-prediction module, used only in training, as the layer after the last
        self._keys_to_ignore_on_load_unexpected = [rf"(^|\.)layers\.{config.num_hidden_layers}\."]
        # Initialize weights and apply final processing
        self.post_init()


class OmoikaneForCausalLM(DeepseekV3ForCausalLM):
    def forward(self, **super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, OmoikaneForCausalLM

        >>> model = OmoikaneForCausalLM.from_pretrained("Goekdeniz-Guelmez/Omoikane-50M-A27M-Base")
        >>> tokenizer = AutoTokenizer.from_pretrained("Goekdeniz-Guelmez/Omoikane-50M-A27M-Base")

        >>> inputs = tokenizer("Plants create energy through a process known as", return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_new_tokens=20)
        >>> text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        ```"""
        return super().forward(**super_kwargs)


__all__ = [
    "OmoikaneConfig",
    "OmoikanePreTrainedModel",
    "OmoikaneModel",
    "OmoikaneForCausalLM",
]
