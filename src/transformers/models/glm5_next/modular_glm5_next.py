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
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...integrations import (
    use_kernel_forward_from_hub,
    use_kernel_func_from_hub,
    use_kernelized_func,
)
from ...integrations.accelerate import force_accelerate_hooks
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast, MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging, torch_compilable_check
from ...utils.generic import is_flash_attention_requested, merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..auto import CONFIG_MAPPING, AutoConfig
from ..auto.modeling_auto import AutoModel
from ..deepseek_v2.modeling_deepseek_v2 import DeepseekV2Attention
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE, DeepseekV3TopkRouter
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection
from ..exaone4_5.modeling_exaone4_5 import Exaone4_5_Model
from ..glm46v.modeling_glm46v import Glm46VForConditionalGeneration
from ..glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from ..glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer
from ..inkling.modeling_inkling import causal_conv1d_fn, causal_conv1d_update
from ..llama.modeling_llama import LlamaRMSNorm, eager_attention_forward
from ..minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLExperts
from ..mixtral.modeling_mixtral import load_balancing_loss_func
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
from ..qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated
from ..qwen3_next.modeling_qwen3_next import apply_mask_to_padding_states


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="zai-org/GLM-5-Next")
@strict
class Glm5NextTextConfig(GlmMoeDsaConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of routed expert groups.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer feed-forward schedule. Values are `"dense"` or `"sparse"`.
    layer_types (`list[str]`, *optional*):
        Per-layer attention cache schedule. Values are `"linear_attention"` for
        KDA layers and `"full_attention"` for MLA layers.
    swiglu_limit (`float`, *optional*, defaults to 10.0):
        Clamp limit applied to SwiGLU gate/up projections.
    linear_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each head in linear attention.
    linear_num_heads (`int`, *optional*, defaults to 64):
        Number of heads used in linear attention layers.
    linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
        Kernel size of the convolution used in linear attention layers.
    linear_lower_bound (`float`, *optional*, defaults to -5.0):
        Whether the forget gate has a lower bound to apply to the decay.
    hc_mult (`int`, *optional*, defaults to 4):
        Number of MHC residual streams.
    hc_eps (`float`, *optional*, defaults to 1e-6):
        Numerical floor used by MHC Sinkhorn normalization.
    hc_sinkhorn_iters (`int`, *optional*, defaults to 20):
        Number of Sinkhorn iterations used by MHC routing.
    """

    model_type = "glm5_next_vl_text"
    base_config_key = "text_config"

    num_hidden_layers: int = 45
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    head_dim: int = 0
    max_position_embeddings: int = 4096  # TODO: check this value on relase
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    qk_rope_head_dim: int = 0
    moe_intermediate_size: int = 2048
    num_experts_per_tok: int = 8
    n_routed_experts: int = 288
    swiglu_limit: float | None = 10.0
    linear_head_dim: int = 128
    linear_num_heads: int = 64
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float | None = -5.0
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001

    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = 154820

    # TODO: add when we have an indexer trained
    index_head_dim = AttributeError()
    index_n_heads = AttributeError()
    index_topk = AttributeError()
    indexer_types = AttributeError()

    rope_parameters = AttributeError()
    first_k_dense_replace = AttributeError()
    mlp_bias = AttributeError()

    # TODO: After indexer: add conversion to dsa layer type + indexer conversion
    # TODO: Validate arch with indexer specific things
    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(3, self.num_hidden_layers) + ["sparse"] * (
                self.num_hidden_layers - 3
            )

        if self.layer_types is None:
            kda_layers = [idx for idx in range(self.num_hidden_layers) if idx % 4 != 3]
            self.layer_types = [
                "linear_attention" if layer_idx in kda_layers else "full_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]

        # Convert dict to attributes (if given)
        linear_attn_dict = kwargs.pop("linear_attn_config", None)
        if linear_attn_dict is not None:
            self.linear_head_dim = linear_attn_dict.get("head_dim", self.linear_head_dim)
            self.linear_num_heads = linear_attn_dict.get("num_heads", self.linear_num_heads)
            self.linear_conv_kernel_dim = linear_attn_dict.get("short_conv_kernel_size", self.linear_conv_kernel_dim)
            self.linear_lower_bound = linear_attn_dict.get("lower_bound", self.linear_lower_bound)

            # Additional lower bound logic as per original dict
            if linear_attn_dict.get("safe_gate", False) and self.linear_lower_bound is None:
                self.linear_lower_bound = -5.0

        # NOTE: this forces an intentional override as we have the convention of head_dim being the RoPE based dim
        kwargs.pop("head_dim", None)
        self.head_dim = self.qk_rope_head_dim
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim

        PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring(checkpoint="zai-org/GLM-5-Next")
@strict
class Glm5NextConfig(PreTrainedConfig):
    r"""
    image_token_id (`int`, *optional*, defaults to 154854):
        The image token index to encode the image prompt.
    video_token_id (`int`, *optional*, defaults to 154855):
        The video token index to encode the video prompt.
    image_start_token_id (`int`, *optional*, defaults to 154830):
        The image start token index to encode the start of image.
    image_end_token_id (`int`, *optional*, defaults to 154831):
        The image end token index to encode the end of image.
    video_start_token_id (`int`, *optional*, defaults to 154832):
        The video start token index to encode the start of video.
    video_end_token_id (`int`, *optional*, defaults to 154833):
        The video end token index to encode the end of video.

    ```python
    >>> from transformers import Glm5NextConfig

    >>> # Initializing a GLM-5-Next style configuration
    >>> configuration = Glm5NextConfig()
    ```"""

    model_type = "glm5_next_vl"
    sub_configs = {"vision_config": AutoConfig, "text_config": Glm5NextTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 154854
    video_token_id: int = 154855
    image_start_token_id: int = 154830
    image_end_token_id: int = 154831
    video_start_token_id: int = 154832
    video_end_token_id: int = 154833
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "glm_ocr_vision")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["glm_ocr_vision"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            # Flat (text-only) GLM-5-Next checkpoints store the text fields at the
            # top level; forward them so `text_config` is populated for BC.
            self.text_config = self.sub_configs["text_config"](**kwargs)

        super().__post_init__(**kwargs)


class Glm5NextTextRMSNorm(LlamaRMSNorm):
    pass


class Glm5NextTextMLP(Qwen2MoeMLP):
    def __init__(self, config, intermediate_size=None):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Optional clamping
        if self.swiglu_limit is not None:
            gate = gate.clamp(min=None, max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm5NextTextExperts(MiniMaxM3VLExperts):
    def __init__(self, config):
        super().__init__(config)
        del self.limit
        del self.swiglu_alpha
        self.intermediate_dim = config.moe_intermediate_size

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        # Optional clamping
        if self.swiglu_limit is not None:
            gate = gate.clamp(min=None, max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        # Simple swiglu instead of alpha
        return F.silu(gate) * up


class Glm5NextTextTopkRouter(DeepseekV3TopkRouter):
    pass


class Glm5NextTextMoE(DeepseekV3MoE):
    def __init__(self, config: Glm5NextConfig):
        super().__init__(config)
        self.experts = Glm5NextTextExperts(config)
        self.gate = Glm5NextTextTopkRouter(config)
        self.shared_experts = Glm5NextTextMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )


class Glm5NextTextHyperConnection(DeepseekV4HyperConnection):
    pass


class Glm5NextTextHyperHead(nn.Module):
    """Final GLM-5-Next HC-stream collapse. Unlike DeepSeek-V4, this is an unweighted mean."""

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=2)


class Glm5NextTextForgetGate(nn.Module):
    def __init__(self, config: Glm5NextTextConfig):
        super().__init__()
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.qkv_dim = self.head_dim * self.num_heads

        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.qkv_dim))
        self.A_log = nn.Parameter(torch.empty(self.num_heads))

        self.safe_gate_lower_bound = config.linear_lower_bound

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_shape = (*hidden_states.shape[:2], -1, self.head_dim)

        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (forget_gate.float() + self.dt_bias.float().view(1, 1, -1)).view(hidden_shape)
        A_log = self.A_log.float().view(1, 1, self.num_heads, 1)
        decay_rate = torch.exp(A_log)

        # Safe lower bound decay
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * torch.sigmoid(decay_rate * g)

        # Softplus "log(1 + exp(x))" with uper bound restraint to avoid overflows
        # NOTE: Softplus for larger values (e.g. 20+), Softplus(x) == x
        g_softplus = torch.where(g > 20.0, g, torch.log(1.0 + torch.exp(g)))

        return -decay_rate * g_softplus


@use_kernel_forward_from_hub("RMSNormGated")
class Glm5NextTextRMSNormGated(Qwen3_5RMSNormGated):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__(hidden_size, eps, kwargs)
        self.activation = "sigmoid"

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype

        # Strict FP32 norm (do not downcast on the weights)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states

        # Apply gating
        hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))

        return hidden_states.to(input_dtype)


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """
    This function is intended to align with the l2norm implementation in the FLA library.

    # NOTE: FLA compares against `F.normalize` but does + eps instead of max(..., eps) leading to a slight differences
    """
    # main difference to qwen's gdn variation: intentionally use sqrt and / to match original triton
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


@use_kernel_func_from_hub("recurrent_kimi_delta_attention")
def recurrent_kimi_delta_attention(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    # calculations happen in float as states are more susceptible to rounding errors
    initial_dtype = query.dtype
    query, key, value, g, beta = [x.to(torch.float32) for x in (query, key, value, g, beta)]

    # important: FLA calculates these in fp32 so we do this after the float casts
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # shapes and other metadata
    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, sequence_length, num_heads, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    # recurrent iteration
    for i in range(sequence_length):
        q_i = query[:, i]
        k_i = key[:, i]
        v_i = value[:, i]
        g_i = g[:, i][..., None].exp()
        b_i = beta[:, i][..., None]

        last_recurrent_state = last_recurrent_state * g_i
        kv_mem = (last_recurrent_state * k_i[..., None]).sum(dim=-2)
        delta = (v_i - kv_mem) * b_i

        last_recurrent_state = last_recurrent_state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, i] = (last_recurrent_state * q_i.unsqueeze(-1)).sum(dim=-2)

    return core_attn_out.to(initial_dtype), last_recurrent_state if output_final_state else None


@use_kernel_func_from_hub("chunk_kimi_delta_attention")
def chunk_kimi_delta_attention(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    # calculations happen in float as states are more susceptible to rounding errors
    initial_dtype = query.dtype

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    # important: FLA calculates these in fp32 so we do this after the float casts
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # shapes and other metadata
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    total_sequence_length = sequence_length + pad_size

    # prepare all the relevant input
    query = F.pad(query, (0, 0, 0, pad_size)) * scale
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # reshape to chunks
    query, key, value, g, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, g, k_beta, v_beta)
    ]
    beta = beta.reshape(beta.shape[0], beta.shape[1], -1, chunk_size)

    # Intra chunk
    # Main difference to GDN is the per head application of `g` which was broadcasted across heads instead
    g = g.cumsum(dim=-2)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    decay_mask = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp().float()
    attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(total_sequence_length // chunk_size):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]
        g_i = g[:, :, i]

        # Inter chunk
        attn_inter = (q_i * g_i.exp()) @ last_recurrent_state
        # Intra chunk
        attn_intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i]).sum(dim=-1).masked_fill(mask, 0)
        # New update rule
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime

        core_attn_out[:, :, i] = attn_inter + attn_intra @ v_new
        last_recurrent_state = (
            last_recurrent_state * g_i[:, :, -1].exp().unsqueeze(-1)
            + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)

    return core_attn_out, last_recurrent_state


@use_kernelized_func(
    [chunk_kimi_delta_attention, recurrent_kimi_delta_attention, causal_conv1d_fn, causal_conv1d_update]
)
class Glm5NextTextLinearAttention(nn.Module):
    """Kimi-style KDA (Kimi Linear Attention) for GLM-5-Next."""

    def __init__(
        self,
        config: Glm5NextTextConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        self.qkv_dim = self.head_dim * self.num_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        self.q_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)

        self.conv_dim = self.qkv_dim * 3
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.forget_gate = Glm5NextTextForgetGate(config)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.o_norm = Glm5NextTextRMSNormGated(self.head_dim, eps=self.layer_norm_epsilon)
        self.o_proj = nn.Linear(self.qkv_dim, self.hidden_size, bias=False)

        self.layer_type = config.layer_types[layer_idx]

    @force_accelerate_hooks("conv1d")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        # Zero out padding
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        mixed_qkv = torch.cat(
            [
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            ],
            dim=-1,
        ).transpose(1, 2)

        # Acts for normal prefill but also for multi-token prefill continue
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0]

        # Single token decode path
        if use_precomputed_states and seq_len == 1:
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        # Multi token prefill or simple "full" prefill
        else:
            # Concatenated state for prefill
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )

            # Cut out any tail
            mixed_qkv = mixed_qkv[:, :, -seq_len:]

        query, key, value = torch.split(
            mixed_qkv.transpose(1, 2),
            [self.qkv_dim] * 3,
            dim=-1,
        )

        query = query.view(hidden_shape)
        key = key.view(hidden_shape)
        value = value.view(hidden_shape)

        # Forget gate and input gate
        g = self.forget_gate(hidden_states)
        beta = torch.sigmoid(self.b_proj(hidden_states))

        # KDA
        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = recurrent_kimi_delta_attention(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                **kwargs,
            )
        else:
            core_attn_out, last_recurrent_state = chunk_kimi_delta_attention(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                **kwargs,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state.to(torch.float32), self.layer_idx)

        # Final gated norm and proj
        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)
        output = self.o_norm(core_attn_out, gate).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output


# TODO: add indexer if trained
class Glm5NextTextAttention(DeepseekV2Attention):
    def __init__(self, config: Glm5NextTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_a_layernorm = Glm5NextTextRMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.kv_a_layernorm = Glm5NextTextRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        # LoRA based path is guaranteed based on the config validation
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(key_shape).transpose(1, 2)
        key_states, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Cache update
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # Flash attention head_dim padding
        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

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

        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Glm5NextTextDecoderLayer(GlmMoeDsaDecoderLayer):
    def __init__(self, config: Glm5NextTextConfig, layer_idx: int):
        self.block_type = config.layer_types[layer_idx]

        super().__init__(config, layer_idx)
        self.self_attn = (
            Glm5NextTextLinearAttention(config, layer_idx)
            if self.block_type == "linear_attention"
            else Glm5NextTextAttention(config, layer_idx)
        )

        self.attn_hc = Glm5NextTextHyperConnection(config)
        self.ffn_hc = Glm5NextTextHyperConnection(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, None]:
        dtype = hidden_states.dtype

        residual = hidden_states
        post, comb, hidden_states = self.attn_hc(hidden_states)
        # Self attn
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

        residual = hidden_states
        post, comb, hidden_states = self.ffn_hc(hidden_states)
        # Feed forward
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

        return hidden_states


@auto_docstring
class Glm5NextPreTrainedModel(PreTrainedModel):
    config: Glm5NextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    # needs index based kernel
    _supports_flash_attn = False
    _supports_sdpa = True
    # needs per layer creation, too expensive
    _supports_flex_attn = False
    _supports_attention_backend = True

    _no_split_modules = ["Glm5NextTextDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    # TODO: this can be fixed but is limited by
    # 1. assuming the cache name
    # 2. linear attention not being considered atm
    _is_stateful = True
    _can_compile_fullgraph = True

    _can_record_outputs = {
        "attentions": Glm5NextTextAttention,
        "hidden_states": Glm5NextTextDecoderLayer,
        "router_logits": OutputRecorder(Glm5NextTextTopkRouter, index=0),  # noqa: F821
    }
    _keep_in_fp32_modules_strict = ["e_score_correction_bias", "conv1d", "dt_bias", "A_log"]
    _keys_to_ignore_on_load_unexpected = [r"layers\.45\.", r"layers\.\d+\.shared_head\."]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Glm5NextTextForgetGate):
            # Following FLA initialization
            # NOTE: This is incredibly important so keep it this way at all costs
            if module.safe_gate_lower_bound is not None:
                init.zeros_(module.A_log)
            else:
                init.copy_(
                    module.A_log,
                    init.uniform_(module.A_log, a=1.0, b=16.0).log(),
                )

            init.uniform_(
                module.dt_bias,
                a=math.log(1e-3),
                b=math.log(1e-1),
            )
            dt = module.dt_bias.exp().clamp_min(1e-4)

            # (stable) inverse softplus
            init.copy_(
                module.dt_bias,
                dt + torch.log(-torch.expm1(-dt)),
            )
        elif isinstance(module, Glm5NextTextRMSNormGated):
            init.ones_(module.weight)
        elif isinstance(module, Glm5NextTextHyperConnection):
            init.normal_(module.fn, mean=0.0, std=0.02)
            init.zeros_(module.base)
            init.ones_(module.scale)
        elif isinstance(module, Glm5NextTextExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Glm5NextTextTopkRouter):
            init.zeros_(module.e_score_correction_bias)
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


# Do not inherit from DSv4 as it messes modular prefixes up for the PreTrainedModel
@auto_docstring
class Glm5NextTextModel(Glm5NextPreTrainedModel):
    config: Glm5NextTextConfig

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Glm5NextTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Glm5NextTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.hc_head = Glm5NextTextHyperHead()

        # Initialize weights and apply final processing
        self.post_init()

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
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)

        # TODO: masks change based on the indexer or not
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

        # Key change: NoPE
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                position_embeddings=None,
                input_ids=input_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class Glm5NextModel(Exaone4_5_Model, Glm5NextPreTrainedModel):
    config: Glm5NextConfig
    _no_split_modules = AttributeError()

    def __init__(self, config):
        super().__init__(config)
        self.visual = AutoModel.from_config(config.vision_config)
        self.language_model = Glm5NextTextModel._from_config(config.text_config)
        del self.rope_deltas

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        # Same as in `Glm46V`
        # reshape video_grid_thw -> [b, 3] -> [1, h, w] * frames
        t = video_grid_thw[:, 0]
        hw = video_grid_thw[:, 1:]
        # repeat each (h,w) row `t` times
        flattened_hw = torch.repeat_interleave(hw, t, dim=0)
        prefix_ones = video_grid_thw.new_ones(flattened_hw.shape[0], 1)
        flattened_video_grid_thw = torch.cat([prefix_ones, flattened_hw], dim=1)

        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values_videos, grid_thw=flattened_video_grid_thw, **kwargs)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        vision_outputs.pooler_output = torch.split(vision_outputs.pooler_output, split_sizes)
        return vision_outputs

    # TODO: `get_placeholder_mask` is broken for simultaneous img + vid
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            # GLM 5 Next VL special_video_mask is special_image_mask
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).to(inputs_embeds.device)
        if image_features is not None:
            torch_compilable_check(
                n_image_tokens * inputs_embeds.shape[-1] == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).to(inputs_embeds.device)
        if video_features is not None:
            torch_compilable_check(
                n_video_tokens * inputs_embeds.shape[-1] == video_features.numel(),
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, features: {video_features.shape[0]}",
            )
        return special_image_mask, special_video_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, **kwargs).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, **kwargs).pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        # Only change is the output type to Moe
        return MoeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class Glm5NextForConditionalGeneration(Glm46VForConditionalGeneration, Glm5NextPreTrainedModel):
    """
    Main Glm5Next conditional generation class.
    """

    def __init__(self, config):
        super().__init__(config)
        self.router_aux_loss_coef = config.text_config.router_aux_loss_coef
        self.num_experts = config.text_config.num_local_experts
        self.num_experts_per_tok = config.text_config.num_experts_per_tok

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        output_router_logits: bool | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:

        ```python
        >>> from transformers import AutoProcessor, Glm5NextForConditionalGeneration
        >>> import torch

        >>> model = Glm5NextForConditionalGeneration.from_pretrained("zai-org/GLM-5-Next")
        >>> processor = AutoProcessor.from_pretrained("zai-org/GLM-5-Next")

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        ...             {"type": "text", "text": "Describe the image."},
        ...         ],
        ...     }
        ... ]
        >>> inputs = processor.apply_chat_template(
        ...     messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ... )
        >>> inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        >>> generated_ids = model.generate(**inputs, max_new_tokens=64)
        ```
        """

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        raise AttributeError()


__all__ = [
    "Glm5NextConfig",
    "Glm5NextTextConfig",
    "Glm5NextPreTrainedModel",
    "Glm5NextTextModel",
    "Glm5NextModel",
    "Glm5NextForConditionalGeneration",
]
