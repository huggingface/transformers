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
# coding=utf-8

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ..deepseek_v3 import DeepseekV3MLP, DeepseekV3MoE, DeepseekV3TopkRouter
from ..gemma3.modeling_gemma3 import (
    Gemma3CausalLMOutputWithPast,
    Gemma3ForConditionalGeneration,
    Gemma3Model,
    Gemma3ModelOutputWithPast,
)
from ..glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteDecoderLayer, Glm4MoeLiteModel
from ..gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding, apply_rotary_pos_emb
from ..llama.modeling_llama import LlamaAttention, LlamaRMSNorm, eager_attention_forward
from ..phi3.modeling_phi3 import Phi3MLP


logger = logging.get_logger(__name__)


class TmlTextConfig(PreTrainedConfig):
    model_type = "tml_text"
    base_config_key = "text_config"

    vocab_size: int = 151936
    hidden_size: int = 4096
    num_hidden_layers: int = 48
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    swa_num_attention_heads: int | None = None
    swa_num_key_value_heads: int | None = None
    swa_head_dim: int | None = None
    sliding_window_size: int = 4096
    layer_types: list[int] | None = None
    partial_rotary_factor: int | None = None
    rel_extent: float = 10000.0
    rope_theta: float = 10000.0
    log_scaling_n_floor: int | None = None
    log_scaling_alpha: float = 1.0
    rms_norm_eps: float = 1e-6
    use_embed_norm: bool = True
    q_bias: bool = False
    o_bias: bool = False
    use_sconv: bool = False
    sconv_kernel_size: int = 4
    mlp_layer_types: list[int] | None = None
    dense_intermediate_size: int = 14336
    use_global_scale: bool = False
    hidden_act: str = "silu"
    # MoE
    moe_intermediate_size: int = 2048
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    moe_router_type: str = "softmax"
    moe_router_n_group: int = 8
    moe_router_topk_group: int = 4
    moe_router_use_bias_correction: bool = False
    moe_norm_topk_prob: bool = True
    inference_moe_w13_interleaved: bool = True
    logits_mup_width_multiplier: float = 24.0
    rms_norm_eps_moe_gate: float = 1e-6
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | None = 2

    def __post_init__(self, **kwargs):
        self.swa_num_attention_heads = self.swa_num_attention_heads or self.num_attention_heads
        self.swa_num_key_value_heads = self.swa_num_key_value_heads or self.num_key_value_heads
        self.swa_head_dim = self.swa_head_dim or self.head_dim

        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * self.num_hidden_layers
        super().__post_init__(**kwargs)


class TmlAudioConfig(PreTrainedConfig):
    model_type = "tml_audio"
    base_config_key = "audio_config"

    n_mel_bins: int = 80
    mel_vocab_size: int = 256
    text_hidden_size: int | None = None
    rms_norm_eps: float = 1e-6


class TmlVisionConfig(PreTrainedConfig):
    model_type = "tml_vision"
    base_config_key = "vision_config"

    vision_encoder_type: str = "hmlp"
    text_hidden_size: int | None = None
    patch_size: int = 14
    num_channels: int = 3
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6


class TmlConfig(PreTrainedConfig):
    """Top-level multimodal config (`TmlMMConfig` in the SGLang source)."""

    model_type = "tml"
    sub_configs = {
        "text_config": TmlTextConfig,
        "audio_config": TmlAudioConfig,
        "vision_config": TmlVisionConfig,
    }

    text_config: TmlTextConfig | dict | None = None
    audio_config: TmlAudioConfig | dict | None = None
    vision_config: TmlVisionConfig | dict | None = None
    image_token_id: int | None = None
    audio_token_id: int | None = None

    def __post_init_(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config = self.sub_configs["audio_config"](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = self.sub_configs["audio_config"]()

        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        super().__post_init__(**kwargs)


class TmlModelOutputWithPast(Gemma3ModelOutputWithPast):
    pass


class TmlCausalLMOutputWithPast(Gemma3CausalLMOutputWithPast):
    pass


class TmlRMSNorm(LlamaRMSNorm):
    pass


class TmlShortConvolution(nn.Module):
    """Causal depthwise short convolution applied to a branch output before
    the residual add (`attn_sconv` / `mlp_sconv` in `TmlDecoderLayer`).

    NOTE: SGLang instantiates this per-layer with a `layer_id` and
    `sconv_type` (ATTN vs MLP) but the actual math isn't in the provided
    source; this is a standard causal depthwise Conv1d, which is the usual
    implementation for this kind of "short conv" branch (à la Hyena/Mamba
    conv-in-block patterns). Swap in the real kernel if this differs.
    """

    def __init__(self, hidden_size: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            padding=kernel_size - 1,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor, cache: torch.Tensor | None = None) -> torch.Tensor:
        # hidden_states: (batch, seq_len, hidden_size)
        x = hidden_states.transpose(1, 2)  # (batch, hidden, seq_len)
        x = self.conv(x)[..., : hidden_states.shape[1]]
        return x.transpose(1, 2)


def compute_log_scaling_tau(position_ids: torch.Tensor, floor: int | None, alpha: float) -> torch.Tensor | None:
    """ "logn"-style per-position attention-logit scale.

    tau = alpha * log(max(pos + 1, floor)); multiplies attention scores so
    scores don't blow up at positions far beyond the trained context length.
    Returns None if `floor` is None (feature disabled), matching the
    SGLang call site which only invokes this when
    `config.log_scaling_n_floor is not None`.
    """
    if floor is None:
        return None
    pos = (position_ids + 1).clamp(min=floor).to(torch.float32)
    return alpha * torch.log(pos)


class TmlRotaryEmbedding(GPTNeoXRotaryEmbedding):
    pass


class TmlAttention(LlamaAttention):
    def __init__(self, config: TmlTextConfig, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        self.is_local_attn = config.layer_types[self.layer_idx] == "sliding_attention"

        self.num_heads = config.swa_num_attention_heads if self.is_local_attn else config.num_attention_heads
        self.num_key_value_heads = config.swa_num_key_value_heads if self.is_local_attn else config.num_key_value_heads
        self.head_dim = config.swa_head_dim if self.is_local_attn else config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.sliding_window = config.sliding_window_size if self.is_local_attn else None

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.q_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.r_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.o_bias)

    def _apply_r_gate(self, attn_output: torch.Tensor, hidden_states: torch.Tensor, shape: tuple) -> torch.Tensor:
        # best-effort "receptance" gate: r = sigmoid(W_r x), applied
        # elementwise to the attention output before the output projection
        r = torch.sigmoid(self.r_proj(hidden_states).view(*shape))
        return attn_output * r.transpose(1, 2).reshape(attn_output.shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        log_scaling_tau: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

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
            log_scaling_tau=log_scaling_tau,
            **kwargs,
        )

        attn_output = self._apply_r_gate(attn_output, hidden_states, hidden_shape)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class TmlDenseMLP(Phi3MLP):
    def __init__(self, config: TmlTextConfig):
        super().__init__(config)
        self.global_scale = 1.0 / math.sqrt(config.intermediate_size) if config.use_global_scale else 1.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = super().forward(hidden_states)
        return out * self.global_scale if self.global_scale != 1.0 else out


class TmlMoEGate(DeepseekV3TopkRouter):
    def __init__(self, config: TmlTextConfig):
        super().__init__(config)
        self.router_type = config.moe_router_type
        if not config.moe_router_use_bias_correction:
            self.e_score_correction_bias = None

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.router_type == "sigmoid_grouped":
            return super().forward(hidden_states)

        # flat softmax top-k fallback
        logits = F.linear(hidden_states.to(torch.float32), self.weight.to(torch.float32))
        scores = logits.softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, topk_weight.to(hidden_states.dtype)


class TmlMoEExpertMLP(DeepseekV3MLP):
    pass


class TmlMoE(DeepseekV3MoE):
    def __init__(self, config: TmlTextConfig):
        super().__init__(config)
        self.gate = TmlMoEGate(config)
        self.experts = nn.ModuleList(
            [
                TmlMoEExpertMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        if config.n_shared_experts is not None and config.n_shared_experts > 0:
            self.shared_experts = TmlMoEExpertMLP(
                config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
            )
        else:
            self.shared_experts = None


class TmlDecoderLayer(Glm4MoeLiteDecoderLayer):
    def __init__(self, config: TmlTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Maybe use_conv is always `True`, check it!
        self.layer_type = config.layer_types[layer_idx]
        self.attn_sconv = (
            TmlShortConvolution(config.hidden_size, config.sconv_kernel_size) if config.use_sconv else None
        )
        self.mlp_sconv = (
            TmlShortConvolution(config.hidden_size, config.sconv_kernel_size) if config.use_sconv else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        log_scaling_tau: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if self.attn_sconv is not None:
            hidden_states = self.attn_sconv(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.mlp_sconv is not None:
            hidden_states = self.mlp_sconv(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class TmlPreTrainedModel(PreTrainedModel):
    config_class = TmlConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TmlDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.get_text_config().initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, TmlRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, TmlMoEGate):
            module.weight.data.normal_(mean=0.0, std=std)


class TmlTextModel(Glm4MoeLiteModel):
    config: TmlTextConfig

    def __init__(self, config: TmlTextConfig):
        super().__init__(config)
        self.embed_norm = TmlRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embedding_multiplier = config.embedding_multiplier

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
            inputs_embeds = self.embed_tokens(input_ids) / self.embedding_multiplier

        if self.embed_norm is not None:
            inputs_embeds = self.embed_norm(inputs_embeds)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)

        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
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
        log_scaling_tau = compute_log_scaling_tau(
            position_ids, self.config.log_scaling_n_floor, self.config.log_scaling_alpha
        )

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[decoder_layer.layer_type],
                past_key_values=past_key_values,
                log_scaling_tau=log_scaling_tau,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class TmlAudioModel(TmlPreTrainedModel):
    def __init__(self, config: TmlAudioConfig):
        super().__init__(config)
        self.n_mel_bins = config.n_mel_bins
        self.mel_vocab_size = config.mel_vocab_size
        self.encoder = nn.Embedding(config.n_mel_bins * config.mel_vocab_size, config.text_hidden_size)
        self.final_norm = TmlRMSNorm(config.text_hidden_size, eps=1e-6)

        embedding_indices = torch.arange(self.n_mel_bins) * self.mel_vocab_size
        self.register_buffer("embedding_indices", embedding_indices.unsqueeze(0), persistent=False)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        if input_features.shape[1] != self.n_mel_bins:
            raise ValueError("`input_features` have to have exactly `num_mel_bin` length!")

        input_features = input_features.to(torch.int32)
        embedding_indices = self.embedding_indices.to(input_features.device) + input_features

        hidden_states = (
            self.encoder(embedding_indices.reshape(-1))
            .reshape(input_features.shape[0], self.n_mel_bins - 1)
            .sum(axis=1)
        )

        hidden_states = self.final_norm(hidden_states)
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=hidden_states,
        )


def prime_factors(number: int) -> list[int]:
    factors = []

    while number % 2 == 0:
        factors.append(2)
        number //= 2

    for p in range(3, math.isqrt(number) + 1, 2):
        while number % p == 0:
            factors.append(p)
            number //= p

    if number > 1:
        factors.append(number)
    return factors


def plan_out_scales(temporal_patch_size: int, patch_size: int, n_layers: int, n_channels: int) -> torch.LongTensor:
    """
    Plan out the dimensions for each layer in the HMLP encoder.

    This function determines the progression of dimensions (temporal, height, width, channels)
    for a multi-layer perceptual model that processes image/video patches. It follows these
    principles:
    1. Start with small dimensions and increase to full size
    2. Expand spatial dimensions (height/width) first, then temporal
    3. Increase channel count to avoid information bottlenecks
    4. Round channel dimensions to multiples of 64 for hardware efficiency

    The function computes optimal assignments of scale configurations to layers using either:
    - For n_layers >= len(scales): Individual best matching scales for each layer (allowing duplicates)
    - For n_layers < len(scales): Global optimal assignment via linear_sum_assignment

    The first and last scales are always fixed to ensure the proper input and output dimensions.

    Args:
        temporal_patch_size: Temporal dimension of input patches
        patch_size: Spatial dimension (height/width) of input patches
        n_layers: Number of layers in the encoder
        n_channels: Number of input channels (default: 3 for RGB)

    Returns:
        torch.LongTensor of shape `(n_layers + 1, 4)` where the last dim holds values for (t, h, w, c) grids.
    """
    h = torch.cumprod(torch.tensor(prime_factors(patch_size)[::-1]), dim=0)
    t = torch.cumprod(torch.tensor(prime_factors(temporal_patch_size)[::-1]), dim=0)

    h_ch = torch.ceil(h**2 / 64).int() * 64 * n_channels
    t_ch = (h_ch[-1] if len(h_ch) else 64 * n_channels) * t

    base = torch.tensor([[1, 1, 1, n_channels]])
    spatial = torch.stack([torch.ones_like(h), h, h, h_ch], dim=1)
    temporal = torch.stack([t, torch.full_like(t, h[-1]), torch.full_like(t, h[-1]), t_ch], dim=1)
    scales = torch.cat([base, spatial, temporal], dim=0)

    size_reduction = torch.prod(scales[:, :-1], dim=1).float()

    total_elements = patch_size * patch_size * temporal_patch_size * n_channels
    log_ideal_scales = torch.linspace(0, torch.log(torch.tensor(total_elements)), n_layers + 1)
    cost_matrix = torch.abs(log_ideal_scales.unsqueeze(1) - torch.log(size_reduction).unsqueeze(0))

    if n_layers >= scales.shape[0]:
        idxs = torch.argmin(cost_matrix, dim=1)
    else:
        from scipy.optimize import linear_sum_assignment

        _, idxs_np = linear_sum_assignment(cost_matrix.cpu().numpy())
        idxs = torch.tensor(idxs_np)
        # idxs = torch.softmax(-cost_matrix * 10, dim=1).argmax(dim=1)

    idxs[0] = 0
    idxs[-1] = scales.shape[0] - 1
    return scales[idxs]


class TmlVisionEncoderLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, t_fold: int, hw_fold: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.layer_norm = TmlRMSNorm(output_dim)
        self.hw_fold = hw_fold
        self.t_fold = t_fold

    def fold_timespace_to_depth(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Convert a tensor of shape (B, T, H, W, C) to a tensor of shape (B, T // t, H // hw, W //  hw, C * (t * hw**2))
        """
        B, T, H, W, C = hidden_states.shape

        t_new = T // self.t_fold
        h_new = H // self.hw_fold
        w_new = W // self.hw_fold

        hidden_states = hidden_states.reshape(B, t_new, self.t_fold, h_new, self.hw_fold, w_new, self.hw_fold, C)

        hidden_states = hidden_states.permute(0, 1, 3, 5, 2, 4, 6, 7)
        hidden_states = hidden_states.reshape(B, t_new, h_new, w_new, self.t_fold * self.hw_fold * self.hw_fold * C)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.hw_fold > 1 or self.t_fold > 1:
            hidden_states = self.fold_timespace_to_depth(hidden_states)

        hidden_states = self.projection(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return hidden_states


class TmlVisionModel(TmlPreTrainedModel):
    def __init__(self, config: TmlVisionConfig):
        super().__init__(config)
        self.scales = plan_out_scales(
            config.temporal_patch_size,
            config.patch_size,
            config.num_hidden_layers,
            config.num_channels,
        )

        # num_hidden_layers - 1 to encoder and the last to proj to text hidden dim
        self.encoder_layers = nn.ModuleList()
        for i, (start_scale, end_scale) in enumerate(zip(self.scales[:-1], self.scales[1:])):
            shuffle_mult = (
                (end_scale[0] // start_scale[0]) * (end_scale[1] // start_scale[1]) * (end_scale[2] // start_scale[2])
            )
            output_dim = config.text_hidden_dim if i == config.num_hidden_layers - 1 else end_scale[3]
            hw_fold = end_scale[1] // start_scale[1]
            t_fold = end_scale[0] // start_scale[0]
            self.encoder_layers.append(
                TmlVisionEncoderLayer(
                    input_dim=start_scale[3] * shuffle_mult, output_dim=output_dim, hw_fold=hw_fold, t_fold=t_fold
                )
            )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        num_patches = pixel_values.shape[0]
        hidden_states = pixel_values
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states=hidden_states)

        hidden_states = hidden_states.reshape(num_patches, -1)
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=hidden_states,
        )


class TmlModel(Gemma3Model, TmlPreTrainedModel):
    def __init__(self, config: TmlConfig):
        super().__init__(config)
        self.language_model = TmlTextModel(config.text_config)
        self.audio_tower = TmlAudioModel(config.audio_config)
        self.vision_tower = TmlVisionModel(config.vision_config)
        del self.multi_modal_projector

    def get_image_features(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        return self.vision_tower(pixel_values=pixel_values, return_dict=True, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **lm_kwargs: Unpack[TransformersKwargs],
    ) -> tuple | TmlModelOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, TmlForConditionalGeneration

        >>> model = TmlForConditionalGeneration.from_pretrained("google/tml2-3b-mix-224")
        >>> processor = AutoProcessor.from_pretrained("google/tml2-3b-mix-224")

        >>> prompt = "Where is the cat standing?"
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs,)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Where is the cat standing?\nsnow"
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

            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **lm_kwargs,
        )

        return TmlModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class TmlForConditionalGeneration(Gemma3ForConditionalGeneration, GenerationMixin):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        input_features: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | TmlCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, TmlForConditionalGeneration

        >>> model = TmlForConditionalGeneration.from_pretrained("google/gemma-3-4b-it")
        >>> processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

        >>> messages = [
        ...     {
        ...         "role": "system",
        ...         "content": [
        ...             {"type": "text", "text": "You are a helpful assistant."}
        ...         ]
        ...     },
        ...     {
        ...         "role": "user", "content": [
        ...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        ...             {"type": "text", "text": "Where is the cat standing?"},
        ...         ]
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(
        ...     messages,
        ...     tokenize=True,
        ...     return_dict=True,
        ...     return_tensors="pt",
        ...     add_generation_prompt=True
        ... )
        >>> # Generate
        >>> generate_ids = model.generate(**inputs)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "user\nYou are a helpful assistant.\n\n\n\n\n\nWhere is the cat standing?\nmodel\nBased on the image, the cat is standing in a snowy area, likely outdoors. It appears to"
        ```
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            input_features=input_features,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return TmlCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        input_features=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- custom `pixel_values/input_features` handling
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if is_first_iteration or not use_cache:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["input_features"] = input_features

        return model_inputs

    def create_masks_for_generate(self, **super_kwargs):
        raise AttributeError("Don't need to override for TML")


__all__ = [
    "TmlConfig",
    "TmlTextConfig",
    "TmlAudioConfig",
    "TmlVisionConfig",
    "TmlPreTrainedModel",
    "TmlTextModel",
    "TmlAudioModel",
    "TmlVisionModel",
    "TmlForConditionalGeneration",
]
