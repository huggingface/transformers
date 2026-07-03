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
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ..deepseek_v3 import DeepseekV3MLP, DeepseekV3MoE, DeepseekV3TopkRouter
from ..gemma3.modeling_gemma3 import Gemma3DecoderLayer, Gemma3ForConditionalGeneration, Gemma3Model
from ..gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding, apply_rotary_pos_emb
from ..llama.modeling_llama import LlamaAttention, LlamaRMSNorm, eager_attention_forward
from ..phi3.modeling_phi3 import Phi3MLP


logger = logging.get_logger(__name__)


class TmlTextConfig(PreTrainedConfig):
    model_type = "tml_text"
    base_config_key = "text_config"

    vocab_size: int = 151936
    padded_vocab_size: int | None = None
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
    logits_mup_width_multiplier: float | None = None
    rms_norm_eps_moe_gate: float = 1e-6
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | None = 2

    def __post_init__(self, **kwargs):
        self.padded_vocab_size = self.padded_vocab_size or self.vocab_size
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

    audio_mode: str = "dmel"
    n_mel_bins: int = 80
    mel_vocab_size: int = 256
    decoder_dmodel: int | None = None
    use_audio_norm: bool = True
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


# =============================================================================
# Norm / short-conv / rotary building blocks
# =============================================================================


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
        is_local = config.layer_types[self.layer_idx] == "sliding_attention"

        self.num_heads = config.swa_num_attention_heads if is_local else config.num_attention_heads
        self.num_key_value_heads = config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        self.head_dim = config.swa_head_dim if is_local else config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.sliding_window = config.sliding_window_size if is_local else None

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


class TmlDecoderLayer(Gemma3DecoderLayer):
    def __init__(self, config: TmlTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = TmlAttention(config, layer_idx)
        self.mlp = TmlMoE(config) if config.mlp_layer_types[layer_idx] == "moe" else TmlDenseMLP(config)
        self.input_layernorm = TmlRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TmlRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Gemma3 has extra pre/post-feedforward norms and a post-attn norm
        # applied *after* the residual add; TML's source doesn't, so drop
        # them rather than leaving unused/mismatched params.
        self.pre_feedforward_layernorm = nn.Identity()
        self.post_feedforward_layernorm = nn.Identity()

        self.attn_sconv = (
            TmlShortConvolution(config.hidden_size, config.sconv_kernel_size) if config.use_sconv else None
        )
        self.mlp_sconv = (
            TmlShortConvolution(config.hidden_size, config.sconv_kernel_size) if config.use_sconv else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        log_scaling_tau: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            log_scaling_tau=log_scaling_tau,
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


@auto_docstring
class TmlTextModel(TmlPreTrainedModel):
    config_class = TmlTextConfig
    _no_split_modules = ["TmlDecoderLayer"]

    def __init__(self, config: TmlTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.padded_vocab_size

        self.embed_tokens = nn.Embedding(config.padded_vocab_size, config.hidden_size, self.padding_idx)
        self.embed_norm = TmlRMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_embed_norm else None
        self.layers = nn.ModuleList(
            [TmlDecoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)]
        )
        self.norm = TmlRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = TmlRotaryEmbedding(config)
        self.has_sliding_layers = len(config.local_layer_ids) > 0
        self.gradient_checkpointing = False

        self.post_init()

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
            inputs_embeds = self.embed_tokens(input_ids)
            if self.embed_norm is not None:
                inputs_embeds = self.embed_norm(inputs_embeds)
        # NOTE: when `inputs_embeds` is passed in directly (e.g. from the
        # multimodal wrapper after scattering audio/vision features), we
        # assume embed_norm was already applied to the text portion upstream
        # -- mirrors the SGLang `TmlCausalLLM.forward` comment.

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
        causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        log_scaling_tau = compute_log_scaling_tau(
            position_ids, self.config.log_scaling_n_floor, self.config.log_scaling_alpha
        )

        for layer in self.layers:
            mask_type = "sliding_attention" if self.config.layer_types[layer.layer_id] else "full_attention"
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[mask_type],
                past_key_values=past_key_values,
                log_scaling_tau=log_scaling_tau,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class TmlAudioTower(TmlPreTrainedModel):
    pass


class TmlVisionTower(TmlPreTrainedModel):
    pass


class TmlModel(Gemma3Model, TmlPreTrainedModel):
    def __init__(self, config: TmlConfig):
        super().__init__(config)
        self.language_model = TmlTextModel(config.text_config)
        self.audio_tower = TmlAudioTower(config.audio_config)
        self.vision_tower = TmlVisionTower(config.vision_config)


class TmlForConditionalGeneration(Gemma3ForConditionalGeneration, GenerationMixin):
    pass


__all__ = [
    "TmlConfig",
    "TmlTextConfig",
    "TmlAudioConfig",
    "TmlVisionConfig",
    "TmlPreTrainedModel",
    "TmlTextModel",
    "TmlAudioTower",
    "TmlVisionTower",
    "TmlForConditionalGeneration",
]
