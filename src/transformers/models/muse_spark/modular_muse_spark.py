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
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...integrations import use_experts_implementation
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MoeModelOutputWithPast,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ProcessorMixin, Unpack
from ...utils import TransformersKwargs, auto_docstring, logging, torch_compilable_check
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ...vision_utils import get_vision_interpolation_indices_and_weights
from ..auto import AutoModel
from ..clip.modeling_clip import CLIPMLP
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3Experts
from ..minimax_m2.modeling_minimax_m2 import MiniMaxM2TopKRouter
from ..muse_glimmer.configuration_muse_glimmer import MuseGlimmerConfig, MuseGlimmerTextConfig, MuseGlimmerVisionConfig
from ..muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerCausalLMOutputWithPast,
    MuseGlimmerForConditionalGeneration,
    MuseGlimmerModel,
    MuseGlimmerModelOutputWithPast,
    MuseGlimmerPreTrainedModel,
    MuseGlimmerRMSNorm,
    MuseGlimmerTextAttention,
    MuseGlimmerTextCenteredRMSNorm,
    MuseGlimmerTextDecoderLayer,
    MuseGlimmerTextMLP,
    MuseGlimmerTextModel,
    MuseGlimmerVisionModel,
    MuseGlimmerVisionPatchEmbedder,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..muse_glimmer.processing_muse_glimmer import MuseGlimmerProcessor
from ..voxtral_realtime.modeling_voxtral_realtime import (
    VoxtralRealtimeCausalConv1d,
    VoxtralRealtimeConv1dPaddingCache,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="meta-models/Muse-Spark-Open-816B-A42B")
@strict
class MuseSparkVisionConfig(MuseGlimmerVisionConfig):
    model_type = "muse_spark_vision"

    base_model_tp_plan = {}


@auto_docstring(checkpoint="meta-models/Muse-Spark-Open-816B-A42B")
@strict
class MuseSparkTextConfig(MuseGlimmerTextConfig):
    r"""
    qk_scale_factor (`float`, *optional*, defaults to 5.4730064863875):
        Multiplier applied to Q after the scaleless QK-norm, on top of the standard `1/sqrt(head_dim)`
        attention scaling.
    output_multiplier (`float`, *optional*, defaults to 0.1767766952966369):
        Scale applied to logits before the final tanh softcap. Equal to `1/sqrt(hidden_size / 256)` for the
        released checkpoint.
    post_norm_eps (`float`, *optional*, defaults to 1e-8):
        Epsilon used for the post-attention and post-FFN norms (which sit between the sub-layer output and the residual).
    layer_rope_theta (`list[float]`, *optional*):
        Per-layer RoPE base theta; `0` disables rotary (NoPE) for that layer. Overrides the global
        `rope_parameters["rope_theta"]`. Defaults to the global theta everywhere except every 4th layer
        counted backward from the last, which is NoPE.
    moe_hidden_size (`int`, *optional*, defaults to 4096):
        Width the MoE block pools `hidden_size` down to before the routed experts, and unpools from
        afterwards. The experts run entirely at this width.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer MLP type — `"dense"` or `"sparse"`. Length must equal
        `num_hidden_layers`. Defaults to all sparse.
    residual_gate_temperature (`float`, *optional*, defaults to 0.3):
        Temperature applied to the residual gate parameter before the sigmoid.
    """

    model_type = "muse_spark_text"

    base_model_tp_plan = {
        "layers.*.mlp.pre_expert_proj": "rowwise_split_input",
        "layers.*.mlp.post_expert_proj": "rowwise_split_input",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_ep_plan = {
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts.post_expert_norm": "replicated_with_grad_allreduce",
        "layers.*.mlp.experts": "ep_dispatch_experts",
    }

    hidden_size: int = 8192
    intermediate_size: int = 16384
    num_hidden_layers: int = 62
    num_attention_heads: int = 128
    num_key_value_heads: int = 16
    head_dim: int = 64
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576

    qk_scale_factor: float = 5.4730064863875
    output_multiplier: float = 0.1767766952966369

    moe_intermediate_size: int = 4096
    num_local_experts: int = 256
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 1.0
    moe_hidden_size: int = 4096
    mlp_layer_types: list[str] | None = None
    residual_gate_temperature: float = 0.3
    output_router_logits: bool = False

    hidden_activation = AttributeError()

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 500_000.0}
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["sparse"] * self.num_hidden_layers
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="meta-models/Muse-Spark-Open-816B-A42B")
@strict
class MuseSparkAudioConfig(PreTrainedConfig):
    r"""
    patch_sizes (`list[int] | tuple[int, ...]`, *optional*, defaults to `(16, 6, 5, 4)`):
        Temporal downsampling factor applied by each stage's patch embedding. Their product is the
        number of waveform samples collapsed into one output token.
    hidden_sizes (`list[int] | tuple[int, ...]`, *optional*, defaults to `(256, 512, 1024, 2048)`):
        Hidden width of each stage.
    depths (`list[int] | tuple[int, ...]`, *optional*, defaults to `(2, 2, 2, 2)`):
        Number of conv blocks in each layer.
    num_hidden_layers (`int`, *optional*, defaults to 4):
        Number of `MuseSparkAudioLayer`s, i.e. resolutions in the pyramid. Redundant with
        `len(hidden_sizes)` and validated against it, so a custom pyramid must set both.
    mlp_ratio (`float`, *optional*, defaults to 4.0):
        Ratio of a layer's MLP intermediate width to its hidden width.
    conv_kernel_size (`int`, *optional*, defaults to 7):
        Kernel size of the depthwise causal convolutions.
    out_hidden_size (`int`, *optional*, defaults to 8192):
        Width the final projection maps to, i.e. the text model's hidden size.

    Example:

    ```python
    >>> from transformers import MuseSparkAudioConfig, MuseSparkAudioModel

    >>> configuration = MuseSparkAudioConfig()
    >>> model = MuseSparkAudioModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "muse_spark_audio"

    patch_sizes: list[int] | tuple[int, ...] = (16, 6, 5, 4)
    hidden_sizes: list[int] | tuple[int, ...] = (256, 512, 1024, 2048)
    depths: list[int] | tuple[int, ...] = (2, 2, 2, 2)
    num_hidden_layers: int = 4
    mlp_ratio: float = 4.0
    conv_kernel_size: int = 7
    hidden_act: str = "gelu"
    rms_norm_eps: float = 1e-6
    out_hidden_size: int = 8192
    initializer_range: float = 0.02

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if not len(self.patch_sizes) == len(self.hidden_sizes) == len(self.depths) == self.num_hidden_layers:
            raise ValueError(
                "`patch_sizes`, `hidden_sizes` and `depths` must have one entry per layer, got "
                f"{len(self.patch_sizes)}, {len(self.hidden_sizes)} and {len(self.depths)} "
                f"for `num_hidden_layers`={self.num_hidden_layers}."
            )

    @property
    def samples_per_token(self) -> int:
        """Waveform samples collapsed into a single output token (1920 == 80 ms at 24 kHz)."""
        return math.prod(self.patch_sizes)


@auto_docstring(checkpoint="meta-models/Muse-Spark-Open-816B-A42B")
@strict
class MuseSparkConfig(MuseGlimmerConfig):
    model_type = "muse_spark"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=MuseSparkTextConfig),
        "vision_config": SubConfigSpec(config_class=MuseSparkVisionConfig),
        "audio_config": SubConfigSpec(config_class=MuseSparkAudioConfig),
    }

    audio_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 200038

    def validate_architecture(self):
        if self.audio_config.out_hidden_size != self.text_config.hidden_size:
            raise ValueError(
                f"`audio_config.out_hidden_size` ({self.audio_config.out_hidden_size}) must match "
                f"`text_config.hidden_size` ({self.text_config.hidden_size}): the audio encoder's "
                "output is scattered directly onto the text embeddings."
            )


class MuseSparkProcessor(MuseGlimmerProcessor):
    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        tokenizer=None,
        feature_extractor=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = getattr(tokenizer, "image_token", "<|patch|>")
        self.image_start_token = getattr(tokenizer, "image_start_token", "<|image_start|>")
        self.image_end_token = getattr(tokenizer, "image_end_token", "<|image_end|>")
        self.video_token = getattr(tokenizer, "video_token", "<|video|>")
        self.video_sep_token = getattr(tokenizer, "video_sep_token", "<|vid_frame_separator|>")
        self.video_start_token = getattr(tokenizer, "video_start_token", "<|vid_start|>")
        self.video_end_token = getattr(tokenizer, "video_end_token", "<|vid_end|>")
        self.audio_token = getattr(tokenizer, "audio_token", "<|audio|>")
        self.audio_start_token = getattr(tokenizer, "audio_start_token", "<|soft_audio_start|>")
        self.audio_end_token = getattr(tokenizer, "audio_end_token", "<|soft_audio_end|>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_start_token_id = tokenizer.convert_tokens_to_ids(self.image_start_token)
        self.image_end_token_id = tokenizer.convert_tokens_to_ids(self.image_end_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)

        ProcessorMixin.__init__(
            self,
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            chat_template=chat_template,
            **kwargs,
        )

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        samples_per_token = self.feature_extractor.samples_per_token
        mask = audio_inputs.get("input_values_mask")
        if mask is not None:
            num_samples = int(mask[audio_idx].sum())
        else:
            num_samples = int(audio_inputs["input_values"][audio_idx].shape[-1])
        num_audio_tokens = -(-num_samples // samples_per_token)
        return self.audio_start_token + self.audio_token * num_audio_tokens + self.audio_end_token


class MuseSparkModelOutputWithPast(MuseGlimmerModelOutputWithPast):
    router_logits: tuple[torch.FloatTensor] | None = None


class MuseSparkCausalLMOutputWithPast(MuseGlimmerCausalLMOutputWithPast):
    router_logits: tuple[torch.FloatTensor] | None = None


class MuseSparkRMSNorm(MuseGlimmerRMSNorm):
    pass


class MuseSparkTextCenteredRMSNorm(MuseGlimmerTextCenteredRMSNorm):
    def forward(self, x):
        output = self._norm(x.float())
        # (x * w).to(bfloat16) instead of x.to(bfloat16) * w
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight).float()
        return output.type_as(x)


class MuseSparkTextInputScaledRMSNorm(MuseGlimmerTextCenteredRMSNorm):
    def forward(self, x):
        # Different order from MuseSparkTextCenteredRMSNorm: first scale, then normalize.
        output = x.float() * (1.0 + self.weight).float()
        output = self._norm(output.type_as(x).float())  # round with fp32->bf16->f32 before norm
        return output.type_as(x)


class MuseSparkTextMLP(MuseGlimmerTextMLP):
    def __init__(self, config: MuseSparkTextConfig):
        super().__init__(config)
        self.act_fn = ACT2FN[config.hidden_act]


class MuseSparkTopkRouter(MiniMaxM2TopKRouter):
    def __init__(self, config: MuseSparkTextConfig):
        super().__init__(config)
        self.output_multiplier = config.output_multiplier
        self.scoring_function = ACT2FN["sigmoid"]
        self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts))
        self.norm_topk_prob_eps = 1e-15
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        # Different from MiniMaxM2: convert to float32 before linear projection and scale by output multiplier.
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        router_logits = router_logits * self.output_multiplier
        routing_weights = self.scoring_function(router_logits)
        scores_for_choice = routing_weights + self.e_score_correction_bias
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        top_k_weights = routing_weights.gather(1, topk_indices)
        # Different from MiniMaxM2: add topk prob eps and scale by routed scaling factor.
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + self.norm_topk_prob_eps)
        top_k_weights = top_k_weights * self.routed_scaling_factor
        return router_logits, top_k_weights, topk_indices


@use_experts_implementation(has_post_expert_norm=True)
class MuseSparkExperts(DeepseekV3Experts):
    def __init__(self, config: MuseSparkTextConfig):
        super().__init__(config)
        self.hidden_dim = config.moe_hidden_size
        self.post_expert_norm = MuseSparkTextInputScaledRMSNorm(self.hidden_dim, eps=config.post_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            # Different from DeepseekV3 which has no post expert norm
            current_hidden_states = self.post_expert_norm(current_hidden_states)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class MuseSparkMoE(nn.Module):
    def __init__(self, config: MuseSparkTextConfig):
        super().__init__()
        self.config = config
        self.gate = MuseSparkTopkRouter(config)
        self.pre_expert_proj = nn.Linear(config.hidden_size, config.moe_hidden_size, bias=False)
        self.pre_expert_norm = MuseSparkTextCenteredRMSNorm(config.moe_hidden_size, eps=config.rms_norm_eps)
        self.experts = MuseSparkExperts(config)
        self.post_expert_proj = nn.Linear(config.moe_hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        _, topk_weights, topk_indices = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.pre_expert_proj(hidden_states)
        hidden_states = self.pre_expert_norm(hidden_states)
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights)
        hidden_states = self.post_expert_proj(hidden_states).view(*orig_shape)
        return hidden_states


class MuseSparkTextAttention(MuseGlimmerTextAttention):
    def __init__(self, config: MuseSparkTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.o_norm = MuseSparkRMSNorm(eps=config.rms_norm_eps, with_scale=False)

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

        query_states = self.qk_norm(query_states) * self.qk_scale_factor
        key_states = self.qk_norm(key_states)

        # NoPE layers receive `position_embeddings=None` from the model.
        if position_embeddings is not None:
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
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = self.o_norm(attn_output).reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(self.gate_proj(hidden_states))
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MuseSparkResidualGate(nn.Module):
    def __init__(self, config: MuseSparkTextConfig, residual_gate_eps: float = 1e-3):
        super().__init__()
        self.gate = nn.Parameter(torch.empty(config.hidden_size))
        self.temperature = config.residual_gate_temperature
        self.eps = residual_gate_eps

    def forward(self, residual: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        w = self.gate / self.temperature
        beta = torch.sigmoid(w)
        alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=self.eps))
        return (alpha * residual + beta * hidden_states).to(residual.dtype)


class MuseSparkTextDecoderLayer(MuseGlimmerTextDecoderLayer):
    def __init__(self, config: MuseSparkTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = MuseSparkMoE(config) if config.mlp_layer_types[layer_idx] == "sparse" else MuseSparkTextMLP(config)
        self.post_attention_layernorm = MuseSparkRMSNorm(
            config.hidden_size, eps=config.post_norm_eps, with_scale=False
        )
        self.post_attention_residual_gate = MuseSparkResidualGate(config)
        self.post_feedforward_layernorm = MuseSparkTextInputScaledRMSNorm(config.hidden_size, eps=config.post_norm_eps)
        self.post_feedforward_residual_gate = MuseSparkResidualGate(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.post_attention_residual_gate(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = self.post_feedforward_residual_gate(residual, hidden_states)

        return hidden_states


class MuseSparkPreTrainedModel(MuseGlimmerPreTrainedModel):
    _keep_in_fp32_modules_strict = [
        "post_attention_residual_gate",
        "post_feedforward_residual_gate",
        "mlp.gate.weight",
        "e_score_correction_bias",
    ]

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, MuseSparkTopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, MuseSparkExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (MuseSparkTextCenteredRMSNorm, MuseSparkTextInputScaledRMSNorm)):
            init.zeros_(module.weight)
        elif isinstance(module, MuseSparkResidualGate):
            init.normal_(module.gate, mean=0.0, std=self.config.initializer_range)


class MuseSparkTextModel(MuseGlimmerTextModel):
    _can_record_outputs = {
        "hidden_states": MuseSparkTextDecoderLayer,
        "attentions": MuseSparkTextAttention,
        "router_logits": OutputRecorder(MuseSparkTopkRouter, index=0),
    }

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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

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
                # NoPE layers (layer_rope_theta == 0) get no position embeddings.
                position_embeddings=position_embeddings if self.config.layer_rope_theta[i] else None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class MuseSparkVisionPatchEmbedder(MuseGlimmerVisionPatchEmbedder):
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        embeddings = self.patch_embedding(pixel_values.to(dtype=self.patch_embedding.weight.dtype))

        # Patch (h, w) reads the position table at (w, h): ref code passes
        # `meshgrid(pos_h, pos_w, indexing="xy")` to `grid_sample` which takes first coord as width.
        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            grid_thw[:, [0, 2, 1]],
            num_grid_per_side=self.num_grid_per_side,
            mode=self.interpolation_mode,
            align_corners=self.interpolation_align_corners,
            spatial_merge_size=1,
            padding=self.interpolation_padding,
            kwargs=kwargs,
        )
        pos_embeds = (self.position_embedding_table(interp_indices) * interp_weights[:, :, None]).sum(1)
        return embeddings + pos_embeds.to(embeddings.dtype)


class MuseSparkVisionModel(MuseGlimmerVisionModel):
    pass


class MuseSparkAudioConv1dPaddingCache(VoxtralRealtimeConv1dPaddingCache):
    pass


class MuseSparkAudioCausalConv1d(VoxtralRealtimeCausalConv1d):
    pass


class MuseSparkAudioPatchEmbedding(nn.Module):
    def __init__(self, config: MuseSparkAudioConfig, layer_idx: int):
        super().__init__()
        # layer 0 consumes the mono waveform itself, so a single input channel
        in_channels = 1 if layer_idx == 0 else config.hidden_sizes[layer_idx - 1]
        out_channels = config.hidden_sizes[layer_idx]
        self.patch_size = config.patch_sizes[layer_idx]
        self.proj = nn.Linear(in_channels * self.patch_size, out_channels, bias=True)
        self.norm = MuseSparkRMSNorm(out_channels, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = hidden_states.shape
        torch_compilable_check(
            seq_len % self.patch_size == 0, f"audio length {seq_len} is not divisible by patch size {self.patch_size}"
        )
        hidden_states = hidden_states.reshape(batch_size, seq_len // self.patch_size, self.patch_size * channels)
        return self.norm(self.proj(hidden_states))


class MuseSparkAudioMLP(CLIPMLP):
    def __init__(self, config: MuseSparkAudioConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_sizes[layer_idx]
        self.intermediate_size = int(self.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.activation_fn = ACT2FN[config.hidden_act]


class MuseSparkAudioConvBlock(nn.Module):
    def __init__(self, config: MuseSparkAudioConfig, layer_idx: int, block_idx: int):
        super().__init__()
        hidden_size = config.hidden_sizes[layer_idx]
        self.conv = MuseSparkAudioCausalConv1d(
            hidden_size,
            hidden_size,
            kernel_size=config.conv_kernel_size,
            cache_key=f"layers.{layer_idx}.conv_blocks.{block_idx}",
            groups=hidden_size,
            bias=True,
        )
        self.norm = MuseSparkRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.mlp = MuseSparkAudioMLP(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_cache: MuseSparkAudioConv1dPaddingCache | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv(hidden_states.transpose(1, 2), padding_cache).transpose(1, 2)
        hidden_states = self.mlp(self.norm(hidden_states))
        return hidden_states + residual


class MuseSparkAudioLayer(nn.Module):
    def __init__(self, config: MuseSparkAudioConfig, layer_idx: int):
        super().__init__()
        self.patch_embedder = MuseSparkAudioPatchEmbedding(config, layer_idx)
        self.conv_blocks = nn.ModuleList(
            [MuseSparkAudioConvBlock(config, layer_idx, block_idx) for block_idx in range(config.depths[layer_idx])]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_cache: MuseSparkAudioConv1dPaddingCache | None = None,
    ) -> torch.Tensor:
        hidden_states = self.patch_embedder(hidden_states)
        for conv_block in self.conv_blocks:
            hidden_states = conv_block(hidden_states, padding_cache)
        return hidden_states


@auto_docstring(
    custom_intro="""
    The Muse Spark Open waveform audio encoder: a fully causal convolutional pyramid that maps raw mono
    24 kHz audio to one embedding per 1920 samples (80 ms, i.e. 12.5 Hz), already in the text model's
    hidden size. It uses no attention and no spectrogram front-end.
    """
)
class MuseSparkAudioModel(MuseSparkPreTrainedModel):
    config: MuseSparkAudioConfig
    main_input_name = "input_values"
    input_modalities = "audio"
    _no_split_modules = ["MuseSparkAudioLayer"]
    _can_record_outputs = {"hidden_states": MuseSparkAudioLayer}

    def __init__(self, config: MuseSparkAudioConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([MuseSparkAudioLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = MuseSparkRMSNorm(config.hidden_sizes[-1], eps=config.rms_norm_eps)
        self.proj = nn.Linear(config.hidden_sizes[-1], config.out_hidden_size, bias=True)
        self.post_proj_norm = MuseSparkRMSNorm(config.out_hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor,
        padding_cache: MuseSparkAudioConv1dPaddingCache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        padding_cache (`MuseSparkAudioConv1dPaddingCache`, *optional*):
            Cache for padding in convolutional layers to maintain state across streaming chunks.
        """
        torch_compilable_check(
            input_values.shape[1] % self.config.samples_per_token == 0,
            (
                f"`input_values` length {input_values.shape[1]} is not a multiple of "
                f"{self.config.samples_per_token}. Prepare the audio with `MuseSparkFeatureExtractor`, which "
                f"pads up to a multiple of it."
            ),
        )
        hidden_states = input_values.unsqueeze(-1)
        for layer in self.layers:
            hidden_states = layer(hidden_states, padding_cache)

        hidden_states = self.post_proj_norm(self.proj(self.norm(hidden_states)))
        return BaseModelOutput(last_hidden_state=hidden_states)


class MuseSparkModel(MuseGlimmerModel):
    def __init__(self, config: MuseSparkConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.audio_tower = AutoModel.from_config(config.audio_config)

    @can_return_tuple
    @auto_docstring
    def get_audio_features(
        self,
        input_values: torch.Tensor,
        input_values_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        input_values_mask (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*):
            Which entries of `input_values` are real audio rather than padding, as produced by
            [`MuseSparkFeatureExtractor`]. Required when batching clips of different lengths.
        """
        audio_outputs = self.audio_tower(input_values=input_values, **kwargs)
        last_hidden_state = audio_outputs.last_hidden_state
        audio_features = last_hidden_state

        if input_values_mask is not None:
            samples_per_token = self.config.audio_config.samples_per_token
            # right-padding only, which is what MuseSparkFeatureExtractor produces
            lengths = input_values_mask.sum(-1)
            num_tokens = (lengths + samples_per_token - 1) // samples_per_token
            positions = torch.arange(audio_features.shape[1], device=audio_features.device)
            token_mask = positions[None, :] < num_tokens[:, None]
            audio_features = audio_features[token_mask]
        else:
            audio_features = audio_features.reshape(-1, audio_features.shape[-1])

        audio_features = self.get_input_embeddings().embed_norm(audio_features)
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=audio_features,
            hidden_states=audio_outputs.hidden_states,
        )

    def get_audio_placeholder_mask(self, input_ids, inputs_embeds, audio_features):
        if input_ids is None:
            audio_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full((), self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            audio_mask = audio_mask.all(-1)
        else:
            audio_mask = input_ids == self.config.audio_token_id

        n_audio_tokens = audio_mask.sum()
        audio_mask = audio_mask.unsqueeze(-1).to(inputs_embeds.device)
        torch_compilable_check(
            n_audio_tokens * inputs_embeds.shape[-1] == audio_features.numel(),
            f"Audio features and audio tokens do not match, tokens: {n_audio_tokens}, "
            f"features: {audio_features.shape[0]}",
        )
        return audio_mask

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
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        input_values: torch.Tensor | None = None,
        input_values_mask: torch.Tensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MuseSparkModelOutputWithPast:
        r"""
        input_values_mask (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*):
            Which entries of `input_values` are real audio rather than padding, as produced by
            [`MuseSparkFeatureExtractor`]. Required when batching clips of different lengths.

        Example:

        ```python
        >>> import os

        >>> from transformers import AutoProcessor, DistributedConfig, MuseSparkModel

        >>> distributed_config = DistributedConfig(
        ...     tp_size=int(os.environ["WORLD_SIZE"]),
        ...     ep_size=int(os.environ["WORLD_SIZE"]),
        ... )

        >>> processor = AutoProcessor.from_pretrained("meta-models/Muse-Spark-Open-816B-A42B")
        >>> model = MuseSparkModel.from_pretrained(
        ...     "meta-models/Muse-Spark-Open-816B-A42B",
        ...     dtype="auto",
        ...     distributed_config=distributed_config,
        ... )

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        ...             {"type": "text", "text": "What is shown in this image?"},
        ...         ],
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(
        ...     messages,
        ...     reasoning_effort="medium",
        ...     add_generation_prompt=True,
        ...     tokenize=True,
        ...     return_dict=True,
        ...     return_tensors="pt",
        ... ).to(model.device).to(model.dtype)

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        ```
        """
        if (pixel_values is not None or pixel_values_videos is not None) and mm_encoder_outputs is not None:
            raise ValueError(
                "You cannot specify both pixel_values/pixel_values_videos and mm_encoder_outputs at the same time"
            )

        if inputs_embeds is None:
            multimodal_mask = (
                (input_ids == self.config.image_token_id)
                | (input_ids == self.config.video_token_id)
                | (input_ids == self.config.audio_token_id)
            )
            llm_input_ids = input_ids.clone()
            llm_input_ids[multimodal_mask] = 0
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True, **kwargs
            )

        if mm_encoder_outputs.get("video") is None and pixel_values_videos is not None:
            mm_encoder_outputs["video"] = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True, **kwargs
            )

        if mm_encoder_outputs.get("image") is not None:
            image_embeds = torch.cat(mm_encoder_outputs["image"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if mm_encoder_outputs.get("video") is not None:
            video_embeds = torch.cat(mm_encoder_outputs["video"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if input_values is not None:
            audio_embeds = self.get_audio_features(input_values, input_values_mask).pooler_output
            audio_embeds = audio_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            audio_mask = self.get_audio_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, audio_features=audio_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return MuseSparkModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class MuseSparkForConditionalGeneration(MuseGlimmerForConditionalGeneration):
    _tp_plan = AttributeError()
    # Tying lm_head.weight and model.language_model.embed_tokens.weight is currently not supported
    # as it requires a custom tp plan
    _tied_weights_keys = AttributeError()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        input_values: torch.Tensor | None = None,
        input_values_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MuseSparkCausalLMOutputWithPast:
        r"""
        input_values_mask (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*):
            Which entries of `input_values` are real audio rather than padding, as produced by
            [`MuseSparkFeatureExtractor`]. Required when batching clips of different lengths.

        Example:

        ```python
        >>> import os

        >>> from transformers import AutoProcessor, DistributedConfig, MuseSparkForConditionalGeneration

        >>> distributed_config = DistributedConfig(
        ...     tp_size=int(os.environ["WORLD_SIZE"]),
        ...     ep_size=int(os.environ["WORLD_SIZE"]),
        ... )

        >>> processor = AutoProcessor.from_pretrained("meta-models/Muse-Spark-Open-816B-A42B")
        >>> model = MuseSparkForConditionalGeneration.from_pretrained(
        ...     "meta-models/Muse-Spark-Open-816B-A42B",
        ...     dtype="auto",
        ...     distributed_config=distributed_config,
        ... )

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        ...             {"type": "text", "text": "What is shown in this image?"},
        ...         ],
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(
        ...     messages,
        ...     reasoning_effort="medium",
        ...     add_generation_prompt=True,
        ...     tokenize=True,
        ...     return_dict=True,
        ...     return_tensors="pt",
        ... ).to(model.device).to(model.dtype)
        >>> input_len = inputs["input_ids"].shape[-1]

        >>> outputs = model.generate(**inputs, max_new_tokens=512)

        >>> response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        >>> print(response)
        The image shows a **Pallas's cat — also known as a manul, *Otocolobus manul*** — walking through snow.
        ...
        ```
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            input_values=input_values,
            input_values_mask=input_values_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # MuseSpark pre-scales logits by `output_multiplier` before the Gemma-style tanh softcap.
        # Together with `final_logit_softcapping = T`, this gives `T * tanh(logits * mult / T)`.
        logits = logits * self.config.text_config.output_multiplier
        logits = logits / self.config.text_config.final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * self.config.text_config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.text_config.vocab_size, **kwargs)

        return MuseSparkCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            router_logits=outputs.router_logits,
        )


__all__ = [
    "MuseSparkTextConfig",
    "MuseSparkVisionConfig",
    "MuseSparkAudioConfig",
    "MuseSparkConfig",
    "MuseSparkPreTrainedModel",
    "MuseSparkTextModel",
    "MuseSparkVisionModel",
    "MuseSparkAudioModel",
    "MuseSparkModel",
    "MuseSparkForConditionalGeneration",
    "MuseSparkProcessor",
]
