# Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch HYV3 model."""

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.output_capturing import OutputRecorder
from ..apertus.modeling_apertus import ApertusAttention
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3DecoderLayer
from ..llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from ..minimax_m2.modeling_minimax_m2 import MiniMaxM2Model, MiniMaxM2SparseMoeBlock
from ..mixtral.modeling_mixtral import MixtralTopKRouter
from ..qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="tencent/Hy3-preview")
@strict
class HYV3Config(PreTrainedConfig):
    r"""
    router_scaling_factor (*float*):
        Scaling factor on the top-k weighs of the MoE expert selection.
    enable_moe_fp32_combine (*bool*):
        Whether to add the shared experts to the final MoE result in fp32 or the base existing dtype of the model.
    mlp_layer_types (`list`, *optional*):
        MLP (Moe vs Dense) pattern for each layer.

    Example:
        ```python
        >>> from transformers import HYV3Config, HYV3Model

        >>> config = HYV3Config()
        >>> model = HYV3Model(config)
        ```
    """

    model_type = "hy_v3"
    default_theta = 11_158_840.0
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_local_experts": "num_experts",
    }
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
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

    vocab_size: int = 120832
    hidden_size: int = 4096
    intermediate_size: int = 13312
    num_hidden_layers: int = 80
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.006
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    num_experts: int | None = 192
    num_experts_per_tok: int | None = 8
    num_shared_experts: int | None = 1
    moe_intermediate_size: int = 1536
    router_scaling_factor: float = 2.826
    enable_moe_fp32_combine: bool = True
    mlp_layer_types: list[str] | None = None
    output_router_logits: bool = False
    rope_parameters: RopeParameters | dict | None = None

    def __post_init__(self, **kwargs):
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * (1 if self.num_hidden_layers > 0 else 0) + ["sparse"] * max(
                self.num_hidden_layers - 1, 0
            )

        super().__post_init__(**kwargs)


class HYV3RMSNorm(LlamaRMSNorm):
    pass


class HYV3RotaryEmbedding(LlamaRotaryEmbedding):
    pass


class HYV3MLP(LlamaMLP):
    def __init__(self, config: HYV3Config, intermediate_size: int | None = None):
        super().__init__(config)
        self.intermediate_size = intermediate_size or config.intermediate_size


class HYV3Attention(ApertusAttention):
    pass


class HYV3TopKRouter(MixtralTopKRouter):
    def __init__(self, config: HYV3Config):
        super().__init__(config)
        self.router_scaling_factor = config.router_scaling_factor

    def forward(
        self,
        hidden_states: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        routing_weights = torch.sigmoid(router_logits)

        scores_for_choice = routing_weights + e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)

        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-20)
        # Key difference: extra scaling factor
        top_k_weights = top_k_weights * self.router_scaling_factor

        return router_logits, top_k_weights, top_k_index


class HYV3Experts(Qwen3MoeExperts):
    pass


class HYV3MoE(MiniMaxM2SparseMoeBlock):
    def __init__(self, config: HYV3Config):
        super().__init__(config)
        del self.jitter_noise
        self.enable_moe_fp32_combine = config.enable_moe_fp32_combine
        self.gate = HYV3TopKRouter(config)
        self.experts = HYV3Experts(config)
        shared_intermediate = config.moe_intermediate_size * config.num_shared_experts
        self.shared_experts = HYV3MLP(config, intermediate_size=shared_intermediate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        _, top_k_weights, top_k_index = self.gate(hidden_states, self.e_score_correction_bias)
        routed_output = self.experts(hidden_states, top_k_index, top_k_weights)

        # Key difference: optional float casting on combing shared experts
        if self.enable_moe_fp32_combine:
            hidden_states = (routed_output.float() + self.shared_experts(hidden_states).float()).to(
                hidden_states.dtype
            )
        else:
            hidden_states = routed_output + self.shared_experts(hidden_states)

        return hidden_states.reshape(batch_size, seq_len, hidden_dim)


class HYV3DecoderLayer(DeepseekV3DecoderLayer):
    def __init__(self, config: HYV3Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = HYV3Attention(config=config, layer_idx=layer_idx)
        self.mlp = HYV3MoE(config) if config.mlp_layer_types[layer_idx] == "sparse" else HYV3MLP(config)
        self.input_layernorm = HYV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HYV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class HYV3PreTrainedModel(LlamaPreTrainedModel):
    # Not supporting multi-token prediction (MTP) atm
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.80.*"]
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _can_record_outputs = {
        "router_logits": OutputRecorder(HYV3TopKRouter, index=0),
        "hidden_states": HYV3DecoderLayer,
        "attentions": HYV3Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, HYV3TopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, HYV3Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, HYV3MoE):
            init.zeros_(module.e_score_correction_bias)


class HYV3Model(MiniMaxM2Model):
    def __init__(self, config: HYV3Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [HYV3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HYV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HYV3RotaryEmbedding(config=config)


class HYV3ForCausalLM(LlamaForCausalLM):
    def __init__(self, config: HYV3Config):
        super().__init__(config)
        self.model = HYV3Model(config)

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
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        outputs: MoeModelOutputWithPast = self.model(
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

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=None,  # Not used in this model
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = [
    "HYV3Config",
    "HYV3ForCausalLM",
    "HYV3Model",
    "HYV3PreTrainedModel",
]
