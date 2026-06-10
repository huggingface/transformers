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
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.output_capturing import OutputRecorder
from ..deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Moe,
)
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from ..mixtral.modeling_mixtral import (
    MixtralForCausalLM,
    MixtralModel,
    load_balancing_loss_func,
)
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeExperts, Qwen2MoeMLP
from ..qwen3.modeling_qwen3 import Qwen3Attention


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="bharatgenai/Param2-17B-A2.4B-Thinking")
@strict
class Param2MoEConfig(PreTrainedConfig):
    r"""
    first_k_dense_replace (`int`, *optional*, defaults to 1):
        Number of dense layers in the shallow layers before switching to MoE layers.
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    router_dtype (`str`, *optional*, defaults to `"fp32"`):
        Data type used for router weight computation. Using float32 improves numerical
        stability of the routing scores.
    partial_rotary_factor (`float`, *optional*, defaults to 1.0):
        Fraction of each attention head's dimension to apply rotary position embeddings
        to. A value of 1.0 applies RoPE to the full head dimension.
    moe_router_enable_expert_bias (`bool`, *optional*, defaults to `True`):
        Whether to add a per-expert learnable scalar bias to routing scores before
        top-k selection. The bias affects routing decisions only; output weights
        use unbiased scores to avoid distorting gradients.
    rope_theta (`float`, *optional*, defaults to 1000000.0):
        Base period (theta) for rotary position embeddings. Larger values extend
        the effective context length.
    score_function (`str`, *optional*, defaults to `"sigmoid"`):
        Activation function used to convert router logits to routing scores.
        `"sigmoid"` gives independent per-expert probabilities; `"softmax"` applies
        competitive normalization across all experts.
    torch_dtype (`str`, *optional*, defaults to `"bfloat16"`):
        Default torch dtype for model weights when loading with `from_pretrained`.

    Example:

    ```python
    >>> from transformers import Param2MoEModel, Param2MoEConfig
    >>> # Initializing a Param2MoE style configuration
    >>> configuration = Param2MoEConfig()
    >>> # Accessing the model configuration
    >>> model = Param2MoEModel(configuration)
    >>> print(model.config)
    ```
    """

    model_type = "param2moe"
    keys_to_ignore_at_inference = ["past_key_values"]

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

    vocab_size: int = 128008
    hidden_size: int = 2048
    intermediate_size: int = 9216
    num_hidden_layers: int = 21
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 3
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    attention_dropout: float | None = 0.0
    head_dim: int | None = 64
    first_k_dense_replace: int = 1
    n_group: int | None = 1
    num_experts: int = 64
    num_shared_experts: int = 2
    routed_scaling_factor: float = 2.5
    topk_group: int | None = 1
    norm_topk_prob: bool | None = True
    num_experts_per_tok: int | None = 6
    moe_intermediate_size: int = 2048
    rope_parameters: RopeParameters | dict | None = None
    router_aux_loss_coef: float = 0.0
    router_dtype: str = "fp32"
    partial_rotary_factor: float = 1.0
    max_window_layers: int = 20
    moe_router_enable_expert_bias: bool = True
    output_router_logits: bool = False
    sliding_window: int | None = None
    rope_theta: float = 1000000.0
    score_function: str = "sigmoid"
    torch_dtype: str = "bfloat16"

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})."
            )


class Param2MoEExperts(Qwen2MoeExperts):
    pass


class Param2MoERouter(nn.Module):
    """
    Sigmoid-based top-k router with per-expert learnable bias.

    Unlike softmax routers, sigmoid gives independent per-expert probabilities.
    The expert_bias shifts routing decisions only; output weights use unbiased
    scores to preserve gradient flow through expert outputs.
    """

    def __init__(self, config: Param2MoEConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.zeros(config.num_experts, config.hidden_size))
        self.expert_bias = nn.Parameter(torch.zeros(config.num_experts))

    def forward(self, hidden_states: torch.Tensor):
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = torch.sigmoid(router_logits)
        _, topk_idx = torch.topk(scores + self.expert_bias.float(), k=self.top_k, dim=-1, sorted=False)
        topk_weights = scores.gather(1, topk_idx)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        topk_weights = (topk_weights * self.routed_scaling_factor).to(hidden_states.dtype)
        return router_logits, topk_weights, topk_idx


class Param2MoESparseMoeBlock(DeepseekV2Moe):
    def __init__(self, config: Param2MoEConfig):
        super().__init__(config)
        self.gate = Param2MoERouter(config)
        self.experts = Param2MoEExperts(config)
        self.shared_experts = Param2MoEMLP(
            config,
            intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        batch_size, seq_len, hidden_dim = hidden_states.shape
        _, routing_weights, selected_experts = self.gate(hidden_states.view(-1, hidden_dim))
        hidden_states = self.experts(hidden_states.view(-1, hidden_dim), selected_experts, routing_weights)
        return hidden_states.view(batch_size, seq_len, hidden_dim) + self.shared_experts(residual)


class Param2MoEMLP(Qwen2MoeMLP):
    pass


class Param2MoERMSNorm(LlamaRMSNorm):
    pass


class Param2MoERotaryEmbedding(LlamaRotaryEmbedding):
    pass


class Param2MoEAttention(Qwen3Attention):
    def __init__(self, config: Param2MoEConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.layer_type
        self.sliding_window = getattr(config, "sliding_window", None)


class Param2MoEDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: Param2MoEConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.self_attn = Param2MoEAttention(config=config, layer_idx=layer_idx)
        self.mlp = (
            Param2MoESparseMoeBlock(config) if layer_idx >= config.first_k_dense_replace else Param2MoEMLP(config)
        )

        self.input_layernorm = Param2MoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Param2MoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Param2MoEPreTrainedModel(LlamaPreTrainedModel):
    config: Param2MoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Param2MoEDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(Param2MoERouter, index=0),
        "hidden_states": Param2MoEDecoderLayer,
        "attentions": Param2MoEAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, Param2MoEExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, Param2MoERouter):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.expert_bias is not None:
                init.zeros_(module.expert_bias)


class Param2MoEModel(MixtralModel):
    pass


class Param2MoEForCausalLM(MixtralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Param2MoEModel(config)
        self.num_experts = config.num_experts

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        output_router_logits (`bool`, *optional*):
            Whether to return raw router logits from every MoE layer.
            Required for computing the auxiliary load-balancing loss during training.
            Defaults to `config.output_router_logits`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Param2MoEForCausalLM

        >>> model = Param2MoEForCausalLM.from_pretrained("bharatgenai/Param2-17B-A2.4B-Thinking")
        >>> tokenizer = AutoTokenizer.from_pretrained("bharatgenai/Param2-17B-A2.4B-Thinking")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

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


__all__ = [
    "Param2MoEConfig",
    "Param2MoEPreTrainedModel",
    "Param2MoEModel",
    "Param2MoEForCausalLM",
]
