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
from torch import nn
 
from ...cache_utils import Cache
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ...utils.output_capturing import OutputRecorder
from transformers.configuration_utils import PretrainedConfig
from ..llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaDecoderLayer,
    LlamaRMSNorm,
)
from ..mixtral.modeling_mixtral import (
    MixtralForCausalLM,
    MixtralModel,
    MixtralPreTrainedModel,
    load_balancing_loss_func,
)
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeExperts, Qwen2MoeMLP
from ..qwen3.modeling_qwen3 import Qwen3Attention
from .configuration_param2moe import Param2MoEConfig


logger = logging.get_logger(__name__)


class Param2MoEExperts(Qwen2MoeExperts):
    pass


class Param2MoERouter(nn.Module):
    """
    Sigmoid-based top-k router with optional per-expert learnable bias.
 
    Design:
    - Raw logits computed in float32 for numerical stability
    - score_function="sigmoid" → independent per-expert probabilities
      (no competition via softmax normalization)
    - Expert bias added to scores for the routing *decision* only;
      the actual output *weights* use the unbiased scores so that the
      bias does not distort gradient flow through the expert outputs
    - Group-limited top-k: when n_group > 1, experts are split into
      n_group groups; only topk_group groups are considered before
      running top-k inside the selected groups
    - Optional weight normalization (norm_topk_prob) and scaling
      (routed_scaling_factor) applied after selection
 
    Returns:
        router_logits  : raw pre-activation logits  [tokens, num_experts]
                         (captured by OutputRecorder for aux loss; index=0)
        topk_weights   : unbiased, (optionally normalized) scaled weights
                         [tokens, num_experts_per_tok]
        topk_idx       : indices of selected experts
                         [tokens, num_experts_per_tok]
    """
 
    def __init__(self, config: Param2MoEConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.score_function = config.score_function
        self.weight = nn.Parameter(torch.zeros(config.num_experts, config.hidden_size))
 
        # Per-expert learnable bias for routing decisions
        if config.moe_router_enable_expert_bias:
            self.expert_bias = nn.Parameter(torch.zeros(config.num_experts))
        else:
            self.expert_bias = None
 
    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: float tensor of shape [num_tokens, hidden_size]
 
        Returns:
            Tuple of (router_logits, topk_weights, topk_idx)
        """
        # ---- 1. Compute raw logits in fp32 --------------------------------
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        # router_logits: [tokens, num_experts]
 
        # ---- 2. Compute unbiased scores (used as output weights) ----------
        if self.score_function == "sigmoid":
            scores = torch.sigmoid(router_logits)
        else:  # "softmax"
            scores = torch.softmax(router_logits, dim=-1)
 
        # ---- 3. Compute biased scores (used for routing decision only) ----
        if self.expert_bias is not None:
            scores_for_routing = scores + self.expert_bias.float()
        else:
            scores_for_routing = scores
 
        # ---- 4. Group-limited top-k (no-op when n_group == 1) -------------
        if self.n_group > 1 and self.topk_group < self.n_group:
            num_tokens = hidden_states.shape[0]
            experts_per_group = self.num_experts // self.n_group
 
            # Max score per group → select topk_group best groups
            group_scores = scores_for_routing.view(num_tokens, self.n_group, experts_per_group).max(dim=-1).values
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
 
            # Build a mask that zeros out experts in non-selected groups
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(num_tokens, self.n_group, experts_per_group)
                .reshape(num_tokens, -1)
            )
            scores_for_routing = scores_for_routing.masked_fill(~score_mask.bool(), 0.0)
 
        # ---- 5. Select top-k experts using biased scores ------------------
        _, topk_idx = torch.topk(scores_for_routing, k=self.top_k, dim=-1, sorted=False)
 
        # ---- 6. Gather *unbiased* scores as actual output weights ---------
        topk_weights = scores.gather(1, topk_idx)
 
        # ---- 7. Normalize and scale ---------------------------------------
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
 
        topk_weights = (topk_weights * self.routed_scaling_factor).to(hidden_states.dtype)
 
        return router_logits, topk_weights, topk_idx
    

class Param2MoESparseMoeBlock(nn.Module):
    """
    MoE feed-forward block combining:
 
    1. **Routed experts** (Param2MoEExperts): selected per-token by Param2MoERouter.
       Only top-k experts are activated; their weighted outputs are summed.
 
    2. **Shared expert** (Param2MoEMLP): always active, applied to every token
       regardless of routing. Output is added to the routed sum.
 
    The router logits (index 0 of Param2MoERouter's output) are automatically
    captured by the OutputRecorder hook defined in Param2PreTrainedModel,
    so this forward does NOT need to return them.
    """
 
    def __init__(self, config: Param2MoEConfig):
        super().__init__()
        self.experts = Param2MoEExperts(config)
        self.gate = Param2MoERouter(config)
        # Shared expert uses moe_shared_expert_intermediate_size (4096),
        # not the routed expert size (moe_intermediate_size = 2048).
        self.shared_experts = Param2MoEMLP(
            config, intermediate_size=config.moe_shared_expert_intermediate_size
        )
 
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Keep a reference for the shared expert residual addition
        residual = hidden_states
        batch_size, seq_len, hidden_dim = hidden_states.shape
 
        # Flatten to [tokens, hidden_size] for the router and experts
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
 
        # Router: logits are captured automatically by OutputRecorder;
        # we only consume the weights and indices here.
        _, routing_weights, selected_experts = self.gate(hidden_states_flat)
 
        # Routed experts: sparse computation over selected_experts
        routed_output = self.experts(hidden_states_flat, selected_experts, routing_weights)
        routed_output = routed_output.view(batch_size, seq_len, hidden_dim)
 
        # Shared expert: dense computation on all tokens, added to routed output
        hidden_states = routed_output + self.shared_experts(residual)
        return hidden_states


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
        self.mlp = Param2MoESparseMoeBlock(config) if layer_idx >= config.first_k_dense_replace else Param2MoEMLP(config)

        self.input_layernorm = Param2MoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Param2MoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Param2MoEPreTrainedModel(MixtralPreTrainedModel):
    _can_record_outputs = {
        "router_logits": OutputRecorder(Param2MoERouter, index=0),
        "hidden_states": Param2MoEDecoderLayer,
        "attentions": Param2MoEAttention,
    }


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
