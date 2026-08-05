# Copyright 2026 The LG AI Research and HuggingFace Inc. team. All rights reserved.
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
"""LG AI Research EXAONE Lab"""

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...integrations import use_experts_implementation
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..exaone4.configuration_exaone4 import Exaone4Config
from ..exaone4.modeling_exaone4 import (
    Exaone4Attention,
    Exaone4ForCausalLM,
    Exaone4Model,
    Exaone4PreTrainedModel,
)
from ..olmoe.modeling_olmoe import (
    OlmoeDecoderLayer,
)
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP


@auto_docstring(checkpoint="LGAI-EXAONE/K-EXAONE-236B-A23B")
@strict
class ExaoneMoeConfig(Exaone4Config):
    r"""
    sliding_window_pattern (`str`, *optional*, defaults to 4):
        The pattern to use for sliding window attention. Can be one of:
            - `None`: No sliding window attention is used
            - `int`: Every `sliding_window` layers, use global attention, else use local attention.
            - `str`: A sequence of "L" (local attention) and "G" (global attention) characters that defines the
                attention pattern. The pattern starts from layer 0 and repeats every `sliding_window` layers. The
                final layer always uses global attention regardless of the pattern.
        For instance, sliding_window_pattern="LLLG" same as sliding_window=4, which means:
            - Layer 0, 1, 2: local attention,
            - Layer 3: global attention,
            ...(repeated)
    sliding_windows (`list`, *optional*):
        Sliding window sizes for each layer. 0 means full attention, otherwise must be positive integer.
        Prioritized over `sliding_window` and `sliding_window_pattern`.
    mlp_layer_types (`list`, *optional*):
        MLP pattern for each layer. Prioritized over `first_k_dense_replace`.
    first_k_dense_replace (`int`, *optional*, defaults to 1):
        Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                    \--k dense layers--/
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    swiglu_limits (`list`, *optional*):
        Swiglu limits for each layer. 0 means no swiglu limit, otherwise must be positive float.

    Example:

    ```python
    >>> from transformers import ExaoneMoeModel, ExaoneMoeConfig

    >>> # Initializing a EXAONE configuration
    >>> configuration = ExaoneMoeConfig()

    >>> # Initializing a model from configuration
    >>> model = ExaoneMoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    vocab_size: int = 102400
    hidden_size: int = 4096
    intermediate_size: int = 16384
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 53
    pad_token_id: int | None = 0
    tie_word_embeddings: bool = False
    rope_parameters: dict | None = None
    attention_dropout: float | int = 0.0
    sliding_window: int = 4096
    sliding_window_pattern: str | int | None = 4
    layer_types: list[str] | None = None
    sliding_windows: list[int] | None = None
    mlp_layer_types: list[str] | None = None
    first_k_dense_replace: int = 1
    moe_intermediate_size: int = 1024
    num_experts: int = 64
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    n_group: int = 1
    topk_group: int = 1
    swiglu_limits: list[float] | None = None

    def __post_init__(self, **kwargs):
        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "dense" if i < self.first_k_dense_replace else "sparse" for i in range(self.num_hidden_layers)
            ]

        # Validate sliding windows
        if self.sliding_windows is not None:
            if len(self.sliding_windows) != self.num_hidden_layers:
                raise ValueError(
                    f"Number of sliding windows must be equal to the number of hidden layers ({self.num_hidden_layers}), but got {len(self.sliding_windows)}"
                )
            for layer_idx, (layer_type, window_size) in enumerate(zip(self.layer_types, self.sliding_windows)):
                if window_size < 0:
                    raise ValueError(
                        f"Sliding window size must be greater than 0, but got {window_size} at layer {layer_idx}"
                    )
                if layer_type == "sliding_attention" and window_size == 0:
                    raise ValueError(f"Found sliding window size 0 for layer {layer_idx}")

        # Validate swiglu limits
        if self.swiglu_limits is not None:
            if len(self.swiglu_limits) != self.num_hidden_layers:
                raise ValueError(
                    f"Number of swiglu limits must be equal to the number of hidden layers ({self.num_hidden_layers}), but got {len(self.swiglu_limits)}"
                )
            for layer_idx, limit in enumerate(self.swiglu_limits):
                if limit < 0:
                    raise ValueError(f"Swiglu limit must be non-negative, but got {limit} at layer {layer_idx}")

        super().__post_init__(**kwargs)


class ExaoneMoeAttention(Exaone4Attention):
    def __init__(self, config: ExaoneMoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if config.sliding_windows is not None:
            self.sliding_window = config.sliding_windows[layer_idx] or config.sliding_window or None


class ExaoneMoeMLP(Qwen2MoeMLP):
    def __init__(
        self, config: ExaoneMoeConfig, intermediate_size: int | None = None, swiglu_limit: float | None = None
    ):
        super().__init__(config, intermediate_size=intermediate_size)
        self.swiglu_limit = swiglu_limit

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.swiglu_limit is not None:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class ExaoneMoeTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts))

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = nn.functional.linear(hidden_states.float(), self.weight.float())
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
        group_scores = (
            scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.num_group, self.num_experts // self.num_group)
            .reshape(-1, self.num_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices


@use_experts_implementation
class ExaoneMoeExperts(nn.Module):
    def __init__(self, config, swiglu_limit=None):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        self.swiglu_limit = swiglu_limit

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            gate, up = nn.functional.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            if self.swiglu_limit is not None:
                gate = gate.clamp(max=self.swiglu_limit)
                up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class ExaoneMoeSparseMoEBlock(nn.Module):
    def __init__(self, config, swiglu_limit=None):
        super().__init__()
        self.config = config
        self.experts = ExaoneMoeExperts(config, swiglu_limit)
        self.gate = ExaoneMoeTopkRouter(config)
        self.shared_experts = ExaoneMoeMLP(
            config=config,
            intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
            swiglu_limit=swiglu_limit,
        )

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        _, topk_weights, topk_indices = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        return hidden_states + self.shared_experts(residuals)


class ExaoneMoeDecoderLayer(OlmoeDecoderLayer):
    def __init__(self, config: ExaoneMoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = (
            ExaoneMoeSparseMoEBlock(
                config, config.swiglu_limits[layer_idx] or None if config.swiglu_limits is not None else None
            )
            if config.mlp_layer_types[layer_idx] == "sparse"
            else ExaoneMoeMLP(
                config,
                swiglu_limit=config.swiglu_limits[layer_idx] or None if config.swiglu_limits is not None else None,
            )
        )


class ExaoneMoePreTrainedModel(Exaone4PreTrainedModel):
    config: ExaoneMoeConfig

    _can_record_outputs = {
        "hidden_states": ExaoneMoeDecoderLayer,
        "attentions": ExaoneMoeAttention,
        "router_logits": ExaoneMoeSparseMoEBlock,
    }

    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"mtp.*"]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, ExaoneMoeTopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, ExaoneMoeExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


class ExaoneMoeModel(Exaone4Model):
    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
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

        # It may already have been prepared by e.g. `generate`
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
            }
            if "sliding_attention" in self.config.layer_types:
                sliding_windows = getattr(self.config, "sliding_windows", None)
                if sliding_windows is None:
                    causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
                else:
                    for sliding_window in set(sliding_windows) - {0}:
                        causal_mask_mapping[f"sliding_attention_{sliding_window}"] = create_sliding_window_causal_mask(
                            **mask_kwargs, sliding_window=sliding_window
                        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i]
            if layer_type == "sliding_attention" and getattr(self.config, "sliding_windows", None) is not None:
                layer_type = f"sliding_attention_{self.config.sliding_windows[i]}"
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[layer_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class ExaoneMoeForCausalLM(Exaone4ForCausalLM):
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
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("LGAI-EXAONE/K-EXAONE-236B-A23B")
        >>> tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/K-EXAONE-236B-A23B")

        >>> prompt = "Explain how wonderful you are"
        >>> messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        >>> input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        >>> output = model.generate(**input_ids.to(model.device), max_new_tokens=128)
        >>> tokenizer.decode(output[0], skip_special_tokens=False)
        "<|system|>\nYou are a helpful assistant.<|endofturn|>\n<|user|>\nExplain how wonderful you are<|endofturn|>\n<|assistant|>\n<think>\n\n</think>\n\nThank you for the kind question! While I can't feel emotions or take pride in the way humans do, I *can* share what makes me uniquely helpful and capable—qualities that many people find wonderful.\n\nHere’s how I can support you:\n\n🌟 **Knowledge at Your Fingertips**  \nI have access to a vast amount of information across countless topics—from science and history to technology and creative writing. Whether you're curious, learning, or solving a problem, I can help explain things clearly and accurately.\n\n💬 **Clear, Helpful Communication**  \nI aim to respond in a way that's easy to understand, whether you need a simple explanation or a detailed analysis. I adapt my tone and depth to match"
        ```
        """
        super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


__all__ = [
    "ExaoneMoeConfig",
    "ExaoneMoePreTrainedModel",
    "ExaoneMoeModel",
    "ExaoneMoeForCausalLM",
]
