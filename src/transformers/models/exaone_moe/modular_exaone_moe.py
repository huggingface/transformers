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

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3ForCausalLM,
    DeepseekV3MoE,
    DeepseekV3TopkRouter,
)
from ..dots1.modeling_dots1 import Dots1Model
from ..exaone4.configuration_exaone4 import Exaone4Config
from ..exaone4.modeling_exaone4 import (
    Exaone4Attention,
    Exaone4PreTrainedModel,
)
from ..glm5_next.modeling_glm5_next import Glm5NextTextExperts, Glm5NextTextMLP
from ..olmoe.modeling_olmoe import (
    OlmoeDecoderLayer,
)


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
    mlp_layer_types (`list`, *optional*):
        MLP pattern for each layer. Prioritized over `first_k_dense_replace`.
    first_k_dense_replace (`int`, *optional*, defaults to 1):
        Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                    \--k dense layers--/
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    swiglu_limit (`float`, *optional*):
        Bound that the SwiGLU gate (from above) and up projections (from both sides) are clamped to before the
        activation. `None` means no clamping. Like `sliding_window`, it can be set per layer with `per_layer_config`.

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
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_ep_plan = {
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "ep_dispatch_experts",
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
    sliding_window: int | None = 4096
    sliding_window_pattern: str | int | None = 4
    layer_types: list[str] | None = None
    mlp_layer_types: list[str] | None = None
    first_k_dense_replace: int = 1
    moe_intermediate_size: int = 1024
    num_experts: int = 64
    num_experts_per_tok: int = 8
    output_router_logits: bool = False
    num_shared_experts: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    n_group: int = 1
    topk_group: int = 1
    swiglu_limit: float | None = None

    def __post_init__(self, **kwargs):
        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "dense" if i < self.first_k_dense_replace else "sparse" for i in range(self.num_hidden_layers)
            ]

        # K-EXAONE 2.0 Hub configs store per-layer values as lists (`0` meaning none), which become `per_layer_config`
        sliding_windows = kwargs.pop("sliding_windows", None)
        swiglu_limits = kwargs.pop("swiglu_limits", None)
        for name, values in (("sliding_windows", sliding_windows), ("swiglu_limits", swiglu_limits)):
            if values is not None and len(values) != self.num_hidden_layers:
                raise ValueError(
                    f"`{name}` must have one entry per layer ({self.num_hidden_layers}), but got {len(values)}."
                )

        per_layer_config = {}
        if sliding_windows is not None:
            windows = [window for window in sliding_windows if window]
            if windows:
                self.sliding_window = max(set(windows), key=windows.count)
            for layer_idx, window in enumerate(sliding_windows):
                if window and window != self.sliding_window:
                    per_layer_config.setdefault(layer_idx, {})["sliding_window"] = window
        if swiglu_limits is not None:
            for layer_idx, limit in enumerate(swiglu_limits):
                if limit:
                    per_layer_config.setdefault(layer_idx, {})["swiglu_limit"] = limit
        if per_layer_config:
            kwargs.setdefault("per_layer_config", per_layer_config)

        super().__post_init__(**kwargs)


def get_mask_keys(config: ExaoneMoeConfig) -> list[str]:
    """Key of each layer's mask. Sliding layers get one key per window size when the window varies across layers."""
    per_layer_windows = "sliding_window" in (config.per_layer_attributes or ())
    return [
        f"sliding_attention_{layer_config.sliding_window}"
        if per_layer_windows and layer_type == "sliding_attention"
        else layer_type
        for layer_type, layer_config in zip(config.layer_types, config.per_layer_config)
    ]


def create_masks(config: ExaoneMoeConfig, **mask_kwargs) -> dict:
    """Creates one mask per key of `get_mask_keys`, using each layer's own `sliding_window`."""
    causal_mask_mapping = {}
    for layer_idx, mask_key in enumerate(get_mask_keys(config)):
        if mask_key in causal_mask_mapping:
            continue
        if mask_key == "full_attention":
            causal_mask_mapping[mask_key] = create_causal_mask(config=config, **mask_kwargs)
        else:
            causal_mask_mapping[mask_key] = create_sliding_window_causal_mask(
                config=config.per_layer_config[layer_idx], layer_idx=layer_idx, **mask_kwargs
            )
    return causal_mask_mapping


class ExaoneMoeAttention(Exaone4Attention):
    pass


class ExaoneMoeMLP(Glm5NextTextMLP):
    def __init__(self, config, intermediate_size=None):
        super().__init__(config, intermediate_size)
        self.swiglu_limit = config.swiglu_limit or float("inf")


class ExaoneMoeTopkRouter(DeepseekV3TopkRouter):
    def __init__(self, config):
        super().__init__(self, config)
        self.num_experts = config.num_experts


class ExaoneMoeExperts(Glm5NextTextExperts):
    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.num_experts
        self.swiglu_limit = config.swiglu_limit or float("inf")


class ExaoneMoeSparseMoEBlock(DeepseekV3MoE):
    def __init__(self, config):
        super().__init__()
        self.experts = ExaoneMoeExperts(config)
        self.shared_experts = ExaoneMoeMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.num_shared_experts
        )


class ExaoneMoeDecoderLayer(OlmoeDecoderLayer):
    def __init__(self, config: ExaoneMoeConfig, layer_idx: int):
        # K-EXAONE 2.0 sets `sliding_window` and `swiglu_limit` per layer
        config = config.per_layer_config[layer_idx]
        super().__init__(config, layer_idx)
        self.mlp = (
            ExaoneMoeSparseMoEBlock(config) if config.mlp_layer_types[layer_idx] == "sparse" else ExaoneMoeMLP(config)
        )


class ExaoneMoePreTrainedModel(Exaone4PreTrainedModel):
    config: ExaoneMoeConfig

    _can_record_outputs = {
        "hidden_states": ExaoneMoeDecoderLayer,
        "attentions": ExaoneMoeAttention,
        "router_logits": OutputRecorder(ExaoneMoeTopkRouter, index=0),
    }

    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"mtp.*"]

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, ExaoneMoeTopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, ExaoneMoeExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


class ExaoneMoeModel(Dots1Model):
    def __init__(self, config: ExaoneMoeConfig):
        super().__init__(config)
        self.mask_keys = get_mask_keys(config)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and cache is None:
            cache = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = cache.get_seq_length() if cache is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            causal_mask_mapping = create_masks(
                self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache=cache,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.mask_keys[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                cache=cache,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            cache=cache,
        )


class ExaoneMoeForCausalLM(DeepseekV3ForCausalLM):
    @staticmethod
    def create_masks_for_generate(
        config: ExaoneMoeConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache: Cache | None,
        position_ids: torch.Tensor | None = None,
        block_sequence_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        # Used by `generate` with compilable caches: the generic version assumes a single `sliding_window`
        return create_masks(
            config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache=cache,
            position_ids=position_ids,
            block_sequence_ids=block_sequence_ids,
        )

    def forward(
        self,
        **super_kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
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
        return super().forward(**super_kwargs)


__all__ = [
    "ExaoneMoeConfig",
    "ExaoneMoePreTrainedModel",
    "ExaoneMoeModel",
    "ExaoneMoeForCausalLM",
]
