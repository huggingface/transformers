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
import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3MoE,
    DeepseekV3NaiveMoe,
    DeepseekV3TopkRouter,
)
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
    mlp_layer_types (`list`, *optional*):
        MLP pattern for each layer. Prioritized over `first_k_dense_replace`.
    first_k_dense_replace (`int`, *optional*, defaults to 1):
        Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                    \--k dense layers--/
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.

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

    def __post_init__(self, **kwargs):
        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "dense" if i < self.first_k_dense_replace else "sparse" for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)


class ExaoneMoeAttention(Exaone4Attention):
    pass


class ExaoneMoeMLP(Qwen2MoeMLP):
    pass


class ExaoneMoeTopkRouter(DeepseekV3TopkRouter):
    def __init__(self, config):
        nn.Module.__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty((config.num_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros(config.num_experts))


class ExaoneMoeExperts(DeepseekV3NaiveMoe):
    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.num_experts


class ExaoneMoeSparseMoEBlock(DeepseekV3MoE):
    def __init__(self, config):
        super().__init__()
        self.experts = ExaoneMoeExperts(config)
        self.shared_experts = ExaoneMoeMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.num_shared_experts
        )
        self.n_routed_experts = config.num_experts


class ExaoneMoeDecoderLayer(OlmoeDecoderLayer):
    def __init__(self, config: ExaoneMoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = (
            ExaoneMoeSparseMoEBlock(config) if config.mlp_layer_types[layer_idx] == "sparse" else ExaoneMoeMLP(config)
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
    pass


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
