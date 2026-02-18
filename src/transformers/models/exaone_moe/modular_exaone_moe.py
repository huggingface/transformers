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

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
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


class ExaoneMoeConfig(Exaone4Config):
    model_type = "exaone_moe"

    r"""
    This is the configuration class to store the configuration of a [`ExaoneMoeModel`]. It is used to
    instantiate a EXAONE MoE model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the K-EXAONE-236B-A23B [LGAI-EXAONE/K-EXAONE-236B-A23B](https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model
    outputs. Read the documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 102400):
            Vocabulary size of the EXAONE MoE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ExaoneMoeModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 16384):
            Dimensionality of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 32768 for EXAONE 3.5).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 53):
            End of stream token id.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        sliding_window (`int`, *optional*, defaults to 4096):
            The size of the sliding window for the sliding window attention.
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
        layer_types (`list`, *optional*):
            Attention pattern for each layer. Prioritized over `sliding_window_pattern`.
        mlp_layer_types (`list`, *optional*):
            MLP pattern for each layer. Prioritized over `first_k_dense_replace`.
        first_k_dense_replace (`int`, *optional*, defaults to 1):
            Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                        \--k dense layers--/
        moe_intermediate_size (`int`, *optional*, defaults to 1024):
            Dimension of the MoE representations.
        num_experts (`int`, *optional*, defaults to 64):
            Number of routed experts.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of selected experts, None means dense model.
        num_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the weights of the routed experts.
        routed_scaling_factor (`float`, *optional*, defaults to 2.5):
            Scaling factor or routed experts.
        n_group (`int`, *optional*, defaults to 1):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 1):
            Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).

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

    def __init__(
        self,
        vocab_size=102400,
        hidden_size=4096,
        intermediate_size=16384,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=53,
        pad_token_id=0,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_dropout=0.0,
        sliding_window=4096,
        sliding_window_pattern=4,
        layer_types=None,
        mlp_layer_types=None,
        first_k_dense_replace=1,
        moe_intermediate_size=1024,
        num_experts=64,
        num_experts_per_tok=8,
        num_shared_experts=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        n_group=1,
        topk_group=1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.rope_parameters = rope_parameters

        self.layer_types = layer_types
        if self.sliding_window is None:
            sliding_window_pattern = 0
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if ((i + 1) % (sliding_window_pattern) != 0 and i < self.num_hidden_layers)
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        self.mlp_layer_types = mlp_layer_types
        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "dense" if i < self.first_k_dense_replace else "sparse" for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.mlp_layer_types, self.num_hidden_layers, attention=False)

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.tie_word_embeddings = tie_word_embeddings

        PreTrainedConfig.__init__(**kwargs)


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
        cache_position: torch.LongTensor | None = None,
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
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


__all__ = [
    "ExaoneMoeConfig",
    "ExaoneMoePreTrainedModel",
    "ExaoneMoeModel",
    "ExaoneMoeForCausalLM",
]
