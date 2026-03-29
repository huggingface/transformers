# Copyright 2025 NAVER CLOUD Corp. and The HuggingFace Inc. team. All rights reserved.
#
# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/granite/modeling_granite.py
# Copyright 2024 IBM and the HuggingFace Inc. team. All rights reserved.
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
"""HyperCLOVAX modular model definition.

HyperCLOVAX is a decoder-only transformer based on Granite with two key modifications:

- **Maximal Update Parametrization (MuP)**: uses per-config scaling factors
  (`attention_multiplier`, `residual_multiplier`, `embedding_multiplier`, `logits_scaling`)
  to enable stable training across model sizes.
- **Peri-Layer Normalization**: optionally applies an extra RMSNorm after each
  sub-layer output when `use_post_norm=True`.
"""

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...cache_utils import Cache
from ...modeling_outputs import CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..granite.configuration_granite import GraniteConfig
from ..granite.modeling_granite import (
    GraniteAttention,
    GraniteDecoderLayer,
    GraniteForCausalLM,
    GraniteMLP,
    GraniteModel,
    GranitePreTrainedModel,
    GraniteRMSNorm,
    GraniteRotaryEmbedding,
)
from ..llama.modeling_llama import (
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
)


@auto_docstring(checkpoint="naver-hyperclovax/HyperCLOVAX-SEED-Think-14B")
@strict
class HyperCLOVAXConfig(GraniteConfig):
    r"""
    embedding_multiplier (`float`, *optional*, defaults to `1.0`):
        Scaling factor applied to the token embedding outputs. Used in MuP to control the
        scale of the embedding activations. When `None`, defaults to `1.0` (no scaling).
    logits_scaling (`float`, *optional*, defaults to `1.0`):
        Scaling factor **multiplied** to the final logits before loss computation or sampling.
        Used in MuP to ensure consistent output scale across model sizes. When `None`,
        defaults to `1.0` (no scaling). Note: unlike [`GraniteConfig`], this is a multiplier,
        not a divisor.
    residual_multiplier (`float`, *optional*, defaults to `1.0`):
        Scaling factor applied to each sub-layer output before adding to the residual stream.
        Used in Maximal Update Parametrization (MuP) to stabilize training across model sizes.
        When `None`, defaults to `1.0` (no scaling).
    attention_multiplier (`float`, *optional*, defaults to `head_dim ** -0.5`):
        Scaling factor applied to attention logits before softmax, replacing the standard
        `1 / sqrt(head_dim)` scaling. Set explicitly for MuP-based training; when `None`,
        defaults to the standard value.
    use_post_norm (`bool`, *optional*, defaults to `False`):
        Whether to apply an extra RMSNorm after each sub-layer output (Peri-Layer Normalization).

    ```python
    >>> from transformers import HyperCLOVAXModel, HyperCLOVAXConfig

    >>> # Initializing a HyperCLOVAX style configuration
    >>> configuration = HyperCLOVAXConfig()

    >>> # Initializing a model from the configuration
    >>> model = HyperCLOVAXModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "hyperclovax"

    head_dim: int | None = None
    # Kept for backward compatibility with older HyperCLOVAX checkpoints; not used by the model.
    pretraining_tp: int | None = 1

    # MuP scaling factors: None means "resolve to the mathematically equivalent default".
    attention_multiplier: float | None = None
    residual_multiplier: float | None = None
    embedding_multiplier: float | None = None
    logits_scaling: float | None = None

    # Peri-Layer Normalization
    use_post_norm: bool = False

    def __post_init__(
        self,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        **kwargs,
    ):
        # Backward compatibility: convert legacy rope_theta / rope_scaling fields
        # (used in older HyperCLOVAX checkpoints) to the current rope_parameters format.
        if self.rope_parameters is None:
            rope_params: dict = {"rope_type": "default", "rope_theta": rope_theta}
            if rope_scaling is not None:
                rope_params.update(rope_scaling)
            self.rope_parameters = rope_params
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        super().__post_init__(**kwargs)

        # Resolve None MuP values to their mathematically equivalent defaults.
        if self.attention_multiplier is None:
            self.attention_multiplier = self.head_dim**-0.5
        if self.residual_multiplier is None:
            self.residual_multiplier = 1.0
        if self.embedding_multiplier is None:
            self.embedding_multiplier = 1.0
        if self.logits_scaling is None:
            self.logits_scaling = 1.0

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})."
            )


class HyperCLOVAXRMSNorm(GraniteRMSNorm):
    pass


class HyperCLOVAXRotaryEmbedding(GraniteRotaryEmbedding):
    pass


class HyperCLOVAXMLP(GraniteMLP):
    pass


class HyperCLOVAXAttention(GraniteAttention):
    pass


class HyperCLOVAXDecoderLayer(GraniteDecoderLayer):
    def __init__(self, config: HyperCLOVAXConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = HyperCLOVAXAttention(config=config, layer_idx=layer_idx)
        self.use_post_norm = config.use_post_norm

        # Peri-Layer Normalization: additional RMSNorm after each sub-layer output
        if self.use_post_norm:
            self.post_norm1 = HyperCLOVAXRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_norm2 = HyperCLOVAXRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if self.use_post_norm:
            hidden_states = self.post_norm1(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.use_post_norm:
            hidden_states = self.post_norm2(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier
        return hidden_states


@auto_docstring
class HyperCLOVAXPreTrainedModel(GranitePreTrainedModel):
    config_class = HyperCLOVAXConfig
    _no_split_modules = ["HyperCLOVAXDecoderLayer"]


@auto_docstring
class HyperCLOVAXModel(GraniteModel):
    def __init__(self, config: HyperCLOVAXConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [HyperCLOVAXDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HyperCLOVAXRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HyperCLOVAXRotaryEmbedding(config=config)


@auto_docstring
class HyperCLOVAXForCausalLM(GraniteForCausalLM):
    def __init__(self, config: HyperCLOVAXConfig):
        super().__init__(config)
        self.model = HyperCLOVAXModel(config)

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
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, HyperCLOVAXForCausalLM

        >>> model = HyperCLOVAXForCausalLM.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-14B")
        >>> tokenizer = AutoTokenizer.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-14B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me? Are you okay?" The man was confused and answered, "Yes." Then the woman asked."
        ```"""
        outputs = self.model(
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
        # MuP: multiply logits by logits_scaling (cf. GraniteForCausalLM which divides)
        logits = self.lm_head(hidden_states[:, slice_indices, :]) * self.config.logits_scaling

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class HyperCLOVAXForSequenceClassification(LlamaForSequenceClassification):
    pass


class HyperCLOVAXForQuestionAnswering(LlamaForQuestionAnswering):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class HyperCLOVAXForTokenClassification(LlamaForTokenClassification):
    pass


__all__ = [
    "HyperCLOVAXConfig",
    "HyperCLOVAXPreTrainedModel",
    "HyperCLOVAXModel",
    "HyperCLOVAXForCausalLM",
    "HyperCLOVAXForSequenceClassification",
    "HyperCLOVAXForQuestionAnswering",
    "HyperCLOVAXForTokenClassification",
]
